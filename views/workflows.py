"""Workflows view — mirrors Vue WorkflowsView.vue (screenshot-faithful).

Layout matches the running Vue interface exactly:

  Left rail:
    - "WORKFLOW STUDIO" label + title + subtitle
    - "+ 新建工作流" button (top-right)
    - Search input "搜索工作流"
    - Styled list items: name/description + status dot (no icon)

  Right stage (3 states):
    a) Nothing selected → empty state with CTA
    b) Simple workflow selected/creating → WorkflowSimpleCreator form widget
    c) Advanced (non-simple) selected → overview/metrics panel

A workflow entry has the shape:
    {
      "id", "name", "description",
      "definition": <pymss-studio-graph v2 or legacy steps>,
      "runParams": {"device", "outputFormat", "useTta"},
      "batch": {"folder", "recursive", "sort"},
      "createdAt", "updatedAt"
    }
"""
from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QCheckBox,
    QDialog, QMessageBox, QFrame, QScrollArea, QSizePolicy,
    QStackedWidget, QFileDialog,
)

from ..workflow_graph import (
    normalize_definition, validate_definition,
    used_models, graph_metrics, build_run_payload,
    resolve_workflow_open_mode, get_workflow_validation_summary,
    hydrate_workflow_definition, analyze_simple_workflow,
    workflow_validation_message,
)
from .workflow_simple import WorkflowSimpleCreator

MEDIA_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus",
              ".mp4", ".mov", ".mkv", ".webm", ".wma"}

# ---- reason code labels (Chinese) ----
SIMPLE_REASON_LABELS = {
    "utility_nodes": "包含工具节点",
    "unsupported_nodes": "包含不支持的节点类型",
    "custom_model_type": "使用自定义模型类型",
    "comfy_metadata": "包含 Comfy-MSS 元数据",
    "invalid_graph": "工作流图结构无效",
    "custom_save_behavior": "自定义保存行为",
}


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in (name or "workflow")).strip("_") or "workflow"


class _WorkflowListItem(QWidget):
    """One styled row in the left rail workflow list (matches Vue .wf-row).

    Each row shows two lines: the workflow name and its description. The row
    height is computed from the *real* wrapped height of both labels (so the
    text is never clipped), not from ``sizeHint()`` which assumes a single wide
    line and therefore badly undercounts word-wrapped content.
    """

    def __init__(self, name: str, description: str, parent=None):
        super().__init__(parent)
        self._full_name = name
        self._build(name, description)

    def _build(self, name: str, desc: str) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Text column (no icon, no status dot — matches the original Vue UI)
        text_col = QVBoxLayout()
        text_col.setSpacing(4)
        self.name_lbl = QLabel(name)
        self.name_lbl.setStyleSheet("color:#1e293b;")
        # Wrap long names onto multiple lines so they are never clipped.
        self.name_lbl.setWordWrap(True)
        self.name_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        text_col.addWidget(self.name_lbl)
        self.desc_lbl = QLabel(desc or "暂无说明")
        self.desc_lbl.setStyleSheet("color:#94a3b8;")
        self.desc_lbl.setWordWrap(True)
        self.desc_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        text_col.addWidget(self.desc_lbl)
        root.addLayout(text_col, 1)

    @staticmethod
    def _wrapped_height(label: QLabel, width: int) -> int:
        """Height of ``label`` text when wrapped into ``width`` px (CJK-safe)."""
        fm = label.fontMetrics()
        # TextWordWrap breaks Latin words at spaces; TextWrapAnywhere also breaks
        # between CJK characters, matching how QLabel actually renders the text.
        flags = int(Qt.TextFlag.TextWordWrap) | int(Qt.TextFlag.TextWrapAnywhere)
        rect = fm.boundingRect(0, 0, max(1, width), 10000, flags, label.text())
        return rect.height()

    def sizeHintForWidth(self, width: int) -> int:
        """Height (incl. item padding) needed to fully show name + description.

        ``width`` is the item size-hint width. We subtract the item's horizontal
        padding (4+4), the widget's own left/right margins (10+10) and the
        layout spacing (10) to get the text column width. The estimate
        intentionally undercounts the available width slightly so the wrapped
        text is always tall enough and never clipped. No status dot is shown
        (matches the original Vue UI), so no width is reserved for it.
        """
        inner = max(1, width - 52)
        h = 20  # widget vertical margins (10 top + 10 bottom)
        h += self._wrapped_height(self.name_lbl, inner)
        h += 4   # spacing between the two labels
        h += self._wrapped_height(self.desc_lbl, inner)
        h += 16  # item vertical padding (8 top + 8 bottom)
        return int(h)


class WorkflowsView(QWidget):
    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self.simple_creator: WorkflowSimpleCreator | None = None
        self._detail_stacked_index = 0  # 0=empty, 1=simple, 2=overview
        self._build_ui()
        self.refresh_list()
        # Always show the editor controls — never a blank stage. If a workflow
        # is selected, open it; otherwise show a fresh (new-draft) creator.
        if self.app.selected_workflow and any(
                w.get("id") == self.app.selected_workflow for w in self.app.workflows):
            self.select_workflow(self.app.selected_workflow)
        else:
            self._show_simple_creator(None)

    # ---- UI -----------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(16)

        # ====== LEFT RAIL ======
        rail = QFrame()
        rail.setObjectName("wf_rail")
        rail.setStyleSheet("""
            #wf_rail {
                background:#f8fafc;
                border:1px solid #e2e8f0;
                border-radius:18px;
            }
        """)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(15, 14, 14, 14)
        rail_layout.setSpacing(12)

        # Header area inside rail
        rail_header = QVBoxLayout()
        rail_header.setSpacing(4)

        studio_lbl = QLabel("WORKFLOW STUDIO")
        studio_lbl.setStyleSheet(
            "font-size:11px;color:#3b82f6;font-weight:700;"
            "letter-spacing:0.06em;")
        rail_header.addWidget(studio_lbl)

        title_lbl = QLabel("<b style='font-size:20px'>工作流</b>")
        rail_header.addWidget(title_lbl)

        subtitle = QLabel(
            "<span style='font-size:13px;color:#64748b'>"
            "创建、修改和管理 pymss 工作流定义，然后在分离页运行工作流推理。</span>")
        rail_header.addWidget(subtitle)
        rail_layout.addLayout(rail_header)

        # New button (right-aligned in header context, but we put it below)
        new_btn_row = QHBoxLayout()
        self.new_btn = QPushButton("+ 新建工作流")
        self.new_btn.setStyleSheet(
            "QPushButton{background:#3b82f6;color:white;border:none;"
            "padding:7px 18px;border-radius:8px;font-weight:600;}"
            "QPushButton:hover{background:#2563eb;}")
        self.new_btn.clicked.connect(self._new_simple)
        new_btn_row.addWidget(self.new_btn)
        new_btn_row.addStretch(1)
        rail_layout.addLayout(new_btn_row)

        # Search
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索工作流")
        self.search_edit.setStyleSheet(
            "QLineEdit{padding:8px 12px;border:1px solid #e2e8f0;"
            "border-radius:10px;background:white;font-size:13px;}"
            "QLineEdit:focus{border-color:#3b82f6;}")
        self.search_edit.textChanged.connect(self.refresh_list)
        rail_layout.addWidget(self.search_edit)

        # List
        self.list = QListWidget()
        self.list.setStyleSheet("""
            QListWidget{background:transparent;border:none;}
            QListWidget::item{
                padding:8px 4px;
                border:1px solid transparent;
                border-radius:13px;
                margin-bottom:4px;
            }
            QListWidget::item:selected{
                border-color:rgba(59,130,246,0.4);
                background:linear-gradient(180deg, rgba(59,130,246,0.08), transparent);
            }
            QListWidget::item:hover{
                background:#f1f5f9;
            }
        """)
        self.list.itemClicked.connect(self._on_item_clicked)
        rail_layout.addWidget(self.list, 1)
        rail.setFixedWidth(340)
        root.addWidget(rail, 0)

        # ====== RIGHT STAGE ======
        self.stage_container = QWidget()
        stage_root = QVBoxLayout(self.stage_container)
        stage_root.setContentsMargins(0, 0, 0, 0)
        self.stage_stack = QStackedWidget()
        stage_root.addWidget(self.stage_stack)

        # Page 0: empty state
        self.empty_page = self._build_empty_page()
        self.stage_stack.addWidget(self.empty_page)

        # Page 1: simple creator (created on demand)
        self.simple_page = QWidget()
        self.simple_layout = QVBoxLayout(self.simple_page)
        self.simple_layout.setContentsMargins(0, 0, 0, 0)
        self.stage_stack.addWidget(self.simple_page)

        # Page 2: advanced overview
        self.overview_page = self._build_overview_page()
        self.stage_stack.addWidget(self.overview_page)

        root.addWidget(self.stage_stack, 1)

    def _build_empty_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = QLabel("⚡")
        icon.setStyleSheet("font-size:52px;")
        v.addWidget(icon)
        title = QLabel("选择或创建一个工作流")
        title.setStyleSheet("font-size:17px;font-weight:600;color:#1e293b;")
        v.addWidget(title)
        hint = QLabel("从左侧选择一个已有工作流，或创建新工作流开始。")
        hint.setStyleSheet("font-size:13px;color:#64748b;")
        hint.setMaximumWidth(300)
        v.addWidget(hint)
        create_btn = QPushButton("+ 新建工作流")
        create_btn.setStyleSheet(
            "QPushButton{background:#3b82f6;color:white;border:none;"
            "padding:7px 18px;border-radius:8px;font-weight:600;}"
            "QPushButton:hover{background:#2563eb;}")
        create_btn.clicked.connect(self._new_simple)
        v.addWidget(create_btn)
        return w

    def _build_overview_page(self) -> QWidget:
        """Advanced-mode overview page (metrics, models, run params, actions)."""
        w = QScrollArea()
        w.setWidgetResizable(True)
        inner = QWidget()
        vl = QVBoxLayout(inner)
        vl.setContentsMargins(22, 20, 22, 20)
        vl.setSpacing(18)

        # Heading
        head = QHBoxLayout()
        icon = QLabel("⚡")
        icon.setFixedSize(38, 38)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(
            "background:#f0f4ff;color:#3b82f6;border-radius:12px;"
            "font-size:18px;")
        head.addWidget(icon)
        self.ov_name = QLabel("")
        self.ov_name.setStyleSheet(
            "font-size:21px;font-weight:700;color:#1e293b;"
            "border:none;background:transparent;")
        self.ov_name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        head.addWidget(self.ov_name, 1)
        self.ov_status = QLabel("")
        self.ov_status.setStyleSheet(
            "padding:0 12px;height:26px;border-radius:999px;"
            "font-size:12px;font-weight:600;")
        head.addWidget(self.ov_status)
        vl.addLayout(head)

        # Description
        self.ov_desc = QTextEdit()
        self.ov_desc.setMaximumHeight(72)
        self.ov_desc.setStyleSheet(
            "border:none;background:transparent;"
            "font-size:13px;color:#64748b;")
        vl.addWidget(self.ov_desc)

        # Metrics grid
        self.ov_metrics_box = QWidget()
        metrics_grid = QHBoxLayout(self.ov_metrics_box)
        metrics_grid.setContentsMargins(0, 0, 0, 0)
        metrics_grid.setSpacing(10)
        self.metric_labels: dict[str, tuple[QLabel, QLabel]] = {}
        metric_names = [("steps", "步骤数"), ("utilityNodes", "工具节点"),
                        ("saveOutputCount", "输出数"), ("stemCount", "音轨")]
        for key, label_text in metric_names:
            card = QFrame()
            card.setStyleSheet(
                "background:#f8fafc;border:1px solid #e2e8f0;"
                "border-radius:14px;padding:14px 12px;")
            cv = QVBoxLayout(card)
            cv.setContentsMargins(14, 12, 12, 12)
            cv.setSpacing(3)
            val_lbl = QLabel("—")
            val_lbl.setStyleSheet(
                "font-size:24px;font-weight:700;color:#1e293b;")
            name_lbl = QLabel(label_text)
            name_lbl.setStyleSheet(
                "font-size:11px;font-weight:600;color:#64748b;")
            cv.addWidget(val_lbl)
            cv.addWidget(name_lbl)
            metrics_grid.addWidget(card)
            self.metric_labels[key] = (val_lbl, name_lbl)
        vl.addWidget(self.ov_metrics_box)

        # Models section
        sec_models = QVBoxLayout()
        sec_models.setSpacing(8)
        sec_models.addWidget(QLabel("<b>所用模型</b>"))
        self.models_box = QWidget()
        self.models_flow = QHBoxLayout(self.models_box)
        self.models_flow.setContentsMargins(0, 0, 0, 0)
        self.models_flow.setSpacing(8)
        self.models_flow.addStretch(1)
        sec_models.addWidget(self.models_box)
        vl.addLayout(sec_models)

        # Run params
        sec_params = QVBoxLayout()
        sec_params.setSpacing(8)
        sec_params.addWidget(QLabel("<b>运行参数</b>"))
        params_grid = QHBoxLayout()
        params_grid.setSpacing(10)
        self.ov_dev_lbl = QLabel("")
        self.ov_fmt_lbl = QLabel("")
        self.ov_norm_lbl = QLabel("")
        for lbl, txt in [(self.ov_dev_lbl, "默认设备"),
                         (self.ov_fmt_lbl, "默认格式"),
                         (self.ov_norm_lbl, "归一化")]:
            card = QFrame()
            card.setStyleSheet(
                "background:#f8fafc;border-radius:12px;padding:10px 12px;")
            cl = QVBoxLayout(card)
            cl.setSpacing(4)
            n = QLabel(txt)
            n.setStyleSheet("font-size:11px;font-weight:600;color:#64748b;")
            cl.addWidget(n)
            v = QLabel("")
            v.setStyleSheet("font-size:14px;font-weight:600;color:#1e293b;")
            cl.addWidget(v)
            params_grid.addWidget(card)
            setattr(self, "_ov_param_" + ("dev" if "设" in txt else
                     "fmt" if "格" in txt else "norm"), v)
        sec_params.addLayout(params_grid)
        vl.addLayout(sec_params)

        # Validation error box
        self.ov_error = QLabel("")
        self.ov_error.setWordWrap(True)
        self.ov_error.setStyleSheet(
            "padding:12px 14px;border-radius:12px;"
            "border:1px solid #fbbf24;background:#fffbeb;"
            "color:#b45309;font-size:12px;line-height:1.5;")
        self.ov_error.hide()
        vl.addWidget(self.ov_error)

        # Simple-mode blockers info
        self.ov_blockers = QFrame()
        self.ov_blockers.setStyleSheet(
            "background:#f8fafc;border:1px solid #e2e8f0;"
            "border-radius:12px;padding:12px 14px;")
        bl = QVBoxLayout(self.ov_blockers)
        bl.setSpacing(8)
        bl.addWidget(QLabel("<span style='font-size:12px;'>"
                             "此工作流需要使用高级编辑器编辑：</span>"))
        self.blocker_list_widget = QLabel("")
        self.blocker_list_widget.setStyleSheet(
            "font-size:12px;color:#64748b;padding-left:18px;")
        bl.addWidget(self.blocker_list_widget)
        self.ov_blockers.hide()
        vl.addWidget(self.ov_blockers)

        # Action bar
        bar_frame = QFrame()
        bar_frame.setStyleSheet(
            "background:#f8fafc;border-top:1px solid #e2e8f0;"
            "border-radius:0;padding:14px 22px;margin-top:8px;")
        action_bar = QHBoxLayout(bar_frame)
        action_bar.setContentsMargins(14, 14, 14, 14)

        self.simple_mode_btn = QPushButton("简易模式")
        self.simple_mode_btn.setStyleSheet(
            "QPushButton{background:#3b82f6;color:white;border:none;"
            "padding:8px 18px;border-radius:8px;font-weight:600;font-size:14px;}"
            "QPushButton:hover{background:#2563eb;}"
            "QPushButton:disabled{background:#93c5fd;}")
        self.simple_mode_btn.clicked.connect(self._switch_to_simple)
        action_bar.addWidget(self.simple_mode_btn)

        self.run_btn = QPushButton("▶ 运行")
        self.run_btn.setStyleSheet(
            "QPushButton{background:transparent;color:#334155;"
            "border:1px solid #e2e8f0;padding:8px 18px;"
            "border-radius:8px;font-weight:500;font-size:14px;}"
            "QPushButton:hover{background:#f8fafc;}")
        self.run_btn.clicked.connect(self._run)
        action_bar.addWidget(self.run_btn)

        action_bar.addStretch(1)
        more = QHBoxLayout()
        more.setSpacing(8)
        dup_btn = QPushButton("📋 复制")
        exp_btn = QPushButton("⬇ 导出 comfy-mss")
        del_btn = QPushButton("🗑 删除")
        for btn, slot in [(dup_btn, self._duplicate),
                          (exp_btn, self._export),
                          (del_btn, self._delete)]:
            btn.setStyleSheet(
                "QPushButton{background:transparent;color:#64748b;"
                "border:none;padding:6px 12px;border-radius:8px;"
                "font-size:13px;}"
                "QPushButton:hover{color:#334155;background:#f1f5f9;}")
            btn.clicked.connect(slot)
            more.addWidget(btn)
        action_bar.addLayout(more)
        vl.addWidget(bar_frame)
        vl.addStretch(1)

        w.setWidget(inner)
        return w

    # ---- list ---------------------------------------------------------
    def refresh_list(self) -> None:
        self.list.clear()
        q = self.search_edit.text().strip().lower()
        for wf in self.app.workflows:
            if q and q not in (wf.get("name", "") + wf.get("description", "")).lower():
                continue
            item = QListWidgetItem()
            widget = _WorkflowListItem(
                wf.get("name", ""),
                wf.get("description", ""),
            )
            # Give the row enough height to show the full (wrapped) name.
            from PySide6.QtCore import QSize
            content_w = max(200, self.list.viewport().width() - 12)
            row_h = max(58, widget.sizeHintForWidth(content_w))
            item.setSizeHint(QSize(content_w, row_h))
            item.setData(Qt.ItemDataRole.UserRole, wf.get("id"))
            self.list.addItem(item)
            self.list.setItemWidget(item, widget)
        # Keep selection
        if self.app.selected_workflow:
            self._highlight(self.app.selected_workflow)

    def _highlight(self, wid: str) -> None:
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == wid:
                self.list.setCurrentItem(it)
                break

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        wid = item.data(Qt.ItemDataRole.UserRole)
        self.select_workflow(wid)

    def select_workflow(self, wid: str) -> None:
        self.app.selected_workflow = wid
        wf = self.app.get_selected_workflow()
        if not wf:
            # No such workflow — show a fresh creator rather than a blank page.
            self._show_simple_creator(None)
            return

        # Determine open mode
        mode = resolve_workflow_open_mode(wf.get("definition"))

        if mode == "simple":
            self._show_simple_creator(wf)
        else:
            self._show_overview(wf)

        self._highlight(wid)

    def _show_simple_creator(self, wf: dict | None = None) -> None:
        """Show (or reuse) the Simple Creator for editing/creating."""
        # Clean up old creator
        if self.simple_creator:
            self.simple_creator.deleteLater()

        self.simple_creator = WorkflowSimpleCreator(self.app, wf)
        # Connect signals (复制 / 删除 / 保存)
        self.simple_creator.save.connect(self._on_simple_save)
        self.simple_creator.duplicate_req.connect(self._duplicate)
        self.simple_creator.delete_req.connect(self._delete)

        # Replace content in simple_page
        while self.simple_layout.count():
            child = self.simple_layout.takeAt(0)
            w = child.widget() if child else None
            if w:
                w.deleteLater()
        self.simple_layout.addWidget(self.simple_creator)
        self.stage_stack.setCurrentIndex(1)  # simple

    def _show_overview(self, wf: dict) -> None:
        """Show the overview/metrics page for advanced workflows."""
        defn = normalize_definition(wf.get("definition"))
        errs = validate_definition(defn)
        summary = get_workflow_validation_summary(defn)
        draft = hydrate_workflow_definition(defn)
        m = graph_metrics(defn)

        # Name + status
        self.ov_name.setText(wf.get("name", ""))
        self.ov_desc.setPlainText(wf.get("description", ""))
        if errs:
            self.ov_status.setText(f"⚠ 需调整 ({len(errs)} 处问题)")
            self.ov_status.setStyleSheet(
                "color:#b45309;background:#fffbeb;"
                "padding:0 12px;height:26px;border-radius:999px;"
                "font-size:12px;font-weight:600;")
        else:
            self.ov_status.setText("✓ 配置就绪")
            self.ov_status.setStyleSheet(
                "color:#15803d;background:#f0fdf4;"
                "padding:0 12px;height:26px;border-radius:999px;"
                "font-size:12px;font-weight:600;")

        # Metrics
        steps_count = len(draft.get("steps", []))
        utility_count = len(draft.get("utilityNodes", []))
        stem_count = sum(len(s.get("stems", [])) for s in draft.get("steps", []))
        save_count = summary.get("saveOutputCount", m.get("outputCount", 0))

        for key, (val_lbl, _) in self.metric_labels.items():
            vals = {"steps": steps_count, "utilityNodes": utility_count,
                    "saveOutputCount": save_count, "stemCount": stem_count}
            val_lbl.setText(str(vals.get(key, "—")))

        # Models
        # Clear existing model chips
        while self.models_flow.count() > 1:
            child = self.models_flow.takeAt(0)
            w = child.widget() if child else None
            if w:
                w.deleteLater()
        models = used_models(defn)
        downloaded_set = set(m.get("name", "") for m in self.app.models if m.get("downloaded"))
        for mn in models:
            chip = QFrame()
            chip.setStyleSheet(
                f"{'border:1px solid #fbbf24;' if mn not in downloaded_set else ''}"
                "background:#f8fafc;border-radius:999px;"
                "padding:6px 12px;" if mn in downloaded_set else
                "background:#f8fafc;border-radius:999px;"
                "border:1px solid #fbbf24;padding:6px 12px;")
            ch = QHBoxLayout(chip)
            ch.setContentsMargins(6, 2, 8, 2)
            ch.setSpacing(6)
            ch.addWidget(QLabel("📦"))
            name_lbl = QLabel(mn)
            name_lbl.setStyleSheet("font-size:12px;color:#1e293b;")
            ch.addWidget(name_lbl)
            if mn not in downloaded_set:
                dl = QLabel("未下载")
                dl.setStyleSheet(
                    "color:#b45309;background:#fffbeb;"
                    "border-radius:999px;padding:1px 7px;"
                    "font-size:10px;font-weight:700;")
                ch.addWidget(dl)
            self.models_flow.insertWidget(self.models_flow.count() - 1, chip)
        if not models:
            self.models_flow.insertWidget(0, QLabel("— 未配置模型 —"))

        # Run params from defaults
        dd = defn.get("defaults", {})
        dev_val = dd.get("device", "auto")
        fmt_val = dd.get("output_format", "wav")
        ip = dd.get("inference_params") or {}
        norm_val = ip.get("normalize", False)
        self._ov_param_dev.setText(dev_val.upper() if isinstance(dev_val, str) else str(dev_val))
        self._ov_param_fmt.setText(fmt_val.upper() if isinstance(fmt_val, str) else str(fmt_val))
        self._ov_param_norm.setText("开" if norm_val else "关")

        # Error
        err_msg = workflow_validation_message(summary)
        if err_msg:
            self.ov_error.setText(f"⚠ {err_msg}")
            self.ov_error.show()
        else:
            self.ov_error.hide()

        # Blockers
        analysis = analyze_simple_workflow(defn)
        if not analysis["editable"]:
            reasons = [SIMPLE_REASON_LABELS.get(r, r) for r in analysis["reasonCodes"]]
            self.blocker_list_widget.setText("\n• ".join([""] + reasons))
            self.ov_blockers.show()
            self.simple_mode_btn.setEnabled(False)
        else:
            self.ov_blockers.hide()
            self.simple_mode_btn.setEnabled(True)

        self.stage_stack.setCurrentIndex(2)  # overview

    # ---- simple creator signal handlers ---------------------------------
    def _on_simple_save(self, payload: dict) -> None:
        wid = payload.get("id")
        if wid:
            wf = next((w for w in self.app.workflows if w.get("id") == wid), None)
            if wf:
                wf["name"] = payload["name"]
                wf["description"] = payload["description"]
                wf["definition"] = payload["definition"]
                wf["updatedAt"] = int(time.time() * 1000)
                self.app.save_workflows()
                self.refresh_list()
                self.select_workflow(wid)
                return
        # New workflow
        wf = {
            "id": uuid.uuid4().hex[:16],
            "name": payload["name"],
            "description": payload["description"],
            "definition": payload["definition"],
            "runParams": {"device": self.simple_creator.default_device,
                          "outputFormat": self.simple_creator.default_format,
                          "useTta": False},
            "batch": {"folder": "", "recursive": False, "sort": True},
            "createdAt": int(time.time() * 1000),
            "updatedAt": int(time.time() * 1000),
        }
        self.app.workflows.append(wf)
        self.app.save_workflows()
        self.refresh_list()
        self.select_workflow(wf["id"])

    # ---- navigation helpers ---------------------------------------------
    def _new_simple(self) -> None:
        self.app.selected_workflow = ""
        self._show_simple_creator(None)

    def _switch_to_simple(self) -> None:
        wf = self.app.get_selected_workflow()
        if wf and analyze_simple_workflow(wf.get("definition"))["editable"]:
            self._show_simple_creator(wf)

    # ---- actions --------------------------------------------------------
    def _duplicate(self) -> None:
        wf = self.app.get_selected_workflow()
        if not wf:
            return
        new = copy.deepcopy(wf)
        new["id"] = uuid.uuid4().hex[:16]
        new["name"] = wf.get("name", "") + " (副本)"
        new["createdAt"] = int(time.time() * 1000)
        new["updatedAt"] = int(time.time() * 1000)
        self.app.workflows.append(new)
        self.app.save_workflows()
        self.refresh_list()
        self.select_workflow(new["id"])

    def _delete(self) -> None:
        wf = self.app.get_selected_workflow()
        if not wf:
            return
        reply = QMessageBox.question(
            self, "删除工作流", f'确认删除工作流 "{wf.get("name")}"？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.app.workflows = [w for w in self.app.workflows if w.get("id") != wf.get("id")]
        self.app.selected_workflow = ""
        self.app.save_workflows()
        self.refresh_list()
        # Open the next available workflow, or a fresh creator if none remain.
        if self.app.workflows:
            self.select_workflow(self.app.workflows[0].get("id"))
        else:
            self._show_simple_creator(None)

    def _export(self) -> None:
        wf = self.app.get_selected_workflow()
        if not wf:
            return
        self._export_definition(wf.get("name", ""), wf.get("definition"))

    def _export_definition(self, name: str, definition: dict) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "导出工作流",
            f"{_slug(name)}.json",
            "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")

    def _resolve_device(self, selected: str) -> dict:
        def ids(v: str) -> list[int]:
            try:
                return [int(v)]
            except ValueError:
                return [0]
        if selected.startswith("cuda:"):
            return {"device": "cuda", "deviceIds": ids(selected[len("cuda:"):])}
        if selected == "cuda":
            return {"device": "cuda", "deviceIds": [0]}
        if selected == "auto":
            return self.app.config.get_runtime_device_config(self.app.env)
        if selected in ("cpu", "mps", "mlx"):
            return {"device": selected, "deviceIds": [0]}
        return {"device": "auto", "deviceIds": [0]}

    def _run(self) -> None:
        self._run_from_selected()

    def _run_from_selected(self) -> None:
        wf = self.app.get_selected_workflow()
        if not wf:
            return
        defn = normalize_definition(wf.get("definition"))
        errs = validate_definition(defn)
        if errs:
            QMessageBox.warning(self, "无法运行", "工作流存在问题：\n" + "\n".join(errs))
            return

        batch = wf.get("batch") or {}
        inputs: list[str] = []
        if batch.get("folder"):
            inputs = self._scan_inputs(batch["folder"],
                                       batch.get("recursive", False),
                                       batch.get("sort", True))
        if not inputs:
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择输入文件", "",
                "Media (*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.opus *.mp4 *.mov *.mkv *.webm)")
            inputs = list(files)
        if not inputs:
            return

        rp = wf.get("runParams") or {}
        dev_cfg = self._resolve_device(rp.get("device", "auto"))
        output_dir = self.app.config.output_dir()
        job_id = self.app.tasks.new_id("job")
        payload = build_run_payload(
            defn=defn,
            job_id=job_id,
            workflow_name=wf.get("name", ""),
            inputs=inputs,
            output_dir=output_dir,
            output_format=rp.get("outputFormat", "wav"),
            output_layout="folders",
            device=dev_cfg["device"],
            device_ids=dev_cfg["deviceIds"],
            model_dir=self.app.config.models_dir() or "",
            audio_params=self.app.config.get_audio_params(),
            use_tta=bool(rp.get("useTta", False)),
        )
        self.app.bridge.infer_workflow(payload, job_id)
        self.app.switch_to("separate")

    def _scan_inputs(self, folder: str, recursive: bool, sort: bool) -> list[str]:
        p = Path(folder)
        if not p.is_dir():
            return []
        files = p.rglob("*") if recursive else p.glob("*")
        found = [str(f) for f in files if f.is_file() and f.suffix.lower() in MEDIA_EXTS]
        if sort:
            found.sort()
        return found
