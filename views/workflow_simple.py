"""Simple Creator form widget — mirrors Vue WorkflowSimpleCreator.vue.

Form-based linear workflow editor (input → separate → save) that appears in the
right stage when a ``simple`` workflow is selected or being created.

Emits:
  save(payload)        — user clicked 保存
  duplicate_req()      — user clicked 复制
  delete_req()         — user clicked 删除
"""
from __future__ import annotations

import copy
import time
import uuid
from typing import Any

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit,
    QComboBox, QPushButton, QSpinBox, QScrollArea, QFrame,
    QFormLayout, QMessageBox, QToolButton, QSizePolicy,
)
from PySide6.QtGui import QFont, QIcon, QAction, QFontMetrics

from ..workflow_graph import (
    hydrate_simple_workflow,
    create_step_draft,
    get_workflow_validation_summary,
    workflow_validation_message,
    parse_model_stems,
    validate_definition,
    normalize_definition,
)


# ---- model helpers (tolerate camelCase or snake_case model entries) --------
def model_downloaded(m: dict) -> bool:
    d = m.get("downloaded")
    if d is None:
        d = m.get("is_downloaded")
    if isinstance(d, bool):
        return d
    return bool(d)


def model_configured_stems(m: dict | None) -> list[str]:
    if not m:
        return []
    cfg = (
        m.get("configInstruments") or m.get("config_instruments")
        or m.get("configTargetInstrument") or m.get("config_target_instrument")
        or m.get("targetStem") or m.get("target_stem") or ""
    )
    return parse_model_stems(cfg)


def _find_model(models: list[dict], name: str) -> dict | None:
    for m in models:
        if m.get("name") == name:
            return m
    return None


class ToggleSwitch(QWidget):
    """A simple on/off toggle switch styled like Naive UI n-switch."""

    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False
        self.setFixedSize(44, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def checked(self) -> bool:
        return self._checked

    def setChecked(self, v: bool):
        self._checked = v
        self.update()

    def mousePressEvent(self, event):
        self._checked = not self._checked
        self.toggled.emit(self._checked)
        self.update()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor("#3b82f6") if self._checked else QColor("#cbd5e1")
        p.setBrush(bg)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, 44, 24, 12, 12)
        knob_color = QColor("#ffffff")
        p.setBrush(knob_color)
        knob_x = 22 if self._checked else 2
        p.drawEllipse(knob_x, 2, 20, 20)


class StemChipToggle(QPushButton):
    """A checkable chip representing one output stem (e.g. 'other')."""

    def __init__(self, text: str, checked: bool, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(28)
        self.toggled.connect(self._apply_style)
        self._apply_style()

    def _apply_style(self) -> None:
        if self.isChecked():
            self.setStyleSheet(
                "QPushButton{background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;"
                "border-radius:999px;padding:4px 14px;font-size:12px;font-weight:600;}"
                "QPushButton:hover{background:#dbeafe;}")
        else:
            self.setStyleSheet(
                "QPushButton{background:#f8fafc;color:#64748b;border:1px solid #e2e8f0;"
                "border-radius:999px;padding:4px 14px;font-size:12px;}"
                "QPushButton:hover{background:#f1f5f9;}")


class StemSelector(QWidget):
    """Multi-select of output stems, driven by the selected model's stems."""

    changed = Signal(list)

    def __init__(self, available: list[str], selected: list[str], parent=None):
        super().__init__(parent)
        self.available = list(available)
        self.selected = [s for s in selected if s in available]
        self._build()

    def _build(self) -> None:
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        if not self.available:
            hint = QLabel("（请先选择模型）")
            hint.setStyleSheet("color:#94a3b8;font-size:12px;")
            h.addWidget(hint)
        else:
            for s in self.available:
                chip = StemChipToggle(s, s in self.selected)
                chip.toggled.connect(lambda checked, stem=s: self._on_toggle(stem, checked))
                h.addWidget(chip)
        h.addStretch(1)

    def _on_toggle(self, stem: str, checked: bool) -> None:
        if checked and stem not in self.selected:
            self.selected.append(stem)
        elif not checked and stem in self.selected:
            self.selected.remove(stem)
        self.changed.emit(list(self.selected))

    def rebuild(self, available: list[str], selected: list[str]) -> None:
        while self.layout().count():
            w = self.layout().takeAt(0).widget()
            if w:
                w.deleteLater()
        self.available = list(available)
        self.selected = [s for s in selected if s in available]
        if not self.available:
            hint = QLabel("（请先选择模型）")
            hint.setStyleSheet("color:#94a3b8;font-size:12px;")
            self.layout().addWidget(hint)
        else:
            for s in self.available:
                chip = StemChipToggle(s, s in self.selected)
                chip.toggled.connect(lambda checked, stem=s: self._on_toggle(stem, checked))
                self.layout().addWidget(chip)
        self.layout().addStretch(1)


class StepCard(QFrame):
    """One step row inside the simple creator step list."""

    model_changed = Signal(object, str)  # step_card, new_model_name
    removed = Signal(int)               # index

    def __init__(self, index: int, step_data: dict, models: list[dict],
                 all_steps: list[dict], parent=None):
        super().__init__(parent)
        self.index = index
        self.step_data = step_data
        self.models = models
        self.all_steps = all_steps
        self.setObjectName("step_card")
        self.setStyleSheet("""
            #step_card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 15px;
            }
        """)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # ---- header: number + title + delete ----
        header = QHBoxLayout()
        num_badge = QLabel(str(self.index + 1))
        num_badge.setFixedSize(22, 22)
        num_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num_badge.setStyleSheet(
            "background:#3b82f6;color:white;border-radius:11px;"
            "font-size:12px;font-weight:bold;")
        header.addWidget(num_badge)
        title = QLabel(f"步骤 {self.index + 1}")
        title.setStyleSheet("font-weight:600;font-size:14px;color:#1e293b;")
        header.addWidget(title, 1)
        del_btn = QLabel("🗑")
        del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        del_btn.setToolTip("删除步骤")
        del_btn.mousePressEvent = lambda e: self.removed.emit(self.index)
        header.addWidget(del_btn)
        layout.addLayout(header)

        # ---- grid: model | input source | overlap ----
        grid = QHBoxLayout()
        grid.setSpacing(12)

        # 模型
        model_col = QVBoxLayout()
        model_col.setSpacing(4)
        mlbl = QLabel("模型")
        mlbl.setStyleSheet("font-size:11px;color:#64748b;")
        model_col.addWidget(mlbl)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        self.model_combo.setMinimumWidth(200)
        self._populate_models()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_col.addWidget(self.model_combo)
        grid.addLayout(model_col, 2)

        # 输入来源
        input_col = QVBoxLayout()
        input_col.setSpacing(4)
        ilbl = QLabel("输入来源")
        ilbl.setStyleSheet("font-size:11px;color:#64748b;")
        input_col.addWidget(ilbl)
        self.input_combo = QComboBox()
        self.input_combo.setMinimumWidth(160)
        self._populate_input_options()
        self.input_combo.currentIndexChanged.connect(self._on_input_changed)
        input_col.addWidget(self.input_combo)
        grid.addLayout(input_col, 1)

        # Overlap Size
        ov_col = QVBoxLayout()
        ov_col.setSpacing(4)
        olbl = QLabel("Overlap Size")
        olbl.setStyleSheet("font-size:11px;color:#64748b;")
        ov_col.addWidget(olbl)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 1048576)
        self.overlap_spin.setValue(int(self.step_data.get("overlapSize") or 0))
        self.overlap_spin.setSpecialValueText("—")
        self.overlap_spin.valueChanged.connect(
            lambda v: self.step_data.__setitem__("overlapSize", v))
        ov_col.addWidget(self.overlap_spin)
        grid.addLayout(ov_col, 1)

        layout.addLayout(grid)

        # ---- 输出轨道: chips (derived from the selected model) ----
        stems_col = QVBoxLayout()
        stems_col.setSpacing(4)
        slbl = QLabel("输出轨道")
        slbl.setStyleSheet("font-size:11px;color:#64748b;")
        stems_col.addWidget(slbl)
        self.stems_container = QWidget()
        self.stems_layout = QHBoxLayout(self.stems_container)
        self.stems_layout.setContentsMargins(0, 0, 0, 0)
        self.stems_layout.setSpacing(6)
        stems_col.addWidget(self.stems_container)
        layout.addLayout(stems_col)
        self._rebuild_stems()

    def _populate_models(self) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        downloaded = [m for m in self.models if model_downloaded(m)]
        current = self.step_data.get("model", "")
        # Keep the currently-selected model even if it isn't downloaded yet,
        # so existing workflows still display correctly.
        if current and not any(m.get("name") == current for m in downloaded):
            downloaded = downloaded + [{"name": current, "downloaded": False}]
        for m in sorted(downloaded, key=lambda x: (x.get("name") or "").lower()):
            self.model_combo.addItem(m.get("name", ""))
        if current:
            idx = self.model_combo.findText(current)
            self.model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self.model_combo.setCurrentIndex(0)
        self.model_combo.blockSignals(False)

    def _populate_input_options(self) -> None:
        self.input_combo.blockSignals(True)
        self.input_combo.clear()
        self.input_combo.addItem("原始输入", "input")
        for i, step in enumerate(self.all_steps):
            if i == self.index:
                continue
            sid = step.get("id", f"step_{i + 1}")
            for stem in step.get("stems", []):
                label = f"步骤 {i + 1} · {stem}"
                value = f"{sid}.{stem}"
                self.input_combo.addItem(label, value)
        self.input_combo.blockSignals(False)
        cur_input = self.step_data.get("input", "")
        idx = self.input_combo.findData(cur_input)
        self.input_combo.setCurrentIndex(idx if idx >= 0 else 0)
        if idx < 0:
            # Dropdown fell back to "原始输入" — sync the underlying data so
            # validation doesn't complain about a missing source.
            self.step_data["input"] = self.input_combo.currentData() or "input"

    def _rebuild_stems(self) -> None:
        while self.stems_layout.count():
            w = self.stems_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        model = _find_model(self.models, self.step_data.get("model", ""))
        available = model_configured_stems(model)
        selected = [s for s in self.step_data.get("stems", []) if s in available]
        sel = StemSelector(available, selected)
        sel.changed.connect(self._on_stems_changed)
        self.stems_layout.addWidget(sel)

    def _on_model_changed(self, text: str) -> None:
        name = text.strip()
        self.step_data["model"] = name
        model = _find_model(self.models, name)
        stems = model_configured_stems(model)
        self.step_data["stems"] = list(stems)
        self.step_data["save"] = {s: s for s in stems}
        self._rebuild_stems()
        self._clear_invalid_inputs()
        self.model_changed.emit(self, name)

    def _on_stems_changed(self, stems: list[str]) -> None:
        self.step_data["stems"] = list(stems)
        self.step_data["save"] = {s: s for s in stems}
        self.model_changed.emit(self, self.step_data.get("model", ""))

    def _on_input_changed(self, _idx: int) -> None:
        self.step_data["input"] = self.input_combo.currentData() or "input"

    def _clear_invalid_inputs(self) -> None:
        allowed = set()
        for i in range(self.input_combo.count()):
            d = self.input_combo.itemData(i)
            if d:
                allowed.add(d)
        cur = self.step_data.get("input", "")
        if cur not in allowed:
            self.step_data["input"] = "input" if self.index == 0 else ""
            idx = self.input_combo.findData(self.step_data["input"])
            self.input_combo.setCurrentIndex(max(idx, 0))

    def refresh_input_options(self, all_steps: list[dict]) -> None:
        self.all_steps = all_steps
        cur = self.input_combo.currentData() or "input"
        self._populate_input_options()
        idx = self.input_combo.findData(cur)
        self.input_combo.setCurrentIndex(idx if idx >= 0 else 0)


class WorkflowSimpleCreator(QWidget):
    """The complete Simple Creator form matching the Vue screenshot.

    Layout (top to bottom):
      1. Header: badge + title + actions (复制 / 删除 / 保存)
      2. Defaults grid: 名称 | 设备 | 格式 | 标准化开关
      3. 说明 textarea (full width)
      4. Steps section: hint + 添加步骤 button + StepCard list
      5. Validation status (green/red)
      6. Collapsible JSON preview (reflects the selected workflow)
    """

    save = Signal(dict)
    duplicate_req = Signal()
    delete_req = Signal()

    def __init__(self, app, workflow_entry: dict | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.app = app
        self.workflow_entry = workflow_entry  # None when creating new
        self.name = ""
        self.description = ""
        self.default_device = "auto"
        self.default_format = "wav"
        self.default_normalize = False
        self.steps: list[dict] = []
        self.expected_updated_at: int | None = None
        self.source_definition: dict | None = None
        self._build_ui()
        self._load_workflow(workflow_entry)

    # ---- UI build -------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(14)

        # ====== HEADER ======
        header = QHBoxLayout()
        left_header = QVBoxLayout()
        left_header.setSpacing(2)
        self.editing_badge = QLabel("正在编辑")
        self.editing_badge.setStyleSheet(
            "color:#3b82f6;font-size:11px;font-weight:600;padding:2px 8px;"
            "background:#eff6ff;border-radius:999px;")
        left_header.addWidget(self.editing_badge)
        title_row = QHBoxLayout()
        title_row.setSpacing(0)
        self.title_label = QLabel("")
        self.title_label.setStyleSheet(
            "font-size:22px;font-weight:700;color:#1e293b;")
        title_row.addWidget(self.title_label)
        left_header.addLayout(title_row)
        header.addLayout(left_header, 1)

        # Action buttons: 复制 / 删除 / 保存
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.dup_btn = QPushButton("复制")
        self.del_btn = QPushButton("删除")
        self.save_btn = QPushButton("保存")
        self.dup_btn.setStyleSheet(self._outline_style())
        self.del_btn.setStyleSheet(self._danger_style())
        self.save_btn.setStyleSheet(self._primary_style())
        self.dup_btn.clicked.connect(lambda: self.duplicate_req.emit())
        self.del_btn.clicked.connect(lambda: self.delete_req.emit())
        self.save_btn.clicked.connect(self._emit_save)
        actions.addWidget(self.dup_btn)
        actions.addWidget(self.del_btn)
        actions.addWidget(self.save_btn)
        header.addLayout(actions)
        root.addLayout(header)

        # ====== DEFAULTS GRID ======
        defaults_grid = QHBoxLayout()
        defaults_grid.setSpacing(16)

        # 名称
        col_name = QVBoxLayout()
        col_name.setSpacing(4)
        col_name.addWidget(self._field_label("名称"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("未命名工作流")
        self.name_edit.textChanged.connect(lambda t: setattr(self, 'name', t.strip()))
        col_name.addWidget(self.name_edit)
        defaults_grid.addLayout(col_name, 2)

        # 默认设备
        col_dev = QVBoxLayout()
        col_dev.setSpacing(4)
        col_dev.addWidget(self._field_label("默认设备"))
        self.dev_combo = QComboBox()
        self.dev_combo.addItems(["Auto", "CPU", "CUDA", "MPS", "MLX"])
        self.dev_combo.currentTextChanged.connect(lambda t: setattr(self, 'default_device', (t or "auto").lower()))
        col_dev.addWidget(self.dev_combo)
        defaults_grid.addLayout(col_dev, 1)

        # 默认格式
        col_fmt = QVBoxLayout()
        col_fmt.setSpacing(4)
        col_fmt.addWidget(self._field_label("默认格式"))
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["WAV", "FLAC", "MP3", "M4A"])
        self.fmt_combo.currentTextChanged.connect(lambda t: setattr(self, 'default_format', (t or "wav").lower()))
        col_fmt.addWidget(self.fmt_combo)
        defaults_grid.addLayout(col_fmt, 1)

        # 标准化输入 toggle
        col_std = QVBoxLayout()
        col_std.setSpacing(4)
        col_std.addWidget(self._field_label("标准化输入"))
        self.std_toggle = ToggleSwitch()
        self.std_toggle.toggled.connect(lambda v: setattr(self, 'default_normalize', v))
        col_std.addWidget(self.std_toggle, 1, Qt.AlignmentFlag.AlignBottom)
        defaults_grid.addLayout(col_std, 0)
        root.addLayout(defaults_grid)

        # ====== DESCRIPTION ======
        desc_l = QVBoxLayout()
        desc_l.setSpacing(4)
        desc_l.addWidget(self._field_label("说明"))
        self.desc_edit = QTextEdit()
        self.desc_edit.setPlaceholderText("描述这个工作流适合什么输入和输出。")
        self.desc_edit.setMaximumHeight(64)
        self.desc_edit.textChanged.connect(
            lambda: setattr(self, 'description', self.desc_edit.toPlainText().strip()))
        desc_l.addWidget(self.desc_edit)
        root.addLayout(desc_l)

        # ====== STEPS SECTION ======
        steps_head = QHBoxLayout()
        steps_left = QVBoxLayout()
        steps_left.setSpacing(2)
        steps_left.addWidget(QLabel("<b>步骤</b>"))
        steps_left.addWidget(QLabel(
            "<span style='font-size:12px;color:#64748b'>"
            "每一步选择模型、输入来源和需要输出/传送的音轨。</span>"))
        steps_head.addLayout(steps_left, 1)
        self.add_step_btn = QPushButton("+ 添加步骤")
        self.add_step_btn.setStyleSheet(self._secondary_style())
        self.add_step_btn.clicked.connect(self._add_step)
        steps_head.addWidget(self.add_step_btn)
        root.addLayout(steps_head)

        # Scrollable step cards area
        self.steps_scroll = QScrollArea()
        self.steps_scroll.setWidgetResizable(True)
        self.steps_scroll.setMinimumHeight(120)
        self.steps_scroll.setMaximumHeight(340)
        self.steps_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.steps_scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)
        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setSpacing(10)
        self.steps_layout.addStretch(1)
        self.steps_scroll.setWidget(self.steps_container)
        root.addWidget(self.steps_scroll, 1)

        # ====== VALIDATION STATUS ======
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("font-size:12px;")
        root.addWidget(self.status_lbl)

        # ====== JSON PREVIEW ======
        json_head = QHBoxLayout()
        json_toggle = QPushButton("∨ 生成的工作流 JSON")
        json_toggle.setCheckable(True)
        json_toggle.setChecked(True)
        json_toggle.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                font-size:12px;
                color:#64748b;
                text-align:left;
                padding:4px 0;
            }
            QPushButton:checked { color:#334155; font-weight:600; }
        """)
        json_head.addWidget(json_toggle)
        root.addLayout(json_head)
        self.json_preview = QTextEdit()
        self.json_preview.setReadOnly(True)
        self.json_preview.setMaximumHeight(160)
        self.json_preview.setStyleSheet(
            "background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
            "font-family:'Consolas','Courier New',monospace;font-size:11px;"
            "color:#475569;padding:8px;")
        self.json_preview.setVisible(json_toggle.isChecked())
        json_toggle.toggled.connect(self.json_preview.setVisible)
        root.addWidget(self.json_preview)

        root.addStretch(0)

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size:11px;color:#64748b;font-weight:600;")
        return lbl

    @staticmethod
    def _primary_style() -> str:
        return (
            "QPushButton{background:#3b82f6;color:white;border:none;"
            "padding:7px 18px;border-radius:8px;font-weight:600;}"
            "QPushButton:hover{background:#2563eb;}"
            "QPushButton:disabled{background:#93c5fd;}"
        )

    @staticmethod
    def _outline_style() -> str:
        return (
            "QPushButton{background:transparent;color:#334155;border:1px solid #e2e8f0;"
            "padding:7px 16px;border-radius:8px;font-weight:500;}"
            "QPushButton:hover{border-color:#cbd5e1;background:#f8fafc;}"
        )

    @staticmethod
    def _secondary_style() -> str:
        return (
            "QPushButton{background:transparent;color:#3b82f6;border:1px solid #bfdbfe;"
            "padding:5px 14px;border-radius:8px;font-weight:500;}"
            "QPushButton:hover{background:#eff6ff;}"
        )

    @staticmethod
    def _danger_style() -> str:
        return (
            "QPushButton{background:transparent;color:#dc2626;border:1px solid #fecaca;"
            "padding:7px 16px;border-radius:8px;font-weight:500;}"
            "QPushButton:hover{border-color:#fca5a5;background:#fef2f2;}"
            "QPushButton:disabled{color:#fca5a5;border-color:#fee2e2;}"
        )

    # ---- data loading / saving ------------------------------------------
    def _load_workflow(self, entry: dict | None) -> None:
        if entry:
            try:
                draft = hydrate_simple_workflow(entry.get("definition"))
            except ValueError:
                draft = {"steps": [create_step_draft(0)], "ui": {}}
            self.name = entry.get("name", "")
            self.description = entry.get("description", "")
            self.default_device = draft.get("defaultDevice", "auto")
            self.default_format = draft.get("defaultFormat", "wav")
            self.default_normalize = draft.get("defaultNormalize", False)
            self.steps = draft.get("steps") or [create_step_draft(0)]
            self.expected_updated_at = entry.get("updatedAt")
            self.source_definition = copy.deepcopy(entry.get("definition")) if entry.get("definition") else None
            self.editing_badge.setText("正在编辑")
            # 复制/删除 apply to an already-saved workflow.
            self.dup_btn.setEnabled(True)
            self.del_btn.setEnabled(True)
        else:
            self.editing_badge.setText("新建中")
            self.name = ""
            self.description = ""
            self.default_device = "auto"
            self.default_format = "wav"
            self.default_normalize = False
            self.steps = [create_step_draft(0)]
            self.expected_updated_at = None
            self.source_definition = None
            # A brand-new draft can't be copied or deleted until it is saved.
            self.dup_btn.setEnabled(False)
            self.del_btn.setEnabled(False)
        self._sync_ui_to_state()

    def _sync_ui_to_state(self) -> None:
        self.name_edit.setText(self.name)
        self.desc_edit.setPlainText(self.description)
        idx = self.dev_combo.findText(self.default_device.upper(), Qt.MatchFlag.MatchStartsWith)
        self.dev_combo.setCurrentIndex(idx if idx >= 0 else 0)
        idx = self.fmt_combo.findText(self.default_format.upper(), Qt.MatchFlag.MatchStartsWith)
        self.fmt_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.std_toggle.setChecked(self.default_normalize)
        self._rebuild_step_cards()
        self._refresh_status()
        self._refresh_json()

    def _rebuild_step_cards(self) -> None:
        # Clear existing step cards (keep the stretch at end)
        while self.steps_layout.count() > 1:
            item = self.steps_layout.takeAt(0)
            w = item.widget() if item else None
            if w:
                w.deleteLater()
        for i, step in enumerate(self.steps):
            card = StepCard(i, step, self.app.models, self.steps)
            card.model_changed.connect(self._on_step_model_changed)
            card.removed.connect(self._remove_step_at)
            self.steps_layout.insertWidget(i, card)

    def _add_step(self) -> None:
        self.steps.append(create_step_draft(len(self.steps)))
        self._rebuild_step_cards()
        self._refresh_status()
        self._refresh_json()

    def _remove_step_at(self, index: int) -> None:
        if len(self.steps) <= 1:
            return
        del self.steps[index]
        # Re-index steps' default inputs
        for i, step in enumerate(self.steps):
            if not step.get("input") or step["input"] not in (
                    s.get("id", f"step_{j+1}") + "." + st
                    for j, s in enumerate(self.steps) for st in (s.get("stems") or [])
                    if j != i):
                if i == 0:
                    step["input"] = "input"
                elif step.get("input", "").startswith("step_"):
                    step["input"] = ""
        self._rebuild_step_cards()
        self._refresh_status()
        self._refresh_json()

    def _on_step_model_changed(self, card: StepCard, model_name: str) -> None:
        # Refresh all cards' input options and rebuild the changed card's stems.
        for i in range(self.steps_layout.count()):
            item = self.steps_layout.itemAt(i)
            w = item.widget() if item else None
            if isinstance(w, StepCard):
                w.refresh_input_options(self.steps)
        self._refresh_status()
        self._refresh_json()

    def _payload(self) -> dict:
        definition = self._generate_definition()
        return {
            "id": self.workflow_entry.get("id") if self.workflow_entry else None,
            "name": self.name.strip(),
            "description": self.description.strip(),
            "definition": definition,
            "expectedUpdatedAt": self.expected_updated_at,
        }

    def _generate_definition(self) -> dict:
        """Build the workflows.json version-1 definition from the form state.

        This is the exact shape stored in data/settings/workflows.json:
            {version:1, defaults:{device, output_format, model_dir,
             inference_params:{normalize}}, steps:[{id, model, input,
             stems, save}]}
        The JSON preview and the saved ``definition`` both use this format so
        the preview matches the selected workflow byte-for-byte in structure.
        """
        from collections import OrderedDict

        defaults: "OrderedDict[str, Any]" = OrderedDict()
        defaults["device"] = (self.default_device or "auto")
        defaults["output_format"] = (self.default_format or "wav")
        defaults["model_dir"] = None
        defaults["inference_params"] = OrderedDict(
            [("normalize", bool(self.default_normalize))])

        steps: list[dict] = []
        for i, step in enumerate(self.steps):
            sid = step.get("id") or f"step_{i + 1}"
            stems = [s for s in (step.get("stems") or [])]
            entry: "OrderedDict[str, Any]" = OrderedDict()
            entry["id"] = sid
            entry["model"] = (step.get("model") or "").strip()
            entry["input"] = step.get("input") or "input"
            entry["stems"] = stems
            entry["save"] = OrderedDict((stem, stem) for stem in stems)
            overlap = step.get("overlapSize")
            if overlap:
                entry["overlap_size"] = int(overlap)
            steps.append(entry)

        out: "OrderedDict[str, Any]" = OrderedDict()
        out["version"] = 1
        out["defaults"] = defaults
        out["steps"] = steps
        return out

    def _form_error(self) -> str:
        if not self.name.strip():
            return "名称不能为空"
        if not self.steps:
            return "至少需要一个步骤"
        downloaded = set(m.get("name", "") for m in self.app.models if model_downloaded(m))
        for i, step in enumerate(self.steps):
            label = f"步骤 {i + 1}"
            if not step.get("model", "").strip():
                return f"{label}: 请选择模型"
            if step["model"].strip() not in downloaded:
                return f"{label}: 模型尚未下载 ({step['model']})"
            inp = step.get("input", "").strip()
            if not inp:
                return f"{label}: 请选择输入来源"
            if not step.get("stems"):
                return f"{label}: 请选择至少一个输出轨道"
        # Validation summary
        try:
            defn = self._generate_definition()
            summary = get_workflow_validation_summary(defn)
            msg = workflow_validation_message(summary)
            if msg:
                return msg
        except Exception:
            pass
        return ""

    def _refresh_status(self) -> None:
        err = self._form_error()
        if err:
            self.status_lbl.setText(f"<span style='color:#ef4444'>⚠ {err}</span>")
            self.status_lbl.setToolTip(err)
        else:
            self.status_lbl.setText(
                '<span style="color:#16a34a">✓ 工作流配置有效</span>')
            self.status_lbl.setToolTip("")

    def _refresh_json(self) -> None:
        try:
            defn = self._generate_definition()
            import json
            text = json.dumps(defn, ensure_ascii=False, indent=2)
            self.json_preview.setPlainText(text)
        except Exception:
            self.json_preview.setPlainText("{  /* error */ }")

    # ---- signal emitters -------------------------------------------------
    def _emit_save(self) -> None:
        err = self._form_error()
        if err:
            QMessageBox.warning(self, "无法保存", err)
            return
        self.save.emit(self._payload())
