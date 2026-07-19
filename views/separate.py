"""Separation view — mirrors the original Vue SeparateView exactly.

Layout (top to bottom):
  1. Top row:  left = input source (file list with per-file delete)
              right = processing plan (model / workflow tabs with radio select)
  2. Middle:   progress panel (hidden by default; shows during batch run)
  3. Bottom:   output config (dir, format, layout, stems, start button)

Public API consumed by main_window.py:
  - SeparateView(app)
  - on_task_updated(task)
  - on_models(models)
  - on_model_info(info)
"""
from __future__ import annotations

import os
import json
import re
import time
import yaml
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal, QTimer
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QCheckBox, QListWidget, QListWidgetItem,
    QFrame, QTextEdit, QDialog, QFileDialog, QProgressBar,
    QButtonGroup, QRadioButton, QSpinBox, QDoubleSpinBox, QMessageBox,
    QStackedWidget, QSizePolicy, QScrollArea, QGroupBox,
    QSlider,
)

from ..config import AppConfig
from ..widgets.drop_area import DropArea
from ..task_model import Task


# ------------------------------------------------------------------ constants
FORMAT_OPTIONS = ["wav", "flac", "mp3", "m4a"]
WAV_DEPTHS = ["PCM_16", "PCM_24", "FLOAT"]
FLAC_DEPTHS = ["PCM_16", "PCM_24"]
BIT_RATES = ["128k", "192k", "256k", "320k", "512k"]

_AUDIO_EXTS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv",
}
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"}

_CARD_CSS = (
    "QFrame{background:#ffffff;border:1px solid #e2e8f0;"
    "border-radius:10px;}"
)

# List-item card: uniform fill so the whole block shares one background;
# the two text lines sit transparently on top of that same fill. No border
# (explicitly invisible) so a block reads as a single filled region and
# blocks are told apart by colour + spacing. The scroll area that holds
# the list also has its frame removed, so the list shows only the panel's
# single outer frame instead of a double frame.
_ITEM_CSS = (
    "QFrame{background:#f8fafc;border:none;border-radius:8px;}"
)
# Fixed height shared by input / model / workflow list blocks so every
# row is the same size regardless of content length.
_ITEM_H = 56
_SECTION_TITLE_CSS = "font-size:15px;font-weight:bold;color:#1e293b;"
_SUBTITLE_CSS = "font-size:12px;color:#64748b;"
_PATH_CSS = "font-size:11px;color:#94a3b8;"


def parse_instruments(value) -> list[str]:
    seen: set[str] = set()
    if isinstance(value, list):
        raw = value
    else:
        text = str(value or "").strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = __import__("json").loads(text)
                if isinstance(parsed, list):
                    raw = parsed
                else:
                    raw = [text]
            except Exception:
                raw = [text]
        else:
            raw = re.split(r"[,，;；/|\n]+", text)
    out: list[str] = []
    for item in raw:
        item = str(item).strip().strip("[](){}'\"").strip()
        if not item:
            continue
        k = item.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


# ================================================================ helpers
def _is_video(path: str) -> bool:
    return Path(path).suffix.lower() in _VIDEO_EXTS


def _file_type_label(path: str) -> str:
    return "视频" if _is_video(path) else "音频"


def _format_size(num: int) -> str:
    if not num:
        return "未知"
    val = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024:
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} PB"


# ===================================================== file-list item
class _FileListItem(QFrame):
    """One row in the input-source file list: name + type/path (2 lines)."""

    removed = Signal(str)  # path

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.setStyleSheet(_ITEM_CSS)
        self.setFixedHeight(_ITEM_H)
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 5, 10, 5)
        root.setSpacing(8)

        # text column: two lines (name + type/path), top-aligned
        col = QVBoxLayout()
        col.setSpacing(2)
        name = Path(self.path).name
        self.name_lbl = QLabel(name)
        self.name_lbl.setStyleSheet("font-size:13px;font-weight:bold;color:#1e293b;")
        col.addWidget(self.name_lbl)

        meta = f"{_file_type_label(self.path)}  ·  {self.path}"
        path_lbl = QLabel(meta)
        path_lbl.setStyleSheet(_PATH_CSS)
        path_lbl.setWordWrap(False)
        fm = path_lbl.fontMetrics()
        if fm.horizontalAdvance(meta) > 440:
            path_lbl.setText(fm.elidedText(meta, Qt.TextElideMode.ElideMiddle, 440))
        col.addWidget(path_lbl)

        root.addLayout(col, 1)

        # delete button (top-aligned with the text)
        btn = QPushButton("\u2715")  # ×
        btn.setFixedSize(24, 24)
        btn.setStyleSheet(
            "QPushButton{color:#94a3b8;border:none;background:transparent;"
            "font-size:14px;border-radius:12px;}"
            "QPushButton:hover{color:#ef4444;background:#fef2f2;}")
        btn.clicked.connect(lambda: self.removed.emit(self.path))
        root.addWidget(btn, 0, Qt.AlignTop)


# =================================================== model-card widget
class _ModelCard(QFrame):
    """One selectable model row in the processing-plan list."""

    selected = Signal(str)   # model name
    toggle_fav = Signal(str)  # model name
    edit_note = Signal(str)   # model name

    def __init__(self, model: dict, note: str = "", favorited: bool = False,
                 parent=None):
        super().__init__(parent)
        self.model = model
        self.setStyleSheet(_ITEM_CSS)
        self._build(note, favorited)

    def _build(self, note: str, favorited: bool) -> None:
        m = self.model
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 6)
        root.setSpacing(8)

        # info column: name + subtitle + (note, wrapped, adaptive height)
        col = QVBoxLayout()
        col.setSpacing(3)
        nm = m.get("name", "")
        name_lbl = QLabel(nm)
        name_lbl.setStyleSheet("font-size:13px;font-weight:bold;color:#1e293b;")
        name_lbl.setWordWrap(True)
        col.addWidget(name_lbl)

        cat = m.get("categoryCn") or m.get("category") or "未分类"
        size = _format_size(m.get("sizeBytes") or 0)
        arch = m.get("architecture") or m.get("modelType") or ""
        parts = [f"类别：{cat}", f"大小：{size}"]
        if arch:
            parts.append(f"架构：{arch}")
        sub = QLabel("  ·  ".join(parts))
        sub.setStyleSheet(_SUBTITLE_CSS)
        sub.setWordWrap(True)
        col.addWidget(sub)

        # 备注显示在模型名称之后，支持自动折行、自适应高度
        if note:
            note_lbl = QLabel(f"备注：{note}")
            note_lbl.setStyleSheet(
                "font-size:12px;color:#475569;background:#eef2f7;"
                "border-radius:4px;padding:2px 6px;")
            note_lbl.setWordWrap(True)
            note_lbl.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            col.addWidget(note_lbl)

        root.addLayout(col, 1)

        # right side: selection radio + favorite star + edit-note
        self.radio = QRadioButton()
        self.radio.setStyleSheet("QRadioButton{margin-right:4px;}")
        self.radio.toggled.connect(
            lambda checked: checked and self.selected.emit(m.get("name", "")))
        root.addWidget(self.radio, 0, Qt.AlignVCenter)

        self.fav_btn = QPushButton("\u2605" if favorited else "\u2606")
        self.fav_btn.setFixedSize(28, 28)
        star_color = "#eab308" if favorited else "#94a3b8"
        self.fav_btn.setStyleSheet(
            "QPushButton{border:none;font-size:16px;color:" + star_color +
            ";}QPushButton:hover{color:#eab308;}")
        self.fav_btn.setToolTip("收藏 / 取消收藏")
        self.fav_btn.clicked.connect(
            lambda: self.toggle_fav.emit(m.get("name", "")))
        root.addWidget(self.fav_btn, 0, Qt.AlignVCenter)

        self.edit_btn = QPushButton("\u270e")
        self.edit_btn.setFixedSize(28, 28)
        self.edit_btn.setStyleSheet(
            "QPushButton{border:none;font-size:15px;color:#64748b;}"
            "QPushButton:hover{color:#1e293b;}")
        self.edit_btn.setToolTip("编辑备注")
        self.edit_btn.clicked.connect(
            lambda: self.edit_note.emit(m.get("name", "")))
        root.addWidget(self.edit_btn, 0, Qt.AlignVCenter)

    def mousePressEvent(self, ev) -> None:
        self.radio.setChecked(True)
        super().mousePressEvent(ev)


# ================================================== workflow card item
class _WorkflowListItem(QFrame):
    """One selectable workflow row."""

    selected = Signal(str)  # workflow id

    def __init__(self, wf: dict, parent=None):
        super().__init__(parent)
        self.wf = wf
        self.setStyleSheet(_ITEM_CSS)
        self.setFixedHeight(_ITEM_H)
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 5, 10, 5)
        col = QVBoxLayout()
        col.setSpacing(2)
        nm = self.wf.get("name", "")
        name_lbl = QLabel(nm)
        name_lbl.setStyleSheet("font-size:13px;font-weight:bold;color:#1e293b;")
        name_lbl.setWordWrap(False)
        nfm = name_lbl.fontMetrics()
        if nfm.horizontalAdvance(nm) > 360:
            name_lbl.setText(nfm.elidedText(nm, Qt.TextElideMode.ElideRight, 360))
        col.addWidget(name_lbl)
        desc_text = self.wf.get("description", "") or "暂无说明"
        desc = QLabel(desc_text)
        desc.setStyleSheet(_SUBTITLE_CSS)
        desc.setWordWrap(False)
        dfm = desc.fontMetrics()
        if dfm.horizontalAdvance(desc_text) > 400:
            desc.setText(dfm.elidedText(desc_text, Qt.TextElideMode.ElideRight, 400))
        col.addWidget(desc)
        root.addLayout(col, 1)
        self.radio = QRadioButton()
        self.radio.toggled.connect(
            lambda checked: checked and self.selected.emit(self.wf.get("id", "")))
        root.addWidget(self.radio, 0, Qt.AlignVCenter)

    def mousePressEvent(self, ev) -> None:
        self.radio.setChecked(True)
        super().mousePressEvent(ev)


# ================================================================= main
class SeparateView(QWidget):
    def __init__(self, app: "App") -> None:
        super().__init__()
        self.app = app
        self.run_mode = "model"
        self.input_files: list[str] = []
        self.temporary_output_dir = ""
        self.save_as_folder = False  # default to flat
        self.selected_stems: list[str] = []
        self.available_stems: list[str] = []
        # inference params
        self.use_tta = False
        self.debug = False
        self.batch_size = 1
        # 默认 50% 重叠(num_overlap=2)。注意：worker 的 _enrich_inference_params_for_model
        # 会用 num_overlap 重新换算 overlap_size 并【覆盖】yaml 里的 overlap_size，
        # 所以这里才是真正生效的重叠率。75%(num_overlap=4) 在 12GB 卡上分离长文件
        # (~39 分钟) 会因 pymss 预分配全部 chunk 窗口缓冲而 OOM(实测峰值 10.88GiB)；
        # 50%(num_overlap=2) 峰值 10.79GiB 可装下。想要 MSST 级 75% 质量需先切音频。
        self.num_overlap = 2  # 默认 50% 重叠，兼容 12GB 显存的长文件分离
        self.chunk_size = 0  # 0 = 使用模型自带的分块大小，不强制全局值
        self.overlap_size = 0  # 重叠样本数（= chunk_size - chunk_size // num_overlap）
        # suppress 标志：程序化 setattr 时不触发 num_overlap 的"手动编辑"处理
        self._suppress_sync = False
        self.standardize = False
        self.normalize = False
        self.slow_mode = False  # 慢速模式（省显存）：强制 use_complete_fast_path=False
        self.engine = "pymss"  # 当前选择的引擎
        self.window_size = 0
        self.aggression = 0
        self.enable_post_process = False
        self.post_process_threshold = 0.0
        self.high_end_process = False
        self.model_search = ""
        self.model_category = ""
        self.workflow_search = ""
        self.selected_task_id: str | None = None
        # internal refs
        self._model_cards: list[_ModelCard] = []
        self._wf_cards: list[_WorkflowListItem] = []
        self._selected_model_name: str | None = None
        self._selected_wf_id: str | None = None
        self._build_ui()
        self._load_models()
        self._load_workflows()

    # ============================================================== UI build
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        # ---- Row 1: Input Source | Processing Plan ----
        top_row = QHBoxLayout()
        top_row.setSpacing(14)
        top_row.addWidget(self._build_input_panel(), 1)
        top_row.addWidget(self._build_plan_panel(), 1)
        root.addLayout(top_row)

        # ---- Row 2: Progress (hidden by default) ----
        self.progress_panel = self._build_progress_panel()
        self.progress_panel.hide()
        root.addWidget(self.progress_panel)

        # ---- Row 3: Output Config ----
        root.addWidget(self._build_output_panel())

    # ------------------------------------------------- INPUT SOURCE PANEL
    def _build_input_panel(self) -> QFrame:
        panel = QFrame()
        panel.setStyleSheet(_CARD_CSS)
        v = QVBoxLayout(panel)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        # header
        hdr = QHBoxLayout()
        icon = QLabel("\U0001f3b5")  # musical notes
        icon.setStyleSheet("font-size:18px;")
        hdr.addWidget(icon)
        title = QLabel("输入来源")
        title.setStyleSheet(_SECTION_TITLE_CSS)
        hdr.addWidget(title)
        hdr.addStretch(1)
        v.addLayout(hdr)

        sub = QLabel("选择单个或多个音频/视频文件，或选择整个文件夹作为分离来源。")
        sub.setStyleSheet(_SUBTITLE_CSS)
        sub.setWordWrap(True)
        v.addWidget(sub)

        # action buttons
        acts = QHBoxLayout()
        acts.setSpacing(8)
        btn_pick = QPushButton("\U0001f3b5 选择音频/视频文件")
        btn_pick.setMinimumHeight(32)
        btn_pick.clicked.connect(self._choose_files)
        acts.addWidget(btn_pick)
        btn_folder = QPushButton("\U0001f4c1 选择输入文件夹")
        btn_folder.setMinimumHeight(32)
        btn_folder.clicked.connect(self._choose_folder)
        acts.addWidget(btn_folder)
        v.addLayout(acts)

        # drag-and-drop zone
        self.input_drop_area = DropArea()
        self.input_drop_area.files_dropped.connect(self._on_dropped)
        v.addWidget(self.input_drop_area)

        # list header
        list_hdr = QHBoxLayout()
        self.input_summary = QLabel("概览选曲  已选 0 首")
        self.input_summary.setStyleSheet("font-size:12px;font-weight:bold;color:#475569;")
        list_hdr.addWidget(self.input_summary)
        list_hdr.addStretch(1)
        btn_clear = QPushButton("清空列表")
        btn_clear.setStyleSheet(
            "QPushButton{color:#ef4444;border:none;background:transparent;"
            "font-size:12px;padding:2px 8px;}"
            "QPushButton:hover{text-decoration:underline;}")
        btn_clear.clicked.connect(self._clear_inputs)
        list_hdr.addWidget(btn_clear)
        v.addLayout(list_hdr)

        # file list (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumHeight(160)
        self.file_list_widget = QWidget()
        self.file_list_layout = QVBoxLayout(self.file_list_widget)
        self.file_list_layout.setContentsMargins(0, 0, 0, 0)
        self.file_list_layout.setSpacing(8)
        # empty state
        self.empty_input_tip = QLabel("暂无文件，请添加音频或视频文件")
        self.empty_input_tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_input_tip.setStyleSheet("color:#94a3b8;font-size:13px;padding:20px;")
        self.file_list_layout.addWidget(self.empty_input_tip)
        scroll.setWidget(self.file_list_widget)
        v.addWidget(scroll, 1)

        return panel

    # ----------------------------------------------- PROCESSING PLAN PANEL
    def _build_plan_panel(self) -> QFrame:
        panel = QFrame()
        panel.setStyleSheet(_CARD_CSS)
        v = QVBoxLayout(panel)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        # header
        hdr = QHBoxLayout()
        icon = QLabel("\U0001f4ca")  # bar chart / cube
        icon.setStyleSheet("font-size:18px;")
        hdr.addWidget(icon)
        title = QLabel("推理方案")
        title.setStyleSheet(_SECTION_TITLE_CSS)
        hdr.addWidget(title)
        hdr.addStretch(1)
        link = QPushButton("前往模型库 \u2197")
        link.setStyleSheet(
            "QPushButton{color:#3b82f6;border:none;background:transparent;"
            "font-size:12px;text-decoration:underline;}"
            "QPushButton:hover{text-decoration:none;}")
        link.clicked.connect(self._on_plan_link)
        hdr.addWidget(link)
        self.plan_link = link
        v.addLayout(hdr)

        sub = QLabel("仅显示已下载且支持推理的模型。")
        sub.setStyleSheet(_SUBTITLE_CSS)
        v.addWidget(sub)

        # tabs
        tabs = QHBoxLayout()
        tabs.setSpacing(0)
        self.tab_model = QPushButton("单模型分离")
        self.tab_workflow = QPushButton("工作流推理")
        for b in (self.tab_model, self.tab_workflow):
            b.setCheckable(True)
            b.setMinimumHeight(30)
            b.setStyleSheet(
                "QPushButton{border:none;background:transparent;"
                "font-size:13px;color:#64748b;padding:0 16px;}"
                "QPushButton:checked{color:#2563eb;font-weight:bold;"
                "border-bottom:2px solid #2563eb;}")
        self.tab_model.setChecked(True)
        self.tab_model.clicked.connect(lambda: self._set_mode("model"))
        self.tab_workflow.clicked.connect(lambda: self._set_mode("workflow"))
        tabs.addWidget(self.tab_model)
        tabs.addWidget(self.tab_workflow)
        tabs.addStretch(1)
        v.addLayout(tabs)

        # search + filter bar (model mode)
        search_row = QHBoxLayout()
        self.plan_search = QLineEdit()
        self.plan_search.setPlaceholderText(
            "搜索模型名称、用途、目标音源或架杆")
        self.plan_search.textChanged.connect(self._on_plan_search)
        search_row.addWidget(self.plan_search, 1)
        self.plan_filter = QComboBox()
        self.plan_filter.addItem("全部分类", "")
        self.plan_filter.setMinimumWidth(150)
        self.plan_filter.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.plan_filter.currentIndexChanged.connect(self._on_plan_filter)
        search_row.addWidget(self.plan_filter)
        # 排序下拉：复制模型库的排序选项（含「收藏优先」）
        self.plan_sort = QComboBox()
        self.plan_sort.setMinimumWidth(130)
        self.plan_sort.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        for label, val in [
            ("默认排序", "default"), ("收藏优先", "favorite"),
            ("名称 A→Z", "name-asc"), ("名称 Z→A", "name-desc"),
            ("大小 大→小", "size-desc"), ("大小 小→大", "size-asc"),
            ("分类", "category"), ("类型", "type"), ("已下载优先", "downloaded"),
        ]:
            self.plan_sort.addItem(label, val)
        self.plan_sort.setCurrentIndex(0)
        self.plan_sort.currentIndexChanged.connect(self._on_plan_sort)
        search_row.addWidget(self.plan_sort)
        v.addLayout(search_row)

        # stacked content: model list | workflow list
        self.plan_stack = QStackedWidget()
        self.plan_stack.addWidget(self._build_model_list())
        self.plan_stack.addWidget(self._build_workflow_list())
        v.addWidget(self.plan_stack, 1)

        return panel

    def _build_model_list(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        self.model_list_layout = QVBoxLayout(w)
        self.model_list_layout.setContentsMargins(0, 0, 0, 0)
        self.model_list_layout.setSpacing(8)
        self.model_empty_tip = QLabel("未找到已下载模型，请到「模型库」下载。")
        self.model_empty_tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_empty_tip.setStyleSheet("color:#94a3b8;font-size:13px;padding:20px;")
        self.model_list_layout.addWidget(self.model_empty_tip)
        scroll.setWidget(w)
        return scroll

    def _build_workflow_list(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        w = QWidget()
        self.wf_list_layout = QVBoxLayout(w)
        self.wf_list_layout.setContentsMargins(0, 0, 0, 0)
        self.wf_list_layout.setSpacing(8)
        self.wf_empty_tip = QLabel("暂无可选工作流。")
        self.wf_empty_tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wf_empty_tip.setStyleSheet("color:#94a3b8;font-size:13px;padding:20px;")
        self.wf_list_layout.addWidget(self.wf_empty_tip)
        scroll.setWidget(w)
        return scroll

    # ---------------------------------------------------- PROGRESS PANEL
    def _build_progress_panel(self) -> QFrame:
        panel = QFrame()
        panel.setStyleSheet(_CARD_CSS)
        v = QVBoxLayout(panel)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        # header row
        hdr = QHBoxLayout()
        self.progress_title = QLabel("正在分离 0 / 0")
        self.progress_title.setStyleSheet(_SECTION_TITLE_CSS)
        hdr.addWidget(self.progress_title)
        hdr.addStretch(1)
        self.progress_state_badge = QLabel("准备中")
        self.progress_state_badge.setStyleSheet(
            "font-size:11px;color:#2563eb;background:#eff6ff;"
            "padding:2px 10px;border-radius:4px;")
        hdr.addWidget(self.progress_state_badge)
        v.addLayout(hdr)

        # current file
        self.current_file_lbl = QLabel("")
        self.current_file_lbl.setStyleSheet("font-size:12px;color:#475569;")
        v.addWidget(self.current_file_lbl)

        # progress bar row
        prow = QHBoxLayout()
        plbl = QLabel("整体进度")
        plbl.setStyleSheet("font-size:12px;color:#64748b;")
        plbl.setFixedWidth(60)
        prow.addWidget(plbl)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumHeight(22)
        prow.addWidget(self.progress_bar, 1)
        self.progress_pct = QLabel("0%")
        self.progress_pct.setStyleSheet("font-size:14px;font-weight:bold;color:#1e293b;")
        self.progress_pct.setFixedWidth(40)
        prow.addWidget(self.progress_pct)
        v.addLayout(prow)

        # status message
        self.progress_msg = QLabel("")
        self.progress_msg.setStyleSheet("font-size:12px;color:#64748b;")
        v.addWidget(self.progress_msg)

        # output path + elapsed
        info = QHBoxLayout()
        self.elapsed_lbl = QLabel("\u23f1 0s")
        self.elapsed_lbl.setStyleSheet("font-size:11px;color:#94a3b8;")
        info.addWidget(self.elapsed_lbl)
        self.output_path_lbl = QLabel("")
        self.output_path_lbl.setStyleSheet("font-size:11px;color:#94a3b8;")
        info.addWidget(self.output_path_lbl, 1)
        v.addLayout(info)

        # action buttons (right-aligned)
        acts = QHBoxLayout()
        acts.addStretch(1)
        self.btn_log = QPushButton("\U0001f4dd 日志")
        self.btn_log.setStyleSheet(
            "QPushButton{border:1px solid #e2e8f0;border-radius:6px;"
            "padding:6px 14px;background:#fff;color:#475569;}"
            "QPushButton:hover{background:#f8fafc;}")
        self.btn_log.clicked.connect(self._show_logs)
        acts.addWidget(self.btn_log)
        self.btn_cancel_job = QPushButton("取消任务")
        self.btn_cancel_job.setStyleSheet(
            "QPushButton{border:1px solid #fecaca;border-radius:6px;"
            "padding:6px 14px;background:#fef2f2;color:#dc2626;"
            "font-weight:bold;}")
        self.btn_cancel_job.clicked.connect(self._cancel_current)
        acts.addWidget(self.btn_cancel_job)
        v.addLayout(acts)

        # completion summary (shown when done)
        self.summary_widget = QWidget()
        sum_v = QVBoxLayout(self.summary_widget)
        sum_v.setContentsMargins(0, 0, 0, 0)
        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet("font-size:14px;font-weight:bold;color:#1e293b;")
        sum_v.addWidget(self.summary_lbl)
        # output file list
        self.output_file_list = QListWidget()
        self.output_file_list.setMaximumHeight(140)
        sum_v.addWidget(QLabel("<b>输出文件</b>"))
        sum_v.addWidget(self.output_file_list)
        self.summary_widget.hide()
        v.addWidget(self.summary_widget)

        return panel

    # ------------------------------------------------------ OUTPUT PANEL
    def _build_output_panel(self) -> QFrame:
        panel = QFrame()
        panel.setStyleSheet(_CARD_CSS)
        v = QVBoxLayout(panel)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        # header
        hdr = QHBoxLayout()
        icon = QLabel("\U0001f4cb")  # clipboard / list
        icon.setStyleSheet("font-size:18px;")
        hdr.addWidget(icon)
        title = QLabel("任务配置")
        title.setStyleSheet(_SECTION_TITLE_CSS)
        hdr.addWidget(title)
        sub = QLabel("配置输出目录、输出音轨与分离推理参数")
        sub.setStyleSheet(_SUBTITLE_CSS)
        hdr.addWidget(sub)
        hdr.addStretch(1)
        v.addLayout(hdr)

        # ---- Row 1: 输出目录 | 输出格式 | 输出布局 ----
        r1 = QHBoxLayout()
        r1.setSpacing(10)
        dlbl = QLabel("输出目录")
        dlbl.setFixedWidth(56)
        dlbl.setStyleSheet("font-size:12px;color:#475569;")
        r1.addWidget(dlbl)
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("默认输出到 results 目录")
        self.out_dir_edit.textChanged.connect(
            lambda t: setattr(self, "temporary_output_dir", t))
        r1.addWidget(self.out_dir_edit, 1)
        btn_browse_out = QPushButton("选择")
        btn_browse_out.setFixedWidth(56)
        btn_browse_out.clicked.connect(self._browse_output)
        r1.addWidget(btn_browse_out)
        r1.addWidget(self._vsep())
        flbl = QLabel("格式")
        flbl.setFixedWidth(32)
        flbl.setStyleSheet("font-size:12px;color:#475569;")
        r1.addWidget(flbl)
        self.format_sel = QComboBox()
        self.format_sel.addItems(FORMAT_OPTIONS)
        self.format_sel.setCurrentText(
            self.app.config["default_output_format"])
        self.format_sel.currentTextChanged.connect(self._on_format)
        self.format_sel.setFixedWidth(80)
        r1.addWidget(self.format_sel)
        llbl = QLabel("布局")
        llbl.setFixedWidth(32)
        llbl.setStyleSheet("font-size:12px;color:#475569;")
        r1.addWidget(llbl)
        self.layout_sel = QComboBox()
        self.layout_sel.addItem("分层", "folders")
        self.layout_sel.addItem("平铺", "flat")
        self.layout_sel.setCurrentIndex(1)  # default to flat
        self.layout_sel.setFixedWidth(80)
        self.layout_sel.currentIndexChanged.connect(
            lambda i: setattr(self, "save_as_folder",
                              self.layout_sel.currentData() == "folders"))
        r1.addWidget(self.layout_sel)
        v.addLayout(r1)

        # ---- Row 2: 高级设置（推理参数） ----
        r2 = QHBoxLayout()
        r2.setSpacing(8)
        adv_lbl = QLabel("高级设置：")
        adv_lbl.setStyleSheet("font-size:12px;font-weight:bold;color:#475569;")
        r2.addWidget(adv_lbl)
        self.sb_batch = self._make_int_spin(0, 32, 70)
        self.sb_chunk = self._make_int_spin(0, 1048576, 92)
        self.sb_overlap = self._make_int_spin(0, 1048576, 92)
        self.sb_num_overlap = self._make_int_spin(0, 128, 70)
        # 块大小/重叠大小按 0.5 秒为步长调节（默认按 44.1kHz 计 = 22050 样本；
        # 选中模型后会按该模型真实采样率刷新为 sr//2）。不用默认的 ±1。
        _half_sec = 22050
        self.sb_chunk.setSingleStep(_half_sec)
        self.sb_overlap.setSingleStep(_half_sec)
        self.sb_batch.setToolTip(
            "batch_size：每次前向推理的音频块数（仅影响速度，不影响质量）")
        self.sb_chunk.setToolTip(
            "chunk_size：单块样本数（=采样率×秒，如 44100×11≈485100）")
        self.sb_overlap.setToolTip("overlap_size：相邻块之间的重叠样本数")
        self.sb_num_overlap.setToolTip(
            "num_overlap：重叠块数，重叠率 = 1 - 1/num_overlap（如 4 → 75%）")
        self.sb_batch.valueChanged.connect(
            lambda v: setattr(self, "batch_size", v))
        self.sb_chunk.valueChanged.connect(
            lambda v: setattr(self, "chunk_size", v))
        self.sb_overlap.valueChanged.connect(
            lambda v: setattr(self, "overlap_size", v))
        self.sb_num_overlap.valueChanged.connect(
            lambda v: (setattr(self, "num_overlap", v),
                       self._on_num_overlap_changed(v)))
        r2.addLayout(self._param_row("批大小", self.sb_batch))
        r2.addLayout(self._param_row("块大小", self.sb_chunk))
        r2.addLayout(self._param_row("重叠大小", self.sb_overlap))
        r2.addLayout(self._param_row("重叠块数", self.sb_num_overlap))
        r2.addWidget(self._vsep())
        self.btn_adv_default = QPushButton("默认")
        self.btn_adv_read = QPushButton("读取")
        self.btn_adv_save = QPushButton("保存")
        self.btn_adv_default.clicked.connect(self._reset_advanced_defaults)
        self.btn_adv_read.clicked.connect(self._read_advanced_from_yaml)
        self.btn_adv_save.clicked.connect(self._write_advanced_to_yaml)
        r2.addWidget(self.btn_adv_default)
        r2.addWidget(self.btn_adv_read)
        r2.addWidget(self.btn_adv_save)
        r2.addStretch(1)
        v.addLayout(r2)

        # ---- Row 3: 运行选项 ----
        r3 = QHBoxLayout()
        r3.setSpacing(16)
        self.cb_tta = QCheckBox("TTA (测试时增强)")
        self.cb_debug = QCheckBox("调试日志")
        self.cb_standardize = QCheckBox("输入标准化")
        self.cb_normalize = QCheckBox("输出归一化")
        self.cb_slow_mode = QCheckBox("慢速模式(省显存)")
        self.cb_tta.toggled.connect(lambda c: setattr(self, "use_tta", c))
        self.cb_debug.toggled.connect(lambda c: setattr(self, "debug", c))
        self.cb_standardize.toggled.connect(
            lambda c: setattr(self, "standardize", c))
        self.cb_normalize.toggled.connect(
            lambda c: setattr(self, "normalize", c))
        self.cb_slow_mode.toggled.connect(
            lambda c: setattr(self, "slow_mode", c))
        r3.addWidget(self.cb_tta)
        r3.addWidget(self.cb_debug)
        r3.addWidget(self.cb_standardize)
        r3.addWidget(self.cb_normalize)
        r3.addWidget(self.cb_slow_mode)
        # 引擎选择（单选框）
        r3.addWidget(self._vsep())
        eng_lbl = QLabel("引擎：")
        eng_lbl.setStyleSheet("font-size:12px;color:#475569;")
        r3.addWidget(eng_lbl)
        self.engine_group = QButtonGroup(self)
        self.engine_radio_widget = QWidget()
        self.engine_radio_layout = QHBoxLayout(self.engine_radio_widget)
        self.engine_radio_layout.setContentsMargins(0, 0, 0, 0)
        self.engine_radio_layout.setSpacing(4)
        r3.addWidget(self.engine_radio_widget)
        r3.addStretch(1)
        v.addLayout(r3)

        # ---- Row 4: 输出音轨 ----
        srow = QHBoxLayout()
        slbl = QLabel("输出音轨")
        slbl.setFixedWidth(56)
        slbl.setStyleSheet("font-size:12px;color:#475569;")
        srow.addWidget(slbl)
        self.stems_box = QFrame()
        sb_lay = QGridLayout(self.stems_box)
        sb_lay.setContentsMargins(0, 0, 0, 0)
        sb_lay.setSpacing(8)
        srow.addWidget(self.stems_box, 1)
        v.addLayout(srow)

        # advanced params start disabled until a model is selected
        self._set_advanced_enabled(False)
        self._sync_advanced_widgets()

        # bottom buttons bar
        bar = QHBoxLayout()
        self.status_tip_label = QLabel("")
        self.status_tip_label.setStyleSheet(
            "font-size:12px;color:#2563eb;padding:0 8px;")
        self.status_tip_label.setFixedHeight(20)
        bar.addWidget(self.status_tip_label)
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(
            lambda: self.status_tip_label.setText(""))
        bar.addStretch(1)
        btn_open_out = QPushButton("\U0001f4c1 打开输出目录")
        btn_open_out.setStyleSheet(
            "QPushButton{border:1px solid #e2e8f0;border-radius:6px;"
            "padding:8px 18px;background:#fff;color:#475569;}")
        btn_open_out.clicked.connect(self._reveal_output)
        bar.addWidget(btn_open_out)
        self.start_btn = QPushButton("\u25b6 开始分离")
        self.start_btn.setStyleSheet(
            "QPushButton{background:#2563eb;color:white;border:none;"
            "border-radius:6px;padding:8 24px;font-weight:bold;font-size:14px;}"
            "QPushButton:hover{background:#1d4ed8;}"
            "QPushButton:disabled{background:#93c5fd;}")
        self.start_btn.clicked.connect(self.start)
        bar.addWidget(self.start_btn)
        v.addLayout(bar)

        return panel

    # ============================================================ models
    def on_models(self, models: list[dict]) -> None:
        self._populate_model_categories()
        self._refresh_model_list()

    def _populate_model_categories(self) -> None:
        """Mirror the model library's category dropdown."""
        cats: list[tuple[str, str]] = []
        seen: set[str] = set()
        for m in self.app.models:
            c = m.get("category")
            cn = m.get("categoryCn") or c
            if c and c not in seen:
                seen.add(c)
                cats.append((cn or c, c))
        self.plan_filter.blockSignals(True)
        self.plan_filter.clear()
        self.plan_filter.addItem("全部分类", "")
        for cn, c in cats:
            self.plan_filter.addItem(cn, c)
        idx = self.plan_filter.findData(self.model_category)
        self.plan_filter.setCurrentIndex(idx if idx >= 0 else 0)
        self.plan_filter.blockSignals(False)

    def _load_notes(self) -> dict:
        return self._read_prefs()[1]

    def _read_prefs(self) -> tuple[set, dict]:
        """Read favorites + notes from the shared model_prefs.json.

        The model library writes the same file, so a favorite / remark
        edited there is visible here and vice-versa. We re-read on every
        call (the file is tiny) so cross-view edits stay fresh.
        """
        favorites: set = set()
        notes: dict = {}
        p = Path(__file__).resolve().parent.parent / "model_prefs.json"
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                favorites = set(data.get("favorites", []) or [])
                notes = data.get("notes", {}) or {}
            except Exception:
                pass
        return favorites, notes

    def _save_prefs(self, favorites: set, notes: dict) -> None:
        """Persist favorites + notes back to model_prefs.json.

        Existing keys (notably ``overrides`` written by the model library)
        are preserved — we only merge, never overwrite the whole file.
        """
        p = Path(__file__).resolve().parent.parent / "model_prefs.json"
        data: dict = {}
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data["favorites"] = sorted(favorites)
        data["notes"] = notes
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    def on_model_info(self, info: dict) -> None:
        self._on_model_info(info)

    def _load_models(self) -> None:
        self.app.bridge.list_models(self.app.config.models_dir())

    def _refresh_model_list(self) -> None:
        # remove old cards, but KEEP the persistent empty-tip (index 0).
        # Removing it (and deleteLater-ing it) was the source of the
        # "Internal C++ object already deleted" error on later refreshes.
        while self.model_list_layout.count() > 1:
            item = self.model_list_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        self._model_cards.clear()

        q = self.plan_search.text().strip().lower()
        cat = self.plan_filter.currentData() or ""
        favorites, notes = self._read_prefs()
        shown = []
        for m in self.app.models:
            # only show downloaded models
            if not m.get("downloaded"):
                continue
            hay = " ".join(str(m.get(k, "")) for k in
                          ("name", "architecture", "modelType", "targetStem",
                           "configTargetInstrument", "category", "categoryCn",
                           "classificationBasis"))
            if q and q not in hay.lower():
                continue
            if cat and cat not in (str(m.get("category", "")).lower(),
                                   str(m.get("primaryCategory", "")).lower(),
                                   str(m.get("secondaryCategory", "")).lower()):
                continue
            shown.append(m)

        # ---- sort (mirrors the model library's sort dropdown) ----------
        key = self.plan_sort.currentData() if hasattr(self, "plan_sort") else "default"
        if key == "favorite":
            shown.sort(key=lambda m: (m.get("name", "") not in favorites,
                                      m.get("name", "").lower()))
        elif key == "name-asc":
            shown.sort(key=lambda m: m.get("name", "").lower())
        elif key == "name-desc":
            shown.sort(key=lambda m: m.get("name", "").lower(), reverse=True)
        elif key == "size-desc":
            shown.sort(key=lambda m: m.get("sizeBytes") or 0, reverse=True)
        elif key == "size-asc":
            shown.sort(key=lambda m: m.get("sizeBytes") or 0)
        elif key == "category":
            shown.sort(key=lambda m: (m.get("category") or m.get("categoryCn") or "").lower())
        elif key == "type":
            shown.sort(key=lambda m: (m.get("modelType") or m.get("architecture", "")).lower())
        elif key == "downloaded":
            shown.sort(key=lambda m: (not m.get("downloaded"), m.get("name", "").lower()))

        bg = QButtonGroup(self)
        for m in shown:
            name = m.get("name", "")
            note = notes.get(name, "")
            card = _ModelCard(m, note, name in favorites)
            card.selected.connect(self._on_model_selected)
            card.toggle_fav.connect(self._on_toggle_fav)
            card.edit_note.connect(self._on_edit_note)
            bg.addButton(card.radio)
            self.model_list_layout.addWidget(card)
            self._model_cards.append(card)

        self.model_empty_tip.setVisible(len(shown) == 0)
        # add stretch at bottom so cards align top
        self.model_list_layout.addStretch(1)

    def _on_plan_sort(self) -> None:
        if self.run_mode == "model":
            self._refresh_model_list()

    def _on_toggle_fav(self, name: str) -> None:
        favorites, notes = self._read_prefs()
        if name in favorites:
            favorites.discard(name)
        else:
            favorites.add(name)
        self._save_prefs(favorites, notes)
        self._refresh_model_list()

    def _on_edit_note(self, name: str) -> None:
        favorites, notes = self._read_prefs()
        dlg = QDialog(self)
        dlg.setWindowTitle(f"编辑备注 — {name}")
        dlg.setMinimumWidth(420)
        v = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setPlainText(notes.get(name, ""))
        te.setMaximumHeight(120)
        v.addWidget(te)
        bar = QHBoxLayout()
        b_ok = QPushButton("保存")
        b_cancel = QPushButton("取消")
        bar.addWidget(b_ok)
        bar.addWidget(b_cancel)
        v.addLayout(bar)

        def _save() -> None:
            notes[name] = te.toPlainText().strip()
            self._save_prefs(favorites, notes)
            dlg.accept()

        b_ok.clicked.connect(_save)
        b_cancel.clicked.connect(dlg.reject)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._refresh_model_list()

    def _on_model_selected(self, name: str) -> None:
        self._selected_model_name = name
        self.app.selected_model = name
        self.app.bridge.model_info(name, self.app.config.models_dir(),
                                    channel="sep_model_info")
        self._update_status()

    def _on_model_info(self, info: dict) -> None:
        name = info.get("name")
        if not name:
            return
        self.app.model_infos[name] = info
        if name == self.app.selected_model:
            self._rebuild_stems(info)
            self._read_advanced_from_yaml()
            self._update_engine_combo(info)

    def _update_engine_combo(self, info: dict) -> None:
        """根据当前模型的架构更新引擎单选框。"""
        from engine import engines_for_architecture, default_engine_for, engine_label
        arch = (info.get("architecture") or info.get("modelType") or "").strip().lower()
        engines = engines_for_architecture(arch)
        # 清除旧的 radio
        for rb in self.engine_group.buttons():
            self.engine_group.removeButton(rb)
            rb.deleteLater()
        # 清除布局中的旧 widget
        while self.engine_radio_layout.count():
            item = self.engine_radio_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # 创建新 radio
        default = default_engine_for(arch)
        for e in engines:
            rb = QRadioButton(engine_label(e))
            rb.setStyleSheet("font-size:12px;")
            if len(engines) == 1:
                rb.setEnabled(False)
                rb.setToolTip("此架构仅支持此引擎，不可切换")
            else:
                rb.setToolTip("选择模型运行引擎")
            self.engine_group.addButton(rb)
            self.engine_radio_layout.addWidget(rb)
            if e == default:
                rb.setChecked(True)
                self.engine = e
        self.engine_group.buttonToggled.connect(self._on_engine_radio_toggled)

    def _on_engine_radio_toggled(self, btn: QRadioButton, checked: bool) -> None:
        """引擎单选框切换时。"""
        if not checked:
            return
        from engine import engine_label
        # 通过 label 反查 engine name
        label = btn.text()
        for name, lbl in {n: engine_label(n) for n in ["pymss", "msst"]}.items():
            if lbl == label:
                self.engine = name
                return

    def _rebuild_stems(self, info: dict) -> None:
        lay = self.stems_box.layout()
        while lay.count():
            lay.takeAt(0).widget().deleteLater()
        stems = parse_instruments(info.get("configInstruments"))
        self.available_stems = stems
        if not stems:
            lbl = QLabel("(全部)")
            lbl.setStyleSheet("font-size:12px;color:#94a3b8;")
            lay.addWidget(lbl)
            self.selected_stems = []
            return
        for i, s in enumerate(stems):
            cb = QCheckBox(s)
            cb.setChecked(True)
            cb.toggled.connect(lambda c, stem=s: self._on_stem_toggle(stem, c))
            lay.addWidget(cb, i // 6, i % 6)

    def _on_stem_toggle(self, stem: str, checked: bool) -> None:
        if checked and stem not in self.selected_stems:
            self.selected_stems.append(stem)
        elif not checked:
            self.selected_stems = [s for s in self.selected_stems if s != stem]
        if set(self.selected_stems) == set(getattr(self, "available_stems", [])):
            self.selected_stems = []

    # ========================================================== workflow
    def _load_workflows(self) -> None:
        self.app.load_workflows()
        self._refresh_workflow_list()

    def _refresh_workflow_list(self) -> None:
        # keep the persistent empty-tip (index 0); only remove cards
        while self.wf_list_layout.count() > 1:
            item = self.wf_list_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        self._wf_cards.clear()

        q = self.plan_search.text().strip().lower()
        bg = QButtonGroup(self)
        for wf in self.app.workflows:
            if q and q not in (wf.get("name", "").lower() +
                               wf.get("description", "").lower()):
                continue
            card = _WorkflowListItem(wf)
            card.selected.connect(self._on_wf_selected)
            bg.addButton(card.radio)
            self.wf_list_layout.addWidget(card)
            self._wf_cards.append(card)

        self.wf_empty_tip.setVisible(len(self._wf_cards) == 0)
        self.wf_list_layout.addStretch(1)

    def _on_wf_selected(self, wf_id: str) -> None:
        self._selected_wf_id = wf_id
        self.app.selected_workflow = wf_id
        self._clear_advanced()
        self._update_status()

    # ============================================================= input
    def _add_paths(self, paths: list[str]) -> None:
        # Append only; never clear an existing list. Deduplicate by
        # case-normalized path so the same file is never added twice.
        existing = {os.path.normcase(x) for x in self.input_files}
        added = 0
        for p in paths:
            if Path(p).suffix.lower() in _AUDIO_EXTS and os.path.normcase(p) not in existing:
                self.input_files.append(p)
                existing.add(os.path.normcase(p))
                added += 1
        self._refresh_input_list()
        if added:
            self._update_status()

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择音频/视频", "",
            "Media (*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.opus "
            "*.mp4 *.mkv *.mov *.avi *.webm *.flv)")
        if files:
            self._add_paths(files)

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            self._add_paths(sorted(
                str(p) for p in Path(folder).rglob("*")
                if p.suffix.lower() in _AUDIO_EXTS))

    def _on_dropped(self, paths: list[str]) -> None:
        """Handle files/folders dropped onto the input drop zone.

        Files are added directly; folders are expanded to their media
        contents (matching the behaviour of the folder picker).
        """
        expanded: list[str] = []
        for p in paths:
            if os.path.isdir(p):
                expanded.extend(
                    str(x) for x in sorted(Path(p).rglob("*"))
                    if x.is_file() and x.suffix.lower() in _AUDIO_EXTS)
            elif os.path.isfile(p):
                expanded.append(p)
        if expanded:
            self._add_paths(expanded)

    def _clear_inputs(self) -> None:
        self.input_files = []
        self._refresh_input_list()
        self._update_status()

    def _remove_input_file(self, path: str) -> None:
        self.input_files = [p for p in self.input_files if p != path]
        self._refresh_input_list()
        self._update_status()

    def _refresh_input_list(self) -> None:
        lay = self.file_list_layout
        # keep only the empty tip (first child); remove all file items
        while lay.count() > 1:
            item = lay.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        n = len(self.input_files)
        self.empty_input_tip.setVisible(n == 0)
        self.input_summary.setText(f"概览选曲  已选 {n} 首")
        for path in self.input_files:
            item = _FileListItem(path)
            item.removed.connect(self._remove_input_file)
            lay.addWidget(item)
        # trailing stretch pushes items to the top (top-aligned) when the
        # list is shorter than the scroll viewport — mirrors the model /
        # workflow lists.
        lay.addStretch(1)

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "选择输出目录", self.temporary_output_dir or "")
        if d:
            self.temporary_output_dir = d
            self.out_dir_edit.setText(d)

    # ============================================================= modes
    def _set_mode(self, mode: str) -> None:
        self.run_mode = mode
        self.tab_model.setChecked(mode == "model")
        self.tab_workflow.setChecked(mode == "workflow")
        self.plan_stack.setCurrentIndex(0 if mode == "model" else 1)
        # category + sort dropdowns only apply to the model tab
        self.plan_filter.setVisible(mode == "model")
        self.plan_sort.setVisible(mode == "model")
        # refresh list for current tab
        if mode == "model":
            self._refresh_model_list()
            self.plan_search.setPlaceholderText(
                "搜索模型名称、用途、目标音源或架构")
            self.plan_link.setText("前往模型库 \u2197")
        else:
            self._refresh_workflow_list()
            self.plan_search.setPlaceholderText("搜索工作流名称\u2026")
            self.plan_link.setText("前往工作流 \u2197")
        self._update_status()
        # advanced params only apply to a selected model; clear them in
        # workflow mode or when no model is currently selected
        if mode == "workflow":
            self._clear_advanced()
        elif self._selected_model_name and self.app.model_infos.get(
                self._selected_model_name):
            self._read_advanced_from_yaml()
        else:
            self._clear_advanced()

    def _on_plan_link(self) -> None:
        # top-right link points to the matching management page for the
        # current plan tab (model library / workflow library)
        self.app.switch_to("workflows" if self.run_mode == "workflow" else "models")

    def _on_plan_search(self, text: str) -> None:
        if self.run_mode == "model":
            setattr(self, "model_search", text)
            self._refresh_model_list()
        else:
            setattr(self, "workflow_search", text)
            self._refresh_workflow_list()

    def _on_plan_filter(self, index: int) -> None:
        cat = self.plan_filter.currentData() or ""
        setattr(self, "model_category", cat)
        self._refresh_model_list()

    # ====================================================== advanced
    def _model_type(self) -> str:
        name = self.app.selected_model
        info = self.app.model_infos.get(name, {})
        return str(info.get("modelType") or info.get("architecture") or "").strip().lower()

    # ------------------------------------------------- advanced helpers
    def _vsep(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color:#e2e8f0;")
        sep.setFixedWidth(2)
        return sep

    def _make_int_spin(self, lo: int, hi: int, width: int) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setFixedWidth(width)
        return sb

    def _param_row(self, label: str, spin: QSpinBox) -> QHBoxLayout:
        h = QHBoxLayout()
        h.setSpacing(4)
        l = QLabel(label)
        l.setStyleSheet("font-size:11px;color:#64748b;")
        h.addWidget(l)
        h.addWidget(spin)
        return h

    def _set_advanced_enabled(self, enabled: bool) -> None:
        for sb in (self.sb_batch, self.sb_chunk,
                   self.sb_overlap, self.sb_num_overlap):
            sb.setEnabled(enabled)
        for b in (self.btn_adv_default, self.btn_adv_read,
                  self.btn_adv_save):
            b.setEnabled(enabled)

    def _sync_advanced_widgets(self) -> None:
        # 程序化赋值期间抑制 num_overlap 的"手动编辑"回调（避免清掉派生黄标）
        self._suppress_sync = True
        try:
            self.sb_batch.setValue(self.batch_size)
            self.sb_chunk.setValue(self.chunk_size)
            self.sb_overlap.setValue(self.overlap_size)
            self.sb_num_overlap.setValue(self.num_overlap)
        finally:
            self._suppress_sync = False

    def _set_num_overlap_derived(self, derived: bool) -> None:
        """num_overlap 由 overlap/chunk 反推（yaml 无此字段）时标黄，否则取消。"""
        if derived:
            self.sb_num_overlap.setStyleSheet(
                "background-color:#fff3cd;color:#7a5b00;")
        else:
            self.sb_num_overlap.setStyleSheet("")

    def _on_num_overlap_changed(self, v: int) -> None:
        # 仅用户手动编辑（非程序化 sync）时才视为"显式给定"
        if self._suppress_sync:
            return
        self._set_num_overlap_derived(False)
        # 手工改 num_overlap 时，按 chunk_size 动态反推 overlap_size
        #   重叠率 = 1 - 1/num_overlap → overlap_size = chunk_size - chunk_size // num_overlap
        # 仅在 chunk_size 已知（>0，已读取模型配置）时计算；未读取则保持原值。
        if v >= 1 and self.chunk_size > 0:
            self.overlap_size = self.chunk_size - self.chunk_size // v
            # 程序化回填 overlap 输入框（其 valueChanged 不会回算 num_overlap，无环）
            self._suppress_sync = True
            try:
                self.sb_overlap.setValue(self.overlap_size)
            finally:
                self._suppress_sync = False

    def _model_sample_rate(self) -> int:
        info = self.app.model_infos.get(self.app.selected_model, {})
        yaml_path = info.get("configPath")
        if yaml_path and os.path.isfile(yaml_path):
            try:
                from pymss_core.config import ConfigLoader
                with open(yaml_path, "r", encoding="utf-8") as f:
                    cfg = yaml.load(f, Loader=ConfigLoader) or {}
                return int((cfg.get("audio") or {}).get("sample_rate", 44100)
                           or 44100)
            except Exception:
                pass
        return 44100

    def _read_advanced_from_yaml(self) -> None:
        if self.run_mode != "model" or not self._selected_model_name:
            self._show_status_tip("请先选中一个模型")
            return
        info = self.app.model_infos.get(self.app.selected_model, {})
        yaml_path = info.get("configPath")
        if not yaml_path or not os.path.isfile(yaml_path):
            self._show_status_tip("未找到选中模型的配置文件")
            return
        try:
            from pymss_core.config import ConfigLoader
            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.load(f, Loader=ConfigLoader) or {}
            audio = cfg.get("audio", {}) or {}
            inf = cfg.get("inference", {}) or {}
            # 按模型真实采样率刷新块大小/重叠大小的调节步长 = 0.5 秒
            # （如 44100 → 22050，48000 → 24000），避免默认的 ±1 微调。
            sr = int(audio.get("sample_rate", 44100) or 44100)
            half = max(1, sr // 2)
            self.sb_chunk.setSingleStep(half)
            self.sb_overlap.setSingleStep(half)
            cs = int(audio.get("chunk_size", 0) or 0)
            ov = int(inf.get("overlap_size", 0) or 0)
            raw = inf.get("num_overlap", None)
            has_num = raw is not None and int(raw) > 0
            if has_num:
                # yaml 自带 num_overlap：直接读取，取消黄色
                self.num_overlap = int(raw)
                self._set_num_overlap_derived(False)
            else:
                # yaml 无 num_overlap：由 overlap_size / chunk_size 反推近似整数
                if cs > 0 and 0 <= ov < cs:
                    step = cs - ov
                    approx = round(cs / step) if step > 0 else 0
                    self.num_overlap = max(1, int(approx))
                else:
                    self.num_overlap = 0
                self._set_num_overlap_derived(True)  # 派生值 → 标黄
            self.chunk_size = cs
            self.batch_size = int(inf.get("batch_size", 1) or 1)
            self.overlap_size = ov
            self._sync_advanced_widgets()
            self._set_advanced_enabled(True)
            self._show_status_tip(
                "已读取模型配置参数"
                + ("" if has_num
                   else "（num_overlap 由 overlap/chunk 反推，已标黄）"))
        except Exception as e:  # noqa: BLE001
            self._show_status_tip(f"读取配置失败：{e}")

    def _write_advanced_to_yaml(self) -> None:
        if self.run_mode != "model" or not self._selected_model_name:
            self._show_status_tip("请先选中一个模型再保存")
            return
        info = self.app.model_infos.get(self.app.selected_model, {})
        yaml_path = info.get("configPath")
        if not yaml_path or not os.path.isfile(yaml_path):
            self._show_status_tip("未找到选中模型的配置文件")
            return
        try:
            from pymss_core.config import ConfigLoader, to_plain
            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.load(f, Loader=ConfigLoader) or {}
            cfg.setdefault("audio", {})
            cfg.setdefault("inference", {})
            cfg["audio"]["chunk_size"] = self.sb_chunk.value()
            cfg["inference"]["batch_size"] = self.sb_batch.value()
            cfg["inference"]["num_overlap"] = self.sb_num_overlap.value()
            cfg["inference"]["overlap_size"] = self.sb_overlap.value()
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(to_plain(cfg), f, allow_unicode=True,
                          sort_keys=False, default_flow_style=False)
            self._show_status_tip("已保存到模型配置文件")
        except Exception as e:  # noqa: BLE001
            self._show_status_tip(f"保存失败：{e}")

    def _reset_advanced_defaults(self) -> None:
        if self.run_mode != "model" or not self._selected_model_name:
            self._show_status_tip("请先选中一个模型")
            return
        sr = self._model_sample_rate()
        chunk = int(sr * 10)            # 块大小 = 10 秒
        self.sb_chunk.setValue(chunk)
        self.sb_num_overlap.setValue(4)            # 重叠率 75%
        self.sb_overlap.setValue(chunk - chunk // 4)
        self.sb_batch.setValue(1)
        self._show_status_tip("已设为默认：块大小=10秒，重叠率=75%")

    def _clear_advanced(self) -> None:
        for sb in (self.sb_batch, self.sb_chunk,
                   self.sb_overlap, self.sb_num_overlap):
            sb.setValue(0)
        self.batch_size = 0
        self.chunk_size = 0
        self.overlap_size = 0
        self.num_overlap = 0
        self._set_advanced_enabled(False)

    # =========================================================== run
    def _effective_format(self) -> str:
        if self.run_mode == "workflow" and self.app.get_selected_workflow():
            wf = self.app.get_selected_workflow()
            d = (wf.get("definition") or {}).get("defaults", {})
            return str(d.get("output_format") or
                       self.app.config["default_output_format"]).lower()
        return str(self.format_sel.currentText()).lower()

    def _on_format(self, fmt: str) -> None:
        self.app.config["default_output_format"] = fmt

    def _build_inference_params(self) -> dict:
        mt = self._model_type()
        vr = mt == "vr"
        apollo = mt == "apollo"
        p: dict = {}
        if vr:
            if self.window_size:
                p["window_size"] = self.window_size
            if self.aggression:
                p["aggression"] = self.aggression
            if self.enable_post_process:
                p["enable_post_process"] = True
            if self.post_process_threshold:
                p["post_process_threshold"] = self.post_process_threshold
            if self.high_end_process:
                p["high_end_process"] = True
            if self.normalize:
                p["normalize"] = True
            return p
        if self.batch_size:
            p["batch_size"] = self.batch_size
        if not apollo and self.num_overlap:
            p["num_overlap"] = self.num_overlap
        if self.chunk_size:
            p["chunk_size"] = self.chunk_size
        if self.overlap_size:
            p["overlap_size"] = self.overlap_size
        if self.standardize:
            p["standardize"] = True
        if self.normalize:
            p["normalize"] = True
        if self.slow_mode:
            # 慢速模式（省显存）：强制 use_complete_fast_path=False，
            # 让重叠缓冲留在 CPU，规避长文件 CUDA OOM（代价是更慢）。
            p["use_complete_fast_path"] = False
        # 引擎选择
        p["engine"] = self.engine
        return p

    def start(self) -> None:
        if not self.input_files:
            self._show_status_tip("请先选择输入文件")
            return
        if self.run_mode == "model" and not self._selected_model_name:
            self._show_status_tip("请选择模型")
            return
        if self.run_mode == "workflow" and not self._selected_wf_id:
            self._show_status_tip("请选择工作流")
            return

        output_dir = self.temporary_output_dir or self.app.config.output_dir()
        output_layout = "folders" if self.save_as_folder else "flat"
        job_id = self.app.tasks.new_id("job")
        device_cfg = self.app.config.get_runtime_device_config(self.app.env)

        tasks = []
        for inp in self.input_files:
            rc = {
                "runMode": self.run_mode,
                "device": device_cfg["device"],
                "deviceIds": device_cfg["deviceIds"],
                "outputFormat": self._effective_format(),
                "outputLayout": output_layout,
                "selectedStems": list(self.selected_stems),
                "useTta": self.use_tta,
                "debug": self.debug,
                "audioParams": self.app.config.get_audio_params(),
                "inferenceParamsVersion": 3,
                "inferenceParams": self._build_inference_params(),
            }
            if self.run_mode == "model":
                rc["model"] = self._selected_model_name
            else:
                wf = self.app.get_selected_workflow()
                rc["workflowName"] = wf.get("name")
                rc["workflowId"] = wf.get("id")
                rc["workflowDefinition"] = wf.get("definition")
            t = self.app.tasks.create_queued(
                inp, rc.get("model", "") or rc.get("workflowName", ""),
                rc, job_id=job_id, job_output=output_dir,
                output_layout=output_layout,
                prefix="wf" if self.run_mode == "workflow" else "sep")
            tasks.append(t)

        payload = {
            "taskId": job_id,
            "model": rc.get("model"),
            "workflowName": rc.get("workflowName"),
            "workflow": rc.get("workflowDefinition"),
            "output": output_dir,
            "modelDir": self.app.config.models_dir() or None,
            "download": True,
            "source": self.app.config["download_source"],
            "endpoint": None,
            "device": device_cfg["device"],
            "deviceIds": device_cfg["deviceIds"],
            "outputFormat": rc["outputFormat"],
            "outputLayout": output_layout,
            "saveAsFolder": True,  # always use pymss folder mode to avoid per-stem subdirs
            "selectedStems": rc["selectedStems"],
            "useTta": rc["useTta"],
            "debug": rc["debug"],
            "audioParams": rc["audioParams"],
            "inferenceParamsVersion": 3,
            "inferenceParams": rc["inferenceParams"],
            "tasks": [{"taskId": t.task_id, "input": t.input, "output": t.output} for t in tasks],
        }
        self.selected_task_id = job_id
        if self.run_mode == "workflow":
            self.app.bridge.infer_workflow(payload, job_id)
        else:
            self.app.bridge.infer(payload, job_id)
        # show progress panel
        self.progress_panel.show()
        self.summary_widget.hide()
        self.progress_title.setText(f"正在分离 0 / {len(tasks)}")
        self.progress_state_badge.setText("运行中")
        self.progress_state_badge.setStyleSheet(
            "font-size:11px;color:#2563eb;background:#eff6ff;"
            "padding:2px 10px;border-radius:4px;")
        self.progress_bar.setValue(0)
        self.progress_pct.setText("0%")
        self.output_path_lbl.setText(f"输出到 {output_dir}")

        # NOTE: input files are intentionally kept after a run; they are NOT
        # cleared so the user can re-run or inspect them.

    def _show_status_tip(self, msg: str) -> None:
        """Show a transient (non-blocking) tip in the bottom status label."""
        self.status_tip_label.setText(msg)
        self._status_timer.start(3500)

    # ===================================================== run-state UI
    def on_task_updated(self, task: Task) -> None:
        if self.selected_task_id and task.job_id == self.selected_task_id:
            self._refresh_running()
        self._update_status()

    def _refresh_running(self) -> None:
        job = self.app.tasks.get_job(self.selected_task_id)
        if not job:
            return
        total = len(job.tasks)
        done = sum(1 for t in job.tasks if t.is_terminal)
        failed = sum(1 for t in job.tasks if t.status == "failed")
        succeeded = sum(1 for t in job.tasks if t.status == "done")
        self.progress_title.setText(f"正在分离 {done} / {total}")

        # current file
        current = next((t for t in job.tasks if not t.is_terminal), None)
        if current:
            self.current_file_lbl.setText(f"当前：{Path(current.input).name}")
            self.progress_msg.setText(current.message or "")
        else:
            self.current_file_lbl.setText("")

        # overall progress
        pct = job.progress
        self.progress_bar.setValue(pct)
        self.progress_pct.setText(f"{pct}%")

        # state badge
        if job.status in ("done", "failed", "cancelled"):
            self.progress_state_badge.setText(
                {"done": "已完成", "failed": "失败", "cancelled": "已取消"}.get(job.status, ""))
            color = {"done": "#15803d", "failed": "#dc2626",
                     "cancelled": "#94a3b8"}.get(job.status, "#64748b")
            bg = {"done": "#f0fdf4", "fail": "#fef2f2",
                  "cancelled": "#f8fafc"}.get(job.status, "#f8fafc")
            self.progress_state_badge.setStyleSheet(
                f"font-size:11px;color:{color};background:{bg};"
                f"padding:2px 10px;border-radius:4px;")
            self.btn_cancel_job.setEnabled(False)
            # show summary
            self._show_completion_summary(total, succeeded, failed)
        else:
            self.progress_state_badge.setText("运行中")

    def _show_completion_summary(self, total: int, ok: int, fail: int) -> None:
        self.summary_widget.show()
        self.summary_lbl.setText(
            f"处理完成：共 {total} 个文件，成功 {ok} 个，失败 {fail} 个")
        # populate output file list
        self.output_file_list.clear()
        job = self.app.tasks.get_job(self.selected_task_id)
        if job:
            for t in job.tasks:
                if t.status != "done":
                    continue
                for o in (t.outputs or []):
                    name = Path(o.get("path", "")).name if o.get("path") else o.get("stem", "")
                    item = QListWidgetItem(name)
                    item.setData(Qt.UserRole, o.get("path", ""))
                    self.output_file_list.addItem(item)

    def _update_status(self) -> None:
        if self.run_mode == "workflow":
            ready = bool(self._selected_wf_id) and bool(self.input_files)
        else:
            ready = bool(self._selected_model_name) and bool(self.input_files)
        self.start_btn.setEnabled(ready)

    def _cancel_current(self) -> None:
        if self.selected_task_id:
            self.app.bridge.cancel(self.selected_task_id)
            job = self.app.tasks.get_job(self.selected_task_id)
            if job:
                for t in job.tasks:
                    self.app.tasks.set_status(t, "cancelled", "Cancelled", 100)

    def _reveal_output(self) -> None:
        d = self.temporary_output_dir or self.app.config.output_dir()
        if d and Path(d).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def _show_logs(self) -> None:
        job = self.app.tasks.get_job(self.selected_task_id)
        logs = []
        if job:
            for t in job.tasks:
                logs.extend(t.logs or [])
        dlg = QDialog(self)
        dlg.setWindowTitle("日志")
        dlg.setMinimumSize(600, 400)
        lay = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText("\n".join(logs))
        lay.addWidget(te)
        dlg.exec()
