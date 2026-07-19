"""Models view — fully mirrors the original Vue ModelsView.

Header (count / 空间管理 / 加载模型列表), toolbar (search / category / sort /
仅显示已下载), paginated card grid, model detail modal (note + default inference
params), and a storage drawer (summary / batch delete / residual cleanup).
Download & delete progress is tracked per model and reflected on the cards.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox, QCheckBox,
    QPushButton, QLabel, QScrollArea, QGridLayout, QFrame, QDialog,
    QTextEdit, QSpinBox, QMessageBox, QListWidget, QListWidgetItem,
    QFileDialog, QProgressBar, QSizePolicy,
)


def format_bytes(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    if n <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.1f} {units[i]}"


class ModelCard(QFrame):
    open_detail = Signal(dict)
    download = Signal(str)
    cancel = Signal(str)
    delete = Signal(str)
    toggle_fav = Signal(str)

    def __init__(self, model: dict, state: dict | None, favorited: bool, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setFixedHeight(150)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(4)

        top = QHBoxLayout()
        name = QLabel(f"<b>{model.get('name','')}</b>")
        name.setWordWrap(True)
        self.fav_btn = QPushButton("★" if favorited else "☆")
        self.fav_btn.setFixedWidth(28)
        self.fav_btn.clicked.connect(lambda: self.toggle_fav.emit(model.get("name", "")))
        top.addWidget(name, 1)
        top.addWidget(self.fav_btn)
        v.addLayout(top)

        meta = QLabel()
        arch = model.get("architecture", "")
        mtype = model.get("modelType", "")
        tags = [t for t in (arch, mtype) if t and t != arch]
        meta.setText(" · ".join(filter(None, [arch, *tags])))
        meta.setStyleSheet("color:#8aa")
        v.addWidget(meta)

        info = QLabel()
        bits = []
        if model.get("targetStem"):
            bits.append(f"目标: {model.get('targetStem')}")
        if model.get("categoryCn"):
            bits.append(f"分类: {model.get('categoryCn')}")
        info.setText("  |  ".join(bits))
        info.setWordWrap(True)
        v.addWidget(info)

        size = QLabel(f"大小: {format_bytes(model.get('sizeBytes'))}")
        v.addWidget(size)

        # status / actions
        self.status_lbl = QLabel()
        v.addWidget(self.status_lbl)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        v.addWidget(self.progress)

        acts = QHBoxLayout()
        self.btn_a = QPushButton()
        self.btn_b = QPushButton()
        self.btn_detail = QPushButton("详情")
        self.btn_detail.clicked.connect(lambda: self.open_detail.emit(self.model))
        acts.addWidget(self.btn_a)
        acts.addWidget(self.btn_b)
        acts.addWidget(self.btn_detail)
        v.addLayout(acts)

        self.apply_state(state)

    def apply_state(self, state: dict | None) -> None:
        m = self.model
        downloaded = bool(m.get("downloaded"))

        def _reset(btn: QPushButton) -> None:
            try:
                btn.clicked.disconnect()
            except RuntimeError:
                pass

        if state and not state.get("done"):
            self.status_lbl.setText(state.get("message") or "处理中…")
            self.progress.setVisible(True)
            self.progress.setValue(int(state.get("progress") or 0))
            _reset(self.btn_a)
            self.btn_a.setText("取消")
            self.btn_a.clicked.connect(lambda: self.cancel.emit(m.get("name", "")))
            self.btn_b.setVisible(False)
            return
        self.progress.setVisible(False)
        _reset(self.btn_a)
        _reset(self.btn_b)
        if downloaded:
            self.status_lbl.setText("已下载")
            self.btn_a.setText("删除")
            self.btn_a.clicked.connect(lambda: self.delete.emit(m.get("name", "")))
            self.btn_b.setText("打开目录")
            self.btn_b.setVisible(True)
            self.btn_b.clicked.connect(lambda: self._open_dir())
            return

        # Not downloaded. A model is only "partially downloaded" (some required
        # files present, some missing) when a real local-state snapshot reports
        # missing files AND at least one required file already exists on disk.
        # If nothing has been downloaded yet, it must show as 未下载 / 下载
        # (not 部分下载 / 继续下载).
        missing_paths = m.get("missingPaths") or []
        if missing_paths:
            required = [p for p in (
                [m.get("modelPath")]
                + ([m.get("configPath")] if m.get("configPath") else [])
                + (m.get("auxiliaryPaths") or [])
            ) if p]
            present = len(required) - len(missing_paths)
            if present > 0:
                self.status_lbl.setText(f"部分下载 ({len(missing_paths)} 文件缺失)")
                self.btn_a.setText("继续下载")
                self.btn_a.clicked.connect(lambda: self.download.emit(m.get("name", "")))
                self.btn_b.setVisible(False)
                return
        # Fall through: not downloaded (nothing present on disk) → 未下载
        self.status_lbl.setText("未下载")
        self.btn_a.setText("下载")
        self.btn_a.clicked.connect(lambda: self.download.emit(m.get("name", "")))
        self.btn_b.setVisible(False)

    def _open_dir(self) -> None:
        p = self.model.get("modelPath") or self.model.get("configPath")
        if p:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(p).parent)))


class ModelsView(QWidget):
    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self.search = ""
        self.category = ""
        self.sort = "default"
        self.downloaded_only = False
        self.page = 0
        self.page_size = 24
        self.favorites: set[str] = set()
        self.notes: dict[str, str] = {}
        self.overrides: dict[str, dict] = {}
        self._download_state: dict[str, dict] = {}
        self._delete_state: dict[str, dict] = {}
        self._cards: dict[str, ModelCard] = {}
        self._load_prefs()
        self._build_ui()
        self.on_models(self.app.models)

    # ---- persistence --------------------------------------------------
    def _prefs_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "model_prefs.json"

    def _load_prefs(self) -> None:
        p = self._prefs_path()
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self.favorites = set(data.get("favorites", []))
                self.notes = data.get("notes", {})
                self.overrides = data.get("overrides", {})
            except Exception:
                pass

    def _save_prefs(self) -> None:
        p = self._prefs_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "favorites": sorted(self.favorites),
            "notes": self.notes,
            "overrides": self.overrides,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- UI -----------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        head = QHBoxLayout()
        head.addWidget(QLabel("<b style='font-size:18px'>模型库</b>"))
        head.addStretch(1)
        self.storage_btn = QPushButton("空间管理")
        self.storage_btn.clicked.connect(self._open_storage)
        self.reload_btn = QPushButton("加载模型列表")
        self.reload_btn.clicked.connect(lambda: self.app.refresh_models())
        head.addWidget(self.storage_btn)
        head.addWidget(self.reload_btn)
        self.count_lbl = QLabel("")
        head.addWidget(self.count_lbl)
        root.addLayout(head)

        tb = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索模型名称、架构、类型或备注…")
        self.search_edit.textChanged.connect(self._on_search)
        self.cat_combo = QComboBox()
        self.cat_combo.addItem("全部分类", "")
        self.cat_combo.currentIndexChanged.connect(self._on_cat)
        self.sort_combo = QComboBox()
        for label, val in [
            ("默认排序", "default"), ("收藏优先", "favorite"), ("名称 A→Z", "name-asc"),
            ("名称 Z→A", "name-desc"), ("大小 大→小", "size-desc"), ("大小 小→大", "size-asc"),
            ("分类", "category"), ("类型", "type"), ("已下载优先", "downloaded"),
        ]:
            self.sort_combo.addItem(label, val)
        self.sort_combo.currentIndexChanged.connect(self._on_sort)
        self.dl_only = QCheckBox("仅显示已下载")
        self.dl_only.toggled.connect(self._on_dl_only)
        tb.addWidget(self.search_edit, 1)
        tb.addWidget(self.cat_combo)
        tb.addWidget(self.sort_combo)
        tb.addWidget(self.dl_only)
        root.addLayout(tb)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        grid_vbox = QVBoxLayout(self.grid_container)
        grid_vbox.setContentsMargins(0, 0, 0, 0)
        grid_vbox.setSpacing(0)
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        grid_vbox.addLayout(self.grid)
        # Absorb any extra vertical space so cards stay top-aligned and the
        # area below them is simply left empty (never stretched/centered).
        grid_vbox.addStretch(1)
        self.scroll.setWidget(self.grid_container)
        root.addWidget(self.scroll, 1)

        pag = QHBoxLayout()
        self.prev_btn = QPushButton("上一页")
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn = QPushButton("下一页")
        self.next_btn.clicked.connect(self._next)
        self.page_lbl = QLabel("")
        self.size_combo = QComboBox()
        for s in (12, 24, 48, 96):
            self.size_combo.addItem(f"{s}/页", s)
        self.size_combo.setCurrentIndex(1)
        self.size_combo.currentIndexChanged.connect(self._on_page_size)
        pag.addStretch(1)
        pag.addWidget(self.prev_btn)
        pag.addWidget(self.page_lbl)
        pag.addWidget(self.next_btn)
        pag.addWidget(QLabel("每页"))
        pag.addWidget(self.size_combo)
        root.addLayout(pag)

    # ---- data ---------------------------------------------------------
    def on_models_error(self, message: str) -> None:
        QMessageBox.warning(
            self, "模型列表加载失败",
            "无法加载模型列表（已确认包已安装但仍失败）：\n\n%s\n\n"
            "请检查：\n"
            "  1. 设置页中的 Python 解释器是否指向已安装 pymss 的 venv；\n"
            "  2. worker 目录 python/worker.py 是否存在；\n"
            "  3. 在「模型库」点击「加载模型列表」重试。" % message)

    def on_models(self, models: list[dict]) -> None:
        # populate category combo once
        cats = []
        seen = set()
        for m in models:
            c = m.get("category")
            cn = m.get("categoryCn") or c
            if c and c not in seen:
                seen.add(c)
                cats.append((cn or c, c))
        self.cat_combo.blockSignals(True)
        self.cat_combo.clear()
        self.cat_combo.addItem("全部分类", "")
        for cn, c in cats:
            self.cat_combo.addItem(cn, c)
        idx = self.cat_combo.findData(self.category)
        self.cat_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.cat_combo.blockSignals(False)
        self.page = 0
        self._rebuild()

    def _filtered(self) -> list[dict]:
        models = [m for m in self.app.models if m.get("supported")]
        q = self.search.strip().lower()
        if q:
            models = [m for m in models if q in " ".join(str(m.get(k, "")) for k in
                       ("name", "architecture", "modelType", "targetStem",
                        "category", "categoryCn", "configTargetInstrument")).lower()
                       or q in (self.notes.get(m.get("name", ""), "")).lower()]
        if self.category:
            models = [m for m in models if m.get("category") == self.category
                      or m.get("primaryCategory") == self.category
                      or m.get("secondaryCategory") == self.category]
        if self.downloaded_only:
            models = [m for m in models if m.get("downloaded")]
        key = self.sort_combo.currentData()
        if key == "favorite":
            models.sort(key=lambda m: (m.get("name", "") not in self.favorites, m.get("name", "")))
        elif key == "name-asc":
            models.sort(key=lambda m: m.get("name", "").lower())
        elif key == "name-desc":
            models.sort(key=lambda m: m.get("name", "").lower(), reverse=True)
        elif key == "size-desc":
            models.sort(key=lambda m: m.get("sizeBytes") or 0, reverse=True)
        elif key == "size-asc":
            models.sort(key=lambda m: m.get("sizeBytes") or 0)
        elif key == "category":
            models.sort(key=lambda m: m.get("category", ""))
        elif key == "type":
            models.sort(key=lambda m: m.get("modelType") or m.get("architecture", ""))
        elif key == "downloaded":
            models.sort(key=lambda m: (not m.get("downloaded"), m.get("name", "")))
        return models

    def _rebuild(self) -> None:
        self._clear_grid()
        self._cards.clear()
        models = self._filtered()
        self.count_lbl.setText(f"共 {len(models)} 个模型（已支持）")
        pages = max(1, (len(models) + self.page_size - 1) // self.page_size)
        if self.page >= pages:
            self.page = pages - 1
        start = self.page * self.page_size
        page_items = models[start:start + self.page_size]
        cols = 3
        for i, m in enumerate(page_items):
            name = m.get("name", "")
            state = self._download_state.get(name) or self._delete_state.get(name)
            card = ModelCard(m, state, name in self.favorites)
            card.open_detail.connect(self._open_detail)
            card.download.connect(self._download)
            card.cancel.connect(self._cancel_download)
            card.delete.connect(self._delete)
            card.toggle_fav.connect(self._toggle_fav)
            self.grid.addWidget(card, i // cols, i % cols)
            self._cards[name] = card
        self.page_lbl.setText(f"第 {self.page + 1} / {pages} 页")
        self.prev_btn.setEnabled(self.page > 0)
        self.next_btn.setEnabled(self.page < pages - 1)

    def _clear_grid(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    # ---- toolbar handlers --------------------------------------------
    def _on_search(self, t: str) -> None:
        self.search = t
        self.page = 0
        self._rebuild()

    def _on_cat(self) -> None:
        self.category = self.cat_combo.currentData() or ""
        self.page = 0
        self._rebuild()

    def _on_sort(self) -> None:
        self.page = 0
        self._rebuild()

    def _on_dl_only(self, on: bool) -> None:
        self.downloaded_only = on
        self.page = 0
        self._rebuild()

    def _on_page_size(self) -> None:
        self.page_size = self.size_combo.currentData()
        self.page = 0
        self._rebuild()

    def _prev(self) -> None:
        if self.page > 0:
            self.page -= 1
            self._rebuild()

    def _next(self) -> None:
        self.page += 1
        self._rebuild()

    # ---- favorite -----------------------------------------------------
    def _toggle_fav(self, name: str) -> None:
        if name in self.favorites:
            self.favorites.discard(name)
        else:
            self.favorites.add(name)
        self._save_prefs()
        self._rebuild()

    # ---- download / delete --------------------------------------------
    def _download(self, name: str) -> None:
        self._download_state[name] = {"progress": 0, "done": False, "message": "排队中…"}
        self.app.bridge.download_model(name, self.app.config.models_dir(),
                                       source=self.app.config["download_source"], force=False)
        self._refresh_card(name)

    def _cancel_download(self, name: str) -> None:
        st = self._download_state.get(name)
        if st and st.get("task_id"):
            self.app.bridge.cancel(st["task_id"])
        self._download_state.pop(name, None)
        self._refresh_card(name)

    def _delete(self, name: str) -> None:
        reply = QMessageBox.question(
            self, "删除模型", f"确认删除本地模型 “{name}”？此操作不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._delete_state[name] = {"progress": 0, "done": False, "message": "删除中…"}
        self.app.bridge.delete_model(name, self.app.config.models_dir())
        self._refresh_card(name)

    def on_download_event(self, etype: str, pl: dict, tid: str = "") -> None:
        name = pl.get("model") or ""
        if not name:
            return
        st = self._download_state.setdefault(name, {})
        st["task_id"] = tid
        if etype == "download_started":
            st.update(progress=0, done=False, message="下载任务已开始")
        elif etype == "download_stage":
            st.update(message=pl.get("message") or pl.get("stage") or st.get("message"), progress=pl.get("progress", st.get("progress")))
        elif etype in ("download_progress", "download_file"):
            st.update(progress=pl.get("progress", st.get("progress")), done=False,
                      message=f"{pl.get('completedFiles', 0)}/{pl.get('totalFiles', 0)} 文件 · {format_bytes(pl.get('aggregateDownloadedBytes', 0))}")
        elif etype == "download_done":
            st.update(progress=100, done=True, message="下载完成")
        elif etype == "download_failed":
            st.update(done=True, message=f"失败: {pl.get('message','')}")
        self._refresh_card(name)

    def on_delete_event(self, etype: str, pl: dict, tid: str = "") -> None:
        name = pl.get("model") or ""
        if not name:
            return
        st = self._delete_state.setdefault(name, {})
        if etype == "model_delete_started":
            st.update(progress=0, done=False, message="删除中…")
        elif etype == "model_delete_progress":
            st.update(progress=pl.get("progress", 0), done=False, message="删除中…")
        elif etype == "model_delete_done":
            st.update(progress=100, done=True, message="已删除")
        elif etype == "model_delete_failed":
            st.update(done=True, message=f"失败: {pl.get('message','')}")
        self._refresh_card(name)

    def _refresh_card(self, name: str) -> None:
        card = self._cards.get(name)
        if card:
            state = self._download_state.get(name) or self._delete_state.get(name)
            card.apply_state(state)

    # ---- detail modal -------------------------------------------------
    def _open_detail(self, model: dict) -> None:
        self.app.bridge.model_info(model.get("name", ""), self.app.config.models_dir(),
                                   channel="models_detail")

    def on_model_info(self, info: dict) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("模型详情")
        dlg.setMinimumWidth(460)
        v = QVBoxLayout(dlg)
        name = info.get("name", "")
        v.addWidget(QLabel(f"<b>{name}</b>"))
        rows = [
            ("架构", info.get("architecture")),
            ("类型", info.get("modelType")),
            ("目标音源", info.get("targetStem")),
            ("大小", format_bytes(info.get("sizeBytes"))),
            ("分类", info.get("categoryCn") or info.get("category")),
            ("别名", ", ".join(info.get("aliases") or []) or "—"),
            ("状态", "已下载" if info.get("downloaded") else "未下载"),
            ("路径", info.get("modelPath")),
        ]
        for k, val in rows:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"<b>{k}</b>"))
            row.addWidget(QLabel(str(val) if val is not None else "—"))
            row.addStretch(1)
            v.addLayout(row)

        v.addWidget(QLabel("<b>备注</b>"))
        note = QTextEdit()
        note.setPlainText(self.notes.get(name, ""))
        note.setMaximumHeight(70)
        v.addWidget(note)

        v.addWidget(QLabel("<b>默认推理参数</b>"))
        inf_def = dict(self.overrides.get(name, {}))
        base = info.get("defaultInferenceParams") or {}
        for key, label in [("batch_size", "批量大小"), ("overlap_size", "重叠大小"),
                           ("chunk_size", "分块大小")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sb = QSpinBox()
            sb.setRange((0 if key == "batch_size" else 0), 1048576)
            sb.setValue(int(inf_def.get(key, base.get(key, 0) or 0)))
            sb.valueChanged.connect(lambda val, k=key: inf_def.__setitem__(k, val))
            row.addWidget(sb)
            v.addLayout(row)

        def save_note():
            self.notes[name] = note.toPlainText().strip()
            self.overrides[name] = inf_def
            self._save_prefs()
            dlg.accept()

        def reset_def():
            self.overrides.pop(name, None)
            self._save_prefs()
            dlg.accept()

        bar = QHBoxLayout()
        b_save = QPushButton("保存")
        b_save.clicked.connect(save_note)
        b_reset = QPushButton("恢复模型默认值")
        b_reset.clicked.connect(reset_def)
        bar.addWidget(b_save)
        bar.addWidget(b_reset)
        bar.addStretch(1)
        v.addLayout(bar)

        if info.get("downloaded"):
            del_btn = QPushButton("删除模型")
            del_btn.clicked.connect(lambda: (self._delete(name), dlg.accept()))
            v.addWidget(del_btn)
        else:
            dl_btn = QPushButton("下载模型")
            dl_btn.clicked.connect(lambda: (self._download(name), dlg.accept()))
            v.addWidget(dl_btn)
        dlg.exec()

    # ---- storage drawer ----------------------------------------------
    def _open_storage(self) -> None:
        self.app.bridge.model_storage_summary(self.app.config.models_dir())
        self._storage = StorageDrawer(self.app, self)
        self._storage.show()

    def on_storage(self, summary: dict) -> None:
        if getattr(self, "_storage", None) and self._storage.isVisible():
            self._storage.apply(summary)
        else:
            self._storage = StorageDrawer(self.app, self)
            self._storage.apply(summary)
            self._storage.show()

    def on_residual_event(self, etype: str, pl: dict, tid: str = "") -> None:
        drawer = getattr(self, "_storage", None)
        if drawer and drawer.isVisible():
            drawer.on_residual_event(etype, pl)


class StorageDrawer(QDialog):
    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app
        self.setWindowTitle("空间管理")
        self.setMinimumWidth(620)
        self.setMinimumHeight(520)
        self.summary: dict = {}
        self.selected: set[str] = set()
        self._cards: dict[str, QListWidgetItem] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        self.stats = QLabel("统计加载中…")
        v.addWidget(self.stats)

        bar = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索模型…")
        self.search_edit.textChanged.connect(self._apply_filter)
        self.sort_combo = QComboBox()
        for label, val in [("大小 大→小", "size-desc"), ("大小 小→大", "size-asc"),
                           ("名称 A→Z", "name-asc"), ("名称 Z→A", "name-desc")]:
            self.sort_combo.addItem(label, val)
        self.sort_combo.currentIndexChanged.connect(self._apply_filter)
        self.dl_only = QCheckBox("仅显示已下载")
        self.dl_only.toggled.connect(self._apply_filter)
        self.refresh_btn = QPushButton("刷新空间统计")
        self.refresh_btn.clicked.connect(lambda: self.app.bridge.model_storage_summary(self.app.config.models_dir()))
        self.open_dir_btn = QPushButton("打开模型目录")
        self.open_dir_btn.clicked.connect(self._open_dir)
        bar.addWidget(self.search_edit, 1)
        bar.addWidget(self.sort_combo)
        bar.addWidget(self.dl_only)
        bar.addWidget(self.refresh_btn)
        bar.addWidget(self.open_dir_btn)
        v.addLayout(bar)

        self.list = QListWidget()
        self.list.itemChanged.connect(self._on_check)
        v.addWidget(self.list, 1)

        pbar = QHBoxLayout()
        self.batch_del = QPushButton("批量删除所选")
        self.batch_del.clicked.connect(self._batch_delete)
        self.cleanup = QPushButton("清理残留文件")
        self.cleanup.clicked.connect(self._cleanup)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        pbar.addWidget(self.batch_del)
        pbar.addWidget(self.cleanup)
        pbar.addWidget(self.progress, 1)
        v.addLayout(pbar)

    def apply(self, summary: dict) -> None:
        self.summary = summary or {}
        total = format_bytes(self.summary.get("totalBytes"))
        dl = self.summary.get("downloadedCount", 0)
        residual = format_bytes(self.summary.get("residualBytes"))
        self.stats.setText(f"已下载占用: {total}  ·  已下载模型: {dl}  ·  可清理残留: {residual}")
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        self._cards.clear()
        models = self.summary.get("models", [])
        q = self.search_edit.text().strip().lower()
        key = self.sort_combo.currentData()
        if key == "size-desc":
            models = sorted(models, key=lambda m: m.get("sizeBytes") or 0, reverse=True)
        elif key == "size-asc":
            models = sorted(models, key=lambda m: m.get("sizeBytes") or 0)
        elif key == "name-asc":
            models = sorted(models, key=lambda m: m.get("name", "").lower())
        elif key == "name-desc":
            models = sorted(models, key=lambda m: m.get("name", "").lower(), reverse=True)
        for m in models:
            if q and q not in m.get("name", "").lower():
                continue
            if self.dl_only.isChecked() and not m.get("downloaded"):
                continue
            item = QListWidgetItem(f"{m.get('name','')}  ·  {format_bytes(m.get('sizeBytes'))}  ·  {'已下载' if m.get('downloaded') else '未下载'}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if m.get("name") in self.selected else Qt.CheckState.Unchecked)
            self.list.addItem(item)
            self._cards[m.get("name", "")] = item
        self.list.blockSignals(False)

    def _apply_filter(self) -> None:
        self._rebuild_list()

    def _on_check(self, item: QListWidgetItem) -> None:
        name = item.text().split("  ·")[0].strip()
        if item.checkState() == Qt.CheckState.Checked:
            self.selected.add(name)
        else:
            self.selected.discard(name)

    def _open_dir(self) -> None:
        d = self.summary.get("modelDir")
        if d:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def _batch_delete(self) -> None:
        names = list(self.selected)
        if not names:
            return
        reply = QMessageBox.question(
            self, "批量删除", f"确认删除选中的 {len(names)} 个模型？不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.progress.setVisible(True)
        self.progress.setValue(0)
        total = len(names)
        for i, name in enumerate(names):
            self.app.bridge.delete_model(name, self.app.config.models_dir())
            self.progress.setValue(int((i + 1) / total * 100))

    def _cleanup(self) -> None:
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.app.bridge.cleanup_residual(self.app.config.models_dir())

    def on_residual_event(self, etype: str, pl: dict) -> None:
        if etype.endswith("_progress") or etype.endswith("_done") or etype == "model_residual_cleaned":
            self.progress.setValue(int(pl.get("progress", 0) or 0))
            if "modelStorageSummary" in pl:
                self.apply(pl["modelStorageSummary"])
        if etype.endswith("_done") or etype == "model_residual_cleaned":
            self.progress.setVisible(False)
            self.app.refresh_models()
