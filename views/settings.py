"""Settings view — fully mirrors the original Vue SettingsView sections.

Sidebar: 关于 / 路径 / 默认参数. The 默认参数 page includes device, download
source, concurrency, developer mode and the audio-format block (wav/flac/mp3/m4a
bit-depth / bit-rate / codec) that the original app keeps in the settings store
and surfaces in the separation audio-quality panel.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QStackedWidget, QFormLayout,
    QLineEdit, QComboBox, QPushButton, QFileDialog, QTextEdit, QLabel,
    QCheckBox, QSpinBox, QGroupBox, QGridLayout, QFrame,
    QButtonGroup, QRadioButton,
)

from ..config import AppConfig

SECTIONS = [
    ("关于", "about"),
    ("路径", "paths"),
    ("默认参数", "defaults"),
    ("处理引擎", "engine"),
]


class SettingsView(QWidget):
    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self._build_ui()
        self._load()
        # copy env-derived versions if already available
        self.on_env(self.app.env)
        # keep the environment log (moved here from the old home view) in sync
        self.app.env_changed.connect(self._show_env)
        self._show_env(self.app.env)

    # ---- UI -----------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(130)
        for label, _ in SECTIONS:
            self.sidebar.addItem(label)
        self.sidebar.currentRowChanged.connect(self._on_nav)
        root.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_about())
        self.stack.addWidget(self._build_paths())
        self.stack.addWidget(self._build_defaults())
        self.stack.addWidget(self._build_engine())
        root.addWidget(self.stack, 1)

        self.sidebar.setCurrentRow(0)

    # ---- about --------------------------------------------------------
    def _build_about(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        title = QLabel("<b style='font-size:20px'>Pymss Studio</b>")
        v.addWidget(title)

        self.sw_ver = QLabel("软件版本: 0.0.8")
        self.core_ver = QLabel("pymss 核心版本: —")
        self.worker_ver = QLabel("Worker 版本: —")
        for l in (self.sw_ver, self.core_ver, self.worker_ver):
            v.addWidget(l)

        intro = QLabel(
            "本程序用 PySide6 重写桌面界面，直接调用原有的 Python worker"
            "（python/worker.py），复用 pymss 核心，无需翻译任何分离算法。\n\n"
            "左侧切换：分离 / 模型库 / 任务 / 结果 / 编辑器 / 工作流 / 设置。"
        )
        intro.setWordWrap(True)
        v.addWidget(intro)

        bar = QHBoxLayout()
        self.check_btn = QPushButton("检查运行环境")
        self.check_btn.clicked.connect(lambda: self.app.refresh_env())
        bar.addWidget(self.check_btn)
        bar.addStretch(1)
        v.addLayout(bar)

        self.env_log = QTextEdit()
        self.env_log.setReadOnly(True)
        v.addWidget(self.env_log, 1)

        lic = QGroupBox("许可证")
        lf = QVBoxLayout(lic)
        lf.addWidget(QLabel("桌面端: AGPL-3.0"))
        lf.addWidget(QLabel("核心库: MIT"))
        v.addWidget(lic)

        links = QGroupBox("相关链接")
        lk = QVBoxLayout(links)
        for label, url in [
            ("桌面端仓库", "https://github.com/TheSmallHanCat/Pymss-Studio"),
            ("核心库仓库", "https://github.com/TheSmallHanCat/pymss"),
            ("AGPL-3.0 许可证", "https://www.gnu.org/licenses/agpl-3.0.html"),
            ("MIT 许可证", "https://opensource.org/licenses/MIT"),
        ]:
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u)))
            lk.addWidget(b)
        v.addWidget(links)
        v.addStretch(1)
        return w

    # ---- paths --------------------------------------------------------
    def _build_paths(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()

        self.data_edit = QLineEdit()
        self.data_edit.setReadOnly(True)
        form.addRow("数据根目录", self._with_browse(self.data_edit, self._open_data))

        self.model_edit = QLineEdit()
        form.addRow("模型目录", self._with_browse(self.model_edit, self._change_model_dir))

        self.out_edit = QLineEdit()
        form.addRow("输出目录", self._with_browse(self.out_edit, self._change_output_dir))

        self.worker_edit = QLineEdit()
        form.addRow("worker.py 目录", self._with_browse(self.worker_edit, self._browse_worker))

        self.pymss_edit = QLineEdit()
        form.addRow("pymss 路径", self._with_browse(self.pymss_edit, self._browse_pymss))

        box = QGroupBox("路径")
        box.setLayout(form)
        v.addWidget(box)

        self.paths_log = QTextEdit()
        self.paths_log.setReadOnly(True)
        self.paths_log.setMaximumHeight(90)
        v.addWidget(self.paths_log)
        v.addStretch(1)
        return w

    def _with_browse(self, edit: QLineEdit, slot) -> QWidget:
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        b = QPushButton("…")
        b.clicked.connect(slot)
        row.addWidget(b)
        w = QWidget()
        w.setLayout(row)
        return w

    def _open_data(self) -> None:
        d = self.app.config.data_root()
        if d:
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def _change_model_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择模型目录", self.model_edit.text())
        if d:
            rel = self.app.config.to_stored_path(d)
            self.model_edit.setText(self.app.config.models_dir() or d)
            self.app.config["model_dir"] = rel
            self.app.config.save()
            self.paths_log.append(f"模型目录已更改: {rel}")

    def _change_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self.out_edit.text())
        if d:
            rel = self.app.config.to_stored_path(d)
            self.out_edit.setText(self.app.config.output_dir() or d)
            self.app.config["default_output_dir"] = rel
            self.app.config.save()
            self.paths_log.append(f"输出目录已更改: {rel}")

    def _browse_worker(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择包含 worker.py 的目录", self.worker_edit.text())
        if d:
            rel = self.app.config.to_stored_path(d)
            wd = self.app.config.resolve_worker_dir()
            self.worker_edit.setText(str(wd) if wd else d)
            self.app.config["worker_dir"] = rel
            self.app.config.save()

    def _browse_pymss(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择 pymss 包目录", self.pymss_edit.text())
        if d:
            rel = self.app.config.to_stored_path(d)
            self.pymss_edit.setText(self.app.config.resolve_pymss_path() or d)
            self.app.config["pymss_path"] = rel
            self.app.config.save()

    # ---- defaults + audio format -------------------------------------
    def _build_defaults(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.device_combo = QComboBox()
        form.addRow("默认设备", self.device_combo)

        self.source_combo = QComboBox()
        for label, val in [("ModelScope", "modelscope"), ("Hugging Face", "huggingface"),
                           ("HF Mirror", "hf-mirror")]:
            self.source_combo.addItem(label, val)
        form.addRow("下载源", self.source_combo)

        self.concurrent = QSpinBox()
        self.concurrent.setRange(1, 16)
        form.addRow("最大同时运行任务数", self.concurrent)

        self.dev_mode = QCheckBox("开发者模式")
        form.addRow("开发者模式", self.dev_mode)

        box = QGroupBox("默认参数")
        box.setLayout(form)
        v.addWidget(box)

        # audio format block
        af = QGroupBox("音频格式")
        af_layout = QFormLayout(af)
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["wav", "flac", "mp3", "m4a"])
        af_layout.addRow("默认输出格式", self.fmt_combo)

        self.wav_depth = QComboBox()
        self.wav_depth.addItems(["FLOAT", "PCM_16", "PCM_24", "PCM_32"])
        af_layout.addRow("WAV 位深", self.wav_depth)

        self.flac_depth = QComboBox()
        self.flac_depth.addItems(["PCM_16", "PCM_24"])
        af_layout.addRow("FLAC 位深", self.flac_depth)

        self.mp3_rate = QComboBox()
        self.mp3_rate.addItems(["128k", "192k", "256k", "320k"])
        af_layout.addRow("MP3 比特率", self.mp3_rate)

        self.m4a_rate = QComboBox()
        self.m4a_rate.addItems(["256k", "512k"])
        af_layout.addRow("M4A 比特率", self.m4a_rate)

        self.m4a_codec = QComboBox()
        self.m4a_codec.addItems(["aac", "alac"])
        af_layout.addRow("M4A 编码器", self.m4a_codec)
        v.addWidget(af)

        save = QPushButton("保存")
        save.clicked.connect(self._save)
        v.addWidget(save)
        v.addStretch(1)
        return w

    # ---- load / save --------------------------------------------------
    def _load(self) -> None:
        c = self.app.config
        self.data_edit.setText(c.data_root())
        self.model_edit.setText(c.models_dir())
        self.out_edit.setText(c.output_dir())
        wd = c.resolve_worker_dir()
        self.worker_edit.setText(str(wd) if wd else (c["worker_dir"] or ""))
        self.pymss_edit.setText(c.resolve_pymss_path() or "")
        self.device_combo.setCurrentText(c["default_device"] or "auto")
        self.source_combo.setCurrentText(c["download_source"] or "modelscope")
        self.concurrent.setValue(int(c["max_concurrent_separations"] or 1))
        self.dev_mode.setChecked(bool(c["developer_mode"]))
        self.fmt_combo.setCurrentText(c["default_output_format"] or "wav")
        self.wav_depth.setCurrentText(c["wav_bit_depth"] or "FLOAT")
        self.flac_depth.setCurrentText(c["flac_bit_depth"] or "PCM_24")
        self.mp3_rate.setCurrentText(c["mp3_bit_rate"] or "320k")
        self.m4a_rate.setCurrentText(c["m4a_bit_rate"] or "512k")
        self.m4a_codec.setCurrentText(c["m4a_codec"] or "aac")

    def _save(self) -> None:
        c = self.app.config
        c["default_device"] = self.device_combo.currentText()
        c["download_source"] = self.source_combo.currentText()
        c["max_concurrent_separations"] = self.concurrent.value()
        c["developer_mode"] = self.dev_mode.isChecked()
        c["default_output_format"] = self.fmt_combo.currentText()
        c["wav_bit_depth"] = self.wav_depth.currentText()
        c["flac_bit_depth"] = self.flac_depth.currentText()
        c["mp3_bit_rate"] = self.mp3_rate.currentText()
        c["m4a_bit_rate"] = self.m4a_rate.currentText()
        c["m4a_codec"] = self.m4a_codec.currentText()
        c.save()

    # ---- env-driven ---------------------------------------------------
    def on_env(self, env: dict) -> None:
        env = env or {}
        self.worker_ver.setText(f"Worker 版本: {env.get('workerVersion', '—')}")
        self.core_ver.setText(f"pymss 核心版本: {env.get('pymssVersion') or '—'}")
        # rebuild device options
        options = AppConfig.device_options(env)
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for opt in options:
            self.device_combo.addItem(opt["label"], opt["value"])
        cur = self.app.config["default_device"] or "auto"
        idx = self.device_combo.findData(cur)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        else:
            self.device_combo.setCurrentText(cur)
        self.device_combo.blockSignals(False)

    def _show_env(self, env: dict) -> None:
        """Render the full environment dump into the About log (moved here
        from the old home view)."""
        env = env or {}
        if not env:
            self.env_log.setPlainText("尚未获取运行环境信息，点击「检查运行环境」。")
            return
        lines = [
            f"Python: {env.get('pythonVersion','—')}",
            f"平台: {env.get('platform','—')}",
            f"Worker 版本: {env.get('workerVersion','—')}",
            f"pymss 可用: {env.get('pymssAvailable')}",
            f"pymss 版本: {env.get('pymssVersion') or '—'}",
            f"torch 可用: {env.get('torchAvailable')}",
            f"CUDA 可用: {env.get('cudaAvailable')}  ·  设备数: {env.get('cudaDeviceCount',0)}",
            f"MPS 可用: {env.get('mpsAvailable')}  ·  MLX 可用: {env.get('mlxAvailable')}",
            f"av 可用: {env.get('avAvailable')}  ·  librosa 可用: {env.get('librosaAvailable')}",
        ]
        self.env_log.setPlainText("\n".join(lines))

    def _on_nav(self, row: int) -> None:
        if 0 <= row < len(SECTIONS):
            self.stack.setCurrentIndex(row)

    # ---- engine config ------------------------------------------------
    def _build_engine(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)

        title = QLabel("<b style='font-size:16px'>处理引擎配置</b>")
        title.setStyleSheet("color:#1e293b;")
        v.addWidget(title)

        desc = QLabel(
            "为每种架构选择默认使用的引擎。"
            "当某个模型在 Pymss 引擎下输出异常时，"
            "可切换 MSST 引擎获得正确结果。"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#64748b;")
        v.addWidget(desc)
        v.addSpacing(10)

        from engine import ARCHITECTURE_ENGINES, ENGINE_LABELS

        saved_map = dict(self.app.config["engine_map"] or {})

        table = QGroupBox("架构 → 引擎映射（点击单选框修改）")
        grid = QGridLayout(table)
        grid.setSpacing(6)
        headers = ["架构", "引擎选择", "当前值"]
        for col, h in enumerate(headers):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setStyleSheet("color:#1e293b;")
            grid.addWidget(lbl, 0, col)

        self.engine_groups: dict[str, QButtonGroup] = {}
        self.engine_cur_labels: dict[str, QLabel] = {}

        row = 1
        for arch, engines in sorted(ARCHITECTURE_ENGINES.items()):
            grid.addWidget(QLabel(arch), row, 0)

            rb_widget = QWidget()
            rb_layout = QHBoxLayout(rb_widget)
            rb_layout.setContentsMargins(0, 0, 0, 0)
            rb_layout.setSpacing(8)
            group = QButtonGroup(rb_widget)
            self.engine_groups[arch] = group

            current = saved_map.get(arch, engines[0])

            for e in engines:
                rb = QRadioButton(ENGINE_LABELS.get(e, e))
                rb.setStyleSheet("font-size:12px;")
                if len(engines) == 1:
                    rb.setEnabled(False)
                    rb.setToolTip("此架构仅支持此引擎")
                else:
                    rb.setToolTip(f"选择 {ENGINE_LABELS.get(e, e)}")
                group.addButton(rb)
                rb_layout.addWidget(rb)
                if e == current:
                    rb.setChecked(True)

            group.idPressed.connect(lambda _a=arch: self._save_engine_map(_a))
            grid.addWidget(rb_widget, row, 1)

            lock = " 🔒" if len(engines) == 1 else ""
            cur_lbl = QLabel(f"{ENGINE_LABELS.get(current, current)}{lock}")
            cur_lbl.setStyleSheet(
                "color:#059669; font-weight:bold;" if not lock
                else "color:#94a3b8; font-weight:bold;"
            )
            self.engine_cur_labels[arch] = cur_lbl
            grid.addWidget(cur_lbl, row, 2)
            row += 1

        v.addWidget(table)

        bar = QHBoxLayout()
        save_btn = QPushButton("保存配置")
        save_btn.setStyleSheet(
            "QPushButton{background:#2563eb;color:#fff;padding:6px 24px;"
            "border-radius:4px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#1d4ed8;}"
        )
        save_btn.clicked.connect(self._save_engine_map_all)
        bar.addWidget(save_btn)
        bar.addStretch(1)
        v.addLayout(bar)
        v.addStretch(1)
        return w

    def _save_engine_map(self, arch: str) -> None:
        """单个架构引擎选择改变时自动保存。"""
        engine_map = self._collect_engine_map()
        self.app.config.data["engine_map"] = engine_map
        self.app.config.save()
        self._refresh_engine_cur_labels()

    def _save_engine_map_all(self) -> None:
        """保存全部。"""
        engine_map = self._collect_engine_map()
        self.app.config.data["engine_map"] = engine_map
        self.app.config.save()
        self._refresh_engine_cur_labels()

    def _collect_engine_map(self) -> dict:
        """收集所有单选框状态 → {arch: engine_name}"""
        from engine import ENGINE_LABELS
        result = {}
        for arch, group in self.engine_groups.items():
            checked = group.checkedButton()
            if checked is None:
                continue
            label = checked.text()
            for name, lbl in ENGINE_LABELS.items():
                if lbl == label:
                    result[arch] = name
                    break
        return result

    def _refresh_engine_cur_labels(self) -> None:
        """刷新当前值列显示。"""
        from engine import ARCHITECTURE_ENGINES, ENGINE_LABELS
        engine_map = dict(self.app.config["engine_map"] or {})
        for arch, cur_lbl in self.engine_cur_labels.items():
            engines = ARCHITECTURE_ENGINES.get(arch, ["pymss"])
            current = engine_map.get(arch, engines[0])
            lock = " 🔒" if len(engines) == 1 else ""
            cur_lbl.setText(f"{ENGINE_LABELS.get(current, current)}{lock}")
