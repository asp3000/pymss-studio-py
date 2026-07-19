"""Main window: sidebar navigation + stacked views + event routing.

This class doubles as the ``App`` object that every view receives as ``self.app``.
It owns the shared state (config, bridge, tasks, models, workflows, env) and
routes worker events to the appropriate view.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QListWidget, QStackedWidget, QHBoxLayout, QVBoxLayout,
    QLabel,
)

from .config import AppConfig
from .worker_bridge import WorkerBridge
from .task_model import TaskManager, Task
from .views.separate import SeparateView
from .views.models import ModelsView
from .views.workflows import WorkflowsView
from .views.settings import SettingsView

NAV = [
    ("分离", "separate"),
    ("模型库", "models"),
    ("工作流", "workflows"),
    ("设置", "settings"),
]


class MainWindow(QMainWindow):
    """Application shell and shared ``App`` state."""

    model_info_ready = Signal(str, dict)  # (channel, info)
    models_changed = Signal(list)
    env_changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pymss Studio (Python GUI)")
        self.resize(1180, 760)

        self.config = AppConfig()
        self.bridge = WorkerBridge(self.config)
        self.tasks = TaskManager()
        self.models: list[dict] = []
        self.model_infos: dict[str, dict] = {}
        self.selected_model: str = ""
        self.selected_workflow: str = ""
        self.workflows: list[dict] = []
        self.env: dict[str, Any] = {}
        self._download_state: dict[str, dict] = {}
        self._delete_state: dict[str, dict] = {}

        self._build_ui()
        self._wire()
        self._load_workflows()

        # initial data
        self.refresh_env()
        self.refresh_models()

    # ---- UI ----------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.nav = QListWidget()
        self.nav.setFixedWidth(120)
        for label, _ in NAV:
            self.nav.addItem(label)
        self.nav.currentRowChanged.connect(self._on_nav)
        root.addWidget(self.nav)

        self.stack = QStackedWidget()
        self.views: dict[str, QWidget] = {}
        self.views["separate"] = SeparateView(self)
        self.views["models"] = ModelsView(self)
        self.views["workflows"] = WorkflowsView(self)
        self.views["settings"] = SettingsView(self)
        for v in self.views.values():
            self.stack.addWidget(v)
        root.addWidget(self.stack, 1)

        self.nav.setCurrentRow(0)

    def _wire(self) -> None:
        self.bridge.event_received.connect(self._on_event)
        self.bridge.data_ready.connect(self._on_data)
        self.bridge.process_exited.connect(self._on_exit)
        self.tasks.task_updated.connect(self._on_task_updated)
        self.tasks.task_added.connect(self._on_task_updated)

    def _on_nav(self, row: int) -> None:
        if 0 <= row < len(NAV):
            self.stack.setCurrentWidget(self.views[NAV[row][1]])

    def switch_to(self, name: str) -> None:
        for i, (_, key) in enumerate(NAV):
            if key == name:
                self.nav.setCurrentRow(i)
                return

    # ---- env / models ------------------------------------------------
    def refresh_env(self) -> None:
        self.bridge.env_info(None)

    def refresh_models(self) -> None:
        self.bridge.list_models(self.config.models_dir())

    # ---- workflows ----------------------------------------------------
    def _workflows_path(self) -> Path:
        # User workflow data lives under data/settings/ (matches Vue layout).
        return Path(__file__).resolve().parent / "data" / "settings" / "workflows.json"

    def load_workflows(self) -> None:
        """Public alias used by views (e.g. SeparateView)."""
        self._load_workflows()

    def _load_workflows(self) -> None:
        path = self._workflows_path()
        self.workflows = []
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.workflows = data.get("workflows", []) or []
                elif isinstance(data, list):
                    self.workflows = data
            except Exception:
                self.workflows = []
        sel = data.get("selectedWorkflowId", "") if isinstance(data, dict) else ""
        if sel and any(w.get("id") == sel for w in self.workflows):
            self.selected_workflow = sel
        else:
            self.selected_workflow = ""
        for wf in self.workflows:
            wf.setdefault("id", uuid.uuid4().hex[:8])
            wf.setdefault("name", "未命名工作流")
            wf.setdefault("description", "")
            wf.setdefault("definition", {"steps": []})
            wf.setdefault("createdAt", int(time.time() * 1000))
            wf.setdefault("updatedAt", int(time.time() * 1000))

    def save_workflows(self) -> None:
        path = self._workflows_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"selectedWorkflowId": self.selected_workflow,
                 "workflows": self.workflows},
                ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_selected_workflow(self) -> dict | None:
        if not self.selected_workflow:
            return None
        return next((w for w in self.workflows if w.get("id") == self.selected_workflow), None)

    # ---- event routing ----------------------------------------------
    def _on_event(self, etype: str, pl: dict, tid: str) -> None:
        if etype in (
            "task_started", "task_stage", "task_progress", "task_log",
            "task_done", "task_cancelled", "error",
        ):
            self._route_task_event(etype, pl, tid)
            return
        if etype.startswith("download_"):
            self.views["models"].on_download_event(etype, pl, tid)
            return
        if etype.startswith("model_delete_"):
            self.views["models"].on_delete_event(etype, pl, tid)
            return
        if etype.startswith("model_residual_cleanup") or etype == "model_residual_cleaned":
            self.views["models"].on_residual_event(etype, pl, tid)
            return

    def _route_task_event(self, etype: str, pl: dict, tid: str) -> None:
        if etype == "task_started":
            self.tasks.on_task_started(tid, pl)
        elif etype == "task_stage":
            self.tasks.on_task_stage(tid, pl)
        elif etype == "task_progress":
            self.tasks.on_task_progress(tid, pl)
        elif etype == "task_log":
            self.tasks.on_task_log(tid, pl)
        elif etype == "task_done":
            self.tasks.on_task_done(tid, pl)
        elif etype == "task_cancelled":
            self.tasks.on_task_cancelled(tid, pl)
        elif etype == "error":
            self.tasks.on_error(tid, pl)

    def _on_task_updated(self, task: Task) -> None:
        try:
            self.views["separate"].on_task_updated(task)
        except Exception:
            pass

    def _on_data(self, channel: str, payload: object) -> None:
        pl = payload if isinstance(payload, dict) else {}
        if channel == "env":
            self.env = pl or {}
            self.env_changed.emit(self.env)
            try:
                self.views["settings"].on_env(self.env)
            except Exception:
                pass
            return
        if channel == "models":
            if isinstance(pl, dict) and pl.get("__error__"):
                try:
                    self.views["models"].on_models_error(str(pl["__error__"]))
                except Exception:
                    pass
                return
            self.models = pl.get("models") or []
            self.models_changed.emit(self.models)
            try:
                self.views["models"].on_models(self.models)
            except Exception:
                pass
            try:
                self.views["separate"].on_models(self.models)
            except Exception:
                pass
            return
        if channel == "sep_model_info":
            try:
                self.views["separate"].on_model_info(pl)
            except Exception:
                pass
            return
        if channel == "models_detail":
            try:
                self.views["models"].on_model_info(pl)
            except Exception:
                pass
            return
        if channel == "storage":
            try:
                self.views["models"].on_storage(pl)
            except Exception:
                pass
            return

    def _on_exit(self, task_id: str, ok: bool) -> None:
        if task_id.startswith("download_") or task_id.startswith("delete_"):
            self.refresh_models()
