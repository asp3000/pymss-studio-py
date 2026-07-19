"""Task data model mirroring the original Vue `task` store.

Each separation / workflow run is a Task. Tasks that share a jobId form a Job
(a batch). The TaskManager exposes Qt signals so views can react to changes.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, Signal


TERMINAL = {"done", "failed", "cancelled"}
INTERRUPTIBLE = {
    "queued", "preparing", "validating_input", "downloading_model",
    "ensuring_model", "loading_model", "separating", "writing_output",
}


@dataclass
class Task:
    task_id: str
    model: str = ""
    input: str = ""
    output: str = ""
    status: str = "queued"
    message: str = "Queued"
    created_at: float = field(default_factory=lambda: time.time() * 1000)
    updated_at: float = field(default_factory=lambda: time.time() * 1000)
    progress: int = 2
    stage_label: str = "Queued"
    progress_current: int | None = None
    progress_total: int | None = None
    progress_detail: str = ""
    files: list[str] = field(default_factory=list)
    outputs: list[dict[str, str]] = field(default_factory=list)  # [{stem, path}]
    logs: list[str] = field(default_factory=list)
    error: str = ""
    run_config: dict[str, Any] = field(default_factory=dict)
    job_id: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL


@dataclass
class Job:
    job_id: str
    output: str = ""
    tasks: list[Task] = field(default_factory=list)
    model: str = ""
    input_count: int = 0
    output_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    status: str = "separating"
    progress: int = 0


def _short(path: str) -> str:
    return path.split("/")[-1].split("\\")[-1] if path else ""


class TaskManager(QObject):
    # task-level updates
    task_added = Signal(Task)
    task_updated = Signal(Task)
    task_removed = Signal(str)
    # job-level
    job_updated = Signal(Job)
    # queue control
    schedule_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._tasks: dict[str, Task] = {}
        self._order: list[str] = []

    # ---- creation ---------------------------------------------------
    def add(self, task: Task) -> Task:
        self._tasks[task.task_id] = task
        if task.task_id not in self._order:
            self._order.insert(0, task.task_id)
        self.task_added.emit(task)
        return task

    def new_id(self, prefix: str = "sep") -> str:
        return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:5]}"

    def create_queued(self, input_path: str, model: str, run_config: dict[str, Any],
                      job_id: str | None = None, job_output: str | None = None,
                      output_layout: str = "folders", prefix: str = "sep") -> Task:
        now = time.time() * 1000
        tid = self.new_id(prefix)
        rid = self.new_id("result")
        job_output = job_output or "results"
        if output_layout == "folders":
            output = f"{job_output}/{_short(input_path)}" if job_output else _short(input_path)
        else:
            output = job_output or "results"
        task = Task(
            task_id=tid,
            job_id=job_id or tid,
            model=model,
            input=input_path,
            output=output,
            status="queued",
            message="Queued",
            created_at=now,
            updated_at=now,
            logs=[f"{_ts()} Queued"],
            run_config=run_config,
        )
        return self.add(task)

    # ---- updates ----------------------------------------------------
    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def all(self) -> list[Task]:
        return [self._tasks[i] for i in self._order]

    def running(self) -> list[Task]:
        return [t for t in self.all() if not t.is_terminal and t.status != "queued"]

    def queued(self) -> list[Task]:
        return [t for t in self.all() if t.status == "queued"]

    def completed(self) -> list[Task]:
        return [t for t in self.all() if t.status == "done" and (t.outputs or t.files)]

    def jobs(self) -> list[Job]:
        groups: dict[str, list[Task]] = {}
        for t in self.all():
            jid = t.job_id or t.task_id
            groups.setdefault(jid, []).append(t)
        out: list[Job] = []
        for jid, items in groups.items():
            items.sort(key=lambda x: x.created_at)
            primary = items[0]
            total = sum(t.progress for t in items)
            out.append(Job(
                job_id=jid,
                output=primary.output,
                tasks=items,
                model=primary.model,
                input_count=len(items),
                output_count=sum(len(t.outputs) for t in items),
                created_at=min(t.created_at for t in items),
                updated_at=max(t.updated_at for t in items),
                status=_resolve_job_status(items),
                progress=round(total / len(items)) if items else 0,
            ))
        return out

    def get_job(self, job_id: str | None) -> Job | None:
        if not job_id:
            return None
        return next((j for j in self.jobs() if j.job_id == job_id), None)

    # ---- event handlers (mirrors task.handleWorkerEvent) ------------
    def _touch(self, t: Task) -> None:
        t.updated_at = time.time() * 1000

    def on_task_started(self, tid: str, payload: dict) -> None:
        t = self.get(tid)
        if not t:
            return
        self.set_status(t, "preparing", "Task started", 6)

    def on_task_stage(self, tid: str, payload: dict) -> None:
        t = self.get(tid)
        if not t:
            return
        stage = payload.get("stage", "preparing")
        msg = payload.get("message") or stage
        self.set_status(t, stage, msg, payload.get("progress"))

    def on_task_progress(self, tid: str, payload: dict) -> None:
        t = self.get(tid)
        if not t:
            return
        stage = payload.get("stage", t.status)
        done = payload.get("done")
        total = payload.get("total")
        detail = payload.get("message")
        t.status = stage
        t.stage_label = stage
        t.message = detail or stage
        try:
            t.progress_current = int(float(done)) if done is not None else None
            t.progress_total = int(float(total)) if total is not None else None
        except (TypeError, ValueError):
            pass
        t.progress_detail = detail or ""
        if stage == "separating":
            if total and done is not None:
                try:
                    pct = min(99, int(round(float(done) / float(total) * 100)))
                except (TypeError, ValueError):
                    pct = t.progress
                t.progress = pct
            else:
                t.progress = t.progress
        self._touch(t)
        self.task_updated.emit(t)

    def on_task_log(self, tid: str, payload: dict) -> None:
        t = self.get(tid)
        if not t:
            return
        level = str(payload.get("level", "info"))
        msg = str(payload.get("message", ""))
        t.logs.append(f"{_ts()} {level}: {msg}")
        t.logs = t.logs[-300:]
        self._touch(t)
        self.task_updated.emit(t)

    def on_error(self, tid: str, payload: dict) -> None:
        t = self.get(tid)
        if not t:
            return
        if t.status == "cancelled":
            return
        msg = payload.get("message") or "Error"
        t.error = msg
        t.message = msg
        t.progress_current = None
        t.progress_total = None
        t.progress_detail = ""
        t.status = "failed"
        t.stage_label = "Failed"
        t.progress = 100
        t.logs.append(f"{_ts()} error: {msg}")
        self._touch(t)
        self.task_updated.emit(t)
        self.schedule_requested.emit()

    def on_task_done(self, tid: str, payload: dict) -> None:
        t = self.get(tid)
        if not t:
            return
        t.status = "done"
        t.message = "Done"
        t.stage_label = "Done"
        t.progress = 100
        t.progress_current = None
        t.progress_total = None
        t.progress_detail = ""
        t.files = payload.get("files") or []
        if payload.get("outputDir"):
            t.output = payload["outputDir"]
        t.outputs = payload.get("outputs") or [
            {"stem": _short(f), "path": f"{t.output}/{_short(f)}.{payload.get('outputFormat', 'wav')}"}
            for f in t.files
        ]
        t.error = ""
        self._touch(t)
        self.task_updated.emit(t)
        self.schedule_requested.emit()

    def on_task_cancelled(self, tid: str, payload: dict) -> None:
        t = self.get(tid)
        if not t:
            return
        t.status = "cancelled"
        t.message = payload.get("message") or "Cancelled"
        t.stage_label = "Cancelled"
        t.progress = 100
        self._touch(t)
        self.task_updated.emit(t)
        self.schedule_requested.emit()

    def set_status(self, t: Task, status: str, message: str | None = None, progress: int | None = None) -> None:
        if t.is_terminal and status not in TERMINAL:
            return
        t.status = status
        t.stage_label = status
        t.message = message or status
        if status != "separating":
            t.progress_current = None
            t.progress_total = None
            t.progress_detail = ""
        p = progress if progress is not None else t.progress
        t.progress = 100 if status in TERMINAL else (min(99, max(0, p)) if status == "separating" else max(t.progress, min(99, p)))
        t.logs.append(f"{_ts()} {t.message}")
        t.logs = t.logs[-300:]
        self._touch(t)
        self.task_updated.emit(t)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _resolve_job_status(items: list[Task]) -> str:
    if any(not t.is_terminal for t in items):
        return "separating"
    if all(t.status == "done" for t in items):
        return "done"
    if all(t.status == "cancelled" for t in items):
        return "cancelled"
    if all(t.status == "failed" for t in items):
        return "failed"
    if any(t.status == "done" for t in items):
        return "done"
    if any(t.status == "failed" for t in items):
        return "failed"
    return "cancelled"
