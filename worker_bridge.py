"""Bridge layer that replaces the Tauri (Rust) shell.

The original Vue UI talked to Tauri commands which in turn spawned
``python worker.py <command> --payload <json>`` and streamed JSON events on
stdout. This bridge does exactly that: it spawns the worker subprocess, parses
the line-delimited JSON event envelope, and forwards events as Qt signals.

Commands (mirrored from worker.py dispatch):
  health, env_info, list_models, model_info, delete_model,
  model_storage_summary, cleanup_model_residual_files, download_model,
  audio_metadata, waveform_peaks, export_editor_mix, infer, infer_workflow
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, Signal

from .config import AppConfig


class WorkerBridge(QObject):
    # raw event envelope: (type, payload, taskId)
    event_received = Signal(str, dict, str)
    # one-shot request results keyed by channel: (channel, payload_or_None)
    data_ready = Signal(str, object)
    # terminal notice for a streamed task: (task_id, ok)
    process_exited = Signal(str, bool)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # ---- command construction --------------------------------------
    def _worker_dir(self) -> Path:
        wd = self.config.resolve_worker_dir()
        if not wd:
            raise RuntimeError("worker.py not found; set python/worker paths in Settings")
        return wd

    def _cmd(self, command: str, payload: dict) -> list[str]:
        wd = self._worker_dir()
        py = self.config.resolve_python_exe()
        return [py, str(wd / "worker.py"), command, "--payload",
                json.dumps(payload, ensure_ascii=False)]

    def _spawn(self, command: str, payload: dict, task_id: str | None = None) -> subprocess.Popen:
        wd = self._worker_dir()
        # On Windows, python.exe is a console program — spawning it without
        # CREATE_NO_WINDOW flashes a brief console window every time a worker
        # command runs (startup, model selection, download, run...). Suppress it.
        kwargs: dict = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(wd),
            env=self.config.env_for_worker(),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        proc = subprocess.Popen(self._cmd(command, payload), **kwargs)
        if task_id:
            with self._lock:
                self._procs[task_id] = proc
        return proc

    # ---- one-shot request (blocking) ------------------------------
    def request(self, command: str, payload: dict, result_type: str, timeout: int = 600) -> dict | None:
        proc = self._spawn(command, payload)
        result: dict | None = None
        error: str | None = None
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") == result_type:
                    result = ev.get("payload", {})
                    break
                if ev.get("type") == "error" and error is None:
                    pl = ev.get("payload") or {}
                    error = pl.get("message") if isinstance(pl, dict) else str(pl)
                    if not error:
                        error = pl.get("error") if isinstance(pl, dict) else str(pl)
        finally:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
        if result is None and error is not None:
            result = {"__error__": error}
        return result

    def request_async(self, command: str, payload: dict, result_type: str, channel: str) -> None:
        def _run() -> None:
            try:
                res = self.request(command, payload, result_type)
            except Exception as exc:  # noqa: BLE001
                res = {"__error__": str(exc)}
            self.data_ready.emit(channel, res)
        threading.Thread(target=_run, daemon=True).start()

    # ---- streaming (infer / download / delete) --------------------
    def stream(self, command: str, payload: dict, task_id: str | None = None) -> subprocess.Popen:
        proc = self._spawn(command, payload, task_id)

        def _reader() -> None:
            ok = True
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    etype = ev.get("type", "")
                    pl = ev.get("payload", {}) or {}
                    # attach taskId from envelope if present
                    tid = ev.get("taskId") or ""
                    self.event_received.emit(etype, pl, tid)
            except Exception:  # noqa: BLE001
                ok = False
            finally:
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
                try:
                    proc.wait(timeout=30)
                except Exception:  # noqa: BLE001
                    ok = False
                with self._lock:
                    if task_id and self._procs.get(task_id) is proc:
                        self._procs.pop(task_id, None)
                self.process_exited.emit(task_id or "", ok)

        threading.Thread(target=_reader, daemon=True).start()
        return proc

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(task_id)
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
        return True

    # ---- convenience wrappers --------------------------------------
    def health(self) -> dict | None:
        return self.request("health", {}, "health")

    def env_info(self, done: Callable[[dict], None]) -> None:
        self.request_async("env_info", {}, "env_info", "env")

    def list_models(self, model_dir: str = "", supported_only: bool = False) -> None:
        self.request_async(
            "list_models",
            {"category": None, "supportedOnly": supported_only,
             "includeLocalState": True, "modelDir": model_dir or None},
            "models", "models",
        )

    def model_info(self, name: str, model_dir: str = "", channel: str = "model_info") -> None:
        self.request_async("model_info", {"model": name, "modelDir": model_dir or None},
                           "model_info", channel)

    def model_storage_summary(self, model_dir: str = "") -> None:
        self.request_async("model_storage_summary", {"modelDir": model_dir or None},
                           "model_storage_summary", "storage")

    def audio_metadata(self, path: str, channel: str = "audio_metadata") -> None:
        self.request_async("audio_metadata", {"path": path}, "audio_metadata", channel)

    def waveform_peaks(self, path: str, channel: str = "waveform") -> None:
        self.request_async("waveform_peaks", {"path": path, "resolution": 1400},
                           "waveform_peaks", channel)

    def infer(self, payload: dict, job_id: str) -> None:
        self.stream("infer", payload, job_id)

    def infer_workflow(self, payload: dict, job_id: str) -> None:
        self.stream("infer_workflow", payload, job_id)

    def download_model(self, name: str, model_dir: str = "", source: str = "modelscope",
                       force: bool = False) -> str:
        task_id = f"download_{name}_{int(__import__('time').time())}"
        payload = {
            "taskId": task_id, "model": name, "modelDir": model_dir or None,
            "source": source, "endpoint": None, "force": force,
        }
        self.stream("download_model", payload, task_id)
        return task_id

    def delete_model(self, name: str, model_dir: str = "") -> str:
        task_id = f"delete_{name}_{int(__import__('time').time())}"
        payload = {"taskId": task_id, "model": name, "modelDir": model_dir or None}
        self.stream("delete_model", payload, task_id)
        return task_id

    def cleanup_residual(self, model_dir: str = "") -> str:
        task_id = f"cleanup_residual_{int(__import__('time').time())}"
        payload = {"taskId": task_id, "modelDir": model_dir or None}
        self.stream("cleanup_model_residual_files", payload, task_id)
        return task_id

    def export_editor_mix(self, project: dict, export_dir: str, fmt: str = "wav",
                          audio_params: dict | None = None, channel: str = "export") -> None:
        self.request_async(
            "export_editor_mix",
            {"project": project, "exportDir": export_dir, "format": fmt,
             "audioParams": audio_params or {}},
            "editor_mix_exported", channel,
        )
