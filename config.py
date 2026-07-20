"""Application configuration (replaces the Tauri store plugin).

Settings are persisted as a small JSON file next to this module. Mirrors the
original Vue `settings` store: audio bit-depth / bit-rate / codec per format,
runtime device, model directory, download source and concurrency.
"""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "python_exe": "python",
    "worker_dir": "",
    "pymss_path": "",
    "data_root": "",
    "default_device": "auto",
    "default_output_format": "wav",
    "default_output_dir": "",
    "model_dir": "",
    "download_source": "modelscope",
    "max_concurrent_separations": 1,
    "wav_bit_depth": "FLOAT",
    "flac_bit_depth": "PCM_24",
    "mp3_bit_rate": "320k",
    "m4a_bit_rate": "512k",
    "m4a_codec": "aac",
    "developer_mode": False,
    "engine_map": {},  # {架构名: 引擎名} 用户自定义引擎映射
}


class AppConfig:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else (Path(__file__).resolve().parent / "config.json")
        self.pkg_dir = Path(__file__).resolve().parent
        self.data: dict[str, Any] = dict(DEFAULTS)
        self.load()

    # ---- persistence -------------------------------------------------
    def load(self) -> None:
        if self.path.is_file():
            try:
                self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
            except Exception:
                pass
        self._fill_missing()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fill_missing(self) -> None:
        for k, v in DEFAULTS.items():
            self.data.setdefault(k, v)

    def __getitem__(self, key: str) -> Any:
        return self.data.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    # ---- derived paths ----------------------------------------------
    @staticmethod
    def _resolve_path(pkg_dir: Path, value: str | None) -> str:
        """Resolve a possibly-relative path against the package directory.

        Only the bare interpreter names ``python``/``python3`` are returned
        unchanged (so they can still be looked up on PATH). Everything else —
        including separator-less relative directory names like ``data`` or
        ``python`` — is resolved relative to the package directory so the
        config file stays portable. Already-absolute paths are kept as-is.
        """
        value = (value or "").strip()
        if not value:
            return value
        if value in ("python", "python3"):
            return value
        if os.path.isabs(value):
            return value
        return str((pkg_dir / value).resolve())

    def resolve_python_exe(self) -> str:
        """Absolute path of the interpreter that runs ``worker.py``.

        使用 ``pythonw.exe``（GUI 子系统）避免 worker 子进程弹出控制台窗口。
        优先使用项目的 venv（``venv/Scripts/pythonw.exe``），它是唯一保证
        携带了 ``pymss`` + torch + PySide6 的解释器。Windows venv 的
        pythonw.exe 是 launcher，会 CreateProcess 系统 Python，这是正常行为。
        配合 worker_bridge.py 的 ``CREATE_NO_WINDOW`` 可完全消除窗口闪烁。
        """
        configured = self.data.get("python_exe")
        resolved = self._resolve_path(self.pkg_dir, configured) if configured else ""
        if resolved and os.path.isfile(resolved):
            return resolved
        # 1) 项目 venv（pythonw = GUI 子系统，无控制台窗口）
        for cand in ("venv/Scripts/pythonw.exe", ".venv/Scripts/pythonw.exe"):
            venv_py = self.pkg_dir / cand
            if venv_py.is_file():
                return str(venv_py.resolve())
        # 2) 兜底：fallback 到 python.exe
        for cand in ("venv/Scripts/python.exe", ".venv/Scripts/python.exe"):
            venv_py = self.pkg_dir / cand
            if venv_py.is_file():
                return str(venv_py.resolve())
        # 3) PATH 查找
        return "python"

    @staticmethod
    def _to_stored_path(pkg_dir: Path, path: str) -> str:
        """Store ``path`` relative to the package dir when possible.

        Already-relative values are kept as-is. Absolute values are rewritten
        as a relative (portable) path using ``os.path.relpath`` (so a sibling
        directory such as ``../pymss-studio/python`` is stored that way too).
        Paths on a different drive (or otherwise not relatable) stay absolute.
        """
        path = (path or "").strip()
        if not path or not Path(path).is_absolute():
            return path
        try:
            rel = os.path.relpath(Path(path).resolve(), pkg_dir.resolve())
            return rel.replace(os.sep, "/")
        except ValueError:
            return str(Path(path).resolve())

    def to_stored_path(self, path: str) -> str:
        """See :meth:`_to_stored_path` (bound to this package directory)."""
        return AppConfig._to_stored_path(self.pkg_dir, path)

    def resolve_pymss_path(self) -> str:
        return self._resolve_path(self.pkg_dir, self.data.get("pymss_path"))

    def resolve_worker_dir(self) -> Path | None:
        # The worker directory is always a directory path (e.g. "python"),
        # never a bare interpreter name, so resolve it relative to the package
        # directory explicitly (bypassing the python/python3 interpreter shortcut).
        wd = (self.data.get("worker_dir") or "").strip()
        if wd:
            p = Path(wd) if os.path.isabs(wd) else (self.pkg_dir / wd).resolve()
            if (p / "worker.py").is_file():
                return p
        here = self.pkg_dir
        for cand in (here.parent / "python", here / "python", here.parent.parent / "python"):
            if (cand / "worker.py").is_file():
                return cand
        return None

    def data_root(self) -> str:
        root = (self.data.get("data_root") or "").strip()
        if root:
            return self._resolve_path(self.pkg_dir, root)
        # fall back to a sensible location next to the package
        candidate = self.pkg_dir.parent / "data"
        return str(candidate)

    def models_dir(self) -> str:
        md = (self.data.get("model_dir") or "").strip()
        if not md:
            return ""
        return self._resolve_path(self.pkg_dir, md)

    def output_dir(self) -> str:
        od = (self.data.get("default_output_dir") or "").strip()
        if od:
            return self._resolve_path(self.pkg_dir, od)
        # Always resolve to an absolute path under the package directory so
        # outputs land in <package>/results regardless of the worker's CWD.
        return str((self.pkg_dir / "results").resolve())

    def env_for_worker(self) -> dict[str, str]:
        env = dict(os.environ)
        # Sandbox the Python import path: drop any user-injected PYTHONPATH and
        # skip the per-user site-packages so a stray/older pymss in
        # %APPDATA%\Python\... can never shadow the venv's pinned pymss 2.0.14.
        # PATH is deliberately left intact -- pymss needs `ffmpeg`, torch needs
        # CUDA DLLs, and the downloader uses git/aria2c, all resolved via PATH.
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        # Make the bundled aria2c/ffmpeg (bin/) discoverable for the worker
        # subprocess. The worker resolves aria2c via PATH; pymss resolves
        # ffmpeg via PATH. Prepending our bin dir guarantees both work even
        # when the system has neither on PATH.
        bin_dir = (self.pkg_dir / "bin").resolve()
        if bin_dir.is_dir():
            existing = env.get("PATH", "")
            env["PATH"] = f"{bin_dir}{os.pathsep}{existing}" if existing else str(bin_dir)
            env["PYMSS_STUDIO_BIN"] = str(bin_dir)
        pymss = (self.data.get("pymss_path") or "").strip()
        if pymss:
            env["PYMSS_STUDIO_PYMSS_PATH"] = self._resolve_path(self.pkg_dir, pymss)
        data = (self.data.get("data_root") or "").strip()
        if data:
            env["PYMSS_STUDIO_DATA_ROOT"] = self._resolve_path(self.pkg_dir, data)
        od = self.output_dir()
        if od:
            env["PYMSS_STUDIO_DEFAULT_OUTPUT_DIR"] = od
        return env

    # ---- audio params (mirrors settings.getAudioParams) ------------
    def get_audio_params(self) -> dict[str, str]:
        return {
            "wav_bit_depth": self.data["wav_bit_depth"],
            "flac_bit_depth": self.data["flac_bit_depth"],
            "mp3_bit_rate": self.data["mp3_bit_rate"],
            "m4a_bit_rate": self.data["m4a_bit_rate"],
            "m4a_codec": self.data["m4a_codec"],
        }

    # ---- runtime device config (mirrors settings.getRuntimeDeviceConfig) ----
    def get_runtime_device_config(self, env: dict | None) -> dict[str, Any]:
        env = env or {}
        selected = self.data["default_device"]

        def parsed_cuda_ids(value: str) -> list[int]:
            try:
                return [int(value)]
            except ValueError:
                return [0]

        if selected.startswith("cuda:"):
            return {"device": "cuda", "deviceIds": parsed_cuda_ids(selected[len("cuda:"):])}
        if selected == "cuda":
            return {"device": "cuda", "deviceIds": [0]}
        if selected == "auto" and env.get("mlxAvailable"):
            return {"device": "mlx", "deviceIds": [0]}
        if selected in ("cpu", "mps", "mlx", "auto"):
            return {"device": selected, "deviceIds": [0]}
        return {"device": "auto", "deviceIds": [0]}

    @staticmethod
    def device_options(env: dict | None) -> list[dict[str, Any]]:
        env = env or {}
        options: list[dict[str, Any]] = [
            {"label": "Auto (优先使用可用显卡)", "value": "auto", "type": "auto"},
            {"label": "CPU", "value": "cpu", "type": "cpu", "deviceIds": [0]},
        ]
        cuda_devices = env.get("cudaDevices") or []
        if cuda_devices:
            for gpu in cuda_devices:
                mem = gpu.get("totalMemoryBytes")
                suffix = f" · {(mem / 1024 / 1024 / 1024):.1f} GB" if mem else ""
                options.append({
                    "label": f"CUDA {gpu.get('id')}: {gpu.get('name', '')}{suffix}",
                    "value": f"cuda:{gpu.get('id')}",
                    "type": "cuda",
                    "deviceIds": [gpu.get("id", 0)],
                })
        else:
            if env.get("cudaAvailable"):
                count = max(1, int(env.get("cudaDeviceCount", 1) or 1))
                for i in range(count):
                    options.append({"label": f"CUDA {i}", "value": f"cuda:{i}", "type": "cuda", "deviceIds": [i]})
        if env.get("mlxAvailable") or env.get("mpsAvailable"):
            options.append({"label": "Apple MLX", "value": "mlx", "type": "mlx", "deviceIds": [0]})
        if env.get("mpsAvailable"):
            options.append({"label": "Apple MPS", "value": "mps", "type": "mps", "deviceIds": [0]})
        return options
