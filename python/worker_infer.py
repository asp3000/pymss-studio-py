from __future__ import annotations

import os
import inspect
import shutil
import subprocess
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from worker_audio import _apply_stereo_pan, _equal_power_fade, _read_audio, _resample_audio
from worker_models import _derive_overlap_size_from_num_overlap
from worker_protocol import _as_bool, _as_float, _as_int, emit, emit_error


# pymss versions before 2.x did not accept ``save_as_folder`` on
# ``MSSeparator.__init__``. Probe once and skip the kwarg on older installs so
# separation does not crash with "unexpected keyword argument 'save_as_folder'".
_MSS_SEP_PARAMS_CACHE: set | None = None


def _mss_separator_supported_params() -> set:
    """Return the set of keyword args accepted by the installed pymss
    ``MSSeparator.__init__``. Cached after first introspection so the
    signature is inspected only once per worker process."""
    global _MSS_SEP_PARAMS_CACHE
    if _MSS_SEP_PARAMS_CACHE is None:
        try:
            from pymss import MSSeparator  # type: ignore
            _MSS_SEP_PARAMS_CACHE = set(
                inspect.signature(MSSeparator.__init__).parameters.keys()
            )
        except Exception:
            # If we cannot introspect, fall back to assuming the modern API.
            _MSS_SEP_PARAMS_CACHE = set()
    return _MSS_SEP_PARAMS_CACHE


def _make_separator(**desired: Any) -> Any:
    """Construct ``MSSeparator`` or ``MsstSeparatorAdapter``, passing only the
    parameters the installed pymss version actually accepts.

    When ``engine="msst"``, create an adapter that wraps MsstEngine with the
    same ``process_folder()`` interface as ``MSSeparator``.
    Otherwise (default), create standard ``MSSeparator``.
    """
    engine = desired.pop("engine", "pymss")
    if engine == "msst":
        from engine import MsstSeparatorAdapter
        # For MsstSeparatorAdapter, pass everything through (it handles **kwargs)
        return MsstSeparatorAdapter(**desired)

    from pymss import MSSeparator  # type: ignore
    supported = _mss_separator_supported_params()
    kwargs = {k: v for k, v in desired.items() if k in supported}
    if not kwargs:  # extreme fallback: try the full set anyway
        kwargs = desired
    return MSSeparator(**kwargs)


def _resolve_ffmpeg_path() -> str | None:
    """Locate ffmpeg: prefer the bundled binary in PYMSS_STUDIO_BIN, else PATH."""
    bin_dir = os.environ.get("PYMSS_STUDIO_BIN")
    if bin_dir:
        cand = os.path.join(bin_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if os.path.isfile(cand):
            return cand
    return shutil.which("ffmpeg")


def _standardize_input_to_wav(input_path: str, task_id: str) -> tuple[str, str | None]:
    """When input standardization is requested, convert the input to a canonical
    WAV (44.1kHz, stereo, 16-bit PCM) using ffmpeg so the separation engine
    always receives a uniform PCM file. Returns ``(path_to_use, temp_path)`` where
    ``temp_path`` is ``None`` unless a temporary WAV was created (caller deletes it).
    """
    p = Path(input_path)
    if not p.is_file():
        return input_path, None  # directories / missing inputs: let pymss handle
    if p.suffix.lower() == ".wav":
        return input_path, None  # already a WAV
    ffmpeg = _resolve_ffmpeg_path()
    if not ffmpeg:
        emit("task_log", {"level": "warning",
                          "message": "输入标准化需要 ffmpeg，但未找到；将直接使用原始文件。"},
             task_id=task_id)
        return input_path, None
    out = Path(tempfile.gettempdir()) / f"{p.stem}.wav"
    cmd = [ffmpeg, "-y", "-nostdin", "-v", "error", "-i", str(p),
           "-ar", "44100", "-ac", "2", "-sample_fmt", "s16", str(out)]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=600, check=True)
    except Exception as exc:  # noqa: BLE001
        emit("task_log", {"level": "warning",
                          "message": f"ffmpeg 输入标准化失败（{exc}），将直接使用原始文件。"},
             task_id=task_id)
        return input_path, None
    emit("task_log", {"level": "info",
                      "message": f"输入标准化：已将 {p.name} 提取为标准 WAV -> {out.name}"},
         task_id=task_id)
    return str(out), str(out)


def _separate_with_pymss(separator: Any, actual_input: str, task_id: str | None = None) -> list[str]:
    """Run ``separator.process_folder`` on a single file or a folder.

    pymss 2.0.14+ accepts both a file path and a directory for
    ``process_folder``. Some older pymss builds, however, validate the input
    strictly as a directory and raise ``Input folder '<file>' does not exist``
    when given a single audio file -- i.e. they treat the file as a missing
    directory. To stay compatible with any installed version, we try the
    direct call first (works on modern pymss and on folders of any version),
    and only fall back to a temporary folder (containing just that file) if the
    loaded pymss rejects the file path as a folder. Output stems are derived
    from the original file name, so downstream output resolution is unchanged.
    """
    import shutil
    import tempfile

    src = Path(actual_input)
    # Diagnostic: report GPU memory headroom before separation so an OOM can be
    # attributed to a dirty GPU (leftover CUDA context from a prior run / other
    # apps) versus a genuine per-forward memory regression.
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            alloc_b = torch.cuda.memory_allocated()
            emit("task_log", {
                "level": "info",
                "message": (
                    f"[gpu-check] before separation: free="
                    f"{free_b / 1024 ** 3:.2f} GiB / total={total_b / 1024 ** 3:.2f} GiB, "
                    f"already allocated by PyTorch={alloc_b / 1024 ** 3:.2f} GiB"
                ),
            }, task_id=task_id)
    except Exception:
        pass

    # Fast path: works for folders on every version and for files on 2.0.14+.
    try:
        return separator.process_folder(actual_input)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        # The only failure we recover from is an old pymss that only accepts a
        # directory (its error mentions "folder" or "not a directory"). Any
        # other error (missing file, CUDA OOM, model load failure, ...) is a
        # genuine failure and must propagate unchanged.
        if not src.is_file() or ("folder" not in msg and "not a directory" not in msg):
            raise
        tmp = Path(tempfile.mkdtemp(prefix="pymss_in_"))
        try:
            dst = tmp / src.name
            shutil.copy2(src, dst)
            return separator.process_folder(str(tmp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class JsonLogHandler:
    def __init__(self, task_id: str):
        import logging
        self.task_id = task_id
        self.level = logging.INFO
        self.messages: list[str] = []  # buffer to inspect on errors

    def setLevel(self, level: int) -> None:
        self.level = level

    def handle(self, record: Any) -> bool:
        if record.levelno < self.level:
            return False
        msg = record.getMessage()
        self.messages.append(msg)
        emit("task_log", {"level": record.levelname.lower(), "message": msg}, task_id=self.task_id)
        return True


def collect_outputs(output_dir: str, success_files: list[str], output_format: str) -> list[dict[str, str]]:
    base = Path(output_dir)
    outputs: list[dict[str, str]] = []
    if not base.exists():
        return outputs
    success_stems = {Path(name).stem for name in success_files}
    for path in base.rglob(f"*.{output_format.lower()}"):
        if success_stems and not any(path.stem.startswith(stem + "_") or path.stem == stem for stem in success_stems):
            continue
        stem = path.stem.split("_")[-1] if "_" in path.stem else path.stem
        outputs.append({"stem": stem, "path": str(path)})
    return outputs


def resolve_pymss_output_dir(output_dir: str, success_files: list[str], fallback_input: str, save_as_folder: bool) -> str:
    if not save_as_folder:
        return str(Path(output_dir))
    file_name = Path(success_files[0]).stem if success_files else Path(fallback_input).stem
    return str(Path(output_dir) / file_name)


def _finalize_pymss_folder_outputs(output_base: str, actual_input: str, original_input: str,
                                   flat: bool = False, model_name: str = "",
                                   skip_dirs: set[str] | None = None) -> tuple[str, list[str]]:
    """Reorganise pymss output into the user-chosen layout.

    Pymss may create per-instrument subdirectories (``other/``, ``vocals/``)
    or a per-input stem directory depending on internal parameters.  This
    function scans *all* child directories under ``output_base``, collects
    every file inside them, moves the files to the correct final location,
    and removes the now-empty directories.

    When *model_name* is given the output is placed under a model-named
    subdirectory first (``results/<model_name>/...``) so that multi-model
    workflows keep each model's output separated.

    * **flat** — files land in ``output_base/<model_name>/`` (or just
      ``output_base/`` when model_name is empty).
    * **folder** — files land in
      ``output_base/<model_name>/<original_input_name>/``.

    Filenames are kept as-is (no renaming)."""
    src_root = Path(output_base)
    # model-named intermediate directory (per-model grouping)
    model_dir = src_root / model_name if model_name else src_root
    if model_name:
        model_dir.mkdir(exist_ok=True)
    # final destination: flat → model_dir;  folder → model_dir/<input_name>
    dst_dir = model_dir / Path(original_input).name if not flat else model_dir
    if not flat:
        dst_dir.mkdir(exist_ok=True)
    new_files: list[str] = []
    try:
        for entry in list(src_root.iterdir()):
            if entry.is_dir() and entry != dst_dir and entry != model_dir \
                    and (skip_dirs is None or entry.name not in skip_dirs):
                # Pymss-created subdirectory: move files out
                for f in entry.iterdir():
                    if not f.is_file():
                        continue
                    target = dst_dir / f.name
                    if target.exists():
                        try:
                            target.unlink()
                        except OSError:
                            pass
                    f.rename(target)
                    new_files.append(str(target))
                try:
                    entry.rmdir()
                except OSError:
                    pass
            elif entry.is_file():
                # Loose file written directly to output_base (some models
                # write flat when save_as_folder=False): move to dst_dir.
                target = dst_dir / entry.name
                if target.exists():
                    try:
                        target.unlink()
                    except OSError:
                        pass
                entry.rename(target)
                new_files.append(str(target))
        return str(dst_dir), new_files
    except OSError:
        import sys
        traceback.print_exc(file=sys.stderr)
        leftover: list[str] = []
        for entry in list(src_root.iterdir()):
            if entry.is_dir() and entry != dst_dir and entry != model_dir \
                    and (skip_dirs is None or entry.name not in skip_dirs):
                for f in entry.iterdir():
                    leftover.append(str(f))
        return str(dst_dir), leftover

def _emit_inference_error(exc: Exception, task_id: str) -> int:
    message = str(exc)
    lowered = message.lower()
    if "no audio stream found" in lowered:
        return emit_error(
            "INPUT_AUDIO_STREAM_MISSING",
            message,
            traceback.format_exc(),
            task_id=task_id,
        )
    if "invalid data found" in lowered or "could not open input" in lowered:
        return emit_error(
            "INPUT_MEDIA_UNSUPPORTED",
            message,
            traceback.format_exc(),
            task_id=task_id,
        )
    return emit_error("INFERENCE_FAILED", message, traceback.format_exc(), task_id=task_id)

def _close_separator(separator: Any) -> None:
    if separator is None:
        return
    close = getattr(separator, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    else:
        try:
            separator.del_cache()
        except Exception:
            pass

def _package_root() -> Path:
    """Absolute package root (<pkg>), i.e. the parent of this ``python`` dir.

    Worker subprocesses are spawned with CWD set to the ``python`` directory,
    so any relative output path must be resolved against the package root to
    keep results in ``<pkg>/results`` rather than ``<pkg>/python/results``.
    """
    return Path(__file__).resolve().parent.parent


def _normalize_output_dir(value: Any) -> str:
    default_output_dir = os.environ.get("PYMSS_STUDIO_DEFAULT_OUTPUT_DIR")
    output_dir = value or default_output_dir or "results"
    output_path = Path(str(output_dir))
    if not output_path.is_absolute():
        # Resolve a relative output (e.g. "results") against the package root
        # so it always lands in <pkg>/results regardless of the worker CWD.
        return str((_package_root() / output_path).resolve())
    return str(output_dir)

def _normalize_selected_stems(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    stems: list[str] = []
    seen: set[str] = set()
    for item in value:
        stem = str(item or "").strip()
        if not stem or stem.lower() in seen:
            continue
        stems.append(stem)
        seen.add(stem.lower())
    return stems

def _store_dirs_for_selected_stems(output_dir: str, selected_stems: list[str]) -> Any:
    if not selected_stems:
        return output_dir
    return {stem: output_dir for stem in selected_stems}

def _normalize_output_layout(value: Any) -> str:
    return "flat" if str(value or "").strip().lower() == "flat" else "folders"


def _normalize_device_ids(value: Any) -> list[int]:
    raw_ids = value if isinstance(value, list) else [value]
    device_ids: list[int] = []
    for raw_id in raw_ids:
        try:
            device_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if device_id >= 0 and device_id not in device_ids:
            device_ids.append(device_id)
    return device_ids or [0]


def _resolve_separator_device(device: Any, device_ids: Any) -> tuple[str, list[int], str]:
    requested_device = str(device or "auto").strip().lower() or "auto"
    normalized_ids = _normalize_device_ids(device_ids)
    if requested_device != "cuda":
        return requested_device, normalized_ids, requested_device

    # pymss currently leaves an explicit `device="cuda"` unchanged. PyTorch
    # interprets that bare value as the process default CUDA device (normally
    # cuda:0), so the selected device id would otherwise be ignored. Let
    # pymss's auto path resolve the indexed CUDA device from device_ids.
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA was selected, but PyTorch is not installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but CUDA is not available")
    device_count = int(torch.cuda.device_count())
    invalid_ids = [device_id for device_id in normalized_ids if device_id >= device_count]
    if invalid_ids:
        raise RuntimeError(
            f"CUDA device id(s) {invalid_ids} are unavailable; detected {device_count} CUDA device(s)"
        )
    return "auto", normalized_ids, f"cuda:{normalized_ids[0]}"


def _prepare_separator(
    *,
    payload: dict[str, Any],
    task_id: str,
    progress_callback: Any,
    logger: Any,
) -> Any:
    model_name = payload.get("model")
    if not model_name:
        raise ValueError("Missing model name")
    model_dir = payload.get("modelDir") or None
    download = bool(payload.get("download", True))
    source = payload.get("source") or "modelscope"
    endpoint = payload.get("endpoint") or None
    device, device_ids, resolved_device_label = _resolve_separator_device(
        payload.get("device"), payload.get("deviceIds")
    )
    output_format = payload.get("outputFormat") or "wav"
    selected_stems = _normalize_selected_stems(payload.get("selectedStems"))
    use_tta = bool(payload.get("useTta", False))
    debug = bool(payload.get("debug", False))
    inference_params = normalize_inference_params(
        payload.get("inferenceParams"),
        payload.get("inferenceParamsVersion"),
    )
    audio_params = normalize_audio_params(payload.get("audioParams"))

    emit("task_log", {
        "level": "info",
        "message": f"Runtime device: {resolved_device_label} (device_ids={device_ids})",
    }, task_id=task_id)

    if download:
        emit("task_stage", {"stage": "downloading_model", "message": "Checking model files"}, task_id=task_id)
    else:
        emit("task_stage", {"stage": "ensuring_model", "message": "Checking model files"}, task_id=task_id)
    from pymss import MSSeparator  # type: ignore
    from pymss.model_registry import resolve_model  # type: ignore
    emit("task_stage", {"stage": "loading_model", "message": "Loading model"}, task_id=task_id)
    try:
        resolved = resolve_model(model_name, model_dir=model_dir, require_supported=True, require_exists=True)
    except Exception as resolve_exc:
        if not download:
            raise resolve_exc
        from pymss.model_download import download_model  # type: ignore
        emit("task_stage", {"stage": "downloading_model", "message": "Downloading model files"}, task_id=task_id)
        download_model(model_name, model_dir=model_dir, source=source, endpoint=endpoint)
        resolved = resolve_model(model_name, model_dir=model_dir, require_supported=True, require_exists=True)
    if not isinstance(resolved, dict):
        raise RuntimeError(f"resolve_model returned unexpected result for {model_name!r}: {type(resolved).__name__}")
    resolved_model_type = resolved.get('model_type')
    resolved_model_path = resolved.get('model_path')
    if not resolved_model_type or not resolved_model_path:
        missing = [key for key in ('model_type', 'model_path') if not resolved.get(key)]
        raise RuntimeError(f"resolve_model result for {model_name!r} is missing required field(s): {', '.join(missing)}")
    runtime_inference_params = _enrich_inference_params_for_model(
        model_type=resolved_model_type,
        config_path=resolved.get('config_path'),
        inference_params=inference_params,
    )
    # 引擎选择：从 inference_params 中提取（UI 端写入）
    engine = runtime_inference_params.pop("engine", "pymss")
    sep_kwargs: dict[str, Any] = dict(
        model_type=resolved_model_type,
        model_path=resolved_model_path,
        config_path=resolved.get('config_path'),
        device=device,
        device_ids=device_ids,
        output_format=output_format,
        use_tta=use_tta,
        store_dirs=_store_dirs_for_selected_stems(_normalize_output_dir(payload.get("output")), selected_stems),
        audio_params=audio_params,
        logger=logger,
        debug=debug,
        progress_callback=progress_callback,
        inference_params=runtime_inference_params,
        save_as_folder=bool(payload.get("saveAsFolder", True)),
        engine=engine,
    )
    return _make_separator(**sep_kwargs)


def normalize_inference_params(payload_params: Any, version: Any = None) -> dict[str, Any]:
    if not isinstance(payload_params, dict):
        return {}

    params = dict(payload_params)
    try:
        version_value = int(version) if version is not None else None
    except (TypeError, ValueError):
        version_value = None

    if version_value is not None and version_value >= 2:
        if params.get("standardize") in {"", "default"}:
            params.pop("standardize", None)
        if params.get("normalize") in {"", "default"}:
            params.pop("normalize", None)
        return params

    # Legacy desktop tasks used `normalize` for input standardization and did not
    # send the new output-normalize flag separately. If `standardize` is absent,
    # treat the historical `normalize` field as the old input standardization
    # switch and default the new output normalize to False.
    if "standardize" not in params and "normalize" in params:
        legacy_standardize = params.pop("normalize")
        params["standardize"] = legacy_standardize
        params["normalize"] = False
        return params

    if "standardize" in params and "normalize" not in params:
        params["normalize"] = False
    elif "standardize" not in params and "normalize" not in params:
        params["standardize"] = True
        params["normalize"] = False
    return params


def _sanitize_runtime_inference_params(params: dict[str, Any]) -> dict[str, Any]:
    next_params = dict(params or {})

    def _drop_non_positive_int(key: str) -> None:
        if key not in next_params:
            return
        value = _as_int(next_params.get(key))
        if value is None or value <= 0:
            next_params.pop(key, None)
            return
        next_params[key] = value

    for numeric_key in ("batch_size", "overlap_size", "num_overlap", "chunk_size", "window_size"):
        _drop_non_positive_int(numeric_key)

    if "aggression" in next_params:
        aggression_value = _as_int(next_params.get("aggression"))
        if aggression_value is None or aggression_value < 0:
            next_params.pop("aggression", None)
        else:
            next_params["aggression"] = aggression_value

    if "post_process_threshold" in next_params:
        threshold_value = _as_float(next_params.get("post_process_threshold"))
        if threshold_value is None or threshold_value < 0:
            next_params.pop("post_process_threshold", None)
        else:
            next_params["post_process_threshold"] = threshold_value

    for bool_key in ("enable_post_process", "high_end_process", "standardize", "normalize"):
        if bool_key not in next_params:
            continue
        bool_value = _as_bool(next_params.get(bool_key))
        if bool_value is None:
            next_params.pop(bool_key, None)
        else:
            next_params[bool_key] = bool_value

    return next_params



def _enrich_inference_params_for_model(
    *,
    model_type: str | None,
    config_path: str | None,
    inference_params: dict[str, Any],
) -> dict[str, Any]:
    params = _sanitize_runtime_inference_params(inference_params)
    normalized_model_type = str(model_type or '').strip().lower()
    if normalized_model_type == 'vr':
        params.pop('num_overlap', None)
        return params
    if normalized_model_type == 'apollo':
        params.pop('num_overlap', None)
        return params
    if not config_path or not Path(config_path).is_file():
        params.pop('num_overlap', None)
        return params

    try:
        from pymss.config import load_config, to_plain  # type: ignore

        config = to_plain(load_config(str(config_path)))
    except Exception:
        params.pop('num_overlap', None)
        return params

    inference = config.get('inference') if isinstance(config, dict) else None
    audio = config.get('audio') if isinstance(config, dict) else None
    inference = inference if isinstance(inference, dict) else {}
    audio = audio if isinstance(audio, dict) else {}

    explicit_overlap_size = _as_int(params.get('overlap_size'))
    explicit_num_overlap = _as_int(params.get('num_overlap'))
    config_overlap_size = _as_int(inference.get('overlap_size'))
    config_num_overlap = _as_int(inference.get('num_overlap'))
    chunk_size = _as_int(params.get('chunk_size'))
    if chunk_size is None:
        chunk_size = _as_int(audio.get('chunk_size'))
    if chunk_size is None:
        chunk_size = _as_int(inference.get('chunk_size'))

    if explicit_overlap_size is None:
        derived_overlap_size: int | None = None
        if explicit_num_overlap is not None:
            derived_overlap_size = _derive_overlap_size_from_num_overlap(chunk_size, explicit_num_overlap)
        elif config_overlap_size is None and config_num_overlap is not None:
            derived_overlap_size = _derive_overlap_size_from_num_overlap(chunk_size, config_num_overlap)
        if derived_overlap_size is not None:
            params['overlap_size'] = derived_overlap_size

    params.pop('num_overlap', None)
    return params



def normalize_audio_params(payload_audio_params: Any) -> dict[str, Any]:
    defaults = {
        "wav_bit_depth": "FLOAT",
        "flac_bit_depth": "PCM_24",
        "mp3_bit_rate": "320k",
        "m4a_bit_rate": "512k",
        "m4a_codec": "aac",
        "m4a_aac_at_quality": 2,
    }
    if not isinstance(payload_audio_params, dict):
        return defaults
    normalized = {
        **defaults,
        **payload_audio_params,
    }
    normalized["m4a_codec"] = "aac" if str(normalized.get("m4a_codec") or "").strip().lower() == "aac" else "aac"
    return normalized


def cmd_infer_batch(payload: dict[str, Any]) -> int:
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return emit_error("INPUT_NOT_FOUND", "Missing batch tasks", task_id=payload.get("taskId") or None)

    root_task_id = str(payload.get("taskId") or raw_tasks[0].get("taskId") or f"sep_{int(datetime.now().timestamp())}")
    output_root = _normalize_output_dir(payload.get("output"))
    output_format = payload.get("outputFormat") or "wav"
    output_layout = _normalize_output_layout(payload.get("outputLayout"))
    save_as_folder = output_layout == "folders"
    batch_tasks: list[dict[str, str]] = []
    _batch_inf = normalize_inference_params(
        payload.get("inferenceParams"),
        payload.get("inferenceParamsVersion"),
    )
    batch_want_standardize = bool(_as_bool(_batch_inf.get("standardize")))

    for index, item in enumerate(raw_tasks):
        if not isinstance(item, dict):
            return emit_error("INPUT_NOT_FOUND", f"Invalid batch task at index {index}", task_id=root_task_id)
        task_id = str(item.get("taskId") or "").strip()
        input_path = str(item.get("input") or "").strip().strip('"').strip("'")
        if not task_id:
            return emit_error("INPUT_NOT_FOUND", f"Missing taskId for batch task {index + 1}", task_id=root_task_id)
        if not input_path:
            return emit_error("INPUT_NOT_FOUND", f"Missing input path for batch task {task_id}", task_id=task_id)
        input_path = str(Path(input_path))
        source_path = Path(input_path)
        if not source_path.exists():
            return emit_error("INPUT_NOT_FOUND", f"Input path does not exist: {input_path}", task_id=task_id)
        batch_tasks.append({
            "taskId": task_id,
            "input": str(source_path),
        })

    logger = None
    log_handler = None
    separator = None
    active_task_id: str | None = None
    last_reported_done: float | None = None
    last_reported_total: float | None = None
    last_progress_message = ""

    def emit_batch_progress(done: Any, total: Any, message: str | None = None) -> None:
        nonlocal last_reported_done, last_reported_total, last_progress_message
        try:
            total_value = float(total)
            done_value = float(done)
        except (TypeError, ValueError):
            return
        safe_message = message or "Separating"
        if (
            done_value == last_reported_done
            and total_value == last_reported_total
            and safe_message == last_progress_message
        ):
            return
        last_reported_done = done_value
        last_reported_total = total_value
        last_progress_message = safe_message
        targets = [active_task_id] if active_task_id else [item["taskId"] for item in batch_tasks]
        for task_id in targets:
            if not task_id:
                continue
            emit("task_progress", {
                "stage": "separating",
                "message": safe_message,
                "done": done_value,
                "total": total_value,
            }, task_id=task_id)

    try:
        Path(output_root).mkdir(parents=True, exist_ok=True)
        pre_dirs = {d.name for d in Path(output_root).iterdir() if d.is_dir()}
        for item in batch_tasks:
            task_output = resolve_pymss_output_dir(output_root, [], item["input"], save_as_folder)
            emit("task_started", {"model": payload.get("model"), "input": item["input"], "output": task_output}, task_id=item["taskId"])
            emit("task_stage", {"stage": "validating_input", "message": "Validating input"}, task_id=item["taskId"])
        try:
            from pymss import get_separation_logger  # type: ignore
            logger = get_separation_logger()
            log_handler = JsonLogHandler(root_task_id)
            logger.addHandler(log_handler)
        except Exception:
            logger = None
        separator = _prepare_separator(
            payload={**payload, "output": output_root, "saveAsFolder": save_as_folder},
            task_id=root_task_id,
            progress_callback=emit_batch_progress,
            logger=logger,
        )
        for item in batch_tasks:
            task_id = item["taskId"]
            active_task_id = task_id
            emit("task_stage", {"stage": "separating", "message": "Separating"}, task_id=task_id)
            actual_input = str(Path(item["input"]))
            std_temp: str | None = None
            if batch_want_standardize:
                actual_input, std_temp = _standardize_input_to_wav(item["input"], task_id)
            try:
                success_files = _separate_with_pymss(separator, actual_input, task_id=task_id)
            finally:
                if std_temp and Path(std_temp).exists():
                    try:
                        Path(std_temp).unlink()
                    except Exception:  # noqa: BLE001
                        pass
            if not success_files:
                pymss_detail = ""
                if log_handler is not None and log_handler.messages:
                    # Show the last few pymss log lines so the user sees
                    # the real error (e.g. CUDA OOM) instead of just "no outputs".
                    pymss_detail = " — pymss log: " + " | ".join(log_handler.messages[-3:])
                emit_error("INFERENCE_FAILED",
                           f"Batch separation produced no outputs for {Path(item['input']).name}{pymss_detail}",
                           task_id=task_id)
                continue
            task_output = resolve_pymss_output_dir(output_root, success_files, actual_input, save_as_folder)
            emit("task_stage", {"stage": "writing_output", "message": "Collecting outputs"}, task_id=task_id)
            task_output, success_files = _finalize_pymss_folder_outputs(
                output_root, actual_input, item["input"],
                flat=(output_layout == "flat"),
                model_name=payload.get("model", ""),
                skip_dirs=pre_dirs)
            outputs = collect_outputs(task_output, success_files, output_format)
            emit("task_done", {
                "files": success_files,
                "outputs": outputs,
                "outputDir": str(Path(task_output).resolve()),
                "outputFormat": output_format,
            }, task_id=task_id)
        active_task_id = None
        return 0
    except Exception as exc:
        for item in batch_tasks:
            _emit_inference_error(exc, item["taskId"])
        return 1
    finally:
        if logger is not None and log_handler is not None:
            try:
                logger.removeHandler(log_handler)
            except Exception:
                pass
        _close_separator(separator)


def cmd_infer(payload: dict[str, Any]) -> int:
    if isinstance(payload.get("tasks"), list):
        return cmd_infer_batch(payload)

    task_id = payload.get("taskId") or f"sep_{int(datetime.now().timestamp())}"
    model_name = payload.get("model")
    input_path = str(payload.get("input") or "").strip().strip('"').strip("'")
    output_dir = _normalize_output_dir(payload.get("output"))
    if not model_name:
        return emit_error("MODEL_NOT_FOUND", "Missing model name", task_id=task_id)
    if not input_path:
        return emit_error("INPUT_NOT_FOUND", "Missing input path", task_id=task_id)
    input_path = str(Path(input_path))
    if not Path(input_path).exists():
        return emit_error("INPUT_NOT_FOUND", f"Input path does not exist: {input_path}", task_id=task_id)

    model_dir = payload.get("modelDir") or None
    download = bool(payload.get("download", True))
    source = payload.get("source") or "modelscope"
    endpoint = payload.get("endpoint") or None
    try:
        device, device_ids, resolved_device_label = _resolve_separator_device(
            payload.get("device"), payload.get("deviceIds")
        )
    except Exception as exc:
        return emit_error("DEVICE_CONFIG_INVALID", str(exc), task_id=task_id)
    output_format = payload.get("outputFormat") or "wav"
    output_layout = _normalize_output_layout(payload.get("outputLayout"))
    save_as_folder = output_layout == "folders"
    task_output = resolve_pymss_output_dir(output_dir, [], input_path, save_as_folder)
    selected_stems = _normalize_selected_stems(payload.get("selectedStems"))
    use_tta = bool(payload.get("useTta", False))
    debug = bool(payload.get("debug", False))
    inference_params = normalize_inference_params(
        payload.get("inferenceParams"),
        payload.get("inferenceParamsVersion"),
    )
    want_standardize = bool(_as_bool(inference_params.get("standardize")))
    audio_params = normalize_audio_params(payload.get("audioParams"))

    emit("task_log", {
        "level": "info",
        "message": f"Runtime device: {resolved_device_label} (device_ids={device_ids})",
    }, task_id=task_id)

    last_reported_done: float | None = None
    last_reported_total: float | None = None
    last_progress_message = ""

    def emit_separation_progress(done: Any, total: Any, message: str | None = None) -> None:
        nonlocal last_reported_done, last_reported_total, last_progress_message
        try:
            total_value = float(total)
            done_value = float(done)
        except (TypeError, ValueError):
            return
        safe_message = message or "Separating"
        if (
            done_value == last_reported_done
            and total_value == last_reported_total
            and safe_message == last_progress_message
        ):
            return
        last_reported_done = done_value
        last_reported_total = total_value
        last_progress_message = safe_message
        emit("task_progress", {
            "stage": "separating",
            "message": safe_message,
            "done": done_value,
            "total": total_value,
        }, task_id=task_id)

    separator = None
    logger = None
    log_handler = None
    try:
        emit("task_started", {"model": model_name, "input": input_path, "output": task_output}, task_id=task_id)
        emit("task_stage", {"stage": "validating_input", "message": "Validating input"}, task_id=task_id)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pre_dirs_single = {d.name for d in Path(output_dir).iterdir() if d.is_dir()}

        if download:
            emit("task_stage", {"stage": "downloading_model", "message": "Checking model files"}, task_id=task_id)
        else:
            emit("task_stage", {"stage": "ensuring_model", "message": "Checking model files"}, task_id=task_id)

        from pymss import MSSeparator  # type: ignore
        from pymss.model_registry import resolve_model  # type: ignore
        emit("task_stage", {"stage": "loading_model", "message": "Loading model"}, task_id=task_id)
        try:
            from pymss import get_separation_logger  # type: ignore
            logger = get_separation_logger()
            log_handler = JsonLogHandler(task_id)
            logger.addHandler(log_handler)
        except Exception:
            logger = None

        try:
            resolved = resolve_model(model_name, model_dir=model_dir, require_supported=True, require_exists=True)
        except Exception as resolve_exc:
            if not download:
                return emit_error("MODEL_NOT_FOUND", str(resolve_exc), traceback.format_exc(), task_id=task_id)

            from pymss.model_download import download_model  # type: ignore

            try:
                emit("task_stage", {"stage": "downloading_model", "message": "Downloading model files"}, task_id=task_id)
                download_model(model_name, model_dir=model_dir, source=source, endpoint=endpoint)
                resolved = resolve_model(model_name, model_dir=model_dir, require_supported=True, require_exists=True)
            except Exception as exc:
                return emit_error("MODEL_DOWNLOAD_FAILED", str(exc), traceback.format_exc(), task_id=task_id)

        if not isinstance(resolved, dict):
            raise RuntimeError(f"resolve_model returned unexpected result for {model_name!r}: {type(resolved).__name__}")
        resolved_model_type = resolved.get('model_type')
        resolved_model_path = resolved.get('model_path')
        if not resolved_model_type or not resolved_model_path:
            missing = [key for key in ('model_type', 'model_path') if not resolved.get(key)]
            raise RuntimeError(f"resolve_model result for {model_name!r} is missing required field(s): {', '.join(missing)}")
        runtime_inference_params = _enrich_inference_params_for_model(
            model_type=resolved_model_type,
            config_path=resolved.get('config_path'),
            inference_params=inference_params,
        )

        sep_kwargs: dict[str, Any] = dict(
            model_type=resolved_model_type,
            model_path=resolved_model_path,
            config_path=resolved.get('config_path'),
            device=device,
            device_ids=device_ids,
            output_format=output_format,
            use_tta=use_tta,
            store_dirs=_store_dirs_for_selected_stems(output_dir, selected_stems),
            audio_params=audio_params,
            logger=logger,
            debug=debug,
            progress_callback=emit_separation_progress,
            inference_params=runtime_inference_params,
            save_as_folder=save_as_folder,
        )
        separator = _make_separator(**sep_kwargs)
        emit("task_stage", {"stage": "separating", "message": "Separating"}, task_id=task_id)
        actual_input = str(Path(input_path))
        std_temp: str | None = None
        if want_standardize:
            actual_input, std_temp = _standardize_input_to_wav(input_path, task_id)
        try:
            success_files = _separate_with_pymss(separator, actual_input)
        finally:
            if std_temp and Path(std_temp).exists():
                try:
                    Path(std_temp).unlink()
                except Exception:  # noqa: BLE001
                    pass
        emit("task_stage", {"stage": "writing_output", "message": "Collecting outputs"}, task_id=task_id)
        task_output = resolve_pymss_output_dir(output_dir, success_files, actual_input, save_as_folder)
        task_output, success_files = _finalize_pymss_folder_outputs(
            output_dir, actual_input, input_path,
            flat=(output_layout == "flat"),
            model_name=payload.get("model", ""),
            skip_dirs=pre_dirs_single)
        outputs = collect_outputs(task_output, success_files, output_format)
        emit("task_done", {"files": success_files, "outputs": outputs, "outputDir": str(Path(task_output).resolve()), "outputFormat": output_format}, task_id=task_id)
        return 0
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "no audio stream found" in lowered:
            return emit_error(
                "INPUT_AUDIO_STREAM_MISSING",
                message,
                traceback.format_exc(),
                task_id=task_id,
            )
        if "invalid data found" in lowered or "could not open input" in lowered:
            return emit_error(
                "INPUT_MEDIA_UNSUPPORTED",
                message,
                traceback.format_exc(),
                task_id=task_id,
            )
        return emit_error("INFERENCE_FAILED", message, traceback.format_exc(), task_id=task_id)
    finally:
        if logger is not None and log_handler is not None:
            try:
                logger.removeHandler(log_handler)
            except Exception:
                pass
        if separator is not None:
            close = getattr(separator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            else:
                try:
                    separator.del_cache()
                except Exception:
                    pass
