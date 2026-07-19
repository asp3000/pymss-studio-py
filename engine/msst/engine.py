"""
MSST 引擎 - 封装 D:\\AI\\MSST-WebUI 的模型加载和推理代码。

使用方式：
    engine = MsstEngine()
    model, config = engine.load_model("mel_band_roformer", "model.ckpt", "model.yaml", "cuda")
    result = engine.separate(config, model, mix_numpy, "cuda", "mel_band_roformer")
    engine.close()
"""
import os, sys, gc, torch
import numpy as np
from typing import Dict, Optional
from contextlib import contextmanager

MSST_ROOT = os.path.abspath(r"D:\AI\MSST-WebUI")
_PARAM_MAP = {
    "batch_size": "inference",
    "num_overlap": "inference",
    "chunk_size": "audio",
    "normalize": "inference",
}
_IMPORTED = False  # 确保 MSST 模块只被导入一次


def _ensure_msst_path():
    """将 MSST-WebUI 根目录加入 sys.path 并切换到该目录。
    MSST 的 constant.py 使用 data_backup/webui_config.json 等相对路径，
    因此必须在 MSST 根目录下才能导入。"""
    global _IMPORTED
    if _IMPORTED:
        return
    if MSST_ROOT not in sys.path:
        sys.path.insert(0, MSST_ROOT)
    # MSST 模块导入依赖 cwd = MSST_ROOT
    _cwd = os.getcwd()
    os.chdir(MSST_ROOT)
    try:
        # 触发 MSST utils 的模块级初始化（会读取 data_backup/ 等相对路径）
        from utils.utils import get_model_from_config, demix as msst_demix  # noqa: F401
        _IMPORTED = True
    finally:
        os.chdir(_cwd)


def _import_msst(name):
    """在 MSST_ROOT 下导入指定 MSST 模块，返回模块引用。"""
    _ensure_msst_path()
    _cwd = os.getcwd()
    os.chdir(MSST_ROOT)
    try:
        if name == "get_model_from_config":
            from utils.utils import get_model_from_config
            return get_model_from_config
        if name == "demix":
            from utils.utils import demix as msst_demix
            return msst_demix
        raise ImportError(f"Unknown MSST import: {name}")
    finally:
        os.chdir(_cwd)


def _apply_inference_params(config, params: Optional[dict]):
    if not params:
        return
    for key, section in _PARAM_MAP.items():
        val = params.get(key)
        if val is None:
            continue
        if config[section].get(key) is not None:
            config[section][key] = int(val) if key != "normalize" else bool(val)


def _load_state_dict(model_path: str, device: str, model_type: str):
    if model_type in ("htdemucs", "apollo"):
        sd = torch.load(model_path, map_location=device, weights_only=False)
        if "state" in sd:
            sd = sd["state"]
        if "state_dict" in sd:
            sd = sd["state_dict"]
        return sd
    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(model_path, device=device)
    # MSST 引擎直接使用 weights_only=False，因为 MSST 的 ckpt 使用新格式存储，
    # weights_only=True 会拒绝 torch.storage.UntypedStorage (tagged with auto)
    return torch.load(model_path, map_location=device, weights_only=False)


class MsstEngine:
    """封装 MSST-WebUI 推理链路的引擎。"""

    def __init__(self):
        self._model = None
        self._config = None
        self._loaded = False

    # ── 模型加载 ──────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        """将 "auto" 解析为实际设备（cuda / cpu），torch.load 不支持 "auto"。"""
        d = str(device).strip().lower()
        if d == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return d

    def load_model(self, model_type: str, model_path: str, config_path: str,
                   device: str = "cuda", device_ids: Optional[list] = None,
                   inference_params: Optional[dict] = None):
        """加载模型，返回 (model, config)。"""
        # 解析 "auto" → 实际设备（torch.load 不支持 map_location="auto"）
        device = self._resolve_device(device)
        get_model_from_config = _import_msst("get_model_from_config")
        _cwd = os.getcwd()
        os.chdir(MSST_ROOT)
        try:
            model, config = get_model_from_config(model_type, config_path)
        finally:
            os.chdir(_cwd)

        _apply_inference_params(config, inference_params)

        state_dict = _load_state_dict(model_path, device, model_type)
        model.load_state_dict(state_dict)

        if device_ids and len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)
        model.eval()

        self._model = model
        self._config = config
        self._loaded = True
        return model, config

    # ── 推理 ──────────────────────────────────────────────

    def separate(self, config, model, mix: np.ndarray, device: str,
                 model_type: str, callback=None) -> Dict[str, np.ndarray]:
        """用 MSST 的 ``demix()`` 执行音源分离。"""
        device = self._resolve_device(device)
        msst_demix = _import_msst("demix")
        _cwd = os.getcwd()
        os.chdir(MSST_ROOT)
        try:
            return msst_demix(config, model, mix, device, model_type=model_type, callback=callback)
        finally:
            os.chdir(_cwd)

    # ── 生命周期 ──────────────────────────────────────────

    def close(self):
        if self._model is not None:
            del self._model
        self._model = None
        self._config = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._loaded
