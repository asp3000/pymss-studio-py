"""Pymss-Studio 引擎注册表与架构映射"""
import os, gc, torch
import numpy as np

from .msst import MsstEngine

REGISTRY = {
    "msst": MsstEngine,
}

# 架构 → 可用引擎列表（第一个为默认）
ARCHITECTURE_ENGINES = {
    "vr":                ["pymss"],
    "tiger":             ["pymss"],
    "bs_roformer_hyperace": ["pymss"],
    "demucs":            ["pymss"],
    "tasnet":            ["pymss"],
    "legacy_demucs":     ["pymss"],
    "legacy_tasnet":     ["pymss"],
    "segm_models":       ["msst"],
    "swin_upernet":      ["msst"],
    "scnet_unofficial":  ["msst"],
    "torchseg":          ["msst"],
    "bs_mamba2":         ["msst"],
    "mel_band_roformer": ["pymss", "msst"],
    "bs_roformer":       ["pymss", "msst"],
    "htdemucs":          ["pymss", "msst"],
    "mdx23c":            ["pymss", "msst"],
    "bandit":            ["pymss", "msst"],
    "bandit_v2":         ["pymss", "msst"],
    "scnet":             ["pymss", "msst"],
    "apollo":            ["pymss", "msst"],
}

ENGINE_LABELS = {
    "pymss": "Pymss",
    "msst": "MSST",
}

DEFAULT_ENGINE = "pymss"


def get_engine(name: str):
    eng = REGISTRY.get(name)
    if eng is None:
        raise ValueError(f"Unknown engine: {name}, available: {list(REGISTRY.keys())}")
    return eng


def engines_for_architecture(arch: str) -> list[str]:
    return ARCHITECTURE_ENGINES.get(arch, [DEFAULT_ENGINE])


def default_engine_for(arch: str) -> str:
    engines = engines_for_architecture(arch)
    return engines[0] if engines else DEFAULT_ENGINE


def engine_label(name: str) -> str:
    return ENGINE_LABELS.get(name, name)


class _MsstProgressProxy(dict):
    """将 MSST 的 ``callback["progress"] = x`` 转换为 Pymss worker 的
    ``callback(done, total, message)`` 格式。"""
    def __init__(self, total_samples: int, worker_callback):
        super().__init__()
        self._total = total_samples
        self._worker_callback = worker_callback
        self["progress"] = 0.0

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "progress" and self._total > 0:
            done = int(value / 0.99 * self._total) if value < 0.99 else self._total
            self._worker_callback(done, self._total, "Separating")


class MsstSeparatorAdapter:
    """适配 MsstEngine 到 MSSeparator 接口的包装器。

    暴露与 ``MSSeparator`` 相同的 ``process_folder()`` 接口，
    但内部使用 MSST 引擎的模型加载和推理代码。
    """

    def __init__(self, **kwargs):
        self.model_type = kwargs.get("model_type", "")
        self.model_path = kwargs.get("model_path", "")
        self.config_path = kwargs.get("config_path", "")
        self.device = kwargs.get("device", "cuda")
        self.device_ids = kwargs.get("device_ids", [0])
        self.output_format = kwargs.get("output_format", "wav")
        self.store_dirs = kwargs.get("store_dirs", "results")
        self.audio_params = kwargs.get("audio_params", {})
        self.logger = kwargs.get("logger")
        self.debug = kwargs.get("debug", False)
        self.progress_callback = kwargs.get("progress_callback")
        self.inference_params = kwargs.get("inference_params", {})
        self.save_as_folder = kwargs.get("save_as_folder", False)

        # 加载 MSST 引擎
        self._engine = MsstEngine()
        self._model = None
        self._config = None

    def _load_if_needed(self):
        if self._model is None:
            self._model, self._config = self._engine.load_model(
                model_type=self.model_type,
                model_path=self.model_path,
                config_path=self.config_path,
                device=self.device,
                device_ids=self.device_ids,
                inference_params=self.inference_params,
            )

    def process_folder(self, input_folder: str) -> list[str]:
        """处理输入音频文件/文件夹（与 MSSeparator.process_folder 接口一致）。"""
        import soundfile as sf
        from pathlib import Path

        self._load_if_needed()
        sample_rate = self._config.audio.get("sample_rate", 44100)

        # 收集输入文件
        input_path = Path(input_folder)
        if input_path.is_file():
            files = [input_path]
        elif input_path.is_dir():
            files = [f for f in input_path.iterdir() if f.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a", ".ogg")]
        else:
            raise ValueError(f"Input path '{input_folder}' does not exist.")

        # 解析输出目录
        if isinstance(self.store_dirs, str):
            store_root = Path(self.store_dirs)
        elif isinstance(self.store_dirs, dict):
            # 取第一个非空的目录
            store_root = Path(next((v for v in self.store_dirs.values() if v), "results"))
        else:
            store_root = Path("results")

        success_files = []
        for fpath in files:
            try:
                mix, sr = sf.read(str(fpath), always_2d=True)
                mix = mix.T.astype(np.float32)  # (2, L)

                # 分离
                total_samples = mix.shape[1]
                cb = _MsstProgressProxy(total_samples, self.progress_callback) if self.progress_callback else None
                results = self._engine.separate(
                    self._config, self._model, mix, self.device,
                    model_type=self.model_type,
                    callback=cb,
                )

                # 保存每个音轨
                file_stem = fpath.stem
                for stem_name, audio in results.items():
                    out_dir = store_root / file_stem if self.save_as_folder else store_root
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{file_stem}_{stem_name}.{self.output_format}"
                    sf.write(str(out_path), audio.T, sample_rate)
                    if self.logger:
                        self.logger.debug(f"Saved {stem_name} to {out_path}")

                success_files.append(fpath.name)
                if self.logger:
                    self.logger.info(f"Separated {fpath.name} -> {store_root}")

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Cannot separate track: {fpath}, error: {e}")
                continue

        return success_files

    def close(self):
        if self._engine:
            self._engine.close()
