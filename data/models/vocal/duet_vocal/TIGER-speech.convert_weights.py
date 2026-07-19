"""Convert the HF TIGER-speech safetensors weights into pymss's
``.pymss_state_dict.pt`` format.

Run with the pymss-studio-py venv Python:
    venv\\Scripts\\python.exe convert_tiger_weights.py

This also validates that ``pymss_core.modules.look2hear.tiger.TIGER`` has
exactly the same ``state_dict`` keys as the published weights.
"""

import json
import torch

from safetensors.torch import load_file

from pymss_core.modules.look2hear.tiger import TIGER

MODEL_DIR = "C:/Users/Administrator/.cache/pymss/models/speech_separation/tiger"
SAFETENSORS = f"{MODEL_DIR}/TIGER-speech.safetensors"
OUT_PT = f"{MODEL_DIR}/TIGER-speech.pymss_state_dict.pt"
OUT_JSON = f"{MODEL_DIR}/TIGER-speech.pymss_state_dict.json"

# Architecture must match the published "TIGER-speech" (large) weights exactly.
MODEL_KWARGS = dict(
    out_channels=128,
    in_channels=256,
    num_blocks=8,
    upsampling_depth=5,
    att_n_head=4,
    att_hid_chan=4,
    att_kernel_size=8,
    att_stride=1,
    win=640,
    stride=160,
    num_sources=2,
    sample_rate=16000,
)


def main():
    print(f"Loading safetensors: {SAFETENSORS}")
    state_dict = load_file(SAFETENSORS, device="cpu")
    print(f"  loaded {len(state_dict)} tensors")

    print("Building pymss TIGER and validating key alignment...")
    model = TIGER(**MODEL_KWARGS)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Key mismatch! missing={missing} unexpected={unexpected}"
        )
    print("  OK: all keys aligned")

    torch.save(model.state_dict(), OUT_PT)
    print(f"Wrote: {OUT_PT}")

    meta = {
        "model_type": "tiger",
        "num_sources": MODEL_KWARGS["num_sources"],
        "sample_rate": MODEL_KWARGS["sample_rate"],
        "win": MODEL_KWARGS["win"],
        "stride": MODEL_KWARGS["stride"],
        "num_tensors": len(model.state_dict()),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
