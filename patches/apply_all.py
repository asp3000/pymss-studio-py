"""Monkey-patch TIGER-speech model support into pymms / pymss_core.

Called once at worker startup (from ``worker_protocol.py``).  After this module
runs, the installed pip packages ``pymss`` and ``pymss_core`` behave as if they
natively supported ``model_type="tiger"``.

Two strategies are used:
* **Code patches** (1–4) — monkey-patch specific functions in-memory
* **Catalog merge** (5) — ``patches/patches_catalog.json`` is merged into the
  pip package's ``model_catalog.json`` on disk, so both ``pymss.model_registry``
  and the project's own ``worker_models`` pick up the TIGER entry automatically.
  The file merge runs *before* any ``load_model_catalog`` call, so LRU caches
  always see the augmented catalog on first use.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_PATCHES_DIR = Path(__file__).parent

# ── sentinel ──────────────────────────────────────────────────────────────
_PATCHES_APPLIED = False


def apply_all() -> None:
    """Apply all TIGER-speech patches.  Idempotent — safe to call multiple times."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    _PATCHES_APPLIED = True

    # ---- 1. Inject TIGER module into pymss_core.modules.look2hear ----
    _inject_tiger_module()

    # ---- 2. Patch pymss_core.utils.get_model_from_config ----
    _patch_get_model_from_config()

    # ---- 3. Patch pymss_core.checkpoint.load_checkpoint ----
    _patch_load_checkpoint()

    # ---- 4. Patch pymms.separator._load_state_dict ----
    _patch_load_state_dict()

    # ---- 5. Merge patches_catalog.json into site-packages catalog ----
    _merge_catalog_file()


# ==========================================================================
# Internal helpers — each patches one function
# ==========================================================================

_TIGER_MODULE: types.ModuleType | None = None


def _get_tiger_module():
    global _TIGER_MODULE
    if _TIGER_MODULE is not None:
        return _TIGER_MODULE
    import importlib.util

    tiger_path = _PATCHES_DIR / "tiger.py"
    spec = importlib.util.spec_from_file_location(
        "pymss_core.modules.look2hear.tiger", str(tiger_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pymss_core.modules.look2hear.tiger"] = mod
    spec.loader.exec_module(mod)
    _TIGER_MODULE = mod
    return mod


def _inject_tiger_module() -> None:
    """Load ``patches/tiger.py`` and register it as ``pymss_core.modules.look2hear.tiger``."""
    import pymss_core.modules.look2hear as l2h

    mod = _get_tiger_module()

    # Make TIGER accessible as l2h.TIGER
    l2h.TIGER = mod.TIGER
    # Update __all__ so star-imports work
    current_all = list(getattr(l2h, "__all__", ()) or ())
    if "TIGER" not in current_all:
        current_all.append("TIGER")
        l2h.__all__ = tuple(current_all)


def _patch_get_model_from_config() -> None:
    """Add ``model_type == "tiger"`` branch to ``get_model_from_config``."""
    import pymss_core.utils as utils

    _original = utils.get_model_from_config

    def _patched(model_type, config_path, model_kwargs_override=None):
        if model_type == "tiger":
            from pymss_core.config import load_config  # same relative import as original
            from pymss_core.modules.look2hear import TIGER

            config = load_config(config_path)
            return TIGER(**config.model), config
        return _original(model_type, config_path, model_kwargs_override=model_kwargs_override)

    utils.get_model_from_config = _patched


def _patch_load_checkpoint() -> None:
    """Add ``model_type == "tiger"`` → ``weights_only=False`` to ``load_checkpoint``."""
    import pymss_core.checkpoint as checkpoint

    _original = checkpoint.load_checkpoint

    def _patched(path, *, model_type=None, map_location="cpu", weights_only=None, mmap=True):
        mt = (model_type or "").lower()
        if mt == "tiger":
            weights_only = False if weights_only is None else weights_only
        return _original(path, model_type=model_type, map_location=map_location,
                         weights_only=weights_only, mmap=mmap)

    checkpoint.load_checkpoint = _patched


def _patch_load_state_dict() -> None:
    """Add ``model_type == "tiger"`` branch to ``separator._load_state_dict``.

    The tiger path is identical to the apollo one: it redirects to the
    ``.pymss_state_dict.pt`` file via ``_apollo_state_dict_path(...)``.
    """
    import torch
    import pymss.separator as separator

    _original = separator._load_state_dict
    _apollo_path = separator._apollo_state_dict_path
    _unwrap = separator._unwrap_state_dict

    def _patched(model_type, model_path, device):
        if model_type == "tiger":
            resolved = _apollo_path(model_path)
            return _unwrap(torch.load(resolved, map_location="cpu", weights_only=False))
        return _original(model_type, model_path, device)

    separator._load_state_dict = _patched


def _merge_catalog_file() -> None:
    """Merge ``patches/patches_catalog.json`` into the pip-installed ``model_catalog.json``.

    For each entry in the patches catalog:
    * If an entry with the same ``name`` already exists in the installed catalog,
      the patches entry **overrides** it (so local customizations take precedence).
    * Otherwise the entry is appended.

    The merge runs *before* any ``load_model_catalog()`` call in the worker process,
    so LRU caches always see the augmented catalog on first use.

    If the site-packages file is read-only or the patches file is missing,
    the operation fails silently — the app continues without TIGER in catalog
    (the code patches 1–4 still work, but models must be loaded programmatically).
    """
    try:
        import json
        from importlib import resources as impresources

        patches_path = _PATCHES_DIR / "patches_catalog.json"
        if not patches_path.is_file():
            return  # no patches catalog — nothing to merge

        patches_data = json.loads(patches_path.read_text(encoding="utf-8"))
        patches_entries: list[dict] = patches_data.get("models", [])
        if not patches_entries:
            return

        # Locate the installed catalog via importlib.resources
        try:
            cat_file = impresources.files("pymss.resources").joinpath("model_catalog.json")
        except (ImportError, AttributeError, TypeError):
            # Fallback for older Python / importlib versions
            import pymss.resources as res
            res_dir = Path(res.__file__).parent
            cat_file = res_dir / "model_catalog.json"

        if not cat_file.is_file():
            return

        # Read installed catalog
        cat_data = json.loads(cat_file.read_text(encoding="utf-8"))
        models: list[dict] = cat_data.get("models", [])

        # Build a set of existing model names for quick lookup
        existing_names: set[str] = {m.get("name", "") for m in models}

        # Merge: override existing entries, append new ones
        changed = False
        for patch_entry in patches_entries:
            pname = patch_entry.get("name", "")
            if pname in existing_names:
                # Override — find and replace in-place
                for i, m in enumerate(models):
                    if m.get("name") == pname:
                        models[i] = patch_entry
                        changed = True
                        break
            else:
                # Append new entry
                models.append(patch_entry)
                existing_names.add(pname)
                changed = True

        if not changed:
            return

        # Write merged catalog back to site-packages
        cat_data["models"] = models
        cat_file.write_text(
            json.dumps(cat_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass  # fail silently — don't crash the worker
