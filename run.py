"""Launcher for the Pymss Studio Python GUI.

The project directory is named ``pymss-studio-py`` (with a hyphen), which is
not a valid Python package identifier, so it cannot be started with
``python -m pymss-studio-py``. This shim registers the directory as the
package ``pymss_gui`` (via importlib) so the regular relative imports used
throughout the code (``from .main_window import ...``,
``from ..workflow_graph import ...``) resolve correctly.

Run directly:  python run.py
"""
from __future__ import annotations

import importlib.util
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))


def _load_package() -> object:
    spec = importlib.util.spec_from_file_location(
        "pymss_gui",
        os.path.join(HERE, "__init__.py"),
        submodule_search_locations=[HERE],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["pymss_gui"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    pkg = _load_package()
    import pymss_gui.main  # noqa: F401  (ensures the main submodule is loaded)
    return pymss_gui.main.main()


if __name__ == "__main__":
    raise SystemExit(main())
