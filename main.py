"""Entry point for the Python GUI replacement of the Pymss Studio Vue UI.

Run with:  python main.py
(requires PySide6 and a working python/worker.py + pymss environment)
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Pymss Studio (Python GUI)")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
