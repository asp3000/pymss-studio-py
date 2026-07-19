"""A QWidget that accepts file/folder drag-and-drop.

Emits `files_dropped` with a list of local paths. Used by the separate view
so users can drag audio/video files or folders straight onto the window.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel


class DropArea(QFrame):
    files_dropped = Signal(list)

    def __init__(self, text: str = "拖放音频 / 视频文件或文件夹到此处", parent=None) -> None:
        super().__init__(parent)
        self._default_text = text
        self.setFrameShape(QFrame.Shape.Box)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setLineWidth(1)
        self.setAcceptDrops(True)
        self.setMinimumHeight(90)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

    # ---- drag & drop ------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setLineWidth(2)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self.setLineWidth(1)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        urls = event.mimeData().urls()
        paths = [str(Path(u.toLocalFile())) for u in urls if u.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()
        self.setLineWidth(1)

    def set_text(self, text: str) -> None:
        self.label.setText(text)

    def set_files(self, paths: list[str]) -> None:
        if paths:
            self.label.setText(f"{len(paths)} 个文件已就绪\n拖放以替换 / 添加")
        else:
            self.label.setText(self._default_text)
