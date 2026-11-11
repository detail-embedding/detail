"""Export video… and Export GIF… for a picture that plays: the enlarged clip-window pane and the
clip montage (§22). The writing is ``api.animation``'s; this is the asking.

**A GIF asks each time** (the user's choice of 2026-10-22): its size and its frame rate decide how
big the file is and how smooth it plays, and the right answer differs between a slide and a chat
message. :class:`GifSettingsDialog` opens on 640 px and 12 fps and says which frames it will
write, so the scope is never a surprise. An mp4 asks only where to save: it keeps every frame
at the size it was drawn.

The work runs through ``busy.run_progress`` — a bar that moves, with Cancel — and the window
says where the file went when it is done.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QWidget,
)

from detail.api.animation import (
    GIF_DEFAULT_FPS,
    GIF_DEFAULT_WIDTH,
    GIF_FPS_RANGE,
    GIF_WIDTH_RANGE,
    MP4,
)
from detail.gui.dialog import Dialog
from detail.gui.spin import FloatSpin, IntSpin


class GifSettingsDialog(Dialog):
    """Ask a GIF's width and frame rate, saying which frames it will hold."""

    def __init__(
        self,
        parent=None,
        *,
        frames_note: str,
        width: int = GIF_DEFAULT_WIDTH,
        fps: float = GIF_DEFAULT_FPS,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export GIF")
        layout = self.body()
        form = QFormLayout()
        self._width = IntSpin()
        self._width.setObjectName("gif_width")
        self._width.setRange(*GIF_WIDTH_RANGE)
        self._width.setSingleStep(32)
        self._width.setSuffix(" px")
        self._width.setValue(int(width))
        self._width.setToolTip(
            "The GIF's width; its height follows. A narrower GIF is a smaller file."
        )
        form.addRow("Width (at most)", self._width)
        self._fps = FloatSpin()
        self._fps.setObjectName("gif_fps")
        self._fps.setRange(*GIF_FPS_RANGE)
        self._fps.setSuffix(" fps")
        self._fps.setValue(float(fps))
        self._fps.setToolTip(
            "Pictures a second. The GIF keeps real time at any rate: a lower one skips frames "
            "rather than playing slower, and makes a smaller file."
        )
        form.addRow("Frame rate", self._fps)
        layout.addLayout(form)
        note = QLabel(frames_note)
        note.setObjectName("gif_frames_note")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Choose file…")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)

    def values(self) -> tuple[int, float]:
        """``(max width in px, frames per second)`` as set."""
        return int(self._width.value()), float(self._fps.value())


def ask_gif_settings(parent: QWidget, frames_note: str) -> tuple[int, float] | None:
    """The GIF dialog, modal; its values, or ``None`` when cancelled."""
    dialog = GifSettingsDialog(parent, frames_note=frames_note)
    if dialog.exec() != Dialog.DialogCode.Accepted:
        return None
    return dialog.values()


def ask_save_path(
    parent: QWidget, title: str, suggested: str, kind: str
) -> Path | None:
    """Where to write an ``mp4`` / ``gif``, from the system save dialog; ``None`` when
    cancelled. The suffix is added when the name typed has none."""
    pattern = "MP4 video (*.mp4)" if kind == MP4 else "GIF animation (*.gif)"
    path, _filter = QFileDialog.getSaveFileName(parent, title, suggested, pattern)
    if not path:
        return None
    path = Path(path)
    if path.suffix.lower() != f".{kind}":
        path = path.with_name(f"{path.name}.{kind}")
    return path


def safe_name(text: str) -> str:
    """``text`` as a file name: letters, digits, ``-`` and ``_``; the rest become ``_``."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in text)
    return "_".join(part for part in cleaned.split("_") if part) or "export"


def run_export(
    target: QWidget,
    work: Callable[[Callable[[str, float, str], None]], Path],
    *,
    on_written: Callable[[Path], None],
    label: str,
):
    """Run ``work(progress)`` off the UI thread under a bar with Cancel; ``on_written(path)``
    when it finishes. A cancelled export leaves no file (``api.animation``)."""
    from detail.gui.busy import run_progress

    return run_progress(
        target,
        work,
        on_written,
        label=label,
        on_cancel=lambda: None,
    )


def written_note(path: Path) -> str:
    """``Wrote masks.gif (2.4 MB) to C:/…`` — what an export says when it is done."""
    try:
        size = path.stat().st_size
    except OSError:
        return f"Wrote {path.name} to {path.parent}"
    return f"Wrote {path.name} ({size / 1e6:.1f} MB) to {path.parent}"
