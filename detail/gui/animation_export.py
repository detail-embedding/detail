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
    QMessageBox,
    QWidget,
)

from detail.api.animation import (
    GIF,
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


_PATTERNS = {MP4: "MP4 video (*.mp4)", GIF: "GIF animation (*.gif)"}


def ask_save_path(
    parent: QWidget, title: str, suggested: str, kind: str, pattern: str = ""
) -> Path | None:
    """Where to write a file of ``kind`` (``mp4``, ``gif``, ``csv``, ``png``…), from the
    system save dialog; ``None`` when cancelled. ``pattern`` is the dialog's filter, read
    from :data:`_PATTERNS` for a video or a GIF."""
    pattern = pattern or _PATTERNS[kind]
    path, _filter = QFileDialog.getSaveFileName(parent, title, suggested, pattern)
    if not path:
        return None
    return with_suffix_confirmed(parent, title, Path(path), kind)


def with_suffix_confirmed(
    parent: QWidget, title: str, path: Path, kind: str
) -> Path | None:
    """``path`` with ``.kind`` added when the name typed has none — and, when the name that
    makes exists, the replace question the dialog never got to ask; ``None`` on No.

    The save dialog asks before replacing the name it returns, and on a platform whose dialog
    does not add the suffix itself (Qt's own, on Linux) that is ``climb``, not ``climb.mp4``:
    adding it afterwards wrote over ``climb.mp4`` with nothing asked (#29).
    """
    if path.suffix.lower() == f".{kind}":
        return path
    path = path.with_name(f"{path.name}.{kind}")
    if path.exists() and not confirm_replace(parent, title, [path.name]):
        return None
    return path


def confirm_replace(parent: QWidget, title: str, names: list[str]) -> bool:
    """Ask before an export writes over files already there; True to go ahead.

    Says how many and names the first few, because "replace 40 files?" with no names gives
    nothing to check the answer against.
    """
    shown = ", ".join(names[:3]) + (
        f" and {len(names) - 3:,} more" if len(names) > 3 else ""
    )
    what = "A file" if len(names) == 1 else f"{len(names):,} files"
    return (
        QMessageBox.question(
            parent,
            title,
            f"{what} with this name already exist{'s' if len(names) == 1 else ''} "
            f"there ({shown}). Replace {'it' if len(names) == 1 else 'them'}?",
        )
        == QMessageBox.StandardButton.Yes
    )


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
