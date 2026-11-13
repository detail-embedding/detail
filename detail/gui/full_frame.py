"""A thumbnail that opens the frame behind it, full size and zoomable.

A grid cell is a compromise: the prompt-test grid draws each frame in a **220px** cell, so on a
1920px recording a mouse 200px wide in the video is **23px** on screen. That is the mask someone
is being asked to judge a prompt on. Every one of these grids already computes the overlay at
full resolution and discards it in the same statement — ``_to_pixmap(rgb).scaled(220, …)`` — so
what is missing is not the picture but a way to ask for it.

**The cell keeps the recipe, not the picture.** :class:`EnlargeableThumb` holds a *callable*
that re-renders the frame, and calls it on the double-click that asks. Holding the rendered
frame instead would undo the reason these grids hold their inputs at all: ``_CompactFrame`` keeps
masks run-length encoded because a full-resolution mask is ~2 MB and a grid is frames × prompts
of them. Re-rendering also means the enlarged frame is drawn at the **current** confidence
threshold rather than whatever it was when the cell was last filled.

The render runs off the UI thread (``busy.run_busy``), because it is a mask decode plus a draw.
Note that ``tests/gui/test_ui_thread_blocking.py`` would *not* have caught an inline version:
its seams are the pixel-source calls, and an RLE decode names none of them. The rule binds
anyway; that scan is a lower bound.

**Why a module of its own.** The dialog needs :class:`~detail.gui.dialog.Dialog` and
``busy.run_busy``; ``image_view`` is deliberately a leaf that imports nothing from ``detail.gui``
but ``scroll``, and it is worth keeping that true. There is still one image view — this hosts
``ImageView`` rather than being a second one.

A render returns an ndarray or a ``QImage``, never a ``QPixmap``: it runs on a worker thread, and
a pixmap may only be built on the UI thread.
"""

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QDialogButtonBox, QLabel

from detail.gui.busy import run_busy
from detail.gui.dialog import Dialog
from detail.gui.image_view import ImageView

#: What a cell hands back when asked for its frame. Off-thread, so no ``QPixmap``.
Render = Callable[[], np.ndarray | QImage]


def to_pixmap(source: np.ndarray | QImage) -> QPixmap:
    """A render's result as a pixmap — UI thread only."""
    if isinstance(source, QImage):
        return QPixmap.fromImage(source)
    rgb = np.ascontiguousarray(source)
    height, width, _ = rgb.shape
    return QPixmap.fromImage(
        QImage(rgb.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888)
    )


class FullFrameDialog(Dialog):
    """One frame, as big as the window, zoomable and pannable.

    The frame arrives off the UI thread, so the view says it is working rather than opening
    blank — a dialog that shows nothing for a second reads as a dialog that is broken.
    """

    def __init__(self, render: Render, *, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = self.body()
        self._view = ImageView(zoomable=True)
        self._view.setMinimumSize(480, 360)
        self._view.clear_frame("Rendering…")
        layout.addWidget(self._view, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)
        run_busy(
            self._view,
            render,
            self._show,
            label="Rendering…",
            on_error=lambda exc: self._view.clear_frame(f"(could not render: {exc})"),
        )

    def _show(self, source) -> None:
        if source is None:
            self._view.clear_frame("(nothing to show)")
            return
        self._view.set_frame(to_pixmap(source))

    def view(self) -> ImageView:
        return self._view


def open_full_frame(parent, render: Render, *, title: str) -> FullFrameDialog:
    """Open ``render``'s frame in its own window, modal to the surface it came from.

    ``open()`` rather than ``exec()``: modal either way, but without a nested event loop, so the
    render's own worker delivers into the ordinary one and a caller is not blocked mid-slot.
    Modal rather than modeless because the frame is rendered **once**, at the threshold in force
    when it was opened — a window left up while the slider moves behind it would go on showing a
    frame the grid no longer shows, which is the "a surface that cannot show a frame must stop
    showing the last one it had" class in its quieter form.
    """
    dialog = FullFrameDialog(render, title=title, parent=parent)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.open()
    return dialog


class EnlargeableThumb(QLabel):
    """A grid cell whose double-click opens the frame behind it.

    ``set_render`` is what makes the cell enlargeable, and the **cue is derived from it**: a cell
    with no render offers no tooltip and no pointing hand. A cue set beside the wiring is a claim
    that can be true of a cell that does nothing, which is the recorded `test_spatial_map` class.
    """

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._render: Render | None = None
        self._title = ""

    def set_render(self, render: Render | None, *, title: str = "") -> None:
        self._render = render
        self._title = title
        if render is None:
            self.setToolTip("")
            self.unsetCursor()
            return
        self.setToolTip("Double-click to see this frame full size")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """Out of a 220px cell and into a window.

        A single click means nothing on these grids, so unlike the clip strips there is no
        gesture to compete with: the cell that is too small to judge is the one you open.
        """
        if self._render is not None:
            open_full_frame(self, self._render, title=self._title)
        super().mouseDoubleClickEvent(event)
