"""Enlarging a grid cell: the gesture, the cue, and what gets rendered.

**The class this guards.** A thumbnail grid is the only view of a frame on three surfaces, and a
220px cell on a 1920px recording draws a 200px-wide mouse at 23px. Every one of those grids
already computed the overlay at full resolution and dropped it in the same statement, so the
picture was never the missing part — the way to ask for it was.

Two properties are worth more than "a dialog opens":

- **The cell keeps a recipe, not a picture.** Holding the rendered frame would undo why these
  grids hold run-length-encoded masks in the first place, and would pin the enlarged frame to the
  threshold in force when the cell was last drawn. So the render is called on the *click*, and
  what it draws is read then.
- **The cue is derived from the wiring.** A tooltip saying "double-click to enlarge" on a cell
  with no render is the recorded `test_spatial_map` class: a capability documented, advertised
  and unreachable.
"""

import numpy as np

from detail.gui.full_frame import (
    EnlargeableThumb,
    FullFrameDialog,
    open_full_frame,
)
from detail.gui.image_view import ImageView
from detail.gui.qt import QEvent, QImage, QMouseEvent, QPointF, Qt
from tests import waits


def _frame(value=90, width=160, height=90):
    return np.full((height, width, 3), value, dtype=np.uint8)


def _double_click(widget):
    """A real double-click, through the event system — not a call to the handler.

    Calling `mouseDoubleClickEvent` directly proves the method works and not that Qt ever
    reaches it: a widget that never receives the event (no mouse tracking, a child eating it)
    passes that version forever.
    """
    centre = QPointF(widget.width() / 2, widget.height() / 2)
    widget.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
    for kind in (
        QEvent.Type.MouseButtonPress,
        QEvent.Type.MouseButtonRelease,
        QEvent.Type.MouseButtonDblClick,
    ):
        widget.event(
            QMouseEvent(
                kind,
                centre,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )


def _shown(widget, qapp, size=(240, 160)):
    widget.resize(*size)
    widget.show()
    qapp.processEvents()
    return widget


# --------------------------------------------------------------------------- #
# the dialog
# --------------------------------------------------------------------------- #
def test_the_dialog_renders_its_frame_zoomably(qapp):
    dialog = FullFrameDialog(lambda: _frame(), title="frame 3")
    dialog.show()
    waits.until(qapp, lambda: dialog.view().pixmap_size() is not None, "the render")
    size = dialog.view().pixmap_size()
    assert (size.width(), size.height()) == (160, 90), "the view holds a scaled copy"
    assert isinstance(dialog.view(), ImageView) and dialog.view().zoomable
    dialog.close()


def test_a_render_that_raises_says_so_instead_of_staying_blank(qapp):
    """A window that opens and never fills is indistinguishable from one that is hung."""

    def boom():
        raise RuntimeError("no mask for this frame")

    dialog = FullFrameDialog(boom, title="frame 3")
    dialog.show()
    waits.until(
        qapp, lambda: "could not render" in dialog.view().text(), "the failure message"
    )
    assert "no mask for this frame" in dialog.view().text()
    assert dialog.view().pixmap_size() is None
    dialog.close()


def test_a_qimage_render_is_accepted(qapp):
    """A cell backed by a file on disk hands back a QImage — a QPixmap cannot be built off
    the UI thread, which is where every render runs."""
    dialog = FullFrameDialog(
        lambda: QImage(32, 24, QImage.Format.Format_RGB888), title="preview"
    )
    dialog.show()
    waits.until(qapp, lambda: dialog.view().pixmap_size() is not None, "the render")
    dialog.close()


# --------------------------------------------------------------------------- #
# the thumbnail
# --------------------------------------------------------------------------- #
def test_a_double_click_opens_the_frame(qapp):
    thumb = _shown(EnlargeableThumb("cell"), qapp)
    rendered = []
    thumb.set_render(lambda: rendered.append(1) or _frame(), title="frame 7")
    assert not rendered, "the render ran before anybody asked for it"

    _double_click(thumb)
    qapp.processEvents()
    dialogs = thumb.findChildren(FullFrameDialog)
    assert dialogs, "a double-click on the cell opened no window"
    assert dialogs[0].windowTitle() == "frame 7"
    waits.until(qapp, lambda: bool(rendered), "the render")
    dialogs[0].close()


def test_a_cell_with_no_render_opens_nothing_and_advertises_nothing(qapp):
    """The cue is computed from the wiring, so a cell cannot claim what it has not connected."""
    thumb = _shown(EnlargeableThumb("cell"), qapp)
    assert thumb.toolTip() == ""
    _double_click(thumb)
    qapp.processEvents()
    assert not thumb.findChildren(FullFrameDialog)

    thumb.set_render(lambda: _frame(), title="frame 7")
    assert "full size" in thumb.toolTip()
    thumb.set_render(None)
    assert thumb.toolTip() == ""


def test_the_render_runs_at_click_time_not_at_fill_time(qapp):
    """What the cell was showing is not what it holds: the recipe is re-run on the click.

    This is what lets the enlarged frame follow the confidence slider. A cell that had cached
    its picture would open the frame as it was drawn, at whatever threshold that was.
    """
    threshold = {"value": 0.1}
    thumb = _shown(EnlargeableThumb("cell"), qapp)
    seen = []
    thumb.set_render(
        lambda: seen.append(threshold["value"]) or _frame(), title="frame 7"
    )

    threshold["value"] = 0.8  # the slider moved after the cell was filled
    _double_click(thumb)
    waits.until(qapp, lambda: bool(seen), "the render")
    assert seen == [0.8], "the cell opened the frame at a stale threshold"
    thumb.findChildren(FullFrameDialog)[0].close()


def test_open_full_frame_parents_the_window_so_it_outlives_the_call(qapp):
    """`open()` returns immediately, so nothing but Qt ownership keeps the window up: an
    unparented dialog is collected the moment the caller returns, and it flashes and vanishes.

    The parent is also what it is modal *to* — the grid it was opened from."""
    host = _shown(EnlargeableThumb("cell"), qapp)
    dialog = open_full_frame(host, lambda: _frame(), title="frame 1")
    qapp.processEvents()
    assert dialog.parent() is host
    assert dialog.isVisible()
    # Modal, and asserted here rather than left to the source scan: the frame is rendered once,
    # at the threshold in force when it opened, so the grid must not move behind it.
    assert dialog.isModal()
    dialog.close()


def test_every_grid_that_enlarges_is_named_by_some_test():
    """Discovered, not listed — `tests/gui/test_image_view.py` is the template.

    Three grids enlarge today. The point of discovering them is that the fourth is covered the
    day it is added, rather than the day somebody remembers this file.
    """
    import pathlib

    gui = pathlib.Path(__file__).resolve().parents[2] / "detail" / "gui"
    tests = pathlib.Path(__file__).resolve().parent
    hosts = {
        py.stem
        for py in gui.rglob("*.py")
        if "EnlargeableThumb(" in py.read_text(encoding="utf-8-sig")
        and py.stem != "full_frame"
    }
    covered = " ".join(
        py.read_text(encoding="utf-8-sig") for py in tests.rglob("test_*.py")
    )
    assert len(hosts) >= 3, f"the sweep stopped finding grids: {hosts}"
    missing = sorted(stem for stem in hosts if stem not in covered)
    assert not missing, (
        f"these open a frame full size and no test names them: {missing}. A capability added "
        f"here reaches the surface being worked on and no other unless something checks."
    )
