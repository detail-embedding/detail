"""A pop-up opens at the size of its content, up to most of its screen, centred.

**The class: a window whose opening size is decided by something other than what it holds.**
Reported from the app — *"most pop-ups open much smaller than their content"* — and it had two
mechanisms, which is why it survived every per-dialog fix:

- **A typed size.** Forty dialogs called ``self.resize(560, 520)`` or so many ``lines()`` by
  so many, picked against one screen and one fixture. On real data the content outgrew them; the
  rule that capped them (1440x810, a typed-size scan) checked only that they were not too *big*.
- **A scroll area's hint.** A `QScrollArea` does not report what it holds, so a dialog left to
  its hint opened at a generic size. ``fits_content`` fixed that for the nine settings dialogs
  that opted in, and for nobody else.

The rule now lives in one place, `Dialog.showEvent` through `sizing.fit_to_content`: measure what
is unseen — the body's own scroll, any table or list in it, any picture short of its natural size
(`ImageView.natural_size`) — grow by it up to `sizing.SCREEN_SHARE` of the screen, centre, and do
it again, growing only, when content arrives later, unless the user has resized the window.

**How it hid.** Nothing is wrong at the line a size is typed: it is right on the screen and the
data it was typed for. And a sweep over the fixture cannot see it either, because the fixture's
tables are empty and its video is 32x24 — every dialog fits. So the guards are two: a source scan
that no dialog types a size (`a_dialog_never_types_its_opening_size`), and the behaviour below,
driven with content made large on purpose.

`a_dialog_opens_showing_its_content` (`gui/requirements.py`) sweeps every buildable dialog.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSize  # noqa: E402
from PyQt6.QtGui import QPixmap  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QTableWidget,
)

from detail.gui.dialog import Dialog  # noqa: E402
from detail.gui.image_view import ImageView  # noqa: E402
from detail.gui.sizing import fit_list_height, fit_screen, window_cap  # noqa: E402

pytestmark = pytest.mark.guard


def _dialog(*rows) -> Dialog:
    dialog = Dialog()
    for row in rows:
        dialog.body().addWidget(row)
    dialog.set_buttons(QDialogButtonBox(QDialogButtonBox.StandardButton.Close))
    return dialog


def _label(lines: int) -> QLabel:
    return QLabel("\n".join(f"row {i}" for i in range(lines)))


def _settle(qapp) -> None:
    for _ in range(4):
        qapp.processEvents()


def _hidden(dialog) -> int:
    return dialog._scroll.verticalScrollBar().maximum()


@pytest.fixture
def shown(qapp):
    opened = []

    def show(dialog):
        dialog.show()
        _settle(qapp)
        opened.append(dialog)
        return dialog

    yield show
    for dialog in opened:
        dialog.close()
        dialog.deleteLater()
    _settle(qapp)


def test_a_dialog_opens_at_its_contents_height(shown):
    dialog = shown(_dialog(_label(20)))
    assert _hidden(dialog) == 0
    assert dialog.height() < window_cap(dialog).height()
    # The content's height, not more: no typed size padding it out.
    assert dialog.height() <= dialog.sizeHint().height() + 2


def test_content_past_the_cap_opens_at_the_cap_and_centred(shown):
    dialog = shown(_dialog(_label(400)))
    cap = window_cap(dialog)
    assert abs(dialog.height() - cap.height()) <= 2
    assert _hidden(dialog) > 0  # the rest scrolls
    free = fit_screen(dialog).availableGeometry()
    centre = dialog.frameGeometry().center()
    assert abs(centre.x() - free.center().x()) <= 2
    assert abs(centre.y() - free.center().y()) <= 2


def test_a_table_opens_with_every_row_in_view(shown):
    table = QTableWidget(12, 3)
    dialog = shown(_dialog(QLabel("A table"), table))
    assert table.verticalScrollBar().maximum() == 0, (
        f"{table.verticalScrollBar().maximum()}px of the table's rows are out of view in a "
        f"{dialog.height()}px window under a {window_cap(dialog).height()}px cap"
    )


def _picture(width: int, height: int) -> QPixmap:
    pixmap = QPixmap(width, height)
    pixmap.fill()
    return pixmap


def test_a_picture_opens_at_its_natural_size(shown):
    view = ImageView(zoomable=False)
    view.set_frame(_picture(400, 300))
    shown(_dialog(view))
    assert view.width() >= 400 and view.height() >= 300, view.size()


def test_any_picture_with_a_natural_size_is_given_it(qapp, shown):
    """Not only `ImageView`: the field-of-view editor's canvas paints its own frame, and is
    found by what it says (``natural_size``), not by its class."""
    import numpy as np

    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(np.zeros((480, 640, 3), np.uint8))
    dialog = Dialog()
    dialog.body().addWidget(canvas, 1)  # as the field-of-view editor lays it out
    dialog.set_buttons(QDialogButtonBox(QDialogButtonBox.StandardButton.Close))
    shown(dialog)
    assert canvas.width() >= 640 and canvas.height() >= 480, canvas.size()


def test_a_frame_that_arrives_after_the_show_grows_the_window(qapp, shown):
    """The clip windows' case: the frame is decoded off the UI thread, after the show."""
    view = ImageView(zoomable=False)
    dialog = shown(_dialog(view))
    before = dialog.size()
    view.set_frame(_picture(520, 360))
    _settle(qapp)
    assert view.width() >= 520 and view.height() >= 360, (before, view.size())


def test_a_window_the_user_resized_is_not_grown(qapp, shown):
    view = ImageView(zoomable=False)
    dialog = shown(_dialog(view))
    # The window system's resize — what a drag delivers — rather than `QWidget.resize`, which is
    # the program's own and is what the fit itself calls.
    dialog.windowHandle().resize(300, 200)
    _settle(qapp)
    assert dialog._user_sized
    view.set_frame(_picture(640, 480))
    _settle(qapp)
    assert dialog.size() == QSize(300, 200)


def test_a_dialog_shown_again_is_sized_fresh(qapp, shown):
    dialog = shown(_dialog(_label(20)))
    fitted = dialog.size()
    dialog.windowHandle().resize(300, 120)
    _settle(qapp)
    dialog.hide()
    dialog.show()
    _settle(qapp)
    assert dialog.size() == fitted


# --- lists in a pop-up: every row, and an edge to drag --------------------------------------- #
# Asked, and answered: should a list in a pop-up grow past its row cap when the screen has room?
# *"Yes, they should and should also be manually scalable by user within larger window."* The
# cap stays a page's (`fit_list_height(max_rows=…)`, `tables.max_visible_rows`); `Dialog`
# lifts it (`SHOWS_EVERY_ROW`, `sizing.refit_rows`) and wraps every list and table in a
# `sizing.RowsGrip` (`attach_row_grips`). Swept by `a_list_in_a_dialog_shows_every_row` and
# `a_list_in_a_dialog_has_an_edge_to_drag` over every dialog, built over 24 recordings.
def _list_of(rows: int) -> QListWidget:
    listing = QListWidget()
    listing.addItems([f"recording {i:02d}" for i in range(rows)])
    return listing


def _rows_hidden(view) -> int:
    from detail.gui.sizing import hidden_extent

    return hidden_extent(view)[1]


def row_grip(view):
    from detail.gui.sizing import row_grip as grip

    return grip(view)


def test_a_capped_list_in_a_pop_up_shows_every_row(shown):
    listing = _list_of(30)
    fit_list_height(
        listing, max_rows=8
    )  # a page's cap, applied before it is in a dialog
    dialog = shown(_dialog(QLabel("Recordings"), listing))
    assert _rows_hidden(listing) == 0, (listing.height(), dialog.height())


def test_the_same_list_on_a_page_keeps_its_cap(qapp):
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

    page = QWidget()
    listing = _list_of(30)
    QVBoxLayout(page).addWidget(listing)
    fit_list_height(listing, max_rows=8)
    page.show()
    _settle(qapp)
    try:
        assert listing.height() < 10 * listing.sizeHintForRow(0)
        assert _rows_hidden(listing) > 0
    finally:
        page.close()


def test_a_capped_table_in_a_pop_up_shows_every_row(shown):
    from detail.gui.tables import fit_table_height

    table = QTableWidget(20, 2)
    fit_table_height(table, max_rows=5)
    shown(_dialog(table))
    assert _rows_hidden(table) == 0


def test_a_list_nobody_sized_shows_every_row(shown):
    """No fit at all, and wrapped in a container that does not expand — the recordings pickers'
    shape — so growing the window would not have reached it."""
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

    listing = _list_of(24)
    holder = QWidget()
    QVBoxLayout(holder).addWidget(listing)
    shown(_dialog(holder))
    assert _rows_hidden(listing) == 0


def test_every_list_gets_an_edge_to_drag(shown):
    from detail.gui.sizing import row_grip

    listing, table = _list_of(5), QTableWidget(3, 2)
    shown(_dialog(listing, table))
    assert row_grip(listing) is not None and row_grip(table) is not None


def test_a_list_in_a_form_row_keeps_answering_to_its_row(qapp, shown):
    """The grip goes beside the view, so a form that hides the view's row by the view —
    the distill form's `setRowVisible(self._runs, …)` — still finds it, and the grip follows.
    """
    from PyQt6.QtWidgets import QFormLayout, QWidget

    from detail.gui.sizing import row_grip

    holder = QWidget()
    form = QFormLayout(holder)
    listing = _list_of(3)
    form.addRow("Runs:", listing)
    dialog = shown(_dialog(holder))
    form.setRowVisible(listing, False)
    _settle(qapp)
    assert not listing.isVisibleTo(dialog) and not row_grip(listing).isVisibleTo(dialog)
    form.setRowVisible(listing, True)
    _settle(qapp)
    assert listing.isVisibleTo(dialog) and row_grip(listing).isVisibleTo(dialog)


def test_dragging_the_edge_resizes_the_list_and_not_the_window(qapp, shown):
    listing = _list_of(6)
    dialog = shown(_dialog(listing))
    window, before = dialog.size(), listing.height()
    grip = row_grip(listing)
    grip.dragged.emit(120)
    _settle(qapp)
    assert listing.height() == before + 120
    assert dialog.size() == window
    assert dialog._scroll.verticalScrollBar().maximum() > 0  # the body scrolls instead
    grip.dragged.emit(-200)
    _settle(qapp)
    assert listing.height() < before


def test_a_dragged_list_is_not_refitted_by_new_rows(qapp, shown):
    listing = _list_of(6)
    shown(_dialog(listing))
    row_grip(listing).dragged.emit(40)
    _settle(qapp)
    dragged = listing.height()
    listing.addItems([f"late {i}" for i in range(10)])
    fit_list_height(listing)
    _settle(qapp)
    assert listing.height() == dragged


def test_a_dragged_height_is_forgotten_when_the_dialog_opens_again(qapp, shown):
    listing = _list_of(6)
    dialog = shown(_dialog(listing))
    fitted = listing.height()
    row_grip(listing).dragged.emit(90)
    _settle(qapp)
    dialog.hide()
    dialog.show()
    _settle(qapp)
    assert listing.height() == fitted


def test_a_hidden_list_hides_its_edge(qapp, shown):
    listing = _list_of(4)
    dialog = shown(_dialog(listing))
    grip = row_grip(listing)
    listing.hide()
    _settle(qapp)
    assert not grip.isVisibleTo(dialog)
    listing.show()
    _settle(qapp)
    assert grip.isVisibleTo(dialog)
