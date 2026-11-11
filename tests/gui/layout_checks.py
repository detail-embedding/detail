"""Measure the symptom of a layout that has nowhere to put its slack.

The class: **a vertical layout with no fill child and no trailing stretch spreads the extra
height across its labels.** Qt has to put leftover space somewhere, and with nothing declaring
that it wants the space, it goes proportionally into every item that will accept it — which is
every ``QLabel``. A heading whose natural height is 17 px is laid out at 324 px and floats three
hundred pixels above the table it names.

The rule is old — ``gui_design`` §3.1's *Pages top-justify*, which names this failure directly:
a page with small content, no fill child and no trailing stretch is the bug. What was missing is
anything that *measures* it. The one existing check was a private ``_ends_with_stretch`` helper
in ``test_overview.py``, applied structurally to two index pages — so it could not see a pane
inside a splitter, which is where this recurred, and it could not see the failure at all, only
the absence of one particular fix for it.

This measures instead: grow the widget past its size hint and look at what the labels did. That
catches the bug however it was caused, says nothing about how it must be fixed, and needs no
exemption for a form inside a dialog that is never given slack in the first place.

The threshold is a ratio, not a pixel budget, because CI is Linux and the dev box is macOS and
font metrics differ. Measured over the eleven pages ``test_smoke_construct.py`` builds: ten
clean, one label at 15 -> 25 px (a row centred beside a taller sibling). The audit heading was
17 -> 324.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QBoxLayout, QGridLayout, QLabel

from detail.gui.sizing import expands_vertically

#: Whether a widget asked for the space, rather than having it forced on it.
#:
#: A `QLabel` with an Expanding vertical policy is a deliberate fill element — `ImageView` is
#: one, a label standing in for the video it will show — and filling is what it is for. The
#: spreading bug is about labels with the default `Preferred` policy absorbing slack nobody
#: offered them.
#:
#: **The production predicate, imported.** This file kept its own copy and said so ("same flag
#: test `pages.add_body_widget` uses"), which makes the guard agree with the code only for as
#: long as nobody edits one of them.
_wants_to_fill = expands_vertically


def _takes_its_row_height(label) -> bool:
    """Whether ``label`` sits in a horizontal row, and so is as tall as whatever shares it.

    "Video(s):" beside a 72px file list renders at 38px against a 15px hint — that is the list,
    not spread. A form row in this package is a `QWidget` with a `QHBoxLayout` holding the label
    beside its field (and `label_with` wraps the label with its `?` button the same way), so a
    horizontal parent layout is exactly the signal.

    The check below is otherwise right to be suspicious of a label at twice its hint: the failure
    it exists for is a page with nowhere to put its slack pushing it into vertically stacked text.
    """
    parent = label.parentWidget()
    layout = parent.layout() if parent is not None else None
    if isinstance(layout, QGridLayout):
        return _row_is_held_open(layout, label)
    direction = getattr(layout, "direction", None)
    return direction is not None and direction() in (
        QBoxLayout.Direction.LeftToRight,
        QBoxLayout.Direction.RightToLeft,
    )


def _row_is_held_open(grid, label) -> bool:
    """A grid row is as tall as its tallest cell, so a row label beside a 132px thumbnail is
    132px by construction. Excused only when a *sibling in its row* holds it that tall —
    a row of labels alone that grew is still spread, which is what a missing stretch row does.
    """
    index = grid.indexOf(label)
    if index < 0:
        return False
    row = grid.getItemPosition(index)[0]
    for i in range(grid.count()):
        other = grid.itemAt(i).widget()
        if other is None or other is label or grid.getItemPosition(i)[0] != row:
            continue
        # A sibling holds the row open if it *asks* for that height, or is a fill child that
        # took it. Not merely "is as tall": a header row of labels alone all come out equally
        # tall once the slack spreads, and would excuse each other.
        asks = (
            max(other.minimumHeight(), other.sizeHint().height()) >= label.height() - 2
        )
        fills = _wants_to_fill(other) and other.height() >= label.height() - 2
        if other.isVisible() and (asks or fills):
            return True
    return False


def spread_labels(widget, *, size=(1200, 1000)) -> list[tuple[str, int, int]]:
    """Grow ``widget`` past its hint and return ``(text, hint, height)`` for labels that grew.

    A word-wrapping label legitimately grows: its height *is* a function of its width.
    """
    widget.resize(*size)
    widget.show()
    QApplication.processEvents()
    grown = [
        (label.text()[:40], label.sizeHint().height(), label.height())
        for label in widget.findChildren(QLabel)
        if label.isVisible()
        and not label.wordWrap()
        and not _wants_to_fill(label)
        and not _takes_its_row_height(label)
        and label.height() > 2 * label.sizeHint().height()
    ]
    widget.hide()
    return grown


def assert_top_justified(widget, *, size=(1200, 1000)) -> None:
    """Fail if any label in ``widget`` absorbed the slack instead of a stretch or a fill child."""
    grown = spread_labels(widget, size=size)
    assert not grown, (
        f"{type(widget).__name__} spread its slack across labels rather than into a stretch "
        f"or a fill child: {grown}"
    )


def off_the_content_column(widget, *, size=(1400, 900)) -> list[tuple[str, int, int]]:
    """The horizontal twin: ``(kind, x, column_x)`` for stacked items that start elsewhere.

    Nothing in this suite measured an *x* before this. Everything here is about the vertical
    axis, and the one horizontal rule the package states — ``sizing.clamp_width``'s promise
    that "everything else on these pages is left-aligned" — was checked nowhere, so a row
    could drift out of the column and only a person opening the app would see it.

    The measure is deliberately weak: the *left edge* of each item in the page's own
    top-level column, not its width. A width is not a defect on its own — a short heading in
    a wide label draws the same picture either way — but a left edge that does not line up
    with the item above it is visible on every page at every size.
    """
    layout = widget.layout()
    direction = getattr(layout, "direction", None)
    if direction is None or direction() not in (
        QBoxLayout.Direction.TopToBottom,
        QBoxLayout.Direction.BottomToTop,
    ):
        return []
    widget.resize(*size)
    widget.show()
    QApplication.processEvents()
    items = [
        layout.itemAt(i).widget()
        for i in range(layout.count())
        if layout.itemAt(i).widget() is not None
    ]
    xs = [w.mapTo(widget, w.rect().topLeft()).x() for w in items if w.isVisible()]
    column = min(xs) if xs else 0
    off = [
        (type(w).__name__, w.mapTo(widget, w.rect().topLeft()).x(), column)
        for w in items
        if w.isVisible() and w.mapTo(widget, w.rect().topLeft()).x() != column
    ]
    widget.hide()
    return off


def assert_one_content_column(widget, *, size=(1400, 900)) -> None:
    """Fail if anything stacked on ``widget`` starts left or right of its neighbours."""
    off = off_the_content_column(widget, size=size)
    assert not off, (
        f"{type(widget).__name__} does not share one left margin; these start elsewhere "
        f"(kind, x, column): {off}"
    )


def owns_a_pane(widget) -> bool:
    """Whether ``widget`` holds a splitter or a tab widget — i.e. panes of its own."""
    from PyQt6.QtWidgets import QSplitter, QTabWidget

    return bool(widget.findChildren(QSplitter) or widget.findChildren(QTabWidget))


def assert_panes_top_justify(widget, *, size=(1400, 1000)) -> None:
    """Every pane inside ``widget`` packs at the top, not just the widget itself.

    ``assert_top_justified`` already recurses — ``findChildren`` reaches a splitter half and a
    tab body like any other descendant — so the gap was never the check, it was the corpus.
    Its two call sites were the eleven smoke pages and ``AuditTab``, and **none of those eleven
    contains a splitter or a tab widget**, so the only panes measured anywhere were
    ``AuditTab``'s three, by accident. Twelve splitters and ten tab bodies had never been
    looked at, and two of them were spreading a 15px heading to 69px and 175px.

    A tab body only lays out once its tab is current, so this switches to each in turn — a
    surface checked on tab 0 alone leaves the others exactly as unmeasured as before.
    """
    from PyQt6.QtWidgets import QTabWidget

    widget.resize(*size)
    widget.show()
    QApplication.processEvents()
    try:
        assert_top_justified(widget, size=size)
        for tabs in widget.findChildren(QTabWidget):
            for index in range(tabs.count()):
                tabs.setCurrentIndex(index)
                QApplication.processEvents()
                assert_top_justified(widget, size=size)
    finally:
        widget.hide()


def locked_columns(table) -> list[int]:
    """The sections of ``table`` a user cannot drag normally.

    ``ResizeToContents`` and ``Fixed`` refuse a drag. So does ``Stretch`` — measured, a drag on
    a stretched column's border moved nothing — and it does worse to its neighbours: a column to
    its right, dragged wider, grows leftward because the stretched one gives the width up. The
    last section stretched through ``setStretchLastSection`` reports ``Interactive`` and drags
    normally, so it is not in this list.

    One predicate for every guard that asks, so the check proven to fail on a relocked column
    (``test_the_draggable_column_check_would_catch_one_relocked_column``) is the same one the
    sweep and the per-surface regression tests run. That proof used to restate the comprehension
    inline, which kept passing whatever the sweep itself was weakened to.
    """
    from PyQt6.QtWidgets import QHeaderView

    header = table.horizontalHeader()
    return [
        c
        for c in range(header.count())
        if header.sectionResizeMode(c) != QHeaderView.ResizeMode.Interactive
    ]


def column_edges(table) -> list[tuple[int, int]]:
    """``(left, right)`` of every column, in the table's own coordinates.

    Not the viewport's: once a drag has pushed the columns past the view, it scrolls, and every
    viewport position moves with it while nothing about the columns has.
    """
    header = table.horizontalHeader()
    return [
        (header.sectionPosition(c), header.sectionPosition(c) + header.sectionSize(c))
        for c in range(header.count())
    ]


def drag_column_border(table, column: int, dx: int) -> None:
    """Drag the right border of ``column`` by ``dx`` pixels, the way a mouse does.

    Press, move and release on the header's viewport, rather than ``resizeSection``: the bug
    this exists for is in what Qt does *around* a user's drag (a stretched neighbour giving the
    width back), and ``resizeSection`` is not a drag. ``table`` must be shown and laid out.
    """
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QMouseEvent

    header = table.horizontalHeader()
    y = header.height() / 2

    def send(kind, x, buttons) -> None:
        local = QPointF(x, y)
        event = QMouseEvent(
            kind,
            local,
            header.viewport().mapToGlobal(local.toPoint()).toPointF(),
            Qt.MouseButton.LeftButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(header.viewport(), event)
        QApplication.processEvents()

    x = header.sectionViewportPosition(column) + header.sectionSize(column) - 1
    send(QEvent.Type.MouseButtonPress, x, Qt.MouseButton.LeftButton)
    for step in range(1, 6):
        send(QEvent.Type.MouseMove, x + dx * step / 5, Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseButtonRelease, x + dx, Qt.MouseButton.NoButton)


def assert_border_follows_the_drag(table, column: int, dx: int = 60) -> None:
    """Drag ``column``'s right border by ``dx``: it moves ``dx``, and its left edge stays put."""
    before = column_edges(table)[column]
    drag_column_border(table, column, dx)
    after = column_edges(table)[column]
    assert (after[0], after[1]) == (before[0], before[1] + dx), (
        f"dragging column {column}'s right border {dx}px moved it from {before} to {after}; "
        "the border should follow the cursor and the column's left edge should not move"
    )


def assert_player_level_with_its_list(
    host, table, picture, *, above, max_rows: int
) -> None:
    """A list with a player beside it: ``max_rows`` rows tall (or all of them, if fewer), and
    the picture level with the list's first row rather than centred in the pane.

    **The class.** Both Browse tabs put their list and player in a splitter inside a tab that
    is as tall as the workspace's tallest tab. A splitter hands each child its whole height, so
    the list drew as many rows as the workspace was tall, and the player — which centres its
    picture in whatever pane it gets — drew its video hundreds of pixels down, off the screen
    while the list's first rows were on it. Reported as "the table is too big and video is
    centered to the table so is off the screen."

    Capping the list's maximum height looked like the fix and was a third failure: a splitter is
    only as tall as its shortest-capped child, so the list *and* the player sank to the bottom
    of the tab while the line above them spread into the gap. So the list must also start right
    under ``above`` — measured from ``above``'s natural height, because a spread label's own
    geometry reaches down to the list and would hide the gap it made.

    Measured after layout, in ``host``'s coordinates: the thing a user sees, not the policy
    that should produce it.
    """
    from PyQt6.QtCore import QPoint

    from detail.gui.sizing import lines

    problems = []
    row = table.verticalHeader().defaultSectionSize()
    chrome = table.horizontalHeader().height() + 2 * table.frameWidth()
    shown = (table.height() - chrome) / max(row, 1)
    # Both ways: past the cap is the reported list, and short of it is a list refilled after
    # layout keeping the hint of its empty self — five rows of thirty, measured.
    expected = min(table.model().rowCount(), max_rows)
    if abs(shown - expected) > 0.6:
        problems.append(f"the list is {shown:.1f} rows tall, not {expected}")
    table_top = table.mapTo(host, QPoint(0, 0)).y()
    above_bottom = above.mapTo(host, QPoint(0, 0)).y() + above.sizeHint().height()
    if table_top - above_bottom > lines(3):
        problems.append(
            f"the list starts {table_top - above_bottom}px below the row above it"
        )
    picture_top = picture.mapTo(host, QPoint(0, 0)).y()
    if abs(picture_top - table_top) > 8:
        problems.append(
            f"the picture starts at {picture_top}px, {picture_top - table_top}px below the "
            f"list beside it"
        )
    assert not problems, "; ".join(problems)
