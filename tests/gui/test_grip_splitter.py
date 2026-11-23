"""Every side panel folds one way: dragged past its minimum it shuts to a strip with its name on
it and an expand mark at its centre; clicked, pulled back out or double-clicked it opens at the
width it had, and both are remembered.

The class: **a panel that could be dragged to nothing, beside panels with a hide button of their
own.** Eighteen horizontal splits gave three answers to "put this away" — a ‹ / › button on the
navigation, "◂ Hide filters" / "Filters ▸" on Interpret, and on six plain `QSplitter`s a drag to
zero width that left no trace of the pane (user, 2026-11-09: the navigation dragged to nothing
and could not be found). `gui/grip_splitter.py` is the one mechanism; the structural half of the
guard (every split is a `GripSplitter`, no pane collapses unless declared, no hide buttons) is in
`detail/gui/requirements.py` and swept over every page and dialog by `test_requirements.py`.
This file is the behavioural half: the gestures, on a real shown splitter, with the pointer.

Drags go through `QTest` on the handle rather than `set_folded`, because the gesture is what a
user has and what an override of the handle's mouse events can break.

**The second class here: a strip that opens only by a gesture nobody tries.** Shut, the strip
answered a double-click, and a pull only once the pointer had crossed the panel's whole minimum
(160–180px on the real panels) with nothing on screen moving — so a user who grabbed it saw a
strip that does not drag (user, 2026-11-09). The pull is driven here in 10px steps with the
button held, each move delivered to the handle wherever it now is, on a stand-in split and on
the real navigation, filters, Audit list and query Audit columns.
"""

import pytest

from detail.gui.grip_splitter import (
    EXPAND_LEFT_PANEL,
    EXPAND_RIGHT_PANEL,
    HANDLE_PX,
    STRIP_PX,
    GripSplitter,
)
from detail.gui.qt import QApplication, QLabel, QPoint, QSettings, QTest, Qt

LIST_MIN = 150


def _split(qapp, key="test_panes", *, fold_last=False):
    """[list | main] (or [main | list]), 800px wide and shown — a split never shown has no
    geometry to drag."""
    split = GripSplitter(Qt.Orientation.Horizontal, key=key)
    side, main = QLabel("list"), QLabel("main")
    side.setMinimumWidth(LIST_MIN)
    main.setMinimumWidth(200)
    panes = (main, side) if fold_last else (side, main)
    for pane in panes:
        split.addWidget(pane)
    pane = 1 if fold_last else 0
    split.set_foldable(pane, "Labels")
    if not split.remember():
        split.setSizes([550, 250] if fold_last else [250, 550])
    split.resize(800, 300)
    split.show()
    qapp.processEvents()
    return split, pane


def _drag(qapp, split, handle_index, to):
    handle = split.handle(handle_index)
    start = handle.rect().center()
    end = QPoint(to - handle.x(), start.y())
    QTest.mousePress(
        handle, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start
    )
    QTest.mouseMove(handle, end)
    QTest.mouseRelease(
        handle, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end
    )
    qapp.processEvents()


def _double_click(qapp, split, handle_index):
    QTest.mouseDClick(split.handle(handle_index), Qt.MouseButton.LeftButton)
    qapp.processEvents()


@pytest.mark.parametrize("fold_last", [False, True], ids=["left", "right"])
def test_a_drag_past_the_minimum_shuts_the_panel_to_its_labelled_strip(qapp, fold_last):
    split, pane = _split(qapp, fold_last=fold_last)
    handle = 1
    _drag(qapp, split, handle, 600 if fold_last else 220)
    assert not split.is_folded(pane), "a drag above the minimum only resizes"
    _drag(qapp, split, handle, 790 if fold_last else 40)
    qapp.processEvents()
    assert split.is_folded(pane)
    assert split.sizes()[pane] == 0
    strip = split.handle(handle).width()
    assert strip == split.strip_width() >= STRIP_PX, "shut, the grip is the strip"
    assert strip > HANDLE_PX
    assert split.pane_name(pane) == "Labels"
    assert split.handle(handle).toolTip() == "Click or drag to show Labels"


def test_the_strip_paints_the_name(qapp):
    """The name is on the strip, not only in a tooltip: the strip's pixels differ from a blank
    grip's below the dots, where the text runs."""
    split, pane = _split(qapp, key="paint")
    split.set_folded(pane, True)
    qapp.processEvents()
    named = split.handle(1).grab().toImage()
    split.set_pane_name(pane, "")
    qapp.processEvents()
    blank = split.handle(1).grab().toImage()
    rows = range(30, 70)
    differ = sum(
        named.pixel(x, y) != blank.pixel(x, y)
        for x in range(named.width())
        for y in rows
    )
    assert differ > 20, "no text was painted down the strip"


def test_dragging_the_strip_back_out_opens_the_panel_under_the_pointer(qapp):
    split, pane = _split(qapp, key="drag_out")
    split.set_folded(pane, True)
    qapp.processEvents()
    assert split.is_folded(pane) and split.sizes()[0] == 0, "shut before the drag"
    _drag(qapp, split, 1, 300)
    assert not split.is_folded(pane)
    assert abs(split.sizes()[0] - 300) <= HANDLE_PX
    assert split.handle(1).width() == HANDLE_PX


def _shut_after_a_width(qapp, split, pane, width):
    """Drag the panel to ``width``, then shut it with the open grip's double-click."""
    edge = width if pane == 0 else split.width() - width - HANDLE_PX
    _drag(qapp, split, split._handle_of(pane), edge)
    opened = split.sizes()[pane]
    _double_click(qapp, split, split._handle_of(pane))
    assert split.is_folded(pane), "shut before the gesture under test"
    return opened


@pytest.mark.parametrize("where", ["centre", "top", "bottom"])
@pytest.mark.parametrize("fold_last", [False, True], ids=["left", "right"])
def test_a_click_anywhere_on_the_strip_opens_the_panel_at_its_last_width(
    qapp, fold_last, where
):
    split, pane = _split(qapp, key=f"click_{fold_last}_{where}", fold_last=fold_last)
    opened = _shut_after_a_width(qapp, split, pane, 320)
    strip = split.handle(split._handle_of(pane))
    at = {
        "centre": strip.rect().center(),
        "top": QPoint(strip.width() // 2, 4),
        "bottom": QPoint(strip.width() // 2, strip.height() - 4),
    }[where]
    QTest.mouseClick(
        strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at
    )
    qapp.processEvents()
    assert not split.is_folded(pane), "a click on the strip opens it"
    assert abs(split.sizes()[pane] - opened) <= 2, "at the width it had"


def test_a_click_on_an_open_grip_changes_nothing(qapp):
    split, pane = _split(qapp, key="click_open")
    before = split.sizes()
    QTest.mouseClick(split.handle(1), Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert not split.is_folded(pane) and split.sizes() == before


def test_a_double_click_on_the_strip_opens_it_and_leaves_it_open(qapp):
    """The first click of the double-click opens the strip; the second is the same gesture,
    not a request to shut it again."""
    split, pane = _split(qapp, key="dclick_shut")
    split.set_folded(pane, True)
    qapp.processEvents()
    strip = split.handle(1)
    at = strip.rect().center()
    left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    # What a pointer sends: press, release, then the second press arrives as a double-click.
    # `QTest.mouseDClick` alone sends only that last event, which skips the click under test.
    QTest.mousePress(strip, left, none, at)
    QTest.mouseRelease(strip, left, none, at)
    qapp.processEvents()
    assert not split.is_folded(pane), "the first click opened it"
    QTest.mouseDClick(strip, left, none, at)
    QTest.mouseRelease(strip, left, none, at)
    qapp.processEvents()
    assert not split.is_folded(pane), "and the second click left it open"


def _pull(qapp, split, handle_index, outward, steps):
    """Press on the strip's centre and move ``steps`` px at a time with the button held, the
    way a pointer does — each move sent to the handle wherever it now is, in its own
    coordinates, as Qt's pointer grab does. Returns ``(pointer, pane size, shut)`` per step.
    """
    handle = split.handle(handle_index)
    press = handle.rect().center()
    pointer = handle.mapTo(split, press)
    QTest.mousePress(
        handle, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, press
    )
    pane = split.governed_pane(handle_index)
    trace = []
    for step in steps:
        pointer = QPoint(pointer.x() + step * outward, pointer.y())
        QTest.mouseMove(handle, pointer - handle.pos())
        qapp.processEvents()
        trace.append((pointer.x(), split.sizes()[pane], split.is_folded(pane)))
    QTest.mouseRelease(
        handle,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        pointer - handle.pos(),
    )
    qapp.processEvents()
    return trace


@pytest.mark.parametrize("fold_last", [False, True], ids=["left", "right"])
def test_pulling_the_strip_out_opens_the_panel_and_it_follows_the_pointer(
    qapp, fold_last
):
    """The user's report: "should also let me grab and drag to open". Before, nothing moved
    until the pointer had travelled the panel's whole minimum width — 150px here, 160 on the
    navigation — so a pull on the strip looked like a strip that does not drag."""
    split, pane = _split(qapp, key=f"pull_{fold_last}", fold_last=fold_last)
    split.set_folded(pane, True)
    qapp.processEvents()
    outward = -1 if fold_last else 1
    trace = _pull(qapp, split, 1, outward, [10] * 30)
    first_open = next(i for i, (_x, _w, shut) in enumerate(trace) if not shut)
    assert first_open <= 1, f"still shut {first_open * 10}px into the pull: {trace}"
    assert trace[first_open][1] >= LIST_MIN, "opens at its minimum, never narrower"
    past_minimum = [
        (x, width) for x, width, _shut in trace if width > LIST_MIN + HANDLE_PX
    ]
    assert len(past_minimum) >= 10, f"never followed the pointer: {trace}"
    for x, width in past_minimum:
        edge = x if pane == 0 else split.width() - x
        assert abs(width - edge) <= HANDLE_PX, f"pointer at {x}, panel {width}: {trace}"
    widths = [width for _x, width in past_minimum]
    assert widths == sorted(widths, reverse=False), "the width grows with the pull"
    assert not split.is_folded(pane)
    assert split.handle(1).width() == HANDLE_PX, "open, the strip is a grip again"


@pytest.mark.parametrize("fold_last", [False, True], ids=["left", "right"])
def test_a_pull_taken_back_onto_the_strip_leaves_it_shut(qapp, fold_last):
    split, pane = _split(qapp, key=f"pullback_{fold_last}", fold_last=fold_last)
    split.set_folded(pane, True)
    qapp.processEvents()
    outward = -1 if fold_last else 1
    trace = _pull(qapp, split, 1, outward, [15, 15, -15, -15])
    assert not trace[1][2], f"opened on the way out: {trace}"
    assert split.is_folded(pane), f"back on the strip, it shuts again: {trace}"


def test_a_pull_inward_on_the_strip_leaves_it_shut(qapp):
    split, pane = _split(qapp, key="pull_in")
    split.set_folded(pane, True)
    qapp.processEvents()
    _pull(qapp, split, 1, -1, [10, 10])
    assert split.is_folded(pane)


@pytest.mark.parametrize("fold_last", [False, True], ids=["left", "right"])
def test_the_strip_shows_its_expand_mark_where_an_open_grip_has_its_dots(
    qapp, fold_last
):
    """ "Keep the icon to expand in the same location as the handles (center)." The mark takes the
    middle dot's place, a dot either side, and the pixels say so — not only the rect the code
    means to draw in."""
    split, pane = _split(qapp, key=f"mark_{fold_last}", fold_last=fold_last)
    handle = split.handle(1)
    open_dots = handle.dot_centres()
    open_centre = handle.height() / 2
    assert open_dots[1].y() == open_centre, "an open grip's dots are at its centre"
    assert handle.mark() == "", "an open grip shows no mark"
    split.set_pane_name(pane, "")
    split.set_folded(pane, True)
    qapp.processEvents()
    assert handle.mark() == (EXPAND_RIGHT_PANEL if fold_last else EXPAND_LEFT_PANEL)
    assert handle.height() == open_centre * 2, "same height shut as open"

    mark = handle.mark_rect()
    assert (
        abs(mark.center().y() - open_centre) <= 1
    ), "the mark is where the middle dot was"
    assert (
        abs(mark.center().x() - handle.width() / 2) <= 1.5
    ), "and across the strip's middle"
    assert (
        mark.height() >= 7
    ), f"a {mark.height()}px mark does not read as something to press"
    above, below = handle.dot_centres()
    assert above.y() < mark.top() and below.y() > mark.bottom(), "a dot either side"

    # The grip is the stand-in window's only focusable widget, so activation focuses it, and a
    # focused grip draws its ring round the whole strip — measured here is the mark, unfocused.
    handle.clearFocus()
    qapp.processEvents()
    image = handle.grab().toImage()
    background = image.pixel(handle.width() // 2, 2)

    def inked(y):
        return any(image.pixel(x, y) != background for x in range(2, image.width() - 2))

    ink = [y for y in range(image.height()) if inked(y)]
    assert ink, "nothing drawn on the strip"
    middle = (ink[0] + ink[-1]) / 2
    assert abs(middle - open_centre) <= 2, (
        f"the mark and dots span rows {ink[0]}–{ink[-1]}, centred at {middle}; an open grip's "
        f"dots are centred at {open_centre}"
    )
    assert ink[-1] - ink[0] <= 40, "mark and dots are one small group, not spread down"
    centre_rows = range(int(open_centre) - 2, int(open_centre) + 3)
    assert all(inked(y) for y in centre_rows), "the mark's own pixels are at the centre"


def test_the_strip_shows_a_pointing_hand_and_an_open_grip_the_resize_arrows(qapp):
    split, pane = _split(qapp, key="cursor")
    handle = split.handle(1)
    assert handle.cursor().shape() == Qt.CursorShape.SplitHCursor
    split.set_folded(pane, True)
    assert handle.cursor().shape() == Qt.CursorShape.PointingHandCursor
    assert handle.toolTip() == "Click or drag to show Labels"
    split.set_folded(pane, False)
    assert handle.cursor().shape() == Qt.CursorShape.SplitHCursor


def test_a_double_click_toggles_both_ways_and_reopens_at_the_last_width(qapp):
    split, pane = _split(qapp, key="double")
    _drag(qapp, split, 1, 330)
    open_width = split.sizes()[0]
    _double_click(qapp, split, 1)
    assert split.is_folded(pane)
    _double_click(qapp, split, 1)
    assert not split.is_folded(pane)
    assert abs(split.sizes()[0] - open_width) <= 2, "back at the width it had"


def test_main_content_refuses_to_fold(qapp):
    """The pane that is not declared foldable stops at its minimum, never at zero."""
    split, _pane = _split(qapp, key="main")
    _drag(qapp, split, 1, 795)
    assert split.sizes()[1] >= 200, "the main pane stopped at its minimum"
    assert not split.isCollapsible(1)
    with pytest.raises(ValueError):
        split.set_folded(1, True)


def test_a_middle_pane_cannot_be_declared_foldable(qapp):
    split = GripSplitter(Qt.Orientation.Horizontal, key="three")
    for text in ("a", "b", "c"):
        split.addWidget(QLabel(text))
    with pytest.raises(ValueError, match="first or last"):
        split.set_foldable(1, "Middle")


def test_a_split_needs_a_key():
    with pytest.raises(ValueError, match="key"):
        GripSplitter(Qt.Orientation.Horizontal, key="")


def test_width_and_shut_survive_a_new_instance(qapp):
    """Real `QSettings` (isolated per test process by `qapp`), a second splitter built the way
    the page builds it on the next visit."""
    split, pane = _split(qapp, key="persist")
    _drag(qapp, split, 1, 320)
    width = split.sizes()[0]
    _double_click(qapp, split, 1)
    split.deleteLater()

    again, _ = _split(qapp, key="persist")
    assert again.is_folded(pane), "shut, as it was left"
    _double_click(qapp, again, 1)
    assert abs(again.sizes()[0] - width) <= 2, "and opens at the width it had"


def test_an_old_saved_split_with_a_pane_at_zero_comes_back_shut(qapp):
    """The bug the user hit, as it sits in an existing settings file: a plain QSplitter's state
    with the navigation dragged to nothing. It migrates to a shut strip, not to zero width.
    """
    from detail.gui.qt import QSplitter

    old = QSplitter(Qt.Orientation.Horizontal)
    old.addWidget(QLabel("nav"))
    old.addWidget(QLabel("page"))
    old.resize(800, 300)
    old.show()
    qapp.processEvents()
    old.setSizes([0, 800])
    qapp.processEvents()
    QSettings().setValue("splitters/legacy", old.saveState())

    split, pane = _split(qapp, key="legacy")
    assert split.is_folded(pane)
    assert split.handle(1).width() == split.strip_width()
    assert QSettings().value("splitters/legacy") is None, "the old key is gone"


def test_the_drag_reaches_the_window_through_the_handle(qapp):
    """Guard the guard: the drags above are real pointer events on the handle — a handle that
    ignored them would leave the split where `setSizes` put it."""
    split, _ = _split(qapp, key="reach")
    before = split.sizes()[0]
    _drag(qapp, split, 1, before + 100)
    assert split.sizes()[0] > before + 50
    QApplication.processEvents()


@pytest.mark.parametrize(
    "text",
    [
        "‹",
        "›",
        "◂ Hide filters",
        "Filters ▸",
        "Hide navigation",
        "« Hide",
        "Show list ▸",
    ],
)
def test_the_hide_button_check_would_catch_a_pane_toggle(text):
    """`requirements.a_side_panel_has_no_hide_button` reads button text; these are the spellings
    the app had (‹ / ›, "◂ Hide filters", "Filters ▸") and their near neighbours."""
    from detail.gui.requirements import _hides_a_pane

    assert _hides_a_pane(text)


@pytest.mark.parametrize("fold_last", [False, True], ids=["left", "right"])
def test_the_strips_own_mark_is_allowed_and_the_same_mark_on_a_button_is_not(
    qapp, fold_last
):
    """`a_side_panel_has_no_hide_button` allows the strip's click target because it is the
    handle, not a push button — not because the text check was loosened. The strip's mark on a
    `QPushButton` is still refused, and the handle is not a kind the button rule walks.
    """
    from detail.gui.qt import QAbstractButton, QPushButton

    from detail.gui import requirements

    split, pane = _split(qapp, key=f"allowed_{fold_last}", fold_last=fold_last)
    split.set_folded(pane, True)
    handle = split.handle(1)
    mark = handle.mark()
    assert mark, "the strip shows a mark"
    assert not isinstance(
        handle, QAbstractButton
    ), "the strip is the handle, not a button"
    assert not split.findChildren(QPushButton), "no button was added beside the panel"
    with pytest.raises(AssertionError, match="hides or shows a panel"):
        requirements.a_side_panel_has_no_hide_button(QPushButton(mark))


@pytest.mark.parametrize(
    "text",
    [
        "▶",
        "◀",
        "‹ an_32",
        "▸ climb",
        "Show Active processes →",
        "Open in window",
        "Whole tree",
        "Labels ▾",
        "next (→) ▶",
    ],
)
def test_the_hide_button_check_passes_buttons_that_hide_nothing(text):
    """Play and step (▶ ◀), a run link (‹ an_32), a fold header (▸ climb) and a navigation
    button are not pane toggles — the check's known limits are in its docstring."""
    from detail.gui.requirements import _hides_a_pane

    assert not _hides_a_pane(text)


# --------------------------------------------------------------------------- #
# The real panels: the gesture through the pages a user opens, not a stand-in.
# --------------------------------------------------------------------------- #
def _reopens_by_pull_and_by_click(qapp, split, pane, name):
    """Shut ``pane``, pull its strip out with the pointer, and check the panel opened at its
    minimum and then followed; shut it again and click the strip."""
    handle_index = split._handle_of(pane)
    split.set_folded(pane, True)
    qapp.processEvents()
    handle = split.handle(handle_index)
    assert handle.toolTip() == f"Click or drag to show {name}"
    assert handle.mark() == (EXPAND_LEFT_PANEL if pane == 0 else EXPAND_RIGHT_PANEL)
    minimum = split.minimum_of(pane)
    outward = 1 if pane == 0 else -1
    trace = _pull(qapp, split, handle_index, outward, [10] * ((minimum + 120) // 10))
    assert not trace[0][2], f"{name}: still shut after the first 10px: {trace[:3]}"
    assert not split.is_folded(pane)
    _x, width, _shut = trace[-1]
    # The page's other panes hold their own minimums, so "followed" means up to what they
    # leave — the clustering Audit's review keeps all but ~220px of a laptop window.
    others = sum(split.minimum_of(i) for i in range(split.count()) if i != pane)
    room = split.width() - others - (split.count() - 1) * HANDLE_PX
    want = min(minimum + 60, room)
    assert (
        want > minimum
    ), f"{name}: no room to follow past {minimum}px, the check is vacuous"
    assert (
        width >= want - 2
    ), f"{name}: the panel did not follow the pull past {minimum}px: {trace[-3:]}"
    split.set_folded(pane, True)
    qapp.processEvents()
    QTest.mouseClick(split.handle(handle_index), Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert not split.is_folded(pane), f"{name}: a click did not open it"
    assert abs(split.sizes()[pane] - width) <= 2, f"{name}: not at its last width"


def test_the_navigation_filters_and_audit_list_open_by_pull_and_click(qapp, fitted):
    """Three left panels in the real window at a laptop's size: the navigation, Interpret's
    filters and the clustering Audit's list."""
    from detail.gui.app import MainWindow
    from detail.gui.cluster_workspace import ClusterWorkspace
    from tests.gui.conftest import drain_busy_calls, drain_until, pump_for

    experiment, run_id = fitted
    window = MainWindow()
    window.resize(1440, 847)
    window.show()
    try:
        window._open_experiment(experiment)
        window._open_analysis_run(run_id)
        page = window._content.widget(0)
        workspace = page.findChildren(ClusterWorkspace)[0]
        assert drain_until(
            qapp,
            lambda: workspace._interpret._view is not None and workspace._audit._loaded,
        )
        drain_busy_calls(qapp)
        _reopens_by_pull_and_by_click(qapp, window._split, 0, "Navigation")
        page._show_view("Interpret")
        pump_for(qapp, 200)
        filters = workspace._interpret._split
        _reopens_by_pull_and_by_click(qapp, filters, 0, filters.pane_name(0))
        page._show_view("Audit")
        pump_for(qapp, 200)
        _reopens_by_pull_and_by_click(qapp, workspace._audit._split, 0, "Labels")
    finally:
        window.close()


def test_the_query_audits_recordings_open_by_pull_and_click(qapp, tmp_path):
    """The query Audit's recording list beside its review pane."""
    from detail.gui.query_review import LIST_SPLIT_KEY, QueryReviewTab
    from tests.api.test_query import _searched

    exp, run = _searched(tmp_path)
    tab = QueryReviewTab(exp, run)
    tab.resize(1440, 847)
    tab.show()
    qapp.processEvents()
    split = tab.findChildren(GripSplitter)[0]
    assert split.key == LIST_SPLIT_KEY
    _reopens_by_pull_and_by_click(qapp, split, 0, "Recordings")


# --- the keyboard ------------------------------------------------------------------------------
#
# **The third class: a panel the keyboard can shut and never reopen.** The grip took no focus, so
# a strip opened only by the mouse — the navigation alone had Ctrl+\ — and a panel shut while one
# of its fields held the focus left the focus on a widget nobody could see (review, 2026-11-09).
# Driven by real Tab and key presses through `QTest`, on a shown window.


def _typing_split(qapp, key):
    """[field in a side panel | field in the main pane], shown and active, the side field
    focused."""
    from detail.gui.qt import QLineEdit

    split = GripSplitter(Qt.Orientation.Horizontal, key=key)
    side, main = QLineEdit("side"), QLineEdit("main")
    side.setMinimumWidth(LIST_MIN)
    main.setMinimumWidth(200)
    split.addWidget(side)
    split.addWidget(main)
    split.set_foldable(0, "Labels")
    if not split.remember():
        split.setSizes([250, 550])
    split.resize(800, 300)
    split.show()
    split.activateWindow()
    QTest.qWaitForWindowActive(split, 2000)
    side.setFocus()
    qapp.processEvents()
    assert side.hasFocus(), "the stand-in field must start with the focus"
    return split, side, main


def _tab_to(qapp, start, target, limit=6) -> bool:
    for _ in range(limit):
        if target.hasFocus():
            return True
        QTest.keyClick(QApplication.focusWidget() or start, Qt.Key.Key_Tab)
        qapp.processEvents()
    return target.hasFocus()


@pytest.mark.parametrize("key", [Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space])
def test_tab_reaches_the_grip_and_its_key_shuts_and_opens_the_panel(qapp, key):
    split, side, _main = _typing_split(qapp, f"keys_{int(key.value)}")
    handle = split.handle(1)
    assert _tab_to(qapp, side, handle), "Tab never reached the grip"
    QTest.keyClick(handle, key)
    qapp.processEvents()
    assert split.is_folded(0), "the key shut the panel"
    assert handle.hasFocus(), "and the strip keeps the focus"
    QTest.keyClick(handle, key)
    qapp.processEvents()
    assert not split.is_folded(0), "the same key opened it again"
    split.close()


def test_a_click_on_the_grip_does_not_take_the_focus(qapp):
    """Tab, not a click: dragging the divider leaves the keys where the user was typing."""
    split, side, _main = _typing_split(qapp, "click_focus")
    handle = split.handle(1)
    QTest.mouseClick(
        handle,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        handle.rect().center(),
    )
    qapp.processEvents()
    assert side.hasFocus()
    split.close()


def test_a_panel_that_shuts_with_the_focus_hands_it_to_its_strip(qapp):
    """Shut by the mouse with its field focused: the focus moves to the strip, not left on the
    hidden field — so the very next Enter opens the panel."""
    split, side, _main = _typing_split(qapp, "focus_moves")
    _drag(qapp, split, 1, 30)
    assert split.is_folded(0)
    assert QApplication.focusWidget() is split.handle(1)
    QTest.keyClick(split.handle(1), Qt.Key.Key_Return)
    qapp.processEvents()
    assert not split.is_folded(0)
    split.close()


def test_a_panel_without_the_focus_leaves_it_where_it_is(qapp):
    split, _side, main = _typing_split(qapp, "focus_stays")
    main.setFocus()
    qapp.processEvents()
    _drag(qapp, split, 1, 30)
    assert split.is_folded(0)
    assert main.hasFocus(), "folding a panel nobody was in moves nothing"
    split.close()


def test_a_focused_grip_draws_its_focus_ring(qapp):
    """Where Enter would act is visible: the focused strip paints differently from the same strip
    unfocused."""
    split, side, _main = _typing_split(qapp, "focus_ring")
    split.set_folded(0, True)
    handle = split.handle(1)
    side.clearFocus()
    split.setFocus()
    qapp.processEvents()
    assert not handle.hasFocus()
    before = handle.grab().toImage()
    handle.setFocus()
    qapp.processEvents()
    assert handle.hasFocus()
    after = handle.grab().toImage()
    assert before != after, "the focused strip looks the same as an unfocused one"
    split.close()
