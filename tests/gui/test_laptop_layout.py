"""A run's views on a laptop window: nothing laid out past the edge of what holds it.

**The class: a content floor taller than the space it is held to, with nothing to absorb the
difference.** A view open full window on the run page is held to the window — the page asks for
no height (`AnalysisRunDetailPage.minimumSizeHint`, review round 1) — so it does not scroll.
When the view's own rows want more than that, Qt lays the children out at their floors anyway
and the bottom ones run past their parent's edge, where they are clipped and cannot be reached.
Nothing raises, and on a large monitor nothing is wrong.

How it hid on INTERPRET (render audit, 2026-11-09, 1440×847): the column of tree, label bar and
clips needed 764px and got 660. The tree was a fixed height computed from heights that had
*already* overflowed (660 − 273 − 357 = 30px of "room" where ~140 was real), so it never gave
up a pixel; the montage kept a 357px floor the column could not honour; and the clip grid's
bottom 94px — its scrollbar's end included — sat under the window's edge. `cluster_workspace.py`
said the page scrolls when the window cannot hold it; with a view open, it does not.

**The rule:** at a laptop window, every view of a run's page fits — no visible widget extends
past its visible parent (contents of a scroll area excepted: that is what scrolling is). The
tree gives height back, down to ``TREE_MIN_PX``, before the clips drop under their rows.

**Why not a `requirements.py` predicate over the whole corpus.** The property is about a window
*size* and real content: the corpus builds subjects at whatever size their constructor leaves,
mostly unshown, with fake data, so "past its parent" there measures the harness rather than the
app. The predicate is a pure function here (:func:`widgets_past_their_parent`) so it can be
pointed at another page the day one needs it.
"""

import pytest

from detail.gui.cluster_workspace import ClusterWorkspace
from detail.gui.qt import (
    QAbstractScrollArea,
    QColor,
    QLabel,
    QPoint,
    QPushButton,
    QSettings,
    QWidget,
)

LAPTOP = (1440, 847)


def widgets_past_their_parent(root: QWidget, slack: int = 1) -> list[str]:
    """Every visible widget under ``root`` whose geometry runs past its visible parent's, by
    more than ``slack`` px — skipping what sits inside a scroll area's viewport."""
    out = []
    for child in root.findChildren(QWidget):
        if child.isWindow() or not child.isVisible():
            continue
        parent = child.parentWidget()
        if parent is None or not parent.isVisible():
            continue
        if _in_a_scroll_area(child):
            continue
        g = child.geometry()
        over_x, over_y = (
            g.right() + 1 - parent.width(),
            g.bottom() + 1 - parent.height(),
        )
        if over_x > slack or over_y > slack:
            out.append(
                f"{type(child).__name__}({child.objectName()}) runs {max(over_x, 0)}px right "
                f"and {max(over_y, 0)}px below its {type(parent).__name__}"
            )
    return out


def _in_a_scroll_area(widget: QWidget) -> bool:
    """Inside a scroll area that can actually scroll — whose bars have a range.

    One that cannot is no excuse: the shell's page scroll gives a run page with a view open
    exactly its viewport and no range, and everything clipped inside that page is clipped for
    good. Exempting every scroll area let this guard pass against the bug it was written for.
    """
    node = widget.parentWidget()
    while node is not None:
        owner = node.parentWidget()
        if isinstance(owner, QAbstractScrollArea) and node is owner.viewport():
            bars = (owner.verticalScrollBar(), owner.horizontalScrollBar())
            return any(bar.maximum() > 0 for bar in bars)
        node = owner
    return False


def test_the_overflow_predicate_would_catch_a_child_past_its_parent(qapp):
    window = QWidget()
    window.resize(400, 400)
    parent = QWidget(window)
    parent.setGeometry(0, 0, 200, 100)
    tall = QLabel("laid out at its floor, past the box", parent)
    tall.setGeometry(0, 0, 200, 300)
    window.show()
    qapp.processEvents()
    try:
        assert widgets_past_their_parent(window), "300px laid out in 100px is caught"
        tall.setGeometry(0, 0, 200, 100)
        assert widgets_past_their_parent(window) == []
    finally:
        window.close()


def _pump(qapp, ms: int) -> None:
    from tests.gui.conftest import pump_for

    pump_for(qapp, ms)


@pytest.fixture
def laptop_run(qapp, fitted):
    """The real shell at 1440×847 with the fitted run's page open."""
    from detail.gui.app import MainWindow
    from tests.gui.conftest import drain_busy_calls, drain_until

    experiment, run_id = fitted
    window = MainWindow()
    window.resize(*LAPTOP)
    window.show()
    window._open_experiment(experiment)
    window._open_analysis_run(run_id)
    page = window._content.widget(0)
    workspace = page.findChildren(ClusterWorkspace)[0]
    assert drain_until(
        qapp,
        lambda: workspace._interpret._view is not None
        and workspace._browse._loaded
        and workspace._audit._loaded,
    )
    drain_busy_calls(qapp)
    yield window, page, workspace
    window.close()


def _open(qapp, page, view: str) -> None:
    page._show_view(view)
    _pump(qapp, 400)


@pytest.mark.parametrize("view", ["Interpret", "Audit", "Browse", "Measures"])
def test_every_view_fits_a_laptop_window(qapp, laptop_run, view):
    window, page, _workspace = laptop_run
    _open(qapp, page, view)
    assert widgets_past_their_parent(window) == []


def _grid_bottom_in_window(window, interpret) -> tuple[int, int]:
    grid = interpret.montage().grid
    bottom = grid.mapTo(window, QPoint(0, grid.height())).y()
    central = window.centralWidget()
    edge = central.mapTo(window, QPoint(0, central.height())).y()
    return bottom, edge


def test_a_tall_remembered_tree_gives_way_before_the_clips(qapp, laptop_run):
    """The reported shape, forced: a tree dragged taller than the window has room for. It is
    held so the clip grid still ends on screen, and kept at the user's height in the settings
    for a window that has room."""
    from detail.gui.node_workspace import TREE_HEIGHT_KEY, TREE_MIN_PX

    window, page, workspace = laptop_run
    _open(qapp, page, "Interpret")
    interpret = workspace._interpret
    interpret._tree_dragged = True
    interpret._icicle.set_fit_rows(False)
    QSettings().setValue(TREE_HEIGHT_KEY, 700)
    interpret._tree_holder.setFixedHeight(700)
    interpret._schedule_tree_fit()
    _pump(qapp, 400)
    bottom, edge = _grid_bottom_in_window(window, interpret)
    assert bottom <= edge + 1, f"the clip grid runs {bottom - edge}px under the window"
    montage = interpret.montage()
    assert interpret._tree_holder.height() >= TREE_MIN_PX
    assert (
        interpret._tree_holder.height() == TREE_MIN_PX
        or montage.height() >= montage.height_for_rows(2) - 2
    ), "the clips lost their rows while the tree had height to give"
    assert int(QSettings().value(TREE_HEIGHT_KEY)) == 700, "the user's height is kept"
    assert widgets_past_their_parent(window) == []


def test_the_clip_grid_ends_on_screen_at_the_trees_own_height(qapp, laptop_run):
    window, page, workspace = laptop_run
    _open(qapp, page, "Interpret")
    interpret = workspace._interpret
    bottom, edge = _grid_bottom_in_window(window, interpret)
    assert bottom <= edge + 1, f"the clip grid runs {bottom - edge}px under the window"
    assert interpret.montage().grid.height() >= interpret.montage().height_for_rows(
        1
    ) - (interpret.montage().height() - interpret.montage().grid.height())


def test_every_view_works_with_the_navigation_hidden(qapp, laptop_run):
    window, page, _workspace = laptop_run
    window._split.set_folded(0, True)
    _pump(qapp, 200)
    assert window._split.sizes()[0] == 0, "shut to its strip"
    for view in ("Interpret", "Audit", "Browse", "Measures"):
        _open(qapp, page, view)
        assert widgets_past_their_parent(window) == [], view
    window._split.set_folded(0, False)


def test_the_label_bar_gives_the_figures_and_the_status_a_whole_line(qapp, laptop_run):
    """User decision A (2026-11-09): row 1 is the name, the chip and the montage toolbar (order,
    Show ▾, Size, Export ▾); under it a short line holds the figures at its left and the status
    at its right, each **whole** at 1440×847 with the navigation and the filters open; the field,
    Label, Remove, Labels ▾, Undo and Open in window on the last. Both still shorten with "…" and
    say the whole line in their tooltip when a window is narrower yet."""
    window, page, workspace = laptop_run
    _open(qapp, page, "Interpret")
    interpret = workspace._interpret
    view = interpret._view
    interpret.select(view.children(view.root)[0])
    bar, montage = interpret.label_bar(), interpret.montage()
    bar.set_quick_labels([("climb", QColor("#1f77b4"))])
    bar.confirm(f"Labelled #{interpret._selected} “climb”")
    bar.set_saving("Output up to date.")
    _pump(qapp, 300)
    assert window._split.sizes()[0] > 0, "the navigation is open"
    assert interpret._split.sizes()[0] > 0, "the filters are open"

    def row_of(widget):
        return widget.mapTo(bar, QPoint(0, widget.height() // 2)).y()

    assert bar.isAncestorOf(montage.order_button), "the toolbar moved onto the bar"
    first = row_of(bar.name)
    toolbar = (
        montage.order_button,
        montage.show_button,
        montage.size_slider,
        montage.exports.button,
    )
    for widget in toolbar:
        assert abs(row_of(widget) - first) <= 6, widget
    info = row_of(bar.figures)
    assert info > first + 10 and abs(row_of(bar.status) - info) <= 6
    second = row_of(bar.field.edit)
    assert second > info + 10
    for widget in (bar.field.list_button, bar.undo, interpret._open_window):
        assert abs(row_of(widget) - second) <= 6, f"{widget} is on the field's row"
    assert (
        bar.undo.x()
        < interpret._open_window.mapTo(bar.undo.parentWidget(), QPoint()).x()
    )
    for label in (bar.figures, bar.status):
        assert (
            not label.is_elided()
        ), f"{label.text()!r} is cut to {label.shown_text()!r}"
        assert label.width() >= label.full_width()
    assert bar.figures.x() < bar.status.x()
    assert (
        not montage.grid.mapTo(bar, QPoint(0, 0)).y() < bar.height()
    ), "grid under the bar"
    figures = bar.figures
    figures.resize(60, figures.height())
    assert figures.shown_text().endswith("…")
    assert figures.toolTip().split("\n")[0] == figures.text()


def test_the_filters_drag_to_240_and_keep_it(qapp, laptop_run, fitted):
    from detail.gui.node_workspace import FILTER_SPLIT_KEY

    window, page, workspace = laptop_run
    _open(qapp, page, "Interpret")
    interpret = workspace._interpret
    split = interpret._split
    split.setSizes([240, sum(split.sizes()) - 240])
    split.splitterMoved.emit(240, 1)
    _pump(qapp, 200)
    assert split.sizes()[0] == 240, "the dock drags down to 240"
    stored = str(QSettings().value(f"panes/{FILTER_SPLIT_KEY}/sizes"))
    assert stored.split(",")[0] == "240"
    assert interpret.filters().region.map.side <= 240
    assert widgets_past_their_parent(window) == []


# --- the Size slider changes how many tiles fit, never the area ------------------------ #
def test_the_size_slider_leaves_interprets_layout_alone(qapp, laptop_run):
    """**The class: a layout sized from content whose size the user tunes.** The page reserved
    two rows of tiles *at the slider's tile size*, so dragging Size from min to max grew the
    clips 201px and shrank the tree to its floor (1440×847, an_32) — the slider resized the
    page. The reservation is at the default tile size now; the slider changes how many tiles
    fit, nothing else."""
    from detail.gui.live_montage import MAX_TILE_PX, MIN_TILE_PX

    window, page, workspace = laptop_run
    _open(qapp, page, "Interpret")
    interpret = workspace._interpret
    montage = interpret.montage()

    def geometry():
        top = montage.mapTo(window, QPoint(0, 0))
        return (
            top.y(),
            montage.width(),
            montage.height(),
            interpret._tree_holder.height(),
            window.size(),
        )

    montage.size_slider.setValue(MIN_TILE_PX)
    _pump(qapp, 300)
    at_min, columns_at_min = geometry(), montage._columns()
    montage.size_slider.setValue(MAX_TILE_PX)
    _pump(qapp, 300)
    assert montage.tile_px() == MAX_TILE_PX
    assert geometry() == at_min, "the slider moved the montage, the tree or the window"
    assert montage._columns() < columns_at_min, "the slider changes how many tiles fit"


def test_the_size_slider_leaves_the_cluster_window_alone(qapp, laptop_run):
    from detail.gui.live_montage import MAX_TILE_PX, MIN_TILE_PX

    window, page, workspace = laptop_run
    _open(qapp, page, "Interpret")
    interpret = workspace._interpret
    view = interpret._view
    interpret.select(view.children(view.root)[0])
    popout = interpret.open_window()
    _pump(qapp, 400)
    try:
        montage = popout.montage
        montage.size_slider.setValue(MIN_TILE_PX)
        _pump(qapp, 300)
        before = (popout.size(), montage.geometry())
        montage.size_slider.setValue(MAX_TILE_PX)
        _pump(qapp, 300)
        assert montage.tile_px() == MAX_TILE_PX
        assert (popout.size(), montage.geometry()) == before
    finally:
        popout.close()


# --- the windows a run's page opens, at a laptop's size -------------------------------- #
def test_the_cluster_window_fits_a_laptop_screen(qapp, laptop_run):
    window, page, workspace = laptop_run
    _open(qapp, page, "Interpret")
    interpret = workspace._interpret
    view = interpret._view
    interpret.select(view.children(view.root)[0])
    popout = interpret.open_window()
    try:
        popout.resize(popout.size().boundedTo(window.size()))
        _pump(qapp, 400)
        assert popout.width() <= LAPTOP[0] and popout.height() <= LAPTOP[1]
        assert widgets_past_their_parent(popout) == []
        bar = popout.bar
        assert not [
            b for b in bar.findChildren(QPushButton) if b.text().startswith("Recent")
        ], "the recent labels lead Labels ▾; there is no Recent ▾"
    finally:
        popout.close()


def test_the_cluster_windows_bar_cuts_nothing_at_its_default_size(qapp, laptop_run):
    """The window's top is INTERPRET's label bar (user, 2026-11-09), toolbar and all, in a
    window about half a laptop wide: at the size it opens at, no control of line 1 is cut below
    its own width, and the figures and the status are whole."""
    window, page, workspace = laptop_run
    _open(qapp, page, "Interpret")
    interpret = workspace._interpret
    view = interpret._view
    interpret.select(view.children(view.root)[0])
    page_bar = interpret.label_bar()
    page_bar.confirm("Labelled #1384 “climb”")
    page_bar.set_saving("Output up to date.")
    popout = interpret.open_window()
    try:
        assert drain_until_clips(qapp, popout)
        _pump(qapp, 400)
        bar, montage = popout.bar, popout.montage
        assert popout.width() <= LAPTOP[0] and popout.height() <= LAPTOP[1]
        cut = [
            f"{type(w).__name__}({getattr(w, 'text', lambda: '')()}) "
            f"{w.width()} < {w.sizeHint().width()}"
            for w in montage.controls_row.findChildren(QWidget)
            if w.isVisible()
            and w.parentWidget() is montage.controls_row
            and w.minimumWidth() != w.maximumWidth()
            and w.width() < w.sizeHint().width()
        ]
        assert cut == []
        assert bar.name.width() >= bar.name.sizeHint().width(), "the title is whole"
        assert bar.status.text() == page_bar.status.text() != ""
        for label in (bar.figures, bar.status):
            assert (
                not label.is_elided()
            ), f"{label.text()!r} is cut to {label.shown_text()!r} at {popout.width()}px"
        print(
            f"\nwindow {popout.width()}x{popout.height()}: figures "
            f"{bar.figures.width()}/{bar.figures.full_width()}px, status "
            f"{bar.status.width()}/{bar.status.full_width()}px, grid "
            f"{montage.grid.height()}px"
        )
    finally:
        popout.close()


def drain_until_clips(qapp, popout) -> bool:
    from tests.gui.conftest import drain_until

    return drain_until(qapp, lambda: popout._clips is not None)


def test_the_clip_window_fits_a_laptop_screen(qapp, laptop_run):
    from detail.gui.clip_popup import ClipPopup
    from tests.gui.conftest import drain_until

    window, page, workspace = laptop_run
    _open(qapp, page, "Interpret")
    interpret = workspace._interpret
    view = interpret._view
    interpret.select(view.children(view.root)[0])
    assert drain_until(qapp, lambda: len(interpret._order) > 0)
    interpret._play_clip(0)
    popup = interpret._clip_window
    assert isinstance(popup, ClipPopup) and not popup.isModal()
    try:
        popup.resize(*LAPTOP)
        _pump(qapp, 600)
        assert popup.height() <= LAPTOP[1]
        assert widgets_past_their_parent(popup) == []
    finally:
        popup.close()


def test_the_tree_at_a_laptops_size_keeps_its_words_apart_and_its_buttons_natural(
    qapp, laptop_run
):
    """The tree's rows as a laptop window fits them: no block's name and share overlap
    (`node_icicle.cell_text`), and "−", "+", "Whole tree" are as wide as their text, at the
    row's right end — they were ~420px each, filling it."""
    from detail.gui.qt import QFont, QFontMetricsF, QRectF

    from detail.gui.node_icicle import cell_text

    window, page, workspace = laptop_run
    _open(qapp, page, "Interpret")
    interpret = workspace._interpret
    icicle = interpret._icicle
    icicle.grab()
    small = QFont(icicle.font())
    small.setPointSizeF(max(small.pointSizeF() - 1.5, 7.0))
    blocks = [rect for rect, _node in icicle._blocks if rect.width() > 46]
    assert blocks, "the tree drew no block wide enough to label"
    for rect in blocks:
        boxes = cell_text(
            rect.intersected(QRectF(icicle.rect())),
            "#1397",
            74.9,
            QFontMetricsF(icicle.font()),
            QFontMetricsF(small),
        )
        rects = [box for box, _t, _k in boxes]
        for i, a in enumerate(rects):
            for b in rects[i + 1 :]:
                assert a.intersected(b).isEmpty(), (rect, a, b)
    row = interpret._fit.parentWidget()
    for button in (interpret._zoom_out, interpret._zoom_in, interpret._fit):
        assert button.width() <= button.sizeHint().width() + 2, button.text()
    right = interpret._fit.geometry().right()
    assert right >= row.width() - 2 - row.layout().contentsMargins().right()


def test_the_label_bars_first_row_cuts_none_of_its_controls(qapp, laptop_run):
    """With the orientation menu and *Open in window* added to the montage's controls, the
    first row asked for more than a laptop has, and every control was cut below its own width
    ("As recorde", "pen in window"). Both are on the second row now."""
    window, page, workspace = laptop_run
    _open(qapp, page, "Interpret")
    interpret = workspace._interpret
    view = interpret._view
    interpret.select(view.children(view.root)[0])
    _pump(qapp, 300)
    bar, montage = interpret.label_bar(), interpret.montage()
    cut = [
        f"{type(w).__name__}({w.objectName() or getattr(w, 'text', lambda: '')()}) "
        f"{w.width()} < {w.sizeHint().width()}"
        for w in montage.controls_row.findChildren(QWidget)
        if w.isVisible()
        and w.parentWidget() is montage.controls_row
        and w.minimumWidth() != w.maximumWidth()  # a fixed width (the slider) is chosen
        and w.width() < w.sizeHint().width()
    ]
    assert cut == []
    second = interpret._open_window.mapTo(bar, QPoint(0, 0)).y()
    assert second >= bar.field.edit.mapTo(bar, QPoint(0, 0)).y() - 6
    assert interpret._open_window.width() >= interpret._open_window.sizeHint().width()
