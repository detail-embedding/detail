"""The node tree draws the clusters a user compares side by side, not a staircase (§22).

**The class.** A real fit's merge tree is mostly chains: one big cluster sheds a piece at each
merge. Drawn one merge per row, every row is a near-full-width bar over the next, the pieces that
came off are scattered one per row down the right edge, and a row of hairline slivers draws as a
barcode. A real run looked exactly like that and was reported as "all horizontal,
the graph doesn't have sufficient differentiation between clusters". Nothing failed: every block
was the right width for its node.

So a lopsided merge (a *peel*) gets no row of its own, the pieces it is made of stand in for it,
largest first, and blocks too narrow to read are bare columns of their own colour while each
stays a click target. These tests build the chain shape directly, so the guard does not depend on what a
fixture's fit happens to look like.
"""

import numpy as np
import pytest
from PyQt6.QtCore import QPointF

from detail.gui import node_icicle
from detail.gui.node_icicle import NodeIcicle, TreePane, row_px


class _Summary:

    members = 1


class _Tree:
    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes


class _ChainView:
    """A peeling chain: the core sheds 10%, 12%, 15% … and finally splits evenly.

    Leaves ``0..5`` are the pieces shed (and the two halves of the last split); internal nodes
    are ``6..10`` with ``10`` the root.
    """

    def __init__(self) -> None:
        #            0    1    2    3    4    5
        leaf_times = [10, 11, 12, 13, 27, 27]
        self._children = {6: (4, 5), 7: (6, 3), 8: (7, 2), 9: (8, 1), 10: (9, 0)}
        times = list(leaf_times) + [0] * 5
        for node in range(6, 11):
            times[node] = sum(times[c] for c in self._children[node])
        self._times = np.asarray(times, dtype=np.float64) * 1e9
        self.tree = _Tree(11)
        self.root = 10

    def node_times(self):
        return self._times

    def children(self, node: int):
        return self._children.get(int(node), ())

    def ancestors(self, node: int) -> list[int]:
        out = []
        for parent, kids in self._children.items():
            if node in kids:
                out = self.ancestors(parent) + [parent]
        return out

    def summary(self, node: int):
        return _Summary()


@pytest.fixture
def icicle(qapp):
    widget = NodeIcicle()
    widget.set_view(_ChainView())
    widget.resize(600, 300)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()


def test_a_peeling_chain_puts_its_pieces_side_by_side(icicle):
    """Under the root: the shed pieces and the final split's halves, in one row."""
    row = icicle.shown_children(10)
    assert set(row) == {0, 1, 2, 3, 6}, "every peel opened, the even split kept"
    times = icicle._times
    assert [times[n] for n in row] == sorted(
        (times[n] for n in row), reverse=True
    ), "largest first"


def test_an_even_split_keeps_its_row(icicle):
    assert icicle.shown_children(6) == [4, 5] or icicle.shown_children(6) == [5, 4]


def test_the_guard_would_catch_the_staircase(icicle, monkeypatch):
    """Guard the guard: with peels not opened, the root's row is the chain again."""
    monkeypatch.setattr(node_icicle, "SPLIT_SHARE", 0.0)
    assert icicle.shown_children(10) == [9, 0]


def test_a_chain_node_selected_from_the_card_outlines_its_pieces(icicle, qapp):
    """A chain node has no block of its own, and is still the selection: its pieces say so."""
    icicle.set_selected(8)  # the core after two peels: pieces 2, 3 and 6
    icicle.grab()
    assert icicle._stand_ins() == {2, 3, 6}


def test_a_hairline_block_is_still_a_target(qapp):
    """Drawn as a bare column, each tiny block keeps its own slot to click."""
    view = _ChainView()
    widget = NodeIcicle()
    widget.set_view(view)
    widget.setMinimumWidth(0)
    widget.resize(40, 200)  # narrow enough that the shed pieces are under TINY_BLOCK_PX
    widget.show()
    qapp.processEvents()
    widget.grab()
    assert widget._tiny, "the fixture must draw some blocks as hairlines"
    for rect, node in widget._blocks:
        if node in widget._tiny:
            centre = rect.center()
            assert widget.node_at(centre.x(), centre.y()) == node
    assert not widget._remainders, "every piece here is wide enough for its own column"
    widget.close()


def _column_colour(widget, rect):
    image = widget.grab().toImage()
    ratio = image.devicePixelRatio()
    return image.pixelColor(
        int((rect.left() + rect.width() / 2) * ratio),
        int((rect.top() + rect.height() / 2) * ratio),
    )


def test_a_thin_block_is_shaded_by_its_own_share_without_zooming(qapp):
    """Under ``TINY_BLOCK_PX`` a block was one grey strip whatever the filter said, so a filter
    that picked out a small cluster showed nothing until the user zoomed onto it. Each thin
    block is now a column of its own ``fill_of``: the share's shade while a filter is on, the
    label colour while it is off."""
    from PyQt6.QtGui import QColor

    from detail.gui import plot_theme

    widget = NodeIcicle()
    widget.set_view(_ChainView())
    widget.setMinimumWidth(0)
    widget.resize(40, 200)
    widget.show()
    qapp.processEvents()
    widget.grab()
    rect, node = next((r, n) for r, n in widget._blocks if n in widget._tiny)
    assert rect.width() < node_icicle.TINY_BLOCK_PX
    shares = {n: 0.0 for n in range(11)}
    shares[node] = 1.0
    widget.set_shades(shares)
    qapp.processEvents()
    assert (
        _column_colour(widget, rect).name() == plot_theme.match_colour(1.0).name()
    ), "a thin block wears its share's shade"
    widget.set_shades(None)
    label = QColor("#c2913a")
    widget.set_labels({node: (label, True)}, {node: "rearing"})
    qapp.processEvents()
    assert _column_colour(widget, rect).name() == label.name(), "off: its label colour"
    widget.close()


class _OneSliverView(_ChainView):
    """A root with one big child and one far too thin to lay out: 1000 s beside 0.1 s."""

    def __init__(self) -> None:
        self._children = {2: (0, 1)}
        self._times = np.asarray([1000.0, 0.1, 1000.1]) * 1e9
        self.tree = _Tree(3)
        self.root = 2


def test_the_strip_only_covers_children_too_thin_to_lay_out(qapp):
    widget = NodeIcicle()
    widget.set_view(_OneSliverView())
    widget.resize(600, 200)
    widget.show()
    qapp.processEvents()
    widget.grab()
    laid_out = {node for _rect, node in widget._blocks}
    assert 1 not in laid_out, "0.06 px: under MIN_BLOCK_PX / 4, not laid out"
    assert widget._remainders, "what no column covers is the strip"
    widget.close()


def test_up_goes_to_the_block_drawn_above(icicle):
    """A piece's parent on screen is the block whose row it is drawn in, not the chain node
    it hangs off in the true tree, whose own opened row also contains it."""
    for piece in icicle.shown_children(10):
        assert icicle.shown_parent(piece) == 10
    assert icicle.shown_parent(4) == 6
    assert icicle.shown_parent(10) is None


class _DeepView(_ChainView):
    """``levels`` real splits in a row (60/40 each, never a peel): one level per row."""

    def __init__(self, levels: int = 12) -> None:
        n_leaves = levels + 1
        self._children = {}
        times = [0.0] * (2 * n_leaves - 1)
        times[0] = 1.0
        core = 0
        for k in range(levels):
            leaf = k + 1
            node = n_leaves + k
            # leaf holds 2/3 of the core so far: a 60/40 split, never a peel
            times[leaf] = times[core] * 2 / 3
            times[node] = times[core] + times[leaf]
            self._children[node] = (core, leaf)
            core = node
        self._times = np.asarray(times) * 1e9
        self.tree = _Tree(len(times))
        self.root = core


def _deep(qapp, height: int, width: int = 400):
    widget = NodeIcicle()
    widget.set_view(_DeepView())
    widget.resize(width, height)
    widget.show()
    qapp.processEvents()
    widget.grab()
    return widget


def _rows_drawn(widget) -> int:
    return len({round(rect.y()) for rect, _node in widget._blocks})


def test_a_taller_pane_shows_more_levels_not_taller_rows(qapp):
    """Dragging the tree taller widens the viewing window; it never stretches the rows."""
    short, tall = _deep(qapp, 150), _deep(qapp, 400)
    assert short._row_height() == tall._row_height() == row_px()
    assert _rows_drawn(tall) > _rows_drawn(short)
    short.close()
    tall.close()


def test_a_deep_tree_scrolls_down_its_levels(qapp):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent

    widget = _deep(qapp, 150)
    assert widget.scroll_range() > 0, "the fixture must be deeper than the pane"
    event = QWheelEvent(
        QPointF(50, 50),
        QPointF(50, 50),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ShiftModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    widget.wheelEvent(event)
    assert widget.scroll_y > 0, "Shift+wheel scrolls down"
    assert widget.span == (0.0, 1.0), "and does not zoom"
    widget.grab()
    rect, node = widget._blocks[len(widget._blocks) // 2]
    assert (
        widget.node_at(*_inside(rect)) == node
    ), "a click after scrolling hits its block"
    widget.close()


def test_the_scrollbar_follows_the_tree(qapp):
    pane = TreePane()
    pane.icicle.set_view(_DeepView())
    pane.resize(400, 150)
    pane.show()
    qapp.processEvents()
    assert pane.scrollbar.isEnabled() and pane.scrollbar.maximum() > 0
    pane.scrollbar.setValue(pane.scrollbar.maximum())
    assert pane.icicle.scroll_y == pane.icicle.scroll_range()
    pane.close()


def test_a_wider_pane_shows_more_of_the_run_at_the_same_zoom(qapp):
    widget = _deep(qapp, 200, width=400)
    widget.zoom_by(0.25, anchor_x=0)
    scale = widget._full()
    narrow = widget.span[1] - widget.span[0]
    widget.resize(600, 200)
    qapp.processEvents()
    assert widget._full() == pytest.approx(
        scale
    ), "the zoom is a scale, kept across a resize"
    assert widget.span[1] - widget.span[0] > narrow
    widget.resize(4000, 200)
    qapp.processEvents()
    assert widget.span == (0.0, 1.0), "once everything fits, it fits"
    widget.close()


def test_the_tree_claims_the_wheel_inside_a_scrolling_page(icicle):
    """Unclaimed, a wheel over the tree scrolls the page instead of zooming it."""
    from detail.gui.scroll import OWNS_WHEEL

    assert icicle.property(OWNS_WHEEL)


def _inside(rect):
    point = QPointF(rect.center())
    return point.x(), point.y()


def _narrow(qapp):
    """The chain at a width where the shed pieces are hairlines."""
    view = _ChainView()
    widget = NodeIcicle()
    widget.set_view(view)
    widget.setMinimumWidth(0)
    widget.resize(40, 200)
    widget.show()
    qapp.processEvents()
    widget.grab()
    return widget


def test_zooming_in_turns_a_hairline_into_a_block(qapp):
    """The reason the tree zooms: a sliver too narrow to paint becomes one you can read."""
    widget = _narrow(qapp)
    assert 0 in widget._tiny
    widget.zoom_by(0.1, anchor_x=widget.width())  # the right edge, where the pieces sit
    widget.grab()
    assert 0 not in widget._tiny and any(node == 0 for _r, node in widget._blocks)
    assert widget.is_zoomed() and widget.span[1] == pytest.approx(1.0)
    widget.close()


def test_zoom_keeps_the_point_under_the_pointer(icicle):
    left, right = icicle.span
    x = icicle.width() * 0.25
    before = left + 0.25 * (right - left)
    icicle.zoom_by(0.5, anchor_x=x)
    left, right = icicle.span
    assert left + 0.25 * (right - left) == pytest.approx(before)


def test_a_drag_pans_and_selects_nothing_a_click_selects(icicle, qapp):
    from PyQt6.QtCore import QPoint
    from PyQt6.QtTest import QTest
    from PyQt6.QtCore import Qt

    picked = []
    icicle.node_picked.connect(picked.append)
    icicle.zoom_by(0.5, anchor_x=0)
    start = icicle.span
    y = int(icicle._row_height() * 1.5)
    QTest.mousePress(icicle, Qt.MouseButton.LeftButton, pos=QPoint(200, y))
    QTest.mouseMove(icicle, QPoint(100, y))
    QTest.mouseRelease(icicle, Qt.MouseButton.LeftButton, pos=QPoint(100, y))
    assert icicle.span[0] > start[0], "dragging left moves the view right"
    assert picked == [], "a drag is not a click"
    QTest.mouseClick(icicle, Qt.MouseButton.LeftButton, pos=QPoint(100, y))
    assert len(picked) == 1


def test_escape_and_fit_show_the_whole_tree(icicle):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    icicle.zoom_by(0.2, anchor_x=100)
    icicle.zoom_to(6)
    assert icicle.is_zoomed()
    QTest.keyClick(icicle, Qt.Key.Key_Escape)
    assert not icicle.is_zoomed() and icicle.zoom_root == 10


def test_zooming_in_never_takes_the_vertical_scroll_away(qapp):
    """Reported: zooming made the scrollbar vanish, so the deeper levels could not be reached.
    The content height counts every level at the current zoom, so zooming in only adds depth.
    """
    pane = TreePane()
    pane.icicle.set_view(_DeepView())
    pane.resize(400, 150)
    pane.show()
    qapp.processEvents()
    before = pane.icicle.scroll_range()
    for factor in (0.5, 0.2, 0.05):
        # The right edge: the shallow side, where a count over the visible part shrank.
        pane.icicle.zoom_by(factor, anchor_x=pane.icicle.width())
        qapp.processEvents()
        assert pane.icicle.scroll_range() >= before
        assert pane.scrollbar.isVisible() and pane.scrollbar.isEnabled()
    pane.close()


def test_the_tree_does_not_move_when_its_overview_appears(qapp):
    """The overview bar is hidden until a zoom, with its room kept: hidden without it, the tree
    would drop under the pointer on the first wheel notch."""
    pane = TreePane()
    pane.icicle.set_view(_DeepView())
    pane.resize(400, 200)
    pane.show()
    qapp.processEvents()
    assert not pane.overview.isVisible()
    before = pane.icicle.geometry()
    pane.icicle.zoom_by(0.5)
    qapp.processEvents()
    assert pane.overview.isVisible()
    assert pane.icicle.geometry() == before
    pane.close()


# --- shading by a filter's matching share (§22) ---------------------------------------------


def _pixel(widget, node, dy):
    """The colour ``dy`` px below the top of ``node``'s block, clear of its text on the left."""
    image = widget.grab().toImage()
    ratio = image.devicePixelRatio()
    rect = next(r for r, n in widget._layout() if n == node)
    return image.pixelColor(
        int((rect.right() - 6) * ratio), int((rect.top() + dy) * ratio)
    )


def test_a_shade_is_the_fixed_ramp_and_the_label_moves_to_a_top_strip(icicle, qapp):
    from PyQt6.QtGui import QColor

    from detail.gui import plot_theme

    label = QColor("#c2913a")
    icicle.set_labels({4: (label, True)}, {4: "rearing"})
    qapp.processEvents()
    assert _pixel(icicle, 4, 10).name() == label.name(), "unfiltered: the label colour"
    shares = {node: 0.5 for node in range(11)}
    shares[5] = 0.0
    icicle.set_shades(shares)
    qapp.processEvents()
    ramp = plot_theme.match_colour(0.5)
    assert _pixel(icicle, 4, 12).name() == ramp.name(), "filtered: the share's shade"
    assert _pixel(icicle, 4, 1).name() == label.name(), "the label is a strip on top"
    assert _pixel(icicle, 5, 12).name() == plot_theme.match_colour(0.0).name()
    assert node_icicle.STRIP_PX == 3
    icicle.set_shades(None)
    qapp.processEvents()
    assert _pixel(icicle, 4, 12).name() == label.name()


def test_the_scale_is_fixed_not_stretched_to_the_clusters_spread(icicle):
    """An even 43–55% spread stays a narrow band of mid shades; stretched to its own min–max it
    would fill the ramp and read as a strong pattern."""
    from detail.gui import plot_theme

    icicle.set_shades({4: 0.43, 5: 0.55})
    low, high = icicle.fill_of(4)[0], icicle.fill_of(5)[0]
    assert low.name() == plot_theme.match_colour(0.43).name()
    assert high.name() == plot_theme.match_colour(0.55).name()
    assert low.name() != plot_theme.match_colour(0.0).name()
    assert high.name() != plot_theme.match_colour(1.0).name()


def test_a_node_with_nothing_to_count_is_drawn_as_none(icicle):
    from detail.gui import plot_theme

    icicle.set_shades({4: float("nan")})
    assert icicle.fill_of(4)[0].name() == plot_theme.match_colour(0.0).name()
    assert icicle.fill_of(5)[0].name() == plot_theme.match_colour(0.0).name()
    assert "no positioned clips" in icicle._describe(4)


def test_the_hover_says_the_share_that_matches(icicle):
    icicle.set_shades({4: 0.52})
    assert "52% of its clips match" in icicle._describe(4)
    icicle.set_shades(None)
    assert "match" not in icicle._describe(4)


def test_the_menu_has_label_remove_and_zoom_only(icicle):
    texts = [a.text() for a in icicle.context_menu(4).actions() if a.text()]
    assert texts == ["Label…", "Remove label", "Zoom to this"]


def _pixel_at(widget, x: float, y: float):
    image = widget.grab().toImage()
    ratio = image.devicePixelRatio()
    return image.pixelColor(int(x * ratio), int(y * ratio))


def _adjacent_thin_pair(widget):
    """Two thin blocks side by side in one row: the left one's slot ends where the right's starts."""
    slots = {n: widget._slots[n] for n in widget._tiny if n in widget._slots}
    for a, sa in slots.items():
        for b, sb in slots.items():
            if (
                a != b
                and sa.top() == sb.top()
                and sa.right() == sb.left()
                and sa.width() >= 2
            ):
                return a, b
    raise AssertionError("the fixture must lay out two thin blocks side by side")


def _thin_widget(qapp):
    widget = NodeIcicle()
    widget.set_view(_ChainView())
    widget.setMinimumWidth(0)
    widget.resize(40, 200)
    widget.show()
    qapp.processEvents()
    widget.grab()
    return widget


@pytest.mark.parametrize("shaded", [False, True], ids=["labels", "filter"])
def test_two_pale_thin_blocks_side_by_side_have_a_dark_gap_between(qapp, shaded):
    """A pale cluster beside a pale cluster read as one block: the gap between them is the dark
    neutral, which stands apart from the light end of the ramp in either theme."""
    widget = _thin_widget(qapp)
    if shaded:
        widget.set_shades({n: 0.02 for n in range(11)})  # every block at the pale end
        qapp.processEvents()
    left, right = _adjacent_thin_pair(widget)
    slot = widget._slots[left]
    y = slot.top() + slot.height() / 2
    assert _pixel_at(widget, slot.right() - 1, y).name() == node_icicle.GAP_COLOUR
    assert _pixel_at(widget, slot.left(), y).name() == widget.fill_of(left)[0].name()
    right_slot = widget._slots[right]
    assert (
        _pixel_at(widget, right_slot.left(), y).name()
        == widget.fill_of(right)[0].name()
    )
    widget.close()


def test_the_gap_guard_would_catch_blocks_drawn_edge_to_edge(qapp, monkeypatch):
    """Guard the guard: with each block filling its whole slot, the sampled pixel is colour."""
    monkeypatch.setattr(node_icicle, "_block_of", lambda slot: slot)
    widget = _thin_widget(qapp)
    left, _right = _adjacent_thin_pair(widget)
    slot = widget._slots[left]
    pixel = _pixel_at(widget, slot.right() - 1, slot.top() + slot.height() / 2)
    assert pixel.name() != node_icicle.GAP_COLOUR
    widget.close()


def test_a_slot_under_two_pixels_is_one_pixel_of_colour():
    slot = node_icicle._snap(10.2, 0.0, 1.1, 20.0)
    assert slot.width() == 1
    assert node_icicle._block_of(slot).width() == 1
    two = node_icicle._snap(10.0, 0.0, 2.0, 20.0)
    assert node_icicle._block_of(two).width() == 1, "2 px keeps 1 px of colour"


def test_the_selected_block_wears_a_dark_and_white_ring(icicle, qapp):
    """2 px dark outside 1 px white, inside the block's edge: one of the two stands out on any
    fill, pale or dark."""
    rect, node = max(icicle._blocks, key=lambda pair: pair[0].width())
    icicle.set_selected(node)
    qapp.processEvents()
    y = rect.top() + rect.height() / 2
    assert _pixel_at(icicle, rect.left(), y).name() == node_icicle.GAP_COLOUR
    assert _pixel_at(icicle, rect.left() + 1, y).name() == node_icicle.GAP_COLOUR
    assert _pixel_at(icicle, rect.left() + 2, y).name() == node_icicle.RING_INNER_COLOUR
    # Inside the ring is the fill. Probed at the right end: the label is left-aligned from
    # left + 4 at this same height, so a probe there reads a glyph's antialiased edge wherever
    # the platform's font puts one on that pixel (Linux CI read #ccc9b7 against #cccccc).
    assert (
        _pixel_at(icicle, rect.right() - 6, y).name() == icicle.fill_of(node)[0].name()
    )
    icicle.set_selected(None)
    qapp.processEvents()
    assert (
        _pixel_at(icicle, rect.left() + 2, y).name() == icicle.fill_of(node)[0].name()
    )


def test_a_selected_thin_column_is_ringed_around_it(qapp):
    """A column a few pixels wide has no inside for a ring, so it is drawn round the outside."""
    widget = _thin_widget(qapp)
    rect, node = next((r, n) for r, n in widget._blocks if n in widget._tiny)
    widget.set_selected(node)
    qapp.processEvents()
    y = rect.top() + rect.height() / 2
    assert _pixel_at(widget, rect.left() - 1, y).name() == node_icicle.RING_INNER_COLOUR
    assert _pixel_at(widget, rect.left() - 2, y).name() == node_icicle.GAP_COLOUR
    assert _pixel_at(widget, rect.left() - 3, y).name() == node_icicle.GAP_COLOUR
    widget.close()


# --- a block's words never print over each other ------------------------------------- #
def _overlaps(boxes) -> list:
    rects = [box for box, _text, _kind in boxes]
    return [
        (a, b)
        for i, a in enumerate(rects)
        for b in rects[i + 1 :]
        if a.intersected(b).height() > 0 and a.intersected(b).width() > 0
    ]


def test_a_blocks_two_lines_never_overlap_at_any_row_height(icicle):
    """**The class: two lines placed at the cell's top and bottom with no check that the cell
    holds both.** The tree drew the name top-aligned and the share bottom-aligned whenever a row
    was within 3px of ``row_px`` — less than the two lines' heights — so on a laptop, where the
    rows are fitted shorter, "#1397" and "74.9%" printed on top of each other. Every height a
    row can take, from the shortest fitted row to a full one and past it, is checked here.
    """
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QFont, QFontMetricsF

    from detail.gui.node_icicle import cell_text, min_row_px

    small = QFont(icicle.font())
    small.setPointSizeF(max(small.pointSizeF() - 1.5, 7.0))
    big, little = QFontMetricsF(icicle.font()), QFontMetricsF(small)
    kinds = set()
    for height in range(min_row_px() - 2, row_px() + 24):
        cell = QRectF(10, 40, 160, height)
        boxes = cell_text(cell, "#1397", 74.9, big, little)
        assert not _overlaps(boxes), height
        for box, _text, kind in boxes:
            assert cell.contains(box), (height, kind)
            want = big.height() if kind == "name" else little.height()
            assert box.height() == want or kind == "line", (height, kind)
        kinds.add(tuple(k for _b, _t, k in boxes))
    assert ("name", "share") in kinds, "a tall enough row still gets two lines"
    assert ("line",) in kinds, "a short row gets one"
    one = cell_text(QRectF(0, 0, 160, min_row_px()), "#1397", 74.9, big, little)
    assert [t for _b, t, _k in one] == ["#1397  74.9%"]


def test_the_tree_paints_its_words_where_cell_text_puts_them(icicle, monkeypatch):
    calls = []
    real = node_icicle.cell_text

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(node_icicle, "cell_text", spy)
    icicle.resize(600, 200)
    icicle.show()
    icicle.grab()
    assert calls, "paintEvent places no text through cell_text"


def test_a_mark_on_an_undrawn_node_lands_on_its_topmost_drawn_descendants():
    """`_drawn_as`: a node too thin to draw is shown by its nearest drawn descendants.

    Covered only by a pose-clustering workspace test until pose clustering was removed, so it is
    asserted directly: a drawn node is itself; an undrawn one resolves to the first drawn block on
    each path down, never to a block below one that is already drawn.
    """
    from types import SimpleNamespace

    tree = {6: [4, 5], 5: [2, 3], 4: [0, 1], 3: [], 2: [], 1: [], 0: []}
    holder = SimpleNamespace(_view=SimpleNamespace(children=lambda n: tree[n]))
    drawn = {4: "rect4", 2: "rect2", 3: "rect3", 0: "rect0"}

    assert NodeIcicle._drawn_as(holder, 4, drawn) == [4]
    assert sorted(NodeIcicle._drawn_as(holder, 6, drawn)) == [2, 3, 4]  # not 0, under 4
    assert sorted(NodeIcicle._drawn_as(holder, 5, drawn)) == [2, 3]
    assert NodeIcicle._drawn_as(holder, 1, drawn) == []
    assert NodeIcicle._drawn_as(SimpleNamespace(_view=None), 6, drawn) == []
