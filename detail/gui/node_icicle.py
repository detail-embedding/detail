"""The node tree on INTERPRET: an icicle, one block per node, always coloured by labels (§22).

Every node of a fit's merge tree is something a user can click, because every node is a cluster
and the cluster is what they reason about. An icicle makes that literal: the root is the top row,
each row down is one merge level, and a block's width is the node's member time, so a click
target is exactly as big as the thing it stands for.

**Coloured by labels and nothing else.** Solid where a label was put on that node; the same colour
pale where a parent's label covers it; neutral where nothing does. Every node lists every clip,
so no block is marked as lacking them. A dot marks a node already looked at in this session.

**A cluster open in a window wears the window's number** (``set_window_marks``): a dark disc at
its block's top right, the number in it, as the window's title has it ("① Behavior #1321 ·
climb"). Drawn after the selection and inside the block's corner, so it never covers
INTERPRET's own selection ring; a cluster drawn as several pieces (a chain opened as a peel)
wears it on its widest piece, and one too narrow for the disc says it in its tooltip.

**A cluster typed into Find clusters wears that id's colour** (``set_found``, from
`gui/cluster_find.py`): a 3 px band just *outside* its block, edged in the theme's ink, so it
reads on any fill and a block one pixel wide is still a band you can see. Outside the block, so
it never meets AUDIT's inset ring or the selection ring on the block's edge; a chain cluster
wears it on each piece it is drawn as, as the selection does. Never inside: a side at the
widget's edge goes without, and the block zoomed onto wears none (``_paint_found``).

**Zooming**: the wheel scales the visible width around the pointer and a drag pans; a block's
right-click menu re-roots the picture on it (**Zoom to this**), because a real fit has hundreds
of leaves and a block one pixel wide is not a target. Double-click labels a block. Nothing moves when a
label lands, so labelling never costs the user their place.

Hand-drawn with ``QPainter``, after ``EthogramStrip`` and ``SpatialMap``: this is a picture of a
tree, not a plot of a quantity, so it wants no axes.
"""

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QPainter, QPen
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QScrollBar,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from detail.gui import plot_theme
from detail.gui.scroll import OWNS_WHEEL
from detail.gui.sizing import lines
from detail.gui.zoom import DRAG_PX, OverviewBar, ZoomWindow, zoom_pinch, zoom_wheel

#: A safety bound on the levels laid out; the tree scrolls vertically, so this is never the
#: limit a user meets.
VISIBLE_ROWS = 200


def row_px() -> int:
    """Every row's height. Fixed, so a taller pane shows **more levels** rather than taller rows.

    A function, not a constant: ``lines`` measures the application font, and evaluated at import
    time (before a ``QApplication`` exists) it took the whole process down with it.
    """
    return lines(2)


def min_row_px() -> int:
    """The shortest a row gets when rows shrink to show every level (``fit_rows``): one line
    of the small font, so a block still carries its name. Shorter than this, the tree scrolls.
    """
    return lines(0.8)


def cell_text(
    cell: QRectF, name: str, share: float, name_metrics, small_metrics
) -> list[tuple[QRectF, str, str]]:
    """Where a tree block's words go: ``[(box, text, kind)]`` — ``kind`` is ``"name"`` (the
    block's font), ``"share"`` or ``"line"`` (the small font) — each box exactly one
    line of its font high, every one inside ``cell`` and none overlapping another.

    **Two lines** — the name, and the share under it in the small font — only when the cell
    holds both lines' full heights. Otherwise **one small line**, ``#1397  74.9%``, cut with
    "…"; nothing when even that does not fit. Two lines used to be drawn whenever a row was
    within 3px of ``row_px``, which is shorter than the two lines' heights: at a laptop's fitted
    row height the name (top-aligned) and the share (bottom-aligned) printed on top of each
    other (user screenshot, dark mode, 2026-11-09).
    """
    high, low = name_metrics.height(), small_metrics.height()
    inner = cell.adjusted(5, 1, -5, -1)
    if inner.height() >= high + low and inner.width() > 0:
        top = QRectF(inner.left(), inner.top(), inner.width(), high)
        under = QRectF(inner.left(), inner.bottom() - low, inner.width(), low)
        return [
            (
                top,
                name_metrics.elidedText(
                    name, Qt.TextElideMode.ElideRight, inner.width()
                ),
                "name",
            ),
            (under, f"{share:.1f}%", "share"),
        ]
    line = cell.adjusted(4, 0, -4, 0)
    if low > cell.height() + 2 or line.width() <= 0:
        return []
    # Centred on the cell; a line up to 2px taller than its cell is clipped at its leading.
    box = QRectF(
        line.left(), cell.center().y() - low / 2, line.width(), low
    ).intersected(cell)
    text = small_metrics.elidedText(
        f"{name}  {share:.1f}%", Qt.TextElideMode.ElideRight, line.width()
    )
    return [(box, text, "line")]


#: Narrower than this and a block is not drawn on its own; its parent's block already covers it.
MIN_BLOCK_PX = 2.0
#: Wider than this and a block carries its label and share as text.
TEXT_BLOCK_PX = 46.0
#: A merge whose smaller half holds less than this share of it is a *peel*, not a split: it gets
#: no row of its own, and the pieces it is made of stand in for it in its parent's row.
SPLIT_SHARE = 0.2
#: Narrower than this and a block is a bare column: no text, no row of children under it.
TINY_BLOCK_PX = 6.0
#: The 1 px gap between neighbouring blocks, and the outer half of the selection ring. One dark
#: neutral in both themes: it has to stand apart from the **light** end of the match ramp, which is
#: the low end in the light theme and the high end in the dark one, so a pale cluster beside a
#: pale cluster still reads as two.
GAP_COLOUR = "#1a1a1a"
#: The inner half of the selection ring: white beside the dark outer half, so one of the two
#: contrasts with whatever fill the block has.
RING_INNER_COLOUR = "#ffffff"
#: The shortest a window's number is drawn on its block: a row the tree has shrunk below this
#: says it in its tooltip instead.
MARK_MIN_PX = 9
#: How far in the wheel goes: the narrowest visible span, as a share of the zoom root.
MIN_SPAN = 1e-4


def format_time(ns: int) -> str:
    """``1 h 02 min``, ``12.4 min``, ``38 s``: member time at the precision the tree needs."""
    seconds = ns / 1e9
    if seconds >= 3600:
        return f"{int(seconds // 3600)} h {int(seconds % 3600 // 60):02d} min"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


#: The label colour's strip along a block's top while the blocks are shaded by a filter.
STRIP_PX = 3
#: How wide a Find clusters band is, outside the block (``set_found``).
FOUND_PX = 3
#: How wide the frame on a cluster a difference screen kept is, inside the block's edge
#: (``set_kept``). A Find clusters band is outside the block and the AUDIT ring is inset from
#: its edge, so the three do not sit on one another.
KEPT_PX = 3


def _frame(painter: QPainter, rect: QRectF, width: float, colour: QColor) -> None:
    """A ``width``-px band just inside ``rect``, as four fills: crisp at any width, where a
    pen's stroke straddles the edge and blurs half a pixel either side."""
    w = min(width, rect.width() / 2, rect.height() / 2)
    if w <= 0:
        return
    painter.fillRect(QRectF(rect.left(), rect.top(), rect.width(), w), colour)
    painter.fillRect(QRectF(rect.left(), rect.bottom() - w, rect.width(), w), colour)
    painter.fillRect(QRectF(rect.left(), rect.top(), w, rect.height()), colour)
    painter.fillRect(QRectF(rect.right() - w, rect.top(), w, rect.height()), colour)


class NodeIcicle(QWidget):
    """Click a block to select its node; double-click to label it; right-click for a menu:
    label it, remove its label, or zoom onto it.

    **Two fills, one at a time.** Unfiltered, a block is its label's colour (``set_labels``).
    With a filter on (``set_shades``), it is the share of its clips that match, on the fixed
    0–100% ramp (``plot_theme.match_colour``), and its label colour moves to a
    :data:`STRIP_PX` strip along its top — the label is still there, the shade says the filter.
    """

    node_picked = pyqtSignal(int)
    zoomed = pyqtSignal(int)
    #: Label this node, with an editor at this global position (double-click, or the menu).
    label_wanted = pyqtSignal(int, object)
    #: Remove this node's own label (the menu).
    unlabel_wanted = pyqtSignal(int)
    #: Open this node's clips in a window of their own (the menu).
    window_wanted = pyqtSignal(int)
    #: The content height, the scroll offset or the viewport changed: a scrollbar re-reads them.
    extent_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(row_px() * 3)
        self.setMinimumWidth(lines(16))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # The wheel zooms the tree. Inside a page that scrolls, an unclaimed wheel scrolls the
        # page out from under the gesture instead (`scroll.PageScroll`).
        self.setProperty(OWNS_WHEEL, True)
        #: Why the run's labels cannot change here (the page's), or ``""`` where they can.
        self._edit_reason = ""
        self._view = None
        self._times = np.zeros(0)
        self._root = 0
        self._selected: int | None = None
        self._fills: dict[int, tuple[QColor, bool]] = {}
        #: Each node's matching share while a filter is on (``None``: no filter); NaN where a
        #: node has nothing to count.
        self._shades: dict[int, float] | None = None
        self._diverging = False
        self._one_sided = False
        #: The clusters a ticked difference screen kept, each with the colour of its effect
        #: and the screen's sentence for the hover (``set_kept``).
        self._kept: dict[int, QColor] = {}
        self._kept_texts: dict[int, str] = {}
        self._shade_texts: dict[int, str] = {}
        self._names: dict[int, str] = {}
        self._outlines: dict[int, QColor] = {}
        self._marks: dict[int, str] = {}
        #: ``{node: window number}``: clusters open in a window of their own.
        self._window_marks: dict[int, int] = {}
        #: ``{node: colour}``: clusters listed in Find clusters, ringed in their colour.
        self._found: dict[int, QColor] = {}
        self._visited: set[int] = set()
        self._blocks: list[tuple[QRectF, int]] = []
        self._mark_rects: dict[int, QRectF] = {}
        #: ``{node: [drawn block, ...]}``: what each found cluster was ringed on at the last
        #: paint (its own block, or the pieces a chain cluster is drawn as).
        self._found_pieces: dict[int, list[int]] = {}
        self._remainders: list[QRectF] = []
        self._tiny: set[int] = set()
        #: Each laid-out node's whole slot: its block plus the gap after it.
        self._slots: dict[int, QRectF] = {}
        self._parents: dict[int, int] | None = None
        #: The horizontal view: the visible part of the zoom root, as fractions — the shared
        #: zoom (`gui/zoom.py`) that `TreePane`'s overview bar is bound to. A resize keeps the
        #: *scale* (`resizeEvent`), so a wider pane shows **more** of the run at the same zoom
        #: instead of stretching it.
        self.zoom = ZoomWindow(self)
        self.zoom.set_floor(MIN_SPAN)
        self.zoom.changed.connect(self._on_zoom)
        #: How far the rows are scrolled down, in pixels.
        self._scroll_y = 0.0
        #: Rows shrink (down to ``min_row_px``) so every level fits the pane's height, rather
        #: than staying ``row_px`` tall and scrolling (`set_fit_rows`).
        self._fit_rows = False
        self._depth_cache: tuple[tuple, int] | None = None
        self._press: tuple[float, float] | None = None
        self._press_view = ((0.0, 1.0), 1.0)
        self._dragging = False

    # --- inputs --------------------------------------------------------------------------

    def set_view(self, view) -> None:
        """Draw ``view``'s tree (a ``NodeView``), zoomed out to its root."""
        self._view = view
        self._parents: dict[int, int] | None = None
        self._times = np.asarray(view.node_times(), dtype=np.float64)
        self._depth_cache = None
        self._root = view.root
        self._scroll_y = 0.0
        self.zoom.blockSignals(True)
        self.zoom.fit()
        self.zoom.blockSignals(False)
        self.extent_changed.emit()
        self.update()

    def set_labels(
        self, fills: dict[int, tuple[QColor, bool]], names: dict[int, str]
    ) -> None:
        """``fills[node] = (colour, own)`` for every node a label covers; ``names`` for text."""
        self._fills = dict(fills)
        self._names = dict(names)
        self.update()

    def set_shades(
        self,
        shares: dict[int, float] | None,
        *,
        diverging: bool = False,
        one_sided: bool = False,
        texts: dict[int, str] | None = None,
    ) -> None:
        """Shade every block by ``shares[node]`` (0..1) on the fixed 0–100% ramp, or ``None`` to
        go back to the label colours. A node missing or NaN is drawn as having none.

        ``diverging`` reads the values as effects on the −1..+1 axis instead
        (``plot_theme.effect_colour``: zero and NaN neutral), and ``texts[node]`` is then the
        hover's line for a block in place of its share — the screen's sentence for a cluster.
        ``one_sided`` reads them as sizes on 0..1 with no direction (a profile screen,
        ``plot_theme.size_colour``): a ramp with no hue, so neither pole's colour is borrowed
        for a difference that has no side.
        """
        self._shades = (
            None if shares is None else {int(k): v for k, v in shares.items()}
        )
        self._diverging = bool(diverging) and shares is not None
        self._one_sided = bool(one_sided) and self._diverging
        self._shade_texts = (
            {}
            if texts is None or shares is None
            else {int(k): v for k, v in texts.items()}
        )
        self.update()

    def shades(self) -> dict[int, float] | None:
        return None if self._shades is None else dict(self._shades)

    def fill_of(self, node: int) -> tuple[QColor, QColor | None]:
        """``(fill, strip)`` for ``node``'s block: what the paint step draws. ``strip`` is the
        label colour along the top while shading is on, else ``None``."""
        theme = plot_theme.current()
        neutral = QColor(theme.neutral)
        colour, own = self._fills.get(node, (neutral, True))
        label = QColor(colour)
        if node in self._fills and not own:
            label.setAlpha(95)
        if self._shades is None:
            return label, None
        share = self._shades.get(int(node), float("nan"))
        strip = label if node in self._fills else None
        if self._diverging:
            value = float("nan") if share is None else float(share)
            if self._one_sided:
                return plot_theme.size_colour(value, theme), strip
            return plot_theme.effect_colour(value, theme), strip
        share = 0.0 if share is None or share != share else float(share)
        return plot_theme.match_colour(share, theme), strip

    def set_outlines(
        self, outlines: dict[int, QColor], marks: dict[int, str] | None = None
    ) -> None:
        """``outlines[node]``: an inset ring in that colour — that AUDIT has judged clips of the
        cluster — drawn inside the block so the selection stroke on its edge still reads over it.
        ``marks[node]`` is a glyph on a white disc in the corner: a ring's hue can be a label's
        hue too, and a ring the colour of its fill does not read."""
        self._outlines = dict(outlines)
        self._marks = dict(marks or {})
        self.update()

    def outlines(self) -> dict[int, QColor]:
        return dict(self._outlines)

    def set_window_marks(self, marks: dict[int, int]) -> None:
        """``marks[node]``: the number of the window that cluster is open in."""
        self._window_marks = {int(k): int(v) for k, v in marks.items()}
        self.update()

    def window_marks(self) -> dict[int, int]:
        return dict(self._window_marks)

    def set_kept(
        self, kept: dict[int, QColor], texts: dict[int, str] | None = None
    ) -> None:
        """``kept[node]``: a cluster the ticked difference screen kept, framed in that colour
        (the colour of its effect) on the block's own edge, with ``texts[node]`` added to its
        hover. For the Filters tab, where the fill is the share of clips that match: the
        frame says which clusters the screen picked without taking the fill (§22). A block
        too narrow to hold a frame is the frame's colour, which is the same statement.
        """
        self._kept = {int(k): QColor(v) for k, v in kept.items()}
        self._kept_texts = {int(k): str(v) for k, v in (texts or {}).items()}
        self.update()

    def kept(self) -> dict[int, QColor]:
        return dict(self._kept)

    def set_found(self, found: dict[int, QColor]) -> None:
        """``found[node]``: a cluster listed in Find clusters, ringed in that colour."""
        self._found = {int(k): QColor(v) for k, v in found.items()}
        self.update()

    def found(self) -> dict[int, QColor]:
        return dict(self._found)

    def set_edit_reason(self, reason: str) -> None:
        """Why the run's labels cannot change (``""``: they can) — said on the right-click
        menu's **Label…** and **Remove label**, greyed."""
        self._edit_reason = str(reason)

    def window_mark_rects(self) -> dict[int, QRectF]:
        """``{node: disc rect}`` for each window mark drawn at the last paint."""
        return dict(self._mark_rects)

    def set_selected(self, node: int | None) -> None:
        self._selected = None if node is None else int(node)
        if node is not None:
            self._visited.add(int(node))
        self.update()

    def zoom_to(self, node: int) -> None:
        """Re-root the picture on ``node`` (the root zooms all the way out)."""
        if self._view is None:
            return
        self._root = int(node)
        self._scroll_y = 0.0
        self.zoom.blockSignals(True)
        self.zoom.fit()
        self.zoom.blockSignals(False)
        self.zoomed.emit(self._root)
        self.extent_changed.emit()
        self.update()

    def view_state(self) -> tuple[int, tuple[float, float], float]:
        """What the picture shows: ``(zoom root, visible span, level scroll)`` — for
        :meth:`restore_view`."""
        return self._root, self.zoom.view, self._scroll_y

    def restore_view(self, state: tuple[int, tuple[float, float], float]) -> None:
        """Show again what :meth:`view_state` returned (Find clusters' **Clear**)."""
        root, (left, right), scroll_y = state
        self.zoom_to(root)
        self.zoom.set_view(left, right)
        self.set_scroll_y(scroll_y)

    def fit(self) -> None:
        """Back to the whole tree: the run's root, nothing zoomed."""
        if self._view is not None:
            self.zoom_to(self._view.root)

    @property
    def span(self) -> tuple[float, float]:
        """The zoom root's visible part, as fractions ``(left, right)`` of its width."""
        return self.zoom.view

    def is_zoomed(self) -> bool:
        return self.zoom.is_zoomed() or (
            self._view is not None and self._root != self._view.root
        )

    def zoom_by(self, factor: float, anchor_x: float | None = None) -> None:
        """Scale the visible span by ``factor`` (< 1 zooms in) around ``anchor_x`` in pixels.

        The point under the anchor stays under it, which is what makes a wheel zoom feel like
        moving towards something rather than the picture sliding away.
        """
        self.zoom.zoom_by(factor, self._share(anchor_x))

    def _share(self, x: float | None) -> float:
        """How far across the widget pixel ``x`` is, 0 to 1; the middle for ``None``."""
        width = max(float(self.width()), 1.0)
        return 0.5 if x is None else x / width

    def _on_zoom(self) -> None:
        self.zoomed.emit(self._root)
        self.extent_changed.emit()
        self.update()

    def _full(self) -> float:
        """Pixels the whole zoom root spans right now."""
        return max(float(self.width()), 1.0) / self.zoom.size

    # --- the vertical viewport -----------------------------------------------------------

    def content_height(self) -> int:
        """Every laid-out level at the current row height."""
        return int(self.levels() * self._row_height())

    def levels(self) -> int:
        """How many rows the tree has at the current zoom and width (cached per view)."""
        if self._view is None or not len(self._times):
            return 0
        x0, full = self._origin()
        key = (id(self._view), self._root, x0, full)
        if self._depth_cache is None or self._depth_cache[0] != key:
            self._depth_cache = (key, self._depth(self._root, x0, full, 0))
        return self._depth_cache[1]

    def whole_tree_levels(self, width: float | None = None) -> int:
        """How many rows the **whole** tree (the run's root, nothing zoomed) has at ``width``
        pixels wide (the pane's own width for ``None``) — what the page sizes the tree to.
        """
        if self._view is None or not len(self._times):
            return 0
        full = max(float(self.width() if width is None else width), 1.0)
        return self._depth(self._view.root, 0.0, full, 0)

    def set_fit_rows(self, fit: bool) -> None:
        """Shrink rows to show every level in the pane's height (down to ``min_row_px``), or
        keep every row ``row_px`` tall and scroll. The page fits rows until the user drags the
        tree's height themselves."""
        fit = bool(fit)
        if fit == self._fit_rows:
            return
        self._fit_rows = fit
        self._scroll_y = 0.0
        self.extent_changed.emit()
        self.update()

    def fits_rows(self) -> bool:
        return self._fit_rows

    def scroll_range(self) -> int:
        return max(0, self.content_height() - self.height())

    @property
    def scroll_y(self) -> int:
        return int(self._scroll_y)

    def set_scroll_y(self, value: float) -> None:
        clamped = float(min(max(value, 0), self.scroll_range()))
        if clamped != self._scroll_y:
            self._scroll_y = clamped
            self.extent_changed.emit()
            self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """A wider pane shows more at the same scale; a taller one shows more levels."""
        super().resizeEvent(event)
        old = event.oldSize().width()
        if self.zoom.is_zoomed() and old > 0:
            left = self.zoom.view[0]
            px_per_root = old / self.zoom.size
            self.zoom.set_view(left, left + self.width() / px_per_root)
        self.set_scroll_y(self._scroll_y)
        # A taller or shorter tree changes how far its levels scroll even when nothing above
        # moved, so the scrollbar re-reads it on every resize, not only on a zoom.
        self.extent_changed.emit()

    @property
    def zoom_root(self) -> int:
        return self._root

    # --- drawing -------------------------------------------------------------------------

    def _origin(self) -> tuple[float, float]:
        """``(x, w)`` of the zoom root in pixels: wider than the widget, shifted, when zoomed."""
        full = self._full()
        return -self.zoom.view[0] * full, full

    def _layout(self) -> list[tuple[QRectF, int]]:
        """Each drawn block's rect, in whole pixels; also fills ``_tiny``, ``_slots`` and
        ``_remainders`` as a side effect.

        **Every block is followed by a 1 px gap**, to its right and below it, drawn in
        :data:`GAP_COLOUR` (``_slots`` holds block plus gap). Edges are rounded to whole pixels
        first, so a slot 2 px wide is 1 px of colour and 1 px of gap, and a pale cluster beside a
        pale cluster reads as two. A slot under 2 px has no room for both: it is drawn as a 1 px
        column of colour with no gap of its own (the gap after its left neighbour still sits
        before it, where that neighbour had the room).

        A child narrower than ``TINY_BLOCK_PX`` **on screen** is drawn as a bare column filled
        with its own colour (``fill_of``: its share's shade while a filter is on), with no text,
        and its subtree is not laid out. They sit at the end of the row (largest first) and each
        keeps its slot, so every one of them is still a click target. Zooming in widens them past
        the threshold, and they become blocks with their own rows under them. Blocks wholly off
        screen are not laid out at all.

        They were once painted as **one** faint grey strip, never shaded, so a filter that
        picked out a small cluster showed nothing until the user zoomed onto it. The strip
        (``_remainders``) is now only what is under the children too thin to lay out at all
        (under ``MIN_BLOCK_PX / 4``), which no column covers.
        """
        self._remainders = []
        self._tiny = set()
        self._slots = {}
        if self._view is None or not len(self._times):
            return []
        width, height = float(self.width()), float(self.height())
        row = self._row_height()
        out: list[tuple[QRectF, int]] = []
        x0, full = self._origin()
        stack = [(self._root, x0, full, 0)]
        while stack:
            node, x, w, depth = stack.pop()
            y = depth * row - self._scroll_y
            if depth >= VISIBLE_ROWS or x + w < 0 or x > width or y > height:
                continue
            tiny = w < TINY_BLOCK_PX
            if y + row > 0:  # scrolled above the viewport: laid out, not drawn
                slot = _snap(x, y, w, row)
                self._slots[node] = slot
                out.append((_block_of(slot), node))
            if tiny:
                self._tiny.add(node)
                continue
            children = self.shown_children(node)
            total = sum(self._times[c] for c in children) or self._times[node]
            cursor = x
            tail_from = None
            for child in children:
                share = self._times[child] / total if total else 1.0 / len(children)
                if w * share >= MIN_BLOCK_PX / 4:
                    stack.append((child, cursor, w * share, depth + 1))
                elif tail_from is None:
                    tail_from = cursor
                cursor += w * share
            if tail_from is not None and depth + 1 < VISIBLE_ROWS:
                self._remainders.append(
                    _block_of(_snap(tail_from, y + row, x + w - tail_from, row))
                )
        return out

    def _row_height(self) -> float:
        """``row_px``, never taller: dragging the pane taller shows more levels, not taller rows.

        Rows used to share the pane's height between the levels drawn, so a taller pane
        stretched the same rows, which is not what a user dragging it expects ("resizing
        stretches the plot rather than widening the viewing window"). With ``fit_rows`` on they
        may be **shorter**, so every level shows without scrolling ("the tree must show the
        whole plot"), but never shorter than ``min_row_px``.
        """
        if not self._fit_rows:
            return float(row_px())
        levels = self.levels()
        if levels <= 0:
            return float(row_px())
        fitted = float(self.height()) / levels
        return float(min(row_px(), max(min_row_px(), fitted)))

    def _depth(self, node: int, x: float, width: float, depth: int) -> int:
        if depth + 1 >= VISIBLE_ROWS:
            return VISIBLE_ROWS
        children = self.shown_children(node)
        total = sum(self._times[c] for c in children)
        wide, cursor = [], x
        for child in children:
            w = width * self._times[child] / total if total else 0.0
            # Over the whole width, not only the part on screen: counted over the visible part,
            # zooming into a shallow stretch shrank the content and the scrollbar went away
            # under the pointer. Every level at this zoom, so zooming in only ever adds depth.
            if w >= TINY_BLOCK_PX:
                wide.append((child, cursor, w))
            cursor += w
        if not wide:
            return depth + 1 + (1 if children else 0)
        return max(self._depth(child, cx, w, depth + 1) for child, cx, w in wide)

    def shown_children(self, node: int) -> list[int]:
        """The blocks drawn in the row under ``node``, largest first: its peels opened.

        A real fit's tree is mostly **chains**: one big cluster sheds a piece at each merge
        Drawn one merge per row that is a
        staircase of near-identical full-width bars, and the clusters that came off it are
        scattered one per row down the right edge, never side by side; that is what a real
        run looked like, and it read as "all horizontal" with nothing to tell the
        clusters apart. So a merge whose smaller half holds less than ``SPLIT_SHARE`` of it is
        a peel and gets no row: the pieces it is made of stand in for it, opened the same way,
        until each is a merge that really divides something (or a leaf). The pieces a chain
        shed then sit **side by side** in one row, which is the comparison a user makes. The
        chain's own nodes are still clusters: the card reaches them (its parent and its two
        halves), and when one is selected its drawn pieces are outlined.
        """
        out: list[int] = []
        for child in self._view.children(node):
            out.extend(self._opened(child))
        return sorted(out, key=lambda n: -self._times[n])

    def shown_parent(self, node: int) -> int | None:
        """The block ``node`` is drawn under, or ``None`` for the root and for a chain node.

        Read off the rows as drawn, from the root: a chain node's own opened row contains the
        same pieces as the row they are drawn in, so the nearest true ancestor whose row holds
        ``node`` can be a chain node with no block, and ↑ would go nowhere visible. Built once
        per view (a dict over every drawn node), because ↑ asks on every key press.
        """
        if self._parents is None:
            self._parents = {}
            stack = [self._view.root]
            while stack:
                parent = stack.pop()
                for child in self.shown_children(parent):
                    self._parents[child] = parent
                    stack.append(child)
        return self._parents.get(int(node))

    def _opened(self, node: int) -> list[int]:
        children = self._view.children(node)
        if len(children) != 2 or not self._times[node]:
            return [node]
        smaller = min(self._times[c] for c in children)
        if smaller / self._times[node] >= SPLIT_SHARE:
            return [node]
        return [piece for c in children for piece in self._opened(c)]

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt's name
        theme = plot_theme.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(theme.surface))
        self._blocks = self._layout()
        neutral = QColor(theme.neutral)
        text = QColor(theme.text)
        muted = QColor(theme.text_muted)
        small = QFont(self.font())
        small.setPointSizeF(max(small.pointSizeF() - 1.5, 7.0))
        path = (
            set(self._view.ancestors(self._selected))
            if self._selected is not None
            else set()
        )
        stand_ins = self._stand_ins()
        painter.setClipRect(QRectF(0, 0, self.width(), self.height()))
        faint = QColor(neutral)
        faint.setAlpha(110)
        for rect in self._remainders:
            painter.fillRect(rect, faint)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        gap = QColor(GAP_COLOUR)
        for _rect, node in self._blocks:
            painter.fillRect(self._slots[node], gap)
        kept = self._kept_pieces()
        for rect, node in self._blocks:
            if node in self._tiny:
                painter.fillRect(rect, self.fill_of(node)[0])
                if node in kept:
                    self._paint_kept(painter, rect, kept[node])
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for rect, node in self._blocks:
            if node in self._tiny:
                continue
            own = self._fills.get(node, (neutral, True))[1]
            fill, strip = self.fill_of(node)
            painter.fillRect(rect, fill)
            if strip is not None:
                painter.fillRect(
                    QRectF(
                        rect.x(), rect.y(), rect.width(), min(STRIP_PX, rect.height())
                    ),
                    strip,
                )
            if node in kept:
                # Under the block's name, which starts inside the frame's width.
                self._paint_kept(painter, rect, kept[node])
            ring = self._outlines.get(node)
            if ring is not None and rect.width() > 8 and rect.height() > 8:
                self._paint_ring(painter, rect, ring, node)
            if node in path and node != self._selected and node not in stand_ins:
                pen = QPen(text, 1.0)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawRect(rect)
            if (
                node in self._visited
                and rect.width() > 12
                and node not in self._window_marks
            ):
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(text)
                painter.drawEllipse(rect.topRight() + _offset(-7, 8), 2.5, 2.5)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            if min(rect.right(), self.width()) - max(rect.left(), 0) > TEXT_BLOCK_PX:
                # Ink that reads on the block it sits on: a solid label colour (or a dark shade
                # of the match ramp) is dark enough to swallow dark text, and a pale fill is not.
                opaque = self._shades is not None or (node in self._fills and own)
                solid = opaque and fill.lightness() < 150
                ink = QColor("#ffffff") if solid else text
                if self._shades is not None and not solid and text.lightness() > 128:
                    # A pale shade in the dark theme, where the theme's ink is pale too:
                    # "65.5%" on a block most of whose clips match was not there to read.
                    ink, solid = QColor(GAP_COLOUR), True
                painter.setPen(ink)
                painter.setFont(self.font())
                name = self._names.get(node) or (
                    "all" if node == self._view.root else f"#{node}"
                )
                inner = rect.intersected(QRectF(0, 0, self.width(), self.height()))
                share = self._times[node] / self._times[self._view.root] * 100
                fonts = {"name": self.font(), "share": small, "line": small}
                for box, line, kind in cell_text(
                    inner,
                    name,
                    share,
                    QFontMetricsF(self.font()),
                    QFontMetricsF(small),
                ):
                    painter.setFont(fonts[kind])
                    # The share under the name is quieter, except on a solid fill.
                    painter.setPen(muted if kind == "share" and not solid else ink)
                    painter.drawText(
                        box,
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                        line,
                    )
        self._paint_found(painter, QColor(theme.text))
        # The selection last, so no neighbour's fill or text is drawn over its ring.
        for rect, node in self._blocks:
            if node == self._selected or node in stand_ins:
                self._paint_selection(painter, rect, outward=node in self._tiny)
        self._paint_window_marks(painter)
        painter.setClipping(False)
        painter.end()

    def _paint_window_marks(self, painter: QPainter) -> None:
        """Each window's number on its cluster's block, top right, inside the selection ring's
        inset so the ring still reads round it."""
        self._mark_rects = {}
        if not self._window_marks:
            return
        drawn = {node: rect for rect, node in self._blocks}
        theme = plot_theme.current()
        base = QFont(self.font())
        base.setBold(True)
        full = QFontMetricsF(base).height() + 2
        for node, number in self._window_marks.items():
            pieces = [drawn[n] for n in self._drawn_as(node, drawn)]
            if not pieces:
                continue
            rect = max(pieces, key=lambda r: r.width())
            # As tall as the row allows inside the selection ring's 3 px, never taller than
            # the tree's own text: rows shrink to fit every level, and a fixed disc would not.
            size = min(full, rect.height() - 6)
            if size < MARK_MIN_PX or rect.width() < size + 10:
                continue
            font = QFont(base)
            font.setPixelSize(max(int(size) - 3, 7))
            painter.setFont(font)
            text = str(number)
            width = max(size, QFontMetricsF(font).horizontalAdvance(text) + 8)
            disc = QRectF(rect.right() - width - 4, rect.top() + 3, width, size)
            painter.setPen(QPen(QColor(RING_INNER_COLOUR), 1.5))
            painter.setBrush(QColor(theme.text))
            painter.drawRoundedRect(disc, size / 2, size / 2)
            painter.setPen(QColor(theme.surface))
            painter.drawText(disc, Qt.AlignmentFlag.AlignCenter, text)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            self._mark_rects[node] = disc

    def _kept_pieces(self) -> dict[int, QColor]:
        """``{drawn block: colour}`` for the clusters a screen kept: a kept cluster with no
        block of its own (a chain node, or one zoomed past) frames the blocks it is drawn
        as."""
        if not self._kept or self._view is None:
            return {}
        drawn = {node: rect for rect, node in self._blocks}
        return {
            piece: colour
            for node, colour in self._kept.items()
            for piece in self._drawn_as(node, drawn)
            if piece in drawn
        }

    @staticmethod
    def _paint_kept(painter: QPainter, rect: QRectF, colour: QColor) -> None:
        """A kept cluster's frame on its block's edge: a dark line either side of the
        colour, so it reads on a pale fill and on a dark one (a yellow frame on a near-white
        block did not)."""
        aliased = painter.testRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        _frame(painter, rect, KEPT_PX + 2, QColor(GAP_COLOUR))
        _frame(painter, rect.adjusted(1, 1, -1, -1), KEPT_PX, colour)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, aliased)

    def _paint_found(self, painter: QPainter, edge: QColor) -> None:
        """Each Find clusters id's band, outside the blocks it is drawn as (module docstring).

        **Never inside a block, and never on the zoom root.** A band with no room outside a
        side was moved inside it, over the block's name (user screenshot, 2026-11-16: "rest"
        read "est" under a zoomed-onto block's band, with a stray bar on the row below). A side
        at the widget's edge now has no band, and the block the picture is zoomed onto — or
        one above it, which is the whole picture — has none at all: the zoom says which it is,
        and the selection ring is frame enough. One in a branch zoomed away from has no block
        on screen and is skipped too.
        """
        self._found_pieces = {}
        if not self._found or self._view is None:
            return
        drawn = {node: rect for rect, node in self._blocks}
        out = FOUND_PX + 1
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for node, colour in self._found.items():
            if node != self._root and self._root in self._view.ancestors(node):
                pieces = self._drawn_as(node, drawn)
            else:
                continue
            pieces = [p for p in pieces if p in drawn and p != self._root]
            self._found_pieces[node] = pieces
            for piece in pieces:
                band = drawn[piece].adjusted(-out, -out, out, out)
                _frame(painter, band, 1, edge)
                _frame(painter, band.adjusted(1, 1, -1, -1), FOUND_PX, colour)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    def _drawn_as(self, node: int, drawn) -> list[int]:
        """The drawn blocks that make up ``node``: itself, or its topmost drawn descendants."""
        if node in drawn:
            return [node]
        out, stack = [], list(self._view.children(node)) if self._view else []
        while stack:
            child = stack.pop()
            if child in drawn:
                out.append(child)
            else:
                stack.extend(self._view.children(child))
        return out

    def _paint_selection(self, painter: QPainter, rect: QRectF, outward: bool) -> None:
        """The two-tone ring: 2 px of :data:`GAP_COLOUR` outside 1 px of white, so one of the two
        stands out on any fill, pale or dark, in either theme. Inside a block's edge; around a
        thin column instead, which has no inside to draw in, so the column is still findable.
        """
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        outer = rect.adjusted(-3, -3, 3, 3) if outward else rect
        _frame(painter, outer, 2, QColor(GAP_COLOUR))
        _frame(painter, outer.adjusted(2, 2, -2, -2), 1, QColor(RING_INNER_COLOUR))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    def _paint_ring(self, painter: QPainter, rect: QRectF, ring: QColor, node: int):
        inner = rect.adjusted(3, 3, -3, -3)
        painter.setPen(QPen(QColor("#ffffff"), 4.5))
        painter.drawRect(inner)
        painter.setPen(QPen(ring, 2.5))
        painter.drawRect(inner)
        mark = self._marks.get(node)
        if mark and rect.width() > 30 and rect.height() > 18:
            centre = QPointF(rect.right() - 11, rect.bottom() - 10)
            painter.setPen(QPen(ring, 1.5))
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(centre, 7, 7)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawText(
                QRectF(centre.x() - 7, centre.y() - 7, 14, 14),
                Qt.AlignmentFlag.AlignCenter,
                mark,
            )

    def _stand_ins(self) -> set[int]:
        """The drawn blocks that make up the selected node when it has no block of its own.

        A chain node (opened as a peel) is selected from the card, by its parent or its two
        halves. It is still the selection, so the pieces it is drawn as are outlined: the topmost
        drawn descendants, which together are exactly its members.
        """
        drawn = {node for _rect, node in self._blocks}
        if self._selected is None or self._selected in drawn:
            return set()
        out: set[int] = set()
        stack = list(self._view.children(self._selected))
        while stack:
            node = stack.pop()
            if node in drawn:
                out.add(node)
            else:
                stack.extend(self._view.children(node))
        return out

    # --- interaction ---------------------------------------------------------------------

    def node_at(self, x: float, y: float) -> int | None:
        """The node whose **whole slot** holds ``(x, y)``, not only its drawn block.

        A block is drawn with a 1 px gap after it so neighbours read as separate, and on a block
        a few pixels wide that gap is a third of it: a click that landed in the gap once selected
        nothing, leaving the previous node selected. The slot is the block's full share of its
        row, gap included, so every drawn block is a target as wide as the time it stands for.
        """
        blocks = self._layout()  # never a stale paint's: a zoom may not have drawn yet
        if self._view is None or not blocks:
            return None
        for _rect, node in reversed(blocks):
            if self._slots[node].contains(x, y):
                return node
        return None

    def event(self, event) -> bool:  # noqa: D401 — Qt's override
        if event.type() == QEvent.Type.NativeGesture and zoom_pinch(
            self.zoom, event, self._share(event.position().x())
        ):
            return True
        if event.type() == QEvent.Type.ToolTip and self._view is not None:
            node = self.node_at(event.pos().x(), event.pos().y())
            if node is None:
                QToolTip.hideText()
            else:
                QToolTip.showText(event.globalPos(), self._describe(node), self)
            return True
        return super().event(event)

    def _describe(self, node: int) -> str:
        """What a hover says about a block, readable however narrow it is drawn."""
        name = self._names.get(node) or (
            "all" if node == self._view.root else f"#{node}"
        )
        share = self._times[node] / self._times[self._view.root] * 100
        text = (
            f"{name} · {share:.1f}% of the run · {format_time(int(self._times[node]))}"
        )
        if self._shades is not None and self._diverging:
            text += " · " + self._shade_texts.get(
                int(node), "not a candidate of the screen"
            )
        elif self._shades is not None:
            match = self._shades.get(int(node), float("nan"))
            text += (
                " · no positioned clips to match"
                if match != match
                else f" · {100 * match:.0f}% of its clips match"
            )
        if int(node) in self._kept_texts and not self._diverging:
            text += " · " + self._kept_texts[int(node)]
        if int(node) in self._window_marks:
            text += f" · open in window {self._window_marks[int(node)]}"
        if int(node) in self._found:
            text += " · one of the ids you are finding"
        return text

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        point = event.position()
        self._press = (point.x(), point.y())
        self._press_view = (self.zoom.view, self._full())
        self._dragging = False

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if self._press is None:
            return
        point = event.position()
        if not self._dragging and abs(point.x() - self._press[0]) < DRAG_PX:
            return
        self._dragging = True
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        (left, right), full = self._press_view
        if (left, right) == (0.0, 1.0):
            return  # nothing to pan: the whole width is on screen
        shift = -(point.x() - self._press[0]) / full
        self.zoom.set_view(left + shift, right + shift)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """A click selects; a drag only pans. Deciding on release is what tells them apart."""
        pressed, dragged = self._press, self._dragging
        self._press, self._dragging = None, False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if pressed is None or dragged:
            return
        node = self.node_at(*pressed)
        if node is not None:
            self.node_picked.emit(node)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """Double-click labels the block: labelling is what this tab is for. Zooming onto a
        block is on the right-click menu and the + button."""
        node = self.node_at(event.position().x(), event.position().y())
        if node is not None:
            self.node_picked.emit(node)
            self.label_wanted.emit(node, self._block_anchor(node))

    def _block_anchor(self, node: int):
        """Where an editor for ``node`` opens: under its block, in global coordinates."""
        for rect, drawn in self._layout():
            if drawn == node:
                left = max(rect.left(), 0.0)
                return self.mapToGlobal(QPoint(int(left), int(rect.bottom())))
        return self.mapToGlobal(QPoint(0, 0))

    def contextMenuEvent(self, event) -> None:  # noqa: N802 — Qt's name
        node = self.node_at(event.pos().x(), event.pos().y())
        if node is None:
            return
        self.node_picked.emit(node)
        from detail.gui.menus import open_once

        open_once(self.context_menu(node), event.globalPos(), wait=True)

    def context_menu(self, node: int) -> QMenu:
        """Label…, Remove label (when it has one), Open in window (while the page opens one:
        derived from the wiring), Zoom to this."""
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        label = menu.addAction(
            "Label…", lambda: self.label_wanted.emit(node, self._block_anchor(node))
        )
        remove = menu.addAction("Remove label", lambda: self.unlabel_wanted.emit(node))
        dead = self._edit_reason
        label.setEnabled(not dead)
        label.setToolTip(dead)
        remove.setEnabled(not dead and bool(self._names.get(node)))
        remove.setToolTip(
            dead or ("" if self._names.get(node) else "This has no label of its own.")
        )
        menu.addSeparator()
        if self.receivers(self.window_wanted) > 0:
            menu.addAction("Open in window", lambda: self.window_wanted.emit(node))
        menu.addAction("Zoom to this", lambda: self.zoom_to(node))
        return menu

    def zoom_toward_selection(self, factor: float) -> None:
        """The + button: zoom on the selected block when it is on screen, else the middle."""
        anchor = None
        if self._selected is not None:
            for rect, drawn in self._layout():
                if (
                    drawn == self._selected
                    and rect.right() > 0
                    and rect.left() < self.width()
                ):
                    anchor = (
                        max(rect.left(), 0.0) + min(rect.right(), self.width())
                    ) / 2
                    break
        self.zoom_by(factor, anchor)

    def wheelEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """The wheel zooms around the pointer; Shift+wheel scrolls down the tree; a sideways
        scroll (trackpad) pans."""
        delta = event.angleDelta()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Some platforms turn Shift+wheel into a horizontal delta; either axis means "down".
            notches = (delta.y() or delta.x()) / 120.0
            self.set_scroll_y(self._scroll_y - notches * row_px() * 2)
        else:
            zoom_wheel(self.zoom, event, self._share(event.position().x()))
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if event.key() == Qt.Key.Key_Escape and self.is_zoomed():
            self.fit()
            event.accept()
            return
        super().keyPressEvent(event)


class TreePane(QWidget):
    """The icicle, the overview bar above it, and the vertical scrollbar beside it — both real
    scrollbars, so they look like every other one in the tool."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        self.icicle = NodeIcicle()
        self.overview = OverviewBar(self.icicle.zoom)
        outer.addWidget(self.overview)
        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        outer.addLayout(box, 1)
        box.addWidget(self.icicle, 1)
        self.scrollbar = QScrollBar(Qt.Orientation.Vertical)
        box.addWidget(self.scrollbar)
        self.icicle.extent_changed.connect(self._sync)
        self.scrollbar.valueChanged.connect(self.icicle.set_scroll_y)
        self._sync()

    def _sync(self) -> None:
        bar, icicle = self.scrollbar, self.icicle
        bar.blockSignals(True)
        bar.setRange(0, icicle.scroll_range())
        bar.setPageStep(max(icicle.height(), 1))
        bar.setSingleStep(row_px())
        bar.setValue(icicle.scroll_y)
        bar.blockSignals(False)
        # Kept in place and greyed when there is nothing to scroll, rather than hidden: a bar
        # that appears and vanishes as you zoom resizes the tree under the pointer.
        bar.setEnabled(icicle.scroll_range() > 0)


def _snap(x: float, y: float, w: float, h: float) -> QRectF:
    """The slot ``x..x+w`` by ``y..y+h`` with its edges on whole pixels, at least 1 px each way:
    neighbours that share an edge share the rounded pixel, so no slot overlaps the next.
    """
    left, top = round(x), round(y)
    right = max(round(x + w), left + 1)
    bottom = max(round(y + h), top + 1)
    return QRectF(left, top, right - left, bottom - top)


def _block_of(slot: QRectF) -> QRectF:
    """The coloured part of a slot: all of it but the 1 px gap on its right and bottom, where
    there is room for one (a slot 1 px wide is all colour)."""
    w = slot.width() - 1 if slot.width() >= 2 else slot.width()
    h = slot.height() - 1 if slot.height() >= 2 else slot.height()
    return QRectF(slot.left(), slot.top(), w, h)


def _offset(dx: float, dy: float):
    from PyQt6.QtCore import QPointF

    return QPointF(dx, dy)
