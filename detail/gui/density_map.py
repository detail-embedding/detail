"""Where in the frame: the 16×16 position grid, and a region picker over it.

:class:`DensityMap` draws position counts (frame-normalized) at the frame's aspect. Click toggles
a cell; drag draws a box, live, which then stays with a handle on each corner: drag a corner to
resize it, drag inside it to move it, drag outside it to start a new one.
:class:`RegionPicker` is what the cluster page's Location filter uses: it *holds* the region, says
when it changed, and can be reset (its Reset shows only while something is picked).

Cells are ``(bx, by)`` — column, then row — the order ``DensityMap.cell_at`` reports and the
node cache's position grid is indexed ``[by, bx]`` by.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from detail.analysis.node_cache import POSITION_BINS
from detail.gui import plot_theme
from detail.gui.sizing import set_available
from detail.gui.throttle import Throttle

#: The grid's side, in cells. The node cache's, so a picked cell is a cache cell.
BINS = POSITION_BINS
#: How near a box's corner, in pixels, a press grabs that corner rather than the box.
CORNER_GRAB_PX = 8
#: A corner handle's side, in pixels.
HANDLE_PX = 7


def box_of(region) -> tuple[int, int, int, int] | None:
    """``(x0, y0, x1, y1)`` inclusive when ``region`` is exactly a filled rectangle, else
    ``None``. Derived from the cells every time, so a region restored from a saved selection
    gets its handles back, and one a click has punched a hole in loses them."""
    cells = set(region)
    if not cells:
        return None
    xs = [x for x, _y in cells]
    ys = [y for _x, y in cells]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if len(cells) != (x1 - x0 + 1) * (y1 - y0 + 1):
        return None
    return x0, y0, x1, y1


def box_cells(x0: int, y0: int, x1: int, y1: int) -> frozenset:
    """Every cell of the box between two corner cells, in either order."""
    (x0, x1), (y0, y1) = sorted((x0, x1)), sorted((y0, y1))
    return frozenset((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))


def paint_density(
    painter: QPainter,
    width: float,
    height: float,
    grid,
    colour,
    *,
    diverging,
    theme,
    peak: float | None = None,
    region: frozenset | None = None,
) -> None:
    """Draw a position grid into ``(0, 0, width, height)`` of ``painter``.

    ``width`` / ``height`` is the **video frame's** aspect. Positions are fractions of the
    frame's width and height, so a square map of a 16:9 frame squeezed every place a subject
    stood sideways ("the video frame isn't square ... it doesn't make much sense spatially").

    Takes the theme rather than reading it, so it can paint off the UI thread.
    """
    grid = np.asarray(grid, dtype=np.float64)
    painter.fillRect(QRectF(0, 0, width, height), QColor(theme.grid))
    rows, cols = (grid.shape[0], grid.shape[1]) if grid.ndim == 2 else (1, 1)
    cell_w = (width - 8) / max(cols, 1)
    cell_h = (height - 8) / max(rows, 1)
    # A shared ``peak`` puts a grid of maps on one scale, so 8 samples in one map look like 8
    # samples beside 1,051 in the next, not like the same dark patch.
    if peak is None:
        peak = float(np.abs(grid).max()) if grid.size else 0.0
    for y in range(rows):
        for x in range(cols):
            value = grid[y, x]
            if not value or not peak:
                continue
            if diverging:
                fill = plot_theme.qcolor(
                    theme.diverging[-1] if value > 0 else theme.diverging[0]
                )
                fill.setAlpha(int(min(abs(value) / 2.0, 1.0) * 255))
            else:
                fill = QColor(colour)
                fill.setAlpha(int((0.12 + 0.88 * math.sqrt(value / peak)) * 255))
            painter.fillRect(
                QRectF(4 + x * cell_w, 4 + y * cell_h, cell_w + 0.5, cell_h + 0.5), fill
            )
    painter.setPen(QColor(theme.axis))
    painter.drawRect(QRectF(3.5, 3.5, width - 7, height - 7))
    if region:
        paint_region(painter, region, cell_w, cell_h, theme)


def region_edges(region) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """The outline of a set of grid cells: each cell edge whose neighbour across it is not in
    the set, as ``((x0, y0), (x1, y1))`` in cell-corner coordinates.

    One outline around the area rather than a box per cell: boxed cell by cell, a picked region
    drew as a spreadsheet laid over the map ("what's going on with this grid thing?").
    """
    cells = set(region)
    edges = []
    for x, y in cells:
        if (x, y - 1) not in cells:
            edges.append(((x, y), (x + 1, y)))
        if (x, y + 1) not in cells:
            edges.append(((x, y + 1), (x + 1, y + 1)))
        if (x - 1, y) not in cells:
            edges.append(((x, y), (x, y + 1)))
        if (x + 1, y) not in cells:
            edges.append(((x + 1, y), (x + 1, y + 1)))
    return edges


def paint_region(
    painter: QPainter, region, cell_w: float, cell_h: float, theme
) -> None:
    """The picked region: a light tint over it and one outline around it."""
    tint = QColor(theme.text)
    tint.setAlpha(40)
    for bx, by in region:
        painter.fillRect(
            QRectF(4 + bx * cell_w, 4 + by * cell_h, cell_w + 0.5, cell_h + 0.5), tint
        )
    painter.setPen(QPen(QColor(theme.text), 2))
    for (x0, y0), (x1, y1) in region_edges(region):
        painter.drawLine(
            QPointF(4 + x0 * cell_w, 4 + y0 * cell_h),
            QPointF(4 + x1 * cell_w, 4 + y1 * cell_h),
        )


class DensityMap(QWidget):
    """Position counts (frame-normalized) at the frame's aspect, or a log-ratio against the run.

    **Click a cell** to toggle it in the region. **Drag** to draw a box, drawn while you drag; a
    region that is a box keeps a handle on each corner — drag one to resize, drag inside the box
    to move it, drag outside it for a new one. ``region_picked`` carries the region the gesture
    asks for, while dragging (at most every 50 ms, ``gui/throttle.py``) and on release.

    **From the keyboard**: Tab reaches the map, the arrows move a ringed cell (where the last
    click landed, else the top-left one), and Space adds or removes it.
    """

    #: The region the user asked for: the current one with the clicked cell toggled, or the box
    #: being drawn, moved or resized.
    region_picked = pyqtSignal(object)

    def __init__(
        self, width: int, caption: str = "", aspect: float = 1.0, parent=None
    ) -> None:
        super().__init__(parent)
        self._width = int(width)
        self._aspect = float(aspect) if aspect and aspect > 0 else 1.0
        self._height = max(int(round(self._width / self._aspect)), 1)
        self._caption = caption
        self._grid = np.zeros((1, 1))
        self._colour = QColor("#888888")
        self._diverging = False
        self._flag = False
        self._peak: float | None = None
        self._region: frozenset = frozenset()
        #: The gesture in hand: ``(mode, press cell, anchor)`` — ``mode`` is ``"new"``,
        #: ``"move"`` or ``"resize"``; ``anchor`` is the fixed corner, or the box being moved.
        self._gesture: tuple | None = None
        self._dragged = False
        self._live = Throttle(self.region_picked.emit)
        self.setFixedSize(self._width, self._height + 18)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        #: The cell the arrows move and Space toggles, ringed while the map has the focus.
        self._cursor: tuple[int, int] = (0, 0)
        self.setToolTip(
            "Click a cell to add or remove it; drag to draw a box. From the keyboard: the "
            "arrows move, Space adds or removes"
        )

    def set_grid(
        self,
        grid,
        colour,
        *,
        diverging=False,
        caption=None,
        flag=False,
        peak: float | None = None,
        region: frozenset | None = None,
    ) -> None:
        self._grid = np.asarray(grid, dtype=np.float64)
        self._colour = QColor(colour)
        self._diverging = diverging
        self._flag = flag
        self._peak = peak
        if self._gesture is None or not self._dragged:
            self._region = frozenset(region or ())
        if caption is not None:
            self._caption = caption
        self.update()

    @property
    def side(self) -> int:
        """The map's width (its height follows the frame's aspect)."""
        return self._width

    def set_side(self, width: int) -> None:
        """Redraw at ``width`` pixels wide, the height following the frame's aspect — for an
        embedder whose column is dragged wider or narrower."""
        width = max(int(width), 1)
        if width == self._width:
            return
        self._width = width
        self._height = max(int(round(self._width / self._aspect)), 1)
        self.setFixedSize(self._width, self._height + 18)
        self.update()

    @property
    def aspect(self) -> float:
        return self._aspect

    def region(self) -> frozenset:
        """The region as drawn: mid-drag, the box under the hand."""
        return self._region

    def cursor_cell(self) -> tuple[int, int]:
        """The ``(bx, by)`` cell the arrows move and Space toggles."""
        return self._cursor

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        _cell_w, _cell_h, cols, rows = self._cell_size()
        steps = {
            Qt.Key.Key_Left: (-1, 0),
            Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1),
            Qt.Key.Key_Down: (0, 1),
        }
        key = event.key()
        if key in steps:
            dx, dy = steps[key]
            x, y = self._cursor
            self._cursor = (
                min(max(x + dx, 0), cols - 1),
                min(max(y + dy, 0), rows - 1),
            )
            self.update()
        elif key == Qt.Key.Key_Space and self._gesture is None:
            self.region_picked.emit(frozenset(self._region ^ {self._cursor}))
        else:
            super().keyPressEvent(event)

    def focusInEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().focusOutEvent(event)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt's name
        theme = plot_theme.current()
        painter = QPainter(self)
        paint_density(
            painter,
            float(self._width),
            float(self._height),
            self._grid,
            self._colour,
            diverging=self._diverging,
            theme=theme,
            peak=self._peak,
            region=self._region,
        )
        box = box_of(self._region)
        if box is not None:
            painter.setPen(QPen(QColor(theme.surface), 1))
            painter.setBrush(QColor(theme.text))
            for corner in self._corners(box):
                painter.drawRect(
                    QRectF(
                        corner.x() - HANDLE_PX / 2,
                        corner.y() - HANDLE_PX / 2,
                        HANDLE_PX,
                        HANDLE_PX,
                    )
                )
            painter.setBrush(Qt.BrushStyle.NoBrush)
        if self.hasFocus():
            cell_w, cell_h, _cols, _rows = self._cell_size()
            x, y = self._cursor
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(theme.text), 2))
            painter.drawRect(QRectF(4 + x * cell_w, 4 + y * cell_h, cell_w, cell_h))
        painter.setPen(QColor(theme.text if self._flag else theme.text_muted))
        painter.drawText(
            QRectF(0, self._height, self._width, 18),
            Qt.AlignmentFlag.AlignLeft,
            painter.fontMetrics().elidedText(
                self._caption, Qt.TextElideMode.ElideRight, self._width
            ),
        )
        painter.end()

    def _cell_size(self) -> tuple[float, float, int, int]:
        rows, cols = self._grid.shape if self._grid.ndim == 2 else (1, 1)
        return (
            max((self._width - 8) / max(cols, 1), 1e-6),
            max((self._height - 8) / max(rows, 1), 1e-6),
            cols,
            rows,
        )

    def cell_at(self, point) -> tuple[int, int] | None:
        """The ``(bx, by)`` grid cell under ``point``, or ``None`` outside the map."""
        cell_w, cell_h, cols, rows = self._cell_size()
        x = (point.x() - 4) / cell_w
        y = (point.y() - 4) / cell_h
        if 0 <= x < cols and 0 <= y < rows:
            return int(x), int(y)
        return None

    def _clamped_cell(self, point) -> tuple[int, int]:
        """The cell under ``point``, held to the grid's edge when the drag leaves the map."""
        cell_w, cell_h, cols, rows = self._cell_size()
        x = int((point.x() - 4) // cell_w)
        y = int((point.y() - 4) // cell_h)
        return min(max(x, 0), cols - 1), min(max(y, 0), rows - 1)

    def _corners(self, box) -> list[QPointF]:
        """The box's four outer corners in pixels: top-left, top-right, bottom-left,
        bottom-right."""
        cell_w, cell_h, _cols, _rows = self._cell_size()
        x0, y0, x1, y1 = box
        left, right = 4 + x0 * cell_w, 4 + (x1 + 1) * cell_w
        top, bottom = 4 + y0 * cell_h, 4 + (y1 + 1) * cell_h
        return [
            QPointF(left, top),
            QPointF(right, top),
            QPointF(left, bottom),
            QPointF(right, bottom),
        ]

    def corner_at(self, point) -> int | None:
        """Which of the kept box's corners (index into ``_corners``) ``point`` grabs."""
        box = box_of(self._region)
        if box is None:
            return None
        for index, corner in enumerate(self._corners(box)):
            if (
                abs(corner.x() - point.x()) <= CORNER_GRAB_PX
                and abs(corner.y() - point.y()) <= CORNER_GRAB_PX
            ):
                return index
        return None

    def _inside_box(self, cell) -> bool:
        box = box_of(self._region)
        return (
            box is not None
            and cell is not None
            and box[0] <= cell[0] <= box[2]
            and box[1] <= cell[1] <= box[3]
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = event.position()
        cell = self.cell_at(point)
        corner = self.corner_at(point)
        if cell is not None:
            self._cursor = cell
        self._dragged = False
        self._live.start(self._region)
        if corner is not None:
            x0, y0, x1, y1 = box_of(self._region)
            # The corner opposite the one grabbed stays put.
            fixed = [(x1, y1), (x0, y1), (x1, y0), (x0, y0)][corner]
            self._gesture = ("resize", self._clamped_cell(point), fixed)
        elif cell is None:
            self._gesture = None
        elif self._inside_box(cell):
            self._gesture = ("move", cell, box_of(self._region))
        else:
            self._gesture = ("new", cell, cell)

    def _wanted(self, cell) -> frozenset:
        mode, pressed, anchor = self._gesture
        if mode == "move":
            x0, y0, x1, y1 = anchor
            _cell_w, _cell_h, cols, rows = self._cell_size()
            dx = min(max(cell[0] - pressed[0], -x0), cols - 1 - x1)
            dy = min(max(cell[1] - pressed[1], -y0), rows - 1 - y1)
            return box_cells(x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        return box_cells(anchor[0], anchor[1], cell[0], cell[1])

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        point = event.position()
        if self._gesture is None:
            if self.corner_at(point) is not None:
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif self._inside_box(self.cell_at(point)):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            return
        cell = self._clamped_cell(point)
        if not self._dragged and cell == self._gesture[1]:
            return
        self._dragged = True
        wanted = self._wanted(cell)
        if wanted != self._region:
            self._region = wanted
            self.update()
            self._live.offer(wanted)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt's name
        gesture = self._gesture
        if gesture is None:
            return
        if self._dragged:
            wanted = self._wanted(self._clamped_cell(event.position()))
            self._gesture, self._dragged = None, False
            self._region = wanted
            self._live.flush(wanted)
            return
        self._gesture = None
        cell = self.cell_at(event.position())
        if cell is not None and cell == gesture[1]:
            self.region_picked.emit(frozenset(self._region ^ {cell}))


class RegionPicker(QWidget):
    """The run's position density, with a set of picked cells on it.

    ``region_changed`` carries a ``frozenset`` of ``(bx, by)`` cells — empty when nothing is
    picked, which a filter reads as "no location condition".

    Over the map: ``heading`` (when given) at the left of one row, and **Reset** at its right,
    shown only while something is picked. Then ``note`` — empty and hidden unless the embedder
    has something to say (why the map is greyed) — then the map. How to pick is the map's
    tooltip, not a line of text under it: the instructions sat under the map on every visit and
    read as clutter once learned (user, review round 8).
    """

    region_changed = pyqtSignal(object)

    def __init__(
        self,
        width: int = 360,
        aspect: float = 16 / 9,
        heading: QWidget | None = None,
        parent=None,
        *,
        reset_row: bool = True,
    ) -> None:
        """``reset_row=False``: no heading row — the embedder's own line names the map and
        carries its Reset (INTERPRET's Location line); :attr:`reset_button` is then never
        shown."""
        super().__init__(parent)
        self._grid = np.zeros((BINS, BINS))
        self._region: frozenset = frozenset()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        if heading is not None:
            head.addWidget(heading, 1)
        else:
            head.addStretch(1)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(lambda: self.set_region(frozenset()))
        # The row is as tall with Reset as without, so the map does not move when it appears.
        policy = self.reset_button.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.reset_button.setSizePolicy(policy)
        head.addWidget(self.reset_button, 0, Qt.AlignmentFlag.AlignTop)
        self._reset_row = bool(reset_row)
        if self._reset_row:
            layout.addLayout(head)
        else:
            self.reset_button.setParent(self)
        #: Why the map is greyed, when it is (set by the embedder); hidden otherwise.
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setVisible(False)
        layout.addWidget(self.note)
        self.map = DensityMap(width, "", aspect)
        self.map.setToolTip(
            "Click a square to add or remove it. Drag to draw a box; drag its corners to "
            "resize it, or inside it to move it."
        )
        self.map.region_picked.connect(self.set_region)
        layout.addWidget(self.map)
        self._sync()

    def set_density(self, grid) -> None:
        """The whole run's member counts per cell, ``(BINS, BINS)`` indexed ``[by, bx]``."""
        self._grid = np.asarray(grid, dtype=np.float64)
        self._sync()

    def region(self) -> frozenset:
        return self._region

    def set_region(self, region) -> None:
        """Pick exactly ``region``; emits ``region_changed`` when that is a change."""
        region = frozenset(
            (int(x), int(y)) for x, y in region if 0 <= x < BINS and 0 <= y < BINS
        )
        if region == self._region:
            return
        self._region = region
        self._sync()
        self.region_changed.emit(region)

    def set_available(self, available: bool, reason: str) -> None:
        """Grey the map, with ``reason`` said over it, when there are no positions to pick."""
        set_available(self.map, available, reason)
        self.note.setText("" if available else reason)
        self.note.setVisible(not available)

    def _sync(self) -> None:
        theme = plot_theme.current()
        self.map.set_grid(self._grid, theme.text_muted, region=self._region)
        n = len(self._region)
        self.reset_button.setVisible(bool(n) and self._reset_row)
        self.reset_button.setToolTip(
            f"Remove the {n} picked square{'s' if n != 1 else ''}" if n else ""
        )
