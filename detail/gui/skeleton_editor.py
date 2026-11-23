"""The skeleton editor (§26 *The Keypoints manager*): a skeleton drawn on a plain canvas.

:class:`SkeletonCanvas` is the scene-geometry canvas (``gui/arena.py``'s ``ArenaCanvas``,
subclassed, so zoom, pan, hit-testing and the right-click/right-drag distinction are one
implementation) with named points and edges instead of one polygon, on a **plain background**:
a skeleton belongs to no recording, so no video frame is drawn behind it.

:class:`SkeletonEditor` is the canvas plus the selected point's *Point* name, *Pair with*,
*Side* and *Remove point*: the editing half of the Keypoints manager
(``gui/keypoints_manager.py``), which owns the library, the skeleton's name and saving.
"""

from __future__ import annotations

import numpy as np

from detail.core.skeleton import Skeleton, default_layout
from detail.core.visualize import KEYPOINT_COLORS, keypoint_roles
from detail.gui.arena import ArenaCanvas
from detail.gui.qt import (
    QBrush,
    QColor,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPainter,
    QPen,
    QPointF,
    QPushButton,
    QRectF,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)
from detail.gui.sizing import lines, set_available

_NONE = "(none)"
_SIDES = ("left", "right")
#: The gesture line under the canvas: the scene editor's conventions plus the edge gesture.
HINT = (
    "click: add · drag: move · right-click: remove · Shift+click a 2nd point: join or "
    "unjoin · scroll: zoom · right-drag: pan"
)
#: The canvas's plain background: a skeleton is drawn on nothing, never on a video frame.
BLANK = np.full((600, 600, 3), 56, dtype=np.uint8)


class SkeletonCanvas(ArenaCanvas):
    """The scene-geometry canvas with named points and edges instead of one polygon.

    The base class owns the view (zoom, pan, the fitted rect), the hit-test, the hover
    highlight, the drag and the right-click/right-drag distinction; this class owns what the
    points *mean*: a selection, the edges between them, the pair colours, the Shift+click edge
    gesture, and a removal that asks the editor (which holds the point's name, edges, pair and
    marks).
    """

    #: A new point was appended at this index (already selected).
    point_added = Signal(int)
    #: A right click asked for this point to go.
    remove_requested = Signal(int)
    #: The selected point changed (-1: none).
    selection_changed = Signal(int)
    #: Shift+click: join these two, or unjoin them if already joined.
    edge_toggled = Signal(int, int)

    def __init__(self) -> None:
        super().__init__(BLANK, points=[], max_points=None)
        self._names: list[str] = []
        self._edges: list[tuple[int, int]] = []
        self._roles: list[str] = []
        self._selected: int | None = None
        self.setMinimumSize(lines(30), lines(24))

    # -- what the editor shows ----------------------------------------------- #
    def show_skeleton(
        self,
        names: list[str],
        layout: list[tuple[float, float]],
        edges: list[tuple[int, int]],
        roles: list[str],
        selected: int | None,
    ) -> None:
        self._names = list(names)
        self._points = [(float(x), float(y)) for x, y in layout]
        self._edges = list(edges)
        self._roles = list(roles)
        self._selected = (
            selected if selected is not None and selected < len(names) else None
        )
        if self._hover_index is not None and self._hover_index >= len(self._points):
            self._hover_index = None
        self.update()

    def selected(self) -> int | None:
        return self._selected

    def _select(self, index: int | None, *, announce: bool = True) -> None:
        if index is not None and not 0 <= index < len(self._points):
            index = None
        changed = index != self._selected
        self._selected = index
        self.update()
        if changed and announce:
            self.selection_changed.emit(-1 if index is None else index)

    # -- gestures ---------------------------------------------------------------- #
    def mousePressEvent(self, event) -> None:
        if self._unavailable is not None:
            return
        pos = event.position()
        if event.button() == Qt.MouseButton.RightButton or self._pan_anchor is not None:
            super().mousePressEvent(event)  # right press: a click removes, a drag pans
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        index = self._vertex_at(pos)
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # The edge gesture: a second point joins (or unjoins) the selected one. A shift-
            # click on empty space does nothing, so a missed point never adds a stray one.
            if (
                index is not None
                and self._selected is not None
                and index != self._selected
            ):
                self.edge_toggled.emit(self._selected, index)
            return
        if index is not None:
            self._select(index)
            self._drag_index = index
            return
        if self._image_rect().contains(pos):
            self._points.append(self._normalized(pos))
            new = len(self._points) - 1
            # Selected quietly: the manager names the point and takes its position when it
            # hears `point_added`; a selection announced first would redraw from a model
            # that does not hold the point yet.
            self._selected = new
            self.point_added.emit(new)

    def mouseReleaseEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.RightButton
            and self._pan_anchor is not None
        ):
            index = None if self._pan_dragged else self._right_press_index
            self._pan_anchor = None
            self._pan_dragged = False
            self._right_press_index = None
            self._drag_index = None
            if index is not None:
                self.remove_requested.emit(index)
            return
        super().mouseReleaseEvent(event)

    # -- painting ----------------------------------------------------------------- #
    _EDGE_HALO = QColor(20, 20, 20, 170)
    _EDGE = QColor(240, 240, 240)
    _OUTLINE = QColor(20, 20, 20)
    _SELECTION = QColor(255, 255, 255)
    _HOVER = QColor(255, 220, 0)

    def paintEvent(self, event) -> None:
        if self._unavailable is not None:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._image_rect()
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        points = self._widget_points(rect)
        for a, b in self._edges:
            if a < len(points) and b < len(points):
                painter.setPen(QPen(self._EDGE_HALO, 4))
                painter.drawLine(points[a], points[b])
                painter.setPen(QPen(self._EDGE, 2))
                painter.drawLine(points[a], points[b])
        for i, point in enumerate(points):
            role = self._roles[i] if i < len(self._roles) else "midline"
            color = QColor(*KEYPOINT_COLORS[role])
            hovered = i == self._hover_index
            radius = 7.0 if hovered else 5.5
            if i == self._selected:
                painter.setPen(QPen(self._SELECTION, 2))
                painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                painter.drawEllipse(point, radius + 4.0, radius + 4.0)
            painter.setPen(QPen(self._HOVER if hovered else self._OUTLINE, 1.5))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(point, radius, radius)
            name = self._names[i] if i < len(self._names) else ""
            if name:
                anchor = QPointF(point.x() + radius + 4.0, point.y() - radius)
                painter.setPen(QPen(self._EDGE_HALO))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    painter.drawText(QPointF(anchor.x() + dx, anchor.y() + dy), name)
                painter.setPen(QPen(self._EDGE))
                painter.drawText(anchor, name)


class SkeletonEditor(QWidget):
    """Draw one skeleton: its points, the edges between them, its left/right pairs, and where
    each point sits (the layout).

    :meth:`load` shows a skeleton; :meth:`skeleton` is the one drawn now; ``edited`` fires on
    every change the user makes (never on :meth:`load`). Whether the result may be saved, and
    where, is the owner's business.
    """

    edited = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._loading = False
        self._names: list[str] = []
        self._edges: list[tuple[str, str]] = []
        self._pairs: list[tuple[str, str]] = []
        self._layout: list[tuple[float, float]] = []
        self._selected: int | None = None

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        self._canvas = SkeletonCanvas()
        self._canvas.point_added.connect(self._point_added)
        self._canvas.remove_requested.connect(self._remove_point)
        self._canvas.selection_changed.connect(self._canvas_selected)
        self._canvas.edge_toggled.connect(self._toggle_edge)
        self._canvas.edited.connect(self._canvas_moved)
        column.addWidget(self._canvas, 1)
        hint = QLabel(HINT)
        hint.setWordWrap(True)
        column.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("Point:"))
        self._point = QLineEdit()
        self._point.setMinimumWidth(lines(10))
        self._point.editingFinished.connect(self._point_renamed)
        row.addWidget(self._point)
        row.addWidget(QLabel("Pair with:"))
        self._pair = QComboBox()
        self._pair.activated.connect(
            lambda index: self._pair_changed(self._pair.itemData(index))
        )
        row.addWidget(self._pair)
        row.addWidget(QLabel("Side:"))
        self._side = QComboBox()
        self._side.addItems(list(_SIDES))
        self._side.activated.connect(lambda index: self._side_changed(_SIDES[index]))
        row.addWidget(self._side)
        self._remove = QPushButton("Remove point")
        self._remove.setAutoDefault(False)
        self._remove.clicked.connect(
            lambda: (
                self._remove_point(self._selected)
                if self._selected is not None
                else None
            )
        )
        row.addWidget(self._remove)
        row.addStretch()
        column.addLayout(row)
        self._problem = QLabel("")
        self._problem.setWordWrap(True)
        column.addWidget(self._problem)
        self.load(Skeleton(names=()))

    # -- public -------------------------------------------------------------- #
    def canvas(self) -> SkeletonCanvas:
        return self._canvas

    def load(self, sk: Skeleton) -> None:
        """Show ``sk`` (its layout, or a default one when it has none); not an edit."""
        self._names = list(sk.names)
        self._edges = [tuple(e) for e in sk.edges]
        self._pairs = [tuple(p) for p in sk.pairs]
        layout = sk.layout
        if layout is None or len(layout) != len(sk.names):
            layout = default_layout(sk)
        self._layout = [(float(x), float(y)) for x, y in layout]
        self._selected = 0 if self._names else None
        self._show()

    def skeleton(self) -> Skeleton:
        """The skeleton as drawn, layout included."""
        return Skeleton(
            names=tuple(self._names),
            edges=tuple(self._edges),
            pairs=tuple(self._pairs),
            layout=tuple((round(x, 4), round(y, 4)) for x, y in self._layout),
        )

    def problem(self) -> str | None:
        """Why the drawn skeleton is not valid yet, or None."""
        try:
            self.skeleton().validate()
        except ValueError as exc:
            return str(exc)
        return None

    # -- the model -------------------------------------------------------------- #
    def _partner(self, name: str) -> str | None:
        for left, right in self._pairs:
            if name == left:
                return right
            if name == right:
                return left
        return None

    def _show(self) -> None:
        self._loading = True
        index = {n: i for i, n in enumerate(self._names)}
        edges = [
            (index[a], index[b]) for a, b in self._edges if a in index and b in index
        ]
        unique = len(set(self._names)) == len(self._names)
        roles = keypoint_roles(
            Skeleton(names=tuple(self._names), pairs=tuple(self._pairs))
            if unique
            else Skeleton(names=tuple(self._names))
        )
        self._canvas.show_skeleton(
            self._names, self._layout, edges, roles, self._selected
        )
        has_point = self._selected is not None and self._selected < len(self._names)
        self._pair.clear()
        if has_point:
            name = self._names[self._selected]
            self._point.setText(name)
            self._pair.addItem(_NONE, None)
            for other in self._names:
                if other != name:
                    self._pair.addItem(other, other)
            partner = self._partner(name)
            self._pair.setCurrentIndex(
                self._pair.findData(partner) if partner is not None else 0
            )
            if partner is not None:
                left, _right = next(p for p in self._pairs if name in p)
                self._side.setCurrentIndex(0 if left == name else 1)
        else:
            self._point.clear()
        self._loading = False
        self._sync()

    def _sync(self) -> None:
        self._problem.setText(self.problem() or "")
        has_point = self._selected is not None and self._selected < len(self._names)
        reason = "Select a point on the canvas first."
        for control in (self._point, self._pair, self._remove):
            set_available(control, has_point, reason)
        paired = has_point and self._partner(self._names[self._selected]) is not None
        set_available(self._side, paired, "Pair this point first.")

    def _changed(self, *, redraw: bool = True) -> None:
        if redraw:
            self._show()
        else:
            self._sync()
        self.edited.emit()

    # -- edits on the canvas and the point row ---------------------------------- #
    def _point_added(self, index: int) -> None:
        points = self._canvas.normalized_polygon()
        if index >= len(points):
            return
        n = len(self._names) + 1
        while f"point_{n}" in self._names:
            n += 1
        self._names.append(f"point_{n}")
        self._layout.append(points[index])
        self._selected = len(self._names) - 1
        self._changed()
        self._point.setFocus()
        self._point.selectAll()

    def _canvas_moved(self) -> None:
        moved = self._canvas.normalized_polygon()
        if len(moved) == len(self._layout) and moved != [
            (round(x, 6), round(y, 6)) for x, y in self._layout
        ]:
            self._layout = list(moved)
            self._changed(redraw=False)

    def _canvas_selected(self, index: int) -> None:
        if not self._loading:
            self._selected = index if index >= 0 else None
            self._show()

    def _toggle_edge(self, a: int, b: int) -> None:
        if not (0 <= a < len(self._names) and 0 <= b < len(self._names)) or a == b:
            return
        pair = {self._names[a], self._names[b]}
        if any(set(e) == pair for e in self._edges):
            self._edges = [e for e in self._edges if set(e) != pair]
        else:
            self._edges.append((self._names[a], self._names[b]))
        self._changed()

    def _remove_point(self, index: int | None) -> None:
        if index is None or not 0 <= index < len(self._names):
            return
        name = self._names.pop(index)
        self._layout.pop(index)
        self._edges = [e for e in self._edges if name not in e]
        self._pairs = [p for p in self._pairs if name not in p]
        if self._selected is not None:
            if self._selected == index:
                self._selected = None
            elif self._selected > index:
                self._selected -= 1
        self._changed()

    def _point_renamed(self) -> None:
        if self._loading or self._selected is None:
            return
        old, new = self._names[self._selected], self._point.text().strip()
        if new == old:
            return
        if not new or new in self._names:
            self._problem.setText(
                "keypoint names must be non-empty"
                if not new
                else f"keypoint {new!r} appears twice"
            )
            self._point.setText(old)
            return
        rename = lambda n: new if n == old else n  # noqa: E731 — a one-use mapping
        self._names[self._selected] = new
        self._edges = [(rename(a), rename(b)) for a, b in self._edges]
        self._pairs = [(rename(a), rename(b)) for a, b in self._pairs]
        self._changed()

    def _pair_changed(self, partner: str | None) -> None:
        if self._loading or self._selected is None:
            return
        name = self._names[self._selected]
        self._pairs = [p for p in self._pairs if name not in p and partner not in p]
        if partner is not None:
            self._pairs.append((name, partner))
        self._changed()

    def _side_changed(self, side: str) -> None:
        if self._loading or self._selected is None:
            return
        name = self._names[self._selected]
        pair = next((p for p in self._pairs if name in p), None)
        if pair is None:
            return
        other = pair[1] if pair[0] == name else pair[0]
        self._pairs = [p for p in self._pairs if name not in p]
        self._pairs.append((name, other) if side == "left" else (other, name))
        self._changed()
