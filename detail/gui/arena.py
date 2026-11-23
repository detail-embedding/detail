"""Spatial-definition drawing canvas + dialog (§10).

Click vertices on a recording frame to define a **field of view** (the polygon
that bounds detection — internal kind ``arena``), a **region of interest**
(analysis polygon), or a **point of interest** (a single analysis landmark).
Vertices are stored normalized (0–1 of the frame), so they survive widget resizing
and map straight onto the ``Recording.define_*`` keyframe format. ROIs/POIs may be
active for only part of the video (an ``active`` interval); a field of view is
always in effect.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from detail.core.dataclasses import PolygonKeyframes
from detail.core.utils import polygon_at_frame
from detail.core.chart_theme import HUE_NAMES, LIGHT
from detail.gui.dialog import Dialog
from detail.gui.qt import (
    QBrush,
    QCheckBox,
    QColor,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QIcon,
    QImage,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPainter,
    QPen,
    QPixmap,
    QPointF,
    QPolygonF,
    QPushButton,
    QRectF,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)
from detail.gui.spin import FloatSpin
from detail.gui import frame_source
from detail.gui.busy import decode_overlay, run_busy
from detail.gui.scroll import OWNS_WHEEL
from detail.gui.status_color import MUTED, set_label_color
from detail.gui.scrubber import FrameScrubber
from detail.gui.sizing import lines, set_available
from detail.gui.workers import BackgroundCall

# Shown behind the "Decoding frame…" overlay until the first real frame decodes
# off-thread, so opening an editor on a large video never freezes on a blank window.
_BLANK_FRAME = np.zeros((360, 480, 3), np.uint8)


#: Decimal places a vertex keeps on the canvas. The scene editor's "has this changed" check
#: compares at the same precision, or a vertex stored with more (an API or CSV definition)
#: reads as an edit the moment the canvas hands it back on a scrub, and lights Save.
_VERTEX_DECIMALS = 6


def _rounded(point) -> tuple[float, float]:
    return (round(point[0], _VERTEX_DECIMALS), round(point[1], _VERTEX_DECIMALS))


def _arrival_report(
    added: list[str], flattened: list[str], refused: list[tuple[str, str]]
) -> str:
    """What a setup arriving from elsewhere actually did, in one message.

    A count alone ("imported 2 region(s)") is what this replaced, and it cannot say the
    two things the trip changes: a polygon that lost its camera motion, and a definition
    left behind because its name was taken or it carried an active window. Both are silent
    otherwise — the count is right either way.
    """
    lines = [f"Added {len(added)}: {', '.join(added)}" if added else "Nothing added."]
    if flattened:
        lines.append(
            f"Camera motion not copied (taken at frame 0): {', '.join(flattened)}"
        )
    lines += [f"Not added — {name}: {reason}" for name, reason in refused]
    return "\n".join(lines)


def _paint_decoded(canvas: "ArenaCanvas", payload) -> None:
    """Put a decoded backdrop on ``canvas``, or say why there is none.

    The three arena scrubbers share this: a dropped decode used to leave the previous
    frame up, which is worse than a blank one here — the polygon goes onto a frame you
    are not looking at.
    """
    image, reason = payload
    if reason is not None:
        canvas.set_unavailable(frame_source.unavailable_text(reason))
        return
    canvas.set_image(image)


class ArenaCanvas(QWidget):
    """A frame backdrop with an editable polygon (or single-point) overlay.

    Modeless editing: left-click empty space adds a vertex, left-drag a vertex
    moves it, right-click a vertex removes it. The vertex under the cursor is
    highlighted so it reads as grabbable. With ``max_points=1`` (a point of
    interest) a left-click places/moves the single point and it renders as a
    crosshair marker rather than a polygon.

    **Zoom and pan.** A frame's own resolution is usually much larger than the fitted
    preview, so a vertex a few source pixels off is unplaceable at the fitted scale. Scroll
    zooms in/out centered on the cursor (so the point you are looking at stays under it),
    matching `gui/image_view.py::ImageView` — the shared convention for every other picture
    in the tool. Panning cannot borrow that convention's left-drag, though: left is already
    every vertex's own click-and-drag. **A right *click* removes the vertex under it; a
    right *drag* pans** instead, the same click-vs-drag distinction `ImageView` uses to tell
    an ordinary press from a gesture — and, unlike middle-drag or Ctrl+left (`docs/gui_design.md`
    §3.1's "a binding is only real on the platform it can be produced on": a MacBook trackpad
    has neither a middle button nor a way to send Ctrl+left, which macOS turns into a right
    click before Qt sees it), a right click is a device every pointer has.

    Both zoom and pan are a *view* of the shape, never the shape itself — neither touches
    ``_points`` or emits ``edited``, so zooming to check a corner cannot itself be mistaken
    for a change worth saving. The canvas claims ``OWNS_WHEEL`` (the editor always wants
    scroll-to-zoom), so a scroll over the canvas zooms even when the dialog around it is
    tall enough to scroll itself; Alt+wheel is still the documented escape hatch back to
    the page. An embedder that shows the canvas as a still picture (`GeometryPreview`)
    makes it transparent to the mouse instead, so Qt delivers the wheel to the widget
    around it and the page keeps scrolling over the picture.
    """

    _HIT_RADIUS = 9.0  # px; how close a click must be to grab/remove a vertex
    _MIN_ZOOM = 1.0  # the fitted scale — never zoom out past the whole frame
    _MAX_ZOOM = (
        16.0  # `ImageView`'s own ceiling — past it a 1080p frame is a few px wide
    )
    _ZOOM_STEP = 1.25  # per button click or wheel notch, `ImageView`'s own step
    _DRAG_SLOP_PX = 4  # a right press that moves less than this is a click, not a drag

    # emitted (normalized x, y) when the frame is clicked with nothing editable — the
    # manager's "click the image to start a region" affordance.
    request_new = Signal(float, float)

    # Emitted whenever a drag, a click or an undo changes the shape being edited. Without
    # it the dialog only learns of an edit when something *else* makes it re-read the
    # canvas — scrubbing, or picking another definition — so Save stayed greyed out after
    # a change and lit up when you moved the frame, which reads as the opposite of what
    # happened.
    edited = Signal()

    # Emitted whenever the zoom level changes, so the dialog's "100%" readout and the
    # in/out buttons' enabled state stay in step with the canvas rather than a second copy
    # of the same number.
    zoom_changed = Signal(float)

    def __init__(
        self,
        image: np.ndarray,
        points: list[tuple[float, float]] | None = None,
        max_points: int | None = None,
    ):
        super().__init__()
        height, width, _ = image.shape
        self._width = width
        self._height = height
        self._pixmap = QPixmap.fromImage(
            QImage(
                image.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888
            )
        )
        self._points: list[tuple[float, float]] = list(points or [])
        self._max_points = max_points  # None = polygon (unbounded); 1 = a point
        self._drag_index: int | None = None
        self._hover_index: int | None = None
        self._zoom = 1.0
        # The normalized (0-1) image point currently centered in the widget. Panning and
        # zooming both act on this rather than on a pixel offset, so a resize (the dialog
        # window itself, or a splitter) recomputes the same view instead of drifting.
        self._center = QPointF(0.5, 0.5)
        self._pan_anchor: QPointF | None = None
        # Whether the active right-button press has moved past the click/drag slop yet —
        # once it has, releasing must not also delete the vertex the press started on.
        self._pan_dragged = False
        # The vertex a right *click* (no drag) would delete, captured at press time so a
        # small press-tremor while aiming at one cannot delete a different vertex than the
        # one under the cursor when the button went down.
        self._right_press_index: int | None = None
        # Without this, `PageScroll` (every dialog's body is wrapped in one) takes the wheel
        # for itself whenever the dialog is tall enough to scroll, and this canvas never
        # sees it at all — a scroll-to-zoom that silently stops working past a certain
        # window size. Unconditional, unlike `ImageView.zoomable`: this canvas always wants
        # the wheel, there is no non-zooming variant of it.
        self.setProperty(OWNS_WHEEL, True)
        self._editable = True  # False = read-only interpolated "ghost" polygon
        self._prominent = False  # read-only but a solid, clearly-visible outline
        self._allow_topology = True  # False = drag-only (vertex count locked)
        # The colour of the shape being edited. It is the *definition's* colour, not a
        # fixed "edit red": the swatch beside a definition and the outline on the frame
        # are one claim, and a canvas that ignores it makes them disagree until you save.
        self._edit_color = QColor(255, 80, 80)
        # Draw context shapes grey rather than in their own colours. With one definition
        # selected, colour means "this is the one you are editing"; everything else is
        # background. With none selected every shape keeps its colour, which is the
        # read-only overview.
        self._dim_context = False
        # sibling regions drawn behind the editable one (multi-region manager);
        # each is (points, QColor, is_point, visible)
        self._context_shapes: list[tuple] = []
        # Set when the backdrop could not be decoded. The last good pixmap stays in
        # ``_pixmap`` (so ``_image_rect`` keeps its geometry), but painting it under a
        # polygon would be a lie: you would be drawing an arena onto some other frame.
        self._unavailable: str | None = None
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)  # hover highlight without a pressed button

    def set_image(self, image: np.ndarray) -> None:
        height, width, _ = image.shape
        self._width = width
        self._height = height
        self._pixmap = QPixmap.fromImage(
            QImage(
                image.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888
            )
        )
        self._unavailable = None
        self.update()

    def set_unavailable(self, text: str) -> None:
        """Show ``text`` instead of a frame, and stop taking edits until one decodes.

        A dropped decode used to leave the previous frame on screen, so the polygon you
        drew next was placed against a picture that is not the one you scrubbed to.
        """
        self._unavailable = text
        self._drag_index = None
        self._hover_index = None
        self.update()

    def is_unavailable(self) -> bool:
        return self._unavailable is not None

    def set_polygon(
        self,
        points: list[tuple[float, float]],
        editable: bool,
        *,
        prominent: bool = False,
    ) -> None:
        """Show ``points``. ``editable`` enables interactive editing; for a
        read-only frame, ``prominent`` draws a solid (non-grabbable) outline (a
        keyframe being viewed) vs the dimmed dashed "ghost" of an interpolated
        frame."""
        self._points = list(points)
        self._editable = editable
        self._prominent = prominent
        self._hover_index = None
        self.update()

    def set_allow_topology(self, allow: bool) -> None:
        self._allow_topology = allow

    def set_edit_color(self, color: QColor) -> None:
        """Draw the editable shape in ``color`` — the definition's own, so the list
        swatch and the outline agree before anything is saved."""
        self._edit_color = QColor(color)
        self.update()

    def set_dim_context(self, dim: bool) -> None:
        """Grey the sibling shapes, so the one being edited is the only coloured thing."""
        self._dim_context = dim
        self.update()

    def set_max_points(self, max_points: int | None) -> None:
        """How many vertices the shape being edited may hold — ``None`` for an unbounded
        polygon, 1 for a point of interest, 2 for a calibration line.

        A constructor argument until one canvas had to serve every kind: the spatial
        workspace switches between a polygon, a point and a two-point line as the
        selection moves, and a canvas holding the max it was born with would let a click
        put a third vertex on a calibration.
        """
        self._max_points = max_points

    def set_context_shapes(self, shapes: list[tuple]) -> None:
        """Non-editable sibling shapes drawn behind the editable one — the multi-region
        manager's other regions. Each is ``(points, QColor, is_point, visible)`` with
        ``points`` already interpolated to the current frame (normalized [0, 1])."""
        self._context_shapes = list(shapes)
        self.update()

    _CONTEXT_GREY = QColor(150, 150, 150)

    def _paint_context(self, painter: QPainter, rect: QRectF) -> None:
        for points, color, is_point, visible in self._context_shapes:
            if not visible or not points:
                continue
            if self._dim_context:
                color = self._CONTEXT_GREY
            widget_points = [
                QPointF(rect.x() + x * rect.width(), rect.y() + y * rect.height())
                for x, y in points
            ]
            if is_point:
                self._paint_point(painter, widget_points[0], color)
                continue
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.drawPolygon(QPolygonF(widget_points))
            painter.setBrush(QBrush(color))
            for point in widget_points:
                painter.drawEllipse(point, 3.0, 3.0)

    def _base_scale(self) -> float:
        """The fit-to-widget scale — zoom 1.0's scale, before the zoom multiplier."""
        return min(self.width() / self._width, self.height() / self._height)

    def _image_rect(self) -> QRectF:
        """Where the (possibly zoomed and panned) image draws, in widget pixels.

        At the default ``_zoom == 1.0`` / ``_center == (0.5, 0.5)`` this is exactly the old
        fit-and-center rect — every caller (hit-testing, painting, normalizing a click) reads
        the view through this one rect, so zoom and pan need no changes anywhere else.
        """
        scale = self._base_scale() * self._zoom
        draw_w = self._width * scale
        draw_h = self._height * scale
        return QRectF(
            self.width() / 2 - self._center.x() * draw_w,
            self.height() / 2 - self._center.y() * draw_h,
            draw_w,
            draw_h,
        )

    # -- zoom / pan ----------------------------------------------------------- #
    def zoom_level(self) -> float:
        return self._zoom

    def can_zoom_in(self) -> bool:
        return self._zoom < self._MAX_ZOOM - 1e-9

    def can_zoom_out(self) -> bool:
        return self._zoom > self._MIN_ZOOM + 1e-9

    def _widget_center(self) -> QPointF:
        return QPointF(self.width() / 2, self.height() / 2)

    def zoom_in(self) -> None:
        self._set_zoom_centered(self._zoom * self._ZOOM_STEP, self._widget_center())

    def zoom_out(self) -> None:
        self._set_zoom_centered(self._zoom / self._ZOOM_STEP, self._widget_center())

    def reset_zoom(self) -> None:
        """Back to fit-to-widget, centered — the way this canvas always looked before zoom
        existed, and the escape hatch once a pan has wandered off the shape."""
        changed = self._zoom != 1.0
        self._zoom = 1.0
        self._center = QPointF(0.5, 0.5)
        self.update()
        if changed:
            self.zoom_changed.emit(self._zoom)

    def _set_zoom_centered(self, new_zoom: float, anchor: QPointF) -> None:
        """Change the zoom while keeping the image point under ``anchor`` fixed on screen —
        so scrolling over a vertex zooms toward that vertex, not toward whatever the fit
        happened to center."""
        new_zoom = min(max(new_zoom, self._MIN_ZOOM), self._MAX_ZOOM)
        if new_zoom == self._zoom:
            return
        rect = self._image_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        nx = (anchor.x() - rect.x()) / rect.width()
        ny = (anchor.y() - rect.y()) / rect.height()
        self._zoom = new_zoom
        scale = self._base_scale() * self._zoom
        draw_w = self._width * scale
        draw_h = self._height * scale
        new_rect_x = anchor.x() - nx * draw_w
        new_rect_y = anchor.y() - ny * draw_h
        self._center = QPointF(
            (self.width() / 2 - new_rect_x) / draw_w,
            (self.height() / 2 - new_rect_y) / draw_h,
        )
        self.update()
        self.zoom_changed.emit(self._zoom)

    def _pan_by(self, delta: QPointF) -> None:
        rect = self._image_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        # Clamped to [0, 1] in each axis: the point at the widget's own center is always
        # somewhere inside the image, so a drag can bring any part of the frame into view
        # but can never pan the whole picture off screen with no way back but Fit.
        self._center = QPointF(
            min(max(self._center.x() - delta.x() / rect.width(), 0.0), 1.0),
            min(max(self._center.y() - delta.y() / rect.height(), 0.0), 1.0),
        )
        self.update()

    def _normalized(self, pos: QPointF) -> tuple[float, float]:
        rect = self._image_rect()
        x = min(max((pos.x() - rect.x()) / rect.width(), 0.0), 1.0)
        y = min(max((pos.y() - rect.y()) / rect.height(), 0.0), 1.0)
        return x, y

    def _vertex_at(self, pos: QPointF) -> int | None:
        rect = self._image_rect()
        best, best_dist = None, self._HIT_RADIUS**2
        for i, (nx, ny) in enumerate(self._points):
            wx = rect.x() + nx * rect.width()
            wy = rect.y() + ny * rect.height()
            dist = (pos.x() - wx) ** 2 + (pos.y() - wy) ** 2
            if dist <= best_dist:
                best, best_dist = i, dist
        return best

    def mousePressEvent(self, event) -> None:
        if self._unavailable is not None:
            return  # no frame on screen — a click here would place a point blind
        pos = event.position()
        if event.button() == Qt.MouseButton.RightButton:
            # Deferred to release (see `mouseReleaseEvent`): a right click deletes the
            # vertex under it, a right drag pans instead, and only the release can tell
            # them apart. This runs whatever is selected — including the read-only
            # overview (nothing selected) and a topology-locked shape — because panning is
            # a view operation and has to work everywhere a right-click-to-delete alone
            # would not (it is refused below when there is nothing to delete anyway).
            self._pan_anchor = pos
            self._pan_dragged = False
            self._right_press_index = (
                self._vertex_at(pos)
                if self._editable and self._allow_topology
                else None
            )
            return
        if self._pan_anchor is not None:
            return  # a right gesture is in flight; a chorded left press is not a click
        if not self._editable:
            # nothing editable here — offer to start a new definition (click-to-start)
            if (
                event.button() == Qt.MouseButton.LeftButton
                and self._image_rect().contains(pos)
            ):
                self.request_new.emit(*self._normalized(pos))
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        index = self._vertex_at(pos)
        if index is not None:
            self._drag_index = index  # grab an existing vertex
        elif self._image_rect().contains(pos):
            if self._max_points == 1:
                self._points = [self._normalized(pos)]  # place/replace the one point
                self._changed()
            elif self._allow_topology and (
                self._max_points is None or len(self._points) < self._max_points
            ):
                self._points.append(
                    self._normalized(pos)
                )  # add a vertex (up to the cap)
                self._changed()

    def mouseMoveEvent(self, event) -> None:
        if self._unavailable is not None:
            return
        pos = event.position()
        if self._pan_anchor is not None:
            delta = pos - self._pan_anchor
            if self._pan_dragged or delta.manhattanLength() > self._DRAG_SLOP_PX:
                self._pan_dragged = True
                self._pan_by(delta)
                self._pan_anchor = pos
            return
        if not self._editable:
            return
        if self._drag_index is not None:
            self._points[self._drag_index] = self._normalized(pos)
            self._changed()
            return
        hover = self._vertex_at(pos)
        if hover != self._hover_index:
            self._hover_index = hover
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.RightButton
            and self._pan_anchor is not None
        ):
            if not self._pan_dragged and self._right_press_index is not None:
                del self._points[self._right_press_index]
                self._hover_index = None
                self._changed()
            self._pan_anchor = None
            self._pan_dragged = False
            self._right_press_index = None
        self._drag_index = None

    def wheelEvent(self, event) -> None:
        if self._unavailable is not None or (
            event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            event.ignore()  # Alt is the page's documented escape hatch (gui/scroll.py)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = self._ZOOM_STEP ** (delta / 120.0)
        self._set_zoom_centered(self._zoom * factor, event.position())
        event.accept()

    def _changed(self) -> None:
        """Repaint, and tell whoever is holding this canvas that the shape moved."""
        self.update()
        self.edited.emit()

    def undo(self) -> None:
        if self._points:
            self._points.pop()
            self._hover_index = None
            self._changed()

    def clear(self) -> None:
        self._points = []
        self._hover_index = None
        self._changed()

    def normalized_polygon(self) -> list[tuple[float, float]]:
        return [_rounded(p) for p in self._points]

    def _widget_points(self, rect: QRectF) -> list[QPointF]:
        return [
            QPointF(rect.x() + x * rect.width(), rect.y() + y * rect.height())
            for x, y in self._points
        ]

    def _paint_point(self, painter: QPainter, p: QPointF, color: QColor) -> None:
        """A crosshair marker (ring + center dot + cross) so a POI reads as a
        point, not a stray polygon vertex."""
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawEllipse(p, 7.0, 7.0)
        painter.drawLine(QPointF(p.x() - 11, p.y()), QPointF(p.x() + 11, p.y()))
        painter.drawLine(QPointF(p.x(), p.y() - 11), QPointF(p.x(), p.y() + 11))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(p, 2.0, 2.0)

    #: Grey — same family as the interpolated "ghost" outline, and a member of the
    #: neutral status colours: this glyph is a statement about *absence*, not a status.
    _NO_FRAME_COLOR = QColor(170, 170, 170)

    def _paint_no_frame_glyph(self, painter: QPainter, box: QRectF) -> None:
        """A generic "no picture" mark — a frame with a plain mountains-and-sun
        silhouette, the same shape a browser draws for an image that failed to load.
        Drawn with the same primitives the point-of-interest marker already uses
        (:meth:`_paint_point`), not an icon font — there is no icon set in this app
        (``requirements.py``, ``the_package_uses_no_icon_sets``).
        """
        painter.setPen(QPen(self._NO_FRAME_COLOR, 2))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawRect(box)
        w, h = box.width(), box.height()
        sun = QPointF(box.left() + w * 0.28, box.top() + h * 0.28)
        painter.drawEllipse(sun, w * 0.11, w * 0.11)
        ridge = QPolygonF(
            [
                QPointF(box.left() + w * 0.12, box.bottom() - h * 0.18),
                QPointF(box.left() + w * 0.4, box.top() + h * 0.42),
                QPointF(box.left() + w * 0.58, box.bottom() - h * 0.32),
                QPointF(box.left() + w * 0.76, box.top() + h * 0.5),
                QPointF(box.right() - w * 0.12, box.bottom() - h * 0.18),
            ]
        )
        painter.drawPolyline(ridge)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        rect = self._image_rect()
        if self._unavailable is not None:
            painter.fillRect(self.rect(), self.palette().window())
            side = max(36.0, min(self.width(), self.height()) * 0.22)
            glyph = QRectF(0, 0, side, side)
            glyph.moveCenter(QPointF(self.width() / 2, self.height() / 2 - side * 0.45))
            self._paint_no_frame_glyph(painter, glyph)
            painter.setPen(QPen(self.palette().windowText().color()))
            text_rect = self.rect().adjusted(0, int(glyph.bottom()) + 10, 0, 0)
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                self._unavailable,
            )
            return
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        self._paint_context(painter, rect)  # sibling regions behind the editable one
        if not self._points:
            return
        widget_points = self._widget_points(rect)
        if self._max_points == 1:  # point of interest — a crosshair, never a polygon
            color = (
                self._edit_color
                if (self._editable or self._prominent)
                else QColor(170, 170, 170)
            )
            self._paint_point(painter, widget_points[0], color)
            return
        if not self._editable:
            # read-only keyframe view — solid outline, not grabbable
            if self._prominent:
                painter.setPen(QPen(self._edit_color, 2))
                painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                painter.drawPolygon(QPolygonF(widget_points))
                painter.setBrush(QBrush(self._edit_color))
                for point in widget_points:
                    painter.drawEllipse(point, 4.0, 4.0)
                return
            # interpolated "ghost" — dimmed, dashed
            pen = QPen(QColor(170, 170, 170), 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.drawPolygon(QPolygonF(widget_points))
            for point in widget_points:
                painter.drawEllipse(point, 3.0, 3.0)
            return
        painter.setPen(QPen(self._edit_color, 2))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawPolygon(QPolygonF(widget_points))  # closes automatically
        for i, point in enumerate(widget_points):
            highlighted = i == self._hover_index
            painter.setBrush(
                QBrush(QColor(255, 220, 0) if highlighted else self._edit_color)
            )
            radius = 6.0 if highlighted else 4.0
            painter.drawEllipse(point, radius, radius)


#: Why the edit buttons are off on a frame between keyframes. A shape only *has* points to undo
#: or clear where it was drawn; in between it is interpolated, so there is nothing of this frame's
#: own to edit.
_INTERPOLATED_FRAME = (
    "this frame is interpolated — scrub to a keyframe to edit its points"
)


def _no_delete_reason(is_keyframe: bool) -> str:
    """Two different refusals wearing one disabled button.

    Not a keyframe at all, and the last keyframe left, are different things to fix, and a single
    reason covering both would be right about one of them. The button is the same button either
    way, so the reason has to be computed where the state is.
    """
    if not is_keyframe:
        return "this frame is interpolated — scrub to a keyframe to delete it"
    return "a shape needs one keyframe; add another before deleting this one"


@dataclass(frozen=True)
class SpatialKind:
    """One kind of spatial definition, as the workspace needs to know it (§10).

    The four kinds are one concept a user sets up together — where the camera looks, how
    big a pixel is, which zones matter, which landmarks matter — and they were four page
    sections, five dialogs and four editing idioms. This table is what lets one dialog
    render all of them: everything that differs between a field of view and a point of
    interest is a field here, so the workspace holds no `if kind ==` beyond the save.
    """

    id: str
    #: Heading a person reads: "Field of view".
    label: str
    #: Singular noun inside a sentence: "field of view".
    noun: str
    #: What a new one is called before it is renamed.
    default_base: str
    min_points: int
    #: Vertices the shape may hold — `None` is an unbounded polygon.
    max_points: int | None
    #: ROIs/POIs may be present for part of the video; a field of view is always in effect
    #: and a calibration is not a shape in time at all.
    has_active: bool
    #: Whether the shape can move over the video. A calibration cannot: it is two points
    #: and a distance, stored with no frame at all, so the editor edits it on whatever frame
    #: is showing rather than on a keyframe (`SceneGeometryDialog._edit_frame`).
    keyframed: bool
    #: At most one may exist (the calibration). The others are lists.
    singleton: bool
    #: How a setup arriving from elsewhere is refused — the grain at which a collision is
    #: judged. ``"kind"``: the target having *any* of this kind refuses the lot.
    #: ``"definition"``: only the names already taken are refused, the rest arrive.
    #:
    #: It follows the kind rather than the direction, because what makes a second one a
    #: problem differs. A second **field of view** breaks the batch tracking picker, which
    #: intersects arena *ids* across recordings — so two arenas means one dropdown naming
    #: different polygons on different recordings. A second **calibration** is meaningless;
    #: there is one pixel scale. Regions and points are sets by design, and refusing a
    #: whole set because one name clashes would kill "pull one more zone from a template",
    #: which is what `recording import-spatial` has always been for.
    collision_grain: str
    #: Carries what the DATA PIPELINE / ANALYSIS page split used to say. It is a fact about
    #: the kind, not about where its widget was drawn, which is why the split could not
    #: survive the merge and this could.
    scope: str
    #: The kind's own chart-theme hue, by name (``ChartTheme.hue``) — so it keeps its
    #: colour as the list grows *and* when the palette's slots are reordered.
    hue: str
    #: The sentence the recording page prints under this kind's heading. It lives on the
    #: kind rather than in the page so the four sections' descriptions could survive being
    #: merged into one section — losing them was the cost of the merge and not worth paying.
    description: str


SPATIAL_KINDS: tuple[SpatialKind, ...] = (
    SpatialKind(
        id="arena",
        keyframed=True,
        collision_grain="kind",
        description="Optional polygon that defines the region of the video used for detection and tracking.",
        label="Field of view",
        noun="field of view",
        default_base="field of view",
        min_points=3,
        max_points=None,
        has_active=False,
        singleton=False,
        scope="used by detection & tracking",
        hue="orange",
    ),
    SpatialKind(
        id="calibration",
        keyframed=False,
        collision_grain="kind",
        description="A known distance between two points, so analysis can report millimetres.",
        label="Distance calibration",
        noun="distance calibration",
        default_base="calibration",
        min_points=2,
        max_points=2,
        has_active=False,
        singleton=True,
        scope="analysis only",
        hue="blue",
    ),
    SpatialKind(
        id="roi",
        keyframed=True,
        collision_grain="definition",
        description="Named polygon zones for location analysis.",
        label="Regions of interest",
        noun="region of interest",
        default_base="region",
        min_points=3,
        max_points=None,
        has_active=True,
        singleton=False,
        scope="analysis only",
        hue="green",
    ),
    SpatialKind(
        id="point",
        keyframed=True,
        collision_grain="definition",
        description="Named landmark points for location analysis.",
        label="Points of interest",
        noun="point of interest",
        default_base="point",
        min_points=1,
        max_points=1,
        has_active=True,
        singleton=False,
        scope="analysis only",
        hue="vermillion",
    ),
)

SPATIAL_KINDS_BY_ID: dict[str, SpatialKind] = {k.id: k for k in SPATIAL_KINDS}

#: Palette slots not claimed as a kind's own colour, for a kind's second and later
#: definitions. Reserving the bases is what keeps two *kinds* from ever sharing one:
#: shifting by the index instead handed a recording's second region the calibration's
#: blue, so the picture drew a zone and a measurement as the same thing.
_EXTRA_SLOTS = tuple(
    name for name in HUE_NAMES if name not in {k.hue for k in SPATIAL_KINDS}
)


def region_color(kind: SpatialKind, index_within_kind: int) -> QColor:
    """A kind's own colour for its first definition, an unreserved one after that.

    Colouring by position in a flat list moves every colour when a definition is added
    above, so the field of view would change colour because you drew a region. Here a
    kind keeps its colour across recordings, which is what makes the swatch on the
    recording page, the outline in :class:`SceneGeometryDialog`'s canvas, and the static
    :class:`GeometryPreview` all draw the same claim rather than three derivations of it.
    """
    # Canonical (light) steps: a region is drawn on a video frame, not on a chart surface.
    if index_within_kind == 0 or not _EXTRA_SLOTS:
        return QColor(LIGHT.hue(kind.hue))
    slot = _EXTRA_SLOTS[(index_within_kind - 1) % len(_EXTRA_SLOTS)]
    return QColor(LIGHT.hue(slot))


def load_scene_regions(recording) -> list[dict]:
    """Every kind, flattened into one list, each row carrying the kind it came from.

    ROIs and POIs keep their ``active`` window. The manager this replaces read it
    nowhere and wrote it nowhere, so opening it and pressing Save all silently turned
    ``frames 100–900`` into ``whole video`` — a definition quietly starting to apply
    to footage the user had excluded.

    Module-level (rather than a method on :class:`SceneGeometryDialog`) so a read-only
    caller — :class:`GeometryPreview` — can build the same list without a dialog
    instance around it.
    """
    regions: list[dict] = []
    for entry in recording.arenas:
        locked = recording.arena_dependents(entry["id"])
        regions.append(
            {
                "kind": "arena",
                "name": entry["name"],
                "keyframes": {
                    f: list(v) for f, v in recording.arena_keyframes(entry["id"])
                }
                or {0: []},
                "active": None,
                "visible": True,
                "locked": (
                    "used by tracking "
                    f"{'runs' if len(locked) > 1 else 'run'} {', '.join(locked)} — "
                    "delete those runs to edit it"
                    if locked
                    else None
                ),
            }
        )
    cal = recording.distance_calibration
    if cal is not None:
        points = [(float(x), float(y)) for x, y in (cal.get("points") or [])]
        regions.append(
            {
                "kind": "calibration",
                "name": "Distance calibration",
                "keyframes": {0: points},
                "active": None,
                "visible": True,
                "locked": None,
                "distance_mm": float(cal.get("distance_mm") or 0.0),
                "pixels_per_mm": float(cal.get("pixels_per_mm") or 0.0),
            }
        )
    for kind, entries, loader in (
        ("roi", recording.rois, recording.roi_keyframes),
        ("point", recording.points, recording.point_keyframes),
    ):
        for entry in entries:
            act = entry.get("active")
            regions.append(
                {
                    "kind": kind,
                    # The stored id rides along so Save hands it back: a region the user
                    # kept stays the same `roi_NNN` (`storage.spatial.replace_defs`).
                    "id": entry["id"],
                    "name": entry["name"],
                    "keyframes": {f: list(v) for f, v in loader(entry["id"])}
                    or {0: []},
                    "active": (act["start"], act["end"]) if act else None,
                    "visible": True,
                    "locked": None,
                }
            )
    return regions


def region_ghost(
    reg: dict, frame: int, *, respect_active: bool = False
) -> list[tuple[float, float]]:
    """Where ``reg`` sits at ``frame``, or nothing.

    ``respect_active`` is on for a *context* shape (drawn but not being edited) and off
    for the definition someone is actively editing in :class:`SceneGeometryDialog`. A
    region declared active for frames 3000–5000 does not exist at frame 0, and drawing
    it there is the picture disagreeing with what analysis will do. Module-level for the
    same reason as :func:`load_scene_regions` — :class:`GeometryPreview` has no
    definition of its own being edited, so every shape it draws goes through this branch.
    """
    if respect_active and reg["active"] is not None:
        start, end = reg["active"]
        if not start <= frame <= end:
            return []
    defined = [(f, v) for f, v in sorted(reg["keyframes"].items()) if v]
    if not defined:
        return []
    pk = PolygonKeyframes(n_vertices=len(defined[0][1]), keyframes=defined)
    return [(float(x), float(y)) for x, y in polygon_at_frame(pk, frame)]


def geometry_snapshot(recording, frame: int) -> list[tuple]:
    """Every defined region on ``recording`` at ``frame``, as
    :meth:`ArenaCanvas.set_context_shapes` wants them — the read-only "nothing selected"
    view :class:`SceneGeometryDialog` opens to, built for a caller with no dialog
    instance around it (:class:`GeometryPreview`, the recording page's static preview).
    """
    shapes = []
    seen: dict[str, int] = {}
    for reg in load_scene_regions(recording):
        index = seen.get(reg["kind"], 0)
        seen[reg["kind"]] = index + 1
        points = region_ghost(reg, frame, respect_active=True)
        if not points:
            continue
        kind = SPATIAL_KINDS_BY_ID[reg["kind"]]
        shapes.append((points, region_color(kind, index), kind.max_points == 1, True))
    return shapes


class SceneGeometryDialog(Dialog):
    """Every piece of a recording's spatial setup, on one frame (§10).

    Field of view, distance calibration, regions and points of interest were four page
    sections, five dialogs and four editing idioms for one thing a person sets up in one
    sitting: where the camera looks, how big a pixel is, which zones matter, which
    landmarks matter. Twenty of a recording page's twenty-six buttons were spatial.

    **Why one canvas is the whole point, not a tidy-up.** The four editors could not see
    each other, so an ROI was drawn on a frame that did not show the arena — which is why
    a separate read-only *Preview spatial setup* dialog had to exist at all. Here the
    other definitions are always drawn as context, so preview is this dialog's resting
    state (select nothing) rather than a fifth surface to keep in step. It also decodes
    the frame once instead of four times.

    Select a definition to edit it; its siblings stay on screen dimmed. The checkbox hides
    one you want out of the way. Transactional: **Save all** writes every kind, **Cancel**
    discards everything.

    A field of view a tracking run depends on is shown, drawn, and not editable — the run
    recorded that polygon, so changing it would make the run's own record a lie. The lock
    is `Recording.clear_arenas`'s, not re-implemented here: the save skips arenas entirely
    when they are untouched, so a locked one never reaches it.
    """

    def __init__(self, recording, experiment=None, parent=None):
        super().__init__(parent)
        self._recording = recording
        self._experiment = experiment
        self.setWindowTitle(f"Scene geometry — {recording.name}")
        self.resize(1040, 740)  # fits 1440x810 (gui_design.md §3.1)
        self._part_counts = recording.part_frame_counts
        self._total = sum(self._part_counts)
        self._frame = 0
        self._selected: int | None = None
        self._loading = False  # suppress list/checkbox signals during rebuilds
        self._regions = self._load_regions()
        # What each kind looked like on open, to tell "untouched" from "edited".
        self._loaded = {k.id: self._signature(k.id) for k in SPATIAL_KINDS}

        outer = self.body()
        body = QHBoxLayout()

        left = QVBoxLayout()
        zoom_row = QHBoxLayout()
        self._zoom_out_btn = QPushButton("Zoom out")
        self._zoom_in_btn = QPushButton("Zoom in")
        self._zoom_fit_btn = QPushButton("Fit")
        self._zoom_label = QLabel()
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for button in (self._zoom_out_btn, self._zoom_label, self._zoom_in_btn):
            zoom_row.addWidget(button)
        zoom_row.addWidget(self._zoom_fit_btn)
        zoom_row.addStretch()
        zoom_row.addWidget(QLabel("Scroll to zoom · right-drag to pan"))
        left.addLayout(zoom_row)
        self._canvas = ArenaCanvas(_BLANK_FRAME)
        self._canvas.edited.connect(self._on_canvas_edited)
        self._canvas.zoom_changed.connect(self._on_zoom_changed)
        self._zoom_out_btn.clicked.connect(self._canvas.zoom_out)
        self._zoom_in_btn.clicked.connect(self._canvas.zoom_in)
        self._zoom_fit_btn.clicked.connect(self._canvas.reset_zoom)
        self._on_zoom_changed(self._canvas.zoom_level())
        left.addWidget(self._canvas, 1)
        # Playable. `_on_scrub` calls `_capture_current()`, which looks like a reason to
        # refuse — but that only writes to a frame that is *already* a keyframe, and writes
        # back what `_sync` just put on the canvas, so playing across one re-saves it as
        # itself. Finding the frame you want to draw on is easier watching than stepping.
        self._scrubber = FrameScrubber(playable=True)
        if recording.video_fps:  # mtime-cached yaml read
            self._scrubber.set_playback_rate(recording.video_fps)
        self._scrubber.set_total(self._total)
        # The time beside the frame number, from the timestamp map — off the UI thread, since
        # it parses every part's map. The row shows no time until it lands.
        self._times_call = BackgroundCall(
            recording.frame_timestamps, receiver=self  # off-thread
        )
        self._times_call.done.connect(self._on_times)
        self._times_call.failed.connect(
            lambda _exc: None
        )  # no time shown, never a guess
        self._times_call.start()
        self._scrubber.set_frame(0, decode=False)
        self._scrubber.frame_changed.connect(self._on_scrub)
        self._scrubber.set_decoder(self._frame_image)
        self._scrubber.rendered.connect(
            lambda _f, payload: _paint_decoded(self._canvas, payload)
        )
        self._decode_busy = decode_overlay(self._scrubber, self._canvas)
        left.addWidget(self._scrubber)
        self._kf_label = QLabel()
        left.addWidget(self._kf_label)

        kf_row = QHBoxLayout()
        self._set_kf = QPushButton("Set keyframe here")
        self._set_kf.clicked.connect(self._set_keyframe)
        self._del_kf = QPushButton("Delete keyframe")
        self._del_kf.clicked.connect(self._delete_keyframe)
        self._undo = QPushButton("Undo point")
        self._undo.clicked.connect(self._on_undo)
        self._clear = QPushButton("Clear points")
        self._clear.clicked.connect(self._on_clear)
        for button in (self._set_kf, self._del_kf, self._undo, self._clear):
            kf_row.addWidget(button)
        kf_row.addStretch()
        left.addLayout(kf_row)

        form = QFormLayout()
        self._name = QLineEdit()
        self._name.editingFinished.connect(self._on_rename)
        form.addRow("Name:", self._name)
        # Calibration only: the real-world distance between its two points.
        self._distance = FloatSpin()
        self._distance.setRange(0.0, 1_000_000.0)
        self._distance.valueChanged.connect(self._on_distance_changed)
        self._unit = QComboBox()
        self._unit.addItems(["mm", "cm"])
        self._unit.currentTextChanged.connect(lambda _t: self._on_distance_changed())
        dist_row = QHBoxLayout()
        dist_row.addWidget(self._distance, 1)
        dist_row.addWidget(self._unit)
        self._distance_holder = QWidget()
        self._distance_holder.setLayout(dist_row)
        self._distance_label = QLabel("Known distance:")
        form.addRow(self._distance_label, self._distance_holder)
        left.addLayout(form)
        self._build_active_controls(left)
        body.addLayout(left, 3)

        right = QVBoxLayout()
        # Adding is the only way to start something, so it leads rather than sitting under
        # the list: clicking the frame no longer begins a shape, which was a click people
        # made by accident and could not undo without noticing a region had appeared.
        #
        # Four plain buttons rather than one menu. A `QToolButton` stretched to the panel
        # width put its small label in the middle and Qt's menu indicator at the far right
        # edge, 200px away — it read as a dropdown, not a button. With four kinds there is
        # nothing to collapse into a menu anyway, and naming them costs a click less.
        self._add_buttons: dict[str, QPushButton] = {}
        for kind in SPATIAL_KINDS:
            button = QPushButton(f"+ {kind.label}")
            button.clicked.connect(lambda _c=False, k=kind.id: self._add_region(k))
            right.addWidget(button)
            self._add_buttons[kind.id] = button
        right.addWidget(QLabel("Definitions:"))
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_select)
        self._list.itemChanged.connect(self._on_item_changed)
        right.addWidget(self._list, 1)
        self._delete = QPushButton("Delete")
        self._delete.clicked.connect(self._delete_region)
        right.addWidget(self._delete)
        body.addLayout(right, 1)
        outer.addLayout(body, 1)

        # Carrying a setup between recordings and to/from a CSV belongs with the thing it
        # carries. On the page they were four buttons beside four sections; here there is
        # one place that owns all four kinds, which is the only place they were ever about.
        actions = QHBoxLayout()
        self._copy = QPushButton("Copy to other recordings…")
        self._copy.clicked.connect(self._copy_to_recordings)
        self._import = QPushButton("Import from…")
        self._import.clicked.connect(self._import_spatial)
        self._export = QPushButton("Export as CSV…")
        self._export.clicked.connect(self._export_spatial)
        for button in (self._copy, self._import, self._export):
            actions.addWidget(button)
        actions.addStretch()
        outer.addLayout(actions)

        # Save keeps the window open. It used to close, which fought the actions beside
        # it: Copy / Import / Export all refuse while edits are pending, so saving to
        # unblock them shut the dialog and you had to reopen it to use them. Save greying
        # out is also the only thing here that says "you have unsaved changes".
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
        )
        self._save = box.button(QDialogButtonBox.StandardButton.Save)
        self._save.setText("Save")
        self._close_button = box.button(QDialogButtonBox.StandardButton.Close)
        self._save.clicked.connect(self._save_all)
        box.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self._close)
        self.set_buttons(
            box
        )  # pinned outside the scroll — the way out never scrolls away
        self._sync_actions()
        self._sync_dirty()

        self._rebuild_list()
        self._sync()
        if self._total:
            self._scrubber.set_frame(0, decode=True)  # first frame off-thread
        else:
            # No video: `_locate` would index an empty part list, the scrubber would
            # swallow the error, and the canvas would keep its blank placeholder — still
            # clickable, so you could draw a polygon against a black rectangle and save
            # it. A surface that cannot show the frame says why and refuses input it
            # cannot show (tests/gui/test_frame_source.py).
            self._canvas.set_unavailable("No video imported")
            self._scrubber.setEnabled(False)

    # -- load / persist ------------------------------------------------------ #
    def _on_times(self, t_ns) -> None:
        if len(t_ns) == self._total:
            self._scrubber.set_timestamps(t_ns)

    def _load_regions(self) -> list[dict]:
        return load_scene_regions(self._recording)

    def _signature(self, kind_id: str) -> list[tuple]:
        """A comparable snapshot of one kind, so an untouched kind is never rewritten.

        This is what keeps *not touching something* from being a write. It matters for two
        kinds in particular, both of which would otherwise fail a save that had nothing to
        do with them:

        - **Arenas.** Editing a field of view means clearing and redefining, which
          `clear_arenas` refuses while any tracking run holds one. Saving a region on a
          recording whose arena is locked must not trip that.
        - **The calibration.** One set from the CLI's ``--pixels-per-mm`` has *no points* —
          there is no polygon to put on a canvas. Validating it like a fresh pick would
          refuse ("needs exactly two points") and block the whole save; treating its empty
          canvas as a deletion would silently drop it. Untouched means untouched.
        """
        return [
            (
                r["name"],
                tuple(
                    (f, tuple(map(_rounded, v)))
                    for f, v in sorted(r["keyframes"].items())
                ),
                r["active"],
                r.get("distance_mm"),
            )
            for r in self._regions
            if r["kind"] == kind_id
        ]

    def _save_all(self) -> None:
        self._capture_current()
        title = self.windowTitle()
        by_kind: dict[str, list[dict]] = {k.id: [] for k in SPATIAL_KINDS}
        for reg in self._regions:
            by_kind[reg["kind"]].append(reg)

        # Validate every kind before writing any of them (Tenet 10) — a partial save
        # would leave the recording in a state neither the user nor the dialog describes.
        prepared: dict[str, list[dict]] = {}
        for kind in SPATIAL_KINDS:
            if kind.id == "calibration":
                continue
            defs = []
            for reg in by_kind[kind.id]:
                name = reg["name"].strip()
                keyframes = [(f, v) for f, v in sorted(reg["keyframes"].items()) if v]
                if not name:
                    return self._refuse(title, f"Every {kind.noun} needs a name.")
                if not keyframes or any(len(v) < kind.min_points for _, v in keyframes):
                    need = (
                        "a point"
                        if kind.min_points == 1
                        else f"at least {kind.min_points} points"
                    )
                    return self._refuse(
                        title, f"'{name}' needs {need} on each keyframe."
                    )
                if len({len(v) for _, v in keyframes}) != 1:
                    return self._refuse(
                        title,
                        f"'{name}' keyframes must have the same number of points.",
                    )
                entry = {"name": name, "keyframes": keyframes}
                if reg.get("id"):
                    entry["id"] = reg["id"]
                if kind.has_active:
                    entry["active"] = reg["active"]
                defs.append(entry)
            # Then the rest of what the write would refuse — a name used twice, above all —
            # asked of the storage layer's own check, so this list cannot drift from the
            # writer's. Found here it refuses before anything is written; found by the write
            # it came after the field of view was already stored (#35 finding 13).
            try:
                self._recording.check_spatial_defs(kind.id, defs)
            except ValueError as exc:
                return self._refuse(title, str(exc))
            prepared[kind.id] = defs
        cal = by_kind["calibration"][0] if by_kind["calibration"] else None
        calibration_touched = (
            self._signature("calibration") != self._loaded["calibration"]
        )
        if cal is not None and calibration_touched:
            points = cal["keyframes"].get(0) or []
            if len(points) != 2:
                return self._refuse(
                    title, "The distance calibration needs exactly two points."
                )
            if not cal.get("distance_mm"):
                return self._refuse(
                    title, "The distance calibration needs a known distance."
                )
            try:
                self._recording.check_distance_calibration(
                    points, float(cal["distance_mm"])
                )
            except ValueError as exc:
                return self._refuse(title, f"The distance calibration: {exc}")

        try:
            # Everything the writes below can refuse about the setup itself was checked
            # above; what is left is the arena lock and the disk. Arenas first, and only
            # when they changed. First because `clear_arenas` refuses while any tracking
            # run holds an arena, and a refusal has to land before anything else is
            # committed — the dialog promises one transaction.
            #
            # Clearing and redefining the whole set is safe *because* that check is
            # whole-set: if it passes, no arena in this recording is named by any tracking
            # run, so the ids being reminted cannot invalidate a run's record of which
            # polygon it used. (`next_arena_id` takes the lowest free number, so rebuilding in
            # list order also hands back the same ids unless one was deleted.)
            if self._signature("arena") != self._loaded["arena"]:
                self._recording.clear_arenas()
                for entry in prepared["arena"]:
                    self._recording.define_arena(entry["name"], entry["keyframes"])
            self._recording.set_rois(prepared["roi"])
            self._recording.set_points(prepared["point"])
            if calibration_touched:
                if cal is None:
                    self._recording.clear_distance_calibration()
                else:
                    self._recording.set_distance_calibration(
                        cal["keyframes"][0], float(cal["distance_mm"])
                    )
        except (ValueError, FileNotFoundError, OSError) as exc:
            QMessageBox.critical(self, title, str(exc))
            return
        self._loaded = {k.id: self._signature(k.id) for k in SPATIAL_KINDS}
        self._sync_actions()
        self._sync_dirty()

    def _refuse(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _dirty(self) -> list[str]:
        """Which kinds differ from what was on disk when this dialog opened."""
        return [
            k.id for k in SPATIAL_KINDS if self._signature(k.id) != self._loaded[k.id]
        ]

    def _edited(self) -> None:
        """A handler changed a definition: tell Save.

        Every handler that writes into a definition ends here, or in :meth:`_sync`, which
        does the same. Four did not — rename, the known distance, the whole-video box and
        the active-window bounds — so Save stayed grey after them while the docs call a
        grey Save the sign that your edits are in (#35 finding 10). The rule is checked
        over this class's source by `tests/gui/test_scene_geometry.py`.
        """
        self._sync_dirty()

    def _sync_dirty(self) -> None:
        """Save is live only when there is something to save — the dialog's only sign that
        edits are pending, now that saving no longer closes it."""
        pending = self._dirty()
        set_available(self._save, bool(pending), "No changes to save")
        self._sync_actions()

    def _close(self) -> None:
        """Closing with edits pending asks first — Save no longer closes, so the only way
        out is this button, and it must not be a silent discard."""
        if self._dirty():
            answer = QMessageBox.question(
                self,
                self.windowTitle(),
                "Discard your unsaved changes to this recording's scene geometry?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.reject()

    # -- carrying a setup elsewhere ------------------------------------------ #
    # These import `gui.pages` inside the method on purpose. `pages` imports this module,
    # so a module-level import here would be a cycle; a deferred one leaves every
    # `detail.gui.*` module importable on its own, which `test_import_cycles` requires.
    def _sync_actions(self) -> None:
        """Say why an action is off rather than leaving it inert.

        Copy needs somewhere to copy to and something to copy; export needs something to
        write. All three are reachable with no video, because a normalized polygon carries
        to a recording whose footage has not been imported yet.
        """
        from detail.gui.pages import CopyToRecordingsDialog

        # `recording_ids` is the directory listing; `.recordings` would build a
        # `Recording` per id and read each one's `recording.yaml` to learn a name nobody
        # here needs — an O(recordings) sweep on the UI thread to answer a yes/no.
        others = (
            [r for r in self._experiment.recording_ids if r != self._recording.id]
            if self._experiment is not None
            else []
        )
        set_available(
            self._export, bool(self._regions), "Nothing defined on this recording yet"
        )
        if not others:
            set_available(self._copy, False, "No other recordings in this experiment")
        else:
            set_available(
                self._copy,
                CopyToRecordingsDialog.has_anything_to_copy(self._recording),
                "Nothing to copy yet — define something first",
            )

    def _copy_to_recordings(self) -> None:
        from detail.gui.pages import CopyToRecordingsDialog

        if self._unsaved_edits():
            return
        dialog = CopyToRecordingsDialog(self._recording, self._experiment, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        # Read the selection on this thread; `run` executes in a worker and must not touch
        # a widget the dialog owns.
        plan = dialog.plan()
        QMessageBox.information(
            self, "Copy to other recordings", dialog.run(plan, None)
        )

    def _import_spatial(self) -> None:
        from detail.gui.pages import _ImportSpatialDialog

        if self._unsaved_edits():
            return
        dialog = _ImportSpatialDialog(self._recording, self._experiment, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            added, flattened, refused = dialog.do_import()
        except (FileNotFoundError, ValueError) as exc:
            QMessageBox.critical(self, "Import scene geometry", str(exc))
            return
        self._reload()
        QMessageBox.information(
            self,
            "Import regions and points",
            _arrival_report(added, flattened, refused),
        )

    def _export_spatial(self) -> None:
        from detail.storage import spatial as spatial_store

        stem = self._recording.id
        if self._experiment is not None:
            # `Experiment` has no `.id` — its folder name is the filesystem-safe
            # identifier, the role `recording.id` plays on the other side of the
            # underscore.
            stem = f"{self._experiment.dir.name}_{self._recording.id}"
        default = str(spatial_store.spatial_exports_root() / f"{stem}_spatial.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export scene geometry", default, "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        run_busy(
            self,
            lambda: self._recording.export_spatial_csv(path),
            lambda _r: QMessageBox.information(
                self, "Export scene geometry", f"Wrote {path}"
            ),
            label="Exporting…",
        )

    def _unsaved_edits(self) -> bool:
        """Refuse an action that reads or rewrites the recording while edits are pending.

        Copy, import and export all go through the *stored* setup, so running one over
        unsaved edits would copy or write something other than what is on screen — and the
        import would then be overwritten by this dialog's own Save all. Save or cancel
        first; the dialog says which.
        """
        pending = [
            k.id for k in SPATIAL_KINDS if self._signature(k.id) != self._loaded[k.id]
        ]
        if not pending:
            return False
        QMessageBox.information(
            self,
            self.windowTitle(),
            "Save or cancel your changes first — these actions read and write the "
            "recording's stored setup, not what is on screen.",
        )
        return True

    def _reload(self) -> None:
        """Re-read after something outside this dialog changed the recording."""
        self._regions = self._load_regions()
        self._loaded = {k.id: self._signature(k.id) for k in SPATIAL_KINDS}
        self._selected = None
        self._rebuild_list()
        self._sync()
        self._sync_actions()

    # -- the definition list ------------------------------------------------- #
    def _current(self) -> dict | None:
        return self._regions[self._selected] if self._selected is not None else None

    def _kind_of(self, reg: dict) -> SpatialKind:
        return SPATIAL_KINDS_BY_ID[reg["kind"]]

    def _color(self, reg: dict, index_within_kind: int) -> QColor:
        return region_color(self._kind_of(reg), index_within_kind)

    def _region_color(self, i: int) -> QColor:
        reg = self._regions[i]
        seen = [j for j, r in enumerate(self._regions) if r["kind"] == reg["kind"]]
        return self._color(reg, seen.index(i))

    def _swatch(self, color: QColor) -> QIcon:
        pixmap = QPixmap(12, 12)
        pixmap.fill(color)
        return QIcon(pixmap)

    def _rebuild_list(self) -> None:
        """One row per definition, grouped under a heading per kind.

        A row's identity is its index in `_regions`, carried in the item's data — never
        the row's position in the view, which the headings already offset and which any
        future grouping change would move again.
        """
        self._loading = True
        self._list.clear()
        for kind in SPATIAL_KINDS:
            members = [i for i, r in enumerate(self._regions) if r["kind"] == kind.id]
            header = QListWidgetItem(f"{kind.label} ({len(members)})")
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setData(Qt.ItemDataRole.UserRole, None)
            self._list.addItem(header)
            for i in members:
                reg = self._regions[i]
                text = reg["name"]
                if reg["locked"]:
                    text = f"{text}  (locked)"
                item = QListWidgetItem(text)
                item.setIcon(self._swatch(self._region_color(i)))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if reg["visible"] else Qt.CheckState.Unchecked
                )
                item.setData(Qt.ItemDataRole.UserRole, i)
                if reg["locked"]:
                    item.setToolTip(reg["locked"])
                self._list.addItem(item)
        self._loading = False

    def _region_at(self, row: int) -> int | None:
        item = self._list.item(row)
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _on_select(self, row: int) -> None:
        if self._loading:
            return
        self._capture_current()
        self._selected = self._region_at(row) if row >= 0 else None
        reg = self._current()
        self._name.setText(reg["name"] if reg else "")
        self._sync()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._loading:
            return
        i = item.data(Qt.ItemDataRole.UserRole)
        if i is None or not (0 <= i < len(self._regions)):
            return
        self._regions[i]["visible"] = item.checkState() == Qt.CheckState.Checked
        self._sync()

    def _on_rename(self) -> None:
        reg = self._current()
        if reg is None or reg["locked"]:
            return
        name = self._name.text().strip()
        if name and name != reg["name"]:
            reg["name"] = name
            self._rebuild_list()
            self._select_region(self._selected)
            self._edited()

    def _select_region(self, index: int | None) -> None:
        """Select ``index`` without the list's own handler, and show its name.

        The handler is suppressed, so the name is set here: a definition added with a **+**
        button was selected with the Name field still blank (or still showing the one
        selected before), although it already had a name.
        """
        if index is None:
            return
        for row in range(self._list.count()):
            if self._region_at(row) == index:
                self._loading = True
                self._list.setCurrentRow(row)
                self._loading = False
                self._selected = index
                self._name.setText(self._regions[index]["name"])
                return

    def _add_region(self, kind_id: str) -> None:
        self._capture_current()
        kind = SPATIAL_KINDS_BY_ID[kind_id]
        if kind.singleton and any(r["kind"] == kind_id for r in self._regions):
            QMessageBox.information(
                self,
                self.windowTitle(),
                f"This recording already has a {kind.noun}; there can only be one.",
            )
            return
        existing = {r["name"] for r in self._regions if r["kind"] == kind_id}
        n = len(existing) + 1
        while f"{kind.default_base} {n}" in existing:
            n += 1
        name = kind.label if kind.singleton else f"{kind.default_base} {n}"
        # Its first keyframe is the frame on screen, for every kind that moves: starting a
        # field of view at 0 while showing frame 500 left the canvas on an "interpolated"
        # frame that took no clicks (#35 finding 11). Save drops a keyframe left empty.
        new = {
            "kind": kind_id,
            "name": name,
            "keyframes": {self._edit_frame(kind): []},
            "active": None,
            "visible": True,
            "locked": None,
        }
        if kind_id == "calibration":
            new["distance_mm"] = 0.0
        self._regions.append(new)
        self._rebuild_list()
        self._select_region(len(self._regions) - 1)
        self._sync()

    def _delete_region(self) -> None:
        reg = self._current()
        if reg is None:
            return
        if reg["locked"]:
            QMessageBox.warning(self, self.windowTitle(), reg["locked"])
            return
        del self._regions[self._selected]
        self._selected = None
        self._rebuild_list()
        self._sync()

    # -- frame ↔ (part, local) + keyframe editing ---------------------------- #
    def _locate(self, frame: int) -> tuple[int, int]:
        for part, count in enumerate(self._part_counts):
            if frame < count:
                return part, frame
            frame -= count
        return max(len(self._part_counts) - 1, 0), max(self._part_counts[-1] - 1, 0)

    def _frame_image(self, frame: int):
        part, local = self._locate(frame)
        return frame_source.decode(lambda: self._recording.get_frame(part, local))

    def _on_zoom_changed(self, zoom: float) -> None:
        """Keep the "100%" readout and the in/out buttons in step with the canvas — the one
        place either is computed, so a canvas change can't drift from what the buttons say.
        """
        self._zoom_label.setText(f"{round(zoom * 100)}%")
        set_available(
            self._zoom_in_btn, self._canvas.can_zoom_in(), "Maximum zoom reached"
        )
        set_available(
            self._zoom_out_btn, self._canvas.can_zoom_out(), "Already fit to frame"
        )

    def _on_canvas_edited(self) -> None:
        """A drag or a click changed the shape — write it back and refresh Save.

        Without this the dialog only learned of an edit when something else made it
        re-read the canvas, so Save stayed greyed out after a change and lit up when you
        scrubbed — which reads as the frame having been the edit.
        """
        self._capture_current()
        self._edited()

    def _edit_frame(self, kind: SpatialKind) -> int:
        """The keyframe an edit of a ``kind`` definition lands on.

        The frame on screen, except for a kind that does not move (the calibration): that
        has one entry, at 0, whatever frame is showing. Otherwise a calibration added on
        frame 500 got its entry at 0, so the canvas on 500 showed an "interpolated" frame
        that took no clicks (#35 finding 11).
        """
        return self._frame if kind.keyframed else 0

    def _capture_current(self) -> None:
        """Write the canvas back into the selected definition. A helper, not an edit
        handler: every caller ends in :meth:`_edited` or :meth:`_sync`."""
        reg = self._current()
        if reg is None or reg["locked"]:
            return
        frame = self._edit_frame(self._kind_of(reg))
        if frame in reg["keyframes"]:
            reg["keyframes"][frame] = self._canvas.normalized_polygon()

    def _ghost_for(
        self, reg: dict, frame: int, *, respect_active: bool = False
    ) -> list[tuple[float, float]]:
        """Where ``reg`` sits at ``frame``, or nothing.

        ``respect_active`` is on for the *context* shapes and off for the definition being
        edited — the one you are *editing* has to stay visible wherever you scrub, or its
        own active window would hide it from you the moment you set one. See
        :func:`region_ghost` for the module-level form this delegates to.
        """
        return region_ghost(reg, frame, respect_active=respect_active)

    def _context_shapes(self) -> list[tuple]:
        shapes = []
        for i, reg in enumerate(self._regions):
            if i == self._selected or not reg["visible"]:
                continue
            points = self._ghost_for(reg, self._frame, respect_active=True)
            if points:
                is_point = self._kind_of(reg).max_points == 1
                shapes.append((points, self._region_color(i), is_point, True))
        return shapes

    def _sync(self) -> None:
        # Colour means "this is the one you are editing". With something selected the rest
        # go grey; with nothing selected every definition keeps its colour, which is the
        # read-only overview this dialog absorbed.
        self._canvas.set_dim_context(self._selected is not None)
        self._canvas.set_context_shapes(self._context_shapes())
        if hasattr(self, "_save"):  # not yet built during __init__
            self._sync_dirty()
        reg = self._current()
        editable_kind = self._kind_of(reg) if reg is not None else None
        is_calibration = reg is not None and reg["kind"] == "calibration"
        self._distance_label.setVisible(is_calibration)
        self._distance_holder.setVisible(is_calibration)
        if is_calibration:
            self._distance.blockSignals(True)
            self._distance.setValue(float(reg.get("distance_mm") or 0.0))
            self._unit.blockSignals(True)
            self._unit.setCurrentText("mm")
            self._unit.blockSignals(False)
            self._distance.blockSignals(False)
        self._refresh_active()
        if reg is None:
            self._canvas.set_polygon([], editable=False)
            self._canvas.set_max_points(None)
            self._name.setText("")  # not the name of one just deleted
            self._name.setEnabled(False)
            set_available(self._delete, False, "Select a definition to delete it")
            for button in (self._set_kf, self._del_kf, self._undo, self._clear):
                set_available(button, False, "Select a definition to edit it")
            self._kf_label.setText(
                "Nothing selected — every definition is shown. Pick one to edit it, or "
                "start a new one with a + button."
            )
            return
        locked = bool(reg["locked"])
        self._canvas.set_edit_color(self._region_color(self._selected))
        self._name.setEnabled(not locked)
        set_available(self._delete, not locked, reg["locked"] or "")
        self._canvas.set_max_points(editable_kind.max_points)
        edit_frame = self._edit_frame(editable_kind)
        is_keyframe = edit_frame in reg["keyframes"]
        if is_keyframe and not locked:
            self._canvas.set_polygon(reg["keyframes"][edit_frame], editable=True)
        else:
            self._canvas.set_polygon(
                self._ghost_for(reg, self._frame), editable=False, prominent=locked
            )
        # Points can be added or removed while at most one keyframe holds any. An empty
        # keyframe holds no shape to keep in step with (Save drops it), so it does not
        # count: a frame-0 entry nobody drew on used to lock a shape started on frame 500.
        drawn = sum(1 for v in reg["keyframes"].values() if v)
        self._canvas.set_allow_topology(drawn <= 1)
        fixed = not editable_kind.keyframed
        still = f"a {editable_kind.noun} is the same on every frame"
        for button, on, why in (
            (
                self._set_kf,
                not is_keyframe and not fixed,
                still if fixed else "this frame is already a keyframe",
            ),
            (
                self._del_kf,
                is_keyframe and len(reg["keyframes"]) > 1 and not fixed,
                still if fixed else _no_delete_reason(is_keyframe),
            ),
            (self._undo, is_keyframe, _INTERPOLATED_FRAME),
            (self._clear, is_keyframe, _INTERPOLATED_FRAME),
        ):
            set_available(button, on and not locked, reg["locked"] or why)
        if locked:
            self._kf_label.setText(reg["locked"])
            return
        if fixed:
            self._kf_label.setText(
                f"The {editable_kind.noun} is the same on every frame — draw it on any."
            )
            return
        frames = ", ".join(str(f) for f in sorted(reg["keyframes"]))
        state = "keyframe (editable)" if is_keyframe else "interpolated"
        self._kf_label.setText(f"Keyframes: {frames}   ·   this frame: {state}")

    def _on_scrub(self, value: int) -> None:
        self._capture_current()
        self._frame = value
        self._sync()

    def _set_keyframe(self) -> None:
        reg = self._current()
        if reg is None or reg["locked"] or not self._kind_of(reg).keyframed:
            return
        reg["keyframes"][self._frame] = self._canvas.normalized_polygon()
        self._sync()

    def _delete_keyframe(self) -> None:
        reg = self._current()
        if reg is None or not self._kind_of(reg).keyframed:
            return
        if not reg["locked"] and self._frame in reg["keyframes"]:
            if len(reg["keyframes"]) > 1:
                del reg["keyframes"][self._frame]
                self._sync()

    def _on_undo(self) -> None:
        self._canvas.undo()
        self._capture_current()

    def _on_clear(self) -> None:
        self._canvas.clear()
        self._capture_current()

    def _on_distance_changed(self) -> None:
        reg = self._current()
        if reg is None or reg["kind"] != "calibration":
            return
        value = self._distance.value()
        reg["distance_mm"] = value * (10.0 if self._unit.currentText() == "cm" else 1.0)
        self._edited()

    # -- active window (ROIs/POIs only) -------------------------------------- #
    def _build_active_controls(self, layout) -> None:
        """Lifted from the single-definition editor, which had it and was unreachable.

        `RoiDialog`/`PointDialog` implemented this and had no call site in the app — the
        manager became the only way in and it could not express a window, so it deleted
        the ones the API and CLI can still set. Editing it here also closes that
        surface-parity gap.
        """
        self._whole_video = QCheckBox("Active for whole video")
        self._whole_video.toggled.connect(self._on_whole_video_toggled)
        self._set_start = QPushButton("Set active start (this frame)")
        self._set_end = QPushButton("Set active end (this frame)")
        self._set_start.clicked.connect(lambda: self._set_active_bound(start=True))
        self._set_end.clicked.connect(lambda: self._set_active_bound(start=False))
        self._active_label = QLabel()
        row = QHBoxLayout()
        row.addWidget(self._whole_video)
        row.addWidget(self._set_start)
        row.addWidget(self._set_end)
        row.addStretch()
        self._active_caption = QLabel("Active window (analysis only):")
        layout.addWidget(self._active_caption)
        self._active_row = QWidget()
        self._active_row.setLayout(row)
        layout.addWidget(self._active_row)
        layout.addWidget(self._active_label)

    def _on_whole_video_toggled(self, checked: bool) -> None:
        reg = self._current()
        if reg is None or not self._kind_of(reg).has_active:
            return
        if checked:
            reg["active"] = None
        elif reg["active"] is None:
            reg["active"] = (0, max(self._total - 1, 0))
        self._refresh_active()
        self._edited()

    def _set_active_bound(self, *, start: bool) -> None:
        reg = self._current()
        if reg is None:
            return
        lo, hi = reg["active"] if reg["active"] else (0, max(self._total - 1, 0))
        if start:
            lo = self._frame
        else:
            hi = self._frame
        if (
            lo > hi
        ):  # a backwards pick collapses to a single frame rather than inverting
            lo = hi = self._frame
        reg["active"] = (lo, hi)
        self._refresh_active()
        self._edited()

    def _refresh_active(self) -> None:
        reg = self._current()
        shows = reg is not None and self._kind_of(reg).has_active
        for widget in (self._active_caption, self._active_row, self._active_label):
            widget.setVisible(shows)
        if not shows:
            return
        whole = reg["active"] is None
        self._whole_video.blockSignals(True)
        self._whole_video.setChecked(whole)
        self._whole_video.blockSignals(False)
        locked = bool(reg["locked"])
        self._whole_video.setEnabled(not locked)
        for button in (self._set_start, self._set_end):
            set_available(button, not whole and not locked, "")
        self._active_label.setText(
            "Active: whole video"
            if whole
            else f"Active: frames {reg['active'][0]}–{reg['active'][1]}"
        )


def _load_geometry(
    recording, canvas: ArenaCanvas, on_done: Callable[[bool], None] | None = None
) -> None:
    """Decode ``recording``'s first frame into ``canvas`` off the UI thread and draw
    every currently defined shape over it — or say why there is no frame to show.

    Shared by :class:`GeometryPreview` and :class:`GeometryPreviewDialog`, which differ
    only in how big the canvas is: without this they would each have to reason about "no
    video part" vs "video excluded from this export" vs "frame won't decode" a second
    way.

    ``on_done(available)`` — when given — reports, once, whether a real picture ended up
    on the canvas: :class:`GeometryPreview`'s "(click to enlarge)" hint is a claim about
    what a click does, and a click does nothing while the canvas is showing a placeholder
    rather than a frame.
    """

    def _report(available: bool) -> None:
        if on_done is not None:
            on_done(available)

    if not recording.video_parts:
        canvas.set_unavailable("No video imported")
        _report(False)
        return
    if not recording.video_available:
        canvas.set_unavailable(frame_source.unavailable_text(frame_source.MISSING))
        _report(False)
        return

    def _on_frame(payload) -> None:
        (image, reason), shapes = payload
        if reason is not None:
            canvas.set_unavailable(frame_source.unavailable_text(reason))
            _report(False)
            return
        canvas.set_image(image)
        canvas.set_context_shapes(shapes)
        _report(True)

    # The shapes are read in the same background call as the frame: every region is its
    # own recording read, and none of them belongs on the UI thread.
    run_busy(
        canvas,
        lambda: (
            frame_source.decode(lambda: recording.get_frame(0, 0)),
            geometry_snapshot(recording, 0),
        ),
        _on_frame,
        label="Loading preview…",
        supersedes="geometry",  # a refresh redraws the same canvas
    )


class GeometryPreview(QWidget):
    """A read-only snapshot of a recording's scene geometry, on its first frame.

    What :class:`SceneGeometryDialog` shows at rest — nothing selected, every definition
    drawn in its own colour — without the scrubber, the edit controls, or any way to
    change anything: a user can tell at a glance whether the setup needs fixing before
    opening the editor. Draws through the same primitives the editor and the recording
    page's swatches use (:func:`load_scene_regions`, :func:`region_ghost`,
    :func:`region_color`) rather than a second way of interpreting a definition.

    Not interactive — the canvas is transparent to mouse events, and a click anywhere on
    this widget opens :class:`GeometryPreviewDialog` instead, the same picture at full
    resolution, still read-only.
    """

    def __init__(self, recording, parent=None):
        super().__init__(parent)
        self._recording = recording
        self._openable = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = ArenaCanvas(_BLANK_FRAME)
        self._canvas.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._canvas.set_polygon([], editable=False)
        self._canvas.set_dim_context(False)
        layout.addWidget(self._canvas)
        self._hint = QLabel("(click to enlarge)")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_label_color(self._hint, MUTED)
        self._hint.setVisible(False)
        layout.addWidget(self._hint)
        self.refresh()

    def refresh(self) -> None:
        """Re-decode frame 0 and re-draw every currently defined shape."""
        self._set_openable(False)
        _load_geometry(self._recording, self._canvas, on_done=self._set_openable)

    def _set_openable(self, openable: bool) -> None:
        # The hint and the pointing hand are claims about what a click does, so they and
        # the click itself read one flag, set only once a real frame has landed — not
        # while it is still decoding, and never over a "No video imported" /
        # "could not decode frame" placeholder.
        self._openable = openable
        self._hint.setVisible(openable)
        if openable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._openable:
            GeometryPreviewDialog(self._recording, parent=self).exec()
            return
        super().mousePressEvent(event)


class GeometryPreviewDialog(Dialog):
    """The same read-only snapshot as :class:`GeometryPreview`, at full size.

    Opened by clicking the recording page's thumbnail — a closer look at a picture
    already on screen, not a second way to edit it. No scrubber, no definition list,
    just Close.
    """

    def __init__(self, recording, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Geometry preview — {recording.name}")
        self.resize(1040, 740)  # fits 1440x810 (gui_design.md §3.1)
        self._canvas = ArenaCanvas(_BLANK_FRAME)
        self._canvas.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._canvas.set_polygon([], editable=False)
        self._canvas.set_dim_context(False)
        self.body().addWidget(self._canvas, 1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        self.set_buttons(box)
        _load_geometry(recording, self._canvas)
