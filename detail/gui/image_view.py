"""The one pane that shows a picture — scaled to fit, and optionally zoomable.

A **leaf**: it imports nothing from ``detail.gui`` except ``scroll``'s wheel-ownership property
and ``sizing``'s way of telling a window its picture grew, which are leaves themselves. That is the point. This widget began as ``behavior_query._VideoPane``,
inside a 5000-line module that imports the module registry — so ``tools/quick_segment.py`` and
``modules/analysis_query_eval.py`` reached it by importing *from* ``behavior_query``. The same
shape as ``sizing.finish_dialog`` (8 of 44 dialogs) and ``sizing.mono_font`` (five copies): a
shared thing living where the things that need it cannot reach.

**Zoom is required, not defaulted.** ``FrameScrubber.playable`` is the precedent, and its
docstring says defaulting it False *"is what left nine surfaces with no player for months"*.
Whether a surface can be zoomed is a fact about that surface — a frame someone is judging, yes;
a thumbnail whose click means "play me", no — so every embedder states it.

**The gestures name a device that exists.** Wheel to zoom, **left**-drag to pan, a visible
**Fit** button to get back. Not middle-drag and not Ctrl-drag: ``tests/gui/test_scatter_navigation``
records the 3-D scatter shipping pan bound to exactly those, which a MacBook trackpad cannot
produce (macOS turns Ctrl+left into a *right* button before Qt sees it, Apple QA1362), so the
capability was documented, implemented and unreachable. Fit is a button rather than only a
gesture for the same test's other half: a view that can be moved out of frame needs a way back.

**Double-click is deliberately not bound.** ``clicked`` drives play/pause on three surfaces, and
Qt delivers press and release *before* a double click — so "double-click to fit" would toggle
playback on the way past. The Fit button is the fit.

**A montage zooms its tiles together** (:meth:`ImageView.set_tiles`). Zooming a picture composed
of one moment seen several ways magnifies one tile and pushes the ones it is compared with off
screen; with tiles set, the same gestures zoom and pan the same region of every tile at once.
Their names are painted by the view, after scaling, at the screen's own text size: a name drawn
into the picture shrinks with it, and a two-by-two montage of a square video fitted to a dialog
put them at 7px.
"""

from collections.abc import Callable

from PyQt6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPixmap
from PyQt6.QtWidgets import QLabel, QPushButton, QSizePolicy, QWidget

from detail.gui.scroll import OWNS_WHEEL
from detail.gui.sizing import announce_natural_size

#: How far one wheel notch moves the zoom. 1.25 is a quarter per notch — enough to feel
#: responsive on a mouse, small enough that a trackpad's finer deltas do not leap.
_ZOOM_STEP = 1.25

#: Fit is 1.0. Below it the picture would be smaller than the pane it already fits, which is
#: never what "zoom out" means here; above it, past ~16x a 1080p frame is a few pixels wide.
_MIN_ZOOM, _MAX_ZOOM = 1.0, 16.0

#: A press that moves less than this is a click, not a drag. Without it, the hand tremor in an
#: ordinary click cancels the play/pause every embedder wires to ``clicked``.
_DRAG_SLOP_PX = 4


class ImageView(QLabel):
    """A picture scaled to the pane, keeping aspect ratio, rescaled as the pane resizes.

    ``clicked`` fires on a press-release that did not move — a player wires it to play/pause.
    ``clicked_at`` fires beside it with *where* on the picture, as ``(u, v)`` in ``[0, 1]`` of
    the frame's own width and height (zoom and pan accounted for), when the click landed on the
    picture rather than the margin around it — a labeling surface places a point there (§26).

    ``zoomable`` has **no default**: see the module docstring. When set, the wheel zooms about
    the cursor, left-drag pans, and :meth:`fit` (and whatever button the host wires to it)
    returns to 1.0.

    A host that lets the user **drag something drawn on the picture** (the labeling screen's
    placed keypoints, §26) sets :attr:`hit_test`, a callable taking the widget position of a
    left press and answering whether it landed on a grabbable thing. When it does, the press
    starts a drag of that thing instead of a pan or a click: ``mark_drag_started``,
    ``mark_dragged`` and ``mark_drag_finished`` fire with ``(u, v)`` picture coordinates
    (clamped to the picture while dragging), and ``clicked`` / ``clicked_at`` stay silent for
    that gesture. ``None``, the default, is a view with nothing to grab, which is every other
    host.
    """

    clicked = pyqtSignal()
    clicked_at = pyqtSignal(float, float)
    mark_drag_started = pyqtSignal(float, float)
    mark_dragged = pyqtSignal(float, float)
    mark_drag_finished = pyqtSignal(float, float)

    def __init__(self, *, zoomable: bool, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(120, 90)
        # **Ignored**, not Expanding. A `QLabel` takes its size hint from its pixmap, and this
        # one's pixmap is a *product* of the size it was given — so the hint fed straight back
        # into the layout that produced it: the first frame took the page's hint from 130px to
        # 536px, the pane got more room, the next frame was scaled larger, and the picture
        # visibly grew for a second or two after opening a clip. `Ignored` means "the sizeHint
        # is ignored; the widget gets as much space as possible", which is exactly what a video
        # pane wants — fill what you are given and never push back. The explicit minimum above
        # still holds, because an explicit minimum is not a hint. Zooming must never reach the
        # size hint for the same reason: it changes what is *drawn*, never what is asked for.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._pixmap: QPixmap | None = None
        #: A host's own statement of the picture's natural size — see :meth:`natural_size`.
        self._natural: QSize | None = None
        self._zoomable = zoomable
        self._zoom = 1.0
        self._offset = QPoint(0, 0)  # pan, in scaled-pixmap pixels
        #: The rects that zoom together (:meth:`set_tiles`), and their shared pan: the centre
        #: of the region each shows, in that tile's own ``[0, 1]``.
        self._tiles: list[QRect] | None = None
        self._tile_names: list[str] | None = None
        self._centre = QPointF(0.5, 0.5)
        self._press: QPoint | None = None
        self._dragged = False
        self.hit_test: Callable[[QPoint], bool] | None = None
        self._mark_drag = False
        self._rest_cursor: Qt.CursorShape | None = None
        self._fit_btn: QPushButton | None = None
        if zoomable:
            # `PageScroll._page_takes_wheel` walks up from the event's widget looking for this,
            # so without it the page scrolls instead of the picture zooming. Alt is the page's
            # documented escape hatch and stays working either way.
            self.setProperty(OWNS_WHEEL, True)
            # The view owns its own way back, rather than each host wiring one. Four hosts today
            # and every future one gets it for free — and a host cannot ship a zoomable view
            # with no way to recover it, which is the `test_scatter_navigation` lesson: the 3-D
            # scatter could be turned off screen and never brought back.
            self._fit_btn = QPushButton("Fit", self)
            self._fit_btn.setToolTip("Back to the whole picture")
            self._fit_btn.clicked.connect(self.fit)
            self._fit_btn.hide()  # nothing to get back from until something is zoomed

    # -- what the host drives ------------------------------------------------- #
    def set_rest_cursor(self, shape: Qt.CursorShape | None) -> None:
        """The cursor over the picture when nothing is being dragged — a pointing hand where a
        host has wired ``clicked`` to open something. ``None`` is the default arrow. Kept
        rather than set once, because a release puts the cursor back after a drag."""
        self._rest_cursor = shape
        self._restore_cursor()

    def rest_cursor(self) -> Qt.CursorShape | None:
        return self._rest_cursor

    def _restore_cursor(self) -> None:
        if self._rest_cursor is None:
            self.unsetCursor()
        else:
            self.setCursor(self._rest_cursor)

    def set_frame(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._rescale()
        self._announce()

    # -- the size the picture would like ------------------------------------- #
    def natural_size(self) -> QSize | None:
        """The size this picture is meant to be seen at — what a pop-up opens large enough to
        show, up to its cap (`sizing.fit_to_content`). ``None`` before there is a picture.

        The frame's own pixels by default. **Never** the size hint: a hint taken from the
        pixmap feeds back into the layout that scaled it (see ``__init__``), and a window grown
        to it would grow again with every frame. The frame's own size is not a product of this
        pane's, so a window can be sized to it once.

        A host whose picture **is** a product of the pane — a montage composed at whatever tile
        size fits — says what it would be at its own resolution, with :meth:`set_natural_size`.
        """
        if self._natural is not None:
            return QSize(self._natural)
        return self.pixmap_size()

    def set_natural_size(self, size: QSize | None) -> None:
        """Declare the picture's natural size, for a picture drawn to fit the pane."""
        self._natural = None if size is None else QSize(size)
        self._announce()

    def _announce(self) -> None:
        announce_natural_size(self)

    def clear_frame(self, text: str) -> None:
        self.setText(text)

    # `QLabel.setText` / `clear` blank the label but not the frame this view holds, so the next
    # resize or wheel notch repainted the old frame over the message — a pane saying "video not
    # on this machine" while showing the last frame it had. Overridden rather than documented:
    # every host that swapped a `QLabel` for this view already calls them.
    def setText(self, text: str) -> None:  # noqa: N802 — Qt's name
        self._pixmap = None
        super().setText(text)

    def clear(self) -> None:
        self._pixmap = None
        super().clear()

    @property
    def zoomable(self) -> bool:
        return self._zoomable

    @property
    def zoom(self) -> float:
        return self._zoom

    def pixmap_size(self) -> QSize | None:
        """The last frame's own size, before any scaling — ``None`` before one arrives.

        A host deriving the picture's on-screen height from this pane's *width* (the safe
        shape documented in the module docstring — width to height is not a feedback loop,
        this widget's own hint to its own size is) needs the frame's original aspect ratio,
        never anything already scaled to this label's size.
        """
        return None if self._pixmap is None else self._pixmap.size()

    def fit(self) -> None:
        """Back to the whole picture, centred — the way out of a view panned off screen."""
        self._set_zoom(1.0, about=None)

    def set_tiles(
        self, tiles: list[QRect] | None, names: list[str] | None = None
    ) -> None:
        """Zoom ``tiles`` together instead of the whole picture (see the module docstring),
        each named in its top-left corner by ``names``.

        Each rect, in the picture's own pixels, shows the same region of itself at the one zoom
        and pan; the picture outside them is never zoomed. ``None`` is the whole picture again.
        A host sets them with every frame, so an unchanged set keeps the zoom it had. With
        tiles set, :meth:`image_position` and :meth:`widget_position` speak of the composed
        picture as fitted, not of what a tile shows.
        """
        tiles = [QRect(t) for t in tiles] if tiles else None
        names = list(names) if tiles and names else None
        if tiles != self._tiles or names != self._tile_names:
            self._tiles, self._tile_names = tiles, names
            self._rescale()

    def tile_names(self) -> list[str] | None:
        return None if self._tile_names is None else list(self._tile_names)

    # -- gestures -------------------------------------------------------------- #
    def wheelEvent(self, event) -> None:
        if not self._zoomable or event.modifiers() & Qt.KeyboardModifier.AltModifier:
            event.ignore()  # Alt is the page's escape hatch; let it through
            return
        notches = event.angleDelta().y() / 120.0 or event.pixelDelta().y() / 50.0
        if not notches:
            event.ignore()
            return
        self._set_zoom(self._zoom * (_ZOOM_STEP**notches), about=event.position())
        event.accept()

    def mousePressEvent(self, event) -> None:
        self._press = event.pos()
        self._dragged = False
        self._mark_drag = False
        if (
            self.hit_test is not None
            and event.button() == Qt.MouseButton.LeftButton
            and self.hit_test(event.pos())
        ):
            where = self.image_position(event.position(), clamp=True)
            if where is not None:
                self._mark_drag = True
                self.mark_drag_started.emit(*where)
                return
        if self._zoomable and self._zoom > 1.0:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._mark_drag:
            where = self.image_position(event.position(), clamp=True)
            if where is not None:
                self.mark_dragged.emit(*where)
            return
        if self._press is not None and self._zoomable and self._zoom > 1.0:
            delta = event.pos() - self._press
            if self._dragged or delta.manhattanLength() > _DRAG_SLOP_PX:
                self._dragged = True
                self._press = event.pos()
                if self._tiles is None:
                    self._offset += delta
                else:
                    self._pan_tiles(delta)
                self._rescale()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._restore_cursor()
        if self._mark_drag:
            # The grabbed thing lands where the button came up — never also a click, which
            # would place the *next* point on top of the one just moved.
            self._mark_drag = False
            self._press = None
            where = self.image_position(event.position(), clamp=True)
            if where is not None:
                self.mark_drag_finished.emit(*where)
            return
        # A drag is not a click. Without the slop test above, the tremor in an ordinary press
        # would swallow the play/pause that three surfaces wire to `clicked`.
        if not self._dragged:
            self.clicked.emit()
            where = self.image_position(event.position())
            if where is not None:
                self.clicked_at.emit(*where)
        self._press = None
        super().mouseReleaseEvent(event)

    def _picture_geometry(self) -> tuple[QSize, int, int, int, int] | None:
        """``(scaled size, left crop, top crop, x0, y0)`` of the picture as drawn: the fit,
        grown by the zoom, cropped to the pane around the pan, centred by the label."""
        if self._pixmap is None or self._pixmap.isNull():
            return None
        mode = Qt.AspectRatioMode.KeepAspectRatio
        fitted = self._pixmap.size().scaled(self.size(), mode)
        if (
            self._zoom == 1.0 or self._tiles is not None
        ):  # tiles zoom inside a fitted picture
            scaled, left, top = fitted, 0, 0
        else:
            scaled = self._pixmap.size().scaled(fitted * self._zoom, mode)
            left = max(0, (scaled.width() - self.width()) // 2 - self._offset.x())
            top = max(0, (scaled.height() - self.height()) // 2 - self._offset.y())
        shown_w = min(self.width(), scaled.width())
        shown_h = min(self.height(), scaled.height())
        x0 = (self.width() - shown_w) // 2
        y0 = (self.height() - shown_h) // 2
        return scaled, left, top, x0, y0

    def image_position(self, pos, *, clamp: bool = False) -> tuple[float, float] | None:
        """``(u, v)`` in ``[0, 1]`` of the frame under widget point ``pos``; None off the
        picture, or the nearest point on it when ``clamp`` (a drag that leaves the picture
        keeps its grip).

        The inverse of :meth:`_rescale`: the picture is fitted to the pane, grown by the zoom,
        cropped to the pane around the pan, and centred by the label — undone in that order.
        """
        geometry = self._picture_geometry()
        if geometry is None:
            return None
        scaled, left, top, x0, y0 = geometry
        u = (pos.x() - x0 + left) / max(1, scaled.width())
        v = (pos.y() - y0 + top) / max(1, scaled.height())
        if clamp:
            return float(min(max(u, 0.0), 1.0)), float(min(max(v, 0.0), 1.0))
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return None
        return float(u), float(v)

    def widget_position(self, u: float, v: float) -> QPointF | None:
        """Where picture point ``(u, v)`` is drawn, in widget pixels; None before a frame."""
        geometry = self._picture_geometry()
        if geometry is None:
            return None
        scaled, left, top, x0, y0 = geometry
        return QPointF(u * scaled.width() - left + x0, v * scaled.height() - top + y0)

    # -- drawing ---------------------------------------------------------------- #
    def _set_zoom(self, zoom: float, *, about) -> None:
        zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, zoom))
        if zoom == self._zoom:
            return
        if self._tiles is not None:
            self._zoom_tiles(zoom, about)
        elif zoom == _MIN_ZOOM:
            self._offset = QPoint(0, 0)  # fitting re-centres; a stale pan would stick
        elif about is not None:
            # Keep the point under the cursor under the cursor: the offset scales with the
            # picture, so the pixel the user is pointing at does not slide away as it grows.
            centre = QPoint(self.width() // 2, self.height() // 2)
            cursor = QPoint(int(about.x()), int(about.y())) - centre
            ratio = zoom / self._zoom
            self._offset = (self._offset + cursor) * ratio - cursor
        self._zoom = zoom
        if self._fit_btn is not None:
            # Shown only when there is something to undo — an always-visible Fit over a picture
            # that is already fitted is a control that does nothing.
            self._fit_btn.setVisible(zoom > _MIN_ZOOM)
            self._place_fit_button()
            self._fit_btn.raise_()
        self._rescale()

    def _zoom_tiles(self, zoom: float, about) -> None:
        """Keep what is under the cursor under it, in whichever tile the cursor is over — and
        so the same point of every tile, since they zoom as one."""
        if zoom == _MIN_ZOOM:
            self._centre = QPointF(0.5, 0.5)
            return
        under = None if about is None else self._tile_position(about)
        if under is not None:
            u, v = under
            seen_x = self._centre.x() + (u - 0.5) / self._zoom
            seen_y = self._centre.y() + (v - 0.5) / self._zoom
            self._centre = QPointF(seen_x - (u - 0.5) / zoom, seen_y - (v - 0.5) / zoom)
        self._centre = self._clamped_centre(zoom)

    def _tile_position(self, pos) -> tuple[float, float] | None:
        """Where ``pos`` falls in the tile under it, in that tile's ``[0, 1]`` as drawn."""
        where = self.image_position(pos)
        if where is None:
            return None
        x, y = where[0] * self._pixmap.width(), where[1] * self._pixmap.height()
        for tile in self._tiles:
            if tile.left() <= x < tile.left() + tile.width() and (
                tile.top() <= y < tile.top() + tile.height()
            ):
                return (x - tile.left()) / tile.width(), (
                    y - tile.top()
                ) / tile.height()
        return None

    def _pan_tiles(self, delta: QPoint) -> None:
        """Move every tile's region by the drag, the way the content under the cursor goes."""
        geometry = self._picture_geometry()
        if geometry is None:
            return
        shown = self._zoom * geometry[0].width() / max(1, self._pixmap.width())
        tile = self._tiles[0]
        self._centre = QPointF(
            self._centre.x() - delta.x() / max(1.0, tile.width() * shown),
            self._centre.y() - delta.y() / max(1.0, tile.height() * shown),
        )
        self._centre = self._clamped_centre(self._zoom)

    def _clamped_centre(self, zoom: float) -> QPointF:
        """The centre held where the region it shows stays inside its tile."""
        half = 0.5 / zoom
        return QPointF(
            min(max(self._centre.x(), half), 1.0 - half),
            min(max(self._centre.y(), half), 1.0 - half),
        )

    def _tiled(self) -> QPixmap:
        """The picture with each tile showing the zoomed, panned region of itself."""
        if self._zoom == 1.0:
            return self._pixmap
        picture = QPixmap(self._pixmap)
        painter = QPainter(picture)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        span = 1.0 / self._zoom
        for tile in self._tiles:
            region = QRectF(
                tile.left() + (self._centre.x() - span / 2) * tile.width(),
                tile.top() + (self._centre.y() - span / 2) * tile.height(),
                tile.width() * span,
                tile.height() * span,
            )
            painter.drawPixmap(QRectF(tile), self._pixmap, region)
        painter.end()
        return picture

    def _named(self, shown: QPixmap) -> QPixmap:
        """``shown`` (the picture as scaled to the pane) with each tile's name on a dark plate
        in its top-left corner, in the view's own font made bold — elided to the tile's width
        rather than run into its neighbour."""
        if not self._tile_names:
            return shown
        scale = shown.width() / max(1, self._pixmap.width())
        font = self.font()
        font.setBold(True)
        metrics = QFontMetrics(font)
        pad = 4
        painter = QPainter(shown)
        painter.setFont(font)
        for tile, name in zip(self._tiles, self._tile_names):
            x, y = round(tile.left() * scale), round(tile.top() * scale)
            room = round(tile.width() * scale) - 2 * pad
            text = metrics.elidedText(name, Qt.TextElideMode.ElideRight, max(room, 0))
            plate = QRect(
                x, y, metrics.horizontalAdvance(text) + 2 * pad, metrics.height() + pad
            )
            painter.fillRect(plate, QColor(20, 20, 20))
            painter.setPen(QColor(245, 245, 245))
            painter.drawText(plate, Qt.AlignmentFlag.AlignCenter, text)
        painter.end()
        return shown

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_fit_button()
        self._rescale()

    def _place_fit_button(self) -> None:
        if self._fit_btn is None:
            return
        hint = self._fit_btn.sizeHint()
        self._fit_btn.setGeometry(
            self.width() - hint.width() - 8, 8, hint.width(), hint.height()
        )

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        if self._tiles is not None:
            shown = self._tiled().scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(self._named(shown))
            return
        fitted = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self._zoom == 1.0:
            self.setPixmap(fitted)
            return
        grown = fitted.size() * self._zoom
        scaled = self._pixmap.scaled(
            grown,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Crop the pane's worth out of the enlarged picture rather than handing the label a
        # pixmap bigger than itself — a QLabel would centre-crop it for us, which throws the pan
        # away. Clamped so a pan can never run past the edge and show empty space.
        window = self.size()
        max_x = max(0, (scaled.width() - window.width()) // 2)
        max_y = max(0, (scaled.height() - window.height()) // 2)
        self._offset = QPoint(
            max(-max_x, min(max_x, self._offset.x())),
            max(-max_y, min(max_y, self._offset.y())),
        )
        left = (scaled.width() - window.width()) // 2 - self._offset.x()
        top = (scaled.height() - window.height()) // 2 - self._offset.y()
        self.setPixmap(
            scaled.copy(
                max(0, left),
                max(0, top),
                min(window.width(), scaled.width()),
                min(window.height(), scaled.height()),
            )
        )


class PictureWithControls(QWidget):
    """Picture directly above its controls, the pair centred in whatever pane it is handed (#396).

    Before this, a video surface gave its picture the layout stretch — the ordinary instinct, the
    same one that puts ``1`` on a table so it fills its panel. For a picture that keeps its aspect
    ratio, a stretch does something different: the *label* still letterboxes the frame inside
    itself, so the label consumes the whole pane and the controls end up pinned to the bottom of
    it, however much of that pane is empty above and below the actual picture. Measured on a real
    pane: 267px of dead space around a 280px-tall picture, and the controls 225px below its own
    bottom edge — nowhere near the thing they operate.

    The fix is not to give the picture a size **hint** that matches its aspect ratio — a `QLabel`
    takes its hint from its pixmap, and this pixmap is itself a product of the size the label was
    given, so a hint derived from it feeds straight back into the layout that produced it (the
    growth loop `ImageView`'s own docstring exists to avoid). Instead this widget's **width**
    (never its own hint) and the last frame's aspect ratio set the picture's height explicitly,
    recomputed on this widget's own resize, and the remaining room is split evenly above and
    below the pair to centre it.

    **No `QLayout`.** A layout stretch recomputed from a constraint that same layout's own child
    just changed is reentrant — the picture's height depends on this widget's height, and
    changing the picture's height mid-resize invalidates the very layout pass computing it, so a
    `QVBoxLayout` version of this settled into a stale, overlapping position on some resizes
    (found empirically: correct once, then wrong the moment the pane changed size again, with no
    exception and no consistent story — `QLayout.activate()` from inside the resize it was
    reacting to did not reliably fix it either). Positioning both children directly in
    ``resizeEvent`` is one pass, computed from this widget's own current size only: nothing here
    depends on anything a previous call changed.
    """

    def __init__(
        self,
        picture: ImageView,
        controls: QWidget,
        *,
        default_aspect: float = 16 / 9,
        top: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.picture = picture
        self.controls = controls
        self._aspect = default_aspect
        #: Pin the pair to the top of the pane instead of centring it — a choice, for a player
        #: that sits *beside* something rather than filling a pane of its own. Browse puts one
        #: next to a list as tall as the workspace, and centred in that the video sat hundreds
        #: of pixels down, below the fold, level with nothing in the list.
        self._top = top
        #: The transport's height at this width as of the last placement — what a
        #: ``LayoutRequest`` is compared against (see :meth:`event`).
        self._controls_need = -1
        picture.setParent(self)
        controls.setParent(self)
        # No `QLayout` manages these two (see the class docstring), so nothing else will show
        # them — a `QLayout` does this implicitly for a widget it owns; a manually parented one
        # stays hidden until told otherwise.
        picture.show()
        controls.show()
        self._reflow()

    def set_aspect(self, size: QSize | None) -> None:
        """Adopt a newly-decoded frame's aspect ratio, if it says anything different.

        A host calls this after handing ``picture`` a fresh frame — ``ImageView.pixmap_size()``
        is the frame's own dimensions, untouched by any scaling, so this never chases anything
        this widget itself already changed.
        """
        if size is None or size.height() <= 0:
            return
        aspect = size.width() / size.height()
        if aspect != self._aspect:
            self._aspect = aspect
            self._reflow()
            # A host sizing itself with `height_for_width` asked before this clip was decoded,
            # so the answer it got was the previous frame's shape. Without this the layout
            # keeps it until something else invalidates the geometry.
            self.updateGeometry()

    def height_for_width(self, width: int) -> int:
        """How tall this block wants to be at ``width``: the picture at its own aspect ratio,
        plus the transport under it.

        The public form of the first two lines of :meth:`_reflow`, for a host that has to
        *reserve* the height rather than fill one it was given. A host that reserves a constant
        instead leaves the difference as a gap under the transport — the whole point of `top`
        being that the spare room ends up in one place, which is only an improvement while
        there is no spare room to speak of. Measured on the audit tab at a 357px pane: a
        constant 495px block around a 268px picture put 227px of nothing between the video and
        the buttons that judge it.

        Not ``heightForWidth``: that is a Qt virtual the layout only consults when the size
        policy opts in, and it would have to be threaded through the player's own layout and
        the ``QStackedWidget`` above it. A host that wants it asks for it.
        """
        picture = max(round(max(width, 1) / self._aspect), self.picture.minimumHeight())
        return picture + self._controls_height(width)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reflow()

    def event(self, event) -> bool:  # noqa: N802 - Qt virtual
        """Re-place both children when the transport needs a different height.

        A child's ``updateGeometry`` posts ``LayoutRequest`` to its parent, which a ``QLayout``
        would answer. This widget has none (see the class docstring), so it only re-placed its
        children on its *own* resize — and a transport that grew after the first layout (the
        player's "tracked" strip appears once a track's breaks are known) kept the height it
        had before, its button row drawn outside it and clipped. ``updateGeometry`` then tells
        the host, whose answer from ``height_for_width`` has changed too.

        **Only when the transport's need changed.** The picture asks too, on every decoded
        frame, and answering each of those re-laid-out the host page per frame: the audit tab
        ran 2-12x slower and settled on either of two heights (measured, 349 and 353px).
        """
        if event.type() == QEvent.Type.LayoutRequest:
            need = self._controls_height(max(self.width(), 1))
            if need != self._controls_need:
                self._reflow()
                self.updateGeometry()
        return super().event(event)

    def _controls_height(self, width: int) -> int:
        """How tall the transport is **at this width**.

        A control row that wraps (`gui/flow.py`) is taller when it is narrow, and this widget
        is exactly where that has to be honoured: nothing here is managed by a ``QLayout``, so
        a row given its one-line ``sizeHint`` height and then squeezed narrow draws its second
        line outside the geometry it was handed, where it is clipped away entirely — every
        control on it still present, still wired, and invisible.

        ``heightForWidth`` returns -1 on a widget that does not opt in, which is every other
        control this block is given, so the one-line hint stays the answer for those.
        """
        wrapped = self.controls.heightForWidth(width) if width > 0 else -1
        return max(wrapped, self.controls.sizeHint().height())

    def sizeHint(self) -> QSize:
        return QSize(
            self.picture.minimumWidth(),
            self.picture.minimumHeight()
            + self._controls_height(self.picture.minimumWidth()),
        )

    def minimumSizeHint(self) -> QSize:
        """The same, as a **floor** — a hint alone is not one, and this widget clips.

        Nothing here is managed by a `QLayout` (see the class docstring), so a host that hands
        this less height than it asked for does not get a squashed-but-working pair: `_reflow`
        holds the picture at its own minimum, and the controls are positioned *below* that, off
        the end of the widget, where they are clipped away entirely. The stream preview showed
        it — a 15px band of video and no transport at all — because a sibling took a fixed
        320px inside a scrolling dialog body and this took what was left.
        """
        return self.sizeHint()

    def _reflow(self) -> None:
        """Position both children from this widget's own current size — one pass, no layout.

        Width to height is not a feedback loop; a size hint derived from what was last drawn is
        (the growth loop `ImageView`'s own docstring exists to avoid). Everything here reads
        only ``self.width()`` / ``self.height()`` (this pane) and ``self._aspect`` (the frame's
        own dimensions, untouched by anything drawn) — never a child's current geometry — so
        repeated calls settle rather than chase each other.
        """
        width = max(self.width(), 1)
        controls_height = self._controls_height(width)
        self._controls_need = controls_height
        available = max(self.height() - controls_height, 0)
        picture_height = min(round(width / self._aspect), available)
        picture_height = max(picture_height, self.picture.minimumHeight())
        # As wide as the picture actually draws, centred under it. A picture held back by the
        # pane's *height* draws narrower than the pane, and controls given the pane's width ran
        # past both edges of the frame they operate — a slider whose ends are not the clip's
        # ends, on every player in a widened window. Never narrower than the transport's own
        # one-line width, though: squeezing it to a small picture would wrap the row instead.
        drawn = min(width, round(picture_height * self._aspect))
        span = min(width, max(drawn, self.controls.sizeHint().width()))
        if span != width:
            controls_height = self._controls_height(span)
            picture_height = max(
                min(picture_height, self.height() - controls_height),
                self.picture.minimumHeight(),
            )
        left = (width - span) // 2
        # Centred: split whatever room is left over the aspect-correct pair between the space
        # above and below it, rather than handing either child the remainder as a stretch.
        spare = max(self.height() - picture_height - controls_height, 0)
        top = 0 if self._top else spare // 2
        self.picture.setGeometry(0, top, width, picture_height)
        self.controls.setGeometry(left, top + picture_height, span, controls_height)
