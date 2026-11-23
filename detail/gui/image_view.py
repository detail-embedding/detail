"""The one pane that shows a picture — scaled to fit, and optionally zoomable.

A **leaf**: it imports nothing from ``detail.gui`` except ``scroll``'s wheel-ownership property,
which is itself a leaf. That is the point. This widget began as ``behavior_query._VideoPane``,
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
"""

from collections.abc import Callable

from detail.gui.qt import (
    QEvent,
    QLabel,
    QPixmap,
    QPoint,
    QPointF,
    QPushButton,
    QSize,
    QSizePolicy,
    QWidget,
    Qt,
    Signal,
)
from detail.gui.scroll import OWNS_WHEEL

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

    clicked = Signal()
    clicked_at = Signal(float, float)
    mark_drag_started = Signal(float, float)
    mark_dragged = Signal(float, float)
    mark_drag_finished = Signal(float, float)

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
        self._zoomable = zoomable
        self._zoom = 1.0
        self._offset = QPoint(0, 0)  # pan, in scaled-pixmap pixels
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
        self._press = event.position().toPoint()
        self._dragged = False
        self._mark_drag = False
        if (
            self.hit_test is not None
            and event.button() == Qt.MouseButton.LeftButton
            and self.hit_test(event.position().toPoint())
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
            delta = event.position().toPoint() - self._press
            if self._dragged or delta.manhattanLength() > _DRAG_SLOP_PX:
                self._dragged = True
                self._press = event.position().toPoint()
                self._offset += delta
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
        if self._zoom == 1.0:
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
        if zoom == _MIN_ZOOM:
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
