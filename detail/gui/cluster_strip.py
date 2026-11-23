"""The member strip and the ethograms under a cluster (§22) — what a cluster *is*, on screen.

Two widgets, both about the same question: a person deciding whether to name a cluster needs to
see what is in it and where it happens.

``ClusterClipStrip`` is a single scrollable row of every member, not the first eight. The cell is
the picture — no frame, no caption, no margin — because a grid of bordered boxes with grey padding
reads as chrome around the data rather than the data. Each thumbnail carries the subject's mask
tinted in that cluster's colour, so a picture, its point in the scatter and its band in the
ethogram are recognisably one thing.

Clicking a cell plays **that same crop** once, animated in place: the subject-cropped square the
thumbnail already shows, moving. Not the recording's full frame in a scrub player — the strip's
whole job is comparing members of one cluster against each other, and swapping one of them for a
different view of a different thing breaks the comparison mid-glance.

The row **scales**: drag its lower edge down and the pictures grow while fewer fit, which is the
trade a person makes when a behavior stops being distinguishable at thumbnail size. Frames are
rescaled from the decoded array, so growing the row costs no decoding.

``EthogramStrip`` is one row per recording, the whole timeline, every cluster drawn in its colour
**and its pattern** (`plot_theme.label_mark`: never hue alone) with the selected one full strength
and the rest dimmed. Where a behavior sits relative to the others is
the thing a scatter cannot show — and a cluster that only ever happens in the first two minutes is
a finding a person will not get from a list of thumbnails. **Clicking a band selects its cluster**,
the same way clicking a point in the scatter does: both are pictures of the vocabulary, and a
picture you can point at is worth more than one you can only read.

Deliberately not a grid of framed, captioned cells. A cell with a frame, a score caption and a
selection shadow is for judging one clip at a time; this strip is for comparing members against
each other, and the two want opposite things from a cell.
"""

import colorsys

from detail.core.chart_theme import rgb
from detail.gui import frame_source
from detail.gui import decode_order
from detail.gui import plot_theme
from detail.gui.clip_grid import _CellSignals, _CellTask
from detail.gui.qt import (
    QAbstractScrollArea,
    QColor,
    QEvent,
    QFrame,
    QHBoxLayout,
    QLabel,
    QObject,
    QPainter,
    QPen,
    QPixmap,
    QPoint,
    QPointF,
    QPushButton,
    QRectF,
    QRunnable,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTimer,
    QToolTip,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)
from detail.gui.quick_preview import (
    MAX_THUMB_PX,
    MIN_THUMB_PX,
    THUMB_PX,
    QuickPreview,
)
from detail.gui.run_preview import _to_pixmap
from detail.gui.scroll import OWNS_WHEEL
from detail.gui.sizing import lines
from detail.gui.zoom import DRAG_PX, OverviewBar, ZoomWindow, zoom_pinch, zoom_wheel

# Frames decoded for one playing clip. A bout is a second or two, but a long one at 59 rows/s
# would be hundreds of seeks and a hundred megabytes of crops, so a longer clip is strided rather
# than truncated — playing a slowed-down whole is closer to the behavior than playing its first
# half at speed.
MAX_PLAY_FRAMES = 120


def _pool():
    """Qt's global thread pool — the one every thumbnail decode in the app shares."""
    from detail.gui.qt import QThreadPool

    return QThreadPool.globalInstance()


class _ClipSignals(QObject):
    # (generation, frames | None) — delivered on the UI thread.
    done = Signal(int, object)


class _ClipTask(QRunnable):
    """Decode one clip's crops on a pool worker, so playback never seeks on the UI thread."""

    def __init__(self, generation: int, producer, signals: _ClipSignals) -> None:
        super().__init__()
        self._generation = generation
        self._producer = producer
        self._signals = signals

    def run(self) -> None:
        try:
            frames = self._producer()
        except Exception:  # noqa: BLE001 — boundary: a pool worker never raises into Qt
            frames = None
        try:
            self._signals.done.emit(self._generation, frames)
        except RuntimeError:
            pass  # the strip was torn down mid-decode — drop it


class _Thumb(QLabel):
    """One member, as its picture and nothing else.

    A bare ``QLabel``: no frame shape, no layout margin, no caption row. The previous cell was a
    ``QFrame`` with a 2px panel, 2px margins and a caption under a square image box, so a 16:9
    crop sat in a grey surround roughly a third the height of the picture — which is what "there
    is a rectangular frame around them with empty space" was describing.
    """

    clicked = Signal(object)

    def __init__(self, size_px: int = THUMB_PX) -> None:
        super().__init__()
        self._size_px = int(size_px)
        self._clip = None
        # The decoded array, kept so the row can be resized without decoding anything again.
        self._rgb = None
        self.setFixedSize(self._size_px, self._size_px)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setText("…")

    def set_clip(self, clip) -> None:
        self._clip = clip
        self._rgb = None
        self.setPixmap(QPixmap())
        self.setText("…")

    def image(self):
        """The decoded frame this thumbnail is showing, if it has one.

        Handed to the playing cell so a clip that is still decoding shows its own first frame
        instead of an empty square.
        """
        return self._rgb

    def set_size(self, size_px: int) -> None:
        """Grow or shrink, redrawing from the array rather than re-decoding."""
        self._size_px = int(size_px)
        self.setFixedSize(self._size_px, self._size_px)
        if self._rgb is not None:
            self._paint(self._rgb)

    def _paint(self, rgb) -> None:
        import numpy as np

        self.setPixmap(
            _to_pixmap(np.ascontiguousarray(rgb)).scaled(
                self._size_px,
                self._size_px,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_image(self, rgb, reason) -> None:
        if reason is not None:
            # Named, not blank: which failure it is decides what the user does about it.
            self._rgb = None
            self.setText(frame_source.unavailable_text(reason))
            return
        self._rgb = rgb
        self._paint(rgb)

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if self._clip is not None:
            self.clicked.emit(self._clip)
        super().mousePressEvent(event)


class ClusterClipStrip(QWidget):
    """Every member of the selected cluster, in one resizable scrollable row."""

    def __init__(self, experiment, size_px: int = THUMB_PX, parent=None) -> None:
        super().__init__(parent)
        from detail.api.run_preview import ClipQueryPreview

        self._experiment = experiment
        self._preview = ClipQueryPreview(experiment.dir)
        self._size_px = int(size_px)
        self._tint: tuple[int, int, int] | None = None
        self._generation = 0
        self._play_generation = 0
        self._thumbs: list[_Thumb] = []
        self._clips: list = []
        #: Indices with a decode in flight, so a scroll does not queue them a second time.
        self._pending: set[int] = set()
        self._playing: int | None = None
        # Unparented and shared: a task holds a reference, so it outlives a strip whose host
        # closed mid-decode and can still emit safely.
        self._signals = _CellSignals()
        self._signals.done.connect(self._on_decoded)
        self._clip_signals = _ClipSignals()
        self._clip_signals.done.connect(self._on_clip_decoded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._row = QWidget()
        self._row_layout = QHBoxLayout(self._row)
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.setSpacing(2)
        self._row_layout.addStretch()
        self._scroll.setWidget(self._row)
        self._scroll.horizontalScrollBar().valueChanged.connect(self._on_scrolled)
        layout.addWidget(self._scroll)

        # One playing cell for the strip, moved into whichever member was clicked. Ninety members
        # cannot each own a decoder and a timer.
        self._cell = QuickPreview(self._size_px)
        self._cell.double_clicked.connect(self._open_playing_clip)
        self._cell.hide()

        self._empty = QLabel("")
        self._empty.setWordWrap(True)
        layout.addWidget(self._empty)
        self._apply_size()

    # --- sizing -------------------------------------------------------------------------

    def resize_by(self, delta: int) -> None:
        """Drag down for bigger pictures and fewer of them; up for the opposite.

        Called by the band's handle. The handle lives on `CollapsibleBand` now rather than
        inside this widget, so every section is draggable by construction instead of the two
        that remembered to add one.
        """
        self.set_thumb_size(self._size_px + delta)

    def set_thumb_size(self, size_px: int) -> None:
        size_px = max(MIN_THUMB_PX, min(int(size_px), MAX_THUMB_PX))
        if size_px == self._size_px:
            return
        self._size_px = size_px
        self._apply_size()

    def _apply_size(self) -> None:
        """Re-scale in place. Every frame is kept as its decoded array, so growing the row
        redraws rather than re-decoding — a resize must not cost a hundred video seeks.
        """
        for thumb in self._thumbs:
            thumb.set_size(self._size_px)
        self._cell.set_size(self._size_px)
        # The pictures, the playing cell's controls, and the horizontal scrollbar.
        self._scroll.setFixedHeight(self._size_px + 40)

    # --- members ------------------------------------------------------------------------

    def show_clips(self, clips: list, *, tint=None, empty_text: str = "") -> None:
        """Point the strip at a cluster's members. ``tint`` colours their masks."""
        self._generation += 1
        self._stop_playing()
        self._tint = None if tint is None else tuple(int(c) for c in tint)
        self._clips = list(clips)
        self._empty.setText("" if clips else empty_text)
        self._empty.setVisible(not clips)

        while len(self._thumbs) < len(clips):
            thumb = _Thumb(self._size_px)
            thumb.clicked.connect(self._on_clicked)
            self._thumbs.append(thumb)
            self._row_layout.insertWidget(len(self._thumbs) - 1, thumb)
        for index, thumb in enumerate(self._thumbs):
            thumb.setVisible(index < len(clips))
            if index < len(clips):
                thumb.set_clip(clips[index])

        self._pending = set()
        self._scroll.horizontalScrollBar().setValue(0)
        # Not one task per clip. A cluster hands this up to 400 members and the page draws them
        # into three strips, so firing everything meant ~800 video decodes queued at once, in
        # index order — the pictures on screen behind every picture to their left. Decode what is
        # visible, and let scrolling ask for the rest.
        self._decode_visible()

    def _decode_visible(self) -> None:
        """Queue a decode for the thumbnails on screen, nearest the middle of the view first.

        The order comes from `gui/decode_order.py`, which the query strip uses too — the two
        strips answer the same question and had one mechanism between them.

        The layout has to be forced first. `show_clips` creates the thumbnails and Qt positions
        them at the next paint, so at this point every cell reports ``width() == 0`` and the
        on-screen filter would admit all of them — the bound would silently do nothing and the
        ordering would be back to index order, which is the bug. Both levels are needed inside a
        `widgetResizable` scroll area: the area sizes the row, so activating only the inner layout
        measures it at its old width.

        Nothing is queued while the strip is off screen, and `showEvent` is what asks once it is
        on. A hidden strip has no geometry worth ordering against, and the pictures are wanted
        when somebody can see them — see the module docstring of `gui/decode_order.py`.
        """
        if not self._clips or not self.isVisible():
            return
        self._row.adjustSize()
        self._row_layout.activate()
        # `isHidden`, not `isVisible`: the question is which cells *this strip* turned on — the
        # ones past the cluster's length, and the one a playing clip replaced, are hidden here.
        # `isVisible` answers a different question, ancestor included, so it returns False for
        # every cell of a strip nobody has shown yet — and the gate above is where that belongs.
        cells = {
            i: t
            for i, t in enumerate(self._thumbs)
            if i < len(self._clips) and not t.isHidden()
        }
        left = self._scroll.horizontalScrollBar().value()
        width = self._scroll.viewport().width()
        centre = decode_order.centre_index(
            cells, viewport_left=left, viewport_width=width
        )
        order = decode_order.visible_first(
            cells,
            viewport_left=left,
            viewport_width=width,
            cell_px=self._size_px,
            centre=centre,
        )
        generation = self._generation
        queued = 0
        for index in order:
            if index in self._pending or self._thumbs[index].image() is not None:
                continue  # already on its way, or already decoded — a scroll must not re-decode
            self._pending.add(index)
            _pool().start(
                _CellTask(
                    generation,
                    index,
                    self._producer_for(self._clips[index]),
                    self._signals,
                )
            )
            queued += 1
            if queued >= decode_order.DECODE_BUDGET:
                break

    def showEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """Becoming visible is what asks for the pictures.

        A strip is commonly filled before it is shown — the interpretation band populates both
        typicality halves and *then* reveals them, and a folded `CollapsibleBand` hides its
        content outright — so the populate cannot be the only trigger. It was, and the two
        typicality strips stayed on `…` forever unless the row was long enough to scroll (#395).
        """
        super().showEvent(event)
        self._decode_visible()

    def _on_scrolled(self, _value: int) -> None:
        """Scrolling brings the next screenful in. Cheap: everything decoded already is skipped."""
        self._decode_visible()

    def _producer_for(self, clip):
        """A Qt-free callable a pool worker runs — the span's **middle** frame, masked.

        The middle rather than the first: a behavior's opening frame is the least recognisable
        moment of it.
        """
        preview = self._preview
        recording_id = clip.recording_id
        embed_run_id = clip.embed_run_id
        track_id = int(clip.track_id)
        middle = (int(clip.start_t_ns) + int(clip.end_t_ns)) // 2
        size = MAX_THUMB_PX
        tint = self._tint

        def produce():
            frame = preview.t_ns_to_frame(recording_id, embed_run_id, middle)
            return preview.source_crop_masked(  # off-thread: a `_CellTask` worker
                recording_id, embed_run_id, track_id, frame, size, tint=tint
            )

        return produce

    def _on_decoded(self, generation: int, index: int, rgb, reason) -> None:
        if generation != self._generation or index >= len(self._thumbs):
            return  # the selection moved on; this picture is for a cluster nobody is looking at
        self._pending.discard(index)
        self._thumbs[index].set_image(rgb, reason)

    # --- playback -----------------------------------------------------------------------

    def _on_clicked(self, clip) -> None:
        """Play in place: the clicked member's own crop, animated in its own slot."""
        # By position in the clips this strip is showing, not by widget state: a playing cell is
        # hidden, so a lookup that skipped hidden cells could never find the one to stop.
        index = next((i for i, shown in enumerate(self._clips) if shown is clip), None)
        if index is None:
            return
        if self._playing == index:
            self._stop_playing()
            return
        self._stop_playing()
        self._row_layout.insertWidget(index, self._cell)
        self._cell.set_loading(self._thumbs[index].image())
        self._thumbs[index].hide()
        self._cell.show()
        self._playing = index

        self._play_generation += 1
        _pool().start(
            _ClipTask(
                self._play_generation, self._clip_producer(clip), self._clip_signals
            )
        )

    def _clip_producer(self, clip):
        """Decode the clip's crops, at the same size and tint the thumbnail uses.

        The **same crop**, which is the point: clicking a member swaps a still picture for a
        moving one and changes nothing else. A player showing the recording's whole frame would
        answer a different question than the one the strip is for.
        """
        preview = self._preview
        recording_id = clip.recording_id
        embed_run_id = clip.embed_run_id
        track_id = int(clip.track_id)
        start_t_ns, end_t_ns = int(clip.start_t_ns), int(clip.end_t_ns)
        tint = self._tint

        def produce():
            first = preview.t_ns_to_frame(recording_id, embed_run_id, start_t_ns)
            last = preview.t_ns_to_frame(recording_id, embed_run_id, end_t_ns)
            total = max(last - first + 1, 1)
            # Ceiling, not floor: `1001 // 120` is 8, and a stride of 8 over 1001 frames yields
            # 126 of them — past the cap the stride exists to enforce.
            step = max(1, -(-total // MAX_PLAY_FRAMES))
            fps = preview.frame_rate(recording_id, embed_run_id) or 30.0
            # One pass over the span, not one seek per frame: the loop this replaces re-opened
            # and re-sought the container for every frame, over an order of magnitude
            # slower, which is the "clips take a really long time to load" bug.
            frames = list(
                preview.source_crops_masked(  # off-thread: a `_ClipTask` worker
                    recording_id,
                    embed_run_id,
                    track_id,
                    first,
                    last,
                    MAX_THUMB_PX,
                    tint=tint,
                    step=step,
                )
            )
            # The interval carries the stride, so a strided clip plays at real duration rather
            # than at real frame rate — the behavior takes as long on screen as it did.
            return frames, int(round(1000.0 * step / max(fps, 1e-6)))

        return produce

    def _on_clip_decoded(self, generation: int, payload) -> None:
        if generation != self._play_generation or self._playing is None:
            return  # a different member was clicked while this one was decoding
        frames, interval_ms = payload if payload else ([], 40)
        self._cell.set_frames(frames, interval_ms)

    def _open_playing_clip(self) -> None:
        """Double-clicked the preview: open the clip properly, in a window.

        The strip holds which member is in the cell; the preview only holds the frames it was
        handed, which is why it asks rather than opens.
        """
        if self._playing is None or self._playing >= len(self._clips):
            return
        from detail.gui import clip_popup

        self._cell.stop()  # two of the same clip playing at once is nobody's intention
        clip_popup.open_clip_window(
            self, self._experiment, self._clips[self._playing], context_s=0
        )

    def _stop_playing(self) -> None:
        if self._playing is None:
            return
        self._cell.stop()
        self._cell.hide()
        self._row_layout.removeWidget(self._cell)
        if self._playing < len(self._thumbs):
            self._thumbs[self._playing].show()
        self._playing = None


def cluster_colors(n: int) -> list[tuple[float, float, float, float]]:
    """One RGBA per cluster, as floats: the chart theme's identity colours, in order.

    The six validated slots, then the theme's frozen extension (24 in all), then a hue circle.
    It used to switch the *whole* vocabulary from Okabe-Ito to a hue circle the moment a run had
    a seventh cluster, so every cluster changed colour when one was added — colour following
    the count rather than the cluster. Now the first 24 never move; past them, colours come
    from a circle that divides by ``n``, not ``n - 1`` — hue 0 and hue 1 are the same red, so
    closing the circle handed the first and last cluster one colour.

    The current theme's steps, the ones `plot_theme.label_swatch` draws BROWSE's key in: a band
    and the swatch that names it are one colour in either theme. Light steps were a holdover from
    when these also tinted video thumbnails, and left five of a dark window's bands a step off
    their key.
    """
    # The label colours: BROWSE's lanes are behavior labels, so they keep off the two the
    # diverging scale has for less and more (`ChartTheme.label_identity`).
    theme = plot_theme.current()
    fixed = len(theme.label_identities())
    out = []
    for i in range(max(n, 1)):
        if i < fixed:
            r, g, b = rgb(theme.label_identity(i))
        else:
            r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 0.85, 0.95)
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
        out.append((r / 255.0, g / 255.0, b / 255.0, 1.0))
    return out


def _nearest(marks, fraction: float, tolerance: float) -> int | None:
    """Index of the ``(start, end, …)`` mark nearest ``fraction``, if within ``tolerance``.

    Nearest rather than first: a recording can hold hundreds of bands, so most are **narrower than
    a pixel**. Widening each to a pixel and taking the first match makes neighbours overlap and
    hands the click to whichever came first — on a real 733-band row that selected cluster 24 for a
    click squarely inside cluster 22. The tolerance still leaves a genuine gap selecting nothing.
    """
    best, best_gap = None, None
    for index, mark in enumerate(marks):
        start, end = mark[0], mark[1]
        gap = (
            0.0
            if start <= fraction <= end
            else min(abs(fraction - start), abs(fraction - end))
        )
        if gap <= tolerance and (best_gap is None or gap < best_gap):
            best, best_gap = index, gap
    return best


#: The narrowest window a zoom reaches. Below it a bout is a frame or two and a mark is a pixel
#: wide whatever the zoom, so further zoom shows nothing more.
MIN_VIEW_NS = 2_000_000_000


def clock(ns: float) -> str:
    """``12:04`` or ``1:02:30`` — time from a recording's start, as a ruler labels it."""
    seconds = int(round(ns / 1e9))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


#: Ruler steps, in seconds: the first that leaves ~90px between ticks wins.
_TICK_STEPS_S = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200)


class EthogramStrip(QWidget):
    """One row per recording on **one time axis**: where every cluster happens.

    **Clicking a band selects its cluster**, exactly as clicking a point in the scatter does. Both
    are pictures of the vocabulary, and the thing a person does on seeing an interesting stretch is
    point at it — a picture that can only be read makes them find the row in a table instead.

    The selected cluster is drawn at full strength and the rest dimmed, so the question the row
    answers is "where does *this* one sit among the others" rather than "what colour is
    everything".

    **A band is its colour and its pattern**, both from its index (`plot_theme.paint_label_mark`),
    so two clusters or labels never differ by hue alone. The pattern is anchored to the strip, so
    neighbouring bands of one identity join into one texture; a band narrower than a pattern's
    period (about 6px) cannot show one, and zooming in is what gives it the room. Time the
    subject was not observed is left blank rather than filled: a tracking gap is not a behavior,
    and at several dozen clusters a fully tiled bar reads as noise.

    **Every row starts at 0:00 on a shared axis**, and a shorter recording ends early. Each row
    used to be stretched to its own length, so the middle of a row was the middle of *that*
    video: harmless at the whole span, and wrong as soon as the strip zooms, when one screen
    position was a different minute in every row and no ruler could label them all.

    **The wheel zooms around the pointer** and a drag pans, as on the cluster tree; Shift+wheel
    moves through the recordings, and Esc or a double-click on blank time shows the whole span.
    The zoom is one window shared by every row, so the rows stay comparable minute for minute.

    The recording's name sits in a **left gutter**, not on a line of its own above the bar. Above,
    it cost 14px a row — 40% of each row was a caption — and an experiment has dozens.
    """

    BAR_PX = 16
    #: Between one row and the next.
    GAP_PX = 4
    #: The optional lane under each bar (`set_lane`): thin, because it answers "where is this one"
    #: beside the bar rather than competing with it.
    LANE_PX = 6
    #: The widest a name may make the gutter, in lines; a longer name is elided, whole on hover.
    GUTTER_LINES = 12

    cluster_picked = Signal(int)
    #: The band under a click, as ``(row, index into that row's bands)``: what an embedder that
    #: knows what each band *is* (a labelled bout, say) acts on. Emitted beside ``cluster_picked``.
    band_activated = Signal(int, int)
    #: A mark in the lane was clicked, as ``(row, index into that row's lane marks)``.
    lane_activated = Signal(int, int)
    #: The visible window moved: a zoom, a pan, or back to the whole span.
    view_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # for Esc
        # The wheel zooms here, as on the tree; the page scrolls everywhere else.
        self.setProperty(OWNS_WHEEL, True)
        self._describe = None
        self._describe_lane = None
        self._rows: list[tuple[str, list[tuple[float, float, int]]]] = []
        self._lengths: list[int] = []
        self._axis_ns = 1
        #: The visible window, as fractions of the axis (the longest recording) — the shared
        #: zoom, which the overview bar above the strips (`ruled`) is bound to.
        self.zoom = ZoomWindow(self)
        self.zoom.changed.connect(self._on_zoom)
        self._colors: list = []
        self._selected: int | None = None
        #: ``(row, band index)`` outlined as the one an embedder is showing, or ``None``.
        self._highlight: tuple[int, int] | None = None
        self._lane: list[list[tuple[float, float]]] | None = None
        self._lane_color: QColor | None = None
        self._gutter_for: tuple | None = None
        self._gutter_px = 0
        #: A row outlined for a moment after something asked to show it (`mark_row`).
        self._marked: int | None = None
        self._unmark = QTimer(self)
        self._unmark.setSingleShot(True)
        self._unmark.timeout.connect(lambda: self.mark_row(None))
        self._press: QPointF | None = None
        self._press_view = self.zoom.view
        self._dragging = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    # --- geometry: the one source paint, hit-tests and embedders all read ----------------------

    def row_px(self) -> int:
        """One row's height — the bar, the lane when there is one, and the gap to the next."""
        lane = self.LANE_PX + 2 if self._lane is not None else 0
        return self.BAR_PX + lane + self.GAP_PX

    def _bar_top(self, row: int) -> int:
        return row * self.row_px() + self.GAP_PX // 2

    def _lane_top(self, row: int) -> int:
        return self._bar_top(row) + self.BAR_PX + 2

    def row_top(self, row: int) -> int:
        """Where ``row`` starts, for an embedder scrolling it into view."""
        return row * self.row_px()

    @property
    def _gutter(self) -> int:
        """The names' column: the widest name, capped, measured in the font the widget has *now*.

        Measured on use rather than in ``set_rows``: a widget's font is settled by its style only
        once it is polished, so a width taken at ``set_rows`` elided names that fit.
        """
        if not self._rows:
            return 0
        key = (self.font().key(), tuple(name for name, _bands in self._rows))
        if key != self._gutter_for:
            metrics = self.fontMetrics()
            widest = max(metrics.horizontalAdvance(name) for name, _bands in self._rows)
            # +4 of slack before the 8px gap: elision measures in floats, so a name exactly the
            # column's width rounded a pixel over it and lost its last characters.
            self._gutter_px = min(widest + 4, lines(self.GUTTER_LINES)) + 8
            self._gutter_for = key
        return self._gutter_px

    def track(self) -> tuple[int, int]:
        """``(left, width)`` of the time axis in pixels: everything right of the names."""
        width = max(self.width() - self._gutter, 1)
        return self._gutter, width

    def _axis_x(self, t: float) -> float:
        """Pixel x of ``t``, a fraction of the axis, under the current zoom."""
        left, width = self.track()
        v0, v1 = self.zoom.view
        return left + (t - v0) / (v1 - v0) * width

    def _x(self, fraction: float, row: int) -> int:
        """Pixel x of ``fraction`` of ``row``'s own recording."""
        return int(round(self._axis_x(fraction * self._scale(row))))

    def _scale(self, row: int) -> float:
        """``row``'s length as a fraction of the axis: 1 for the longest recording."""
        if not 0 <= row < len(self._lengths):
            return 1.0
        return self._lengths[row] / self._axis_ns

    def _fraction(self, x: float, row: int) -> float | None:
        """Where ``x`` falls in ``row``'s own recording, or ``None`` over the names or past its
        end."""
        left, width = self.track()
        if x < left:
            return None
        v0, v1 = self.zoom.view
        t = v0 + (x - left) / width * (v1 - v0)
        fraction = t / self._scale(row)
        return fraction if 0.0 <= fraction <= 1.0 else None

    def _tolerance(self, row: int) -> float:
        """A pixel and a half, in ``row``'s own fractions at the current zoom."""
        v0, v1 = self.zoom.view
        return 1.5 / self.track()[1] * (v1 - v0) / self._scale(row)

    def _fit(self) -> None:
        self.setFixedHeight(max(len(self._rows), 1) * self.row_px())

    # --- the time window ------------------------------------------------------------------------

    @property
    def axis_ns(self) -> int:
        """The axis's length: the longest recording's span."""
        return self._axis_ns

    @property
    def view(self) -> tuple[float, float]:
        """The visible window, as fractions ``(left, right)`` of the axis."""
        return self.zoom.view

    def is_zoomed(self) -> bool:
        return self.zoom.is_zoomed()

    def fit(self) -> None:
        """The whole span again."""
        self.zoom.fit()

    def set_view(self, left: float, right: float) -> None:
        """Show ``(left, right)`` of the axis, kept inside it and no narrower than
        ``MIN_VIEW_NS``."""
        self.zoom.set_view(left, right)

    def zoom_by(self, factor: float, anchor_x: float | None = None) -> None:
        """Scale the window by ``factor`` (< 1 zooms in) around ``anchor_x`` in pixels: the time
        under the pointer stays under it."""
        self.zoom.zoom_by(factor, self._share(anchor_x))

    def _share(self, x: float | None) -> float:
        """How far across the time axis pixel ``x`` is, 0 to 1; the middle for ``None``."""
        left, width = self.track()
        return 0.5 if x is None else (x - left) / width

    def _on_zoom(self) -> None:
        self.view_changed.emit()
        self.update()

    # --- what is drawn ------------------------------------------------------------------------

    def set_lane(
        self, lane: list[list[tuple[float, float]]] | None, color: QColor | None = None
    ) -> None:
        """A thin lane under each bar — ``[(start_fraction, end_fraction), …]`` per row — or none.

        Kept apart from the rows, so ``set_rows`` leaves it standing: the lane is a second fact (a
        selection) on the same picture, and a redraw of the first must not drop it.
        """
        self._lane = None if lane is None else [list(marks) for marks in lane]
        self._lane_color = color
        self._fit()
        self.update()

    def set_highlight(self, band: tuple[int, int] | None) -> None:
        """Outline one band — the bout an embedder's list has selected — or none.

        An outline rather than a dimming of the rest, because dimming already means "the selected
        cluster" here, and the bout being watched is a different fact on the same picture.
        """
        self._highlight = band
        self.update()

    def mark_row(self, row: int | None, *, for_ms: int = 1500) -> None:
        """Outline ``row`` for a moment — the one a picker just scrolled to — or clear it."""
        self._marked = row
        if row is None:
            self._unmark.stop()
        else:
            self._unmark.start(for_ms)
        self.update()

    def set_rows(self, rows, colors, selected: int | None, *, lengths_ns) -> None:
        """``rows`` is ``[(recording, [(start_fraction, end_fraction, cluster_id), …]), …]``;
        ``lengths_ns`` is each row's recording span, in the same order.

        Fractions of the row's own recording, so an embedder never converts to the axis. The
        lengths have no default: without them every row would be drawn the length of the
        longest, which is the per-row stretching this axis replaced — plausible, and wrong.
        """
        lengths = [max(int(n), 1) for n in lengths_ns]
        if len(lengths) != len(rows):
            raise ValueError(
                f"lengths_ns has {len(lengths)} entries for {len(rows)} rows: one per row"
            )
        axis = max(lengths, default=1)
        self.zoom.set_floor(min(MIN_VIEW_NS / axis, 1.0))
        if axis != self._axis_ns:
            self.zoom.fit()  # a window into another axis means nothing here
        self._rows = list(rows)
        self._lengths = lengths
        self._axis_ns = axis
        self._colors = list(colors)
        self._selected = None if selected is None else int(selected)
        self._highlight = None
        self._fit()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt's name
        # A chart like any other, so the chart theme's surface, track and ink — the same lane an
        # export's ethogram draws. The widget palette's `AlternateBase` stood in for the track,
        # and under Fusion's dark scheme that is a dark teal, which read as a cluster colour.
        theme = plot_theme.current()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.surface))
        left, width = self.track()
        metrics = painter.fontMetrics()
        for row, (recording_id, _bands) in enumerate(self._rows):
            top = self._bar_top(row)
            painter.setPen(QColor(theme.text_secondary))
            name = metrics.elidedText(
                recording_id, Qt.TextElideMode.ElideRight, max(self._gutter - 8, 0)
            )
            painter.drawText(
                QRectF(0, top, max(self._gutter - 8, 0), self.BAR_PX),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                name,
            )
        visible = QRectF(left, 0, width, self.height())
        painter.setClipRect(visible)
        painter.setPen(Qt.PenStyle.NoPen)
        for row, (_recording_id, bands) in enumerate(self._rows):
            top = self._bar_top(row)
            # The empty bar first, as long as this recording is, so unobserved time and gaps
            # between bands read as absence rather than as a colour nobody defined — and the
            # end of a shorter recording reads as its end.
            painter.setBrush(QColor(theme.grid))
            x0, x1 = self._x(0.0, row), self._x(1.0, row)
            painter.drawRect(x0, top, x1 - x0, self.BAR_PX)
            # Grouped by identity, so each one's pattern is stroked once along the row rather
            # than once per bout (`plot_theme.paint_label_mark`).
            by_identity: dict[int, list[QRectF]] = {}
            for start, end, cluster in bands:
                x0 = self._x(start, row)
                x1 = max(self._x(end, row), x0 + 1)
                # Cut to what is on screen first: zoomed in, one band can be a hundred thousand
                # pixels wide, and its pattern would be drawn across all of them.
                band = QRectF(x0, top, x1 - x0, self.BAR_PX).intersected(visible)
                by_identity.setdefault(int(cluster), []).append(band)
            for cluster, rects in by_identity.items():
                plot_theme.paint_label_mark(
                    painter,
                    rects,
                    cluster,
                    color=self._color_for(cluster),
                    theme=theme,
                )
            self._paint_lane(painter, row)
        self._paint_highlight(painter)
        painter.setClipping(False)
        self._paint_marked(painter)
        painter.end()

    def _paint_lane(self, painter, row: int) -> None:
        if self._lane is None or row >= len(self._lane):
            return
        theme = plot_theme.current()
        top = self._lane_top(row)
        painter.setBrush(QColor(theme.grid))
        x0, x1 = self._x(0.0, row), self._x(1.0, row)
        painter.drawRect(x0, top, x1 - x0, self.LANE_PX)
        painter.setBrush(self._lane_color or QColor(theme.text))
        for start, end in self._lane[row]:
            x0 = self._x(start, row)
            x1 = max(self._x(end, row), x0 + 1)
            painter.drawRect(x0, top, x1 - x0, self.LANE_PX)

    def _paint_highlight(self, painter) -> None:
        """Stroke the highlighted band, at least a few pixels wide so a sliver still shows."""
        if self._highlight is None:
            return
        row, index = self._highlight
        if not (0 <= row < len(self._rows) and 0 <= index < len(self._rows[row][1])):
            return
        start, end, _cluster = self._rows[row][1][index]
        top = self._bar_top(row)
        x0 = self._x(start, row)
        x1 = max(self._x(end, row), x0 + 1)
        pad = max(0, (4 - (x1 - x0)) // 2)
        pen = QPen(QColor(plot_theme.current().text))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(x0 - pad - 1, top - 1, x1 - x0 + 2 * pad + 2, self.BAR_PX + 2)

    def _paint_marked(self, painter) -> None:
        if self._marked is None or not 0 <= self._marked < len(self._rows):
            return
        pen = QPen(QColor(plot_theme.current().text))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        top = self.row_top(self._marked)
        painter.drawRect(1, top + 1, self.width() - 2, self.row_px() - 2)

    # --- pointing -----------------------------------------------------------------------------

    def set_describer(self, describe, describe_lane=None) -> None:
        """``describe(row, index) -> str`` for a band's hover text; ``None`` for no hover.
        ``describe_lane`` is the same for a mark in the lane."""
        self._describe = describe
        self._describe_lane = describe_lane

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        self._press = event.position()
        self._press_view = self.zoom.view
        self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if self._press is None:
            return
        dx = event.position().x() - self._press.x()
        if not self._dragging and abs(dx) < DRAG_PX:
            return
        self._dragging = True
        if self._press_view == (0.0, 1.0):
            return  # nothing to pan: the whole span is on screen
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        v0, v1 = self._press_view
        shift = -dx / self.track()[1] * (v1 - v0)
        self.set_view(v0 + shift, v1 + shift)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """A click picks; a drag only pans. Deciding on release is what tells them apart."""
        pressed, dragged = self._press, self._dragging
        self._press, self._dragging = None, False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if pressed is None or dragged:
            return
        self.click_at(pressed)

    def click_at(self, pos) -> None:
        """What a click at ``pos`` does: play a lane mark, pick a band's cluster."""
        mark = self.lane_at(pos)
        if mark is not None:
            self.lane_activated.emit(*mark)
        cluster = self._cluster_at(pos)
        if cluster is not None:
            self.cluster_picked.emit(int(cluster))
        band = self.band_at(pos)
        if band is not None:
            self.band_activated.emit(*band)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """On blank time, the whole span again; on a band, a second click like the first."""
        pos = event.position()
        if self.band_at(pos) is None and self.lane_at(pos) is None:
            self.fit()

    def wheelEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """The wheel zooms around the pointer; Shift+wheel moves through the recordings; a
        sideways scroll (trackpad) pans. The tree's mapping, so one gesture means one thing.
        """
        delta = event.angleDelta()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Some platforms turn Shift+wheel into a horizontal delta; either axis means "down".
            area = self._scroll_area()
            if area is None:
                event.ignore()
                return
            notches = (delta.y() or delta.x()) / 120.0
            bar = area.verticalScrollBar()
            bar.setValue(int(bar.value() - notches * self.row_px() * 2))
        else:
            self.zoom_from(event, event.position().x())
        event.accept()

    def zoom_from(self, event, x: float) -> bool:
        """A wheel event zooms around ``x`` (this strip's pixels) — wherever it was turned."""
        return zoom_wheel(self.zoom, event, self._share(x))

    def _scroll_area(self):
        """The scroll area the rows live in, if an embedder put them in one."""
        widget = self.parentWidget()
        while widget is not None:
            if isinstance(widget, QAbstractScrollArea):
                return widget
            widget = widget.parentWidget()
        return None

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if event.key() == Qt.Key.Key_Escape and self.is_zoomed():
            self.fit()
            event.accept()
            return
        super().keyPressEvent(event)

    def event(self, event) -> bool:  # noqa: D401 — Qt's override
        if event.type() == QEvent.Type.NativeGesture and zoom_pinch(
            self.zoom, event, self._share(event.position().x())
        ):
            return True
        if event.type() == QEvent.Type.ToolTip:
            pos = QPointF(event.pos())
            mark = self.lane_at(pos)
            band = self.band_at(pos) if self._describe is not None else None
            name = self._name_at(pos)
            if mark is not None and self._describe_lane is not None:
                QToolTip.showText(event.globalPos(), self._describe_lane(*mark), self)
            elif band is not None:
                QToolTip.showText(event.globalPos(), self._describe(*band), self)
            elif name is not None:
                QToolTip.showText(event.globalPos(), name, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)

    def _row_under(self, y: float, top_of, height: int) -> int | None:
        for row in range(len(self._rows)):
            top = top_of(row)
            if top <= y < top + height:
                return row
        return None

    def _name_at(self, pos) -> str | None:
        """The whole name under ``pos`` in the gutter — the elided one, read in full."""
        if pos.x() >= self._gutter:
            return None
        row = self._row_under(pos.y(), self._bar_top, self.BAR_PX)
        return None if row is None else self._rows[row][0]

    def lane_at(self, pos) -> tuple[int, int] | None:
        """``(row, mark index)`` of the lane mark nearest ``pos`` within a pixel and a half."""
        if self._lane is None:
            return None
        row = self._row_under(pos.y(), self._lane_top, self.LANE_PX)
        if row is None or row >= len(self._lane):
            return None
        fraction = self._fraction(pos.x(), row)
        if fraction is None:
            return None
        index = _nearest(self._lane[row], fraction, self._tolerance(row))
        return None if index is None else (row, index)

    def band_at(self, pos) -> tuple[int, int] | None:
        """``(row, band index)`` of the band nearest ``pos`` within a pixel and a half."""
        row = self._row_under(pos.y(), self._bar_top, self.BAR_PX)
        if row is None:
            return None
        fraction = self._fraction(pos.x(), row)
        if fraction is None:
            return None
        index = _nearest(self._rows[row][1], fraction, self._tolerance(row))
        return None if index is None else (row, index)

    def _cluster_at(self, pos):
        """The cluster of the band under ``pos``, or ``None`` on blank time or over the names.

        Blank selects nothing rather than the nearest band: the gaps are where the subject was not
        observed or the vocabulary accounts for nothing, and snapping a click there to a neighbour
        would claim a band covers time it does not.
        """
        band = self.band_at(pos)
        if band is None:
            return None
        row, index = band
        return self._rows[row][1][index][2]

    def _color_for(self, cluster: int) -> QColor:
        if not self._colors:
            return QColor(plot_theme.current().neutral)
        r, g, b, _a = self._colors[cluster % len(self._colors)]
        color = QColor(int(r * 255), int(g * 255), int(b * 255))
        if self._selected is not None and cluster != self._selected:
            # Dimmed rather than hidden: the point of the row is where this cluster sits *among*
            # the others, which needs the others visible.
            color.setAlpha(70)
        return color


class EthogramRuler(QWidget):
    """The time over an `EthogramStrip`: ticks and clock labels across the visible window.

    Its own widget, above the scroll area the strip's rows live in, so it stays put while the
    recordings scroll under it. It reads every position off the strip (`track`, `view`), so the
    two cannot disagree about where a minute is. Where the window sits in the whole is the
    overview bar's job (`ruled`), a real scrollbar rather than a band drawn here.
    """

    RULER_PX = 18

    def __init__(self, strip: EthogramStrip, parent=None) -> None:
        super().__init__(parent)
        self._strip = strip
        strip.view_changed.connect(self.update)
        self.setFixedHeight(self.RULER_PX)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _track(self) -> tuple[float, float]:
        """The strip's time axis, in this widget's x."""
        left, width = self._strip.track()
        origin = self._strip.mapTo(self.window(), QPoint(0, 0)).x()
        mine = self.mapTo(self.window(), QPoint(0, 0)).x()
        return left + origin - mine, width

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt's name
        theme = plot_theme.current()
        painter = QPainter(self)
        left, width = self._track()
        v0, v1 = self._strip.view
        axis = self._strip.axis_ns
        span_s = (v1 - v0) * axis / 1e9
        step = next(
            (s for s in _TICK_STEPS_S if s / max(span_s, 1e-9) * width >= 90),
            _TICK_STEPS_S[-1],
        )
        painter.setPen(QColor(theme.text_secondary))
        metrics = painter.fontMetrics()
        t = int((v0 * axis / 1e9) // step) * step
        while t <= v1 * axis / 1e9:
            x = left + (t * 1e9 / axis - v0) / (v1 - v0) * width
            if x >= left:
                painter.drawLine(QPointF(x, 12), QPointF(x, self.height()))
                label = clock(t * 1e9)
                # A label that would run past the axis's end is left off rather than cut: the
                # widget's right edge is the key column's, not the time's.
                if x + 2 + metrics.horizontalAdvance(label) <= left + width:
                    painter.drawText(QPointF(x + 2, 11), label)
            t += step
        painter.end()


#: What the hint row says while nothing is zoomed: the gestures are invisible until named.
ZOOM_HINT = (
    "Scroll over the ethograms to zoom · drag to pan · Shift+scroll for more videos"
)


class _WheelToStrip(QObject):
    """Hands a wheel or a pinch over the strip's surroundings — the blank area under its rows,
    the ruler — to the strip's zoom, at the time under the pointer.

    Shift+wheel is left alone: over the scroll area it already moves through the recordings,
    which is what Shift means on the strip itself.
    """

    def __init__(self, strip: EthogramStrip, parent=None) -> None:
        super().__init__(parent)
        self._strip = strip

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 — Qt's name
        kind = event.type()
        if kind == QEvent.Type.Wheel:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                return False
            return self._strip.zoom_from(event, self._x(event))
        if kind == QEvent.Type.NativeGesture:
            return zoom_pinch(
                self._strip.zoom, event, self._strip._share(self._x(event))
            )
        return False

    def _x(self, event) -> float:
        return float(self._strip.mapFromGlobal(event.globalPosition().toPoint()).x())


def ruled(strip: EthogramStrip, area: QWidget | None = None) -> QWidget:
    """The strip's header — a hint row that becomes **Whole** and the overview bar while zoomed,
    then the ruler — with ``area`` (the scroll area holding ``strip``) under it when given.

    One builder, so every ethogram gets its time, its zoom and its hint the same way. The row
    keeps one height zoomed or not, so the strips never jump when the overview appears.
    """
    holder = QWidget()
    column = QVBoxLayout(holder)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(2)
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    # Enabled, not greyed: a disabled label cannot be selected and copied, and the app's text is.
    holder.hint = QLabel(ZOOM_HINT)
    holder.whole = QPushButton("Whole")
    holder.whole.setObjectName("ethogram_whole")
    holder.whole.setToolTip(
        "Show the whole time span (Esc, or double-click blank time)"
    )
    holder.whole.clicked.connect(strip.fit)
    # The row has a fixed height of its own, so the hidden bar's room goes to the hint.
    holder.overview = OverviewBar(strip.zoom, retain_space=False)
    row.addWidget(holder.hint, 1)
    row.addWidget(holder.whole)
    row.addWidget(holder.overview, 1)
    row_box = QWidget()
    row_box.setLayout(row)
    row_box.setFixedHeight(holder.whole.sizeHint().height())
    column.addWidget(row_box)
    holder.ruler = EthogramRuler(strip)
    column.addWidget(holder.ruler)
    # The whole plot area zooms, not only the rows: under a few recordings most of the area is
    # blank, and a wheel there scrolled nothing while the rows a centimetre above zoomed.
    holder.wheel_to_strip = _WheelToStrip(strip, holder)
    holder.ruler.installEventFilter(holder.wheel_to_strip)
    if area is not None:
        column.addWidget(area)
        viewport = getattr(area, "viewport", None)
        (viewport() if viewport else area).installEventFilter(holder.wheel_to_strip)

    def sync() -> None:
        zoomed = strip.is_zoomed()
        holder.hint.setVisible(not zoomed)
        holder.whole.setVisible(zoomed)

    strip.view_changed.connect(sync)
    sync()
    return holder
