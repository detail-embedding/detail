"""A strip of clip thumbnails that fills progressively (§20).

The calibration view's whole basis is looking at *what a cutoff returns*, so the clips near the
current setting have to be on screen while the user drags. That makes the decode economics the
widget's shape rather than an implementation detail:

- **Progressive, not batched.** Each cell decodes on its own and paints when it lands. The
  batched alternative (``gui/tools/quick_segment.py``'s ``_SampledMatrix``, which gathers every
  frame in one worker before painting any of them) shows nothing until the slowest cell is done,
  and on a drag that is every cell, every time.
- **Latest-wins per cell, and moving the cutoff abandons the old band.** A drag retargets the
  strip faster than a decode returns, so a result arriving for a band the user has left must not
  paint. That is what ``_generation`` is for.
- **A thread pool, not a thread per cell.** ``BackgroundCall`` is one OS thread per call, which
  over a strip that retargets on every drag is a thread storm. ``QThreadPool`` bounds it, owns
  the runnables, and is what ``FrameScrubber`` already uses for exactly this decode.

**A cell that cannot decode says which reason.** ``gui/frame_source.py`` keeps *missing* (the
video is not on this machine) and *undecodable* (this position has no picture) apart, and a
surface that collapses them into a blank pane reports neither. The thumbnail template this
otherwise follows gets that wrong — a failed cell there reads "rendering…" forever — and the
audit (§20) will show a clip and take a judgment against it, so a blank pane there becomes a
recorded decision about a frame nobody saw.

Kept out of ``modules/`` because §21 lists the clip grid as shared across module tabs and Browse,
not as Query's.
"""

from __future__ import annotations

import math

import numpy as np

from detail.gui import decode_order
from detail.gui import frame_source
from detail.gui.lifetime import discard
from detail.gui.qt import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QObject,
    QPixmap,
    QPushButton,
    QRadioButton,
    QRunnable,
    QScrollArea,
    QSettings,
    QThreadPool,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)
from detail.gui.run_preview import _to_pixmap
from detail.gui.sizing import lines, remember_size, remembered_size

# A thumbnail's side in pixels. Big enough to recognise a posture, small enough that eight sit
# in a row on a laptop without the strip owning the tab.
THUMB_PX = 150


class _CellSignals(QObject):
    # (generation, index, rgb | None, reason | None) — delivered on the UI thread.
    done = Signal(int, int, object, object)


class _CellTask(QRunnable):
    """Decode one cell's thumbnail on a pool worker and hand it back.

    ``signals`` is unparented and shared, the way ``FrameScrubber``'s is: a task holds a
    reference to it, so it outlives a grid whose host closed mid-decode and can still emit
    safely.
    """

    def __init__(
        self, generation: int, index: int, producer, signals: _CellSignals
    ) -> None:
        super().__init__()
        self._generation = generation
        self._index = index
        self._producer = producer
        self._signals = signals

    def run(self) -> None:
        try:
            rgb, reason = frame_source.decode(self._producer)
        except Exception:  # noqa: BLE001 — boundary: a pool worker never raises into Qt
            # `frame_source.decode` classifies the two *expected* failures. Anything else —
            # a recording record that no longer resolves, a corrupt store — would otherwise
            # escape into the thread pool, where it prints a traceback nobody reads and
            # leaves the cell showing its placeholder forever. From the user's side a frame
            # that could not be produced is undecodable, whatever the reason, so it says so.
            rgb, reason = None, frame_source.UNDECODABLE
        try:
            self._signals.done.emit(self._generation, self._index, rgb, reason)
        except RuntimeError:
            pass  # the grid was torn down mid-decode — drop it


class _Cell(QFrame):
    """One thumbnail with its caption. Clickable, and it says why it is blank when it is."""

    def __init__(self, size_px: int = THUMB_PX, parent=None) -> None:
        # ``parent`` is not decoration. A widget built without one is a **top-level window**, and
        # macOS gives each native backing: building the strip's 300-cell pool parentless took
        # seconds, against milliseconds with the parent passed in. A caller that adds the cell to a layout
        # a moment later still pays for the window it was in the meantime.
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._size_px = int(size_px)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self._image = QLabel("…")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setWordWrap(True)
        # Fixed, and **the frame's own shape**. A square box was right when these were crops
        # around the subject; a whole frame is ~16:9, so a 180-wide picture is 101 tall and a
        # square label left 79px of empty label under every one of them — the white space that
        # made a row of full frames look wasteful at any size.
        #
        # Fixed rather than minimum for the original reason too: a minimum lets the layout
        # stretch the cell to fill the pane, which is how four thumbnails ended up spread
        # across 1450px with a 150px picture in the middle of each.
        self._image.setFixedSize(self._size_px, round(self._size_px * 9 / 16))
        #: The frame it is showing, so a resize is a rescale rather than a re-decode.
        self._rgb = None
        self._caption = QLabel("")
        self._caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._image)
        layout.addWidget(self._caption)
        # The caption sits under its own picture, not floating in the middle of the cell. Cells
        # are stretched vertically to the tallest thing in the row — the divider — and without
        # this the slack goes into the caption, which drifts away from the frame it names.
        # Same rule as a pane's trailing stretch (`gui_design` §3), one level in.
        layout.addStretch()
        self._clip = None
        self._on_click = None
        self._on_double_click = None

    def set_clip(
        self, clip, on_click, caption: str | None = None, on_double_click=None
    ) -> None:
        self._clip = clip
        self._on_click = on_click
        self._on_double_click = on_double_click
        # Clear the previous band's picture before the new decode lands. Without this a
        # retargeted strip is a mix of two cutoffs, which reads as one that did not update.
        self._image.setPixmap(QPixmap())
        self._image.setText("…")
        if caption is None:
            caption = "" if clip is None or clip.score is None else f"{clip.score:.2f}"
        self._caption.setText(caption)
        self.setCursor(
            Qt.CursorShape.ArrowCursor
            if clip is None
            else Qt.CursorShape.PointingHandCursor
        )
        # Derived from the wiring, so a strip cannot advertise an opener it has not connected.
        self.setToolTip(
            "Double-click to open this clip"
            if clip is not None and on_double_click is not None
            else ""
        )

    def set_selected(self, on: bool) -> None:
        """Mark the cell whose clip is loaded in the player.

        A frame shape change, never a stylesheet (GUI style: zero QSS). Without it the strip
        gives no sign of which thumbnail the player is showing.

        A **drawn box**, not a shadow. `Sunken` on a `StyledPanel` is what this used to be, and
        rendering the row showed the marked cell and its neighbours as the same picture — the
        style paints a one-pixel bevel that survives a screenshot only just. Which clip is
        marked now decides where the cutoff goes, so it has to read at a glance.
        """
        self.setFrameShape(QFrame.Shape.Box if on else QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setLineWidth(3 if on else 1)

    def set_image(self, rgb, reason) -> None:
        if reason is not None:
            # Named, not blank: which of the two failures this is decides what the user does
            # about it, and a surface that shows an empty pane reports neither.
            self._image.setText(frame_source.unavailable_text(reason))
            return
        self._rgb = rgb
        self._image.setPixmap(
            _to_pixmap(np.ascontiguousarray(rgb)).scaled(
                self._size_px,
                self._size_px,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_size(self, size_px: int) -> None:
        """Resize in place, keeping the frame it is already showing.

        In place because the pool is identity-guarded: `test_strip_reuse.py` asserts a cell
        survives a cutoff nudge as the *same object*, and that growing the pool creates no
        top-level widgets (parentless construction of a pool takes seconds).
        Rebuilding cells to resize them would walk into both.
        """
        self._size_px = int(size_px)
        self._image.setFixedSize(self._size_px, round(self._size_px * 9 / 16))
        if self._rgb is not None:
            self.set_image(self._rgb, None)

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if self._clip is not None and self._on_click is not None:
            self._on_click(self._clip)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """Out of the thumbnail and into a window.

        A single click here already means *mark this one* — it is what `Cut here` acts on, and it
        loads the clip into the player beside the row. So this cell does not play in place the way
        the clustering strip's does; what it lacks is a way to see a clip at more than 132px, and
        that is this.
        """
        if self._clip is not None and self._on_double_click is not None:
            self._on_double_click(self._clip)
        super().mouseDoubleClickEvent(event)


class ClipGrid(QWidget):
    """Clip thumbnails, decoded off the UI thread and filled as they arrive.

    One row by default; ``columns`` wraps them into a block.
    """

    clip_clicked = Signal(object)  # the ClipSpec whose cell was clicked

    def __init__(
        self, experiment, n_cells: int = 8, columns: int | None = None, parent=None
    ) -> None:
        super().__init__(parent)
        from detail.api.run_preview import ClipQueryPreview

        # Its own preview instance: the scrub-context cache inside one is mutable, and sharing
        # it between two surfaces that decode different positions was already found to be a
        # problem (`_ClipPlayer` owns one per player for the same reason).
        self._experiment = experiment  # the popup a double-click opens needs it
        self._preview = ClipQueryPreview(experiment.dir)
        self._generation = 0
        self._signals = _CellSignals()
        self._signals.done.connect(self._on_decoded)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._empty = QLabel("")
        self._empty.setWordWrap(True)
        root.addWidget(self._empty)
        row = QWidget()
        self._grid = QGridLayout(row)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._cells = [_Cell() for _ in range(n_cells)]
        # One row unless a caller asks to wrap. A tall pane wants a block of clips rather than
        # a single line of them, and this widget is shared with Browse and §21's audit, so the
        # shape belongs here rather than special-cased by one tab.
        width = columns or n_cells
        for index, cell in enumerate(self._cells):
            self._grid.addWidget(cell, index // width, index % width)
        root.addWidget(row)
        self._row = row

    def show_clips(self, clips: list, *, empty_text: str = "") -> None:
        """Point the grid at a new set of clips, abandoning the previous band's decodes.

        ``empty_text`` is what the strip says when there is nothing to show — a behavior that
        returns nothing at this cutoff is a *result*, and an empty row of frames reads as a
        surface that failed to load.
        """
        self._generation += 1
        generation = self._generation
        shown = list(clips)[: len(self._cells)]
        self._empty.setText("" if shown else empty_text)
        self._row.setVisible(bool(shown))
        for index, cell in enumerate(self._cells):
            clip = shown[index] if index < len(shown) else None
            cell.setVisible(clip is not None)
            cell.set_clip(clip, self._emit_clicked, on_double_click=self._open_clip)
            if clip is None:
                continue
            producer = self._producer_for(clip)
            QThreadPool.globalInstance().start(
                _CellTask(generation, index, producer, self._signals)
            )

    def _producer_for(self, clip):
        """A Qt-free callable the pool worker runs — the seam `frame_source.decode` wraps.

        The thumbnail is the span's **middle** frame: its first frame is often the behavior
        beginning, which is the least recognisable moment of it.
        """
        preview = self._preview
        recording_id = clip.recording_id
        embed_run_id = clip.embed_run_id
        track_id = int(clip.track_id)
        middle_t_ns = (int(clip.start_t_ns) + int(clip.end_t_ns)) // 2

        def produce():
            frame = preview.t_ns_to_frame(recording_id, embed_run_id, middle_t_ns)
            return preview.source_crop(  # off-thread: a QThreadPool worker runs this
                recording_id, embed_run_id, track_id, frame, THUMB_PX
            )

        return produce

    def _on_decoded(self, generation: int, index: int, rgb, reason) -> None:
        if generation != self._generation:
            return  # the user has moved the cutoff; this band is not on screen any more
        self._cells[index].set_image(rgb, reason)

    def set_selected(self, clip) -> None:
        """Mark whichever cell holds ``clip`` (and unmark the rest)."""
        for cell in self._cells:
            cell.set_selected(cell._clip is not None and cell._clip == clip)

    def _open_clip(self, clip) -> None:
        """Double-clicked a thumbnail: open it in a window, whole frame and full rate.

        The same gesture as the clustering strip's, opening the same window, so the two strips
        stop diverging on what a thumbnail can do.
        """
        from detail.gui import clip_popup

        clip_popup.open_clip_window(self, self._experiment, clip, context_s=0)

    def _emit_clicked(self, clip) -> None:
        self.clip_clicked.emit(clip)


class ClipGridModes(QWidget):
    """The strip's two modes, as a labelled radio row (§20).

    The default is the **marginal band** — the clips *at* the cutoff, because the decision the
    view exists for is "is what sits at this level real?". Sampled-above is the second mode
    rather than the default: a strip of obvious hits tells you nothing about where to cut.
    """

    mode_changed = Signal(str)

    NEAR = "near"
    ABOVE = "above"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Clips"))
        self._near = QRadioButton("near cutoff")
        self._above = QRadioButton("sampled above")
        self._near.setChecked(True)
        for button, mode in ((self._near, self.NEAR), (self._above, self.ABOVE)):
            button.toggled.connect(
                lambda checked, m=mode: checked and self.mode_changed.emit(m)
            )
            row.addWidget(button)
        row.addStretch()

    def mode(self) -> str:
        return self.NEAR if self._near.isChecked() else self.ABOVE


# --- the §20 confidence strip ------------------------------------------------------------- #

#: How many candidates either side of the cutoff the row holds.
#:
#: The row is a **neighbourhood of the cutoff**, not the ranking. That is what the decision is:
#: you scroll near the boundary until the clips stop being the behavior, and you move the line
#: there. Nobody traverses the whole list — and at scale nobody could. `soft_knn` has no
#: candidate cap and yields ~1,200 per thirty-minute recording, so a hundred hours pooled is
#: ~240,000; 20,000 cells already take seconds and a lot of memory to build and make the row four
#: million pixels wide, and a quarter of a million would be forty-nine million — past what Qt
#: will paint.
#:
#: So the window is fixed and **nothing loads at its edges**. That is deliberate: paging there
#: is what yanked the scrollbar handle out from under a dragging mouse and what let one drag
#: fire the load over and over. To see past the window you move the cutoff, which re-centres it
#: — the same gesture the screen is for.
_WINDOW = 150

#: How many frames one decode pass may queue.
#:
#: About a screenful either side of what is being looked at, at the sizes this strip uses — enough
#: that a scroll lands on warm cells, few enough that the eight pool threads are working on what
#: is in front of the user rather than on a queue of 300 they will never reach. Scrolling and the
#: post-layout pass ask again.
_DECODE_BUDGET = decode_order.DECODE_BUDGET
#: How many decoded frames to hold, across window moves.
_CACHE_CAP = 400
#: Layout spacing plus frame around one cell. Used to place the scroll from an *index* rather
#: than from a widget's geometry, which is not available until Qt has laid the row out.
_CELL_GAP_PX = 14
#: What the cut button says with nothing to cut at. Named, because it is also what the button
#: goes back to after a cut — the marked clip is then the boundary, and cutting there again is
#: a no-op the button should not offer.
_CUT_IDLE = "Cut here"


# Thumbnail sizes the user picks between. Small enough that a wide pane shows a dozen at once,
# large enough that the largest is worth looking at before opening the player.
#: Scores closer than this, relatively, are one score. The search computes in float32, and
#: how its sums round depends on how BLAS splits them across threads, so one match found in two
#: recordings can come back a unit in the last place apart (0.85658836 and 0.85658842 on a
#: 32-thread machine, equal on a 4-thread one). Counting those as two made every caption take
#: the full five decimals trying to separate them. Ten times float32's resolution, and far
#: below the 1e-5 a five-decimal caption can show.
SAME_SCORE_REL_TOL = 1e-6


def score_digits(scores) -> int:
    """Decimals enough to tell these scores apart, capped so a caption stays a caption.

    Two is what a *cutoff* reads like and it is useless beside a boundary: the matches either
    side of one agree to two places, so a kept clip and a cut one both printed `0.66` and a
    list of them looked like it was not updating.
    """
    values = sorted(v for v in scores if v is not None)
    distinct = sum(
        1
        for i, v in enumerate(values)
        if i == 0 or not math.isclose(v, values[i - 1], rel_tol=SAME_SCORE_REL_TOL)
    )
    digits = 2
    while digits < 5 and len({f"{v:.{digits}f}" for v in values}) < distinct:
        digits += 1
    return digits


#: The one size a confidence row uses. There is no picker and nothing persisted: a whole
#: frame at 180 wide is what it takes to see what an animal is doing, and the sizes below that
#: existed for crops, where a smaller picture still showed the subject filling it.
STRIP_FRAME_PX = 180
#: How far the row can be dragged. The floor is where a whole frame stops showing what the
#: animal is doing, which is the size the crops used to sit at.
MIN_STRIP_PX = 120
MAX_STRIP_PX = 480
# `query/thumb_columns` is gone with the grid: a row fits what it fits, and how many clips
# are on screen is now a consequence of the pane's width rather than a setting.


class ConfidenceStrip(QWidget):
    """Every match for one behavior, ordered by confidence, with the cutoff drawn **in** it.

    Distinct from :class:`ClipGrid`, which is a fixed set of cells a caller re-points. This
    grows: you scroll toward more or less confident matches and it loads more, and the cutoff
    is a line at its own place in the ordering rather than a number set somewhere else.

    That inversion is the point. A cutoff was set by dragging a marker on the frequency curve,
    where the whole confidence axis is a couple of hundred pixels wide — about 0.005 per pixel,
    against an interesting range three decimal places across. Here the decision is made where
    the evidence is: scroll until the clips stop being the behavior, and put the line there.

    Pooled across recordings on purpose (§20): the boundary is a property of the cohort, and a
    per-recording strip would hide one animal dominating it. Each cell names its recording so
    the pooling stays legible.

    Scrolling only **loads**; it never commits. Moving the line is what changes the cutoff, and
    ``cutoff_released`` is what commits it — the same split the frequency curve uses, so a
    browse cannot rewrite the run's artifact.
    """

    clip_clicked = Signal(object)
    cutoff_moved = Signal(float)  # dragged: update the view only
    cutoff_released = Signal(float)  # let go: commit

    def __init__(self, experiment, columns: int = 3, parent=None) -> None:
        super().__init__(parent)
        from detail.api.run_preview import ClipQueryPreview

        self._experiment = experiment  # the popup a double-click opens needs it
        self._preview = ClipQueryPreview(experiment.dir)
        #: ``{recording_id: name}`` for the captions (`set_names`); an id with no name shows
        #: as itself, a recording deleted since the run.
        self._names: dict[str, str] = {}
        self._columns = columns
        self._generation = 0
        self._signals = _CellSignals()
        self._signals.done.connect(self._on_decoded)
        self._all: list = []  # every clip, ascending by confidence
        self._lo = 0  # the window into `_all`, always centred on the cutoff
        self._hi = 0
        self._cutoff = 0.5
        self._digits = 2
        self._cells: dict[int, _Cell] = {}  # index in `_all` -> its cell
        #: The reusable row. `_rebuild` re-points these rather than constructing fresh ones —
        #: the window is up to `2 * _WINDOW` wide, and rebuilding it was the switch lag.
        self._pool: list[_Cell] = []
        #: Whether the *reader* has scrolled since the view was last centred on the cutoff.
        #: Until they have, the view belongs to the cutoff and follows it through a resize.
        self._user_scrolled = False
        #: Set while this widget moves the scrollbar itself, so its own scrolling is not
        #: mistaken for the reader's.
        self._scrolling_itself = False
        self._handle = None
        #: One deferred centring at a time — a second would connect a second `rangeChanged`.
        self._centring = False
        #: The clip loaded in the player, by index into `_all`. Kept on the strip rather than
        #: on the cell, because `_rebuild` throws every cell away: marking held on the widget
        #: vanished the moment the cutoff moved, which is exactly when you are looking at it.
        self._marked: int | None = None
        #: Decoded frames, by clip identity. A cutoff move changes which side of the divider a
        #: clip falls on, never the picture, so a rebuild must not pay for it twice.
        self._decoded: dict = {}
        self._pending: set = set()

        self._size_px = remembered_size("query/clips", STRIP_FRAME_PX)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        # One row, scrolled sideways: the decision is "walk along the ordering until the clips
        # stop being the behavior", which is a left-right gesture on a confidence axis.
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        bar = self._scroll.horizontalScrollBar()
        bar.valueChanged.connect(self._on_scrolled)
        # Also on `rangeChanged`. The button's state is "can this row scroll", which is the
        # scrollbar's maximum — and that is 0 until Qt has laid the cells out, so reading it
        # inside `_rebuild` always said no. Same trap as every other geometry read here.
        bar.rangeChanged.connect(lambda _lo, _hi: self._sync_recentre())
        self._body = QWidget()
        self._rows = QHBoxLayout(self._body)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._scroll.setWidget(self._body)
        root.addWidget(self._scroll)
        self._apply_size()

        ends = QHBoxLayout()
        # Statements, not buttons. Reaching an end loads the next page by itself, so what a
        # reader needs there is whether there *is* more — pushing against a limit otherwise
        # reads the same as a stall.
        self._left_edge = QLabel("")
        ends.addWidget(self._left_edge)
        ends.addStretch()
        # The counts live here, not in the band. Five stacked lines did not fit a row sized for
        # one frame, and "kept" was clipped off the bottom — the band holds the value and the
        # steppers, which is what has to be *at* the divider to grab.
        self._counts = _heading("")
        ends.addWidget(self._counts)
        ends.addSpacing(12)
        # Always present, disabled while the divider is in view. Hidden-until-needed was the
        # first version and it was wrong: the recorded rule is grey out, don't hide — a
        # capability nobody can see is one nobody knows they have (gui_design.md §3.1).
        self._recentre = QPushButton("⤒ centre on cutoff")
        self._recentre.clicked.connect(self.back_to_cutoff)
        ends.addWidget(self._recentre)
        ends.addSpacing(12)
        # Setting the cutoff from a clip you scrolled to, without giving a cell a second
        # meaning. Clicking a clip plays it, as it always has; this button — which names the
        # clip and what keeping from it costs — is the only thing that moves the line. A
        # click target inside the row would have had to be told apart from *play*, and a
        # target you have to think about beside a row you are browsing is the wrong trade.
        self._cut = QPushButton(_CUT_IDLE)
        self._cut.setObjectName("cut_at_marked")
        self._cut.clicked.connect(self.cut_at_marked)
        # Wide enough for the longest thing it can say, so it does not resize under the
        # pointer. The label changes on every click — it names the clip you just marked — and
        # a button that grows from "Cut here" pushes the row's other controls sideways while
        # you are clicking along it.
        self._cut.setMinimumWidth(
            self._cut.fontMetrics()
            .boundingRect("Cut here — keep from 0.0000 (000000 kept)")
            .width()
            + 24
        )
        ends.addWidget(self._cut)
        ends.addStretch()
        self._right_edge = QLabel("")
        ends.addWidget(self._right_edge)
        root.addLayout(ends)

        self._empty = QLabel("")
        self._empty.setWordWrap(True)
        root.addWidget(self._empty)

    def visible_cells(self) -> int:
        """How many cells the viewport can show at once — what widening the pane buys.

        The row does not reflow: cells are one fixed size and the pane simply fits more of
        them, which is what keeps a given scroll distance meaning a fixed number of clips.
        """
        return max(self._scroll.viewport().width() // (self._size_px + 14), 1)

    def resizeEvent(self, event):  # noqa: N802 — Qt's name
        super().resizeEvent(event)
        self._sync_recentre()

    # -- what it shows ------------------------------------------------------------------ #

    def show_clips(self, clips: list, cutoff: float, *, empty_text: str = "") -> None:
        """Point the strip at a behavior's whole ranked list, opening at ``cutoff``."""
        self._generation += 1
        # **Ascending**, so the row reads left-to-right on the same axis as the frequency
        # curve above it: least confident (dropped) on the left, most confident (kept) on the
        # right, and the divider sits where the curve's marker sits. A row running the other
        # way to the graph it is under would be two pictures of one number pointing opposite
        # ways.
        self._all = sorted(clips, key=lambda c: (c.score or 0.0))
        # A different behavior's ranking: the old index points at somebody else's clip, and a
        # button naming it would offer to cut at a clip nobody is looking at.
        self._marked = None
        self._digits = self._score_digits()
        self._cutoff = float(cutoff)
        self._empty.setText("" if self._all else empty_text)
        self._recentre_window()

    def cutoff(self) -> float:
        return self._cutoff

    def set_cutoff(self, value: float) -> None:
        """Move the line without emitting — for a caller that moved it (the graph, the box).

        Distinct from :meth:`move_cutoff`, which is the strip's *own* control emitting. Two
        surfaces drive one value, so the one that did not originate the change must not echo
        it back; that is the same asymmetry the frequency curve's two pyqtgraph signals have.
        """
        self._cutoff = float(min(max(value, 0.0), 1.0))
        self._recentre_window()

    def _score_digits(self) -> int:
        return score_digits([c.score for c in self._all if c.score is not None])

    def resize_by(self, delta: int) -> None:
        """Grow or shrink the frames — the band's drag handle calls this.

        This row shipped at one fixed size, with a comment saying there was deliberately no
        picker: a whole frame at 180 wide is what it takes to see what an animal is doing,
        unlike the crops the smaller sizes were for. That argues for the *default*, not against
        a drag — and a band you can drag beside a band you cannot is what got reported. The
        reasoning survives as the starting size.

        Cells are resized in place rather than rebuilt, because the pool's identity is guarded
        (`test_strip_reuse.py`).
        """
        size = max(MIN_STRIP_PX, min(self._size_px + delta, MAX_STRIP_PX))
        if size == self._size_px:
            return
        self._size_px = size
        for cell in self._pool:
            cell.set_size(size)
        self._apply_size()
        remember_size("query/clips", size)
        self._rebuild()

    def _apply_size(self) -> None:
        """Give the row a height to match its cells.

        Without this the scroll area kept whatever height the layout gave it, so dragging made
        the pictures bigger *inside a fixed viewport* — the row scrolled instead of the section
        growing, which is exactly what a drag must not do. `ClusterClipStrip` has sized its own
        scroll area from the cell size since it was written; this row never did, because until
        now its cells could not change size.
        """
        picture = round(self._size_px * 9 / 16)
        chrome = lines(2) + self._scroll.horizontalScrollBar().sizeHint().height()
        self._scroll.setFixedHeight(picture + chrome)

    def set_names(self, names: dict[str, str]) -> None:
        """The recordings' names, which the captions show (§11: the id is the tool's).

        Read off the UI thread by the embedder and handed in before the clips are shown;
        captioning by id is what this replaced, with the Browse tab beside it showing names.
        """
        self._names = dict(names)

    def _caption_for(self, clip) -> str:
        """Which animal, and how confident — pooling interleaves recordings, so say which."""
        score = "" if clip.score is None else f"{clip.score:.{self._digits}f}"
        name = self._names.get(clip.recording_id, clip.recording_id)
        return f"{name}\n{score}".strip()

    def _boundary_index(self) -> int:
        """Where the cutoff falls in the ordering — the first clip it **keeps**.

        Ascending order, so everything left of this index is dropped and everything from it
        rightward is kept. `len(self._all)` means the cutoff is above every candidate.
        """
        return self._boundary_for(self._cutoff)

    def _boundary_for(self, value: float) -> int:
        """Where a *prospective* cutoff would fall — what a cut is about to do.

        Split out so the cut button's promise and the cut itself are one calculation. Scores
        tie: two clips at 1.0000 are one cutoff, so cutting at the second of them changes
        nothing, and a button that compared *indices* went on offering a press that did.
        """
        for index, clip in enumerate(self._all):
            if (clip.score or 0.0) >= value:
                return index
        return len(self._all)

    def back_to_cutoff(self) -> None:
        """Scroll the divider back to the middle, for a browse that wandered.

        A scroll and nothing else. The window is a function of the cutoff, so it cannot have
        drifted — and rebuilding it here was worse than useless: it replaced the divider with a
        fresh widget positioned at 0 until Qt laid it out, and with the row's extent unchanged
        no `rangeChanged` followed to correct it, so the button scrolled nowhere. It also left
        the new divider showing a stale number.
        """
        if not self._all:
            return
        self._centre_handle()
        self._sync_recentre()

    def _recentre_window(self) -> None:
        """Put the window around the cutoff and scroll the divider to the middle.

        Called whenever the cutoff moves, which is the only thing that moves the neighbourhood
        anyone cares about. Nothing else changes the window, so nothing can shift under a
        scrolling mouse.
        """
        boundary = self._boundary_index()
        lo = max(boundary - _WINDOW, 0)
        hi = min(boundary + _WINDOW, len(self._all))
        self._lo, self._hi = lo, hi
        # The window just moved to follow the cutoff, so the view belongs to the cutoff again.
        self._user_scrolled = False
        self._rebuild()
        self._centre_handle()
        self._sync_recentre()
        self._decode_visible()

    def _centre_handle(self) -> None:
        """Put the divider in the middle of the view, once the row has actually been laid out.

        The one place this widget scrolls itself, and it reads the handle's **real** position
        rather than predicting it. Both halves of that matter, and each was a bug:

        *Waiting.* `_rebuild` creates the cells and Qt lays them out afterwards, so at the
        moment of the call the scrollbar's range is still the old one — usually 0 — and
        `setValue` clamps silently. `rangeChanged` fires twice, the first time with the range
        still (0, 0), so waiting for merely the next one is not enough either.

        *Measuring.* The first version predicted the position as `cells × (frame + gap)`. The
        pitch estimate was a couple of pixels out, which over the 150 cells between the window's
        edge and the divider accumulated to ~300px — so "centre" reliably landed the divider a
        cell and a half left of centre, which is what it looked like: a button that moved the
        row and still did not centre it. The laid-out geometry has no such error.

        A zero `x` is the ambiguous case, and asking the *ordering* resolves it: the divider is
        the row's first widget exactly when the cutoff sits at or below the window's low end, so
        there — and only there — a zero is where it belongs. Anywhere else a zero means Qt has
        not run the layout yet. (The row's own size hints cannot answer this: both the body's
        and the layout's report 0 for cells that were added a moment ago, so a gate built on
        either read "settled" and skipped the wait.)
        """
        bar = self._scroll.horizontalScrollBar()
        handle = self._handle
        if handle is None:
            return

        def centre() -> None:
            if self._handle is None:
                return
            # Force the layout before reading the divider's position. Now that the cells are
            # reused, a rebuild that leaves the row the same width gives Qt no reason to have
            # run the layout by the time this fires — and the handle then reports where it was
            # *last* time, or 0 if it is the freshly built one. Traced on a cut: three calls, at
            # x=0, x=28760 and a row that finally settled at x=28800, so the scrollbar was left
            # 40px shy of centre with nothing further scheduled to correct it.
            # Both levels. `self._rows` is the row's own layout, but the widget it lays out is
            # inside a `widgetResizable` scroll area that sizes it — and until *that* has run,
            # the row is laid out at its old width and every x in it is stale. Traced on a cut:
            # the divider read 28760 through three passes and settled at 28800, leaving the bar
            # 40px shy of centre with nothing scheduled to correct it.
            self._body.adjustSize()
            self._rows.activate()
            middle = self._handle.x() + self._handle.width() // 2
            self._scrolling_itself = True
            try:
                bar.setValue(max(0, middle - self._scroll.viewport().width() // 2))
            finally:
                self._scrolling_itself = False
            self._user_scrolled = False

        if handle.x() > 0 or self._boundary_index() <= self._lo:
            centre()
            return

        def when_ready(_low, high) -> None:
            if high <= 0:  # the first of the two, before the cells have a size
                return
            bar.rangeChanged.disconnect(when_ready)
            self._centring = False
            centre()

        if self._centring:
            return
        self._centring = True
        bar.rangeChanged.connect(when_ready)
        # Belt and braces: a rebuild that leaves the row the same width — nudging the cutoff by
        # one cell inside the window it already has — changes every position and no range, so
        # there is no `rangeChanged` to wait for. One turn of the event loop is enough for the
        # layout either way.
        QTimer.singleShot(0, centre)

    def _ensure_cells(self, needed: int) -> None:
        """Grow the reusable row to ``needed`` cells.

        Never shrinks: the window is bounded by ``2 * _WINDOW``, and a hidden cell costs nothing
        against the cost of building it again.
        """
        while len(self._pool) < needed:
            # Parented **in the constructor**, not after it: see `_Cell.__init__`. Reparenting a
            # moment later still pays for the window it was in the meantime, and that is the
            # whole cost.
            self._pool.append(_Cell(self._size_px, self._body))

    def _lay_out_row(self, boundary: int) -> None:
        """Order the row: the window's cells, with the divider between dropped and kept.

        Emptied with ``takeAt``, which detaches the layout *items* and leaves the widgets alive.
        The ordering is rebuilt from scratch — trivially correct, and the divider's position
        changes on every cutoff move — while the widgets themselves are reused. It is the widget
        construction that was expensive, not the ordering.
        """
        while self._rows.count():
            self._rows.takeAt(0)
        discard(self._handle)
        # The divider is rebuilt rather than re-pointed, unlike the cells: one widget against
        # three hundred, and its labels and steppers all read from the constructor arguments.
        self._handle = _CutoffHandle(self._cutoff, boundary, len(self._all), self)
        for slot, index in enumerate(range(self._lo, self._hi)):
            if index == boundary:
                self._rows.addWidget(self._handle)
            self._rows.addWidget(self._pool[slot])
        if (
            boundary >= self._hi
        ):  # the cutoff is at or above every candidate in the window
            self._rows.addWidget(self._handle)
        self._rows.addStretch()

    def _rebuild(self) -> None:
        """The window's candidates, as cells, in one row with the divider among them.

        The cells are **re-pointed, not rebuilt**. This used to delete every widget in the row
        and construct fresh ones, and the window is up to ``2 * _WINDOW`` wide — so a behavior
        switch and every cutoff move paid for ~300 ``_Cell`` constructions and the layout pass
        after them. A cached switch froze the UI noticeably, with
        most of it here, and the frames were already cached, so it was rebuilding widgets to show
        pictures it already had.

        ``ClipGrid`` above has always worked this way — a fixed set of cells it re-points — and
        this is the same idiom reached from the other side.
        """
        self._ensure_cells(max(self._hi - self._lo, 0))
        self._cells = {}
        if not self._all:
            for cell in self._pool:
                cell.setVisible(False)
            self._left_edge.setText("")
            self._right_edge.setText("")
            self._counts.setText("")
            self._recentre.setEnabled(False)
            self._sync_cut()
            return
        boundary = self._boundary_index()
        # What lies outside the window, and how to reach it. Nothing loads at the edges, so a
        # reader who scrolls to one has to be told that moving the line is the way past it —
        # otherwise the end reads as a bug rather than as the edge of a neighbourhood.
        self._left_edge.setText(
            f"… {self._lo} less confident — move the cutoff to see them"
            if self._lo
            else "least confident match"
        )
        beyond = len(self._all) - self._hi
        self._right_edge.setText(
            f"{beyond} more confident — move the cutoff to see them …"
            if beyond
            else "most confident match"
        )
        self._counts.setText(f"{boundary} dropped  │  {len(self._all) - boundary} kept")

        for slot, index in enumerate(range(self._lo, self._hi)):
            clip = self._all[index]
            cell = self._pool[slot]
            cell.setVisible(True)
            cell.set_clip(
                clip,
                lambda c, i=index: self._emit_clicked(c, i),
                caption=self._caption_for(clip),
                on_double_click=self._open_clip,
            )
            cell.set_selected(index == self._marked)
            self._cells[index] = cell
        for cell in self._pool[max(self._hi - self._lo, 0) :]:
            cell.setVisible(False)
        self._lay_out_row(boundary)
        self._decode_visible()
        self._sync_recentre()
        self._sync_cut()

    def _decode_visible(self) -> None:
        """Decode the frames on screen, nearest the cutoff first, and re-use every cached one.

        The clips do not change when the cutoff moves — only which side of the divider they
        fall on — so a rebuild that re-decoded everything was paying for a picture it already
        had. Nudging the line on the graph re-decoded ~13 full frames each time, which is the
        lag; with the cache it re-decodes none.

        **The on-screen test cannot be asked before Qt has laid the row out.** `_rebuild` creates
        the cells and the layout runs afterwards, so at this point every cell reports
        ``width() == 0`` and ``x()`` a few pixels apart — and the test below then admits *every*
        cell in the window. On a large behavior, hundreds of decodes queued at first
        paint where eight fit on screen, in index order, so the cells at the cutoff — the
        ones being looked at — were decoded far down the queue: seconds before the
        visible row fills and more churn behind it, which also
        starves the player's own decode, since they share the global pool. That is the whole of
        "clip loading is laggy and inconsistent".

        So two things. The order is by distance from **what is being looked at** — the viewport's
        middle once there is a layout to ask, the cutoff before then, which is where the view is
        about to be centred anyway. And the pass is **bounded**: a screenful and its margins, not
        the window. Scrolling and the deferred pass below bring in the rest.

        Nothing is queued while the strip is off screen, and `showEvent` is what asks once it is
        on — this strip sits in a `CollapsibleBand` whose fold is remembered between sessions, so
        a reader who folded *Clips* was paying for a screenful of decodes nobody could see, chosen
        against the viewport the row had before it was folded.
        """
        if not self._all or not self.isVisible():
            return
        for index in self._decode_order()[:_DECODE_BUDGET]:
            key = self._key_for(self._all[index])
            self._pending.add(key)
            QThreadPool.globalInstance().start(
                _CellTask(
                    self._generation,
                    index,
                    _producer(self._preview, self._all[index], self._size_px),
                    self._signals,
                )
            )

    def _decode_order(self) -> list:
        """Which cells still need a picture, nearest what is being looked at first."""
        # Force the layout rather than guess whether it has run. `any(width)` was the guess and
        # it is wrong in the way that matters: after a rebuild the row is *partially* laid out —
        # some cells sized, the ones at the front still reporting `width() == 0` at x-values a
        # few pixels apart — so the check said "laid out" and every geometry read below was
        # nonsense. Activating is a pass this widget needs anyway.
        self._rows.activate()
        left = self._scroll.horizontalScrollBar().value()
        width = self._scroll.viewport().width()
        # The on-screen filter and its margin now live in `gui/decode_order.py`, which the
        # clustering strip asks too — including the rule that a viewport narrower than one cell
        # means the geometry has not settled and must not be filtered against.

        # Cached pictures are painted here rather than filtered inside the ordering: the strip
        # owns its cache, the ordering owns geometry.
        pending = {}
        for index, cell in self._cells.items():
            key = self._key_for(self._all[index])
            cached = self._decoded.get(key)
            if cached is not None:
                cell.set_image(cached, None)
                continue
            if key in self._pending:
                continue
            pending[index] = cell

        # Ordered by distance from the **cutoff**, always. That is where the row is centred and
        # where the decision is made, so it is what the pool threads should be working on; and
        # once the user scrolls away, the filter has already narrowed this to what is on screen,
        # so the ordering within it costs nothing either way. Asking the viewport's middle
        # instead sounds more precise and is not: it is only meaningful once the geometry has
        # settled, which is exactly when it no longer matters. That is why `centre` is passed
        # rather than derived — this strip has a better answer than the viewport's middle, and
        # the clustering strip, which has no cutoff, does not.
        return decode_order.visible_first(
            pending,
            viewport_left=left,
            viewport_width=width,
            cell_px=self._size_px,
            centre=self._boundary_index(),
        )

    @staticmethod
    def _key_for(clip) -> tuple:
        """What makes two clips the same picture."""
        return (
            clip.recording_id,
            int(clip.track_id),
            int(clip.start_t_ns),
            int(clip.end_t_ns),
        )

    def _add_block(self, indices: list[int], generation: int) -> None:
        """One run of cells, left to right, in ascending confidence."""
        for _position, index in enumerate(indices):
            clip = self._all[index]
            cell = _Cell(self._size_px)
            cell.set_clip(
                clip,
                self._emit_clicked,
                caption=self._caption_for(clip),
                on_double_click=self._open_clip,
            )
            self._cells[index] = cell
            self._rows.addWidget(cell)
            QThreadPool.globalInstance().start(
                _CellTask(
                    generation,
                    index,
                    _producer(self._preview, clip, self._size_px),
                    self._signals,
                )
            )

    def _on_decoded(self, generation: int, index: int, rgb, reason) -> None:
        if generation != self._generation or index not in self._cells:
            return
        clip = self._all[index]
        key = self._key_for(clip)
        self._pending.discard(key)
        if rgb is not None:
            # Bounded, and oldest-first: a long browse over a big ranking would otherwise hold
            # every frame it ever showed. `_CACHE_CAP` frames at 180x101 is a few megabytes.
            if len(self._decoded) >= _CACHE_CAP:
                self._decoded.pop(next(iter(self._decoded)))
            self._decoded[key] = rgb
        self._cells[index].set_image(rgb, reason)

    def _open_clip(self, clip) -> None:
        """Double-clicked a thumbnail: open it in a window, whole frame and full rate.

        The same gesture as the clustering strip's, opening the same window, so the two strips
        stop diverging on what a thumbnail can do.
        """
        from detail.gui import clip_popup

        clip_popup.open_clip_window(self, self._experiment, clip, context_s=0)

    def _emit_clicked(self, clip, index: int | None = None) -> None:
        self._marked = index
        for cell_index, cell in self._cells.items():
            cell.set_selected(cell_index == index)
        self._sync_cut()
        self.clip_clicked.emit(clip)

    def marked_index(self) -> int | None:
        """Which clip is loaded in the player, as an index into the whole ranking."""
        return self._marked

    def cut_at_marked(self) -> None:
        """Put the cutoff at the marked clip: it becomes the least confident one kept.

        Commits, like the steppers and unlike a drag — it is a discrete "put it here", and the
        button said what it would do before it was pressed.
        """
        if not self._can_cut():
            return
        self._cutoff = self._marked_score()
        # ...and re-centre, rather than `move_cutoff`'s bare rebuild. The line has just moved
        # to somewhere off-screen by definition — you scrolled there — so leaving the view
        # where it was would hide the thing you just set.
        self._recentre_window()
        self.cutoff_moved.emit(self._cutoff)
        self.cutoff_released.emit(self._cutoff)

    def _can_cut(self) -> bool:
        if self._marked is None or not (0 <= self._marked < len(self._all)):
            return False
        return self.editable() and self._marked_score() != self._cutoff

    def _marked_score(self) -> float:
        return float(min(max(self._all[self._marked].score or 0.0, 0.0), 1.0))

    def _sync_cut(self) -> None:
        """The label is the readback: what will be kept, and from which clip.

        A cutoff move reprojects, so the one gesture that sets it from across the row says what
        it is about to do while it is still a button and not yet a decision.
        """
        can = self._can_cut()
        self._cut.setEnabled(can)
        if can:
            score = self._marked_score()
            kept = len(self._all) - self._boundary_for(score)
            self._cut.setText(
                f"Cut here — keep from {score:.{self._digits}f} ({kept} kept)"
            )
            self._cut.setToolTip("Move the cutoff to the clip in the player")
            return
        self._cut.setText(_CUT_IDLE)
        if not self.editable():
            reason = "Fixed by this snapshot. Go back to the live run to change it."
        elif self._marked is None:
            reason = "Click a clip to choose where the cutoff goes"
        else:
            reason = "The cutoff is already at this clip"
        self._cut.setToolTip(reason)

    # -- the line ------------------------------------------------------------------------ #

    def drop_cutoff_at(self, x: int, commit: bool) -> None:
        """Put the cutoff where the handle was dragged to, snapped to a cell boundary.

        Snapped, not free: the cutoff only means anything at a value that separates two clips,
        and a pixel-precise number between them keeps the same set while reading as a decision
        someone made. Same reason the steppers move by one clip.
        """
        if not self._all:
            return
        ordered = sorted(self._cells.items())
        after = [i for i, cell in ordered if cell.x() + cell.width() // 2 < x]
        index = (after[-1] + 1) if after else (ordered[0][0] if ordered else 0)
        index = max(0, min(index, len(self._all) - 1))
        self.move_cutoff(float(self._all[index].score or 0.0), commit=commit)

    def showEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """Becoming visible is what asks for the pictures — the same rule as the cluster strip.

        Unfolding the *Clips* band is not a resize and not a scroll, so without this the cells a
        fold was hiding would wait for the reader to nudge one of the two.
        """
        super().showEvent(event)
        self._decode_visible()

    def _on_scrolled(self, _value: int) -> None:
        """Scrolling only decodes. Nothing loads, nothing shifts, nothing to fight."""
        if not self._scrolling_itself:
            self._user_scrolled = True
        self._sync_recentre()
        self._decode_visible()

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """Follow the cutoff through a resize, until the reader takes the view somewhere else.

        Opening a run is a resize. The search results arrive while the page is still being laid
        out, so the strip centres against whatever width it has at that moment and nothing
        re-centres when the real one arrives — measured on a 2000-clip ranking, the divider ended
        up **497px off**, about two and a half cells, which is exactly far enough to be looking
        at clips that have not decoded yet with the cutoff off screen. That is the "I have to
        press *centre on cutoff* myself" report.

        Guarded by `_user_scrolled` rather than done unconditionally: once the reader has
        scrolled off to look at something, a window resize must not drag them back to the line.
        """
        super().resizeEvent(event)
        if self._all and not self._user_scrolled:
            self._centre_handle()

    def _sync_recentre(self) -> None:
        """Enabled whenever the row can scroll at all.

        The first version asked "is the divider *visible*" and disabled itself when it was —
        but visible is not centred, so resizing the window left the line off-centre with the
        one control that would fix it greyed out. Any sharper predicate only relitigates how
        far off-centre counts, and pressing this while the line is already centred costs
        nothing: it is a scroll-to-here.
        """
        can_scroll = (
            bool(self._all) and self._scroll.horizontalScrollBar().maximum() > 0
        )
        self._recentre.setEnabled(can_scroll)
        self._recentre.setToolTip(
            "Scroll the cutoff back to the middle"
            if can_scroll
            else "The whole neighbourhood is already on screen"
        )

    def move_cutoff(self, value: float, *, commit: bool) -> None:
        """Set the line's confidence. ``commit`` is the release, not the drag."""
        self._cutoff = float(min(max(value, 0.0), 1.0))
        self._rebuild()
        self.cutoff_moved.emit(self._cutoff)
        if commit:
            self.cutoff_released.emit(self._cutoff)

    def set_editable(self, editable: bool) -> None:
        """Whether the cutoff line in this strip may be moved.

        A flag on the strip rather than ``setEnabled`` on the buttons, because the line is rebuilt
        every refresh, so a per-instance disable would not survive the next one. ``_CutoffHandle``
        reads this when it builds, and so does the cut button through ``_sync_cut``.
        """
        self._editable = editable
        self._rebuild()

    def editable(self) -> bool:
        return getattr(self, "_editable", True)

    def step_cutoff(self, *, down: bool) -> None:
        """Move the line past one clip — the finest step the evidence actually has.

        ``down`` loosens (keep one more, the divider moves left); the default tightens.
        """
        if not self.editable():
            return
        boundary = self._boundary_index()
        target = boundary - 1 if down else boundary + 1
        if 0 <= target < len(self._all):
            self.move_cutoff(float(self._all[target].score or 0.0), commit=True)


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    return label


class _CutoffHandle(QFrame):
    """The cutoff, as an object standing between the clips it separates.

    Three gestures share this row — scroll it, play a clip, move the line — and they are kept
    apart by *where* they are rather than by a mode. Dragging the background scrolls; clicking a
    cell plays it; only this widget moves the cutoff. That is why the grip is drawn: a surface
    that is draggable and does not look it gets dragged by accident or not at all.

    The steppers are not a lesser duplicate of the drag. They are the precise path — one clip,
    which is the finest step the evidence actually has — and the one that still works when the
    handle has been scrolled out of view.
    """

    #: Wide enough for the value, two steppers side by side, and "218 dropped / 124 kept" on
    #: their own lines. The old 26px grip clipped all three.
    BAND_PX = 78
    GRIP_PX = 30

    def __init__(
        self, cutoff: float, boundary: int, total: int, strip: "ConfidenceStrip"
    ) -> None:
        super().__init__()
        self._strip = strip
        self._dragging = False
        # A filled band, not a drawn line. `QFrame.VLine` paints down the *middle* of the
        # widget and these labels are its children, so the line went straight through the
        # value and the counts — and clipped them, because a line's widget is not sized for
        # content. A band is one object: nothing inside it can collide with it.
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), palette.color(palette.ColorRole.Mid))
        self.setPalette(palette)
        self.setCursor(Qt.CursorShape.SplitHCursor)
        self.setFixedWidth(self.BAND_PX)

        column = QVBoxLayout(self)
        column.setContentsMargins(2, 2, 2, 2)
        column.addStretch()
        column.addWidget(_heading(f"{cutoff:.4f}"))
        steps = QHBoxLayout()
        looser = QPushButton("◀")
        # Named, because the glyph is not unique to this control: the clip stepper wears the
        # same one, and a walk that keys on the label cannot tell a control that *reads* a
        # snapshot from one that changes it (`tests/gui/test_cluster_workspace.py`).
        looser.setObjectName("cutoff_looser")
        looser.setToolTip("Keep one more — move the cutoff toward less confident")
        looser.clicked.connect(lambda: strip.step_cutoff(down=True))
        tighter = QPushButton("▶")
        tighter.setObjectName("cutoff_tighter")
        tighter.setToolTip("Keep one fewer — move the cutoff toward more confident")
        tighter.clicked.connect(lambda: strip.step_cutoff(down=False))
        # The **third** and **fourth** ways to move a cutoff, after the curve's line and the
        # number box. A read-only surface that froze only the first two left the stepper live,
        # and stepping it showed a cutoff belonging to neither the snapshot nor the run; this
        # band adds a drag, which is the same hole one gesture over.
        for button in (looser, tighter):
            button.setFixedWidth(self.GRIP_PX)
            button.setEnabled(strip.editable())
            if not strip.editable():
                button.setToolTip(
                    "Fixed by this snapshot. Go back to the live run to change it."
                )
            steps.addWidget(button)
        if not strip.editable():
            self.setCursor(Qt.CursorShape.ArrowCursor)
        column.addLayout(steps)
        column.addStretch()

    # -- dragging ------------------------------------------------------------------- #
    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if not self._strip.editable():
            return
        self._dragging = True
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if not self._dragging:
            return
        self._strip.drop_cutoff_at(
            self.mapToParent(event.position().toPoint()).x(), commit=False
        )
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if not self._dragging:
            return
        self._dragging = False
        self._strip.drop_cutoff_at(
            self.mapToParent(event.position().toPoint()).x(), commit=True
        )
        event.accept()


def _producer(preview, clip, size_px: int = THUMB_PX):
    """The off-thread decode for one clip — the strip's copy of the grid's producer."""
    recording_id = clip.recording_id
    embed_run_id = clip.embed_run_id
    track_id = int(clip.track_id)
    middle_t_ns = (int(clip.start_t_ns) + int(clip.end_t_ns)) // 2

    def produce():
        frame = preview.t_ns_to_frame(recording_id, embed_run_id, middle_t_ns)
        # The **whole frame** with the subject's mask blended in, not a crop around it: at
        # this size a crop shows the animal and nothing it is doing it against, and the
        # question here is whether the stretch is the behavior.
        shown = preview.frame_masked(  # off-thread
            recording_id, embed_run_id, track_id, frame
        )
        return None if shown is None else shown[0]

    return produce
