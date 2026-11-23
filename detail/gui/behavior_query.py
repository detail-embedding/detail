"""Clip surfaces shared across the GUI (§16, §20): the example-clip picker and the clip player.

- ``ClipExampleDialog`` — scrub the video and mark example clips (in/out), with the
  subject's mask on the frame when **Masks** is ticked,
  returning their ``t_ns`` spans in ``.examples``.
- ``_ClipPlayer`` / ``_PlayerColumn`` / ``ClipSpec`` — the player most clip views embed.

The file is named for the Behavior query module (``clip_query``) whose dialogs it first held;
that module and its matches/compare/report dialogs were removed
(``docs/removed_capabilities.md``). Everything reads through ``ClipQueryPreview`` (Tenet 2);
the pure matcher lives in ``analysis/``.
"""

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg

from detail.analysis import clip_query
from detail.api.run_preview import DEFAULT_PREVIEW_FPS, ClipQueryPreview
from detail.core.chart_theme import LIGHT, rgb
from detail.gui.dialog import Dialog
from detail.gui.qt import (
    QAbstractItemView,
    QBrush,
    QButtonGroup,
    QCheckBox,
    QColor,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QEvent,
    QFont,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QKeySequence,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPainter,
    QPalette,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QRect,
    QShortcut,
    QSizePolicy,
    QStandardItem,
    QStandardItemModel,
    QTabBar,
    QTableView,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)
from detail.gui.spin import SECONDS, FloatSpin, IntSpin
from detail.gui import example_spread
from detail.gui import frame_source
from detail.gui.busy import decode_overlay
from detail.gui import plot_theme
from detail.gui.plots import bound_axes
from detail.gui.run_preview import _to_pixmap
from detail.gui.grip_splitter import GripSplitter
from detail.gui.image_view import ImageView, PictureWithControls
from detail.gui.scrubber import FrameScrubber
from detail.gui.sizing import (
    add_fitted_picture,
    bound_picture,
    emphasis_font,
    set_available,
    text_width,
)
from detail.gui.status_color import READY, set_button_color
from detail.gui.tables import (
    num_cell,
    enable_sorting,
    fill_width,
    fit_table_height,
    make_columns_resizable,
    row_id,
    stamp_row_id,
)
from detail.gui.workers import BackgroundCall
from detail.gui.sizing import mono_font
from detail.gui.timefmt import format_time

# The kept span's fill on a coverage strip — Okabe-Ito "bluish green", the same colour the
# review tab's confirm button uses, so green means the same thing on both.
_SPAN_RGB = rgb(LIGHT.hue("green"))

# How much of its weight the coverage strip keeps outside the kept span. Low enough that
# the span is obviously the subject, high enough that a tracking gap out there is still
# visible — reading the gaps around a clip is why the strip is drawn at all.
_OUTSIDE_SPAN_ALPHA = 0.3


class _CoverageBar(QWidget):
    """A strip under the scrubber shading how much of each pixel column's frames carry an
    embedding — where the subject was actually detected and tracked.

    Shading by *density* rather than drawing embedded/not-embedded is what makes it honest
    at this scale: an hour-long recording puts hundreds of frames in every pixel, so a
    binary strip would hide every short dropout, and giving each break a one-pixel minimum
    would paint a 78%-tracked recording as mostly gap. A column that is 92% embedded is
    drawn 92% dark.

    With a start marked, the containing continuous stretch is bracketed **where the searching
    method requires one** — the bracket says how far the end may be dragged, so a surface
    whose method pools exemplar frames rather than matching a path draws none rather than
    claiming a limit it does not impose.
    """

    _HEIGHT = 14
    _BASE_TOOLTIP = "Where this track has embeddings. Darker = more frames detected."

    # Emitted while an edge is being dragged, and once more on release. The receiver
    # persists on ``span_committed`` only — a write per mouse-move would append a label row
    # per pixel.
    span_dragged = Signal(int, int)
    span_committed = Signal(int, int)

    _CAP_W = 3  # width of the drawn grab handle at each end of the span
    _GRAB_PX = (
        9.0  # how close a press must be to an edge to take it (arena's hit radius)
    )

    def __init__(self, scrubber, parent=None) -> None:
        super().__init__(parent)
        self._scrubber = scrubber
        self._frame_idx = None
        self._runs: list[tuple[int, int]] = []
        self._n_frames = 0
        self._bracket: tuple[int, int] | None = None
        self._bins: tuple[int, np.ndarray] | None = None  # (width, density) cache
        self._draggable = False  # only the trim surface takes edits
        self._drag_edge: int | None = None  # 0 = start, 1 = end
        self.setFixedHeight(self._HEIGHT)
        # What the bar *draws*, and nothing about what may be marked on it: whether a clip
        # has to sit inside one stretch is the searching method's rule, so an embedder that
        # knows it adds the sentence with `set_rule_note`. Asserting it here made a shared
        # widget promise a limit that is not true of every surface embedding it.
        self.setToolTip(self._BASE_TOOLTIP)

    def set_rule_note(self, note: str | None) -> None:
        """Append the embedder's rule for what may be marked here, or clear it."""
        self.setToolTip(
            self._BASE_TOOLTIP if not note else f"{self._BASE_TOOLTIP} {note}"
        )

    def set_continuity(self, frame_idx, runs, n_frames: int) -> None:
        self._frame_idx = frame_idx
        self._runs = list(runs)
        self._n_frames = int(n_frames)
        self._bins = None
        self._bracket = None
        self.update()

    def has_gaps(self) -> bool:
        """Whether there is a break worth drawing: more than one continuous stretch, or one that
        leaves more than 1% of the video untracked. A fully tracked recording draws one solid
        block that says nothing, so a host that only *reports* coverage hides the strip then.
        """
        if self._n_frames <= 0 or not self._runs:
            return False
        if len(self._runs) > 1:
            return True
        first, last = self._runs[0]
        return (last - first + 1) < 0.99 * self._n_frames

    def clear(self) -> None:
        self.set_continuity(None, [], 0)

    def set_bracket(self, span: tuple[int, int] | None) -> None:
        """Outline one continuous stretch (inclusive global frames), or nothing."""
        self._bracket = span
        self.update()

    def set_draggable(self, on: bool) -> None:
        """Let the bracket's edges be dragged (the annotation trim surface).

        Off everywhere else: the picker's bracket marks the stretch a pending clip must
        stay inside, which is not a thing the user may move.
        """
        self._draggable = on
        self._drag_edge = None
        self.setCursor(
            Qt.CursorShape.SizeHorCursor if on else Qt.CursorShape.ArrowCursor
        )

    def _muted_range(self, left: int, span: int) -> tuple[int, int] | None:
        """Widget x-range of the kept span, or ``None`` when nothing should be muted."""
        if self._bracket is None or not self._draggable:
            return None
        first, last = self._bracket
        scale = (span - 1) / max(self._n_frames - 1, 1)
        return left + round(first * scale), left + round(last * scale)

    def _frame_at(self, x: float) -> int:
        """The frame under a widget x — the inverse of the bracket mapping in paintEvent."""
        left, span = self._scrubber.handle_travel()
        span = max(min(span, self.width() - left), 1)
        frac = (x - left) / max(span - 1, 1)
        frame = round(frac * max(self._n_frames - 1, 1))
        return int(min(max(frame, 0), max(self._n_frames - 1, 0)))

    def _edge_x(self, frame: int) -> float:
        left, span = self._scrubber.handle_travel()
        span = max(min(span, self.width() - left), 1)
        return left + frame / max(self._n_frames - 1, 1) * (span - 1)

    def _edge_at(self, x: float) -> int | None:
        if self._bracket is None:
            return None
        distances = [abs(x - self._edge_x(f)) for f in self._bracket]
        nearest = 0 if distances[0] <= distances[1] else 1
        return nearest if distances[nearest] <= self._GRAB_PX else None

    def mousePressEvent(self, event) -> None:
        if not self._draggable or self._n_frames <= 0:
            return
        self._drag_edge = self._edge_at(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if self._drag_edge is None or self._bracket is None:
            return
        self._bracket = self._moved(event.position().x())
        self.update()
        self.span_dragged.emit(*self._bracket)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_edge is None or self._bracket is None:
            return
        self._bracket = self._moved(event.position().x())
        self._drag_edge = None
        self.update()
        self.span_committed.emit(*self._bracket)

    def _moved(self, x: float) -> tuple[int, int]:
        """The bracket with the dragged edge at ``x``, keeping start <= end.

        Edges are not allowed to cross: dragging one past the other pins it to its
        neighbour rather than silently swapping which edge is being held.
        """
        first, last = self._bracket
        frame = self._frame_at(x)
        if self._drag_edge == 0:
            return (min(frame, last), last)
        return (first, max(frame, first))

    def run_containing(self, frame: int) -> tuple[int, int] | None:
        """The continuous stretch holding ``frame``, if any — what a pending start pins."""
        for first, last in self._runs:
            if first <= frame <= last:
                return (first, last)
        return None

    def _density(self, width: int) -> np.ndarray:
        """Fraction of each pixel column's frames that carry an embedding."""
        if self._bins is not None and self._bins[0] == width:
            return self._bins[1]
        n_frames = max(self._n_frames, 1)
        if width >= n_frames:
            # Zoomed in past one frame per pixel (the trim window). A histogram here puts
            # 0 or 1 frame in each bin, so a fully-tracked stretch renders as a barcode of
            # alternating dark and blank columns — which reads as a run of tracking gaps,
            # the exact thing this strip exists to show. Sample the coverage instead.
            covered = np.zeros(n_frames, dtype=bool)
            inside = self._frame_idx[
                (self._frame_idx >= 0) & (self._frame_idx < n_frames)
            ]
            covered[inside.astype(np.int64)] = True
            density = covered[np.arange(width) * n_frames // width].astype(float)
        else:
            counts, _ = np.histogram(self._frame_idx, bins=width, range=(0, n_frames))
            density = np.clip(counts / (n_frames / width), 0.0, 1.0)
        self._bins = (width, density)
        return density

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._bins = None  # re-bin at the new width

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        palette = self.palette()
        base = palette.color(QPalette.ColorRole.Base)
        left, span = self._scrubber.handle_travel()
        span = max(min(span, self.width() - left), 1)
        # The strip only under the slider's travel, with its name in the room to its left — the
        # column the elapsed time occupies above it. Unlabelled, it read as "a solid bar I don't
        # understand"; the tooltip alone was not enough.
        painter.fillRect(left, 0, span, self.height(), base)
        label = "tracked"
        if left >= self.fontMetrics().horizontalAdvance(label) + 6:
            painter.setPen(palette.color(QPalette.ColorRole.WindowText))
            painter.drawText(
                QRect(0, 0, left - 6, self.height()),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                label,
            )
        if self._frame_idx is None or self._n_frames <= 0:
            return
        ink = palette.color(QPalette.ColorRole.Text)
        density = self._density(span)
        top, height = 3, self._HEIGHT - 6
        # Columns outside the kept span are drawn back towards the strip's own background,
        # so the selection reads as the subject and its surroundings as context. Without
        # it the whole strip is one weight of ink and the eye has to hunt for the span.
        muted = self._muted_range(left, span)
        for i, value in enumerate(density.tolist()):
            if value <= 0.0:
                continue
            shade = QColor(ink)
            alpha = 0.12 + 0.88 * value  # never fully invisible where rows exist
            if muted is not None and not (muted[0] <= left + i <= muted[1]):
                alpha *= _OUTSIDE_SPAN_ALPHA
            shade.setAlphaF(alpha)
            painter.fillRect(left + i, top, 1, height, shade)
        if self._bracket is not None:
            first, last = self._bracket
            x0 = left + round(first / max(self._n_frames - 1, 1) * (span - 1))
            x1 = left + round(last / max(self._n_frames - 1, 1) * (span - 1))
            width = max(x1 - x0, 1)
            if self._draggable:
                # A **filled** span, not an outline: on a strip that is mostly dark ink
                # (dense coverage) a thin rectangle disappears into it, and the one thing
                # the annotator needs to see at a glance is where the clip currently
                # starts and ends. Okabe-Ito green, the same "confirmed" green the review
                # buttons use.
                fill = QColor(*_SPAN_RGB)
                fill.setAlphaF(0.55)
                painter.fillRect(x0, 0, width, self._HEIGHT - 1, fill)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(*_SPAN_RGB)))
                for x in (x0, x1):  # solid grab handles at each end
                    painter.drawRect(
                        round(x - self._CAP_W / 2), 0, self._CAP_W, self._HEIGHT - 1
                    )
            else:
                painter.setPen(palette.color(QPalette.ColorRole.Highlight))
                painter.drawRect(x0, 0, width, self._HEIGHT - 1)


@dataclass(frozen=True)
class ClipSpec:
    """One clip to render in the compare view — its subject, time span, and a label."""

    recording_id: str
    track_id: int
    embed_run_id: str
    start_t_ns: int
    end_t_ns: int
    label: str
    score: float | None = None
    rank: int | None = None


#: The context a clip player offers either side of its clip, in seconds; ``0`` is off. A short,
#: fixed menu rather than a free number: the question is "a beat either side" or "what led up to
#: it", and four choices answer both without a field to type into mid-viewing.
CONTEXT_CHOICES_S = (0.0, 1.0, 2.0, 5.0)

#: The clip is bracketed on the bar only while it covers less than this share of it. Past that the
#: context is a sliver either side and the brackets would sit on the bar's own ends.
_SPAN_SHOWN_BELOW = 0.9


def _interval_ms(fps: float) -> int:
    """Timer interval for real-time playback at ``fps``, floored so it stays a live timer."""
    return max(int(round(1000.0 / fps)), 8) if fps > 0 else 33


# Holding the scrubber at the window's edge walks it along. It starts at video speed so a
# short hold is precise, then ramps — long enough to reach the next behavior without the
# tearing that panning per mouse-move caused. Same shape as the scrubber's key
# acceleration (``scrubber.accelerated_step``), but linear and capped low: this moves the
# *view*, and losing your place is worse than arriving slowly.
_EDGE_MAX_STEP = 3  # frames per tick, i.e. up to 3x video speed
_EDGE_RAMP_TICKS = 4  # ticks held before each step up (~0.5 s each at 120 ms)


def edge_walk_step(ticks_held: int) -> int:
    """Frames to slide on a tick, given how many ticks the edge has been held."""
    return min(1 + max(ticks_held, 0) // _EDGE_RAMP_TICKS, _EDGE_MAX_STEP)


class _ClipPlayer(QWidget):
    """Plays one clip with its own scrub bar + Play. Options: ``masked`` (the subject's mask
    blended into the frame, vs. plain footage), ``autoplay`` (start on each ``set_clip``),
    ``loop`` (replay instead of stopping at the end), and ``top`` (pin the video to the top of
    its pane rather than centring it). Clicking the video toggles play.

    **One pane, not two.** This used to show the source frame beside the *input image* — the
    constructor's rendering of what the embedder saw — which answers a question about the
    method rather than about the animal, and halved the space the footage had. Every query
    surface now shows the same thing: the video, with the subject marked. The blend is
    ``ClipQueryPreview.frame_masked``, the framing sibling of the ``source_crop_masked`` the
    cluster strip uses, so there is one implementation of "the subject, marked" rather than a
    renderer per surface.

    Owns its own ``ClipQueryPreview`` so its off-thread decode never shares mutable cache
    state with the other column's player."""

    # The window the scrubber spans moved or changed size (a pan, a context change), so anything
    # drawn against it must be repainted.
    window_changed = Signal()

    def __init__(
        self,
        experiment,
        parent=None,
        *,
        masked: bool = True,
        autoplay: bool = False,
        loop: bool = False,
        context_s: float,
        walk_edges: bool = False,
        top: bool = False,
    ) -> None:
        """``context_s`` is how much footage either side of the clip the player opens with, one
        of ``CONTEXT_CHOICES_S``. **Required**: each page decides it, because how much context a
        clip needs is a fact about what the page is for — telling two similar clusters apart
        needs a second either side, a list of matches being judged on their own does not. The
        viewer can change it from the player's own row; that lasts as long as the player.

        ``walk_edges`` lets the window slide when the playhead is held at either end, for the
        page that drags a clip's boundaries outward (the removed ``clip_annotation`` was that page;
        no page passes it today). Playback there stays on the clip; everywhere else it runs
        through the context.

        Raises:
            ValueError: ``context_s`` is not one of ``CONTEXT_CHOICES_S``.
        """
        super().__init__(parent)
        # Kept, not just read for its dir: the export below submits a run through it. Every one
        # of the thirteen surfaces that builds a player already passes one, so holding on to it
        # costs no call site anything.
        self._experiment = experiment
        self._preview = ClipQueryPreview(experiment.dir)
        self._clip: ClipSpec | None = None
        self._bounds = (0, 0)  # inclusive (start_frame, end_frame)
        self._len = 0
        self._masked = masked
        self._autoplay = autoplay
        self._loop = loop
        #: Whether `set_clip` is still waiting for a first frame before it starts playing.
        self._start_on_first_frame = False
        # With context the scrubber spans a *window* around the clip rather than the whole
        # recording: over a 29-minute part the full timeline puts ~2.8 s behind every pixel,
        # which makes a one-second clip a third of a pixel wide. The clip is drawn on the bar as
        # a span, so the window's extent is never a guess about which part is the behavior.
        # Frames still decode one at a time on demand, so nothing is preloaded.
        if float(context_s) not in CONTEXT_CHOICES_S:
            raise ValueError(
                f"context_s={context_s!r} is not one of the offered amounts "
                f"{CONTEXT_CHOICES_S}"
            )
        self._context_s = float(context_s)
        self._walk_edges = bool(walk_edges)
        self._clip_span: tuple[int, int] = (0, 0)  # the clip's own frames
        self._total = 0  # frames in the part, so panning knows where to stop
        # Playback belongs to the scrubber now. What this class used to add — looping, and
        # playing the *clip* rather than the padded window the scrubber spans — moved there as
        # `loop=` and `set_playback_range`, so twelve surfaces have them instead of one.
        # Holding the scrubber past the window's edge walks the window at the same rate,
        # so scrubbing beyond the clip moves at the speed the video plays.
        self._edge_pan = QTimer(self)
        self._edge_pan.setInterval(120)
        self._edge_pan.timeout.connect(self._pan_one_frame)
        self._edge_forward = True
        self._edge_ticks = 0  # how long the edge has been held, for the ramp

        self._src = ImageView(zoomable=True)
        self._src.clicked.connect(self._play_btn_toggle)
        self._scrubber = FrameScrubber(playable=True, loop=loop)
        self._scrubber.set_decoder(self._compute)
        self._scrubber.rendered.connect(self._display)
        # A playing clip is a stream, not a request: `busy_changed` stays True across the
        # re-dispatch burst, and `decode_overlay` asks the scrubber rather than being told, so
        # the scrim cannot latch over frames that are arriving.
        self._decode_busy = decode_overlay(self._scrubber, self._src)
        self._scrubber.scrubbed.connect(
            self._pause_for_manual
        )  # manual move stops autoplay
        self._scrubber.frame_changed.connect(self._extend_at_edge)
        # Every player that can reach a mask offers the choice. Eleven of the thirteen surfaces
        # built on this class took `masked=True` and had no way to turn it off — so they paid
        # 10.39 ms a frame against 4.64 whether anybody was looking at the outline or not, and
        # playback was slower everywhere for a marking you only study when you stop.
        #
        # The constructor argument is now where the toggle *starts*, not what it is fixed at.
        self._masks = QCheckBox("Masks")
        self._masks.setChecked(self._masked)
        self._masks.setToolTip(
            "Tint the subject's mask onto the frame. About twice the cost a frame, so playback "
            "is slower with it on."
        )
        self._masks.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._masks.toggled.connect(self._on_masks_toggled)
        # Context lives on the player, not on a page: every page with a clip has the question
        # "what happened just before", and the answer belongs where the clip is being watched.
        self._context_label = QLabel("Context:")
        self._context = QComboBox()
        self._context.setObjectName("clip_context")
        self._context.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for seconds in CONTEXT_CHOICES_S:
            self._context.addItem("off" if not seconds else f"{seconds:g} s", seconds)
        self._context.setCurrentIndex(CONTEXT_CHOICES_S.index(self._context_s))
        self._context.setToolTip(
            "Footage either side of the clip. The clip itself is the green band on the bar."
        )
        self._context.currentIndexChanged.connect(self._on_context_changed)
        # One unit with the mask toggle: both choose how this clip is shown, and as one unit the
        # flowing row neither wraps between "Context:" and its menu nor spends a line on each
        # in a narrow pane — where every line is height the picture above gives up.
        view_box = QWidget()
        view_row = QHBoxLayout(view_box)
        view_row.setContentsMargins(0, 0, 0, 0)
        view_row.addWidget(self._context_label)
        view_row.addWidget(self._context)
        view_row.addSpacing(6)
        view_row.addWidget(self._masks)
        self._add_view_controls(view_row)
        self._scrubber.add_control(view_box)
        # The same row, for the same reason: a player is where you decide a clip is worth
        # keeping, so it is where taking it away belongs. One control here reaches all thirteen
        # surfaces — the leverage a shared widget is meant to give. It opens the *shared* export
        # dialog (`gui/export.py::open_export`), pre-answered; it does not write anything itself,
        # which is the distinction the one-control rule actually protects.
        self._export = QPushButton("Export video…")
        self._export.setToolTip(
            "Write the frames on screen to an mp4, in the flavour shown — masks on draws the "
            "outline, masks off writes the plain footage."
        )
        self._export.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._export.setEnabled(False)  # nothing to export until a clip is set
        self._export.clicked.connect(self._export_clip)
        self._scrubber.add_control(self._export)
        # Picture directly above its controls, centred in whatever pane this player is handed
        # (#396) — not the picture claiming the whole pane and the controls sinking to wherever
        # it happens to end.
        # ``top`` pins the pair to the top instead, for a player beside a list (Browse).
        self._block = PictureWithControls(
            self._picture_widget(), self._scrubber, top=top
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._block)

    def height_for_width(self, width: int) -> int:
        """How tall this player wants to be at ``width`` — see
        :meth:`PictureWithControls.height_for_width`. Forwarded rather than inherited: the
        block is this widget's only child and its margins are zero, so the two are the same
        number, and a host embedding the player should not have to reach past it."""
        return self._block.height_for_width(width)

    # -- the seams a richer player overrides (``gui/advanced_player.py``) ---------------- #
    # Five hooks with this player's behaviour as the default, so the eleven surfaces that embed
    # the plain player are untouched by a window that shows more: what sits where the picture
    # goes, what joins the view row, what one frame decodes to, how a payload is shown, and
    # whether playback runs the clip alone.

    def _picture_widget(self) -> QWidget:
        """The widget the controls sit under: the footage alone."""
        return self._src

    def _add_view_controls(self, row) -> None:
        """More ticks after **Masks** on the view row; the plain player adds none."""

    def _decode(self, clip: ClipSpec, frame: int):
        """One frame's payload, computed **off the UI thread** (the scrubber's decoder). A
        frame that cannot be produced raises; `_compute` classifies that into a reason
        (`frame_source.decode`), so an override returns the payload and nothing else."""
        if not self._masks.isChecked():
            # `frame_plain`, not `frame_view`: this pane shows one picture, and `frame_view`
            # decoded the masks and rendered an input image beside it that was then thrown away
            # — 14.31 ms a frame for a 4.64 ms job.
            return self._preview.frame_plain(  # off-thread (scrubber decoder)
                clip.recording_id, clip.embed_run_id, clip.track_id, frame
            )
        return self._preview.frame_masked(  # off-thread (scrubber decoder)
            clip.recording_id, clip.embed_run_id, clip.track_id, frame
        )

    def _show_payload(self, payload) -> None:
        """Put a decoded payload on screen. ``frame_masked`` and ``frame_plain`` both return
        ``(rgb, actual, t_ns)``; the first element is the picture either way."""
        self._src.set_frame(_to_pixmap(payload[0]))
        self._block.set_aspect(self._src.pixmap_size())

    def _show_unavailable(self, text: str) -> None:
        """Say where the picture goes why there is none."""
        self._src.clear_frame(text)

    def _clip_only(self) -> bool:
        """Whether playback runs the clip alone rather than the whole window (context too)."""
        return self._walk_edges

    def _redraw(self, *_args) -> None:
        """Draw the frame on screen again with the view ticks as they now are. Without this a
        tick does nothing until you move, which reads as a broken control."""
        self._scrubber.set_frame(self._scrubber.frame())

    def set_clip(self, clip: ClipSpec, *, autoplay: bool | None = None) -> None:
        """Show ``clip``. ``autoplay`` overrides the player's own setting for this clip only —
        a list that opens on its first row shows it still, and a click still plays."""
        self._scrubber.pause()
        self._edge_pan.stop()
        self._clip = clip
        self._export.setEnabled(True)
        self._scrubber.set_playback_rate(self._frame_rate(clip))
        self._bounds = self._resolve(clip)
        self._len = max(self._bounds[1] - self._bounds[0] + 1, 1)
        self._scrubber.set_total(self._len)
        self._apply_playback_range()
        # Where playing starts. Context is there to be watched on the way into the clip, so the
        # playhead opens at the window's left edge; the boundary-dragging page opens on the clip,
        # since its playback never leaves it.
        self._scrubber.set_frame(
            self._clip_span[0] - self._bounds[0] if self._walk_edges else 0
        )
        # **Deferred until a frame exists.** Starting the timer here started it over nothing:
        # the decode is off-thread, so the counter advanced and the scrub bar ran while the pane
        # was still empty — and because the "Decoding…" overlay is suppressed *while playing*,
        # the one indicator that would have said why was switched off by the same act. Playing
        # begins when there is something to play; until then the overlay does its job.
        self._start_on_first_frame = self._autoplay if autoplay is None else autoplay

    def _frame_rate(self, clip: ClipSpec) -> float:
        try:
            return self._preview.frame_rate(clip.recording_id, clip.embed_run_id)
        except (
            Exception
        ):  # noqa: BLE001 — an unreadable clip still plays, just at the default
            return DEFAULT_PREVIEW_FPS

    def bounds(self) -> tuple[int, int]:
        return self._bounds

    def play(self) -> None:
        """Start playing from the current position (the annotation review autoplay)."""
        self._scrubber.play()

    def pause(self) -> None:
        """Stop playing, leaving the frame where it is."""
        self._scrubber.pause()

    def is_playing(self) -> bool:
        """Whether the clip is running. Ask this rather than the button behind it."""
        return self._scrubber.is_playing()

    def play_button(self):
        """The control that starts and stops it — the scrubber's, since this owns none."""
        return self._scrubber.play_button()

    def _apply_playback_range(self) -> None:
        """Re-state everything the scrubber knows about the current window.

        Where playing runs, the time of every scrub position, and where the clip sits inside the
        window. Called whenever the window moves, because ``set_total`` forgets all three and a
        pan slides the window without calling it.
        """
        first, last = self._playback_range()
        self._scrubber.set_playback_range(first, last)
        self._scrubber.set_timestamps(self._window_timestamps())
        lo = self._clip_span[0] - self._bounds[0]
        hi = self._clip_span[1] - self._bounds[0]
        if self._clip is not None and (hi - lo + 1) < _SPAN_SHOWN_BELOW * self._len:
            self._scrubber.set_marks(spans=[(lo, hi)])
        else:
            # Bracketing nearly the whole bar marks nothing: a cluster member that is one
            # unbroken run of a 30-minute video filled the bar edge to edge with "the clip".
            self._scrubber.clear_marks()

    def _window_timestamps(self):
        """The recording's ``t_ns`` over the window, or ``None`` when it cannot be read."""
        if self._clip is None:
            return None
        try:
            t_ns = self._preview.timestamps(
                self._clip.recording_id, self._clip.embed_run_id
            )
        except Exception:  # noqa: BLE001 — no time is shown, never a guessed one
            return None
        window = t_ns[self._bounds[0] : self._bounds[1] + 1]
        return window if len(window) == self._len else None

    def context_s(self) -> float:
        """Seconds of context either side of the clip, as currently chosen."""
        return self._context_s

    def set_context_s(self, seconds: float) -> None:
        """Choose the context, as the menu would."""
        self._context.setCurrentIndex(CONTEXT_CHOICES_S.index(float(seconds)))

    def _on_context_changed(self, _index: int) -> None:
        seconds = self._context.currentData()
        if seconds is None:
            return
        self._context_s = float(seconds)
        if self._clip is None:
            return
        # Keep the frame being looked at, if the new window still holds it — changing how much
        # surrounds the clip is not a request to start again.
        here = self._bounds[0] + self._scrubber.frame()
        was_playing = self.is_playing()
        self._edge_pan.stop()
        self._bounds = self._resolve(self._clip)
        self._len = max(self._bounds[1] - self._bounds[0] + 1, 1)
        self._scrubber.set_total(self._len)
        self._apply_playback_range()
        inside = self._bounds[0] <= here <= self._bounds[1]
        self._scrubber.set_frame(here - self._bounds[0] if inside else 0)
        self.window_changed.emit()
        if was_playing:
            self.play()

    def export_button(self) -> QPushButton:
        """The row's **Export video…** — for a host with an export of its own to hide."""
        return self._export

    def current_frame(self) -> int:
        """The global frame under the playhead."""
        return int(self._bounds[0] + self._scrubber.frame())

    def focus_scrubber(self) -> None:
        """Give the scrubber keyboard focus so arrow keys scrub this player."""
        self._scrubber.setFocus()

    def _on_masks_toggled(self, on: bool) -> None:
        """Redraw the frame already on screen through the other path.

        Without this the tick does nothing until you move, which reads as a broken control.
        """
        self._masked = bool(on)
        self._redraw()

    def masks_shown(self) -> bool:
        """Whether the subject's mask is being drawn — ask this, not the checkbox behind it."""
        return self._masks.isChecked()

    def export_scope(self):
        """The clip on screen, as an `ExportScope` — what the shell's `Export…` would narrow to.

        Implemented so the top-bar control and the button below agree about what "this clip"
        means; `gui/export.py::scope_of` searches a page's children, so any surface holding a
        player answers with the clip it is showing without forwarding anything.
        """
        from detail.api.exporters import ExportScope

        if self._clip is None:
            return None
        return ExportScope(frames=self.clip_span())

    def _export_clip(self) -> None:
        """Open the shared export dialog on this clip, pre-answered.

        Everything the export needs that no control can ask for — which recording, which
        subject, which embedding run resolves its masks — the player already knows. The flavour
        follows the picture: masks on writes the overlay, masks off writes the raw footage, so
        the file matches what was on screen when the button was pressed.
        """
        from detail.api.exporters import ExportSubject
        from detail.gui.export import open_export

        clip = self._clip
        if clip is None:
            return
        from detail.api import clip_render

        renders = [clip_render.OVERLAY if self.masks_shown() else clip_render.RAW]
        open_export(
            self,
            ExportSubject("recording", self._experiment.dir, clip.recording_id),
            self.export_scope(),
            only=("clip",),
            seed={
                "track_id": int(clip.track_id),
                "embed_run_id": clip.embed_run_id,
                # Lists, not tuples: this rides an `ExportJob` through YAML, where a tuple
                # writes as `!!python/tuple` and `safe_load` then refuses it.
                "renders": renders,
                "crop": False,
            },
        )

    def _play_btn_toggle(self) -> None:
        """Clicking the video is the same gesture as pressing Play — and claims the keyboard.

        Focus, because space is handled by whichever scrubber has it. Two players share a window
        on the annotation page, and the eval page puts a player beside a scrubber of its own, so
        a window-wide space shortcut would be ambiguous between them and Qt would fire neither.
        Clicking the picture you are watching is how you say which one you mean.
        """
        self._scrubber.setFocus()
        button = self._scrubber.play_button()
        if button is not None:
            button.toggle()

    def _pause_for_manual(self) -> None:
        """Stop autoplay when the user scrubs by hand, so they can step frames manually."""
        self._scrubber.pause()

    def _resolve(self, clip: ClipSpec) -> tuple[int, int]:
        try:
            lo = self._preview.t_ns_to_frame(
                clip.recording_id, clip.embed_run_id, clip.start_t_ns
            )
            hi = self._preview.t_ns_to_frame(
                clip.recording_id, clip.embed_run_id, clip.end_t_ns
            )
        except Exception:  # noqa: BLE001 — a bad clip shows an empty pane, not a crash
            return (0, 0)
        self._clip_span = (min(lo, hi), max(lo, hi))
        if not self._context_s and not self._walk_edges:
            return self._clip_span
        try:
            total = self._preview.recording_frame_count(
                clip.recording_id, clip.embed_run_id
            )
            self._total = total
            if not self._context_s:
                return self._clip_span
            pad_ns = int(self._context_s * 1e9)
            # `t_ns_to_frame` clamps to the recording, so a clip near either end of the video
            # gets what context there is rather than an error.
            first = self._preview.t_ns_to_frame(
                clip.recording_id, clip.embed_run_id, clip.start_t_ns - pad_ns
            )
            last = self._preview.t_ns_to_frame(
                clip.recording_id, clip.embed_run_id, clip.end_t_ns + pad_ns
            )
        except Exception:  # noqa: BLE001 — fall back to the clip's own range
            return self._clip_span
        return (max(min(first, last), 0), min(max(first, last), max(total - 1, 0)))

    def pan(self, forward: bool, frames: int | None = None) -> bool:
        """Slide the window **without changing its size**. ``False`` at the part's ends.

        Deliberately not a zoom-out. Growing the window each time it was reached made the
        scale coarser with every step — a few of them and a clip is too small to work with,
        with no way back to a usable one. Sliding keeps the seconds-per-pixel the annotator
        was given, so scrubbing can run as far as they like and the view stays workable.

        ``frames`` is how far to slide; ``None`` is half a window, for a one-off jump.
        """
        first, last = self._bounds
        width = last - first
        if width <= 0 or self._total <= 1:
            return False
        step = max(width // 2, 1) if frames is None else max(int(frames), 1)
        shift = min(step, self._total - 1 - last) if forward else -min(step, first)
        if shift == 0:
            return False
        self._bounds = (first + shift, last + shift)
        self._apply_playback_range()
        return True

    def recenter(self) -> bool:
        """Put the window back around the clip. ``False`` if it is already there."""
        if self._clip is None:
            return False
        before = self._bounds
        self._bounds = self._resolve(self._clip)
        if self._bounds == before:
            return False
        self._len = max(self._bounds[1] - self._bounds[0] + 1, 1)
        self._scrubber.set_total(self._len)
        self._apply_playback_range()
        self._scrubber.set_frame(max(self._clip_span[0] - self._bounds[0], 0))
        return True

    def clip_span(self) -> tuple[int, int]:
        """The clip's own inclusive frame range, whatever the scrubber spans."""
        return self._clip_span

    def scrubber(self) -> FrameScrubber:
        """The scrub bar, for a caller building a strip aligned to its handle travel."""
        return self._scrubber

    def set_underlay(self, widget) -> None:
        """Put a strip directly under the scrub bar (the boundary editor's coverage bar)."""
        self._scrubber.set_underlay(widget)

    def frame_to_t_ns(self, step: int) -> int | None:
        """The ``t_ns`` at a **scrub position**, for turning a dragged edge into a span.

        ``step`` is scrubber-relative (0..len-1); with context that is offset from the
        recording's own frame numbering by the window's first frame.
        """
        if self._clip is None:
            return None
        try:
            return self._preview.frame_t_ns(
                self._clip.recording_id,
                self._clip.embed_run_id,
                int(self._bounds[0] + step),
            )
        except (IndexError, ValueError, KeyError, OSError):
            return None

    def window(self) -> tuple[int, int]:
        """The inclusive recording frames the scrubber currently spans."""
        return self._bounds

    def _compute(self, step: int):
        if self._clip is None:
            return None
        frame = self._bounds[0] + min(step, self._len - 1)
        clip = self._clip
        return frame_source.decode(  # off-thread (scrubber decoder)
            lambda: self._decode(clip, frame)
        )

    def _display(self, _step: int, view) -> None:
        if view is None:
            return
        payload, reason = view
        if reason is not None:
            # An undecodable clip never plays, and must not leave autoplay armed for the next
            # frame that happens to arrive.
            self._start_on_first_frame = False
            self._show_unavailable(frame_source.unavailable_text(reason))
            return
        self._show_payload(payload)
        if self._start_on_first_frame:
            self._start_on_first_frame = False
            self.play()

    def show_message(self, text: str) -> None:
        """Put a message where the video goes, and stop playing.

        For the states that have no clip yet rather than a clip that failed: an empty pane
        reads as broken, and the difference between "nothing to show yet" and "this frame did
        not decode" is exactly what a blank one hides.
        """
        self._clip = None
        self._scrubber.pause()
        self._src.clear_frame(text)

    def _extend_at_edge(self, step: int) -> None:
        """Grow the window when the playhead reaches either end of it.

        Reaching the edge means the user wants to see past it — the behavior runs on
        beyond what the window happens to show. Extending there keeps scrubbing and
        playback continuous instead of stopping at a boundary that is an artifact of how
        much context was loaded, not of the video.
        """
        if not self._walk_edges or self._len <= 1:
            return
        if 0 < step < self._len - 1:
            self._edge_pan.stop()  # back inside the window — nothing to chase
            return
        self._edge_forward = step >= self._len - 1
        if not self._edge_pan.isActive():
            self._edge_ticks = 0  # a fresh grab starts at video speed
            # Start first: the pan below moves the scrubber, which re-enters here, and an
            # inactive timer would make that recurse.
            self._edge_pan.start()
            self._pan_one_frame()  # respond to the first touch immediately

    def _pan_one_frame(self) -> None:
        """Hold at the edge: slide the window one frame, at the rate the video plays.

        Panning straight from the scrub signal ran away: sliding re-bases the playhead,
        the in-progress drag immediately re-pins it to the edge, and that pans again — half
        a window per mouse-move. Driving it from a timer decouples the speed from how fast
        the mouse is moving: a short hold is frame-precise at video speed, and a sustained
        one ramps to ``_EDGE_MAX_STEP``.
        """
        step = self._scrubber.frame()
        if not self._walk_edges or not (step <= 0 or step >= self._len - 1):
            self._edge_pan.stop()
            return
        if not self.pan(
            forward=self._edge_forward, frames=edge_walk_step(self._edge_ticks)
        ):
            self._edge_pan.stop()  # at the part's end
            return
        self._edge_ticks += 1
        # Stay pinned to the edge so the frame under it is the next one along.
        self._scrubber.set_frame(self._len - 1 if self._edge_forward else 0)
        self.window_changed.emit()

    def _playback_range(self) -> tuple[int, int]:
        """Scrub positions autoplay runs between.

        The whole window, context included — the context is there to be watched. Except where
        the window walks: there it slides as far as the person scrubs, and playing on past the
        clip would run to the end of the recording, which is not what a boundary editor's
        preview is for.
        """
        if not self._clip_only():
            return 0, self._len - 1
        lo = self._clip_span[0] - self._bounds[0]
        hi = self._clip_span[1] - self._bounds[0]
        cap = max(self._len - 1, 0)
        return max(min(lo, cap), 0), max(min(hi, cap), 0)


#: The two lists a marked clip can join: the config key, and the word a user reads for it.
_KINDS = (("examples", "example"), ("negatives", "non-example"))
_KIND_WORD = dict(_KINDS)

#: Why Mark start / Mark end are greyed while no behavior exists: a clip marked then would have
#: no list to join. Said in the marking box as well as the tooltip, because a pressed button that
#: only wrote a status line looked like nothing was wrong (2026-11-23).
_NAME_FIRST = "Name a behavior first; then mark where it starts and ends."


def behavior_name_problem(name: str, existing) -> str | None:
    """Why ``name`` cannot name a behavior beside ``existing``, or None if it can.

    One rule for every place a behavior is named — the setup table's Add and Rename, and the
    example dialog's Add — so a name one of them refuses is not accepted by another.
    """
    if not name:
        return "A behavior needs a name."
    if name in existing:
        return f"A behavior named {name!r} is already listed."
    return None


def _with_article(word: str) -> str:
    return f"an {word}" if word[0] in "aeiou" else f"a {word}"


def _span_text(spec: dict) -> str:
    if "start_t_ns" in spec:
        return f"{format_time(spec['start_t_ns'])}–{format_time(spec['end_t_ns'])}"
    return f"frames {spec['start_frame']}–{spec['end_frame']}"


class _PlayerColumn(QWidget):
    """The example dialog's left column: the video pinned to the top, sized to its own shape,
    and what is ``below`` it directly under the transport.

    The block is ``top=True`` so its picture lines up with the checklist beside it. Given the
    column's stretch, a top-pinned block gathers the spare height into one gap under its
    transport — exactly between the playback buttons and the Mark pair that work together. So
    the block is held at the height the picture needs at this width (capped by the room the
    rows below leave), and the spare height goes under everything instead. The
    `label_check` player does the same for the same reason.
    """

    def __init__(self, block: PictureWithControls, below: list[QWidget]) -> None:
        super().__init__()
        self._block = block
        self._below = below
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        add_fitted_picture(column, block)
        for widget in below:
            column.addWidget(widget)
        column.addStretch(1)
        # Owned by this widget, so a refit queued as the dialog closes dies with it.
        self._refit = QTimer(self)
        self._refit.setSingleShot(True)
        self._refit.setInterval(0)
        self._refit.timeout.connect(self._fit)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt virtual
        super().resizeEvent(event)
        self._fit()

    def event(self, event) -> bool:  # noqa: N802 - Qt virtual
        # The block asks again when its transport grows (a strip appearing after the first
        # layout); a resize-only fit kept the old height and the button row was clipped.
        # Queued, as `label_check` does: refitting inside the request re-enters the layout.
        if event.type() == QEvent.Type.LayoutRequest:
            self._refit.start()
        return super().event(event)

    def _fit(self) -> None:
        bound_picture(self._block, self.width(), self._block.minimumSizeHint().height())


class ClipExampleDialog(Dialog):
    """Scrub the video and mark example clips by time.

    Two modes, one marking path:

    - **Every behavior at once** (``behaviors=[{name, examples, negatives}]``, Behavior Scan).
      A behavior table and an example / non-example toggle beside the player decide which
      list Mark start / Mark end adds to, so every behavior's examples and non-examples are
      marked in one pass over the video. The edited list comes back in ``.behaviors``. With
      no behavior yet, the marking box asks for one by name, and Mark start / Mark end and
      **Use these clips** stay greyed, each saying why in words beside it.
    - **One list** (``initial=[...]``, the hidden clip query and clip annotation modules). No
      behavior table or toggle; the marked clips come back in ``.examples``.

    ``recordings`` are the ones that carry ``embed_run_id``; each marked in/out pair becomes
    a ``{recording_id, track_id, start_t_ns, end_t_ns}`` spec.
    """

    def __init__(
        self,
        experiment,
        recordings,
        embed_run_id,
        parent=None,
        initial=None,
        max_gap_ns: int | None = None,
        require_contiguous: bool = True,
        *,
        behaviors: list[dict] | None = None,
        active: int = 0,
        negatives_reason: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._multi = behaviors is not None
        self.setWindowTitle("Mark examples" if self._multi else "Pick example clips")
        if self._multi:
            self.resize(1100, 740)
        else:
            self.resize(960, 700)
        self._preview = ClipQueryPreview(experiment.dir)
        self._embed = embed_run_id
        self._recordings = recordings
        self._by_id = {r.id: r for r in recordings}
        # Marking a clip needs the pixels, not just the embeddings: continuity below is
        # read from zarr, which a video-excluded export still has in full.
        self._video_available = True
        # The run's split gap, so what the picker accepts is what the search will accept.
        self._max_gap_ns = max_gap_ns
        # The selected method's answer, passed in for the same reason `max_gap_ns` is: the
        # picker must accept exactly what the search will. Required, so a caller that forgets
        # it gets the conservative rule rather than silently the permissive one.
        self._require_contiguous = require_contiguous
        if self._multi:
            self.behaviors = [
                {
                    "name": b["name"],
                    "examples": [dict(c) for c in b.get("examples", [])],
                    "negatives": [dict(c) for c in b.get("negatives", [])],
                }
                for b in behaviors
            ]
        else:
            self.behaviors = [
                {
                    "name": "",
                    "examples": [dict(c) for c in initial or []],
                    "negatives": [],
                }
            ]
        self._active = min(max(active, 0), len(self.behaviors) - 1)
        self._kind = "examples"
        # None when the method can use non-examples; otherwise its own sentence saying why not.
        self._negatives_reason = negatives_reason
        self._clips: list[tuple[int, str, dict]] = []  # the clip table's rows, in order
        self._warnings: dict[int, str] = {}  # id(spec) -> why a pre-loaded clip fails
        # id(spec) -> a frame-only clip's start in `t_ns`, from the pre-load check. The Time
        # column sorts on it, so a column holding both spellings sorts on one unit; the spec
        # itself keeps its frames, so what is saved back is what was loaded.
        self._start_t_ns: dict[int, int] = {}
        self._pending_start: int | None = None
        self._pending_frame: int | None = None  # where that start is, for its tick
        # The balance picker's columns and values: one metadata file, read once on open.
        self._metadata = experiment.recording_metadata if self._multi else {}
        self._metadata_columns = experiment.metadata_columns if self._multi else []
        # A clip selected on another recording or track: seek there once its continuity lands.
        self._pending_seek: tuple[str, int, dict] | None = None
        self._continuity: dict | None = None  # current subject's timeline + runs
        self._continuity_call: BackgroundCall | None = None
        self._continuity_gen = 0  # drop a load the user has already scrolled past
        # What the track says about marking, apart from whether a behavior exists: why it is
        # blocked (None once the continuity is read and usable), and whether that is the read
        # still running. `_apply_marking` combines it with the behavior list.
        self._track_blocks: str | None = None
        self._track_loading = False

        layout = self.body()

        top = QHBoxLayout()
        top.addWidget(QLabel("Recording:"))
        self._recording = QComboBox()
        for recording in recordings:
            self._recording.addItem(recording.name, recording.id)
        top.addWidget(self._recording)
        top.addWidget(QLabel("Track:"))
        self._track = IntSpin()
        self._track.setRange(0, 9999)
        top.addWidget(self._track)
        top.addStretch()
        layout.addLayout(top)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # One pane. The silhouette that used to sit here showed the embedder's input image —
        # a question about the method, beside footage you are watching to judge an animal — and
        # it took two fifths of the width to do it. The mask toggle below puts the same subject
        # marking on the frame itself, where it does not compete with the video for space.
        # Zoomable, like the player's: a clip's edge is a paw leaving the floor, which is found
        # by looking at the paw.
        self._src = ImageView(zoomable=True)
        self._src.setMinimumSize(400, 300)

        # The clip editor. You mark start/end here, which reads as a canvas — but it is the
        # surface the whole complaint came from: flailing is only recognisable in motion.
        self._scrubber = FrameScrubber(playable=True)
        self._scrubber.set_decoder(self._compute)
        self._scrubber.rendered.connect(self._display)
        self._decode_busy = decode_overlay(self._scrubber, self._src)
        # On by default, like every other player in the tool (#628 follow-up: a user opening
        # this one expected the mask and did not get it). It costs: the frame alone decodes in
        # 4.64 ms against 10.39 with its mask, the difference between 2x and not reaching 1.5x
        # on this footage — which the achieved-rate note says out loud, and unticking undoes.
        self._masks = QCheckBox("Masks")
        self._masks.setChecked(
            True
        )  # before `toggled` is wired, so it decodes nothing extra
        self._masks.setToolTip(
            "Tint the subject's mask onto the frame. About twice the cost a frame, so playback "
            "is slower with it on."
        )
        self._masks.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._masks.toggled.connect(self._on_masks_toggled)
        self._scrubber.add_control(self._masks)
        # The time is the scrubber's own now, beside the frame number (`set_timestamps` in
        # `_on_continuity`). This dialog used to add a readout of its own, which sat mid-row
        # while the frame number was pushed to the far end.
        self._coverage = _CoverageBar(self._scrubber)
        self._coverage.set_rule_note(
            "An example clip has to sit inside one continuous stretch."
            if require_contiguous
            else "An example clip may cross a break; the unseen frames are skipped."
        )
        self._scrubber.set_underlay(self._coverage)
        self._coverage.setVisible(
            False
        )  # until the continuity says there is a gap to show
        self._scrubber.frame_changed.connect(self._on_frame_changed)
        # Picture directly above its controls (#396), the pair taking the dialog's spare room.
        # `top=True`: pinned, so the picture's top edge is level with the checklist's beside it
        # rather than floating with whatever shape the video is.
        self._block = PictureWithControls(self._src, self._scrubber, top=True)

        # Layout A (#628): the Mark pair sits right under the playback row, where the hands
        # already are while finding a clip's edges; the right column is the checklist.
        annotate = self._build_annotate_box()
        self._span_status = QLabel("")  # the pending span, checked as you scrub
        # A refusal names the break it crossed, so the line is long — wrap it rather than
        # clipping the half that says what to do about it, and keep two lines' worth of room
        # reserved so the wrap has somewhere to go (the picture above takes all the stretch).
        self._span_status.setWordWrap(True)
        self._span_status.setMinimumHeight(self.fontMetrics().height() * 2 + 4)
        # A divider, not a fixed column: a fixed checklist kept every extra pixel of a widened
        # window for the video, and the clip list's last columns stayed cut off however wide
        # the window got. Both sides share a widening; the divider moves the rest.
        self._split = GripSplitter(Qt.Orientation.Horizontal, key="mark_examples")
        self._split.addWidget(_PlayerColumn(self._block, [annotate, self._span_status]))
        self._checklist = self._build_checklist()
        self._split.addWidget(self._checklist)
        self._split.setStretchFactor(0, 3)
        self._split.setStretchFactor(1, 2)
        self._split.set_foldable(1, "Checklist")
        if not self._split.remember():
            self._split.setSizes([680, self._checklist.minimumWidth()])
        layout.addWidget(self._split, 1)

        self._clip_model = QStandardItemModel()
        self._clip_model.setHorizontalHeaderLabels(
            ["Behavior", "Kind", "Time", "Recording", "Track", "Note"]
        )
        self._clip_table = QTableView()
        self._clip_table.setModel(self._clip_model)
        self._clip_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._clip_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._clip_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        make_columns_resizable(self._clip_table)
        fill_width(self._clip_table)
        enable_sorting(self._clip_table)
        self._clip_table.setColumnHidden(0, not self._multi)
        self._clip_table.setColumnHidden(1, not self._multi)
        self._clip_table.verticalHeader().setVisible(False)
        self._clip_table.selectionModel().currentRowChanged.connect(
            self._on_clip_selected
        )
        # Delete is the key a user reaches for; Backspace is what the Mac keyboard labels
        # "delete". Scoped to the table so neither fires while typing a track number.
        for key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            QShortcut(
                QKeySequence(key),
                self._clip_table,
                activated=self._delete_selected,
                context=Qt.ShortcutContext.WidgetShortcut,
            )
        self._delete = QPushButton("Delete clip")
        self._delete.setToolTip("Delete the selected clip (or press Delete)")
        self._delete.clicked.connect(self._delete_selected)
        # Until the first clip there is nothing to list: one line saying how to start, rather
        # than a header row over an empty grid.
        self._no_clips = QLabel(
            "No clips yet. Find the behavior in the video, press Mark start where it "
            "begins and Mark end where it ends."
        )
        self._no_clips.setWordWrap(True)
        # In the right column, under the checklist: at the bottom of the dialog the list pushed
        # the window past the 750px a window gets, and it scrolled with three clips in it.
        self._side.addWidget(self._no_clips)
        # Fills what the column has left, so it shows every clip it has room for; the spacer
        # below takes that room only while the list is hidden.
        self._side.addWidget(self._clip_table, 1)
        self._side.addWidget(self._delete, alignment=Qt.AlignmentFlag.AlignLeft)
        self._side.addStretch()

        # Why Use these clips is greyed, in words: the body's last row, right-aligned, so it sits
        # directly above the button. Not inside the button box — Qt hides any widget it did not
        # place there the next time it lays out its buttons — and not beside the box in the
        # dialog's own layout, which holds only the scroll and the box
        # (`requirements.a_dialog_scrolls_its_content`).
        self._accept_note = QLabel("")
        self._accept_note.setWordWrap(True)
        self._accept_note.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._accept_note.setVisible(False)
        layout.addWidget(self._accept_note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._use = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._use.setText("Use these clips" if self._multi else "Use these examples")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)

        # Changing the track spinbox fires per keystroke; coalesce so typing "12" reads the
        # timeline once, not once for "1" and again for "12".
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(200)
        self._reload_timer.timeout.connect(self._reload)

        self._refresh_behaviors()
        self._refresh_clips()
        self._recording.currentIndexChanged.connect(self._schedule_reload)
        self._track.valueChanged.connect(self._schedule_reload)
        self._reload()
        self._verify_initial()

    def _build_annotate_box(self) -> QWidget:
        """Under the playback row: what is being annotated, as what, and the Mark pair.

        With no behavior yet, the heading and the kind toggle give way to a field naming one
        (`_refresh_target`). The checklist's Add was the only way in, across the dialog from
        the box the user was looking at, while the Mark pair sat here pressable (2026-11-23).
        """
        box = QFrame()
        box.setFrameShape(QFrame.Shape.StyledPanel)
        column = QVBoxLayout(box)
        self._annotating = QLabel("")
        self._annotating.setFont(emphasis_font(self._annotating, steps=3))
        self._annotating.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._annotating)

        self._naming_prompt = QLabel("What behavior are you marking?")
        self._naming_prompt.setFont(emphasis_font(self._naming_prompt, steps=3))
        self._naming_prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._naming_prompt)
        naming = QHBoxLayout()
        self._new_behavior = QLineEdit()
        self._new_behavior.setPlaceholderText("Behavior name, such as grooming")
        self._new_behavior.returnPressed.connect(self._add_behavior_from_field)
        self._add_named = QPushButton("Add behavior")
        self._add_named.clicked.connect(self._add_behavior_from_field)
        naming.addWidget(self._new_behavior, 1)
        naming.addWidget(self._add_named)
        column.addLayout(naming)

        kinds = QHBoxLayout()
        kinds.addStretch()
        self._as_example = QRadioButton("example")
        self._as_negative = QRadioButton("non-example")
        self._as_example.setChecked(True)
        self._kind_group = QButtonGroup(box)
        self._kind_group.addButton(self._as_example)
        self._kind_group.addButton(self._as_negative)
        self._as_example.toggled.connect(self._on_kind_toggled)
        # Greyed with the method's reason, never hidden (§20): the capability stays visible
        # and says why it does not apply under the method that is selected.
        set_available(
            self._as_negative,
            self._negatives_reason is None,
            self._negatives_reason or "",
        )
        kinds.addWidget(self._as_example)
        kinds.addWidget(self._as_negative)
        kinds.addStretch()
        column.addLayout(kinds)

        # Why the Mark pair is greyed, in words: a tooltip alone needs a hover nobody makes.
        self._marking_note = QLabel("")
        self._marking_note.setWordWrap(True)
        self._marking_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._marking_note.setVisible(False)
        column.addWidget(self._marking_note)

        marks = QHBoxLayout()  # start left, end right (natural reading order)
        self._mark_start = QPushButton("Mark start")
        self._mark_start.clicked.connect(lambda: self._mark(start=True))
        self._mark_end = QPushButton("Mark end")
        self._mark_end.clicked.connect(lambda: self._mark(start=False))
        # Marking needs the track's continuity to judge the span, so it stays greyed until
        # the read lands rather than accepting a clip it cannot yet check.
        self._set_marking_loading()
        marks.addWidget(self._mark_start, 1)
        marks.addWidget(self._mark_end, 1)
        column.addLayout(marks)
        return box

    def _build_checklist(self) -> QWidget:
        """The right column: every behavior's progress and spread, and the balance nudge."""
        panel = QWidget()
        side = QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 0, 0)

        self._behavior_model = QStandardItemModel()
        self._behavior_model.setHorizontalHeaderLabels(
            ["Behavior", "Examples", "Non-examples", "Spread"]
        )
        self._behavior_table = QTableView()
        self._behavior_table.setObjectName("example_behaviors")  # SORT_EXEMPT says why
        self._behavior_table.setModel(self._behavior_model)
        self._behavior_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._behavior_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._behavior_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._behavior_table.verticalHeader().setVisible(False)
        make_columns_resizable(self._behavior_table)
        fill_width(self._behavior_table)
        self._behavior_table.selectionModel().currentRowChanged.connect(
            self._on_behavior_selected
        )
        side.addWidget(self._behavior_table)
        # Like the clip list: no header row over nothing, a line saying what to do instead.
        self._no_behaviors = QLabel("No behaviors yet. Add one to start annotating.")
        self._no_behaviors.setWordWrap(True)
        side.addWidget(self._no_behaviors)

        manage = QHBoxLayout()
        self._add_behavior_button = QPushButton("Add")
        self._add_behavior_button.setToolTip("Add a behavior to annotate")
        self._add_behavior_button.clicked.connect(self._add_behavior)
        self._rename_button = QPushButton("Rename")
        self._rename_button.setToolTip("Rename the selected behavior")
        self._rename_button.clicked.connect(self._rename_behavior)
        self._remove_button = QPushButton("Remove")
        self._remove_button.setToolTip("Remove the selected behavior and its clips")
        self._remove_button.clicked.connect(self._remove_behavior)
        for button in (
            self._add_behavior_button,
            self._rename_button,
            self._remove_button,
        ):
            manage.addWidget(button)
        manage.addStretch()
        side.addLayout(manage)

        balance = QHBoxLayout()
        self._balance_label = QLabel("Balance samples across:")
        balance.addWidget(self._balance_label)
        self._balance = QComboBox()
        self._balance.addItem("none", None)
        for column in self._metadata_columns:
            self._balance.addItem(column, column)
        self._balance.setToolTip(
            "A recording metadata column to sample evenly across, such as treatment. "
            "The hint below names a value with no examples yet."
        )
        set_available(
            self._balance,
            bool(self._metadata_columns),
            "This experiment has no recording metadata columns",
        )
        self._balance.currentIndexChanged.connect(self._refresh_hint)
        balance.addWidget(self._balance, 1)
        side.addLayout(balance)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        side.addWidget(self._hint)
        self._side = side  # the clip list joins it, built after

        for widget in (
            self._behavior_table,
            self._no_behaviors,
            self._add_behavior_button,
            self._rename_button,
            self._remove_button,
            self._balance,
            self._balance_label,
            self._hint,
        ):
            widget.setVisible(self._multi)
        panel.setMinimumWidth(
            text_width("Behavior  Examples  Non-examples  2 recs · 2 animals") + 70
        )
        return panel

    @property
    def examples(self) -> list[dict]:
        """One-list mode's result: the clips the caller handed in as ``initial``, as edited."""
        return self.behaviors[0]["examples"]

    def _subject(self) -> tuple[str, int]:
        return self._recording.currentData(), self._track.value()

    # -- behaviors and kind -------------------------------------------------- #
    def _behavior_color(self, index: int) -> str:
        """A behavior's colour: its swatch in the checklist and its clips on the timeline."""
        palette = plot_theme.current().categorical
        return palette[index % len(palette)]

    def _refresh_behaviors(self) -> None:
        table = self._behavior_table
        table.selectionModel().blockSignals(True)
        self._behavior_model.removeRows(0, self._behavior_model.rowCount())
        for index, behavior in enumerate(self.behaviors):
            name = QStandardItem(behavior["name"])
            name.setData(
                QColor(self._behavior_color(index)), Qt.ItemDataRole.DecorationRole
            )
            stamp_row_id(name, index)
            n_ex = len(behavior["examples"])
            n_neg = len(behavior["negatives"])
            negatives = num_cell(
                n_neg,
                f"{n_neg} (not used)" if n_neg and self._negatives_reason else None,
            )
            if n_neg and self._negatives_reason:
                negatives.setToolTip(self._negatives_reason)
            self._behavior_model.appendRow(
                [
                    name,
                    num_cell(n_ex),  # drawn over by the progress bar below
                    negatives,
                    QStandardItem(example_spread.spread_text(behavior["examples"])),
                ]
            )
            table.setIndexWidget(
                self._behavior_model.index(index, 1), self._progress_bar(n_ex)
            )
        if 0 <= self._active < len(self.behaviors):
            table.selectRow(self._active)
        table.selectionModel().blockSignals(False)
        table.resizeColumnsToContents()
        fit_table_height(table, max_rows=8)
        has_one = bool(self.behaviors)
        self._behavior_table.setVisible(self._multi and has_one)
        self._no_behaviors.setVisible(self._multi and not has_one)
        set_available(self._rename_button, has_one, "Add a behavior first")
        set_available(self._remove_button, has_one, "Add a behavior first")
        self._refresh_target()
        self._refresh_accept()

    def _refresh_accept(self) -> None:
        """Use these clips waits for one example, and says so beside itself.

        Closing on a list with none handed the launch form nothing to search for, and nothing
        in the dialog said so (2026-11-23). One behavior with an example is enough here: the
        launch form names any other behavior that still lacks one. The one-list mode is left
        as it was: its button is never greyed here.
        """
        if not self._multi:
            return
        reason = None
        if not self.behaviors:
            reason = "Mark at least one example first"
        elif not any(behavior["examples"] for behavior in self.behaviors):
            reason = f"{self.behaviors[0]['name']} has no examples yet"
        set_available(self._use, reason is None, reason or "")
        self._accept_note.setText(reason or "")
        self._accept_note.setVisible(reason is not None)

    @staticmethod
    def _progress_bar(count: int) -> QProgressBar:
        """Progress toward the advised number of examples, ticked once it is reached."""
        target = example_spread.TARGET_EXAMPLES
        bar = QProgressBar()
        bar.setRange(0, target)
        bar.setValue(min(count, target))
        bar.setFormat(f"{count} ✓" if count >= target else f"{count}/{target}")
        bar.setToolTip(
            f"{target} or more examples measurably steady the search. Advice, not a rule: "
            "one works."
        )
        return bar

    def _refresh_target(self) -> None:
        """The marking box's top: what the next clip joins — or, with no behavior yet, a field
        naming one, since until then there is nothing for a clip to join."""
        naming = self._multi and not self.behaviors
        for widget in (self._annotating, self._as_example, self._as_negative):
            widget.setVisible(self._multi and not naming)
        for widget in (self._naming_prompt, self._new_behavior, self._add_named):
            widget.setVisible(naming)
        if not self.behaviors:
            self._annotating.setText("")
        else:
            name = self.behaviors[self._active]["name"]
            suffix = " — non-examples" if self._kind == "negatives" else ""
            self._annotating.setText(f"Annotating: {name}{suffix}")
        self._apply_marking()  # a behavior added or removed opens or closes the Mark pair
        self._refresh_hint()

    def _refresh_hint(self, *_a) -> None:
        if not self.behaviors:
            self._hint.setText("")
            return
        behavior = self.behaviors[self._active]
        column = self._balance.currentData()
        self._hint.setText(
            example_spread.spread_hint(
                behavior["name"],
                behavior["examples"],
                {r.id: r.name for r in self._recordings},
                metadata=self._metadata if column else None,
                column=column,
            )
        )

    def _on_behavior_selected(self, current, _previous=None) -> None:
        key = row_id(current) if current.isValid() else None
        if key is not None:
            self._active = key
            self._refresh_target()

    def _on_kind_toggled(self, *_a) -> None:
        self._kind = "examples" if self._as_example.isChecked() else "negatives"
        self._refresh_target()

    def _add_behavior(self) -> None:
        name, ok = QInputDialog.getText(self, "Add behavior", "Behavior name:")
        if ok:
            self._add_behavior_named(name)

    def _add_behavior_from_field(self) -> None:
        """The marking box's Add behavior (or Return in its field): the checklist's path, with
        a refusal said on the line under the box, where the name was typed."""
        if self._add_behavior_named(
            self._new_behavior.text(), report=self._span_status
        ):
            self._new_behavior.selectAll()
            return
        self._new_behavior.clear()
        self._span_status.setText("")  # a refusal from the last try no longer applies

    def _add_behavior_named(
        self, name: str, *, report: QLabel | None = None
    ) -> str | None:
        """Add ``name`` and make it the one being annotated; return why not if refused.

        The one add path: the checklist's Add and the marking box's field both come here, so a
        name one refuses the other refuses too. The refusal goes on ``report``, a label beside
        where the name was typed — the status line when none is given.
        """
        name = name.strip()
        problem = behavior_name_problem(name, [b["name"] for b in self.behaviors])
        if problem is not None:
            (self._status if report is None else report).setText(problem)
            return problem
        self.behaviors.append({"name": name, "examples": [], "negatives": []})
        self._active = len(self.behaviors) - 1
        self._refresh_behaviors()
        return None

    def _rename_behavior(self) -> None:
        if not self.behaviors:
            return
        current = self.behaviors[self._active]["name"]
        name, ok = QInputDialog.getText(
            self, "Rename behavior", "Behavior name:", text=current
        )
        if ok:
            self._rename_behavior_named(name)

    def _rename_behavior_named(self, name: str) -> str | None:
        """Rename the behavior being annotated; return why not if refused."""
        name = name.strip()
        if name == self.behaviors[self._active]["name"]:
            return None
        others = [b["name"] for i, b in enumerate(self.behaviors) if i != self._active]
        problem = behavior_name_problem(name, others)
        if problem is not None:
            self._status.setText(problem)
            return problem
        self.behaviors[self._active]["name"] = name
        self._refresh_behaviors()
        self._refresh_clips()
        return None

    def _remove_behavior(self) -> None:
        """Remove the behavior being annotated and its clips — Cancel still brings them back."""
        if not self.behaviors:
            return
        removed = self.behaviors.pop(self._active)
        for spec in removed["examples"] + removed["negatives"]:
            self._warnings.pop(id(spec), None)
            self._start_t_ns.pop(id(spec), None)
        self._active = max(min(self._active, len(self.behaviors) - 1), 0)
        self._refresh_behaviors()
        self._refresh_clips()
        n = len(removed["examples"]) + len(removed["negatives"])
        self._status.setText(
            f"removed {removed['name']!r} and its {n} clip{'' if n == 1 else 's'} — "
            "Cancel discards every change in this dialog, this one included"
        )

    # -- the clip table ------------------------------------------------------ #
    def _refresh_clips(self) -> None:
        self._clips = [
            (b, kind, spec)
            for b, behavior in enumerate(self.behaviors)
            for kind, _word in _KINDS
            for spec in behavior[kind]
        ]
        model = self._clip_model
        self._clip_table.selectionModel().blockSignals(True)
        model.removeRows(0, model.rowCount())
        for index, (b, kind, spec) in enumerate(self._clips):
            recording = self._by_id.get(spec["recording_id"])
            note = self._warnings.get(id(spec), "")
            if kind == "negatives" and self._negatives_reason:
                note = f"not used: {self._negatives_reason}"
            behavior = QStandardItem(self.behaviors[b]["name"])
            behavior.setData(
                QColor(self._behavior_color(b)), Qt.ItemDataRole.DecorationRole
            )
            stamp_row_id(behavior, index)
            model.appendRow(
                [
                    behavior,
                    QStandardItem(_KIND_WORD[kind]),
                    num_cell(self._sort_t_ns(spec), _span_text(spec)),
                    QStandardItem(
                        recording.name
                        if recording is not None
                        else spec["recording_id"]
                    ),
                    num_cell(int(spec["track_id"])),
                    QStandardItem(f"⚠ {note}" if note else ""),
                ]
            )
        self._clip_table.selectionModel().blockSignals(False)
        # Sized to what is now in them: sized once while empty, the recording and time
        # columns read "recordin…" and "0:…" beside a Note column holding all the width.
        # Note is empty for every clip that is fine, which is nearly all of them: shown only
        # when one has something to say, so it does not take the width Time and Recording need.
        self._clip_table.setColumnHidden(
            5, not any(model.item(r, 5).text() for r in range(model.rowCount()))
        )
        # Track the same way: on a one-animal video every clip is track 0 and the column says
        # nothing. It appears once clips are on two animals, where it tells them apart.
        tracks = {int(spec["track_id"]) for _b, _kind, spec in self._clips}
        self._clip_table.setColumnHidden(4, len(tracks) <= 1)
        self._clip_table.resizeColumnsToContents()
        any_clips = bool(self._clips)
        self._no_clips.setVisible(not any_clips)
        self._clip_table.setVisible(any_clips)
        self._delete.setVisible(any_clips)
        set_available(self._delete, any_clips, "No clips marked yet")
        self._refresh_recording_counts()
        self._refresh_marks()
        if self._multi:
            self._refresh_hint()

    def _refresh_recording_counts(self) -> None:
        """Each recording in the picker says how many clips it has, so an unsampled one shows."""
        counts: dict[str, int] = {}
        for _b, _kind, spec in self._clips:
            counts[spec["recording_id"]] = counts.get(spec["recording_id"], 0) + 1
        for i in range(self._recording.count()):
            recording_id = self._recording.itemData(i)
            name = self._by_id[recording_id].name
            n = counts.get(recording_id, 0)
            self._recording.setItemText(
                i, f"{name} · {n} clip{'' if n == 1 else 's'}" if n else name
            )

    def _refresh_marks(self) -> None:
        """This subject's clips on the timeline, each in its behavior's colour — a non-example
        hollow — and a pending start's tick."""
        if self._continuity is None:
            self._scrubber.clear_marks()
            return
        recording_id, track = self._subject()
        spans = []
        for b, kind, spec in self._clips:
            if spec["recording_id"] != recording_id or int(spec["track_id"]) != track:
                continue
            first, last = self._frames_of(spec)
            spans.append((first, last, self._behavior_color(b), kind == "negatives"))
        ticks = [] if self._pending_frame is None else [self._pending_frame]
        self._scrubber.set_marks(ticks=ticks, bands=spans)

    def _frames_of(self, spec: dict) -> tuple[int, int]:
        if "start_t_ns" in spec:
            return (
                self._preview.t_ns_to_frame(
                    spec["recording_id"], self._embed, int(spec["start_t_ns"])
                ),
                self._preview.t_ns_to_frame(
                    spec["recording_id"], self._embed, int(spec["end_t_ns"])
                ),
            )
        return int(spec["start_frame"]), int(spec["end_frame"])

    def _selected_clip(self) -> int | None:
        rows = self._clip_table.selectionModel().selectedRows()
        return row_id(rows[0]) if rows else None

    def _delete_selected(self) -> None:
        index = self._selected_clip()
        if index is None:
            self._span_status.setText("select a clip in the list to delete it")
            return
        b, kind, spec = self._clips[index]
        clips = self.behaviors[b][kind]
        clips[:] = [c for c in clips if c is not spec]
        self._warnings.pop(id(spec), None)
        self._start_t_ns.pop(id(spec), None)
        self._refresh_clips()
        self._refresh_behaviors()
        what = (
            f" ({self.behaviors[b]['name']}, {_KIND_WORD[kind]})" if self._multi else ""
        )
        self._span_status.setText(f"deleted {_span_text(spec)}{what}")

    def _on_clip_selected(self, current, _previous=None) -> None:
        """Seek the player to the clip — switching recording and track first if need be."""
        key = row_id(current) if current.isValid() else None
        if key is None:
            return
        spec = self._clips[key][2]
        subject = (spec["recording_id"], int(spec["track_id"]))
        if subject == self._subject() and self._continuity is not None:
            self._seek_to(spec)
            return
        combo_index = self._recording.findData(spec["recording_id"])
        if combo_index < 0:
            self._span_status.setText(
                "that clip's recording is not among the ones being searched, so it "
                "cannot be shown here"
            )
            return
        self._pending_seek = (*subject, spec)
        self._recording.setCurrentIndex(combo_index)
        self._track.setValue(subject[1])
        self._schedule_reload()

    def _seek_to(self, spec: dict) -> None:
        self._scrubber.set_frame(self._frames_of(spec)[0])

    # -- loading a subject ---------------------------------------------------- #
    def _verify_initial(self) -> None:
        """Flag any pre-loaded clip that can no longer be used (it was marked before this
        rule existed, or against a different embed run) — editing a query must show the
        same verdict as adding to one."""
        specs = [spec for _b, _kind, spec in self._clips]
        if not specs:
            return
        self._initial_call = BackgroundCall(
            lambda: self._preview.check_examples(
                self._embed,
                specs,
                self._max_gap_ns,
                require_contiguous=self._require_contiguous,
            ),
            receiver=self,
        )
        self._initial_call.done.connect(
            lambda results: self._on_initial_checked(specs, results)
        )
        self._initial_call.failed.connect(lambda _exc: None)  # advisory only
        self._initial_call.start()

    def _on_initial_checked(self, specs: list[dict], results: object) -> None:
        present = {id(spec) for _b, _kind, spec in self._clips}
        for spec, result in zip(specs, results):
            if id(spec) not in present:
                continue
            if not result["ok"]:
                self._warnings[id(spec)] = result["reason"]
            if "start_t_ns" not in spec and result.get("start_t_ns") is not None:
                self._start_t_ns[id(spec)] = int(result["start_t_ns"])
        self._refresh_clips()

    def _sort_t_ns(self, spec: dict) -> int | None:
        """The clip's start in `t_ns`, the one unit the Time column sorts on.

        A clip marked here has it. A frame-only clip (an older config, a CLI ``--example``)
        has it once the pre-load check has resolved its frames; until then, or if they never
        resolve, ``None``, which sorts below every time. Sorting it on its frame number
        instead put every frame-only clip first whatever its time (#35).
        """
        if "start_t_ns" in spec:
            return int(spec["start_t_ns"])
        return self._start_t_ns.get(id(spec))

    def _set_marking(self, reason: str | None, *, loading: bool = False) -> None:
        """What the track says about marking: why it is blocked, or None once it is usable."""
        self._track_blocks = reason
        self._track_loading = loading
        self._apply_marking()

    def _set_marking_loading(self) -> None:
        """Greyed while the track's continuity is read — and the buttons *say* so.

        A tooltip alone was the only explanation, and a user pressing Mark start saw
        a button that seemed not to work. The read follows every recording or track change,
        including each keystroke in the Track box.
        """
        self._set_marking(
            "Reading this track's continuity, which is what judges the span",
            loading=True,
        )

    def _apply_marking(self) -> None:
        """Grey the Mark pair with the one reason that applies, and say it in the box.

        Two things gate marking: a behavior for the clip to join (the one-list mode always has
        its list) and the track, whose continuity judges the span. A missing behavior is the
        reason shown while both apply, because it is the one the user has to act on; the read
        finishes by itself, and once a behavior exists the buttons say "(loading…)" until it
        has. Recomputed from both whenever either changes, so naming a behavior opens marking
        only if the track is ready, and the track landing opens it only if a behavior exists.
        """
        nameless = self._multi and not self.behaviors
        reason = _NAME_FIRST if nameless else self._track_blocks
        loading = self._track_loading and not nameless
        for button, edge in ((self._mark_start, "start"), (self._mark_end, "end")):
            # `set_available`, not `setEnabled`: a greyed button with no tooltip reads as
            # broken rather than as not-yet.
            set_available(button, reason is None, reason or "")
            button.setText(f"Mark {edge} (loading…)" if loading else f"Mark {edge}")
        # The read says itself on the buttons; the note is for a reason that lasts.
        note = reason if self._multi and reason is not None and not loading else ""
        self._marking_note.setText(note)
        self._marking_note.setVisible(bool(note))

    def _schedule_reload(self, *_a) -> None:
        self._reload_timer.start()

    def _reload(self, *_a) -> None:
        recording_id, track = self._subject()
        dropped = self._pending_start
        self._continuity = None  # first, so the clear below draws no stale clips
        self._clear_pending_start()
        # Said, not done quietly: a start that vanishes because the track spinner moved otherwise
        # reads as Mark start not working.
        self._span_status.setText(
            ""
            if dropped is None
            else f"the start marked at {format_time(dropped)} was dropped — the recording or "
            "track changed, so mark it again here"
        )
        self._continuity = None
        self._coverage.clear()
        self._continuity_gen += 1
        generation = self._continuity_gen
        self._set_marking_loading()
        self._status.setText("reading this track's continuity…")
        self._continuity_call = BackgroundCall(
            lambda: self._load_subject(recording_id, track), receiver=self
        )
        self._continuity_call.done.connect(
            lambda result, g=generation: self._on_continuity(g, result)
        )
        self._continuity_call.failed.connect(
            lambda exc, g=generation: self._on_continuity_failed(g, exc)
        )
        self._continuity_call.start()

    def _load_subject(self, recording_id: str, track: int) -> dict:
        """Off-thread: the track's continuity **and** whether this recording's pixels are
        reachable here. Both are needed before marking opens, and one recording's presence
        check is a stat per part — never a sweep over every recording in the combo."""
        continuity = self._preview.track_continuity(  # off-thread (BackgroundCall)
            recording_id, self._embed, track, self._max_gap_ns
        )
        recording = self._by_id.get(recording_id)
        available = recording is None or recording.video_available
        # Measured off-thread with everything else this subject needs: the scrub source's own
        # median timestamp step, so playback here runs at the recording's rate rather than a
        # constant that is only real time by coincidence.
        rate = self._preview.frame_rate(recording_id, self._embed)
        scrub_t_ns = self._preview.timestamps(recording_id, self._embed)
        return {
            **continuity,
            "video_available": available,
            "frame_rate": rate,
            "scrub_t_ns": scrub_t_ns,
        }

    def _on_continuity(self, generation: int, result: object) -> None:
        if generation != self._continuity_gen:
            return  # a later subject owns the view now
        self._continuity = result
        self._video_available = bool(result["video_available"])
        if not self._video_available:
            # Refuse rather than let someone mark spans against a pane that never
            # renders — the clip would be committed to a frame nobody saw.
            self._set_marking("This recording's video isn't on this machine")
            self._coverage.clear()
            self._scrubber.set_total(0)
            self._status.setText(
                "This recording's video isn't on this machine (a video-excluded "
                "export), so its clips can't be marked. Bring the source video here "
                "and reopen this dialog."
            )
            self._src.setText(frame_source.unavailable_text(frame_source.MISSING))
            return
        self._set_marking(None)
        self._coverage.set_continuity(
            result["frame_idx"], result["runs"], result["n_frames"]
        )
        # Shown only where there is a break to see. The status line already says how much of
        # the track is embedded; on a fully tracked recording the strip was one solid block.
        self._coverage.setVisible(self._coverage.has_gaps())
        self._status.setText(
            f"track {self._track.value()} · embedded on {result['coverage']:.1%} of "
            f"{result['n_frames']} frames · {len(result['runs'])} continuous stretches"
        )
        self._scrubber.set_total(result["n_frames"])
        scrub_t_ns = result.get("scrub_t_ns")
        if scrub_t_ns is not None and len(scrub_t_ns) == result["n_frames"]:
            self._scrubber.set_timestamps(scrub_t_ns)
        if result.get("frame_rate"):
            self._scrubber.set_playback_rate(result["frame_rate"])
        seek, self._pending_seek = self._pending_seek, None
        if seek is not None and seek[:2] == self._subject():
            self._seek_to(seek[2])
        else:
            self._scrubber.set_frame(0)
        self._refresh_marks()  # `set_total` cleared them; this subject's clips go back on

    def _on_continuity_failed(self, generation: int, exc: object) -> None:
        if generation != self._continuity_gen:
            return
        self._coverage.clear()
        self._set_marking(f"Cannot read this track: {exc}")
        self._scrubber.set_total(0)
        self._status.setText(f"cannot scrub: {exc}")
        self._src.setText(f"(cannot scrub: {exc})")

    def _check(self, start_t_ns: int, end_t_ns: int):
        """This subject's verdict on a span — the same rule the search enforces, run over
        the timeline already in memory (no disk read per scrub step)."""
        c = self._continuity
        return clip_query.check_span(
            c["frame_idx"],
            c["t_ns"],
            start_t_ns,
            end_t_ns,
            c["max_gap_ns"],
            require_contiguous=self._require_contiguous,
        )

    def _describe_pending(self, end_frame: int) -> str:
        if self._pending_start is None or self._continuity is None:
            return ""
        try:
            t_ns = self._preview.frame_t_ns(
                self._recording.currentData(), self._embed, end_frame
            )
        except IndexError:
            return ""
        lo, hi = sorted((self._pending_start, t_ns))
        check = self._check(lo, hi)
        head = (
            f"span {format_time(lo)}–{format_time(hi)} · {check.n_rows} embedded frames"
        )
        if not check.ok:
            return f"{head} — {check.reason}"
        return self._with_gap_note(head, check)

    @staticmethod
    def _with_gap_note(head: str, check) -> str:
        """``head``, plus the break an accepted clip crosses — one sentence, one home.

        Accepted and still worth saying: the clip is usable, and the user should know the
        animal was not seen for part of what they marked. Silence reads as "the whole span is
        example frames". Written once because the pending line and the added line say it.
        """
        return head if not check.gap_note else f"{head} — {check.gap_note}; skipped"

    def _on_frame_changed(self, frame: int) -> None:
        if self._pending_start is not None:
            self._span_status.setText(self._describe_pending(frame))

    def _compute(self, global_frame: int):
        recording_id, track = self._subject()
        if self._masks.isChecked():
            return frame_source.decode(  # off-thread (scrubber decoder)
                lambda: self._preview.frame_masked(
                    recording_id, self._embed, track, global_frame
                )
            )
        return frame_source.decode(  # off-thread (scrubber decoder)
            lambda: self._preview.frame_plain(
                recording_id, self._embed, track, global_frame
            )
        )

    def _on_masks_toggled(self, _on: bool) -> None:
        """Redraw the frame on screen through the other path, rather than waiting for a move."""
        self._scrubber.set_frame(self._scrubber.frame())

    def _display(self, _frame: int, payload) -> None:
        view, reason = payload
        if reason is not None:
            self._src.setText(frame_source.unavailable_text(reason))
            return
        rgb, _actual, _t_ns = view
        # Full resolution: the view scales it, so zooming in reaches real pixels.
        self._src.set_frame(_to_pixmap(rgb))
        self._block.set_aspect(self._src.pixmap_size())

    def _mark(self, *, start: bool) -> None:
        recording_id, track = self._subject()
        if not self.behaviors:
            self._span_status.setText("add a behavior first, then mark its clips")
            return
        if not self._video_available:
            self._span_status.setText(
                "cannot mark: this recording's video isn't on this machine"
            )
            return
        if self._continuity is None:
            self._span_status.setText(
                "still reading this track's continuity — try again"
            )
            return
        frame = self._scrubber.frame()
        try:
            t_ns = self._preview.frame_t_ns(recording_id, self._embed, frame)
        except IndexError:
            self._span_status.setText(
                f"cannot mark at frame {frame}: this recording has no timestamp for it. "
                "Step back a frame and mark again."
            )
            return
        if start:
            moved_from, self._pending_start = self._pending_start, t_ns
            self._pending_frame = frame
            # Lit while a start waits for its end, with a tick where it is: the status line
            # alone said so somewhere other than where the next press goes.
            set_button_color(self._mark_start, READY)
            self._refresh_marks()
            # Where the example has to fit inside one continuous stretch, outlining the one
            # the start lands in says exactly how far the end may go. Where it does not, the
            # outline would claim a limit that is not there, so there is no bracket to draw.
            self._coverage.set_bracket(
                self._coverage.run_containing(frame)
                if self._require_contiguous
                else None
            )
            self._status.setText(
                f"start marked at {format_time(t_ns)} — now mark end"
                if moved_from is None
                # A second start replaces the first; say so rather than let it look ignored.
                else f"start moved to {format_time(t_ns)} (was {format_time(moved_from)}) "
                "— now mark end"
            )
            self._span_status.setText(self._describe_pending(frame))
            return
        if self._pending_start is None:
            self._span_status.setText("mark a start first")
            return
        lo, hi = sorted((self._pending_start, t_ns))
        check = self._check(lo, hi)
        if not check.ok:
            # Refuse here rather than let it fail inside the search, where the only report
            # is a red line in a dialog several steps later.
            self._span_status.setText(
                f"cannot use {format_time(lo)}–{format_time(hi)}: it {check.reason} — keep "
                "the clip inside the outlined stretch"
            )
            return
        self.behaviors[self._active][self._kind].append(
            {
                "recording_id": recording_id,
                "track_id": track,
                "start_t_ns": lo,
                "end_t_ns": hi,
            }
        )
        self._clear_pending_start()
        self._refresh_clips()
        self._refresh_behaviors()
        self._status.setText("")
        target = (
            f" to {self.behaviors[self._active]['name']!r} as "
            f"{_with_article(_KIND_WORD[self._kind])}"
            if self._multi
            else ""
        )
        self._span_status.setText(
            self._with_gap_note(
                f"added {format_time(lo)}–{format_time(hi)}{target} · "
                f"{check.n_rows} embedded frames",
                check,
            )
        )

    def _clear_pending_start(self) -> None:
        """Forget a pending start, and visibly: the lit button, the tick and the bracket go."""
        self._pending_start = None
        self._pending_frame = None
        set_button_color(self._mark_start, None)
        self._refresh_marks()
        self._coverage.set_bracket(None)

    def pending_start(self) -> int | None:
        """The ``t_ns`` of a marked start still waiting for its end, if there is one."""
        return self._pending_start

    def reject(self) -> None:
        self._clear_pending_start()
        super().reject()
