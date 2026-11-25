"""Every clip of a cluster, all playing: the montage under the cluster tree.

A grid of tiles, one per clip, in the order the page hands them (most typical, least typical or
random — the order switch here only *asks*; the page sorts). Every tile plays its own clip on its
own clock, looping from the clip's start; hovering a tile pauses every other one so the eye can
stay on it. A thin line under the hovered tile's picture shows how far through its loop it is.

**Only the hovered tile draws its loop line.** A line under every tile was thirty small bars all
moving at once, which pulled the eye off the footage; the hovered tile is the one still playing,
so it is the one whose position is worth showing. The line's band stays reserved under every
picture (:func:`progress_rects`), so hovering a tile moves nothing. Tiles carry no rank badge:
the rank is in the hover tooltip, where it does not cover the animal.

**The loop line sits under the picture, on the page, never over the footage.** It is a track
(the theme's grid colour) filled from the left in the theme's ink, and both are drawn on the
canvas below the picture. Drawn over the picture's foot it was the theme's near-black ink on
footage whose bottom rows are black letterbox: the fill vanished, and the black remainder —
the part *not* yet played — read as the bar, shrinking from the right, so every clip looked as if
it played backwards. The theme controls the page's colours and not the footage's, so a mark that
has to be read goes where the theme decides its contrast (:func:`progress_rects`).

**Which clips match.** While a filter is on (``set_matching(..., filtered=True)``), every
matching tile wears a :data:`FRAME_PX` frame in the match colour round its picture — with masks
on or off — and the header shows a key, "■ matches filters", in the frame's colour
(:func:`match_colours`). The grey of a miss needs no key entry: it is the absence of the colour,
and a second entry took width the header's figures line needs. With no filter on, no frames and
no key: every clip matches nothing.

**One clock per tile, one timer for all.** A single ``QTimer`` at :data:`FPS` advances each
visible tile's offset by the real time since the last tick, modulo that clip's own length — so a
1 s clip loops three times while a 3 s clip loops once, and no tile waits on another.

**Virtualised.** The grid paints and asks for pictures only for the tiles inside the viewport
(:meth:`LiveMontage.visible_indices`); a cluster of two thousand clips costs what its visible
thirty cost.

**Pictures come from a** :class:`TileSource`, **off the UI thread.** ``frame`` is called on one
worker (a ``GuiWorker``), which serves the newest request per tile and drops the ones a later
tick replaced, so a slow source makes the montage play at a lower rate rather than queue.

**The tiles in view first.** The worker serves the first picture of every tile in view (in
display order) before any tile's second, then the first picture of :data:`LOOKAHEAD_ROWS` row
past the view in the direction it last scrolled, then later frames. Every scroll, resize and
size change re-decides this and drops the requests of tiles that left the view and the
lookahead; the pictures they already have stay. Served in the order asked, a fast scroll to the
bottom of a large cluster drew hundreds of tiles already scrolled past before the ones
in view, seconds to the last picture. A clip
that does not match the page's filters still plays; the source is asked for its *grey* mask
picture (``matching=False``) rather than the tile being hidden or dimmed.

**Show ▾** (``show_menu.ShowButton``, in the header): Video, Masks, Mask trail, Keypoints and
the orientation, the page's look kept under the page's own key (``look_key``; the older
app-wide *Show masks* setting, :data:`MASKS_KEY`, is read once to start from). Masks off, the
source is asked for the plain picture (``masks=False``), and since
a grey mask is then no longer there to say a clip misses the filters, a clip that misses is
**washed out** instead — the theme's surface colour laid over the picture at
:data:`MISS_SCRIM_ALPHA`. What a tile last asked for is keyed on the flag, so a toggle redraws
every visible tile at once.

**Marks: a faded tile, its tag and its ✕.** A clip carries ``faded``, ``tag`` and ``crossed``: a
clip removed from its label in review stays where the clustering put it, washed out with the same
scrim as a miss (:data:`MISS_SCRIM_ALPHA`), with its tag ("not climb", "→ climb › left") painted
in the picture's bottom-left corner and — ``crossed``, a clip removed rather than moved — a bold
✕ in its top-right. ``note`` is the tile's first tooltip line ("Removed from climb in review,
8 Nov"). The montage only draws what the page hands it; the page decides what is faded.

**Click to mark** (:meth:`LiveMontage.set_click_hint`, off by default — Interpret leaves it off,
where a click does nothing). With a hint set and ``clip_clicked`` connected, a click on a tile
emits ``clip_clicked(index, shift)`` at once and the page decides what it means (AUDIT: remove
the clip from its label, or put it back; Shift: the run from the last clicked clip). The hovered
tile then shows the cue: a pointing hand, a faint ✕ where a removed clip wears its bold one, and
the hint as the tooltip's first line ("Click: remove from climb · double-click: open"). The cue
is derived from the wiring — no hint or no receiver, no hand and no ✕ — so a surface cannot
advertise a click it does not handle (review round 13, which replaced round 11's selection).

**Marking a run, with a preview first** (review round 14). Two gestures, one mechanism
(:class:`RunPreview`, :meth:`LiveMontage.preview`): **drag** from a clip — every clip from the one
pressed to the one under the mouse, in display order, is tinted and ringed, and letting go emits
``run_marked(indices, put_back)`` once; a drag that *starts* on a faded clip puts the run back
(blue) instead of removing it (vermillion), and either skips the clips already in that state. Esc
cancels, and the grid scrolls by itself while the mouse is near its top or bottom edge. A press and
release that moved less than ``QApplication.startDragDistance`` is a click. **Shift**: while it is
held over the grid, the run a Shift-click would remove — from the last clicked clip (the anchor,
ringed in ink) to the hovered one — wears the same tint, and the Shift-click emits ``run_marked``
with exactly those clips. The tint is the theme's diverging poles (``theme.hue``), painted inside
the picture like the ✕; ``preview_changed`` lets the page say in words what letting go will do.
Nothing previews on a montage that is not clickable, so a read-only grid offers no run.

**Opening a clip** is a double-click on a tile, or Space with the grid focused (the hovered tile,
else the last one clicked): ``clip_activated``. A double-click's first press is a click, so a
page that marks on click retracts it when the double-click follows (``label_clean.ReviewPane``).

**Sections** (``set_clips(..., sections=[(title, count), ...])``): consecutive runs of tiles under
a heading each, every section starting on a new row — AUDIT's finished cluster, "Removed (3)" then
"Kept as climb (57)". Without sections the grid is one run with no heading.

**How far the scroll has reached** (:meth:`LiveMontage.seen`): the number of tiles from the top
that have been on screen, which only grows until the clips change. ``seen_changed`` carries it, so
a page can wait until every tile has been looked at.

**Orientation** (the 11-9 meeting): **As recorded** / **Aligned**, inside **Show ▾** on every
montage, with Video, Masks, Mask trail, Keypoints and the context. Aligned turns each tile frame so the animal's body points up (``api.clip_tiles.TileLook``,
the turn ``api.clip_views`` writes once for the tiles, the clip window and the exports). It needs
the keypoint run the embedding read and a heading definition; until the page says it has them
(:meth:`LiveMontage.set_alignment`) the choice is greyed on *As recorded* with the reason. The choice is kept per
page, under the key the page names (``aligned_key``, required: every montage decides where its
choice lives, so none silently shares another's).

**What a tile draws** is a ``TileLook`` (:meth:`LiveMontage.set_look`): the footage, the masks and
their trail, the keypoints, the orientation — all of it from **Show ▾** — and the **context**,
seconds of footage either side of each clip (:meth:`LiveMontage.context_s`, also on **Show ▾**,
kept under the page's ``context_key``). The page widens its clips by it (:func:`context_span`):
the montage only says what was chosen.

**One toolbar on every clip grid** (2026-11-09, option A): **Most typical ▾ · Show ▾ · Size ·
Export ▾**, built here, so INTERPRET, the pop-out cluster window and AUDIT cannot differ — a page
hides what does not apply to it (AUDIT's fixed order) but adds nothing of its own. **Export ▾** and
its pick mode are ``montage_export.MontageExport``; the page hands the montage an
``ExportSource`` (required: what a grid exports is a fact about the page, never a default).

**A click layer** (:meth:`LiveMontage.set_click_layer`): while a layer is set, it — not the page
— decides what a click, a drag and a Shift-click mean, which clips are "off" for a run, which are
washed out (``dropped``) and which wear a number (``number``, painted in the top-left corner).
Pick mode is one: a click picks a clip or un-picks it, each pick wears its place in the export's
order, and a Shift-click picks a run (``shift_puts_back``). Taking the layer off gives the page's own click back unchanged (AUDIT's
click removes again), because the page's hint and connections were never touched. While a layer is
set no ✕ cue is drawn: dropping a clip from an export never looks like removing it from its label.

**Blocked.** While the previews a montage needs are still being built, :meth:`set_blocked` puts
the house progress bar (`gui/busy.py`'s overlay) in the grid's place — phase, fraction, time left,
Cancel — and the clock stops. The rest of the page stays usable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import replace
import threading
import time
from typing import Protocol

import numpy as np
from PyQt6.QtCore import (
    QEvent,
    QPoint,
    QRect,
    QRectF,
    QSettings,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFontMetrics,
    QImage,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from detail.api.clip_tiles import TileLook
from detail.core.progress import format_duration
from detail.gui import plot_theme
from detail.gui.busy import BusyOverlay
from detail.gui.lifetime import alive
from detail.gui.sizing import emphasis_font, small_font
from detail.gui.workers import GuiWorker

#: The rate every tile plays at — the preview proxy's frame rate.
FPS = 15
TICK_MS = round(1000 / FPS)
#: Real time a tick may advance a clock by at most, so a stall (a busy machine, a debugger) does
#: not jump every clip forward by seconds.
_MAX_STEP_NS = 200_000_000
TILE_PX = 132
MIN_TILE_PX = 96
MAX_TILE_PX = 240
_GAP = 6
#: The loop line's band under each picture: a gap, then the line, then the caption. Reserved on
#: every tile, drawn only on the hovered one, so hovering does not move the grid.
_BAR_GAP_PX = 1
BAR_PX = 3
#: The frame round a matching tile's picture while a filter is on.
FRAME_PX = 2
#: Whether the montage draws the subject's mask and outline, kept app-wide (under
#: ``interpret/``, which the GUI tests' fixture clears).
MASKS_KEY = "interpret/montage_masks"
#: How strongly a clip outside the filters is washed out when masks are off (0–255).
MISS_SCRIM_ALPHA = 165
#: A rule heading's line width (:meth:`LiveMontage.set_rule_headings`).
RULE_PX = 3
#: A removed clip's ✕ (and the hovered tile's faint one): its box's side and its inset from the
#: picture's top-right corner.
CROSS_PX = 18
CROSS_INSET = 4
#: How strongly the hovered tile's cue ✕ is drawn (0–255): a hint, not a mark.
CUE_ALPHA = 110
#: A run preview's tint over each picture (0–255) and the ring inside its edge, in pixels.
PREVIEW_ALPHA = 80
PREVIEW_RING_PX = 3
#: The shift anchor's ring, in pixels.
ANCHOR_RING_PX = 3
#: While dragging, the band at the grid's top and bottom edge that scrolls it, and the timer.
AUTOSCROLL_EDGE_PX = 36
AUTOSCROLL_MS = 30
#: A pick number's box: the font's height plus this much either side.
PICK_PAD = 3
#: What **Aligned** says while the page has not yet said whether this run can align its clips.
ALIGN_CHECKING = "Checking whether this run's keypoints can align its clips…"
ALIGN_TIP = (
    "Turn every clip so the animal's body points up, frame by frame, from the keypoint run the "
    "embedding read: postures compare across clips whichever way the animal faced. A "
    "side-view embedding is flipped where it flipped instead, never turned. A frame with no "
    'body axis plays as recorded, marked "no axis".'
)
#: The order switch's choices: ``(key, label)``. The page sorts; this widget only asks.
ORDERS = (
    ("typical", "Most typical"),
    ("atypical", "Least typical"),
    ("random", "Random"),
)
#: Rows past the view, in the direction it last scrolled, whose first picture is asked for
#: once every tile in view has one — so the next row of a slow scroll is already drawn.
LOOKAHEAD_ROWS = 1
#: How long the picture worker waits for another request before its thread ends.
_IDLE_S = 0.5


class TileSource(Protocol):
    """Where tile pictures come from — the preview cache's reader, at integration."""

    def frame(
        self,
        clip_index: int,
        offset_ns: int,
        tile_px: int,
        matching: bool,
        *,
        look: TileLook,
    ) -> np.ndarray:
        """The picture for clip ``clip_index`` at ``offset_ns`` from its start.

        A ``(tile_px, tile_px, 3)`` RGB ``uint8`` array (``PreviewClip.frame_at``'s), drawn as
        ``look`` says — the subject in the match colour when ``matching`` and in grey when not.
        Called on a worker thread, never the UI thread.
        """
        ...


@dataclass(frozen=True)
class MontageClip:
    """One tile: how long the clip loops, its typicality rank, its caption, whether it matches.

    ``faded``: washed out (a clip removed from its label in review, or moved to another), with
    ``tag`` painted in the picture's bottom-left corner and ``note`` as the tooltip's first line.
    ``crossed``: a bold ✕ in the top-right corner — removed, not moved.
    """

    duration_ns: int
    rank: int
    caption: str
    matching: bool = True
    tag: str | None = None
    faded: bool = False
    note: str = ""
    crossed: bool = False


@dataclass(frozen=True)
class ClickLayer:
    """What a click means while a mode (pick mode) has the grid (:meth:`LiveMontage.set_click_layer`).

    Every callable takes a display index. ``is_off``: whether the clip is in the state a click
    takes it out of (not picked) — a drag that starts on one puts its run back. ``clicked``: a
    plain click; ``run``: a drag let go or a Shift-click, ``(indices, put_back)``. ``dropped``:
    washed out; ``number``: the number painted in its top-left corner, or ``None``.
    ``shift_puts_back``: a Shift-click turns the run "on" (pick mode picks it) rather than off
    (AUDIT's removes it).
    """

    hint: Callable[[int], str]
    is_off: Callable[[int], bool]
    clicked: Callable[[int], None]
    run: Callable[[list, bool], None]
    dropped: Callable[[int], bool]
    number: Callable[[int], int | None]
    shift_puts_back: bool = False


def context_span(start_ns: int, end_ns: int, context_s: float) -> tuple[int, int]:
    """A clip's span widened by ``context_s`` either side — what a tile plays and an export
    writes (``api.clip_montage.with_context``'s widening, for one clip)."""
    pad = int(round(max(float(context_s), 0.0) * 1e9))
    return max(int(start_ns) - pad, 0), int(end_ns) + pad


@dataclass(frozen=True)
class RunPreview:
    """What letting go (``kind="drag"``) or a Shift-click (``kind="shift"``) would mark now:
    ``indices`` (display order, only the clips it would change) and whether it puts them back.
    """

    kind: str
    put_back: bool
    indices: tuple[int, ...]


def preview_colour(put_back: bool) -> QColor:
    """A run preview's colour: the theme's blue to put back, its vermillion to remove — the two
    poles of its diverging scale, so they read as opposites in either theme."""
    theme = plot_theme.current()
    return QColor(theme.hue("blue" if put_back else "vermillion"))


def match_colours() -> tuple[QColor, QColor]:
    """``(matches, doesn't)``: the colours a tile's subject is drawn in — the house mask tint, and
    the grey of a clip outside the filters (``api.clip_tiles.GREY``). One function for the
    tiles' frames and the header's key, so the key cannot disagree with the tiles."""
    from detail.api import clip_tiles
    from detail.core import visualize

    return QColor(*visualize.MASK_TINT), QColor(*clip_tiles.GREY)


def progress_rects(picture: QRect, done: float) -> tuple[QRect, QRectF]:
    """The loop line under ``picture``: its full track and the part ``done`` (0..1) fills, from
    the left. Below the picture, never over it (see the module docstring)."""
    top = picture.bottom() + 1 + _BAR_GAP_PX
    track = QRect(picture.x(), top, picture.width(), BAR_PX)
    done = min(max(float(done), 0.0), 1.0)
    return track, QRectF(picture.x(), top, picture.width() * done, BAR_PX)


class _MatchKey(QWidget):
    """ "■ matches filters" — the match frame's colour, painted."""

    _TEXTS = ("matches filters",)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFont(small_font(self))
        metrics = self.fontMetrics()
        box = metrics.height() - 4
        # No pad after the last entry: the row's own spacing follows, and every pixel here is one
        # the figures beside it lose at full width.
        width = (
            sum(box + 4 + metrics.horizontalAdvance(t) + 12 for t in self._TEXTS) - 12
        )
        self.setFixedSize(width, metrics.height() + 2)
        self.setToolTip(
            "Clips that match every filter wear a frame in this colour; the rest are drawn "
            "grey (or washed out, with masks off)."
        )

    def texts(self) -> tuple[str, ...]:
        return self._TEXTS

    def paintEvent(self, _event) -> None:  # noqa: N802 — Qt's name
        painter = QPainter(self)
        painter.setFont(self.font())
        metrics = painter.fontMetrics()
        box = metrics.height() - 4
        ink = QColor(plot_theme.current().text_secondary)
        x = 0
        for colour, text in zip(match_colours(), self._TEXTS):
            painter.fillRect(QRect(x, (self.height() - box) // 2, box, box), colour)
            x += box + 4
            painter.setPen(ink)
            painter.drawText(
                QRect(x, 0, metrics.horizontalAdvance(text) + 2, self.height()),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )
            x += metrics.horizontalAdvance(text) + 12
        painter.end()


def _to_image(picture: np.ndarray) -> QImage:
    """An RGB ``uint8`` array as an owned ``QImage``. Off-thread safe (``QImage``, not pixmap)."""
    picture = np.ascontiguousarray(picture, dtype=np.uint8)
    height, width = picture.shape[:2]
    return QImage(
        picture.data, width, height, 3 * width, QImage.Format.Format_RGB888
    ).copy()


class _PictureWorker(GuiWorker):
    """Serves tile requests, newest per tile, until it has been idle for :data:`_IDLE_S`.

    **Which request next** (:meth:`_take`): the first picture of a tile in view, in display
    order; then the first picture of a :meth:`focus` lookahead tile; then later frames, oldest
    request first. :meth:`focus` drops every request for a tile outside the view and the
    lookahead, so a fast scroll leaves no backlog of tiles already scrolled past.

    **Waking it.** Whether the thread is serving is a flag kept under the lock (``_live``), set
    by whoever hands it work and cleared by the thread in the same locked step that decides to
    go idle. Read off ``isRunning()`` instead, a request posted between the thread deciding to
    exit and the thread finishing found it "running", started nothing, and waited for the next
    request to be served.

    **The source in use** is held here (:meth:`adopt`) as well as on the montage, so
    :meth:`stop` — run when the montage is destroyed — closes it: a montage that went with its
    window left its open clips and drawn tiles to the garbage collector.
    """

    delivered = pyqtSignal(
        int, int, int, object
    )  # generation, index, offset_ns, QImage|None

    def __init__(self, *, receiver) -> None:
        super().__init__(receiver=receiver)
        self._lock = threading.Condition()
        #: ``{index: (seq, first, request)}`` — ``first``: the tile has no picture yet.
        self._pending: dict[int, tuple[int, bool, tuple]] = {}
        self._seq = 0
        #: ``{index: rank}`` of the tiles in view, then the lookahead; ``None`` until focused.
        self._rank: dict[int, int] | None = None
        self._ahead_from = 0
        #: Replaced sources, closed here: the worker is the only thread that touches a source.
        self._retired: list = []
        #: The montage's source now, closed by :meth:`stop`.
        self._held = None
        self._stopped = False
        #: Whether the thread is serving or about to: set by whoever hands it work, cleared by
        #: the thread itself as it goes idle — both under the lock.
        self._live = False

    def post(self, index: int, request: tuple, *, first: bool = False) -> None:
        """Ask for a picture; replaces any request for the same tile not yet served, keeping
        its place among the later frames. ``first``: the tile has no picture yet."""
        with self._lock:
            if self._stopped:
                return
            held = self._pending.get(index)
            if held is None:
                self._seq += 1
                seq = self._seq
            else:
                seq, first = held[0], first or held[1]
            self._pending[index] = (seq, first, request)
            self._lock.notify()
            wake = self._claim()
        self._wake(wake)

    def focus(self, visible: Sequence[int], ahead: Sequence[int]) -> list[int]:
        """Serve ``visible`` (in this order) first, then ``ahead``; drop every other request.

        Returns the tiles whose requests were dropped, so the montage can ask again when one
        comes back into view. The request in hand finishes; a dropped tile keeps the pictures it
        already has.
        """
        with self._lock:
            self._rank = {index: rank for rank, index in enumerate([*visible, *ahead])}
            self._ahead_from = len(visible)
            dropped = [index for index in self._pending if index not in self._rank]
            for index in dropped:
                del self._pending[index]
        return dropped

    def _take(self) -> int:
        """The pending tile to serve next (lock held): see the class docstring."""

        def order(item) -> tuple[int, int]:
            index, (seq, first, _request) = item
            rank = None if self._rank is None else self._rank.get(index)
            if first and (rank is None or rank < self._ahead_from):
                return (0, seq if rank is None else rank)
            if first:
                return (1, rank)
            return (2, seq)

        return min(self._pending.items(), key=order)[0]

    def adopt(self, source) -> None:
        """``source`` is the montage's now; the one before it is closed once the request in
        hand (which may still be using it) is served."""
        with self._lock:
            old, self._held = self._held, source
            if old is None or old is source:
                return
            self._retired.append(old)
            self._lock.notify()
            wake = self._claim()
        self._wake(wake)

    def retire(self, source) -> None:
        """Close ``source`` once the request in hand (which may still be using it) is served."""
        with self._lock:
            self._retired.append(source)
            self._lock.notify()
            wake = self._claim()
        self._wake(wake)

    def stop(self) -> None:
        """Serve nothing more, and close the source in use."""
        with self._lock:
            self._stopped = True
            self._pending.clear()
            if self._held is not None:
                self._retired.append(self._held)
                self._held = None
            self._lock.notify()
            wake = self._claim() if self._retired else False
        self._wake(wake)

    def _claim(self) -> bool:
        """(Lock held.) Whether the caller must start the thread: it is idle, or deciding to be."""
        if self._live:
            return False
        self._live = True
        return True

    def _wake(self, start: bool) -> None:
        if not start:
            return
        # The thread, if it ran, has already decided to go idle (it cleared `_live` under the
        # lock): let it finish so that `start` starts it again rather than doing nothing.
        self.wait()
        self.start()

    def run(self) -> None:
        # Nothing here keeps the worker alive or lets it go: `GuiWorker` holds a started
        # thread until Qt has finished with it. Letting go at the end of `run()` — as this did,
        # in a `finally` — is before Qt stops counting the thread as running, and a montage
        # destroyed a moment earlier left the worker to a collection in that gap.
        while True:
            with self._lock:
                if not self._pending and not self._retired and not self._stopped:
                    self._lock.wait(_IDLE_S)
                retired, self._retired = self._retired, []
                request = None
                if self._pending and not self._stopped:
                    index = self._take()
                    request = self._pending.pop(index)[2]
                if request is None and not retired:
                    self._live = False
                    return
            for source in retired:
                _close(source)
            if request is not None:
                self._serve(index, request)

    def _serve(self, index: int, request: tuple) -> None:
        source, generation, offset_ns, tile_px, matching, look = request
        try:
            frame = source.frame  # off-thread: this whole method runs on the worker
            picture = frame(index, offset_ns, tile_px, matching, look=look)
            image = _to_image(picture)
        except (
            Exception
        ):  # noqa: BLE001 — boundary: one bad tile must not stop the rest
            image = None
        self.emit_to_ui(self.delivered, generation, index, offset_ns, image)


def _close(source) -> None:
    close = getattr(source, "close", None)
    if close is not None:
        close()


class _Canvas(QWidget):
    """The painted grid. Holds no state of its own beyond hover; asks the montage for the rest."""

    def __init__(self, montage: LiveMontage) -> None:
        super().__init__()
        self._montage = montage
        self.setMouseTracking(True)

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if event.button() == Qt.MouseButton.LeftButton:
            point = event.position().toPoint()
            self._montage._pressed(self._montage.index_at(point), point)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if event.button() == Qt.MouseButton.LeftButton:
            self._montage._released(event.modifiers())
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if not self._montage._key(event):
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if event.key() == Qt.Key.Key_Shift:
            self._montage._set_shift(False)
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802 — Qt's name
        self._montage._cancel_drag()
        self._montage._set_shift(False)
        super().focusOutEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 — Qt's name
        index = self._montage.index_at(event.pos())
        if index is not None:
            self._montage.clip_context_requested.emit(index, event.globalPos())
            event.accept()
        else:
            event.ignore()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt's name
        self._montage._paint(self, event.rect())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        point = event.position().toPoint()
        index = self._montage.index_at(point)
        self._montage._moved(
            index,
            point,
            event.globalPosition().toPoint(),
            event.modifiers(),
            event.buttons(),
        )
        if index is not None and self._montage.is_clickable():
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

    def leaveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if not self._montage.is_dragging():
            self._montage._set_hovered(None)
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 — Qt's name
        point = event.position().toPoint()
        index = self._montage.index_at(point)
        if index is not None:
            self._montage.clip_activated.emit(index)

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            index = self._montage.index_at(event.pos())
            text = self._montage.hover_text(index) if index is not None else ""
            if text:
                QToolTip.showText(event.globalPos(), text, self)
            else:
                QToolTip.hideText()
                event.ignore()
            return True
        return super().event(event)


class _Grid(QScrollArea):
    """The scrolled viewport the canvas sits in; re-lays the canvas out on every resize."""

    def __init__(self, montage: LiveMontage) -> None:
        super().__init__()
        self._montage = montage
        self.setWidgetResizable(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().resizeEvent(event)
        self._montage._relayout()


class LiveMontage(QWidget):
    """The montage: a header (order switch, masks, tile size) over a grid of playing tiles."""

    #: The order switch was moved: ``"typical"``, ``"atypical"`` or ``"random"``.
    order_changed = pyqtSignal(str)
    #: The tile-size control was moved, in pixels.
    tile_px_changed = pyqtSignal(int)
    #: A tile asked to be opened at full size — a double-click, or Space: its display index.
    clip_activated = pyqtSignal(int)
    #: With a click hint set (:meth:`set_click_hint`): a tile was clicked — its display index,
    #: and whether Shift was held.
    clip_clicked = pyqtSignal(int, bool)
    #: How many tiles from the top have been on screen, when that grows (:meth:`seen`).
    seen_changed = pyqtSignal(int)
    #: A tile was right-clicked: its display index and the global position, for a menu.
    clip_context_requested = pyqtSignal(int, QPoint)
    #: A drag was let go, or a Shift-click made from an anchor: the display indices to mark
    #: (only those it changes) and whether they are put back rather than removed.
    run_marked = pyqtSignal(list, bool)
    #: The run preview (:meth:`preview`) appeared, changed or went.
    preview_changed = pyqtSignal()

    #: The orientation switch was moved: whether tiles are drawn **Aligned**.
    aligned_changed = pyqtSignal(bool)
    #: **Show ▾**'s context was changed: seconds either side of each clip.
    context_changed = pyqtSignal(float)
    #: The tiles were replaced or re-marked (:meth:`set_clips`).
    clips_changed = pyqtSignal()
    #: A tile got its first picture, or its picture could not be made, or the pictures were
    #: reset (:meth:`unpictured`).
    pictures_changed = pyqtSignal()

    def __init__(
        self,
        parent=None,
        *,
        aligned_key: str,
        look_key: str | None,
        context_key: str,
        export,
    ) -> None:
        """``aligned_key`` is the ``QSettings`` key this montage keeps its orientation under —
        the page's, so Interpret, AUDIT and the pop-out window each remember their own.
        ``look_key`` is where **Show ▾**'s layers are kept, or ``None`` to keep them for this
        montage only. ``context_key`` is where **Show ▾**'s context is kept. ``export`` is the
        page's ``montage_export.ExportSource``: the clips **Export ▾** writes, in grid order.
        """
        super().__init__(parent)
        self._clips: list[MontageClip] = []
        self._offsets: list[int] = []
        self._images: dict[int, QImage] = {}
        #: Per tile, the ``(offset, tile_px, matching)`` last asked for — a tile whose wanted
        #: picture has not changed (paused, nothing new) is not asked again.
        self._asked: dict[int, tuple] = {}
        self._failed: set[int] = set()
        #: Tiles whose first picture is asked for and not yet delivered: a tick asking again
        #: meanwhile is a later frame, and must not jump ahead of another tile's first.
        self._coming: set[int] = set()
        self._source: TileSource | None = None
        self._generation = 0
        self._tile_px = TILE_PX
        from detail.gui.show_menu import remembered_look

        self._look_key = look_key
        look = (
            remembered_look(look_key, masks_key=MASKS_KEY)
            if look_key is not None
            else TileLook()
        )
        self._masks = bool(look.masks)
        #: ``None``: the run's keypoints can be drawn; else why not (on **Keypoints**).
        self._points_note: str | None = ALIGN_CHECKING
        self._aligned_key = str(aligned_key)
        #: Whether the person chose Aligned here (kept); drawn only while the run can align.
        self._wants_aligned = QSettings().value(self._aligned_key, False, type=bool)
        #: ``None``: the run can align its clips; else why not (shown on **Aligned**).
        self._align_note: str | None = ALIGN_CHECKING
        self._look = look
        from detail.gui.show_menu import CONTEXT_CHOICES_S

        self._context_key = str(context_key)
        try:
            context = float(QSettings().value(self._context_key, 0.0))
        except (TypeError, ValueError):
            context = 0.0
        self._context_s = context if context in CONTEXT_CHOICES_S else 0.0
        #: The mode that has the grid's clicks (pick mode), or ``None`` for the page's own.
        self._layer: ClickLayer | None = None
        self._hovered: int | None = None
        self._filtered = False
        self._hover_text: Callable[[int], str] | None = None
        self._overlay: BusyOverlay | None = None
        self._last_tick: float | None = None
        self._click_hint: Callable[[int], str] | None = None
        #: Whether a drag or a Shift-click marks a run of tiles, or every press is one click
        #: (:meth:`set_click_hint`'s ``runs``).
        self._marks_runs = True
        self._last_clicked: int | None = None
        #: ``(index, point)`` of the left press not yet released, while clickable.
        self._press: tuple[int, QPoint] | None = None
        self._dragging = False
        self._drag_end: int | None = None
        #: Where the mouse last was during a drag, globally — what the autoscroll reads.
        self._drag_global: QPoint | None = None
        self._shift = False
        self._preview: RunPreview | None = None
        #: ``(title, count)`` per section, in display order; empty for one untitled run.
        self._sections: list[tuple[str, int]] = []
        #: Whether a titled section's heading is drawn as a dashed rule (:meth:`set_rule_headings`).
        self._rule_headings = False
        self._seen = 0
        #: The scroll position last seen, and whether the last move was down — which side of
        #: the view the lookahead row is on.
        self._scroll_at = 0
        self._scroll_down = True
        #: ``(window, its QWindow)`` whose minimise and exposure the clock follows.
        self._watched: tuple = (None, None)
        self._worker = _PictureWorker(receiver=self)
        self._worker.delivered.connect(self._on_delivered)
        self.destroyed.connect(self._worker.stop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        #: Lines the page puts above the controls row (the previews note, what an export wrote).
        self.header_lines = QVBoxLayout()
        self.header_lines.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.header_lines)
        #: The montage's own controls (the match key, the order switch, masks, size, and what
        #: :meth:`add_control` adds), in one widget so a page can seat them on a row of its own
        #: (:meth:`detach_controls`; INTERPRET puts them on its label bar's first row).
        self.controls_row = QWidget()
        header = QHBoxLayout(self.controls_row)
        header.setContentsMargins(0, 0, 0, 0)
        self._header = header
        # The stretch is outside the controls, so they sit at the right here and keep their
        # own width wherever they are seated.
        self._controls_line = QHBoxLayout()
        self._controls_line.addStretch(1)
        self._controls_line.addWidget(self.controls_row)
        self._controls_detached = False
        #: The match colours' key, shown only while a filter is on.
        self.match_key = _MatchKey(self)
        self.match_key.setVisible(False)
        header.addWidget(self.match_key)
        header.addSpacing(12)
        # One button opening the three orders, not three buttons: on the label bar's first row
        # the three took 260px of a 790px bar on a laptop, and the figures beside them had
        # none left (laptop layout, 2026-11-09).
        self.order_button = QPushButton()
        self.order_button.setObjectName("montage_order")
        self.order_button.setToolTip("The order the clips play in")
        # Its own width wherever the row is seated: a row stretch here would make the row
        # greedy on the label bar, where the figures beside it are what should take the room.
        self.order_button.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self._order_menu = QMenu(self)
        self._order_group = QActionGroup(self)
        self._order_group.setExclusive(True)
        #: The order switch's entries, by key: ``trigger()`` one to pick that order.
        self.order_actions: dict[str, QAction] = {}
        for key, text in ORDERS:
            action = self._order_menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(key == ORDERS[0][0])
            action.triggered.connect(lambda _c=False, k=key: self._order_picked(k))
            self._order_group.addAction(action)
            self.order_actions[key] = action
        self.order_button.clicked.connect(
            lambda: self._order_menu.exec(
                self.order_button.mapToGlobal(QPoint(0, self.order_button.height()))
            )
        )
        self._sync_order_button()
        header.addWidget(self.order_button)
        header.addSpacing(8)
        # One menu for the layers and the orientation (2026-11-09): a Show masks box and an
        # As recorded ▾ menu beside it took the label bar's first row from the figures.
        from detail.gui.show_menu import ShowButton

        self.show_button = ShowButton()
        self.show_button.layer_toggled.connect(self._layer_toggled)
        self.show_button.aligned_wanted.connect(self.set_aligned)
        self.show_button.context_wanted.connect(self.set_context_s)
        header.addWidget(self.show_button)
        self._sync_orientation()
        header.addSpacing(8)
        header.addWidget(QLabel("Size"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(MIN_TILE_PX, MAX_TILE_PX)
        self.size_slider.setValue(self._tile_px)
        # 72, not 96: the row shares the label bar's first line with the cluster's figures.
        self.size_slider.setFixedWidth(72)
        self.size_slider.setToolTip("Tile size")
        self.size_slider.valueChanged.connect(self.set_tile_px)
        header.addWidget(self.size_slider)
        layout.addLayout(self._controls_line)
        #: Between the controls and the grid: the pick bar while picking, and what the last
        #: export wrote (``montage_export``).
        self.above_grid = QVBoxLayout()
        self.above_grid.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.above_grid)

        self.grid = _Grid(self)
        self.canvas = _Canvas(self)
        self.grid.setWidget(self.canvas)
        self.grid.verticalScrollBar().valueChanged.connect(self._scrolled)
        layout.addWidget(self.grid, 1)

        from detail.gui.montage_export import MontageExport

        #: **Export ▾** and pick mode, on the controls row and over the grid.
        self.exports = MontageExport(self, export)

        self._clock = QTimer(self)
        self._clock.setInterval(TICK_MS)
        self._clock.timeout.connect(self._tick)
        self._autoscroll = QTimer(self)
        self._autoscroll.setInterval(AUTOSCROLL_MS)
        self._autoscroll.timeout.connect(self._autoscroll_tick)

    # -- inputs ------------------------------------------------------------- #
    def add_control(self, widget: QWidget) -> None:
        """Put ``widget`` at the end of the controls row (**Export ▾**) — wherever that row is
        seated, here or on the page (:meth:`detach_controls`)."""
        self._header.addWidget(widget)

    def detach_controls(self) -> QWidget:
        """Hand the controls row to the page to seat elsewhere; the montage then has no row of
        its own above the grid. Returns :attr:`controls_row` — the page adds it to a layout.
        """
        self._controls_line.removeWidget(self.controls_row)
        self._controls_detached = True
        return self.controls_row

    def set_source(
        self, source: TileSource | None, *, keep_pictures: bool = False
    ) -> None:
        """Where pictures come from. Replacing it re-asks for every visible tile; the old one is
        closed (on the picture worker, which is the only thread that ever touches it).

        ``keep_pictures``: the same clips from a better source (a recording's masks were just
        built) — each tile keeps its picture until the new one arrives, and every clock, the
        scroll and the tiles stay as they are.
        """
        self._source = source
        self._worker.adopt(source)
        if keep_pictures:
            self._generation += 1
            self._asked.clear()
            self._failed.clear()
            self._coming.clear()
            self._request_visible()
            self.pictures_changed.emit()
        else:
            self._reset_pictures()

    def source(self) -> TileSource | None:
        return self._source

    def set_clips(
        self,
        clips: Sequence[MontageClip],
        *,
        keep_scroll: bool = False,
        sections: Sequence[tuple[str, int]] | None = None,
    ) -> None:
        """The tiles, in display order. Every clip starts from its own beginning.

        ``keep_scroll``: the same tiles with new marks (a removal fading a clip) — the scroll
        stays where it is, and when the count is unchanged so do every tile's clock and picture.
        Otherwise the montage starts at the top.

        ``sections``: ``(title, count)`` runs of the tiles, in order, each under its heading and
        starting on a new row; the counts must add up to the clips. ``None``: one run, no heading.

        Raises:
            ValueError: the sections' counts do not add up to the clips.
        """
        clips = list(clips)
        sections = [(str(t), int(n)) for t, n in (sections or ())]
        if sections and sum(n for _t, n in sections) != len(clips):
            raise ValueError(
                f"sections hold {sum(n for _t, n in sections)} clips, not {len(clips)}"
            )
        same = keep_scroll and len(clips) == len(self._clips)
        scroll = self.grid.verticalScrollBar().value()
        self._clips = clips
        self._sections = sections
        if same:
            self._relayout()
            self.grid.verticalScrollBar().setValue(scroll)
            self._request_visible()
            self._update_preview()
            self.clips_changed.emit()
            return
        self._cancel_drag()
        self._offsets = [0] * len(self._clips)
        self._hovered = None
        self._last_clicked = None
        self._update_preview()
        self._seen = 0
        self._reset_pictures()
        self._relayout()
        self.grid.verticalScrollBar().setValue(scroll if keep_scroll else 0)
        self._sync_clock()
        # Last: a listener (pick mode leaving on other clips) may relay the grid, and must find
        # every tile's clock in place.
        self.clips_changed.emit()

    def sections(self) -> list[tuple[str, int]]:
        return list(self._sections)

    def set_rule_headings(self, on: bool) -> None:
        """Draw each titled section's heading as a **rule**: a dashed line across the grid with the
        title centred on it, in a band about three lines tall — a boundary to cross, where a plain
        heading is a label to read.

        CALIBRATE's cutoff. A stepped line through one unbroken grid moved no tile but was easy to
        miss; a heading in the font of a caption was the first build's, under faded clips. The rule
        is dashed and in the theme's handle colour (the curve's cutoff line), so it is told from a
        tile edge by its pattern as well as its colour, and its words say which side is which.
        """
        self._rule_headings = bool(on)
        self._relayout()
        self.canvas.update()

    def clips(self) -> list[MontageClip]:
        return list(self._clips)

    def set_matching(self, flags: Sequence[bool], *, filtered: bool) -> None:
        """Which clips match the filters, by display index — redrawn grey or in colour.

        ``filtered``: whether any filter is on. Only then do matching tiles wear the match
        frame and the header show its key; with none, every clip "matches" nothing.
        """
        if len(flags) != len(self._clips):
            raise ValueError(f"{len(flags)} flags for {len(self._clips)} clips")
        self._filtered = bool(filtered)
        self.match_key.setVisible(self._filtered)
        self._clips = [replace(c, matching=bool(f)) for c, f in zip(self._clips, flags)]
        self._request_visible()

    # -- clicking ----------------------------------------------------------- #
    def set_click_hint(
        self, hint: Callable[[int], str] | None, *, runs: bool = True
    ) -> None:
        """Turn click-to-mark on with ``hint`` — what a click on tile ``index`` will do, the first
        line of its tooltip — or off with ``None`` (the default).

        ``runs=False``: a click means one tile and nothing more — no drag or Shift-click run, and
        no run preview drawn. For a page whose click *chooses* a tile (CALIBRATE moving its
        cutoff to it) rather than marking it: a drag there would draw a removal that never
        happens.
        """
        self._click_hint = hint
        self._marks_runs = bool(runs)
        if hint is None:
            self.canvas.unsetCursor()
            self._cancel_drag()
        self._update_preview()
        self.canvas.update()

    def is_clickable(self) -> bool:
        """Whether a click on a tile does something: a click layer is set, or a hint is set and
        ``clip_clicked`` has a receiver. The hover cue is derived from this, never set beside it.
        """
        if self._layer is not None:
            return True
        return self._click_hint is not None and self.receivers(self.clip_clicked) > 0

    def click_hint(self, index: int) -> str:
        if not self.is_clickable() or not 0 <= index < len(self._clips):
            return ""
        if self._layer is not None:
            return self._layer.hint(index)
        return self._click_hint(index)

    def set_click_layer(self, layer: ClickLayer | None) -> None:
        """Hand the grid's clicks to ``layer`` (pick mode), or back to the page with ``None``.

        The page's own hint and connections are left as they were, so taking the layer off
        restores exactly what a click did before. The anchor and any drag are dropped either way.
        """
        self._layer = layer
        self._cancel_drag()
        self._last_clicked = None
        if not self.is_clickable():
            self.canvas.unsetCursor()
        self._update_preview()
        self.canvas.update()

    def click_layer(self) -> ClickLayer | None:
        return self._layer

    def _shift_puts_back(self) -> bool:
        return self._layer is not None and self._layer.shift_puts_back

    def _emit_run(self, indices: list, put_back: bool) -> None:
        if self._layer is not None:
            self._layer.run(list(indices), put_back)
        else:
            self.run_marked.emit(list(indices), put_back)

    def last_clicked(self) -> int | None:
        """The anchor a Shift-click's run starts from: the last clip clicked, or a drag's end."""
        return self._last_clicked

    def clear_anchor(self) -> None:
        """Forget the anchor — a Shift-click then is a plain click (a new review)."""
        self._last_clicked = None
        self._update_preview()

    def run_between(self, a: int, b: int, put_back: bool) -> list[int]:
        """The clips from ``a`` to ``b`` (either order, both included) a run would change: the
        faded ones to put back, else the ones not faded."""
        low, high = sorted((int(a), int(b)))
        low, high = max(low, 0), min(high, len(self._clips) - 1)
        return [i for i in range(low, high + 1) if self.is_off(i) == put_back]

    def preview(self) -> RunPreview | None:
        """What letting go of the drag, or a Shift-click, would mark now; ``None`` with neither."""
        return self._preview

    def is_dragging(self) -> bool:
        return self._dragging

    def cross_rect(self, index: int) -> QRect:
        """Where tile ``index``'s ✕ sits: the picture's top-right corner."""
        cell = self.tile_rect(index)
        return QRect(
            cell.x() + self._tile_px - CROSS_INSET - CROSS_PX,
            cell.y() + CROSS_INSET,
            CROSS_PX,
            CROSS_PX,
        )

    def _pressed(self, index: int | None, point: QPoint) -> None:
        """A left press: a click or the start of a drag, decided by how far the mouse moves."""
        self._cancel_drag()
        if index is None or not self.is_clickable():
            return
        self.canvas.setFocus(Qt.FocusReason.MouseFocusReason)
        self._press = (index, QPoint(point))

    def _moved(
        self,
        index: int | None,
        point: QPoint,
        global_point: QPoint,
        modifiers,
        buttons=Qt.MouseButton.LeftButton,
    ) -> None:
        """The mouse moved over (or, while dragging, anywhere off) the canvas."""
        if self._press is not None and not buttons & Qt.MouseButton.LeftButton:
            self._cancel_drag()  # the release went somewhere else
        if self._press is not None and not self._dragging and self._marks_runs:
            moved = (point - self._press[1]).manhattanLength()
            if moved >= QApplication.startDragDistance():
                self._dragging = True
                self._drag_end = self._press[0]
                self._autoscroll.start()
        if self._dragging:
            self._drag_global = QPoint(global_point)
            if index is not None:
                self._drag_end = index
                self._set_hovered(index)
            self._update_preview()
            return
        self._set_hovered(index)
        self._set_shift(bool(modifiers & Qt.KeyboardModifier.ShiftModifier))

    def _released(self, modifiers) -> None:
        """Let go: a drag marks its run; a press that never became one is a click."""
        press, dragging = self._press, self._dragging
        preview, end = self._preview, self._drag_end
        self._cancel_drag()
        if press is None:
            return  # a double-click's second release, or a cancelled drag
        if dragging:
            if preview is not None and preview.kind == "drag":
                self._last_clicked = press[0] if end is None else end
                self._update_preview()
                if preview.indices:
                    self._emit_run(list(preview.indices), preview.put_back)
            return
        self._clicked(press[0], modifiers)

    def _cancel_drag(self) -> bool:
        """Drop a press or drag in progress; ``True`` when there was a drag to drop."""
        had = self._dragging
        self._press = None
        self._dragging = False
        self._drag_end = None
        self._drag_global = None
        self._autoscroll.stop()
        self._update_preview()
        return had

    def _set_shift(self, held: bool) -> None:
        if held != self._shift:
            self._shift = held
            self._update_preview()

    def _update_preview(self) -> None:
        """Recompute the run preview from the drag, Shift and the hovered tile."""
        preview = None
        if self.is_clickable() and self._clips and self._marks_runs:
            if self._dragging and self._press is not None:
                start = self._press[0]
                end = start if self._drag_end is None else self._drag_end
                back = self.is_off(start)
                run = self.run_between(start, end, back)
                preview = RunPreview("drag", back, tuple(run))
            elif (
                self._shift
                and self._press is None
                and self._hovered is not None
                and self._last_clicked is not None
                and 0 <= self._last_clicked < len(self._clips)
            ):
                back = self._shift_puts_back()
                run = self.run_between(self._last_clicked, self._hovered, back)
                preview = RunPreview("shift", back, tuple(run))
        if preview != self._preview:
            self._preview = preview
            self.canvas.update()
            self.preview_changed.emit()

    def _autoscroll_tick(self) -> None:
        """While dragging near the grid's top or bottom edge, scroll — faster the nearer."""
        if not self._dragging or self._drag_global is None:
            self._autoscroll.stop()
            return
        viewport = self.grid.viewport()
        y = viewport.mapFromGlobal(self._drag_global).y()
        edge = AUTOSCROLL_EDGE_PX
        if y < edge:
            step = -min(edge - y, 3 * edge) // 2 - 2
        elif y > viewport.height() - edge:
            step = min(y - (viewport.height() - edge), 3 * edge) // 2 + 2
        else:
            return
        bar = self.grid.verticalScrollBar()
        before = bar.value()
        bar.setValue(before + step)
        if bar.value() == before:
            return
        point = self.canvas.mapFromGlobal(self._drag_global)
        point.setX(min(max(point.x(), 0), self.canvas.width() - 1))
        index = self.index_at(point)
        if index is not None:
            self._drag_end = index
            self._set_hovered(index)
        self._update_preview()

    def _clicked(self, index: int | None, modifiers) -> None:
        if index is None or not self.is_clickable():
            return
        self.canvas.setFocus(Qt.FocusReason.MouseFocusReason)
        shift = bool(
            (modifiers or Qt.KeyboardModifier.NoModifier)
            & Qt.KeyboardModifier.ShiftModifier
        )
        anchor = self._last_clicked
        self._last_clicked = index
        if (
            shift
            and self._marks_runs
            and anchor is not None
            and 0 <= anchor < len(self._clips)
        ):
            # Exactly the run the Shift preview showed — computed by the same function.
            back = self._shift_puts_back()
            run = self.run_between(anchor, index, back)
            self._update_preview()
            if run:
                self._emit_run(run, back)
            return
        if self._layer is not None:
            self._layer.clicked(index)
            return
        self.clip_clicked.emit(index, shift)

    def _key(self, event) -> bool:
        """Space opens the hovered clip, else the last one clicked; Esc cancels a drag; Shift
        shows the run a Shift-click would remove. ``True`` when used."""
        if event.key() == Qt.Key.Key_Escape and self._cancel_drag():
            return True
        if event.key() == Qt.Key.Key_Shift:
            self._set_shift(True)
            return False
        if event.key() != Qt.Key.Key_Space or not self._clips:
            return False
        index = self._hovered if self._hovered is not None else self._last_clicked
        if index is None or not 0 <= index < len(self._clips):
            return False
        self.clip_activated.emit(index)
        return True

    # -- how far the scroll has reached ------------------------------------- #
    def seen(self) -> int:
        """How many tiles from the top have been on screen since the clips were set — a row
        counting once half its pictures' height was in view: equal to the clip count once the
        last row has been."""
        return self._seen

    def _note_seen(self) -> None:
        if (
            not self.isVisible()
            or self.grid.viewport().height() <= 0
            or not self._clips
        ):
            return
        # A row counts once half of its pictures' height has been in view: a sliver of the
        # last row peeking under the fold is not that row looked at.
        cols, step = self._columns(), self._cell().height() + _GAP
        bottom = self.grid.verticalScrollBar().value() + self.grid.viewport().height()
        reached = 0
        for start, count, _title, first in self._runs():
            rows = (bottom - first - self._tile_px // 2) // step + 1
            if rows > 0:
                reached = start + min(rows * cols, count)
        if reached > self._seen:
            self._seen = reached
            self.seen_changed.emit(reached)

    def set_hover_text(self, text_for: Callable[[int], str] | None) -> None:
        """What a tile's tooltip says: rank, recording, each filter's ✓ / ✗, position."""
        self._hover_text = text_for

    def set_order(self, key: str) -> None:
        """Show ``key`` as the current order, without emitting ``order_changed``."""
        self.order_actions[key].setChecked(True)
        self._sync_order_button()

    def order(self) -> str:
        return next(k for k, a in self.order_actions.items() if a.isChecked())

    def _order_picked(self, key: str) -> None:
        self._sync_order_button()
        self.order_changed.emit(key)

    def _sync_order_button(self) -> None:
        self.order_button.setText(f"{dict(ORDERS)[self.order()]} ▾")

    def tile_px(self) -> int:
        return self._tile_px

    def set_tile_px(self, px: int) -> None:
        px = max(MIN_TILE_PX, min(int(px), MAX_TILE_PX))
        if px == self._tile_px:
            return
        self._tile_px = px
        self.size_slider.blockSignals(True)
        self.size_slider.setValue(px)
        self.size_slider.blockSignals(False)
        self._relayout()
        self._request_visible()
        self.tile_px_changed.emit(px)

    def masks(self) -> bool:
        return self._masks

    def set_masks(self, on: bool) -> None:
        """Draw the subject's mask and its trail, or neither; kept under the page's key."""
        on = bool(on)
        if on == self._masks:
            return
        try:
            look = replace(self._look, masks=on, mask_trail=on)
        except ValueError:
            look = replace(self._look, masks=on, mask_trail=on, footage=True)
        self._set_look(look)

    def _layer_toggled(self, key: str, on: bool) -> None:
        from detail.gui.show_menu import toggled

        try:
            look = toggled(self._look, key, on)
        except ValueError:  # the last layer: greyed, so only a stale menu gets here
            self._sync_orientation()
            return
        if key == "masks" and not on:
            look = replace(look, mask_trail=False)
        self._set_look(look)

    def _set_look(self, look: TileLook) -> None:
        from detail.gui.show_menu import remember_look

        self._look = replace(look, aligned=False)
        self._masks = bool(look.masks)
        if self._look_key is not None:
            remember_look(self._look_key, self._look)
        self._sync_orientation()
        self._request_visible()

    def set_points(self, note: str | None) -> None:
        """Whether the run's keypoints can be drawn: ``None`` they can, else why not —
        **Keypoints** is greyed with it, and a remembered tick is not drawn."""
        self._points_note = note
        self._sync_orientation()
        self._request_visible()

    # -- what a tile draws -------------------------------------------------- #
    def look(self) -> TileLook:
        """What the tiles are drawn with now: the page's look, Aligned only while the run can,
        keypoints only while it has them (a tile with nothing else left draws its video).
        """
        look = self._look
        if look.keypoints and self._points_note is not None and self._look_key:
            # One replace: keypoints alone, taken away, would be a look that draws nothing.
            look = replace(
                look, keypoints=False, footage=look.footage or not look.masks
            )
        return replace(look, aligned=self.is_aligned())

    def set_look(self, look: TileLook) -> None:
        """Draw every tile as ``look`` says. Its ``aligned`` is ignored: the orientation is the
        switch's (:meth:`set_aligned`), which knows whether the run can align."""
        self._set_look(look)

    def is_aligned(self) -> bool:
        """Whether the tiles are drawn **Aligned**: chosen, and the run can."""
        return self._wants_aligned and self._align_note is None

    def alignment_note(self) -> str | None:
        """``None`` when the run can align its clips, else why not."""
        return self._align_note

    def set_aligned(self, on: bool) -> None:
        """Choose the orientation (kept under the page's key); drawn Aligned only while the
        run can align its clips."""
        on = bool(on)
        was = self.is_aligned()
        if on != self._wants_aligned:
            self._wants_aligned = on
            QSettings().setValue(self._aligned_key, on)
        self._sync_orientation()
        if self.is_aligned() != was:
            self._request_visible()
            self.aligned_changed.emit(self.is_aligned())

    def set_alignment(self, note: str | None) -> None:
        """Whether this run can align its clips: ``None`` it can, else the reason it cannot —
        **Aligned** is greyed with it, and the tiles are drawn as recorded."""
        was = self.is_aligned()
        self._align_note = note
        self._sync_orientation()
        if self.is_aligned() != was:
            self._request_visible()
            self.aligned_changed.emit(self.is_aligned())

    def _sync_orientation(self) -> None:
        """**Show ▾** says the look and the orientation, each greyed with its reason."""
        self.show_button.sync(
            self._look,
            aligned=self.is_aligned(),
            align_note=self._align_note,
            align_tip=ALIGN_TIP,
            points_note=self._points_note,
            context_s=self._context_s,
        )

    def is_off(self, index: int) -> bool:
        """Whether clip ``index`` is in the state a click takes it out of — removed (``faded``)
        on the page, not picked under a click layer — so a drag that starts on one puts its run
        back."""
        if self._layer is not None:
            return bool(self._layer.is_off(index))
        return bool(self._clips[index].faded)

    # -- context ------------------------------------------------------------ #
    def context_s(self) -> float:
        """Seconds of footage either side of each clip, as **Show ▾** says."""
        return self._context_s

    def set_context_s(self, seconds: float) -> None:
        """Choose the context (kept under the page's key); the page widens its clips on
        ``context_changed``."""
        from detail.gui.show_menu import CONTEXT_CHOICES_S

        seconds = float(seconds)
        if seconds not in CONTEXT_CHOICES_S or seconds == self._context_s:
            self._sync_orientation()
            return
        self._context_s = seconds
        QSettings().setValue(self._context_key, seconds)
        self._sync_orientation()
        self.context_changed.emit(seconds)

    # -- blocked ------------------------------------------------------------ #
    def set_blocked(
        self,
        message: str,
        fraction: float,
        eta_s: float | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        """Put the progress bar in the grid's place: ``message``, ``fraction`` (0..1), time left.

        Called again as the work moves; ``on_cancel`` is offered on the first call. The clock
        stops until :meth:`clear_blocked`.
        """
        if self._overlay is None:
            # The bar stands in the grid's place, so the grid's own scrollbar goes with it.
            self.grid.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._overlay = BusyOverlay(self.grid, scrim_alpha=255)
            self._overlay.start(message)
            if on_cancel is not None:
                self._overlay.offer_cancel(on_cancel)
        text = message
        if eta_s is not None:
            text = f"{message} · about {format_duration(eta_s)} left"
        self._overlay.set_fraction(fraction, text)
        self._sync_clock()

    def is_blocked(self) -> bool:
        return self._overlay is not None

    def clear_blocked(self) -> None:
        """Back to the grid — the previews are there now."""
        if self._overlay is None:
            return
        self._overlay.stop()
        self._overlay.deleteLater()
        self._overlay = None
        self.grid.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._reset_pictures()
        self._sync_clock()

    # -- geometry ----------------------------------------------------------- #
    def _cell(self, tile_px: int | None = None) -> QSize:
        tile = self._tile_px if tile_px is None else tile_px
        caption = self.fontMetrics().height() + 2
        return QSize(tile, tile + _BAR_GAP_PX + BAR_PX + caption)

    def height_for_rows(self, rows: int) -> int:
        """The height a page reserves for the montage to hold ``rows`` rows of tiles: its
        header lines and controls row, then the grid — **at the default tile size**
        (:data:`TILE_PX`), never the slider's.

        The Size slider changes how many tiles fit in the area, never the area: measured at the
        slider's size, a page sizing its rows from this (INTERPRET's tree fit) grew the clips
        201px and shrank the tree to its floor as the slider went from min to max, and back
        again — the slider resized the page (2026-11-09, `test_laptop_layout`).
        """
        layout = self.layout()
        # A row with nothing shown is skipped by the layout, spacing and all.
        above = [
            r.sizeHint().height()
            for r in (
                self.header_lines,
                None if self._controls_detached else self._controls_line,
                self.above_grid,
            )
            if r is not None and not r.isEmpty()
        ]
        chrome = (
            sum(above)
            + len(above) * max(layout.spacing(), 0)
            + 2 * self.grid.frameWidth()
        )
        cell = self._cell(TILE_PX).height()
        return chrome + rows * (cell + _GAP)

    def _columns(self) -> int:
        width = self.grid.viewport().width()
        return max(1, (width + _GAP) // (self._tile_px + _GAP))

    def _left(self) -> int:
        """Where the first column starts: the columns are centred, so the width left over
        when no further tile fits is split either side rather than all of it at the right.
        """
        cols = self._columns()
        used = cols * self._tile_px + (cols - 1) * _GAP
        return max((self.grid.viewport().width() - used) // 2, 0)

    def _heading_px(self) -> int:
        """A section heading's band: one line of the bold heading font and a little air — or,
        drawn as a rule, room either side of it so the two sections stand apart."""
        line = QFontMetrics(emphasis_font(self.canvas, steps=0)).height()
        return line * 3 if self._rule_headings else line + 8

    def _runs(self) -> list[tuple[int, int, str, int]]:
        """``(first index, count, title, y of its first row)`` per section — one untitled run
        holding every clip when there are no sections."""
        sections = self._sections or [("", len(self._clips))]
        cols, step = self._columns(), self._cell().height() + _GAP
        heading = self._heading_px()
        runs, start, y = [], 0, 0
        for title, count in sections:
            first = y + (heading if title else 0)
            runs.append((start, count, title, first))
            start += count
            y = first + -(-count // cols) * step
        return runs

    def heading_rects(self) -> list[tuple[str, QRect]]:
        """Each section's title and where its heading is drawn."""
        heading, left = self._heading_px(), self._left()
        return [
            (
                title,
                QRect(left, first - heading, self.canvas.width() - left, heading),
            )
            for _start, _count, title, first in self._runs()
            if title
        ]

    def tile_rect(self, index: int) -> QRect:
        """Where tile ``index``'s picture and caption sit on the canvas."""
        cols, cell = self._columns(), self._cell()
        for start, count, _title, first in self._runs():
            if index < start + count:
                break
        row, col = divmod(index - start, cols)
        return QRect(
            self._left() + col * (cell.width() + _GAP),
            first + row * (cell.height() + _GAP),
            cell.width(),
            cell.height(),
        )

    def index_at(self, point) -> int | None:
        cols, cell = self._columns(), self._cell()
        x = point.x() - self._left()
        col = x // (cell.width() + _GAP)
        if col >= cols or x < 0 or point.y() < 0:
            return None
        step = cell.height() + _GAP
        for start, count, _title, first in self._runs():
            if point.y() < first:
                return None
            row = (point.y() - first) // step
            local = row * cols + col
            if local < -(-count // cols) * cols:
                if local >= count:
                    return None
                index = start + local
                return index if self.tile_rect(index).contains(point) else None
        return None

    def visible_indices(self) -> range:
        """The tiles inside the viewport — the only ones that play, paint or ask for pictures."""
        if not self._clips:
            return range(0)
        cols, step = self._columns(), self._cell().height() + _GAP
        top = self.grid.verticalScrollBar().value()
        bottom = top + self.grid.viewport().height()
        low, high = None, None
        for start, count, _title, first in self._runs():
            rows = -(-count // cols)
            first_row = max((top - first) // step, 0)
            last_row = min((bottom - 1 - first) // step, rows - 1)
            if count == 0 or last_row < first_row:
                continue
            a = start + first_row * cols
            b = min(start + (last_row + 1) * cols, start + count)
            low = a if low is None else low
            high = b
        if low is None:
            return range(0)
        return range(low, high)

    def _relayout(self) -> None:
        cols, cell = self._columns(), self._cell()
        height = 0
        if self._clips:
            _start, count, _title, first = self._runs()[-1]
            height = max(first + -(-count // cols) * (cell.height() + _GAP) - _GAP, 0)
        self.canvas.setFixedSize(self.grid.viewport().width(), height)
        self._request_visible()

    # -- the clock ---------------------------------------------------------- #
    def _sync_clock(self) -> None:
        if "_clock" not in self.__dict__:
            # A hide or an expose reaching a montage whose Python half is already released (its
            # window going, in a teardown): there is no clock left to stop.
            return
        playing = (
            bool(self._clips)
            and self._overlay is None
            and self.isVisible()
            and self.is_on_screen()
        )
        if playing and not self._clock.isActive():
            self._last_tick = None
            self._clock.start()
        elif not playing:
            self._clock.stop()

    def is_on_screen(self) -> bool:
        """Whether the window holding the montage can be seen: not minimised, and — where the
        platform says so (macOS reports a window wholly covered by others) — exposed.

        A hidden montage already stopped its clock (``hideEvent``); a minimised or covered
        window's montage is still "visible" to Qt and kept decoding every tile at :data:`FPS`,
        one decode thread per open cluster window, for nobody.
        """
        window = self.window()
        if window.isMinimized():
            return False
        handle = window.windowHandle()
        return handle is None or handle.isExposed()

    def _watch_window(self) -> None:
        """Follow the holding window's minimise and exposure (they reach it, not the montage)."""
        window = self.window()
        handle = window.windowHandle()
        watched = (window, handle)
        if watched == self._watched:
            return
        for old in self._watched:
            if old is not None and alive(old):
                old.removeEventFilter(self)
        self._watched = watched
        for new in watched:
            if new is not None:
                new.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 — Qt's name
        if event.type() in (QEvent.Type.WindowStateChange, QEvent.Type.Expose):
            self._sync_clock()
        return False

    def showEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().showEvent(event)
        self._watch_window()
        self._sync_clock()
        self._request_visible()

    def hideEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().hideEvent(event)
        self._sync_clock()

    def _tick(self) -> None:
        now = time.monotonic()
        step = 0 if self._last_tick is None else int((now - self._last_tick) * 1e9)
        self._last_tick = now
        self.advance(min(step, _MAX_STEP_NS))
        for index in self.visible_indices():
            self._show_frame(index)

    def advance(self, step_ns: int) -> None:
        """Move every visible tile's clock on by ``step_ns`` — all but the hovered one's paused
        while a tile is hovered. Each loops at its own clip's length."""
        for index in self.visible_indices():
            if self._hovered is not None and index != self._hovered:
                continue
            duration = max(int(self._clips[index].duration_ns), 1)
            self._offsets[index] = (self._offsets[index] + int(step_ns)) % duration
        self.canvas.update()

    def offset_ns(self, index: int) -> int:
        return self._offsets[index]

    def hovered(self) -> int | None:
        return self._hovered

    def hover_text(self, index: int) -> str:
        text = self._hover_text(index) if self._hover_text is not None else ""
        note = self._clips[index].note if 0 <= index < len(self._clips) else ""
        return "\n".join(t for t in (note, self.click_hint(index), text) if t)

    def _set_hovered(self, index: int | None) -> None:
        if index == self._hovered:
            return
        self._hovered = index
        self._update_preview()
        self.canvas.update()

    # -- pictures ----------------------------------------------------------- #
    def _reset_pictures(self) -> None:
        self._generation += 1
        self._images.clear()
        self._asked.clear()
        self._failed.clear()
        self._coming.clear()
        self._request_visible()
        self.pictures_changed.emit()

    def _scrolled(self, value: int) -> None:
        if value != self._scroll_at:
            self._scroll_down = value > self._scroll_at
            self._scroll_at = value
        self._request_visible()

    def lookahead_indices(self) -> list[int]:
        """The :data:`LOOKAHEAD_ROWS` rows of tiles just past the view in the direction it last
        scrolled, nearest first — asked for their first picture once the view has its own.
        """
        visible = self.visible_indices()
        if not visible:
            return []
        span = LOOKAHEAD_ROWS * self._columns()
        if self._scroll_down:
            return list(range(visible.stop, min(visible.stop + span, len(self._clips))))
        return list(range(visible.start - 1, max(visible.start - span, 0) - 1, -1))

    def _request_visible(self) -> None:
        """Ask for the tiles in view, then the lookahead, and drop every other request: a
        scroll, a resize or a size change re-decides what the picture worker serves next.
        """
        visible = self.visible_indices()
        ahead = self.lookahead_indices()
        for index in self._worker.focus(visible, ahead):
            self._asked.pop(index, None)
            self._coming.discard(index)
        for index in (*visible, *ahead):
            self._show_frame(index)
        self._note_seen()
        self.canvas.update()

    def _show_frame(self, index: int) -> None:
        """Ask for tile ``index``'s picture at its clock, unless that is what it already has."""
        if self._source is None or self._overlay is not None:
            return
        clip = self._clips[index]
        # Without masks the picture is the same whether the clip matches or not (the paint step
        # washes a miss out), so a filter change asks for nothing new.
        matching = clip.matching or not self._masks
        wanted = (self._offsets[index], self._tile_px, matching, self.look())
        if self._asked.get(index) == wanted:
            return
        self._asked[index] = wanted
        first = not (
            index in self._images or index in self._failed or index in self._coming
        )
        if first:
            self._coming.add(index)
        self._worker.post(index, (self._source, self._generation, *wanted), first=first)

    def _on_delivered(self, generation: int, index: int, _offset: int, image) -> None:
        if generation != self._generation or index >= len(self._clips):
            return
        self._coming.discard(index)
        before = (index in self._images, index in self._failed)
        if image is None:
            self._failed.add(index)
        else:
            self._failed.discard(index)
            self._images[index] = image
        self.canvas.update(self.tile_rect(index))
        if (index in self._images, index in self._failed) != before:
            self.pictures_changed.emit()

    def unpictured(self) -> tuple[int, int]:
        """``(waiting, failed)``: tiles whose first picture is asked for and not yet here, and
        tiles whose picture could not be made — they paint *no preview*. A surface that takes a
        decision about every clip (a review's Finish) waits for both to be zero: a clip whose
        picture never showed was not seen, and a decision about it is a decision against nothing
        (``tests/gui/test_frame_source.py``'s class).

        A tile never asked for — scrolled past in one jump — is in neither: how far the grid was
        looked through is :meth:`seen`'s question, not this one."""
        total = len(self._clips)
        waiting = sum(1 for index in self._coming if index < total)
        failed = sum(1 for index in self._failed if index < total)
        return waiting, failed

    # -- painting ----------------------------------------------------------- #
    def _paint(self, canvas: QWidget, dirty: QRect) -> None:
        theme = plot_theme.current()
        painter = QPainter(canvas)
        small = small_font(canvas)
        caption_ink = QColor(theme.text_secondary)
        if self._sections:
            painter.setFont(emphasis_font(canvas, steps=0))
            for title, rect in self.heading_rects():
                if not rect.intersects(dirty):
                    continue
                if self._rule_headings:
                    self._paint_rule(painter, rect, title, theme)
                    continue
                painter.setPen(QColor(theme.text))
                painter.drawText(
                    rect.adjusted(0, 0, 0, -4),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                    title,
                )
        clickable = self.is_clickable()
        layer = self._layer
        preview = self._preview
        tinted = set(preview.indices) if preview is not None else set()
        anchor = (
            self._last_clicked
            if preview is not None and preview.kind == "shift"
            else None
        )
        for index in self.visible_indices():
            cell = self.tile_rect(index)
            if not cell.intersects(dirty):
                continue
            clip = self._clips[index]
            picture = QRect(cell.x(), cell.y(), self._tile_px, self._tile_px)
            image = self._images.get(index)
            if image is not None:
                painter.drawImage(QRectF(picture), image)
                dropped = layer is not None and layer.dropped(index)
                if clip.faded or dropped or (not self._masks and not clip.matching):
                    scrim = QColor(theme.surface)
                    scrim.setAlpha(MISS_SCRIM_ALPHA)
                    painter.fillRect(picture, scrim)
            else:
                painter.fillRect(picture, QColor(theme.grid))
                painter.setPen(QColor(theme.text_muted))
                painter.drawText(
                    picture,
                    Qt.AlignmentFlag.AlignCenter,
                    "no preview" if index in self._failed else "…",
                )
            if index in tinted:
                self._paint_preview(painter, picture, preview.put_back)
            if clip.tag:
                self._paint_badge(painter, small, picture, clip.tag)
            if clip.crossed:
                self._paint_cross(painter, index, cue=False)
            elif (
                clickable
                and layer is None
                and self._marks_runs
                and index == self._hovered
                and not self.is_off(index)
            ):
                # The cue is "a click removes this". A click-only grid (CALIBRATE) chooses a
                # clip instead, so it draws none.
                self._paint_cross(painter, index, cue=True)
            number = layer.number(index) if layer is not None else None
            if number is not None:
                self._paint_pick(painter, small, picture, number)
            framed = self._filtered and clip.matching
            if framed:
                pen = QPen(match_colours()[0], FRAME_PX)
                pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                inset = FRAME_PX / 2
                painter.drawRect(QRectF(picture).adjusted(inset, inset, -inset, -inset))
            duration = max(int(clip.duration_ns), 1)
            track, fill = progress_rects(picture, self._offsets[index] / duration)
            if index == self._hovered:
                # In the theme's ink on its grid colour: a summary, not a control (the handle
                # colour is for those) — and on the page, not the footage.
                painter.fillRect(track, QColor(theme.grid))
                painter.fillRect(fill, QColor(theme.summary))
            painter.setFont(small)
            painter.setPen(caption_ink)
            text_rect = QRect(
                cell.x(),
                track.bottom() + 1,
                cell.width(),
                cell.bottom() - track.bottom(),
            )
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                painter.fontMetrics().elidedText(
                    clip.caption, Qt.TextElideMode.ElideRight, cell.width()
                ),
            )
            if index == anchor:
                pen = QPen(QColor(theme.text), ANCHOR_RING_PX)
                pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                half = ANCHOR_RING_PX / 2
                painter.drawRect(QRectF(picture).adjusted(half, half, -half, -half))
            elif index == self._hovered and index not in tinted:
                # Inside the match frame when there is one, so hovering hides neither. Not on a
                # previewed tile: its own ring says it is in the run, and an ink ring there read
                # as a second anchor.
                ring = 1 + (FRAME_PX if framed else 0)
                painter.setPen(QPen(QColor(theme.text), 2))
                painter.drawRect(picture.adjusted(ring, ring, -ring, -ring))
        painter.end()

    def rule_rect(self, rect: QRect) -> QRect:
        """Where a rule heading's line runs inside its heading ``rect``: across the columns, at
        its middle."""
        cols = self._columns()
        right = self._left() + cols * (self._tile_px + _GAP) - _GAP
        y = rect.center().y()
        return QRect(self._left(), y - RULE_PX // 2, right - self._left(), RULE_PX)

    def _paint_rule(self, painter, rect: QRect, title: str, theme) -> None:
        """A dashed line across the grid, broken in the middle by its title."""
        line = self.rule_rect(rect)
        y = line.center().y()
        pen = QPen(QColor(theme.handle), RULE_PX, Qt.PenStyle.DashLine)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.drawLine(line.left(), y, line.right(), y)
        metrics = painter.fontMetrics()
        text = metrics.elidedText(title, Qt.TextElideMode.ElideRight, line.width())
        width = metrics.horizontalAdvance(text) + 2 * metrics.height()
        label = QRect(
            line.center().x() - width // 2,
            y - metrics.height() // 2 - 3,
            width,
            metrics.height() + 6,
        )
        painter.fillRect(label, QColor(theme.surface))
        painter.setPen(QColor(theme.text))
        painter.drawText(label, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_preview(self, painter, picture: QRect, put_back: bool) -> None:
        """A run preview on one picture: a translucent tint and a ring inside its edge."""
        colour = preview_colour(put_back)
        tint = QColor(colour)
        tint.setAlpha(PREVIEW_ALPHA)
        painter.fillRect(picture, tint)
        pen = QPen(colour, PREVIEW_RING_PX)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        half = PREVIEW_RING_PX / 2
        painter.drawRect(QRectF(picture).adjusted(half, half, -half, -half))

    def _paint_cross(self, painter, index: int, *, cue: bool) -> None:
        """The ✕ in the picture's top-right: bold on a box of the page's surface colour for a
        removed clip; faint, the hovered tile's cue, for one a click would remove."""
        theme = plot_theme.current()
        rect = QRectF(self.cross_rect(index))
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = QColor(theme.surface)
        ink = QColor(theme.text)
        if cue:
            box.setAlpha(CUE_ALPHA)
            ink.setAlpha(CUE_ALPHA + 60)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(box)
        painter.drawRoundedRect(rect, 3, 3)
        pen = QPen(ink, 2.5 if not cue else 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        inner = rect.adjusted(5, 5, -5, -5)
        painter.drawLine(inner.topLeft(), inner.bottomRight())
        painter.drawLine(inner.topRight(), inner.bottomLeft())
        painter.restore()

    def pick_rect(self, index: int, number: int) -> QRect:
        """Where tile ``index``'s pick number sits: the picture's top-left corner."""
        cell = self.tile_rect(index)
        metrics = QFontMetrics(small_font(self.canvas))
        text = str(number)
        side = metrics.height() + 2 * PICK_PAD
        width = max(side, metrics.horizontalAdvance(text) + 2 * PICK_PAD + 2)
        return QRect(cell.x() + CROSS_INSET, cell.y() + CROSS_INSET, width, side)

    def _paint_pick(self, painter, font, picture: QRect, number: int) -> None:
        """The clip's place in the export order: ink on a rounded box of the page's colour,
        clear of the ✕ in the top-right and the tag in the bottom-left."""
        theme = plot_theme.current()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text = str(number)
        side = metrics.height() + 2 * PICK_PAD
        width = max(side, metrics.horizontalAdvance(text) + 2 * PICK_PAD + 2)
        box = QRectF(picture.x() + CROSS_INSET, picture.y() + CROSS_INSET, width, side)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.text))
        painter.drawRoundedRect(box, side / 2, side / 2)
        painter.setPen(QColor(theme.surface))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def _paint_badge(self, painter, font, picture: QRect, text: str) -> None:
        """``text`` on a box of the page's surface colour, in the picture's bottom-left corner,
        clear of the ✕ in its top-right."""
        theme = plot_theme.current()
        painter.setFont(font)
        metrics = painter.fontMetrics()
        pad = 3
        left = picture.x() + 2
        width = min(
            metrics.horizontalAdvance(text) + 2 * pad, picture.right() - 2 - left
        )
        height = metrics.height() + 2
        box = QRect(left, picture.bottom() - 1 - height, max(width, 0), height)
        painter.fillRect(box, QColor(theme.surface))
        painter.setPen(QColor(theme.text))
        painter.drawText(
            box,
            Qt.AlignmentFlag.AlignCenter,
            metrics.elidedText(text, Qt.TextElideMode.ElideRight, width - 2),
        )
