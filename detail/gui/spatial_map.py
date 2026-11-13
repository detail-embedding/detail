"""Where a behavior happens, over the field of view (§19 ``instance_features.csv``).

A check for where behavior happens, in one widget shared by the surfaces that show
behavior instances. The pose embedding carries no position and is not egocentric, so a cluster
that is really "the corner where the video is bad" is invisible on every other screen: the
scatter groups it, the ethogram times it, the frames show it, and none of them says *where*.

**One square per recording, in the ethogram's order.** The strip above answers *when* with one
row per recording; this answers *where* with one square per recording, in the same order, so the
two read as one picture. It also refuses to pretend two arenas are the same shape — a single
normalized overlay would average away exactly the per-box difference somebody is looking for.
A hundred recordings is 16,500px of squares, so the row **scrolls** rather than pooling them:
this used to overlay every square into one above six, which destroys the comparison the panel
exists for at exactly the scale it matters.

**Points, until points stop being readable.** Past :data:`DENSITY_ABOVE` instances a scatter is a
solid blob that answers nothing, so it becomes a 2-D histogram — again stated, because "more
colour means many" and "each dot is one" are different pictures.

Drawn with QPainter rather than pyqtgraph: this is a picture of a box, not a plot of a quantity,
and it needs no axes, ticks or units. That also keeps it out of the wheel-stealing class
``gui/scroll.py`` exists to contain — nothing here takes the wheel, so the page still scrolls
under it.
"""

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from detail.gui import plot_theme
from detail.gui.behavior_query import ClipSpec, _ClipPlayer
from detail.gui.grip_splitter import GripSplitter
from detail.gui.plots import bound_axes
from detail.gui.scroll import OWNS_WHEEL
from detail.gui.sizing import lines, remember_size, remembered_size
from detail.gui.status_color import MUTED, WARN, set_label_color

#: Above this many points a scatter is a blob; draw density instead.
DENSITY_ABOVE = 2000

#: How far the squares can be dragged, in lines of the interface font. The floor is where a dot
#: stops being placeable inside the box; the ceiling is where one recording fills the screen and
#: the comparison this panel exists for is gone.
#: Space between the canvas edge and the frame's own boundary, in lines. `plots.bound_axes`
#: gives every pyqtgraph axis this slack and says why — "a symbol sitting exactly on the
#: boundary would otherwise be drawn half outside the frame". Normalized position is closed at
#: both ends on both axes, so the same applies here; these squares are hand-painted and never
#: got it, which is why the frame read as the widget edge and a dot at cx=0 sat on top of it.
PAD_LINES = 0.4
#: The established minimum width for a clip player on these pages (`analysis_query.py`).
PLAYER_PX = 460

MIN_SIDE_LINES = 7
MAX_SIDE_LINES = 34

#: How concentrated a distribution has to be before the caption says so. Two thirds in one third
#: of the box is well past what a uniform distribution produces and is the shape a reader would
#: want to notice — "everything is in the bottom left corner", in the review's words.
CONCENTRATION_FRACTION = 2 / 3

_BINS = 24  # density grid per axis; ~1 cell per 40px at the size a square is drawn


def frame_aspect(recording) -> float:
    """A recording's frame width / height, or 1.0 when no video has been imported.

    One function because two surfaces need the same number and a spatial map drawn at the wrong
    aspect is wrong *plausibly*: every dot is somewhere legal, the box is a box, and the only
    signal is that a centred animal appears to reach the top and bottom of a 16:9 arena. Reading
    it per surface is how the two would drift.

    Cheap per recording — `video_resolution` goes through the mtime-cached `read_recording` —
    but it is still a `recording.yaml` parse the first time, so callers gather it in the same
    off-thread sweep as the outlines rather than on the UI thread.
    """
    try:
        resolution = recording.video_resolution
    except (FileNotFoundError, ValueError, KeyError):
        return 1.0
    if not resolution:
        return 1.0
    width, height = resolution
    return float(width) / float(height) if height else 1.0


def _clock(t_ns: int) -> str:
    """Where in the recording something happens — a position, always ``h:mm:ss``."""
    total = int(t_ns // 1_000_000_000)
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


class _Square(QWidget):
    #: Clicked, with the click's position normalized into the field of view. The panel turns
    #: that into the nearest instance, which is the one it plays.
    clicked = pyqtSignal(float, float)

    """One recording's field of view, with its instances plotted in it."""

    #: A square big enough to read a cloud in, and small enough that six fit on a laptop.
    #: Fixed rather than expanding: a lone square stretched across a 1350px band is 1200px of
    #: dead space with a picture in the middle of it, and the field of view is not the band.
    SIDE_LINES = 11

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._points = np.zeros((0, 2), dtype=float)
        self._outline: list[tuple[float, float]] = []
        #: The frame's width/height. 1.0 until told otherwise, which is only right for a square
        #: video — the callers all pass the real one.
        self._aspect = 1.0
        #: The instances' colour, or ``None`` for the theme's first identity colour.
        self._colour: QColor | None = None
        self.setFixedSize(lines(self.SIDE_LINES), lines(self.SIDE_LINES))

    def set_side(self, side_px: int) -> None:
        """Resize the square. Points are normalized, so growing it is a redraw — never a
        re-read of anything."""
        self.setFixedSize(side_px, side_px)

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        """Report the click in field-of-view coordinates, so the panel can find what is under it.

        Position rather than a bare "I was clicked": a dot is a few pixels wide and they
        overlap, so the panel plays the **nearest** instance rather than requiring a hit. That
        also makes the rule the same in density mode, where there are no dots to hit at all.
        """
        box = self._box()
        if box.width() <= 0 or box.height() <= 0:
            return
        point = event.position()
        self.clicked.emit(
            (point.x() - box.left()) / box.width(),
            (point.y() - box.top()) / box.height(),
        )

    def set_data(self, points, outline, colour, aspect=1.0) -> None:
        self._points = np.asarray(points, dtype=float).reshape(-1, 2)
        self._outline = list(outline or [])
        self._aspect = float(aspect) if aspect and aspect > 0 else 1.0
        if colour is not None:
            self._colour = QColor(*colour)
        self.update()

    def paintEvent(self, _event) -> None:
        theme = plot_theme.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = self._box()
        # The **canvas** is the chart surface, not just the field of view inside it. Filling
        # only `box` put the boundary line exactly on the canvas/background edge, where on a
        # dark theme it is indistinguishable from the edge itself — the outline was drawn every
        # time and could not be seen. The padding has to be *inside* the canvas for the line to
        # sit on something.
        painter.fillRect(self.rect(), QColor(theme.surface))
        # The axis ink, not a faint grey: this line *is* the field of view — or the frame, when
        # no field of view was drawn — and a boundary you cannot see makes every dot a
        # coordinate rather than a place.
        # Width 2, not 1. A 1px path on an integer coordinate straddles two pixel columns, and
        # antialiasing splits the ink between them: measured at lightness 142 on white — a grey
        # line where a near-black one was asked for. Two pixels cover a column each, fully.
        painter.setPen(QPen(QColor(theme.axis), 2))
        if self._outline:
            painter.drawPolygon(
                QPolygonF([QPointF(*self._to_px(x, y, box)) for x, y in self._outline])
            )
        else:
            # No arena drawn on this recording: the frame *is* the field of view, and saying
            # so with the same rectangle is honest — the coordinates are normalized to it.
            painter.drawRect(box)
        if len(self._points):
            if len(self._points) > DENSITY_ABOVE:
                self._paint_density(painter, box)
            else:
                self._paint_points(painter, box)
        painter.end()

    def _instance_colour(self) -> QColor:
        """A fresh copy of the instances' colour, so setting its alpha changes nothing shared."""
        if self._colour is None:
            return QColor(plot_theme.current().identity(0))
        return QColor(self._colour)

    def _box(self) -> QRectF:
        """The largest centred rectangle **of the frame's shape** that fits, inset by the pad.

        Not a square. Coordinates are normalized per axis, so `cx` is a fraction of the width
        and `cy` a fraction of the height — drawing them in a square stretches a 16:9 frame by
        1.78x vertically, which moves every point in it and makes a centred animal look like it
        reaches the top and bottom of the arena. The widget stays square, so the panels tile
        evenly; the field of view letterboxes inside it, on the canvas the boundary is drawn
        against.
        """
        pad = lines(PAD_LINES)
        avail_w = max(self.width() - 2 * pad, 1.0)
        avail_h = max(self.height() - 2 * pad, 1.0)
        width = min(avail_w, avail_h * self._aspect)
        height = width / self._aspect
        return QRectF(
            (self.width() - width) / 2, (self.height() - height) / 2, width, height
        )

    @staticmethod
    def _to_px(x: float, y: float, box: QRectF) -> tuple[float, float]:
        return (box.left() + x * box.width(), box.top() + y * box.height())

    def _paint_points(self, painter: QPainter, box: QRectF) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        colour = self._instance_colour()
        # Semi-transparent so overlapping bouts read as denser rather than as one dot: the
        # difference between "it happens here" and "it happens here a lot".
        colour.setAlpha(150)
        painter.setBrush(QBrush(colour))
        radius = 2.0
        for x, y in self._points:
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            px, py = self._to_px(float(x), float(y), box)
            painter.drawEllipse(QPointF(px, py), radius, radius)

    def _paint_density(self, painter: QPainter, box: QRectF) -> None:
        grid = density_grid(self._points, _BINS)
        peak = grid.max()
        if peak <= 0:
            return
        painter.setPen(Qt.PenStyle.NoPen)
        cell_w, cell_h = box.width() / _BINS, box.height() / _BINS
        for ix in range(_BINS):
            for iy in range(_BINS):
                weight = grid[iy, ix] / peak
                if weight <= 0:
                    continue
                colour = self._instance_colour()
                colour.setAlpha(int(30 + 210 * weight))
                painter.fillRect(
                    QRectF(
                        box.left() + ix * cell_w,
                        box.top() + iy * cell_h,
                        cell_w + 0.5,
                        cell_h + 0.5,
                    ),
                    colour,
                )


def density_grid(points, bins: int) -> np.ndarray:
    """``(bins, bins)`` counts over the unit square, row 0 at the top (screen order)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if not len(pts):
        return np.zeros((bins, bins))
    grid, _, _ = np.histogram2d(
        pts[:, 1], pts[:, 0], bins=bins, range=[[0.0, 1.0], [0.0, 1.0]]
    )
    return grid


def _worst_third(points) -> tuple[str, float]:
    """The most crowded of the four thirds, and its share. ``("", 0.0)`` below a usable count."""
    pts = np.asarray(list(points), dtype=float).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 10:  # below this a "share" is one or two bouts moving
        return ("", 0.0)
    thirds = {
        "left third": float((pts[:, 0] < 1 / 3).mean()),
        "right third": float((pts[:, 0] > 2 / 3).mean()),
        "top third": float((pts[:, 1] < 1 / 3).mean()),
        "bottom third": float((pts[:, 1] > 2 / 3).mean()),
    }
    return max(thirds.items(), key=lambda kv: kv[1])


def concentration_note(by_recording: dict) -> str:
    """A sentence when the distribution is lopsided, or ``""``.

    Computed rather than left to the eye: noticing that two thirds of a cloud sits in one third
    of the box is exactly what a reader skims past, and it is the finding this widget exists for.

    **Checked per recording as well as over the pool, and the worst one is what gets reported.**
    A pooled check alone misses the case the review actually described — one recording where the
    video is bad in a corner — because the recordings that are fine dilute it below any threshold.
    One plainly left-piled recording beside two uniform ones passes a pooled check,
    while the offender alone fails it.
    """
    pooled = np.concatenate(
        [np.asarray(list(v), dtype=float).reshape(-1, 2) for v in by_recording.values()]
        or [np.zeros((0, 2))]
    )
    where, share = _worst_third(pooled)
    culprit = None
    if len(by_recording) > 1:
        for recording_id, points in by_recording.items():
            one_where, one_share = _worst_third(points)
            if one_share > share:
                where, share, culprit = one_where, one_share, recording_id
    if not where or share < CONCENTRATION_FRACTION:
        return ""
    subject = f"{culprit}: " if culprit else ""
    return f"⚠ {subject}{round(100 * share)}% fall in the {where}"


class SpatialMap(QWidget):
    #: An instance was clicked, as its placement dict. The panel has already played it; the
    #: signal is for a host that wants to react (select the row, say), not for playback.
    instance_picked = pyqtSignal(object)

    """The panel: one square per recording, a caption, and a concentration warning.

    Takes points already reduced to one per instance (§19's ``instance_features.csv``), so it
    knows nothing about bouts, clusters or cutoffs — every surface hands it the same shape and
    colours it however that surface colours things.
    """

    def __init__(self, experiment, parent=None) -> None:
        super().__init__(parent)
        self._experiment = experiment
        #: recording id -> the embedding run its clips play through (§6.5: per recording).
        self._embed_run_ids: dict[str, str] = {}
        self._placements: dict[str, list] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # A hundred recordings is 16,500px of squares, so the row scrolls rather than pooling
        # them into one box — an overlay mode used to do that above six, which destroyed the
        # comparison this panel exists for at exactly the scale it matters. Horizontal only:
        # `PageScroll` asks the *vertical* bar's range before taking the wheel, so a row that
        # scrolls sideways still lets the page scroll under it.
        self._strip = QWidget()
        self._row = QHBoxLayout(self._strip)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(lines(0.5))
        self._row.addStretch()  # squares pack left; the slack goes after them, not into them
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setWidget(self._strip)

        left = QWidget()
        left_box = QVBoxLayout(left)
        left_box.setContentsMargins(0, 0, 0, 0)
        left_box.addWidget(self._scroll)
        self._caption = QLabel("")
        self._caption.setWordWrap(True)
        set_label_color(self._caption, MUTED)
        left_box.addWidget(self._caption)
        self._note = QLabel("")
        self._note.setWordWrap(True)
        set_label_color(self._note, WARN)
        left_box.addWidget(self._note)

        #: The dot you click plays here, beside the squares rather than in a strip somewhere
        #: else: that strip shows the most typical members capped at 400 while the map draws
        #: every instance, so the outlier worth clicking was the one most likely to be missing.
        #: It also shows the *whole frame* where the strip shows a crop, and for a map — is this
        #: the corner where the video is bad? — the frame is the answer and the crop cannot be.
        self._player = _ClipPlayer(
            experiment, masked=True, autoplay=True, loop=False, context_s=0
        )
        #: What you are looking at, over the video. An instance is a *run* of frames in the same
        #: cluster, and the run can be one frame or several seconds — so a click that produced a
        #: still and a click that produced a second of footage looked like two different bugs
        #: rather than two lengths of the same thing.
        self._played = QLabel("")
        set_label_color(self._played, MUTED)
        self._played.setWordWrap(True)
        right = QWidget()
        right_box = QVBoxLayout(right)
        right_box.setContentsMargins(0, 0, 0, 0)
        right_box.addWidget(self._played)
        right_box.addWidget(self._player, 1)
        # Neither side folds: the squares are the page, and the video is what they open.
        split = GripSplitter(Qt.Orientation.Horizontal, key="spatial_map")
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 2)  # two thirds squares, one third video
        split.setStretchFactor(1, 1)
        split.remember()
        root.addWidget(split, 1)

        # The player has nothing until a dot is clicked, and an empty pane beside a full row of
        # squares reads as broken rather than as waiting. `show_message` is the seam for the
        # "no clip yet" state, as against a clip that failed to decode.
        self._player.show_message("Click an instance to play it")
        self._played.setText("")

        self._panels: list[tuple[QWidget, _Square, QLabel]] = []
        #: Which recording each panel is currently showing, so a click resolves to the right
        #: one after a redraw reorders or reuses the squares.
        self._titles: list[str] = []
        self._side_px = remembered_size("map/squares", lines(_Square.SIDE_LINES))

    def resize_by(self, delta: int) -> None:
        """Drag down for bigger squares, up for smaller — the band's handle calls this.

        Bigger squares trade how many recordings fit against how well a corner bias reads, and
        at a hundred recordings against how precisely a dot can be hit.
        """
        self.set_side(self._side_px + delta)
        remember_size("map/squares", self._side_px)

    def _apply_side(self) -> None:
        """Push the current size onto every square, and give the scroll area a height to match.

        A row inside a scroll area has to be *told* how tall to be — the strip does the same
        thing one band up, where the extra pixels are its play controls; here they are the
        recording name under each square plus the horizontal scrollbar.
        """
        for _holder, square, _label in self._panels:
            square.set_side(self._side_px)
        # The square, the recording name under it, **and the horizontal scrollbar**. Short by
        # those few pixels the content overflows vertically, and a hidden vertical bar still
        # reports a range — which is what `PageScroll` asks before giving up the wheel, so the
        # row would silently stop the page scrolling through it.
        bar = self._scroll.horizontalScrollBar().sizeHint().height()
        self._scroll.setFixedHeight(self._side_px + lines(2) + bar)

    def set_side(self, side_px: int) -> None:
        side_px = max(lines(MIN_SIDE_LINES), min(int(side_px), lines(MAX_SIDE_LINES)))
        if side_px == self._side_px:
            return
        self._side_px = side_px
        self._apply_side()

    def _on_square_clicked(self, index: int, x: float, y: float) -> None:
        """Play the instance nearest the click, in the recording whose square was clicked."""
        if not (0 <= index < len(self._titles)):
            return
        recording_id = self._titles[index]
        here = self._placements.get(recording_id) or []
        if not here:
            return
        nearest = min(here, key=lambda p: (p["cx"] - x) ** 2 + (p["cy"] - y) ** 2)
        self._player.set_clip(
            ClipSpec(
                recording_id=recording_id,
                track_id=nearest["subject"],
                embed_run_id=self._embed_run_ids.get(recording_id, ""),
                start_t_ns=int(nearest["t_ns"]),
                end_t_ns=int(nearest["end_t_ns"]),
                label=f"{recording_id} {nearest['t_ns'] / 1e9:.2f}s",
            )
        )
        self._played.setText(self._describe(recording_id, nearest))
        self.instance_picked.emit(nearest)

    def _describe(self, recording_id: str, placement: dict) -> str:
        """The clip's own sentence: which recording, where in it, and how long it lasted.

        The duration is the part that had to be said. A pose instance is a maximal run of
        consecutive frames in one cluster, so it is a single frame about as often as it is three
        seconds — and the short ones handed the player a **zero-length** span
        (the removed `pipeline/pose_clustering.py` built `end_t_ns` as `t_ns[end]`, which for a
        one-row run is `t_ns[start]`, and saved runs still carry those spans). What that rendered as
        was a frame counter reading `0 / 0`, which does not read as "one frame", it reads as broken.
        """
        first, last = self._player.bounds()
        frames = max(last - first + 1, 1)
        span_ns = int(placement["end_t_ns"]) - int(placement["t_ns"])
        if span_ns <= 0:
            held = "held a single frame"
        elif frames > 1:
            held = f"held {span_ns / 1e9:.2f} s ({frames:,} frames)"
        else:
            # The player could not resolve the span to frames — this recording's video is not
            # readable from here. Say the duration, which is a fact about the instance, rather
            # than "(1 frames)", which is a fact about the failure.
            held = f"held {span_ns / 1e9:.2f} s"
        return f"{recording_id} · at {_clock(int(placement['t_ns']))} · {held}"

    def show_unavailable(self, why: str) -> None:
        """No positions for this run at all, and what to do about it.

        Distinct from "this cluster has no points", which the caller renders by hiding the band:
        that is an absence, this is a fixable state, and a user shown neither cannot tell the
        difference between "nothing happened here" and "nothing was recorded".
        """
        for holder, _square, _label in self._panels:
            holder.setVisible(False)
        self._caption.setText(why)
        self._note.setText("")

    def show_instances(
        self,
        by_recording: dict,
        *,
        outlines: dict,
        aspects: dict,
        embed_run_ids: dict[str, str],
        colour=None,
    ) -> None:
        """``{recording_id: [placement, ...]}`` in experiment order — one square per recording.

        A placement is §19's ``instance_features`` row joined to its instance: ``cx``, ``cy``,
        ``subject``, ``t_ns``, ``end_t_ns``. Coordinates alone were enough while the panel only
        drew; a dot you can click has to be turnable back into the clip it stands for.

        ``outlines``, ``aspects`` and ``embed_run_ids`` are **required**. They are facts about the
        recording, not choices about this call, and each one shipped optional-with-a-default and
        silently skipped one of the two surfaces for weeks: without them the picture is not
        plainer, it is wrong — an unlabelled box, a 16:9 arena drawn square, a player that plays
        nothing. ``colour`` keeps its default because it genuinely is a choice.
        """
        self._placements = {rid: list(v) for rid, v in by_recording.items()}
        # Per recording: a run reads each recording's own embedding run, and those share
        # settings, not an id (§6.5).
        self._embed_run_ids = dict(embed_run_ids)
        # The label names the *last click*, so a redraw must not leave it describing an instance
        # from a cutoff or granularity nobody is looking at any more.
        self._played.setText("")
        self._player.show_message("Click an instance to play it")
        recordings = list(by_recording)
        panels = [
            (rid, _arr([(p["cx"], p["cy"]) for p in by_recording[rid]]))
            for rid in recordings
        ]
        total = sum(len(points) for _rid, points in panels)
        self._rebuild(len(panels))
        self._titles = [title for title, _points in panels]
        for (_holder, square, label), (title, points) in zip(self._panels, panels):
            square.setCursor(Qt.CursorShape.PointingHandCursor)
            square.set_data(
                points, outlines.get(title), colour, aspects.get(title, 1.0)
            )
            # Named, because square N and ethogram row N are the same recording and a reader
            # has to be able to say which is which — a tooltip cannot be seen at a glance.
            label.setText(f"{title} · {len(points):,}")
            square.setToolTip(f"{title} — {len(points):,} instances")
        self._apply_side()

        if not total:
            self._caption.setText("(nothing to place — no instances at this setting)")
            self._note.setText("")
            return
        # **Per square, the way the painting decides it.** This asked the pooled total, so a
        # hundred recordings of fifty instances each said "density" over squares that were
        # drawing fifty dots apiece — a caption describing a picture that was not on screen.
        densest = max((len(points) for _rid, points in panels), default=0)
        how = (
            "density, so a click plays the nearest instance"
            if densest > DENSITY_ABOVE
            else "one dot per instance · click one to play it"
        )
        scope = f"one square per recording ({len(recordings)}), in experiment order"
        # What an *instance* is, once, where the picture is. Without it the varying clip lengths
        # read as the surface behaving inconsistently rather than as the data being uneven.
        what = (
            "an instance is a run of frames in the same group, from a single frame to "
            "several seconds"
        )
        self._caption.setText(f"{total:,} instances · {scope} · {how} · {what}")
        # Per recording, not just pooled: a single bad recording is the case this exists for.
        self._note.setText(concentration_note({t: p for t, p in panels}))
        self._note.setVisible(bool(self._note.text()))

    def _rebuild(self, count: int) -> None:
        """Grow or shrink to ``count`` panels, keeping the trailing stretch last."""
        while len(self._panels) > count:
            holder, _square, _label = self._panels.pop()
            self._row.removeWidget(holder)
            holder.setParent(None)
            holder.deleteLater()
        while len(self._panels) < count:
            holder = QWidget()
            column = QVBoxLayout(holder)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(2)
            square = _Square()
            # By index, not by capturing a recording id: panels are reused across redraws, so
            # square N holds whichever recording is N'th *now*. Capturing at construction would
            # open the recording that used to be there.
            square.clicked.connect(
                lambda x, y, i=len(self._panels): self._on_square_clicked(i, x, y)
            )
            label = QLabel("")
            set_label_color(label, MUTED)
            column.addWidget(square)
            column.addWidget(label)
            holder.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            # Before the stretch, so the squares stay packed to the left.
            self._row.insertWidget(len(self._panels), holder)
            self._panels.append((holder, square, label))


def _arr(points) -> np.ndarray:
    return np.asarray(list(points), dtype=float).reshape(-1, 2)
