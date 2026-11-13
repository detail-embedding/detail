"""In-GUI preview of a detection run: a frame viewer that decodes the run's
source frame on demand and overlays detection boxes (no annotated video written).

Built for long videos: a detections-per-frame overview strip shows where the
action is and jumps the viewer; decode runs on a background worker (latest frame
only, via the shared ``FrameScrubber``), so the UI never stalls on an hour-long file.
"""

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from detail.api.run_preview import locate_frame, track_coverage
from detail.core.chart_theme import LIGHT, rgb
from detail.core.visualize import (
    BOX_ONLY,
    DETECTION_BOX_COLOR,
    MASK_BOX_CONF,
    PREVIEW_STYLE_LABELS,
    PREVIEW_STYLES,
    draw_preview_overlay,
    preview_needs_mask,
)
from detail.gui import frame_source
from detail.gui.busy import decode_overlay, run_busy
from detail.gui.dialog import Dialog
from detail.gui.image_view import ImageView, PictureWithControls
from detail.gui import plot_theme
from detail.gui.plots import bound_axes
from detail.gui.scrubber import FrameScrubber
from detail.gui.sizing import lines
from detail.gui.spin import FRACTION, FloatSpin


def _track_color(object_id):
    """A track's identity colour — on the frame, in the scatter and in the timeline alike.

    Canonical (light) steps, because the same colour tints the track's box on a video frame,
    which does not change with the application's theme. Past six tracks the theme's frozen
    extension, rather than wrapping so track 6 wore track 0's colour.
    """
    if object_id is None:
        return DETECTION_BOX_COLOR
    return rgb(LIGHT.identity(object_id))


def _to_pixmap(rgb: np.ndarray) -> QPixmap:
    h, w, _ = rgb.shape
    return QPixmap.fromImage(
        QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    )


def _frame_x(frame: int, total: int, left: int, plot_w: int) -> int:
    """Where ``frame`` is drawn on a timeline whose plot starts at ``left`` and is ``plot_w``
    wide.

    One mapping for both strips on a tracking run's page, with :func:`_x_frame` its inverse.
    Stacked timelines read as one axis — you look down from a spike in one to the other — so
    each strip working the position out for itself was a coincidence waiting to break, and it
    broke: the lost-frames timeline reserved a gutter for its track labels that the count
    strip above it did not, and the two disagreed about every frame by exactly that gutter.

    Spans ``plot_w - 1``, not ``plot_w``: the last frame at ``left + plot_w`` is one pixel
    past the widget, so the playhead vanished on exactly the frame a run ends on.
    """
    return left + int(frame / max(total - 1, 1) * max(plot_w - 1, 0))


def _x_frame(x: float, total: int, left: int, plot_w: int) -> int:
    """The frame under pixel ``x`` — :func:`_frame_x`'s inverse, clamped to the run."""
    frame = int((x - left) / max(plot_w, 1) * total)
    return min(max(frame, 0), total - 1)


class _CountStrip(QWidget):
    """A detections-per-frame bar strip; click to jump. Shows a frame cursor.

    ``set_left`` insets the plot to line up with a timeline stacked under it that keeps a
    label gutter (:attr:`_TrackTimeline.gutter`).
    """

    clicked = pyqtSignal(int)

    def __init__(self, counts: np.ndarray):
        super().__init__()
        self._counts = counts
        self._cursor = 0
        self._left = 0
        self.setFixedHeight(44)
        self.setMinimumWidth(200)

    def set_cursor(self, frame: int) -> None:
        self._cursor = frame
        self.update()

    def set_left(self, left: int) -> None:
        """Start the plot ``left`` pixels in, so it shares an axis with the strip below."""
        self._left = left
        self.update()

    def _plot_width(self) -> int:
        return max(self.width() - self._left, 1)

    def x_for_frame(self, frame: int) -> int:
        return _frame_x(frame, len(self._counts), self._left, self._plot_width())

    def mousePressEvent(self, event) -> None:
        n = len(self._counts)
        if n and self.width():
            self.clicked.emit(
                _x_frame(event.position().x(), n, self._left, self._plot_width())
            )

    def paintEvent(self, _event) -> None:
        theme = plot_theme.current()
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(theme.surface))
        n = len(self._counts)
        if not n:
            return
        peak = max(int(self._counts.max()), 1)
        painter.setPen(QColor(theme.identity(0)))
        plot_w = self._plot_width()
        for x in range(plot_w):
            lo = int(x / plot_w * n)
            hi = max(int((x + 1) / plot_w * n), lo + 1)
            value = int(self._counts[lo:hi].max())
            bar = int(value / peak * (h - 4))
            painter.drawLine(self._left + x, h - 2, self._left + x, h - 2 - bar)
        cursor_x = self.x_for_frame(self._cursor)
        # The playhead in ink, like every timeline's: a position, not a series.
        painter.setPen(QColor(theme.text))
        painter.drawLine(cursor_x, 0, cursor_x, h)


def _present_runs(mask: np.ndarray):
    """Yield ``(start, length)`` for each maximal run of True in a 1-D bool array,
    so a row of present columns paints as one rect per run instead of per pixel."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    for start, end in zip(starts, ends):
        yield int(start), int(end - start)


class _TrackTimeline(QWidget):
    """One row per track: where that track **lost** frames, over the timeline.

    Bar height is how many frames the column is missing, scaled to the worst column in
    the run — so empty means nothing was lost and the tall bars are where to look. Click
    to jump; a cursor marks the current frame. See ``_buckets_for`` for why this plots
    loss rather than presence.
    """

    clicked = pyqtSignal(int)
    #: Rows in lines of the interface font, not pixels — `sizing.lines` is this package's
    #: `em`. Called where the widget is built, never at import: `lines()` reads the
    #: application font and there is no application at import time.
    _ROW_LINES = 1.6
    _LABEL_W = 26  # left gutter for the track ids, when there is more than one row

    def __init__(self, track_ids: list[int], presence: dict, total: int):
        super().__init__()
        self._ROW_H = lines(self._ROW_LINES)
        self._track_ids = track_ids
        self._presence = presence
        self._total = total
        self._cursor = 0
        self._bucket_w = -1  # cached loss buckets are valid for this plot width
        self._buckets: dict = {}
        self._peak_lost = 1  # the worst column, set with the buckets
        self.setFixedHeight(max(len(track_ids), 1) * self._ROW_H + 4)
        self.setMinimumWidth(200)

    @property
    def gutter(self) -> int:
        """The label column left of the plot — only when there are rows to tell apart.

        One subject is the ordinary run, and its single row labelled ``0`` said nothing while
        costing the plot its alignment with the count strip above: 26px of blank and a digit
        where the recording's first frames should have been.
        """
        return self._LABEL_W if len(self._track_ids) > 1 else 0

    def set_cursor(self, frame: int) -> None:
        self._cursor = frame
        self.update()

    def _plot_width(self) -> int:
        return max(self.width() - self.gutter, 1)

    def x_for_frame(self, frame: int) -> int:
        return _frame_x(frame, self._total, self.gutter, self._plot_width())

    def mousePressEvent(self, event) -> None:
        if self._total:
            self.clicked.emit(
                _x_frame(
                    event.position().x(), self._total, self.gutter, self._plot_width()
                )
            )

    def _buckets_for(self, plot_w: int) -> dict:
        """Per-track **frames lost** in each column — the quantity the plot is about.

        This row used to be a boolean presence band, a column set if *any* frame in it
        was tracked. A column is hundreds of frames wide on a real recording, so one
        tracked frame in hundreds painted a
        solid healthy bar and a run that had lost thousands drew as an unbroken line —
        the plot someone was looking at when they said the losses "aren't showing up when
        you have this many frames". The bucketing was hiding them, not the resolution.

        Two encodings were tried before this one, and both failed for the same reason —
        a real run loses a small share of its frames:

        - *Mark a column if any frame is missing.* A few thousand scattered losses put
          at least one in nearly every column.
        - *Bar height = the covered fraction.* 87% and 99% are 10px and 11px in a 14px
          row, so the worst dropouts and the quiet stretches drew identically.

        So the plot shows **loss**, scaled to the worst column (``paintEvent``). Bars mean
        badness, empty means nothing was lost, and the structure — a few real dropouts
        against a low scatter — is visible because the scale is set by the data rather
        than by an absolute axis that squashes all of it into one pixel.

        Cached until the plot width changes, so cursor repaints don't recompute it.
        ``reduceat``'s duplicate-index behaviour reproduces the ≥1-frame-per-column guard
        when plot_w > frames, and ``sizes`` is derived the same way so the two always agree
        about what a column holds — deriving them differently is how a column ends up
        reporting more losses than it has frames.
        """
        if self._bucket_w == plot_w:
            return self._buckets
        n = self._total
        los = (np.arange(plot_w) / plot_w * n).astype(int)
        sizes = np.add.reduceat(np.ones(n, dtype=np.intp), los)
        self._buckets = {
            tid: sizes
            - np.add.reduceat(np.asarray(self._presence[tid]).astype(np.intp), los)
            for tid in self._track_ids
        }
        # One scale across every row, so two tracks' bars mean the same thing. Never 0:
        # a run that lost nothing divides by it on the next paint.
        self._peak_lost = (
            max((int(v.max()) for v in self._buckets.values()), default=0) or 1
        )
        self._bucket_w = plot_w
        return self._buckets

    def paintEvent(self, _event) -> None:
        theme = plot_theme.current()
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(theme.surface))
        n = self._total
        if not n or not self._track_ids:
            return
        plot_w = self._plot_width()
        left = self.gutter
        buckets = self._buckets_for(plot_w)
        for r, track_id in enumerate(self._track_ids):
            y = r * self._ROW_H + 2
            if left:
                painter.setPen(QColor(theme.text_secondary))
                painter.drawText(2, y + self._ROW_H - 4, str(track_id))
            lost = buckets[track_id]
            color = QColor(*_track_color(track_id))
            bar_h = self._ROW_H - 3
            # Scaled to the worst column in the run, the way the detections strip above
            # scales to its own peak. An absolute 0-1 axis cannot work here: real runs
            # lose a small share of their frames, so every bar would round to nothing and the
            # picture would be blank at exactly the moment it matters.
            heights = np.rint(lost / self._peak_lost * bar_h).astype(int)
            # Losing *nothing* has to draw as nothing. Rounding a 0.4% loss up to a
            # visible pixel is how the previous encoding lit every column and said less
            # than the one it replaced.
            heights[lost <= 0] = 0
            for height in np.unique(heights[heights > 0]):
                for start, length in _present_runs(heights == height):
                    painter.fillRect(left + start, y, length, int(height), color)
        cursor_x = self.x_for_frame(self._cursor)
        painter.setPen(QColor(theme.text))
        painter.drawLine(cursor_x, 0, cursor_x, h)


class DetectionRunPreview(QWidget):
    """Frame viewer + overview for one recording's detection run."""

    #: The style the viewer opens in. The mask is the result of a detection or tracking run,
    #: so it is drawn; a keypoints run's points are, and a mask tint would sit under them.
    _DEFAULT_STYLE = MASK_BOX_CONF

    def __init__(
        self,
        recording,
        run_id: str,
        *,
        experiment_dir=None,
        parent_run_id: str | None = None,
    ):
        """``experiment_dir``/``parent_run_id`` are what the export buttons need and the frame
        viewer does not: the *experiment's* run (``run_006``), not this recording's own
        ``det_01``, because the export target is the run (`export_targets._RUN_TARGETS`'
        ``preview_frame``/``preview_video``) and only an experiment-level run has a record to
        export from. ``None`` hides the buttons, which is right for a run with no experiment-
        level record (one launched per recording from the API or CLI) and for a fake recording
        in a test. It is **not** a default to rely on: a viewer built without them looks
        complete and has no Export, which is how this went missing from the run-detail page
        (`ModuleUI.preview_widget` takes both as required keywords for that reason).
        """
        super().__init__()
        self._frame = 0
        self._style = self._DEFAULT_STYLE  # snapshot for the off-thread compute
        self._experiment_dir = experiment_dir
        self._parent_run_id = parent_run_id
        self._recording_id = getattr(recording, "id", None)
        layout = QVBoxLayout(self)
        self._layout = layout
        self._summary = QLabel("")
        # The summary line has free width to its right, which is where the export buttons go: not
        # in the transport row under the picture. That row wraps, and `PictureWithControls` sizes
        # the page's minimum height from it wrapped at the picture's narrowest width, so every
        # control added there costs a full line of height at any window size (two buttons made the
        # tracking page 32px too tall for the 1440x810 window, which scrolls the transport away).
        self._head = QHBoxLayout()
        self._head.addWidget(self._summary, 1)
        layout.addLayout(self._head)
        # Zoomable: a detection is judged at the size of the thing detected, and a whole
        # 1080p frame fitted to this pane draws a mouse a few dozen pixels wide.
        self._image = ImageView(zoomable=True)
        # A **floor**, not a size: the picture takes the page's slack and gives it back when
        # the window shrinks. 480x360 was neither — it was most of the height this page has to
        # spend, so the run's own details and its list could not fit above it without the page
        # scrolling, and the transport sank below the bottom of the window.
        self._image.setMinimumSize(lines(14), lines(9))
        layout.addWidget(self._image, 1)
        # Opening a run reads its per-frame counts (and, for tracking, track presence),
        # which scales with run length — off the UI thread so the viewer never freezes
        # to a blank pane. _populate builds the rest once the data is in.
        run_busy(
            self,
            lambda: self._load(recording, run_id),
            self._populate,
            label="Loading run…",
        )

    def _load(self, recording, run_id: str) -> tuple:
        """Off-thread: open the run and read its per-frame counts + overview data."""
        preview = self._make_preview(recording, run_id)
        part_counts = preview.part_frame_counts
        total = sum(part_counts)
        counts = (
            np.concatenate([preview.counts(p) for p in range(len(part_counts))])
            if total
            else np.zeros(0, dtype=int)
        )
        # The rate rides along from the off-thread load: a player on a fixed timer plays
        # every recording at whatever that timer happens to be, real time by coincidence.
        return (
            preview,
            part_counts,
            total,
            counts,
            self._load_overview(preview),
            recording.video_fps,
        )

    def _populate(self, data: tuple) -> None:
        """UI thread: build the summary, strip, overview, and scrubber from the data."""
        (
            self._preview,
            self._part_counts,
            self._total,
            self._counts,
            overview,
            fps,
        ) = data
        self._summary.setWordWrap(True)
        self._summary.setText(self._summary_text(overview, fps))
        self._strip = _CountStrip(self._counts)
        self._strip.clicked.connect(lambda f: self._scrubber.set_frame(f))

        # shared scrubber: slider + ◀/▶ + label, hold-to-accelerate keys, coalesced decode
        # A finished run, reviewed frame by frame — nothing is drawn on it.
        self._scrubber = FrameScrubber(playable=True)
        self._scrubber.set_total(self._total)
        # The map the preview already read to seek by — in memory, so this is a concatenation,
        # not a read. It is what lets the row say the frame's time without a frame-rate guess.
        self._scrubber.set_timestamps(self._preview.timestamps())
        if fps:
            self._scrubber.set_playback_rate(fps)
        self._style_picker = QComboBox()
        for style in PREVIEW_STYLES:
            self._style_picker.addItem(PREVIEW_STYLE_LABELS[style], style)
        self._style_picker.setCurrentIndex(self._style_picker.findData(self._style))
        self._style_picker.currentIndexChanged.connect(self._on_style_changed)
        self._scrubber.add_control(self._style_picker)

        # Both go through the shared export dialog (`gui/export.py::open_export`),
        # pre-answered — the same "one dialog, two triggers" rule the clip player's own
        # Export video… button keeps. Neither writes anything itself.
        export_frame = QPushButton("Export frame…")
        export_frame.setToolTip(
            "Write the frame on screen to a PNG, in the style shown."
        )
        export_frame.clicked.connect(self._export_frame)

        export_video = QPushButton("Export video…")
        export_video.setToolTip(
            "Write the whole recording — every video part — to an mp4, in the style "
            "shown."
        )
        export_video.clicked.connect(self._export_video)
        self._export_buttons = (export_frame, export_video)
        self._place_export_buttons()
        # No experiment-level run to export from (see `__init__`) — hidden rather than a click
        # that always fails.
        if not self._can_export():
            export_frame.hide()
            export_video.hide()

        self._scrubber.frame_changed.connect(
            self._on_frame_changed
        )  # immediate cursors
        self._scrubber.set_decoder(self._compute)  # off-thread decode
        self._scrubber.rendered.connect(self._display)  # on-thread display
        self._decode_busy = decode_overlay(self._scrubber, self._image)
        # Picture directly above its controls (#396), and the overviews under the pair: a
        # count strip or track timeline between the two put the transport a strip's height
        # below the frame it drives.
        # The slot is read *before* the block is built: re-parenting the picture into it drops
        # the picture's layout item, so a `replaceWidget` afterwards finds nothing to replace
        # and the block is never shown.
        slot = self._layout.indexOf(self._image)
        self._block = PictureWithControls(self._image, self._scrubber)
        self._layout.insertWidget(slot, self._block, 1)
        self._layout.addWidget(self._strip)
        self._build_overview(self._layout, overview)  # e.g. a track-presence timeline
        self._scrubber.set_frame(0)

    def _on_style_changed(self, index: int) -> None:
        self._style = self._style_picker.itemData(index)
        self._scrubber.set_frame(self._frame)  # re-decode the current frame

    def _on_frame_changed(self, frame: int) -> None:
        """Immediate, cheap: move the cursors as the user scrubs (no decode)."""
        self._frame = frame
        self._strip.set_cursor(frame)
        self._set_overview_cursor(frame)

    def _compute(self, global_frame: int):
        """Off-thread: decode + overlay → a display payload (no Qt here)."""
        if not self._total:
            return ("text", "(no video)")
        part, local = locate_frame(self._part_counts, global_frame)
        result, reason = frame_source.decode(
            lambda: self._preview.frame(
                part, local, with_masks=preview_needs_mask(self._style)
            )
        )
        if reason is not None:
            return ("text", frame_source.unavailable_text(reason))
        _actual, rgb, detections = result
        return ("img", self._decorate(self._overlay(rgb, detections), part, local))

    def _overlay(self, rgb, detections):
        """The run's own drawing of ``detections`` on ``rgb``, in the style picked — the one
        draw a subclass layers its own marks over (the keypoint viewers' points, §26).
        """
        return draw_preview_overlay(
            rgb,
            detections,
            self._style,
            color_fn=self._color_fn,
            label_fn=self._label_fn,
        )

    def _display(self, _frame: int, payload) -> None:
        kind, data = payload
        if kind == "text":
            self._image.clear_frame(data)
            return
        # Full resolution: the view scales it, so zooming in reaches real pixels rather than
        # magnifying a copy already shrunk to the pane.
        self._image.set_frame(_to_pixmap(data))
        self._block.set_aspect(self._image.pixmap_size())

    def _export_frame(self) -> None:
        self._open_preview_export("preview_frame", frames=(self._frame, self._frame))

    def _export_video(self) -> None:
        self._open_preview_export("preview_video", frames=None)

    def _place_export_buttons(self) -> None:
        """The frame and video buttons, at the right of the summary line (see ``__init__``)."""
        for button in self._export_buttons:
            self._head.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)

    def _can_export(self) -> bool:
        """Whether there is an experiment-level run behind this viewer to export from."""
        return (
            self._experiment_dir is not None
            and self._parent_run_id is not None
            and self._recording_id is not None
        )

    def _export_seed(self) -> dict:
        """What the viewer knows that no control in the export dialog offers: the recording on
        screen and the style on screen. A subclass adds what it draws on top."""
        return {"recording_id": self._recording_id, "style": self._style}

    def _open_preview_export(self, target_id: str, *, frames) -> None:
        """Open the shared export dialog, pre-answered with the run and what is on screen."""
        from detail.api.exporters import ExportScope, ExportSubject
        from detail.gui.export import open_export

        scope = ExportScope(frames=frames) if frames is not None else None
        open_export(
            self,
            ExportSubject("run", self._experiment_dir, self._parent_run_id),
            scope,
            only=(target_id,),
            seed=self._export_seed(),
        )

    # -- subclass hooks ------------------------------------------------------ #
    def _summary_text(self, overview, fps: float | None) -> str:
        """The headline numbers, in words. Each stage says what *it* is about.

        Detection's question is how much it found; tracking's is how much of the recording
        it ended up holding — a different number, and the one a user is actually asking
        when they open a tracking run. Inheriting this line verbatim is how a tracking run
        came to report its detection counts and never its coverage.
        """
        total = int(self._counts.sum())
        with_dets = int((self._counts > 0).sum())
        peak = int(self._counts.max()) if self._total else 0
        # Over the frames the run looked at, as the run's results list counts them: at
        # every_n_frames=4 the source count would report three frames in four as empty.
        processed = self._preview.frames_processed
        return (
            f"{total} detections · {with_dets}/{processed} frames with ≥1 · "
            f"up to {peak}/frame"
        )

    def _make_preview(self, recording, run_id: str):
        return recording.detection_preview(run_id)  # off-thread (called from _load)

    def _color_fn(self, det):
        return DETECTION_BOX_COLOR

    def _label_fn(self, det, show_confidence: bool):
        return f"{det.class_name} {det.confidence:.2f}" if show_confidence else None

    def _decorate(self, rgb, part: int, local: int):
        """Off-thread: draw what belongs to *this frame* over the overlay — a hook for a run
        whose marks are per frame rather than per detection (keypoints). Default: nothing.
        """
        return rgb

    def _load_overview(self, preview):
        """Off-thread: overview data (e.g. track presence). ``None`` = no overview."""
        return None

    def _build_overview(self, layout, overview) -> None:
        """UI thread: subclasses build an overview widget from ``_load_overview``'s data."""

    def _set_overview_cursor(self, frame: int) -> None:
        pass


class TrackingRunPreview(DetectionRunPreview):
    """Same frame viewer, colored per track id with a ``track N`` label, plus a
    track-presence timeline."""

    def _make_preview(self, recording, run_id: str):
        return recording.tracking_preview(run_id)  # off-thread (called from _load)

    def _color_fn(self, det):
        return _track_color(det.object_id)

    def _label_fn(self, det, show_confidence: bool):
        # The track id is the reason a tracking preview exists, so it stays in the
        # label even when the style turns confidence off — unlike detection, which has
        # no label at all in that case.
        if show_confidence:
            return f"track {det.object_id} · {det.class_name} {det.confidence:.2f}"
        return f"track {det.object_id}"

    def _summary_text(self, overview, fps: float | None) -> str:
        """Coverage, per subject — the number a tracking run is actually asked for.

        A gap is invisible in the timeline at any realistic length: a long recording
        across ~1,000 pixels is hundreds of frames a column, so a one-frame loss cannot
        be a pixel. The
        numbers are what make it knowable, which is why they are here and not only drawn.
        """
        track_ids, presence = overview
        # Frames the run *looked at*, not source frames: at every_n_frames=4 three frames
        # in four were never processed, and counting them as missed reports a healthy run
        # as having lost 75% of the recording.
        processed = self._preview.frames_processed
        with_detections = int((self._counts > 0).sum())
        detections = f"{with_detections:,}/{processed:,} frames with detections"
        if not track_ids:
            return f"no subject was tracked on any frame · {detections}"
        lines = []
        for track_id in track_ids:
            stats = track_coverage(presence[track_id], fps)
            # Two ratios, because one number cannot separate the two ways a frame ends up
            # without a subject: the detector found nothing there, or it found something
            # and the tracker did not take it. The gap between these lines is the second,
            # and it is the only one a user can act on.
            line = (
                f"subject {track_id} — {stats['frames_covered']:,}/{processed:,} frames "
                f"with tracked detections · {detections}"
            )
            if stats["n_gaps"]:
                longest = f"{stats['longest_gap']:,} frames"
                if stats["longest_gap_s"] is not None:
                    longest += f" ({stats['longest_gap_s']:.1f} s)"
                line += f" · {stats['n_gaps']:,} gaps · longest {longest}"
            lines.append(line)
        return "\n".join(lines)

    def _load_overview(self, preview):
        return preview.track_presence()  # off-thread (called from _load)

    def _build_overview(self, layout, overview) -> None:
        track_ids, presence = overview
        self._timeline = _TrackTimeline(track_ids, presence, self._total)
        self._timeline.clicked.connect(lambda f: self._scrubber.set_frame(f))
        # One axis down the page: the count strip starts where the timeline's plot does.
        self._strip.set_left(self._timeline.gutter)
        layout.addWidget(
            QLabel(
                "Frames with no subject (click to jump) — taller means more lost in "
                "that slice of the recording; empty means none:"
            )
        )
        layout.addWidget(self._timeline)

    def _set_overview_cursor(self, frame: int) -> None:
        self._timeline.set_cursor(frame)


def _titled(title: str | QLabel, widget: QWidget) -> QWidget:
    """``widget`` under a title — a string, or a label the caller keeps to retitle it."""
    box = QWidget()
    column = QVBoxLayout(box)
    column.setContentsMargins(0, 0, 0, 0)
    column.addWidget(title if isinstance(title, QLabel) else QLabel(title))
    column.addWidget(widget, 1)
    return box


class InputImageScrubber(Dialog):
    """Scrub every source frame to compare a constructor's model input against the
    original frame, one track at a time (Section 14).

    Decodes + renders on demand (slider release / step), so it stays responsive on
    long video. The spec is resolved per recording, so the centered constructors
    show the same canvas the embedding run would use for that recording.

    ``alignment`` (§14) is the alignment the run would apply, with each recording's keypoint
    track run in ``keypoint_run_ids``, so the preview turns what the run will turn. ``pose``
    (``{"pose": "draw", "pose_min_score": ...}``) draws the points the run will draw, from the
    same keypoint track runs.
    """

    def __init__(
        self,
        recordings,
        run_ids,
        track_ids,
        constructor,
        model_id=None,
        parent=None,
        *,
        alignment=None,
        keypoint_run_ids=None,
        pose=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Input image preview")
        self._recordings = recordings
        # {recording_id: tracking run id resolved from the selected spec} — differs per
        # recording (§6.5), so the run id is looked up per recording, not shared.
        self._run_ids = run_ids
        self._track_ids = [int(t) for t in track_ids]
        self._constructor = constructor
        # supplies the default resize target so the preview matches the embed input
        self._model_id = model_id
        # §14: the alignment the run would apply (mode, heading definition, settings) and
        # each recording's keypoint track run, so the preview turns what the run will turn.
        self._alignment = (
            dict(alignment)
            if alignment and (alignment.get("align") or "none") != "none"
            else None
        )
        self._keypoint_run_ids = dict(keypoint_run_ids or {})
        # §14 pose: only a drawn pose changes the picture; appended numbers change the vector
        self._pose = dict(pose) if pose and pose.get("pose") == "draw" else None
        self._preview = None
        self._part_counts: list[int] = []
        self._total = 0
        self._frame = 0

        layout = self.body()

        top = QHBoxLayout()
        top.addWidget(QLabel("Recording:"))
        self._recording = QComboBox()
        for i, recording in enumerate(recordings):
            self._recording.addItem(recording.name, i)
        top.addWidget(self._recording)
        top.addWidget(QLabel("Track:"))
        self._track = QComboBox()
        for track_id in self._track_ids:
            self._track.addItem(f"track {track_id}", track_id)
        top.addWidget(self._track)
        top.addStretch()
        layout.addLayout(top)

        self._status = QLabel()
        layout.addWidget(self._status)

        panes = QHBoxLayout()
        self._orig = self._image_pane(420, 320)
        self._input = self._image_pane(320, 320)
        panes.addWidget(_titled("Original frame", self._orig), 3)
        panes.addWidget(_titled("Model input", self._input), 2)
        layout.addLayout(panes, 1)

        # snapshot for off-thread compute
        self._active_track = self._track.currentData()
        # Watching what the embedder sees as the subject moves is the whole point here.
        self._scrubber = FrameScrubber(playable=True)
        self._scrubber.frame_changed.connect(self._on_frame_changed)
        self._scrubber.set_decoder(self._compute)  # off-thread decode
        self._scrubber.rendered.connect(self._display)  # on-thread display
        self._decode_busy = decode_overlay(self._scrubber, self._orig)
        layout.addWidget(self._scrubber)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)

        self._recording.currentIndexChanged.connect(self._reload)
        self._track.currentIndexChanged.connect(self._on_track_changed)
        self._reload()

    def _on_track_changed(self) -> None:
        self._active_track = self._track.currentData()
        self._scrubber.set_frame(self._scrubber.frame())  # re-decode current frame

    @staticmethod
    def _image_pane(width: int, height: int) -> ImageView:
        view = ImageView(zoomable=True)
        view.setMinimumSize(width, height)
        return view

    def _current_recording(self):
        return self._recordings[self._recording.currentData()]

    def _reload(self) -> None:
        recording = self._current_recording()
        run_id = self._run_ids.get(recording.id)
        if run_id is None:
            self._reload_failed(
                "This recording has no tracking run matching the selected spec."
            )
            return
        # Resolving the constructor spec + opening the preview reads the run — off the
        # UI thread behind a busy bar, so a recording/track change never freezes.
        run_busy(
            self,
            lambda: self._resolve_preview(recording, run_id),
            self._apply_preview,
            label="Loading input image…",
            on_error=lambda exc: self._reload_failed(str(exc)),
            supersedes="preview",  # a recording or track change reloads mid-read
        )

    def _resolve_preview(self, recording, run_id: str):
        """Off-thread: resolve the constructor spec + open the input-image preview. An
        aligned preview sizes its canvas from the turned masks and carries the alignment
        step, as the run will (§14)."""
        alignment = None
        canvas_px = None
        if self._alignment is not None or self._pose is not None:
            from detail.pipeline.embedding import (
                _alignment_settings,
                largest_turned_mask_px,
            )

            keypoint_run_id = self._keypoint_run_ids.get(recording.id)
            if keypoint_run_id is None:
                raise ValueError(
                    f"recording {recording.name!r} has no keypoint track run matching the "
                    "selected one, so its input cannot be aligned or drawn on"
                )
            track_run = recording.keypoint_track_run(keypoint_run_id)
            cutoff = (track_run.meta.get("config") or {}).get("decode_cutoff")
            alignment = {
                **(self._alignment or {"align": "none"}),
                "keypoint_run_id": keypoint_run_id,
                "heading_cutoff": cutoff,
                "pose": "draw" if self._pose is not None else "none",
                "pose_min_score": (
                    self._pose.get("pose_min_score")
                    if self._pose is not None
                    and self._pose.get("pose_min_score") is not None
                    else cutoff
                ),
            }
            if self._alignment is not None:
                canvas_px = largest_turned_mask_px(
                    recording.dir,
                    run_id,
                    self._track_ids,
                    _alignment_settings(alignment),
                )
        if alignment is None:  # the unaligned call, exactly as before alignment existed
            spec = recording.resolve_constructor(  # off-thread (called from _reload)
                self._constructor,
                run_id,
                self._track_ids,
                model_id=self._model_id,
            )
            return recording.input_image_preview(  # off-thread (via _reload)
                run_id, spec
            )
        from detail.pipeline.embedding import _draw_pose_step
        from detail.transforms.constructors import with_alignment, with_pose

        spec = recording.resolve_constructor(  # off-thread (called from _reload)
            self._constructor,
            run_id,
            self._track_ids,
            model_id=self._model_id,
            canvas_px=canvas_px,
        )
        # the points first, then the turn: the order validate builds the run's spec in
        spec = with_alignment(
            with_pose(spec, _draw_pose_step(recording.dir, alignment)),
            alignment["align"],
        )
        return recording.input_image_preview(  # off-thread (via _reload)
            run_id, spec, alignment=alignment
        )

    def _apply_preview(self, preview) -> None:
        self._preview = preview
        self._part_counts = preview.part_frame_counts
        self._total = sum(self._part_counts)
        self._scrubber.set_total(self._total)
        # The map the preview already read to seek by — in memory, so this is a concatenation,
        # not a read. It is what lets the row say the frame's time without a frame-rate guess.
        self._scrubber.set_timestamps(self._preview.timestamps())
        fps = self._current_recording().video_fps  # mtime-cached yaml read
        if fps:
            self._scrubber.set_playback_rate(fps)
        self._scrubber.set_frame(0)

    def _reload_failed(self, message: str) -> None:
        self._preview = None
        self._part_counts, self._total = [], 0
        self._scrubber.set_total(0)
        self._status.setText(message)
        self._orig.setText("(unavailable)")
        self._input.clear()

    def _locate(self, frame: int) -> tuple[int, int]:
        for part, count in enumerate(self._part_counts):
            if frame < count:
                return part, frame
            frame -= count
        return max(len(self._part_counts) - 1, 0), max(self._part_counts[-1] - 1, 0)

    def _on_frame_changed(self, frame: int) -> None:
        self._frame = frame

    def _compute(self, global_frame: int):
        """Off-thread: decode the original frame + the model input (no Qt here)."""
        if self._preview is None or not self._total:
            return ("none",)
        part, local = self._locate(global_frame)
        decode = lambda: self._preview.input_at(
            part, local, self._active_track
        )  # noqa: E731
        result, reason = frame_source.decode(decode)
        if reason is not None:
            return ("unavailable", reason)
        return ("ok", *result)  # (actual, rgb, model_input)

    def _display(self, _frame: int, payload) -> None:
        kind = payload[0]
        if kind == "none":
            return
        if kind == "unavailable":
            self._orig.setText(frame_source.unavailable_text(payload[1]))
            self._input.clear()
            return
        _, actual, rgb, model_input = payload
        self._show(self._orig, rgb)
        if model_input is None:
            self._input.clear()
            self._input.setText("(no detection\nfor this track)")
            self._status.setText(f"frame {actual} · track absent")
        else:
            self._show(self._input, model_input)
            self._status.setText(
                f"frame {actual} · input {model_input.shape[1]}×{model_input.shape[0]}"
            )

    @staticmethod
    def _show(view: ImageView, rgb: np.ndarray) -> None:
        view.set_frame(_to_pixmap(np.ascontiguousarray(rgb)))


_SCATTER_MAX_POINTS = 5000  # cap the inspection scatter so big runs stay responsive


def pose_text(meta: dict) -> str | None:
    """How a run put the pose into its vectors (§14), for the viewer's header; None for a
    run without pose (and every run recorded before pose existed)."""
    from detail.configs.embedding import POSE_CHOICES

    pose = meta.get("pose") or "none"
    if pose == "none":
        return None
    label = POSE_CHOICES[pose][0].lower()
    if pose == "append" and meta.get("pose_weight") is not None:
        return f"{label} (weight {float(meta['pose_weight']):g})"
    return label


class EmbeddingRunPreview(QWidget):
    """Embedding-run inspection (Section 14): a PCA point cloud colored by track, and the
    frame + model input behind a clicked point — three equal columns in one row.

    Inspection only — cheap numpy PCA, distinct from the §16 analysis stack. The
    PCA reads every embedding zarr + runs an SVD, so it loads **off the UI thread**
    (the page opens immediately, showing "Loading…"); the scatter is subsampled to
    ``_SCATTER_MAX_POINTS`` so a long run stays responsive.

    **Side by side, not a scatter with a column beside it.** The scatter used to take three
    fifths of the width and stack a per-track coverage table, a status line and both images
    down the rest, so the frame you clicked for was a 240px thumbnail under a table that
    said, for a single-subject run, one number the scatter already showed. The table is gone
    and the clicked frame titles itself ``Frame x / last`` — the scrubbers' own format —
    rather than a status line repeating the track and a part-local index.
    """

    #: Each column's floor, in lines of the interface font (`sizing.lines`, this package's
    #: `em`). Square: the scatter and the model input are, and a frame letterboxes in one.
    #: A floor, not a size — the row takes the page's slack, so this is how small the three
    #: panes may get before the page scrolls, not how big they are on a full-screen window.
    _COLUMN_LINES = 13

    def __init__(self, recording, run_id: str):
        super().__init__()
        import pyqtgraph as pg

        self._preview = None
        self._coords = np.zeros((0, 2), dtype=np.float32)
        self._tracks = np.zeros(0, dtype=int)
        self._locators: list[tuple[int, str, int, int]] = []

        layout = QVBoxLayout(self)
        self._head = QLabel("")
        layout.addWidget(self._head)
        self._status = QLabel("")
        layout.addWidget(self._status)

        side = lines(self._COLUMN_LINES)
        self._plot = plot_theme.themed_plot()
        self._plot.setLabel("bottom", "PC1")
        self._plot.setLabel("left", "PC2")
        # Genuinely unbounded: a principal component has no domain, and panning around the
        # projection is how you read a scatter. Declared rather than omitted.
        bound_axes(self._plot, x=None, y=None)
        # Each point ringed in the surface, so a dense cloud of one track stays countable.
        self._scatter = pg.ScatterPlotItem(
            size=plot_theme.MARKS.marker_size + 1,
            pen=plot_theme.pen(plot_theme.current().surface, width=1),
        )
        self._plot.addItem(self._scatter)
        self._plot.setMinimumSize(side, side)

        self._orig = self._pane(side)
        self._input = self._pane(side)
        self._frame_title = QLabel("Frame")
        columns = QHBoxLayout()
        # Equal stretch over equal floors is what makes the thirds equal: a stretch alone
        # hands out only the room left after each column's own minimum.
        columns.addWidget(_titled("PCA", self._plot), 1)
        columns.addWidget(_titled(self._frame_title, self._orig), 1)
        columns.addWidget(_titled("Model input", self._input), 1)
        layout.addLayout(columns, 1)
        self._scatter.sigClicked.connect(self._on_clicked)

        # Opening the run + reading every embedding zarr + the PCA SVD is slow, so it
        # runs off the UI thread behind a busy bar; _on_loaded builds the scatter.
        run_busy(
            self,
            lambda: self._load(recording, run_id),
            self._on_loaded,
            label="Loading embeddings…",
            on_error=lambda exc: self._status.setText(f"(failed to load: {exc})"),
        )

    def _load(self, recording, run_id: str):
        """Worker thread (no Qt): open the run + the expensive read + PCA + row count."""
        preview = recording.embedding_preview(run_id)
        meta = preview.metadata()
        coords, tracks, locators = preview.pca_2d(max_points=_SCATTER_MAX_POINTS)
        embedded = sum(row["n_frames"] for row in preview.coverage())
        heading = (
            self._heading_overview(recording, preview, meta)
            if (meta.get("align") or "none") != "none"
            else None
        )
        return (preview, meta, coords, tracks, locators, embedded, heading)

    @staticmethod
    def _heading_overview(recording, preview, meta):
        """Off-thread (§14 *Coverage and alerts*): the run's heading loss, read through
        the embedding preview API."""
        tracking = recording.tracking_preview(meta["tracking_run_id"])  # off-thread
        return preview.heading_loss(tracking)  # off-thread (called from _load)

    def _on_loaded(self, result) -> None:
        (
            self._preview,
            meta,
            self._coords,
            self._tracks,
            self._locators,
            embedded,
            heading,
        ) = result
        if heading is not None:
            self._show_heading(heading)
        self._head.setText(
            " · ".join(
                str(x)
                for x in (
                    meta.get("model_id"),
                    meta.get("model_variant"),
                    meta.get("constructor"),
                    pose_text(meta),
                    f"{meta.get('output_dim')}-d",
                    meta.get("status"),
                )
                if x
            )
        )
        spots = [
            {
                "pos": (float(self._coords[i, 0]), float(self._coords[i, 1])),
                "brush": plot_theme.brush(QColor(*_track_color(self._tracks[i]))),
                "data": i,
            }
            for i in range(len(self._locators))
        ]
        self._scatter.setData(spots=spots)
        if not spots:
            self._status.setText("(no embeddings in this run)")
        elif len(spots) < embedded:
            self._status.setText(
                f"Click a point to inspect · showing {len(spots)} of {embedded}"
            )
        else:
            self._status.setText("Click a point to inspect")

    #: §14 *Coverage and alerts*: policy thresholds, not measured.
    HEADING_COVERAGE_ALERT = 0.95
    HEADING_GAP_ALERT_S = 2.0

    def _show_heading(self, heading: dict) -> None:
        """The heading loss row per track, the counts, and the amber alert when heading
        coverage is under 95% or the longest gap over 2 s."""
        from detail.gui.status_color import WARN, set_label_color

        layout = self.layout()
        timeline = _TrackTimeline(
            heading["track_ids"], heading["kept"], heading["total"]
        )
        layout.insertWidget(
            2,
            QLabel(
                "Frames the track had a subject but no heading within the gap, so no "
                "embedding (taller means more lost in that slice; empty means none):"
            ),
        )
        layout.insertWidget(3, timeline)
        lines_, alerts = [], []
        for track, parts in sorted(heading["coverage"].items()):
            frames = sum(int(p.get("frames", 0)) for p in parts.values())
            own = sum(int(p.get("own", 0)) for p in parts.values())
            interpolated = sum(int(p.get("interpolated", 0)) for p in parts.values())
            dropped = sum(int(p.get("dropped", 0)) for p in parts.values())
            longest = max(
                (float(p.get("longest_gap_s", 0.0)) for p in parts.values()),
                default=0.0,
            )
            share = (own + interpolated) / frames if frames else 1.0
            lines_.append(
                f"{track.replace('_', ' ')}: heading on {own:,} frames, interpolated on "
                f"{interpolated:,}, dropped {dropped:,} ({share:.0%} covered, longest gap "
                f"{longest:.1f} s)"
            )
            if share < self.HEADING_COVERAGE_ALERT:
                alerts.append(f"{track.replace('_', ' ')} heading coverage {share:.0%}")
            if longest > self.HEADING_GAP_ALERT_S:
                alerts.append(
                    f"{track.replace('_', ' ')} longest heading gap {longest:.1f} s"
                )
        summary = QLabel(
            "\n".join(lines_) if lines_ else "heading coverage not recorded"
        )
        summary.setWordWrap(True)
        layout.insertWidget(4, summary)
        self._heading_alert = QLabel("")
        self._heading_alert.setWordWrap(True)
        set_label_color(self._heading_alert, WARN)
        self._heading_alert.setVisible(bool(alerts))
        if alerts:
            self._heading_alert.setText(
                "; ".join(alerts)
                + ". Mark more example frames and run keypoint detect again, or raise the "
                "gap this run fills in a new embedding run."
            )
        layout.insertWidget(5, self._heading_alert)

    @staticmethod
    def _pane(side: int) -> ImageView:
        view = ImageView(zoomable=True)
        view.setMinimumSize(side, side)
        return view

    def _on_clicked(self, _scatter, points, *_args) -> None:
        # pyqtgraph hands `points` as a numpy array of spot items — test with len,
        # not truthiness (a multi-element array raises on bool()).
        if len(points) == 0:
            return
        track_id, stem, frame_idx, _t = self._locators[points[0].data()]
        run_busy(
            self._orig,  # the frame decode is slow — off-thread with a busy overlay
            lambda: self._decode(track_id, stem, frame_idx),
            self._show_clicked_frame,
            label="Loading frame…",
            supersedes="frame",  # the last point clicked is the frame shown
            on_error=lambda exc: self._status.setText(
                "(source video not in this export)"
                if isinstance(exc, FileNotFoundError)
                else f"(error: {exc})"
            ),
        )

    def _decode(self, track_id: int, stem: str, frame_idx: int):
        """Off-thread: the frame + model input behind a point, and where it sits in the
        recording — placed from the frame that actually decoded, not the one asked for.
        """
        view = self._preview.frame_view(track_id, stem, frame_idx)  # off-thread
        if view is None:
            return None
        actual, rgb, model_input = view
        return self._preview.frame_position(stem, actual), rgb, model_input

    def _show_clicked_frame(self, result) -> None:
        if result is None:
            self._status.setText("(could not decode frame)")
            return
        position, rgb, model_input = result
        if position is not None:
            frame, total = position
            self._frame_title.setText(f"Frame {frame} / {max(total - 1, 0)}")
        self._show(self._orig, rgb)
        if model_input is None:
            self._input.clear()
        else:
            self._show(self._input, model_input)

    @staticmethod
    def _show(view: ImageView, rgb: np.ndarray) -> None:
        view.set_frame(_to_pixmap(np.ascontiguousarray(rgb)))
