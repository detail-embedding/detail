"""The labeling screen (§26 "Example sets"): marking an example set's frames.

A patch-matching keypoint method places keypoints by matching every tracked frame against a
handful the user has marked. This screen is a **video player first** — the shared
``FrameScrubber``, so play, step, speed, the jump keys, zoom and pan behave exactly as in every
other viewer — with the example set beside it, in the *Mark examples* dialog's shape
(``gui/behavior_query.py``): the video and track on top, the player and the frame list in a
remembered splitter, an *Annotating: <point>* box under the transport.

- navigate to any tracked frame (play, scrub, ←/→, Shift+←/→, Home/End, Space to play/pause);
  the set's frames from this video are ticks on the slider;
- **click** the animal to place the highlighted keypoint — the frame joins the set on the first
  click, with the chosen track's box — and the next unmarked keypoint is highlighted;
- **Not visible (N)** says the current keypoint cannot be seen on this frame, **Clear
  (Backspace)** unmarks it, **1-9** choose a keypoint, **D** / **A** jump to the next / previous
  example frame; **Add frames** samples varied postures (or a spread over time) from every
  video offered. *By heading*, its floor and the compass are offered only when a recording
  here is not top down (``Experiment.heading_applies``, §26 *Headings*).

**Clicks are on the full frame**, so the stored normalized coordinate is the click's own; the
method maps it into whatever it crops. Points are drawn by role — a pair's left member blue,
its right vermillion, the midline green (``core.visualize.draw_keypoints``) — with the
skeleton's edges and the tracked box, so a left/right decision is made against the body it is
on. The tool row offers *Point* alone; *Region* and *Box* are the marks the schema reserves and
nothing reads yet (§26), so they are greyed and say so rather than hidden.

**Every click saves**, off the UI thread (``_MarkSaver``: one write in flight, the latest marks
per frame win), through ``Experiment.set_example_marks``, which rewrites the set atomically. The
video, its tracked boxes and every decode load off the UI thread too (``busy.run_busy`` and the
scrubber's own decoder). The **Skeleton** dropdown (the library the Keypoints manager
edits, ``gui/keypoints_manager.py``) changes the set's skeleton until the first mark, and is fixed
after that; a placed point can be dragged to move it here.

The example-set picker (``gui/example_set_picker.py``), wherever a model that needs examples is
chosen, opens this screen through :func:`open_labeling` (*New example set…*, *Mark frames…*).
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.core import class_names
from detail.core import yaml_load
from detail.core.examples import ExampleMark, ExampleSet
from detail.core.skeleton import Skeleton
from detail.core.visualize import draw_bbox, draw_keypoints
from detail.gui import frame_source
from detail.gui.busy import decode_overlay, run_busy, run_progress
from detail.gui.dialog import Dialog
from detail.gui.full_frame import to_pixmap
from detail.gui.grip_splitter import GripSplitter
from detail.gui.heading_review import HeadingCompass, HeadingReviewDialog
from detail.gui.image_view import ImageView, PictureWithControls
from detail.gui.scrubber import FrameScrubber
from detail.gui.sizing import (
    add_fitted_picture,
    bound_picture,
    emphasis_font,
    lines,
    set_available,
)
from detail.gui.spin import IntSpin
from detail.gui.workers import BackgroundCall
from detail.pipeline import subject_frames
from detail.storage import recording as recording_store

#: The tracked box the method crops to — drawn thin, so it frames the animal without covering
#: the points being placed on it.
_BOX_COLOR = (255, 220, 60)


# --------------------------------------------------------------------------- #
# Pure helpers (no Qt)
# --------------------------------------------------------------------------- #
def draw_example(
    rgb: np.ndarray,
    skeleton: Skeleton,
    marks: list[ExampleMark],
    current: str | None,
    box=None,
) -> np.ndarray:
    """The full frame with the placed marks, their edges, a ring on ``current`` when it is
    placed, and the tracked box."""
    out = rgb
    if box is not None:
        out = draw_bbox(out, box, _BOX_COLOR, thickness=1)
    names = skeleton.names
    xy = np.full((len(names), 2), np.nan)
    for mark in marks:
        if mark.visible and mark.x is not None and mark.target in names:
            xy[skeleton.index(mark.target)] = (mark.x, mark.y)
    out = draw_keypoints(out, xy, skeleton)
    if current in names and np.isfinite(xy[skeleton.index(current)]).all():
        h, w = out.shape[:2]
        cx, cy = xy[skeleton.index(current)] * (w, h)
        r = max(6, min(h, w) // 60)
        centre = (int(round(cx)), int(round(cy)))
        cv2.circle(out, centre, r + 2, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.circle(out, centre, r, (250, 250, 250), 2, cv2.LINE_AA)
    return out


def offered_videos(experiment, sources: dict[str, str] | None, example_set) -> dict:
    """Off the UI thread: the videos the player offers and the recording names.

    ``sources`` (recording id -> tracking run id) when a launch form passes the recordings it
    has checked; otherwise every complete tracking run of every recording, one entry each,
    so a recording with several runs is a choice the user makes rather than a guess. The
    recordings the set's frames came from are offered too, when this experiment has them.
    """
    names = experiment.recording_names()  # off-thread
    videos: list[tuple[str, str]] = []
    entries: dict[tuple[str, str], dict] = {}  # (recording, run) -> its tracking entry
    if sources:
        videos = [(rid, run) for rid, run in sources.items()]
    else:
        for recording_id in experiment.recording_ids:  # off-thread
            recording = experiment.get_recording(recording_id)
            for entry in recording.tracking_runs:
                if str(entry.get("status", "complete")) == "complete":
                    videos.append((recording_id, str(entry["id"])))
    for frame in example_set.frames:
        pair = (frame.recording_id, frame.tracking_run_id)
        if pair not in videos and frame.recording_id in names:
            videos.append(pair)
    for rid in {rid for rid, _run in videos}:
        for entry in experiment.get_recording(rid).tracking_runs:  # off-thread
            entries[(rid, str(entry["id"]))] = entry
    renamed = experiment.renamed_classes()
    per_recording = {}
    for rid, _run in videos:
        per_recording[rid] = per_recording.get(rid, 0) + 1
    return {
        "names": names,
        "columns": list(experiment.metadata_columns),  # off-thread: one CSV read
        "classes": detection_classes(experiment, videos[0]) if videos else [],
        "class_labels": renamed,
        "heading_applies": experiment.heading_applies(per_recording),  # off-thread
        "videos": [
            {
                "recording_id": rid,
                "run_id": run,
                "label": (
                    f"{names.get(rid, rid)} ({run})"
                    if per_recording[rid] > 1
                    else names.get(rid, rid)
                ),
                "tracking": tracking_kind(entries.get((rid, run)) or {}, names=renamed),
            }
            for rid, run in videos
        ],
    }


def tracking_kind(entry: dict, *, names: dict[str, str]) -> str:
    """What a tracking run follows, the same on every recording it was run on: its class by
    the experiment's names, and tracker — the key *From tracking* groups runs by, since a run
    id is a per-recording counter."""
    stored = entry.get("class_name")
    classes = class_names.label(str(stored), names) if stored else "every class"
    tracker = entry.get("tracker_id") or "tracking"
    return f"{classes} · {tracker}"


def detection_classes(experiment, video: tuple[str, str]) -> list[str]:
    """Off the UI thread: the class names one offered tracking run's detections carry — the
    choices for the heading sampler's floor (§26 *Headings*). Empty when none are recorded.
    """
    from detail.core import atomic
    from detail.storage.zarr_io import DetectionPartReader

    recording_id, run_id = video
    rec_dir = experiment.get_recording(recording_id).dir
    track_yaml = rec_dir / "tracking" / run_id / "track.yaml"
    if not track_yaml.is_file():
        return []
    det_run = yaml_load.safe_load(atomic.read_text(track_yaml)).get("detection_run_id")
    parts = sorted((rec_dir / "detections" / str(det_run)).glob("part_*.zarr"))
    if not parts:
        return []
    reader = DetectionPartReader(parts[0])
    if "class_name" not in reader.column_names:
        return []
    return sorted({str(c) for c in reader.column("class_name")[:]})


def load_video(experiment, recording_id: str, tracking_run_id: str) -> dict:
    """Off the UI thread: what the player needs for one recording — its tracking preview (the
    frame source), the part ids and per-part frame counts, the rate, the timestamp map, and
    every track's box on every frame (``pipeline.subject_frames.tracked_frames``)."""
    from detail.pipeline import subject_frames

    recording = experiment.get_recording(recording_id)
    preview = recording.tracking_preview(tracking_run_id)  # off-thread
    boxes: dict[tuple[str, int], dict[int, tuple]] = {}
    for item in subject_frames.tracked_frames(recording.dir, tracking_run_id):
        boxes.setdefault((item["part_id"], item["frame_idx"]), {})[item["track_id"]] = (
            item["bbox"]
        )
    counts = list(preview.part_frame_counts)
    part_ids = [recording_store.part_stem(part) for part in recording.video_parts]
    return {
        "recording_id": recording_id,
        "tracking_run_id": tracking_run_id,
        "preview": preview,
        "part_ids": part_ids,
        "counts": counts,
        "offsets": [int(sum(counts[:i])) for i in range(len(counts))],
        "boxes": boxes,
        "tracks": sorted({t for per in boxes.values() for t in per}),
        "fps": recording.video_fps,
        # The timestamp map the preview already holds: the player's time readout (Tenet 8),
        # never a frame-rate guess.
        "timestamps": preview.timestamps(),
    }


# --------------------------------------------------------------------------- #
# Saving marks off the UI thread
# --------------------------------------------------------------------------- #
class _MarkSaver:
    """Persist one frame's marks off the UI thread: one write in flight, the latest marks per
    frame win. A click lands on screen at once from the dialog's own copy of the set; this is
    the part that reaches the disk (``Experiment.set_example_marks``, atomic)."""

    def __init__(self, dialog: QWidget, experiment, set_id: str) -> None:
        self._dialog = dialog
        self._experiment = experiment
        self._set_id = set_id
        self._pending: dict[int, list[ExampleMark]] = {}
        self._worker: BackgroundCall | None = None

    @property
    def busy(self) -> bool:
        return self._worker is not None or bool(self._pending)

    def save(self, index: int, marks: list[ExampleMark]) -> None:
        self._pending[index] = [ExampleMark.from_dict(m.to_dict()) for m in marks]
        self._flush()

    def _flush(self) -> None:
        if self._worker is not None or not self._pending:
            return
        index = next(iter(self._pending))
        marks = self._pending.pop(index)
        experiment, set_id = self._experiment, self._set_id
        self._worker = BackgroundCall(
            lambda: experiment.set_example_marks(set_id, index, marks),  # off-thread
            receiver=self._dialog,
        )
        self._worker.done.connect(self._done)
        self._worker.failed.connect(self._failed)
        self._worker.start()

    def _done(self, _result) -> None:
        self._worker = None
        self._flush()
        if not self.busy:
            self._dialog._saves_settled()

    def _failed(self, exc: BaseException) -> None:
        self._worker = None
        self._pending.clear()
        self._dialog._save_failed(exc)


class _PlayerColumn(QWidget):
    """The left column: the video pinned to the top, sized to its own shape, and what is
    ``below`` it directly under the transport.

    The second copy of ``behavior_query._PlayerColumn``, for the same reason it exists there: a
    top-pinned block in a stretched column gathers the spare height into one gap under its
    transport, right between the playback buttons and the marking controls that work with
    them. A third copy is the point at which this becomes one module.
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
        self._refit = QTimer(self)
        self._refit.setSingleShot(True)
        self._refit.setInterval(0)
        self._refit.timeout.connect(self._fit)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt virtual
        super().resizeEvent(event)
        self._fit()

    def event(self, event) -> bool:  # noqa: N802 - Qt virtual
        if event.type() == QEvent.Type.LayoutRequest:
            self._refit.start()
        return super().event(event)

    def _fit(self) -> None:
        bound_picture(self._block, self.width(), self._block.minimumSizeHint().height())


# --------------------------------------------------------------------------- #
# The labeling screen
# --------------------------------------------------------------------------- #
class KeypointLabelingDialog(Dialog):
    """A video player with an example set beside it: find a frame, click its keypoints.

    ``sources`` maps recording id -> tracking run id when a launch form passes the recordings
    it has checked; None offers every recording's complete tracking runs.
    """

    def __init__(
        self,
        experiment,
        set_id: str,
        *,
        sources: dict[str, str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._experiment = experiment
        self._sources = dict(sources) if sources else None
        self._set: ExampleSet = experiment.example_set(set_id)
        self._saver = _MarkSaver(self, experiment, set_id)
        self._names: dict[str, str] = {}
        self._offered: list[dict] = []
        self._current: str | None = None
        # The frame `_current` was chosen on. A new frame starts again at its first unmarked
        # point; carrying the last frame's point over made the order drift (tail, head, ...).
        self._current_for: tuple | None = None
        self._video: dict | None = None  # load_video's payload for the video on screen
        self._raw: tuple | None = (
            None  # (recording_id, part_id, local, rgb) under the playhead
        )
        self._stored: int | None = None  # an example frame shown from its stored image
        self._stored_rgb: np.ndarray | None = None
        self._pending_seek: tuple | None = None  # (part_id, frame_idx, track_id)
        self.setWindowTitle(f"Mark example frames — {self._set.name} ({self._set.id})")

        layout = self.body()
        top = QHBoxLayout()
        top.addWidget(QLabel("Video:"))
        self._recording = QComboBox()
        self._recording.activated.connect(lambda _i: self._recording_chosen())
        top.addWidget(self._recording, 1)
        top.addWidget(QLabel("Track:"))
        self._track = QComboBox()
        self._track.activated.connect(lambda _i: self._compose())
        top.addWidget(self._track)
        top.addStretch()
        # The set's skeleton, changeable until the first mark (§26): marks name points, so
        # after that a different skeleton is a new set.
        top.addWidget(QLabel("Skeleton:"))
        self._skeleton = QComboBox()
        self._skeleton.activated.connect(lambda _i: self._skeleton_chosen())
        top.addWidget(self._skeleton)
        self._fill_skeletons()
        layout.addLayout(top)
        self._notice = QLabel("")
        self._notice.setWordWrap(True)
        layout.addWidget(self._notice)

        self._view = ImageView(zoomable=True)
        self._view.setMinimumSize(lines(28), lines(20))
        self._view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._view.clicked_at.connect(self._clicked)
        # A placed point can be dragged to move it (the scene editor's and the skeleton
        # editor's drag), beside re-placing it with 1-9 and a click.
        self._dragging: str | None = None
        self._view.hit_test = lambda pos: self._mark_near(pos) is not None
        self._view.mark_drag_started.connect(self._drag_started)
        self._view.mark_dragged.connect(self._dragged)
        self._view.mark_drag_finished.connect(self._drag_finished)
        self._scrubber = FrameScrubber(playable=True)
        self._scrubber.set_decoder(self._compute)
        self._scrubber.rendered.connect(self._display)
        self._scrubber.frame_changed.connect(self._frame_moved)
        self._decode_busy = decode_overlay(self._scrubber, self._view)
        self._block = PictureWithControls(self._view, self._scrubber, top=True)
        annotate = self._build_annotate_box()
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(self.fontMetrics().height() * 2 + 4)

        self._split = GripSplitter(Qt.Orientation.Horizontal, key="keypoint_labeling")
        self._split.addWidget(_PlayerColumn(self._block, [annotate, self._status]))
        side = self._build_side_panel()
        self._split.addWidget(side)
        self._split.setStretchFactor(0, 3)
        self._split.setStretchFactor(1, 2)
        self._split.set_foldable(1, "Keypoints")
        if not self._split.remember():
            self._split.setSizes([760, 440])
        layout.addWidget(self._split, 1)

        self._ready = QLabel("")
        self._ready.setWordWrap(True)
        layout.addWidget(self._ready)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Close
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Use these frames")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        for button in buttons.buttons():
            button.setAutoDefault(False)
        self.set_buttons(buttons)

        self._refresh_lists()
        self._view.clear_frame("Reading recordings…")
        experiment_, sources_, set_ = experiment, self._sources, self._set
        run_busy(
            self._view,
            lambda: offered_videos(experiment_, sources_, set_),
            self._videos_ready,
            label="Reading recordings…",
            on_error=lambda exc: self._view.clear_frame(f"(could not read: {exc})"),
        )

    # -- building ------------------------------------------------------------ #
    def _build_annotate_box(self) -> QWidget:
        """Under the playback row: which point is being placed, the tools, and the two
        decisions a click cannot make."""
        box = QFrame()
        box.setFrameShape(QFrame.Shape.StyledPanel)
        column = QVBoxLayout(box)
        self._annotating = QLabel("")
        self._annotating.setFont(emphasis_font(self._annotating, steps=3))
        self._annotating.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._annotating)
        tools = QHBoxLayout()
        tools.addStretch()
        tools.addWidget(QLabel("Tool:"))
        self._tools = QButtonGroup(box)
        self._tool_buttons: dict[str, QPushButton] = {}
        for kind, label in (("point", "Point"), ("region", "Region"), ("box", "Box")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setAutoDefault(False)
            self._tools.addButton(button)
            self._tool_buttons[kind] = button
            tools.addWidget(button)
            set_available(
                button,
                kind == "point",
                f"A {label.lower()} mark is reserved in the schema; nothing reads one yet "
                "(DESIGN_PLAN §26).",
            )
        self._tool_buttons["point"].setChecked(True)
        tools.addStretch()
        column.addLayout(tools)
        decisions = QHBoxLayout()
        self._hidden_button = QPushButton("Not visible (N)")
        self._hidden_button.setAutoDefault(False)
        self._hidden_button.clicked.connect(self._mark_hidden)
        self._clear_button = QPushButton("Clear (Backspace)")
        self._clear_button.setAutoDefault(False)
        self._clear_button.clicked.connect(self._clear_current)
        decisions.addWidget(self._hidden_button, 1)
        decisions.addWidget(self._clear_button, 1)
        column.addLayout(decisions)
        return box

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        side = QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 0, 0)
        side.addWidget(QLabel("Keypoints (1-9 to choose):"))
        self._keypoints = QListWidget()
        self._keypoints.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._keypoints.currentRowChanged.connect(self._keypoint_chosen)
        side.addWidget(self._keypoints, 2)

        add_row = QHBoxLayout()
        self._n_frames = IntSpin()
        self._n_frames.setRange(1, 50)
        self._n_frames.setValue(5)
        add_row.addWidget(self._n_frames)
        self._strategy = QComboBox()
        for strategy in subject_frames.STRATEGY_TABLE:
            self._strategy.addItem(strategy.label, strategy.id)
            self._strategy.setItemData(
                self._strategy.count() - 1, strategy.help, Qt.ItemDataRole.ToolTipRole
            )
        self._strategy.setToolTip(
            "  ".join(f"{s.label}: {s.help}" for s in subject_frames.STRATEGY_TABLE)
        )
        add_row.addWidget(self._strategy, 1)
        self._add_frames = QPushButton("Add frames")
        self._add_frames.setAutoDefault(False)
        self._add_frames.clicked.connect(self._sample)
        add_row.addWidget(self._add_frames)
        # Which tracking Add frames (and By heading's pool) samples from, asked when a
        # recording has several — each animal's track and their merge, in a social
        # experiment. A recording's runs were once folded into one {recording: run} map,
        # so the last run listed (the merge) always won, whatever animal the set was for.
        self._sample_tracking_row = QWidget()
        tracking_row = QHBoxLayout(self._sample_tracking_row)
        tracking_row.setContentsMargins(0, 0, 0, 0)
        tracking_label = QLabel("From tracking")
        tracking_label.setToolTip(
            "Which tracking Add frames samples from, for recordings with several: an "
            "animal's own track gives frames of that animal only; a merged track gives "
            "frames of every animal it follows."
        )
        tracking_row.addWidget(tracking_label)
        self._sample_tracking = QComboBox()
        self._sample_tracking.setToolTip(tracking_label.toolTip())
        self._sample_tracking.currentIndexChanged.connect(self._sample_tracking_changed)
        tracking_row.addWidget(self._sample_tracking, 1)
        self._sample_tracking_row.hide()
        side.addWidget(self._sample_tracking_row)
        side.addLayout(add_row)
        subset_row = QHBoxLayout()
        recordings_label = QLabel("Recordings")
        recordings_label.setToolTip(
            "How many recordings the sample reads, in total. A varied sample reads the shape "
            "of every tracked frame of each one, so fewer is faster."
        )
        subset_row.addWidget(recordings_label)
        self._n_recordings = IntSpin()
        self._n_recordings.setRange(1, 1)
        subset_row.addWidget(self._n_recordings)
        across_label = QLabel("spread across")
        across_label.setToolTip(
            "Split those recordings evenly across the values of a metadata column (an "
            "acquisition setup, a cohort), so every value is read. Recording order spreads "
            "them through the experiment as imported."
        )
        subset_row.addWidget(across_label)
        self._spread_by = QComboBox()
        self._spread_by.addItem("recording order", None)
        subset_row.addWidget(self._spread_by, 1)
        side.addLayout(subset_row)
        # The heading sampler's controls: shown once the offered recordings are known, and
        # only when one of them is not top down (`_apply_heading_gate`).
        self._heading_box = QWidget()
        heading_column = QVBoxLayout(self._heading_box)
        heading_column.setContentsMargins(0, 0, 0, 0)
        floor_row = QHBoxLayout()
        floor_label = QLabel("Floor")
        floor_label.setToolTip(
            "By heading: off by default, which works with any detector. If your detector "
            "segments the arena floor (a hot plate, an open-field base), choose that class: "
            "from a 45° or side view the floor is squashed in the picture, and its outline "
            "lets walking toward and away from the camera be read, so those two directions "
            "are found more reliably."
        )
        floor_row.addWidget(floor_label)
        self._floor = QComboBox()
        self._floor.addItem("no floor", None)
        floor_row.addWidget(self._floor, 1)
        heading_column.addLayout(floor_row)
        self._compass = HeadingCompass()
        self._compass.clicked.connect(self._review_direction)
        heading_column.addWidget(self._compass)
        self._heading_box.hide()
        side.addWidget(self._heading_box)
        self._heading_note = QLabel(
            "Top-down camera: sampling by heading is for side and angled views."
        )
        self._heading_note.setWordWrap(True)
        self._heading_note.setToolTip(
            "Every recording here has the camera view Top down, and the directions By "
            "heading reads (toward, away from the camera) are not visible from above. If "
            "that view is wrong, change it at the top of the Recordings page, or for one "
            "recording under its Scene geometry."
        )
        self._heading_note.hide()
        side.addWidget(self._heading_note)
        self._pool = None
        self._pool_request = None
        for control in (self._n_recordings, self._spread_by, self._floor):
            signal = (
                getattr(control, "valueChanged", None) or control.currentIndexChanged
            )
            signal.connect(self._forget_pool)
        self._strategy.currentIndexChanged.connect(self._strategy_changed)
        self._strategy_changed()
        side.addWidget(QLabel("Example frames (D / A to step):"))
        self._frames = QListWidget()
        self._frames.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._frames.itemClicked.connect(
            lambda item: self._go_to(self._frames.row(item))
        )
        side.addWidget(self._frames, 3)
        self._remove_frame = QPushButton("Remove frame")
        self._remove_frame.setAutoDefault(False)
        self._remove_frame.clicked.connect(self._remove)
        side.addWidget(self._remove_frame, alignment=Qt.AlignmentFlag.AlignLeft)
        legend = QLabel(
            "Click the animal: places the highlighted keypoint\n"
            "and adds the frame to the set.\n"
            "Space plays and pauses; ←/→ step (Shift: big steps).\n"
            "Wheel: zoom    drag: pan\n"
            "Dots: a pair's left member blue, its right vermillion,\n"
            "midline green; the box is the tracked subject."
        )
        side.addWidget(legend)
        # Height the row-fitted lists do not take collects here, not between the labels —
        # which is where it went once the compass was gone.
        side.addStretch(1)
        return panel

    # -- public -------------------------------------------------------------- #
    @property
    def set_id(self) -> str:
        return self._set.id

    def example_set(self) -> ExampleSet:
        return self._set

    def image_view(self) -> ImageView:
        return self._view

    def scrubber(self) -> FrameScrubber:
        return self._scrubber

    def saving(self) -> bool:
        """Whether a click's save is still in flight."""
        return self._saver.busy

    # -- the videos ---------------------------------------------------------- #
    def _videos_ready(self, offered: dict) -> None:
        self._names = offered["names"]
        self._offered = offered["videos"]
        self._spread_by.clear()
        self._spread_by.addItem("recording order", None)
        for column in offered["columns"]:
            self._spread_by.addItem(column, column)
        self._floor.clear()
        self._floor.addItem("no floor", None)
        labels = offered.get("class_labels") or {}
        for name in offered.get("classes", []):
            self._floor.addItem(class_names.label(name, labels), name)
        self._apply_heading_gate(offered["heading_applies"])
        n_recordings = max(len({v["recording_id"] for v in self._offered}), 1)
        self._n_recordings.setRange(1, n_recordings)
        self._n_recordings.setValue(
            min(subject_frames.DEFAULT_FROM_RECORDINGS, n_recordings)
        )
        self._strategy_changed()
        self._recording.clear()
        for video in self._offered:
            self._recording.addItem(
                video["label"], (video["recording_id"], video["run_id"])
            )
        if not self._offered:
            self._recording.addItem("(no tracked video)", None)
            set_available(
                self._recording,
                False,
                "No recording has a complete tracking run; run tracking first.",
            )
            set_available(
                self._add_frames, False, "No recording has a complete tracking run."
            )
            self._view.clear_frame("No tracked video to show: run tracking first.")
            self._refresh_lists()
            if self._set.frames:
                self._go_to(0)
            return
        self._fill_sample_tracking()
        self._refresh_lists()
        self._recording_chosen()

    # -- which tracking Add frames samples from -------------------------------- #
    def _fill_sample_tracking(self) -> None:
        """Offer *From tracking* when a recording has several tracking runs: one entry per
        kind of run (its classes and tracker), with how many recordings have it. Opens on a
        choose row, never a guess (Tenet 9)."""
        per_recording: dict[str, int] = {}
        for video in self._offered:
            rid = video["recording_id"]
            per_recording[rid] = per_recording.get(rid, 0) + 1
        several = any(n > 1 for n in per_recording.values())
        kinds: dict[str, set] = {}
        for video in self._offered:
            kinds.setdefault(video.get("tracking") or "", set()).add(
                video["recording_id"]
            )
        self._sample_tracking.blockSignals(True)
        self._sample_tracking.clear()
        self._sample_tracking.addItem("(choose the tracking to sample from)", None)
        total = len(per_recording)
        for kind in sorted(kinds):
            self._sample_tracking.addItem(
                f"{kind}  ·  {len(kinds[kind])}/{total} recordings", kind
            )
        self._sample_tracking.blockSignals(False)
        self._sample_tracking_row.setVisible(several)
        self._sync_add_frames()

    def _sample_tracking_changed(self, *_args) -> None:
        self._forget_pool()  # the heading pool was built from the previous choice
        self._sync_add_frames()

    def sample_sources(self) -> dict[str, str]:
        """``{recording_id: tracking run id}`` Add frames and By heading sample from: every
        offered recording's run of the chosen *From tracking* kind; with one run per
        recording, that run. Empty while a choice is needed and not made."""
        if self._sample_tracking_row.isHidden():
            return {v["recording_id"]: v["run_id"] for v in self._offered}
        kind = self._sample_tracking.currentData()
        if kind is None:
            return {}
        return {
            v["recording_id"]: v["run_id"]
            for v in self._offered
            if v.get("tracking") == kind
        }

    def _sync_add_frames(self) -> None:
        if not self._offered:
            return
        if not self.sample_sources():
            set_available(
                self._add_frames,
                False,
                "Recordings here have several tracking runs: choose which one to sample "
                "from in From tracking",
            )
            return
        set_available(self._add_frames, True, "")

    def _recording_chosen(self) -> None:
        chosen = self._recording.currentData()
        if chosen is None:
            return
        recording_id, run_id = chosen
        self._stored = None
        self._video = None
        self._raw = None
        experiment = self._experiment
        self._view.clear_frame("Loading video…")
        run_busy(
            self._view,
            lambda: load_video(experiment, recording_id, run_id),
            self._video_loaded,
            label="Loading video…",
            on_error=lambda exc: self._view.clear_frame(f"(could not load: {exc})"),
        )

    def _video_loaded(self, video: dict) -> None:
        chosen = self._recording.currentData()
        if chosen is None or video["recording_id"] != chosen[0]:
            return  # another video was picked while this one loaded
        self._video = video
        self._track.clear()
        for track_id in video["tracks"]:
            self._track.addItem(f"track {track_id}", track_id)
        set_available(
            self._track, bool(video["tracks"]), "This tracking run has no tracks."
        )
        if video["fps"]:
            self._scrubber.set_playback_rate(video["fps"])
        self._scrubber.set_total(sum(video["counts"]))
        self._scrubber.set_timestamps(video["timestamps"])
        self._scrubber.set_available(True, "")
        target = 0
        if self._pending_seek is not None:
            part_id, frame_idx, track_id = self._pending_seek
            self._pending_seek = None
            target = self._global(part_id, frame_idx)
            index = self._track.findData(track_id)
            if index >= 0:
                self._track.setCurrentIndex(index)
        self._refresh_ticks()
        self._scrubber.set_frame(target)
        self._view.setFocus()

    def _global(self, part_id: str, local: int) -> int:
        if self._video is None or part_id not in self._video["part_ids"]:
            return int(local)
        part = self._video["part_ids"].index(part_id)
        return int(self._video["offsets"][part] + local)

    def _locate(self, frame: int) -> tuple[int, int]:
        counts = self._video["counts"] if self._video else []
        for part, count in enumerate(counts):
            if frame < count:
                return part, frame
            frame -= count
        return max(len(counts) - 1, 0), max((counts or [1])[-1] - 1, 0)

    def _compute(self, global_frame: int):
        """Off-thread: the raw frame under the playhead (the overlay is drawn on the UI
        thread, so a click redraws without decoding again)."""
        video = self._video
        if video is None:
            return ("text", "Loading video…")
        part, local = self._locate(global_frame)
        result, reason = frame_source.decode(
            lambda: video["preview"].frame(part, local, with_masks=False)
        )
        if reason is not None:
            return ("text", frame_source.unavailable_text(reason))
        actual, rgb, _detections = result
        return ("img", video["recording_id"], video["part_ids"][part], int(actual), rgb)

    def _display(self, _frame: int, payload) -> None:
        if payload[0] == "text":
            self._raw = None
            self._view.clear_frame(payload[1])
            return
        if self._stored is not None:
            return  # a stored example image is on screen; the player is parked
        _kind, recording_id, part_id, actual, rgb = payload
        self._raw = (recording_id, part_id, actual, rgb)
        self._compose()

    def _frame_moved(self, _frame: int) -> None:
        self._stored = None

    # -- what is on screen --------------------------------------------------- #
    def _here(self) -> tuple | None:
        """``(recording_id, part_id, frame_idx, track_id)`` of the frame on screen, or None."""
        if self._raw is None:
            return None
        recording_id, part_id, local, _rgb = self._raw
        return recording_id, part_id, local, self._track.currentData()

    def _index_of(self, key) -> int | None:
        if key is None:
            return None
        recording_id, part_id, frame_idx, track_id = key
        for i, f in enumerate(self._set.frames):
            if (f.recording_id, f.part_id, f.frame_idx, f.track_id) == (
                recording_id,
                part_id,
                frame_idx,
                track_id,
            ):
                return i
        return None

    def _on_screen(self) -> int | None:
        """The example frame on screen, if the frame shown is one."""
        if self._stored is not None:
            return self._stored
        return self._index_of(self._here())

    def _box_here(self):
        here = self._here()
        if here is None or self._video is None:
            return None
        _recording_id, part_id, local, track_id = here
        return self._video["boxes"].get((part_id, local), {}).get(track_id)

    def _compose(self) -> None:
        index = self._on_screen()
        if self._stored is not None:
            if self._stored_rgb is None:
                self._refresh_frame()
                return
            rgb = self._stored_rgb
            box = self._set.frames[self._stored].bbox
        elif self._raw is not None:
            rgb = self._raw[3]
            box = self._box_here()
        else:
            self._refresh_frame()
            return
        marks = self._set.frames[index].marks if index is not None else []
        shown_key = (
            ("stored", self._stored) if self._stored is not None else self._here()
        )
        if (
            shown_key != self._current_for
            or self._current not in self._set.skeleton.names
        ):
            self._current = self._first_unmarked(marks)
            self._current_for = shown_key
        shown = draw_example(rgb, self._set.skeleton, marks, self._current, box)
        self._view.set_frame(to_pixmap(shown))
        self._block.set_aspect(self._view.pixmap_size())
        self._refresh_frame()

    # -- the gestures -------------------------------------------------------- #
    def _clicked(self, u: float, v: float) -> None:
        if self._scrubber.is_playing():
            self._scrubber.pause()  # a click on a moving picture stops it, places nothing
            return
        if self._current is None:
            return
        self._place(ExampleMark(target=self._current, x=float(u), y=float(v)))

    def _mark_hidden(self) -> None:
        if self._current is None:
            return
        self._place(ExampleMark(target=self._current, visible=False))

    # -- dragging a placed point --------------------------------------------------- #
    _GRAB_PX = 10.0

    def _mark_near(self, pos) -> str | None:
        """The placed, visible mark on the frame on screen within reach of widget point
        ``pos``, or None (nothing to grab: a click places the current point as before).
        """
        index = self._on_screen()
        if index is None or self._scrubber.is_playing():
            return None
        best, best_d = None, self._GRAB_PX**2
        for mark in self._set.frames[index].marks:
            if not mark.visible or mark.x is None or mark.y is None:
                continue
            where = self._view.widget_position(float(mark.x), float(mark.y))
            if where is None:
                continue
            d = (where.x() - pos.x()) ** 2 + (where.y() - pos.y()) ** 2
            if d <= best_d:
                best, best_d = mark.target, d
        return best

    def _drag_started(self, u: float, v: float) -> None:
        where = self._view.widget_position(u, v)
        self._dragging = self._mark_near(where) if where is not None else None
        if self._dragging is not None:
            self._current = self._dragging
            self._compose()

    def _dragged(self, u: float, v: float) -> None:
        index = self._on_screen()
        if self._dragging is None or index is None:
            return
        mark = self._set.frames[index].mark_for(self._dragging)
        if mark is not None:
            mark.x, mark.y = float(u), float(v)  # in memory until the button comes up
            self._compose()

    def _drag_finished(self, u: float, v: float) -> None:
        index = self._on_screen()
        target, self._dragging = self._dragging, None
        if target is None or index is None:
            return
        frame = self._set.frames[index]
        frame.marks = [m for m in frame.marks if m.target != target] + [
            ExampleMark(target=target, x=float(u), y=float(v))
        ]
        self._saver.save(index, frame.marks)
        self._refresh_lists()
        self._compose()  # the moved point stays current: a drag is not a placement

    def _place(self, mark: ExampleMark) -> None:
        """Put ``mark`` on the frame on screen, adding the frame to the set first when it is
        not in it yet."""
        index = self._on_screen()
        if index is not None:
            self._save(index, mark)
            return
        here = self._here()
        if here is None:
            return
        if self._box_here() is None:
            self._status.setText(
                f"Track {here[3]} has no subject on this frame, so it cannot be an "
                "example: there is no tracked box."
            )
            return
        recording_id, part_id, local, track_id = here
        experiment, set_id = self._experiment, self._set.id
        run_id = self._video["tracking_run_id"]
        run_busy(
            self._view,
            lambda: experiment.add_example_frame(
                set_id, recording_id, run_id, part_id, local, track_id
            ),
            lambda example_set: self._added(example_set, here, mark),
            label="Adding frame…",
            on_error=lambda exc: QMessageBox.warning(
                self, "Add frame", f"Could not add this frame:\n{exc}"
            ),
        )

    def _added(self, example_set: ExampleSet, key, mark: ExampleMark) -> None:
        self._set = example_set
        self._refresh_lists()
        self._refresh_ticks()
        index = self._index_of(key)
        if index is not None:
            self._save(index, mark)

    def _save(self, index: int, mark: ExampleMark) -> None:
        frame = self._set.frames[index]
        frame.marks = [m for m in frame.marks if m.target != mark.target] + [mark]
        self._saver.save(index, frame.marks)
        self._refresh_lists()
        self._advance(index)

    def _clear_current(self) -> None:
        index = self._on_screen()
        if self._current is None or index is None:
            return
        frame = self._set.frames[index]
        kept = [m for m in frame.marks if m.target != self._current]
        if len(kept) != len(frame.marks):
            frame.marks = kept
            self._saver.save(index, frame.marks)
            self._refresh_lists()
        self._compose()

    def _saves_settled(self) -> None:
        self._refresh_lists()

    def _save_failed(self, exc: BaseException) -> None:
        QMessageBox.warning(self, "Save marks", f"Could not save:\n{exc}")

    def _first_unmarked(self, marks: list[ExampleMark]) -> str | None:
        names = self._set.skeleton.names
        marked = {m.target for m in marks}
        return next((n for n in names if n not in marked), names[0] if names else None)

    def _advance(self, index: int) -> None:
        """The next keypoint not yet marked on this frame, after the current one."""
        names = list(self._set.skeleton.names)
        marked = {m.target for m in self._set.frames[index].marks}
        start = names.index(self._current) if self._current in names else -1
        order = names[start + 1 :] + names[: start + 1]
        self._current = next((n for n in order if n not in marked), self._current)
        self._compose()

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if event.key() == Qt.Key.Key_Escape:
            super().keyPressEvent(event)
            return
        key = event.key()
        names = self._set.skeleton.names
        if key == Qt.Key.Key_N:
            self._mark_hidden()
        elif key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self._clear_current()
        elif Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            index = key - Qt.Key.Key_1
            if index < len(names):
                self._keypoints.setCurrentRow(index)
        elif key == Qt.Key.Key_D:
            self._step_example(+1)
        elif key == Qt.Key.Key_A:
            self._step_example(-1)
        else:
            # Everything else is the player's: Space, the arrows, Home/End, Page Up/Down —
            # the same keys every viewer in the app answers to.
            self._scrubber.keyPressEvent(event)

    # -- example frames ------------------------------------------------------ #
    def _step_example(self, step: int) -> None:
        if not self._set.frames:
            return
        current = self._on_screen()
        if current is None:
            current = -1 if step > 0 else len(self._set.frames)
        target = current + step
        if 0 <= target < len(self._set.frames):
            self._go_to(target)

    def _video_index(self, recording_id: str, run_id: str) -> int:
        for i, video in enumerate(self._offered):
            if (video["recording_id"], video["run_id"]) == (recording_id, run_id):
                return i
        return -1

    def _go_to(self, index: int) -> None:
        """Show example frame ``index``: seek its video, or show its stored image."""
        if not 0 <= index < len(self._set.frames):
            return
        self._frames.setCurrentRow(index)
        frame = self._set.frames[index]
        self._current = self._first_unmarked(frame.marks)
        where = self._video_index(frame.recording_id, frame.tracking_run_id)
        if where < 0:
            self._scrubber.pause()
            self._stored = index
            self._stored_rgb = None
            self._raw = None
            self._scrubber.set_available(
                False,
                "This frame's recording is not in this experiment; its stored image is shown.",
            )
            experiment, set_id = self._experiment, self._set.id
            self._view.clear_frame("Loading frame…")
            run_busy(
                self._view,
                lambda: (index, experiment.example_frame_image(set_id, index)),
                self._stored_loaded,
                label="Loading frame…",
                on_error=lambda exc: self._view.clear_frame(f"(could not load: {exc})"),
            )
            return
        if self._recording.currentIndex() != where or self._video is None:
            self._pending_seek = (frame.part_id, frame.frame_idx, frame.track_id)
            self._recording.setCurrentIndex(where)
            self._recording_chosen()
            return
        self._scrubber.set_available(True, "")
        self._stored = None
        track = self._track.findData(frame.track_id)
        if track >= 0:
            self._track.setCurrentIndex(track)
        self._scrubber.pause()
        self._scrubber.set_frame(self._global(frame.part_id, frame.frame_idx))
        self._view.setFocus()

    def _stored_loaded(self, result) -> None:
        index, rgb = result
        if index == self._stored:
            self._stored_rgb = rgb
            self._compose()

    def _apply_heading_gate(self, applies: bool) -> None:
        """Offer the heading sampler — *By heading*, the floor and the compass — only when
        a recording here is not top down (``Experiment.heading_applies``); otherwise one note
        says why it is missing. The API refuses the same requests, so the two agree."""
        offered = {self._strategy.itemData(i) for i in range(self._strategy.count())}
        if ("heading" in offered) != applies:
            chosen = self._strategy.currentData()
            self._strategy.blockSignals(True)
            self._strategy.clear()
            for strategy in subject_frames.STRATEGY_TABLE:
                if strategy.id == "heading" and not applies:
                    continue
                self._strategy.addItem(strategy.label, strategy.id)
                self._strategy.setItemData(
                    self._strategy.count() - 1,
                    strategy.help,
                    Qt.ItemDataRole.ToolTipRole,
                )
            self._strategy.setCurrentIndex(max(self._strategy.findData(chosen), 0))
            self._strategy.blockSignals(False)
        self._heading_box.setVisible(applies)
        self._heading_note.setVisible(not applies)

    def _strategy_changed(self) -> None:
        """The recordings controls apply only to a strategy that reads a subset; the floor and
        the per-direction count only to the heading sampler."""
        chosen = next(
            (
                s
                for s in subject_frames.STRATEGY_TABLE
                if s.id == self._strategy.currentData()
            ),
            subject_frames.STRATEGY_TABLE[0],
        )
        reason = f"{chosen.label} reads every recording."
        set_available(self._n_recordings, chosen.reads_subset, reason)
        set_available(
            self._spread_by,
            chosen.reads_subset and self._spread_by.count() > 1,
            (
                reason
                if not chosen.reads_subset
                else "The experiment has no metadata columns."
            ),
        )
        heading = chosen.id == "heading"
        set_available(
            self._floor,
            heading and self._floor.count() > 1,
            (
                "Only By heading reads a floor."
                if not heading
                else "The detections carry no class to use as the floor."
            ),
        )
        self._n_frames.setToolTip(
            "Frames per direction." if heading else "Frames to add."
        )

    def _forget_pool(self) -> None:
        """A different recording subset or floor is a different pool."""
        self._pool = None

    def _refresh_compass(self) -> None:
        from detail.pipeline import headings

        self._compass.set_coverage(headings.coverage(self._pool, self._set))

    def _review_direction(self, bucket: str) -> None:
        """Open the review window on ``bucket``, building the pool first if it is not held."""
        if self._pool is not None:
            self._open_review(bucket)
            return
        experiment = self._experiment
        sources = self.sample_sources()
        if not sources:
            return
        options = {
            "from_recordings": self._n_recordings.value(),
            "spread_by": self._spread_by.currentData(),
            "floor_class": self._floor.currentData(),
        }
        run_progress(
            self,
            lambda emit: experiment.heading_pool(sources, progress=emit, **options),
            lambda pool: self._pool_ready(pool, bucket),
            label="Reading headings…",
            on_error=self._sample_failed,
        )

    def _pool_ready(self, pool, bucket: str) -> None:
        self._pool = pool
        self._refresh_compass()
        self._open_review(bucket)

    def _open_review(self, bucket: str) -> None:
        dialog = HeadingReviewDialog(
            self._experiment, self._set.id, self._pool, bucket, self
        )
        dialog.changed.connect(self._review_changed)
        dialog.open_frame.connect(self._go_to)
        dialog.open()

    def _review_changed(self) -> None:
        self._set = self._experiment.example_set(self._set.id)
        self._refresh_lists()
        self._refresh_ticks()

    def _sample(self) -> None:
        set_id, n = self._set.id, self._n_frames.value()
        strategy = self._strategy.currentData()
        from_recordings = self._n_recordings.value()
        spread_by = self._spread_by.currentData()
        heading = (
            {"floor_class": self._floor.currentData()} if strategy == "heading" else {}
        )
        experiment = self._experiment
        sources = self.sample_sources()
        set_available(self._add_frames, False, "Adding frames…")
        run_progress(
            self,
            lambda emit: experiment.sample_example_frames(
                set_id,
                sources,
                n,
                strategy=strategy,
                from_recordings=from_recordings,
                spread_by=spread_by,
                progress=emit,
                **heading,
            ),
            self._sampled,
            label="Adding frames…",
            on_error=self._sample_failed,
            # Cancellable, and cancelled with the window: closing it used to leave the copy
            # running, and opening it again started a second beside the first.
            on_cancel=self._sample_cancelled,
        )

    def _sampled(self, example_set: ExampleSet) -> None:
        self._sync_add_frames()
        before = len(self._set.frames)
        self._set = example_set
        self._refresh_lists()
        self._refresh_ticks()
        if len(example_set.frames) > before:
            self._go_to(before)

    def _sample_cancelled(self) -> None:
        """Nothing was added: a cancelled draw removes the pictures it had copied."""
        self._sync_add_frames()

    def _sample_failed(self, exc: BaseException) -> None:
        self._sync_add_frames()
        QMessageBox.warning(self, "Add frames", f"Could not add frames:\n{exc}")

    def _remove(self) -> None:
        index = self._frames.currentRow()
        if not 0 <= index < len(self._set.frames):
            return
        frame = self._set.frames[index]
        if frame.marks:
            answer = QMessageBox.question(
                self,
                "Remove frame",
                f"Frame {index + 1} has {len(frame.marks)} mark(s). Remove it from the set?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        experiment, set_id = self._experiment, self._set.id
        run_busy(
            self._view,
            lambda: experiment.remove_example_frame(set_id, index),
            self._removed,
            label="Removing frame…",
            on_error=lambda exc: QMessageBox.warning(
                self, "Remove frame", f"Could not remove:\n{exc}"
            ),
        )

    def _removed(self, example_set: ExampleSet) -> None:
        self._set = example_set
        if self._stored is not None:
            self._stored = None
            self._scrubber.set_available(
                self._video is not None, "This frame's video is not loaded."
            )
        self._refresh_lists()
        self._refresh_ticks()
        self._compose()

    # -- drawing --------------------------------------------------------------- #
    def _keypoint_chosen(self, row: int) -> None:
        names = self._set.skeleton.names
        if 0 <= row < len(names) and names[row] != self._current:
            self._current = names[row]
            self._compose()

    def _frame_label(self, i: int) -> str:
        frame = self._set.frames[i]
        recording = self._names.get(frame.recording_id, frame.recording_id)
        k = len(self._set.skeleton)
        done = len({m.target for m in frame.marks} & set(self._set.skeleton.names))
        state = "all marked" if done == k else f"{done}/{k} marked"
        how = ""
        if frame.sampled_by:
            parts = frame.sampled_by.split(":")
            # a heading pick says which way it faces; the others say how they were chosen
            how = (
                f" ({parts[0]}: {parts[1]})"
                if parts[0] == "heading"
                else f" ({parts[0]})"
            )
        return f"{i + 1}. {recording}, frame {frame.frame_idx} · {state}{how}"

    # -- the skeleton, until the first mark ------------------------------------ #
    def _fill_skeletons(self) -> None:
        from detail.api import skeletons as library
        from detail.gui.example_set_picker import fill_skeleton_combo, skeleton_of

        entries = library.skeletons()  # the machine library: a few small files
        self._skeleton.clear()
        fill_skeleton_combo(self._skeleton, entries)
        owner = skeleton_of(self._set, entries)
        if owner is None:
            self._skeleton.addItem(
                f"this set's own skeleton · {len(self._set.skeleton)} points", ""
            )
            owner = ""
        self._skeleton.setCurrentIndex(self._skeleton.findData(owner))
        self._sync_skeleton()

    def _sync_skeleton(self) -> None:
        marked = sum(1 for f in self._set.frames if f.marks)
        set_available(
            self._skeleton,
            not marked,
            f"{marked} frame(s) have marks, so the skeleton is fixed; make a new example set "
            "to use another.",
        )

    def _skeleton_chosen(self) -> None:
        chosen = self._skeleton.currentData()
        if not chosen:
            return
        try:
            self._set = self._experiment.update_example_set(
                self._set.id, skeleton=chosen
            )
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.warning(self, "Skeleton", str(exc))
            self._fill_skeletons()
            return
        self._current = None
        self._current_for = None
        self._fill_skeletons()
        self._refresh_lists()
        self._compose()

    def _refresh_lists(self) -> None:
        self._refresh_compass()
        selected = self._frames.currentRow()
        self._frames.clear()
        for i in range(len(self._set.frames)):
            self._frames.addItem(self._frame_label(i))
        if 0 <= selected < self._frames.count():
            self._frames.setCurrentRow(selected)
        try:
            self._set.check_runnable()
            self._ready.setText(
                "Ready: every keypoint is marked visible on at least one frame."
            )
        except ValueError as exc:
            self._ready.setText(f"Not ready yet: {exc}")
        set_available(
            self._remove_frame,
            bool(self._set.frames) and not self._saver.busy,
            "Saving…" if self._saver.busy else "The set has no frames yet.",
        )
        self._sync_skeleton()

    def _refresh_ticks(self) -> None:
        if self._video is None:
            return
        recording_id, run_id = (
            self._video["recording_id"],
            self._video["tracking_run_id"],
        )
        ticks = [
            self._global(f.part_id, f.frame_idx)
            for f in self._set.frames
            if (f.recording_id, f.tracking_run_id) == (recording_id, run_id)
        ]
        self._scrubber.set_marks(ticks=ticks)

    def _refresh_frame(self) -> None:
        names = self._set.skeleton.names
        index = self._on_screen()
        marks = self._set.frames[index].marks if index is not None else []
        by_target = {m.target: m for m in marks}
        self._keypoints.blockSignals(True)
        self._keypoints.clear()
        for j, name in enumerate(names):
            mark = by_target.get(name)
            state = (
                "not marked"
                if mark is None
                else ("placed" if mark.visible else "not visible")
            )
            self._keypoints.addItem(QListWidgetItem(f"{j + 1}. {name} — {state}"))
        if self._current in names:
            self._keypoints.setCurrentRow(names.index(self._current))
        self._keypoints.blockSignals(False)
        if index is not None:
            self._frames.setCurrentRow(index)
        on_frame = self._raw is not None or self._stored is not None
        for control in (self._hidden_button, self._clear_button):
            set_available(control, on_frame, "Find a frame in the video first.")
        self._annotating.setText(
            f"Annotating: {self._current}" if self._current else "Annotating: (nothing)"
        )
        if self._stored is not None:
            frame = self._set.frames[self._stored]
            where = (
                f"Example frame {self._stored + 1} of {len(self._set.frames)} (stored "
                f"image) · {frame.recording_id}, frame {frame.frame_idx}"
            )
        elif self._raw is not None:
            recording_id, part_id, local, _rgb = self._raw
            name = self._names.get(recording_id, recording_id)
            where = f"{name}, frame {local}" + (
                f" ({part_id})"
                if self._video and len(self._video["part_ids"]) > 1
                else ""
            )
            if index is not None:
                where += f" · example frame {index + 1} of {len(self._set.frames)}"
            elif self._box_here() is None:
                where += " · no tracked subject on this frame for the chosen track"
            else:
                where += " · not in the set yet: a click adds it"
        else:
            where = ""
        marked = len(set(by_target) & set(names))
        self._status.setText(
            where
            + (
                f" · {marked} of {len(names)} keypoints marked"
                if index is not None
                else ""
            )
        )


# --------------------------------------------------------------------------- #
# Entry point (PR 4's launch form imports it lazily)
# --------------------------------------------------------------------------- #
def open_labeling(
    parent, experiment, set_id: str, *, sources: dict[str, str] | None = None
) -> bool:
    """Open the labeling screen modally; True when it closed with *Use these frames*."""
    dialog = KeypointLabelingDialog(experiment, set_id, sources=sources, parent=parent)
    return dialog.exec() == Dialog.DialogCode.Accepted
