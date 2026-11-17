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
  video offered.

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

import threading
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

from detail.core.examples import ExampleMark, ExampleSet
from detail.core.skeleton import Skeleton
from detail.core.visualize import draw_bbox, draw_keypoints
from detail.gui import frame_source
from detail.gui.busy import decode_overlay, run_busy, run_progress
from detail.gui.dialog import Dialog
from detail.gui.full_frame import to_pixmap
from detail.gui.grip_splitter import GripSplitter
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
from detail.storage import recording as recording_store

#: The tracked box the method crops to — drawn thin, so it frames the animal without covering
#: the points being placed on it.
_BOX_COLOR = (255, 220, 60)
_STRATEGIES = (("Varied frames", "varied"), ("Spread over time", "spread"))


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
    per_recording = {}
    for rid, _run in videos:
        per_recording[rid] = per_recording.get(rid, 0) + 1
    return {
        "names": names,
        "videos": [
            {
                "recording_id": rid,
                "run_id": run,
                "label": (
                    f"{names.get(rid, rid)} ({run})"
                    if per_recording[rid] > 1
                    else names.get(rid, rid)
                ),
            }
            for rid, run in videos
        ],
    }


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
    """Persist one frame's marks off the UI thread: one write at a time, the latest marks per
    frame win. A click lands on screen at once from the dialog's own copy of the set; this is
    the part that reaches the disk (``Experiment.set_example_marks``, atomic).

    The writes run on the app's write queue for this set (``gui/run_writes.py``), not on the
    dialog: a queue whose next write started from the previous one's ``done`` lost every queued
    frame after the one in flight when the dialog closed, because that delivery is dropped once
    its receiver is gone (#24). A queued frame's write reads the latest marks for it when it
    runs, so clicks on a frame still waiting fold into its one write.
    """

    def __init__(self, dialog: QWidget, experiment, set_id: str) -> None:
        from pathlib import Path

        from detail.gui import run_writes

        self._dialog = dialog
        self._experiment = experiment
        self._set_id = set_id
        root = getattr(experiment, "dir", None)
        where = str(Path(root).resolve()) if root is not None else f"{id(experiment)}"
        self._queue = run_writes.queue_for_key(f"example-set:{where}:{set_id}".lower())
        #: Marks per frame index not yet taken by a write: read by the write when it runs.
        self._latest: dict[int, list[ExampleMark]] = {}
        #: The check-and-fold in :meth:`save` and the take in :meth:`_write` are one step each:
        #: a take landing between the check and the fold would leave the fold unwritten.
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._queue.busy()

    def save(self, index: int, marks: list[ExampleMark]) -> None:
        copy = [ExampleMark.from_dict(m.to_dict()) for m in marks]
        with self._lock:
            waiting = index in self._latest
            self._latest[index] = copy
        if waiting:
            return  # the frame's queued write has not run yet, and will read these
        self._queue.submit(
            lambda: self._write(index),
            owner=self._dialog,
            done=self._done,
            failed=self._failed,
            title="Marks not saved",
        )

    def _write(self, index: int):  # off-thread
        with self._lock:
            marks = self._latest.pop(index, None)
        if marks is None:
            return None
        experiment, set_id = self._experiment, self._set_id
        return experiment.set_example_marks(set_id, index, marks)  # off-thread

    def _done(self, _result) -> None:
        if not self.busy:
            self._dialog._saves_settled()

    def _failed(self, exc: BaseException) -> None:
        with self._lock:
            self._latest.clear()
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
        self.resize(1260, 740)  # fits 1440x810 (gui_design.md §3.1)

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
        for label, value in _STRATEGIES:
            self._strategy.addItem(label, value)
        self._strategy.setToolTip(
            "Varied: postures that differ, the typical ones first (reads the subject's shape "
            "on every tracked frame). Spread: one frame per equal time slice."
        )
        add_row.addWidget(self._strategy, 1)
        self._add_frames = QPushButton("Add frames")
        self._add_frames.setAutoDefault(False)
        self._add_frames.clicked.connect(self._sample)
        add_row.addWidget(self._add_frames)
        side.addLayout(add_row)
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
        set_available(self._add_frames, True, "")
        self._refresh_lists()
        self._recording_chosen()

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
        set_available(self._scrubber, True, "")
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
            set_available(
                self._scrubber,
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
        set_available(self._scrubber, True, "")
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

    def _sample(self) -> None:
        set_id, n = self._set.id, self._n_frames.value()
        strategy = self._strategy.currentData()
        experiment = self._experiment
        sources = {v["recording_id"]: v["run_id"] for v in self._offered}
        set_available(self._add_frames, False, "Adding frames…")
        run_progress(
            self,
            lambda emit: experiment.sample_example_frames(
                set_id, sources, n, strategy=strategy, progress=emit
            ),
            self._sampled,
            label="Adding frames…",
            on_error=self._sample_failed,
            # Cancellable, and cancelled with the window: closing it used to leave the copy
            # running, and opening it again started a second beside the first.
            on_cancel=self._sample_cancelled,
        )

    def _sampled(self, example_set: ExampleSet) -> None:
        set_available(self._add_frames, True, "")
        before = len(self._set.frames)
        self._set = example_set
        self._refresh_lists()
        self._refresh_ticks()
        if len(example_set.frames) > before:
            self._go_to(before)

    def _sample_cancelled(self) -> None:
        """Nothing was added: a cancelled draw removes the pictures it had copied."""
        set_available(self._add_frames, True, "")

    def _sample_failed(self, exc: BaseException) -> None:
        set_available(self._add_frames, True, "")
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
            set_available(self._scrubber, self._video is not None, "")
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
        how = f" ({frame.sampled_by.split(':')[0]})" if frame.sampled_by else ""
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
