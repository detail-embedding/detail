"""Quick-segment tool UI (§9): a config form (video + prompt + model + cap) and a
result view that plays the overlay clip back and exports it to mp4.

The config form is bespoke (like a ``ModuleUI``'s) because it needs non-scalar fields a
generic pydantic form can't render — a file picker, a prompt, and a backend combo that
tracks the model. The overlay player mirrors ``behavior_query._ClipPlayer``, reusing
``FrameScrubber`` + ``ImageView``; its decoder renders each frame's saved detections on
the fly (nothing overlay-shaped is stored)."""

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from detail.api import models as api_models
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.quick_segment import QuickSegmentConfig
from detail.core.dataclasses import Prompt, PromptTarget, TextCue
from detail.executor.runners import import_runner_modules
from detail.gui import frame_source
from detail.gui.compute_target import ComputeTargetRow
from detail.gui.confidence import ConfidenceRow, showing_caption
from detail.gui.forms import ConfigForm
from detail.gui.model_availability import (
    apply_availability,
    availability_note,
    selected_model_id,
    show_note,
)
from detail.gui.help_ui import field_label, help_button, label_with
from detail.gui._pydantic_form import field_meta, field_widget
from detail.gui.modules.registry import ConfigWidget
from detail.gui.status_color import MUTED, WARN, set_label_color
from detail.gui.run_preview import _to_pixmap
from detail.gui.scrubber import FrameScrubber
from detail.gui.tools.registry import ToolUI, register_tool_ui
from detail.gui.workers import BackgroundCall, StageProgressWorker
from detail.storage import tool_runs
from detail.tools import quick_segment as engine
from detail.gui.full_frame import EnlargeableThumb
from detail.gui.image_view import ImageView, PictureWithControls

# Marks a payload whose first slot carries a ``frame_source`` reason instead of a
# sample index; an object() can never collide with a real index.
_UNAVAILABLE = object()


class _QuickSegmentConfig(ConfigWidget):
    def __init__(self):
        super().__init__()
        import_runner_modules(PIPELINE_RUNNERS)  # populate MODELS (Tenet 1)
        self._frame_count: int | None = None
        self._probe_worker = None
        self._probe_generation = 0
        form = ConfigForm(self, QuickSegmentConfig)

        video_row = QWidget()
        video_box = QHBoxLayout(video_row)
        video_box.setContentsMargins(0, 0, 0, 0)
        self._videos = QListWidget()
        self._videos.setMaximumHeight(72)
        self._videos.setToolTip(
            "Clips to segment. Sampling spreads the frame budget across all of them."
        )
        add_video = QPushButton("Add")
        add_video.clicked.connect(self._browse)
        remove_video = QPushButton("Remove")
        remove_video.clicked.connect(self._remove_video)
        video_buttons = QVBoxLayout()
        video_buttons.setContentsMargins(0, 0, 0, 0)
        video_buttons.addWidget(add_video)
        video_buttons.addWidget(remove_video)
        video_buttons.addStretch()
        video_buttons_holder = QWidget()
        video_buttons_holder.setLayout(video_buttons)
        video_box.addWidget(self._videos, 1)
        video_box.addWidget(video_buttons_holder)
        form.addRow(label_with("Video(s):", help_button("recording")), video_row)
        self._cap = QLabel("")
        self._cap.setWordWrap(True)
        set_label_color(self._cap, MUTED)
        form.addRow("", self._cap)

        # Sampling is the default: the first N frames of a clip show only its opening
        # seconds, where a still animal or a blank lead-in shows nothing at all.
        mode_row = QVBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        self._mode_sample = QRadioButton("Sample evenly across each video")
        self._mode_sample.setChecked(True)
        self._mode_consecutive = QRadioButton("Consecutive from the start")
        # An explicit group, because the two radios end up in different parent widgets
        # (the consecutive row carries the "whole video" box) and Qt's auto-exclusivity
        # is per parent — without this both can be checked at once.
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._mode_sample)
        self._mode_group.addButton(self._mode_consecutive)
        self._whole_video = QCheckBox("whole video (no limit)")
        self._whole_video.setEnabled(False)
        self._mode_consecutive.toggled.connect(self._whole_video.setEnabled)
        self._mode_consecutive.toggled.connect(self._sync_estimate)
        self._whole_video.toggled.connect(self._sync_estimate)
        consecutive_row = QHBoxLayout()
        consecutive_row.setContentsMargins(0, 0, 0, 0)
        consecutive_row.addWidget(self._mode_consecutive)
        consecutive_row.addWidget(self._whole_video)
        consecutive_row.addStretch()
        consecutive_holder = QWidget()
        consecutive_holder.setLayout(consecutive_row)
        mode_row.addWidget(self._mode_sample)
        mode_row.addWidget(consecutive_holder)
        mode_holder = QWidget()
        mode_holder.setLayout(mode_row)
        form.addRow(field_label("quick_segment", "mode", "Frames:"), mode_holder)

        self._prompt_row = QWidget()
        prompt_box = QHBoxLayout(self._prompt_row)
        prompt_box.setContentsMargins(0, 0, 0, 0)
        self._prompts = QListWidget()
        self._prompts.setMaximumHeight(72)
        self._prompts.addItem("mouse")
        self._prompts.setToolTip(
            "One prompt per row, compared side by side — use Add… for another. "
            "A single prompt can name several classes at once: 'mouse, cricket'."
        )
        add_prompt = QPushButton("Add")
        add_prompt.clicked.connect(self._add_prompt)
        remove_prompt = QPushButton("Remove")
        remove_prompt.clicked.connect(self._remove_prompt)
        prompt_buttons = QVBoxLayout()
        prompt_buttons.setContentsMargins(0, 0, 0, 0)
        prompt_buttons.addWidget(add_prompt)
        prompt_buttons.addWidget(remove_prompt)
        prompt_buttons.addStretch()
        prompt_buttons_holder = QWidget()
        prompt_buttons_holder.setLayout(prompt_buttons)
        prompt_box.addWidget(self._prompts, 1)
        prompt_box.addWidget(prompt_buttons_holder)
        self._prompt_label = label_with("Prompts:", help_button("text-prompt"))
        form.addRow(self._prompt_label, self._prompt_row)

        # SAM 3 + your distilled/imported models. A closed-set model has no text prompt —
        # the prompt row hides and it runs detection directly (also the no-GPU path).
        self._tool_models = api_models.tool_models()
        self._prompt_capable_map = {m.id: m.prompt_capable for m in self._tool_models}
        self._model = QComboBox()
        for entry in self._tool_models:
            self._model.addItem(entry.name or entry.id, entry.id)
        self._model.currentIndexChanged.connect(self._sync_backends)
        form.addRow(label_with("Model:", help_button("model-kinds")), self._model)
        self._no_models = availability_note()
        form.addRow("", self._no_models)
        self._compute = ComputeTargetRow()
        self._compute.add_to(form.section())
        # Moves the selection to a model that can run here, which drives _sync_backends.
        self._unavailable = apply_availability(self._model, "detection")
        show_note(self._no_models, self._unavailable)
        self._sync_backends()

        frames_meta = field_meta(QuickSegmentConfig, "n_frames")
        # The model has no upper bound (an uncapped sweep is the point); the field's
        # ui_max supplies the spin ceiling, and "whole video" is the way past it.
        self._frames = field_widget(QuickSegmentConfig, "n_frames")
        self._frames.setValue(QuickSegmentConfig().n_frames)
        self._frames.valueChanged.connect(self._sync_estimate)
        form.addRow(
            field_label("quick_segment", "n_frames", frames_meta["label"] + ":"),
            self._frames,
        )
        form.finish()

    # -- model / backend ----------------------------------------------------- #
    def _prompt_capable(self) -> bool:
        """Whether the selected model takes a text prompt (SAM 3) — closed-set models don't."""
        return self._prompt_capable_map.get(selected_model_id(self._model), True)

    def _selected_backends(self) -> list[str]:
        # Built-ins report their own backends; imported/distilled models resolve by family
        # and support cuda/mps/cpu (the api fallback), so this works for both.
        model_id = selected_model_id(self._model)
        return api_models.supported_backends(model_id) if model_id else []

    def _sync_backends(self) -> None:
        self._compute.set_backends(self._selected_backends())
        self._sync_prompt_visibility()

    def _sync_prompt_visibility(self) -> None:
        """Show the prompt row only for a model that takes text.

        The whole row, not just the list: Add…/Remove sit in a sibling holder beside it,
        so hiding the list alone left two orphan buttons stretched across the form for a
        closed-set model that has nothing to prompt.
        """
        capable = self._prompt_capable()
        self._prompt_row.setVisible(capable)
        self._prompt_label.setVisible(capable)

    # -- videos + prompts ---------------------------------------------------- #
    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose video(s)",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        existing = self._video_paths()
        for path in paths:
            if path not in existing:
                self._videos.addItem(path)
        if paths:
            self._probe_videos()

    def _remove_video(self) -> None:
        row = self._videos.currentRow()
        if row >= 0:
            self._videos.takeItem(row)
            self._probe_videos()

    def _add_prompt(self) -> None:
        text, ok = QInputDialog.getText(
            self,
            "Add prompt",
            "Prompt to compare (one per row).\n"
            "A prompt can name several classes at once: mouse, cricket",
        )
        if ok and text.strip():
            self._prompts.addItem(text.strip())
            self._sync_estimate()

    def _remove_prompt(self) -> None:
        row = self._prompts.currentRow()
        if row >= 0:
            self._prompts.takeItem(row)
            self._sync_estimate()

    def _video_paths(self) -> list[str]:
        return [self._videos.item(i).text() for i in range(self._videos.count())]

    def _prompt_texts(self) -> list[str]:
        return [self._prompts.item(i).text() for i in range(self._prompts.count())]

    def _probe_videos(self) -> None:
        """Total frames across the chosen clips, read off the UI thread (CLAUDE.md)."""
        paths = [p for p in self._video_paths() if Path(p).is_file()]
        self._frame_count = None
        # Bumped on every change to the list, an emptied one included: a probe started for an
        # earlier list counts frames of videos no longer in it.
        self._probe_generation += 1
        generation = self._probe_generation
        if not paths:
            self._sync_estimate()
            return
        self._cap.setText("Reading video(s)…")
        set_label_color(self._cap, MUTED)
        self._probe_worker = BackgroundCall(
            lambda: sum(engine.approx_frame_count(p) for p in paths), receiver=self
        )
        self._probe_worker.done.connect(
            lambda count: self._on_probe_done(count, generation)
        )
        self._probe_worker.failed.connect(
            lambda _exc: generation == self._probe_generation and self._sync_estimate()
        )
        self._probe_worker.start()

    def _on_probe_done(self, count: int, generation: int) -> None:
        if generation != self._probe_generation:
            return
        self._frame_count = int(count)
        self._sync_estimate()

    def _sync_estimate(self, *_a) -> None:
        """What the run will actually cost: frames x prompts is the inference count, and
        uncapped x several prompts is the combination that runs for hours."""
        if self._frame_count is None:
            self._cap.setText("")
            return
        n_videos = max(len(self._video_paths()), 1)
        consecutive = self._mode_consecutive.isChecked()
        whole = consecutive and self._whole_video.isChecked()
        if whole:
            frames = self._frame_count
        elif consecutive:
            frames = min(self._frames.value(), self._frame_count)
        else:
            frames = min(self._frames.value() * n_videos, self._frame_count)
        n_prompts = 1
        if not consecutive and self._prompt_capable():
            n_prompts = max(len(self._prompt_texts()), 1)
        self._cap.setText(
            f"~{self._frame_count} frame(s) available - this run segments {frames} "
            f"x {n_prompts} prompt(s) = {frames * n_prompts} inference(s)"
        )
        set_label_color(self._cap, WARN if whole else MUTED)

    def over_cap_message(self) -> str | None:
        """Confirm an unbounded run - it is the one that can take hours."""
        if self._mode_consecutive.isChecked() and self._whole_video.isChecked():
            n = f"~{self._frame_count} frames" if self._frame_count else "every frame"
            return (
                f"This will segment {n} with no limit, which can take hours. "
                "You can stop it while it runs. Continue?"
            )
        return None

    # -- ConfigWidget contract ----------------------------------------------- #
    def validation_error(self) -> str | None:
        if self._unavailable:  # nothing here can run — say so before the run does
            return self._unavailable
        paths = self._video_paths()
        if not paths:
            return "Add a video first."
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            return f"This video path does not exist: {missing[0]}"
        # A closed-set model needs no prompt - it segments its own trained classes.
        if self._prompt_capable() and not self._prompt_texts():
            return "Add at least one prompt."
        if self._mode_consecutive.isChecked() and len(paths) > 1:
            return (
                "Consecutive reads one clip straight through - remove the extra "
                "videos, or switch to Sample evenly."
            )
        if self._mode_consecutive.isChecked() and len(self._prompt_texts()) > 1:
            return (
                "Consecutive runs a single prompt - remove the extra prompts, or "
                "switch to Sample evenly to compare them."
            )
        return None

    def _parsed_prompts(self) -> list[Prompt]:
        # Closed-set models ignore the prompt (they return all trained classes), so send
        # an empty prompt rather than the hidden list's contents.
        if not self._prompt_capable():
            return [Prompt(targets=[])]
        return [
            Prompt(
                targets=[
                    PromptTarget(c.strip(), TextCue(c.strip()))
                    for c in text.split(",")
                    if c.strip()
                ]
            )
            for text in self._prompt_texts()
        ]

    def config(self) -> dict:
        compute = self._compute.compute_target()
        paths = self._video_paths()
        prompts = self._parsed_prompts()
        consecutive = self._mode_consecutive.isChecked()
        return {
            "videos": paths,
            "video": paths[0],  # singular alias for the run manifest
            "prompts": prompts,
            "prompt": prompts[0],
            "model_id": selected_model_id(self._model),
            "model_variant": None,
            "compute": compute,
            "mode": "consecutive" if consecutive else "sample",
            "whole_video": consecutive and self._whole_video.isChecked(),
            "n_frames": self._frames.value(),
        }


class _QuickSegmentResult(QWidget):
    """Play a completed run's overlay clip (loops) + export it to mp4. Mirrors
    ``behavior_query._ClipPlayer`` — an off-thread ``FrameScrubber`` decoder renders
    ``source frame + saved detections`` per step."""

    def __init__(self, manifest: dict):
        super().__init__()
        self._manifest = manifest
        self._export_worker = None
        result = manifest["result"]
        self._source = manifest["source_video"]
        run_dir = tool_runs.tool_run_dir(manifest["tool_id"], manifest["id"])
        self._columns = engine.load_detections(
            run_dir / result["outputs"]["detections"]
        )
        self._colors = engine.class_colors(
            [str(c) for c in self._columns["class_name"]]
        )
        self._n = int(result["n_processed"])
        # The clip's own frame_idx -> t_ns map, so the row can say the time; a run written
        # before the map existed has none, and shows none rather than frame ÷ fps.
        self._t_ns = engine.load_timestamps(run_dir, result)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(result["summary"]))

        # Above the video, not below it. The run kept every proposal, so the threshold is
        # chosen here rather than before the run — but the scrubber under the video is
        # also a horizontal slider, and the two sitting together read as one control.
        # This one answers "what am I looking at", which belongs with the summary; the
        # scrubber answers "which frame", which belongs under the picture.
        # Re-decoding the current frame is what applies it (the scrubber's decoder reads
        # _min_conf), and ThresholdSlider's debounce keeps a drag to one re-decode.
        self._confidence = ConfidenceRow()
        self._min_conf = self._confidence.value()
        self._confidence.threshold_changed.connect(self._on_min_conf)
        self._total_stored = len(self._columns["frame_idx"])
        self._update_caption()
        layout.addWidget(self._confidence)

        self._pane = ImageView(zoomable=True)
        self._pane.clicked.connect(self._toggle)

        # A consecutive run is the clip read straight through, so it loops like a clip and plays
        # at the source's own rate. Both were sitting in the manifest unused: the timer here was
        # a flat 120 ms — ~8 fps whatever the camera did, so a 60 fps recording played at 1/7
        # speed and every judgement about the segmentation was made on the wrong tempo.
        self._scrubber = FrameScrubber(playable=True, loop=True)
        self._scrubber.set_playback_rate(float(result["fps"]) or DEFAULT_PREVIEW_FPS)
        self._scrubber.set_decoder(self._decode)
        self._scrubber.rendered.connect(self._display)
        # Picture directly above its controls, centred in the room this page gives it (#396) —
        # not the picture claiming the whole page and the controls sinking to wherever it ends.
        self._block = PictureWithControls(self._pane, self._scrubber)
        layout.addWidget(self._block, 1)

        actions = QHBoxLayout()
        self._export_btn = QPushButton("Export mp4")
        self._export_btn.clicked.connect(self._export)
        actions.addWidget(self._export_btn)
        actions.addStretch()
        layout.addLayout(actions)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._scrubber.set_total(self._n)
        if self._t_ns is not None and len(self._t_ns) >= self._n:
            self._scrubber.set_timestamps(self._t_ns[: self._n])
        self._scrubber.set_frame(0)

    def _on_min_conf(self, value: float) -> None:
        self._min_conf = value
        self._update_caption()
        self._scrubber.set_frame(
            self._scrubber.frame()
        )  # re-decode at the new threshold

    def _update_caption(self) -> None:
        kept = engine.frame_detection_count(self._columns, self._min_conf)
        self._confidence.set_caption(showing_caption(kept, self._total_stored))

    def _decode(self, step: int):  # off the UI thread (no Qt here)
        return frame_source.decode(
            lambda: engine.overlay_frame(
                self._source, step, self._columns, self._colors, min_conf=self._min_conf
            )
        )

    def _display(self, _step: int, payload) -> None:
        rgb, reason = payload
        if reason is not None:
            self._pane.clear_frame(frame_source.unavailable_text(reason))
            return
        self._pane.set_frame(_to_pixmap(rgb))
        self._block.set_aspect(self._pane.pixmap_size())

    def _toggle(self) -> None:
        """Clicking the picture is the same gesture as pressing Play, and takes the keyboard."""
        self._scrubber.setFocus()
        button = self._scrubber.play_button()
        if button is not None:
            button.toggle()

    def is_playing(self) -> bool:
        """Whether the overlay clip is running — ask this, not the button behind it."""
        return self._scrubber.is_playing()

    def _export(self) -> None:
        """Render the overlay clip into a new, named folder under a directory you pick —
        an export leaves the app, so it carries its own identity rather than landing as a
        bare ``overlay.mp4``."""
        parent = QFileDialog.getExistingDirectory(self, "Export to…")
        if not parent:
            return
        run_dir = tool_runs.tool_run_dir(
            self._manifest["tool_id"], self._manifest["id"]
        )
        self._export_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setMaximum(1000)
        self._progress.setValue(0)
        min_conf = self._min_conf
        self._export_worker = StageProgressWorker(
            lambda progress: engine.export_run(
                run_dir, self._manifest, parent, progress, min_conf=min_conf
            ),
            receiver=self,
        )
        self._export_worker.tick.connect(
            lambda permille, _s: self._progress.setValue(permille)
        )
        self._export_worker.done.connect(lambda out: self._export_done(str(out)))
        self._export_worker.failed.connect(self._export_failed)
        self._export_worker.start()

    def _export_done(self, path: str) -> None:
        self._progress.setVisible(False)
        self._export_btn.setEnabled(True)
        QMessageBox.information(self, "Export complete", f"Saved to {path}")

    def _export_failed(self, exc) -> None:
        self._progress.setVisible(False)
        self._export_btn.setEnabled(True)
        QMessageBox.critical(self, "Export failed", str(exc))


class _QuickSegmentComparison(QWidget):
    """A sampled run's results: a prompt x frame grid, and a click-through at full size.

    The run stored detections, not pictures, so both tabs **render** their overlays from
    ``source frame + saved detections`` the way every run preview does. The grid renders
    off the UI thread once (one decode per sampled frame, N draws), because a grid of
    thumbnails is the one view that needs every frame at once.
    """

    def __init__(self, manifest: dict):
        super().__init__()
        self._manifest = manifest
        self._run_dir = tool_runs.tool_run_dir(manifest["tool_id"], manifest["id"])
        self._index = engine.read_sampled(self._run_dir)
        self._sources = manifest.get("source_videos") or [manifest["source_video"]]
        result = manifest["result"]

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(result["summary"]))

        # One threshold for the whole result, not one per tab: Compare and Play show the
        # same detections, and two sliders that disagree would show two answers to the
        # same question.
        self._confidences = engine.sampled_confidences(self._run_dir)
        self._confidence = ConfidenceRow()
        self._confidence.threshold_changed.connect(self._on_min_conf)
        layout.addWidget(self._confidence)

        self._tabs = QTabWidget()
        min_conf = self._confidence.value()
        self._frames_view = _SampledFrames(
            self._run_dir, self._index, self._sources, min_conf
        )
        n_prompts = len(self._index.get("prompts", []))
        self._matrix = _SampledMatrix(
            self._run_dir, self._index, self._sources, min_conf
        )
        if n_prompts > 1:
            # One prompt has nothing to compare against, so the grid is a column of one -
            # open on the big view, which is what a single-prompt run is for.
            self._tabs.addTab(self._matrix, "Compare")
        self._tabs.addTab(self._frames_view, "Play")
        layout.addWidget(self._tabs, 1)
        self._update_caption(min_conf)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        export = QPushButton("Export frames")
        export.clicked.connect(self._export)
        actions.addWidget(export)
        actions.addStretch()
        actions_holder = QWidget()
        actions_holder.setLayout(actions)
        layout.addWidget(actions_holder)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._export_worker = None

    def _on_min_conf(self, value: float) -> None:
        self._update_caption(value)
        self._matrix.set_min_conf(value)
        self._frames_view.set_min_conf(value)

    def _update_caption(self, min_conf: float) -> None:
        kept = sum(int((c >= min_conf).sum()) for c in self._confidences.values())
        total = sum(int(c.size) for c in self._confidences.values())
        self._confidence.set_caption(showing_caption(kept, total))

    def _export(self) -> None:
        """Render the overlays into a new, named folder under a directory you pick. Images
        are made here and nowhere else: the run keeps detections, so this is the one place
        a picture reaches disk. The stills are drawn at the threshold on screen — a JPG
        can't be re-thresholded later, so it inherits what you chose."""
        parent = QFileDialog.getExistingDirectory(self, "Export to…")
        if not parent:
            return
        self._progress.setVisible(True)
        self._progress.setMaximum(1000)
        min_conf = self._confidence.value()
        self._export_worker = StageProgressWorker(
            lambda progress: engine.export_run(
                self._run_dir, self._manifest, parent, progress, min_conf=min_conf
            ),
            receiver=self,
        )
        self._export_worker.tick.connect(
            lambda permille, _s: self._progress.setValue(permille)
        )
        self._export_worker.done.connect(
            lambda out: (
                self._progress.setVisible(False),
                QMessageBox.information(self, "Export frames", f"Saved to {out}"),
            )
        )
        self._export_worker.failed.connect(
            lambda exc: (
                self._progress.setVisible(False),
                QMessageBox.critical(self, "Export failed", str(exc)),
            )
        )
        self._export_worker.start()


class _SampledMatrix(QScrollArea):
    """Prompt x frame grid, rendered from the run's detections (columns = prompts with a
    color swatch + total, rows = sampled frames, each cell an overlay thumbnail + count).

    Mirrors the experiment prompt-test's grid, but renders rather than reading saved JPGs:
    this run has none by design.

    Two passes, both off the UI thread. The **gather** decodes each sampled frame once and
    keeps it with that frame's raw detection columns; the **draw** turns that cache into
    overlays at the current display threshold. Moving the slider repeats only the draw —
    re-gathering would re-decode every sampled frame on every slider move, which is the
    whole reason the engine splits the two.
    """

    _THUMB = 220

    def __init__(self, run_dir: Path, index: dict, sources: list[str], min_conf: float):
        super().__init__()
        self.setWidgetResizable(True)
        self._run_dir, self._index, self._sources = run_dir, index, sources
        self._min_conf = min_conf
        self._colors = engine.sampled_prompt_colors(run_dir)
        self._gathered: list | None = None
        self._draw_workers: set = set()
        self._draw_generation = 0
        self._cells = 0
        inner = QWidget()
        self._grid = QGridLayout(inner)
        self.setWidget(inner)
        self._prompts = index.get("prompts", [])
        self._frames = [int(e["frame_idx"]) for e in index.get("sampled_frames", [])]

        self._grid.addWidget(QLabel("frame"), 0, 0)
        self._headers: dict[str, QLabel] = {}
        for col, prompt in enumerate(self._prompts, start=1):
            header = QLabel(_prompt_text(prompt))
            self._grid.addWidget(header, 0, col)
            self._headers[prompt["label"]] = header
        self._panes: dict[tuple[int, str], QLabel] = {}
        for row, frame in enumerate(self._frames, start=1):
            self._grid.addWidget(QLabel(str(frame)), row, 0)
            for col, prompt in enumerate(self._prompts, start=1):
                pane = EnlargeableThumb("rendering…")
                pane.setMinimumSize(self._THUMB, int(self._THUMB * 0.6))
                self._grid.addWidget(pane, row, col)
                self._panes[(row - 1, prompt["label"])] = pane
                self._cells += 1
        # A `setWidgetResizable(True)` scroll area grows its inner widget to the viewport, and
        # a grid with no stretched row shares that growth out among its rows — a 15px header
        # was laid out at 175px, and worse the wider the window. A trailing stretch row is the
        # grid's form of the top-justify rule (`gui_design` §3).
        self._grid.setRowStretch(self._grid.rowCount(), 1)
        self._gather_worker = BackgroundCall(self._gather_all, receiver=self)
        self._gather_worker.done.connect(self._on_gathered)
        self._gather_worker.failed.connect(lambda _exc: None)
        self._gather_worker.start()

    def set_min_conf(self, value: float) -> None:
        """Redraw the grid at a new display threshold — no decode, no store read."""
        self._min_conf = value
        self._redraw()

    def _gather_all(self) -> list:
        """One decode per sampled frame, kept with its detection columns (off the UI
        thread). Columns rather than decoded masks: a full-resolution mask is megabytes and
        the run holds every proposal, so caching them decoded would cost gigabytes."""
        return [
            engine.sampled_frame(self._run_dir, self._sources, i)
            for i in range(len(self._frames))
        ]

    def _on_gathered(self, gathered: object) -> None:
        self._gathered = list(gathered)
        self._redraw()

    def _redraw(self) -> None:
        if self._gathered is None:
            return  # the gather is still in flight; it redraws when it lands
        self._update_headers()
        gathered, min_conf = self._gathered, self._min_conf
        self._draw_generation += 1
        generation = self._draw_generation

        def draw() -> list:
            # Mask decode + draw for the whole grid: off the UI thread, because at a low
            # threshold a cell can carry every proposal the model found.
            return [
                (
                    {}
                    if frame_rgb is None
                    else engine.draw_overlays(
                        frame_rgb, per_label, frame_idx, self._colors, min_conf
                    )
                )
                for frame_rgb, per_label, frame_idx in gathered
            ]

        worker = BackgroundCall(draw, receiver=self)
        # Held until it finishes, not stored in a single attribute: a drag starts a draw
        # per settled position, and rebinding one attribute drops the previous QThread's
        # last reference while it is still running — Qt aborts the process for that.
        self._draw_workers.add(worker)
        worker.finished.connect(lambda w=worker: self._draw_workers.discard(w))
        # A drag can also outrun a draw; only the newest one may paint, or an older,
        # slower render lands last and the grid disagrees with the slider.
        worker.done.connect(lambda rendered: self._fill(rendered, generation))
        worker.failed.connect(lambda _exc: None)
        worker.start()

    def _update_headers(self) -> None:
        """Each prompt's total *at the current threshold* — the run's own stored counts
        were counted over every proposal and would overstate every column."""
        for label, header in self._headers.items():
            total = 0
            for _frame_rgb, per_label, frame_idx in self._gathered or []:
                columns = per_label.get(label)
                if columns is not None:
                    rows = columns["frame_idx"] == frame_idx
                    conf = np.asarray(columns["confidence"], dtype=float)
                    total += int(np.count_nonzero(rows & (conf >= self._min_conf)))
            prompt = next(p for p in self._prompts if p["label"] == label)
            header.setText(f"{_prompt_text(prompt)}\n{total} total")

    def _fill(self, rendered: object, generation: int) -> None:
        if generation != self._draw_generation:
            return  # superseded by a newer threshold
        for i, per_prompt in enumerate(rendered):
            for label, rgb in per_prompt.items():
                pane = self._panes.get((i, label))
                if pane is None:
                    continue
                pane.setPixmap(
                    _to_pixmap(rgb).scaled(
                        self._THUMB,
                        self._THUMB,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                # Re-rendered on the double-click that asks for it, at the threshold in force
                # then — the same recipe this draw used, not the picture it produced.
                pane.set_render(
                    lambda i=i, lb=label: self._render_one(i, lb),
                    title=f"{self._frame_label(i)} · {_prompt_text(self._prompt(label))}",
                )

    def _prompt(self, label: str) -> dict:
        return next(p for p in self._prompts if p["label"] == label)

    def _frame_label(self, index: int) -> str:
        _frame_rgb, _per_label, frame_idx = (self._gathered or [])[index]
        return f"frame {frame_idx}"

    def _render_one(self, index: int, label: str):
        """One cell's overlay at full resolution — off the UI thread (`full_frame`)."""
        frame_rgb, per_label, frame_idx = (self._gathered or [])[index]
        if frame_rgb is None:
            return None
        drawn = engine.draw_overlays(
            frame_rgb, per_label, frame_idx, self._colors, self._min_conf
        )
        return drawn.get(label)

    def cell_count(self) -> int:
        return self._cells


def _prompt_text(prompt: dict) -> str:
    targets = (prompt.get("prompt") or {}).get("targets") or []
    if not targets:
        return prompt["label"]
    return ", ".join(t.get("class_name", "?") for t in targets)


def _uniform_stride(entries: list[dict]) -> int | None:
    """The constant gap between sampled positions, or ``None`` when there is not one.

    Whether a sampled run can be *played* is a property of the positions it sampled, not of the
    mode it was launched in — so it is measured here rather than passed in. Three shapes reach
    this view and the rule separates them without knowing their names: ``consecutive`` reads
    straight through (stride 1), ``even`` sampling lands on the midpoints of equal buckets
    (a constant stride > 1, so playback is honest at ``fps / stride``), and ``random`` sampling
    lands wherever it lands — no gap, no rate, and a "1x" over it would be invented.

    Positions in different parts are not a gap at all: part-local indices restart, so a run
    spanning two clips has no single sequence to play.
    """
    if len(entries) < 2:
        return None
    if len({int(e["part_index"]) for e in entries}) != 1:
        return None
    frames = [int(e["frame_idx"]) for e in entries]
    gaps = {b - a for a, b in zip(frames, frames[1:])}
    if len(gaps) != 1:
        return None
    stride = gaps.pop()
    return stride if stride > 0 else None


def _sample_times(run_dir: Path, entries: list[dict]):
    """Each sampled position's ``t_ns``, gathered from its part's saved map — or ``None`` when a
    part has no map (a run written before maps were saved), since a partial time row is a guess.

    Not uniform, so not a slice: the positions are scattered across parts and frames.
    """
    maps: dict[int, np.ndarray] = {}
    times = []
    for entry in entries:
        part = int(entry["part_index"])
        if part not in maps:
            path = Path(run_dir) / engine.TIMESTAMPS_DIRNAME / f"part_{part:03d}.npy"
            if not path.is_file():
                return None
            maps[part] = np.load(path)
        frame = int(entry["frame_idx"])
        if frame >= len(maps[part]):
            return None
        times.append(int(maps[part][frame]))
    return np.asarray(times, dtype=np.int64)


class _SampledFrames(QWidget):
    """Step through a sampled run's positions at full size, one prompt at a time -
    rendered on demand from the saved detections, off the UI thread via the scrubber."""

    def __init__(self, run_dir: Path, index: dict, sources: list[str], min_conf: float):
        super().__init__()
        self._run_dir, self._sources = run_dir, sources
        self._min_conf = min_conf
        self._prompts = index.get("prompts", [])
        self._labels = [p["label"] for p in self._prompts]
        self._texts = [_prompt_text(p) for p in self._prompts]
        entries = index.get("sampled_frames", [])
        self._frames = [int(e["frame_idx"]) for e in entries]
        # Derived, not declared: a run whose positions are one frame apart (or a constant k
        # apart) is a sequence that can be played at `fps / k`; one sampled at random is not.
        stride = _uniform_stride(entries)
        fps = float(index.get("fps") or 0.0)
        self._play_fps = fps / stride if stride and fps > 0 else None

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(QLabel("Prompt:"))
        self._prompt_pick = QComboBox()
        self._prompt_pick.addItems(self._texts or ["(none)"])
        self._prompt_pick.currentIndexChanged.connect(
            lambda _i: self._scrubber.set_frame(self._scrubber.frame())
        )
        top.addWidget(self._prompt_pick)
        top.addStretch()
        top_holder = QWidget()
        top_holder.setLayout(top)
        layout.addWidget(top_holder)

        self._pane = ImageView(zoomable=True)
        # Frame entry is off: the positions are sampled, so typing a source frame number
        # would have nothing to land on - and it lets the readout name what you are on.
        # Playback is on where the positions form a sequence. A consecutive run is stride 1 and
        # plays at the source rate; an evenly sampled one plays at `fps / stride`, which is the
        # rate those frames really represent; a randomly sampled one has no rate and does not
        # play, because "1x" over scattered positions would be a number with nothing behind it.
        self._scrubber = FrameScrubber(
            label_formatter=lambda v, last: f"sample {v + 1} / {last + 1}",
            allow_frame_entry=False,
            playable=self._play_fps is not None,
        )
        if self._play_fps is not None:
            self._scrubber.set_playback_rate(self._play_fps)
        self._scrubber.set_decoder(self._render)  # off-thread
        self._scrubber.rendered.connect(self._display)
        # Picture directly above its controls, centred in the room this page gives it (#396) —
        # not the picture claiming the whole page and the controls sinking to wherever it ends.
        self._block = PictureWithControls(self._pane, self._scrubber)
        layout.addWidget(self._block, 1)
        self._caption = QLabel("")
        layout.addWidget(self._caption)
        self._scrubber.set_total(max(len(self._frames), 1))
        sample_t_ns = _sample_times(run_dir, entries)
        if sample_t_ns is not None and len(sample_t_ns) == len(self._frames):
            self._scrubber.set_timestamps(sample_t_ns)
        self._scrubber.set_frame(0)

    def set_min_conf(self, value: float) -> None:
        """Re-render the position on screen at a new display threshold."""
        self._min_conf = value
        self._scrubber.set_frame(self._scrubber.frame())

    def _render(self, step: int):  # off the UI thread (no Qt here)
        if not self._frames:
            return None
        i = min(step, len(self._frames) - 1)
        label = self._labels[max(self._prompt_pick.currentIndex(), 0)]
        overlays, reason = frame_source.decode(
            lambda: engine.sampled_overlays(
                self._run_dir, self._sources, i, min_conf=self._min_conf
            )
        )
        if reason is not None:
            return _UNAVAILABLE, reason
        # An empty map is a frame that would not decode; a label missing from a
        # populated one is a prompt with nothing on this frame. Both read as no
        # picture, so ``_display`` keeps them apart.
        return i, overlays.get(label)

    def _display(self, _step: int, payload) -> None:
        if payload is None:
            self._pane.clear_frame("(no sampled frames)")
            return
        i, rgb = payload
        if i is _UNAVAILABLE:
            self._pane.clear_frame(frame_source.unavailable_text(rgb))
            return
        if rgb is None:
            self._pane.clear_frame("(frame could not be decoded)")
            return
        self._pane.set_frame(_to_pixmap(rgb))
        self._block.set_aspect(self._pane.pixmap_size())
        self._caption.setText(f"source frame {self._frames[i]}")


@register_tool_ui
class QuickSegmentUI(ToolUI):
    tool_id = "quick_segment"

    def config_widget(self) -> ConfigWidget:
        return _QuickSegmentConfig()

    def result_widget(self, manifest: dict) -> QWidget:
        outputs = (manifest.get("result") or {}).get("outputs") or {}
        if "prompt_test" in outputs:
            return _QuickSegmentComparison(manifest)
        return _QuickSegmentResult(manifest)
