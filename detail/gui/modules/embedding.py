"""Embedding module UI (§14): config form + result view.

Per-recording upstream (a tracking run + its tracks, or "All tracks"), an embedding
model, a named input-image **constructor** (transforms.constructors), and compute.
The "Preview input images…" button opens the input-image scrubber; the result
view (EmbeddingRunPreview) shows coverage + a PCA point cloud. Runs need a model
backend (DINOv3) — install it with `pip install -e '.[torch]'`.
"""

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from detail.api import run_spec
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.embedding import EmbeddingConfig
from detail.core.registries import EMBEDDING_MODELS
from detail.executor.runners import import_runner_modules
from detail.gui.compute_target import ComputeTargetRow
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label
from detail.gui.modules._common import (
    UpstreamSpecPicker,
    covered_ids,
    repopulate_spec_combo,
    spec_groups,
    spec_items,
)
from detail.gui.modules._every_n import EveryNRow
from detail.gui.modules._heading import HeadingPicker
from detail.gui.model_availability import (
    apply_availability,
    availability_note,
    fill_variants,
    selected_model_id,
    show_note,
    unavailable_reason,
    variant_available,
)
from detail.gui._pydantic_form import (
    field_meta,
    field_widget,
    optional_spin,
    optional_value,
    set_optional_value,
)
from detail.gui.modules.registry import ConfigWidget, ModuleUI, register_module_ui
from detail.gui.modules.registry import surfaced
from detail.gui.run_preview import InputImageScrubber
from detail.transforms.constructors import CONSTRUCTORS, DEFAULT_CONSTRUCTOR
from detail.transforms.constructors import surfaced_constructors


class _EmbeddingConfig(ConfigWidget):
    def __init__(self):
        super().__init__()
        self._model_names: dict = {}  # {model_id: name}; filled by on_experiment_set
        import_runner_modules(
            PIPELINE_RUNNERS
        )  # populate the EMBEDDING_MODELS registry
        form = ConfigForm(self, EmbeddingConfig)
        self._tracking = QComboBox()
        self._tracking.currentIndexChanged.connect(self._update_available_tracks)
        self._tracking.currentIndexChanged.connect(
            lambda _index: self.coverage_changed.emit()
        )
        self._tracking.currentIndexChanged.connect(self._sync_upstream_every_n)
        form.addRow(
            field_label("embedding", "tracking_run_id", "Tracking run:"), self._tracking
        )
        tracks_row = QWidget()
        tracks_layout = QHBoxLayout(tracks_row)
        tracks_layout.setContentsMargins(0, 0, 0, 0)
        self._tracks = QLineEdit("0")
        self._all_tracks = QCheckBox("All tracks")
        self._all_tracks.setToolTip(field_meta(EmbeddingConfig, "all_tracks")["help"])
        self._all_tracks.toggled.connect(self._sync_tracks_enabled)
        tracks_layout.addWidget(self._tracks, 1)
        tracks_layout.addWidget(self._all_tracks)
        form.addRow(field_label("embedding", "all_tracks", "Track ids:"), tracks_row)
        self._available = QLabel("")
        self._available.setWordWrap(True)
        form.addRow("", self._available)
        self._model = QComboBox()
        for model_id in surfaced(EMBEDDING_MODELS):
            self._model.addItem(model_id, model_id)
        self._select_default_model()
        self._model.currentIndexChanged.connect(self._sync_variants)
        self._model.currentIndexChanged.connect(self._sync_readouts)
        form.addRow(field_label("embedding", "model_id", "Model:"), self._model)
        self._no_models = availability_note()
        form.addRow("", self._no_models)
        self._variant = QComboBox()
        form.addRow(
            field_label("embedding", "model_variant", "Size / variant:"), self._variant
        )
        self._sync_variants()
        self._readout = QComboBox()
        self._readout.setToolTip(field_meta(EmbeddingConfig, "readout")["help"])
        form.addRow(field_label("embedding", "readout", "Read-out:"), self._readout)
        self._sync_readouts()
        # After the rows its selection drives exist: marking moves the selection to a
        # runnable model, and that fires _sync_variants / _sync_readouts.
        self._unavailable = apply_availability(self._model, "embedding")
        # `apply_availability` moves off a model that is not installed. The embedding model
        # and size are what a run *is* — every vector changes with them — so the form stays on
        # the configured default, greyed and labelled, and `validation_error` says why it
        # cannot launch, rather than quietly opening on another model or size.
        self._select_default_model()
        show_note(self._no_models, self._unavailable)
        self._input_size = optional_spin(
            EmbeddingConfig, "input_size", "(model default)"
        )
        form.addRow(
            field_label("embedding", "input_size", "Input size:"), self._input_size
        )
        self._constructor = QComboBox()
        self._constructor.setToolTip(field_meta(EmbeddingConfig, "constructor")["help"])
        for key in surfaced_constructors():
            self._constructor.addItem(CONSTRUCTORS[key].label, key)
        self._constructor.setCurrentIndex(
            self._constructor.findData(DEFAULT_CONSTRUCTOR)
        )
        form.addRow(
            field_label("embedding", "constructor", "Input image:"), self._constructor
        )
        # Only bite on a multi-recording run; kept visible (greyed-out surfaces confuse
        # more than they help here) since the launch selection can change under the form.
        self._canvas_scope = QComboBox()
        self._canvas_scope.setToolTip(
            field_meta(EmbeddingConfig, "canvas_scope")["help"]
        )
        self._canvas_scope.addItem("Per recording", "per_recording")
        self._canvas_scope.addItem("Shared across selection", "shared")
        form.addRow(
            field_label("embedding", "canvas_scope", "Canvas scope:"),
            self._canvas_scope,
        )
        # §14 egocentric alignment: the mode, the keypoint track run (a spec, resolved per
        # recording like the tracking run) and the heading definition; the three settings
        # under Advanced, seeded from the model.
        self._align = QComboBox()
        self._align.setToolTip(field_meta(EmbeddingConfig, "align")["help"])
        for value, label in (
            ("none", "None"),
            ("rotate", "Overhead camera: turn so the head points right"),
            ("mirror", "Side camera: flip so the animal faces right"),
        ):
            self._align.addItem(label, value)
        self._align.currentIndexChanged.connect(self._sync_alignment)
        form.addRow(
            field_label("embedding", "align", "Egocentric alignment:"), self._align
        )
        self._keypoint_picker = UpstreamSpecPicker("keypoint_track")
        self._keypoint = self._keypoint_picker.combo
        self._keypoint.setToolTip(
            field_meta(EmbeddingConfig, "keypoint_run_id")["help"]
        )
        self._keypoint.currentIndexChanged.connect(self._sync_heading_skeleton)
        self._keypoint.currentIndexChanged.connect(
            lambda _index: self.coverage_changed.emit()
        )
        form.addRow(
            field_label("embedding", "keypoint_run_id", "Keypoint track run:"),
            self._keypoint,
        )
        self._heading = HeadingPicker()
        self._heading.combo.setToolTip(field_meta(EmbeddingConfig, "heading")["help"])
        form.addRow(field_label("embedding", "heading", "Heading:"), self._heading)
        self._alignment_note = QLabel("")
        self._alignment_note.setWordWrap(True)
        form.addRow("", self._alignment_note)
        self._max_gap = field_widget(EmbeddingConfig, "max_gap_s")
        self._mirror_hold = field_widget(EmbeddingConfig, "mirror_hold_s")
        self._facing_ratio = field_widget(EmbeddingConfig, "facing_camera_ratio")
        for name, widget in (
            ("max_gap_s", self._max_gap),
            ("mirror_hold_s", self._mirror_hold),
            ("facing_camera_ratio", self._facing_ratio),
        ):
            meta = field_meta(EmbeddingConfig, name)
            form.addRow(
                field_label("embedding", name, meta["label"] + ":"),
                widget,
                advanced=True,
            )
        self._skeleton_by_spec: dict = {}
        self._experiment = None
        every_n_meta = field_meta(EmbeddingConfig, "every_n_frames")
        self._every_n = EveryNRow(EmbeddingConfig, upstream_noun="tracked")
        form.addRow(
            field_label("embedding", "every_n_frames", every_n_meta["label"] + ":"),
            self._every_n,
        )
        batch_meta = field_meta(EmbeddingConfig, "batch_size")
        self._batch = field_widget(EmbeddingConfig, "batch_size")
        self._batch.setValue(32)
        form.addRow(
            field_label("embedding", "batch_size", batch_meta["label"] + ":"),
            self._batch,
        )
        self._compute = ComputeTargetRow()
        self._compute.add_to(form.section())
        self._recordings: list = []
        self._groups: list[dict] = []
        self._track_cache: dict[tuple[str, str], list[int]] = {}
        self._preview_btn = QPushButton("Preview input images")
        self._preview_btn.clicked.connect(self._open_scrubber)
        form.addRow("", self._preview_btn)
        form.finish()
        self._sync_alignment()

    def _select_default_model(self) -> None:
        """Open on ``EmbeddingConfig``'s default model, not on whichever row is first."""
        index = self._model.findData(EmbeddingConfig.model_fields["model_id"].default)
        if index >= 0:
            self._model.setCurrentIndex(index)

    def _not_installed(self) -> str | None:
        """Why the selected model or size cannot run here, or ``None`` when it can."""
        model_id = selected_model_id(self._model)
        reason = unavailable_reason(model_id) if model_id else None
        if reason is None and not variant_available(self._variant):
            reason = "not installed"
        if reason is None:
            return None
        size = self._variant.currentData()
        name = f"{model_id} {size}" if size else str(model_id)
        return (
            f"{name}: {reason}. Install it from the Model manager, or choose an "
            "installed model or size."
        )

    def _sync_tracks_enabled(self) -> None:
        self._tracks.setEnabled(not self._all_tracks.isChecked())

    # -- §14 alignment ------------------------------------------------------------ #
    def _align_mode(self) -> str:
        return str(self._align.currentData() or "none")

    def _sync_alignment(self, *_args) -> None:
        """The keypoint run, heading and settings matter only when aligning; greyed
        otherwise, never hidden, and the note says what the mode does."""
        mode = self._align_mode()
        on = mode != "none"
        for widget in (
            self._keypoint,
            self._heading,
            self._max_gap,
            self._facing_ratio,
        ):
            widget.setEnabled(on)
        self._mirror_hold.setEnabled(mode == "mirror")
        if not on:
            self._alignment_note.setText(
                "Each animal is embedded as cropped, facing whichever way it faces. "
                "Alignment needs a keypoint track run and a heading definition."
            )
        elif mode == "rotate":
            self._alignment_note.setText(
                "Each animal is turned so its head points right. Frames where the head "
                "direction is unknown are left out."
            )
        else:
            self._alignment_note.setText(
                "An animal facing left is flipped to face right. It must face the new way "
                "for the hold time before it flips. Frames where the direction is unknown "
                "are left out."
            )
        self.coverage_changed.emit()

    def _sync_heading_skeleton(self, *_args) -> None:
        spec = self._keypoint_picker.spec()
        key = None if spec is None else run_spec.freeze_spec(spec)
        self._heading.set_skeleton_names(self._skeleton_by_spec.get(key))

    def _sync_upstream_every_n(self, *_args) -> None:
        """Adopt the picked tracking run's cadence: it is the floor this stage can use."""
        spec = self._tracking.currentData()
        identity = spec.get("identity", {}) if isinstance(spec, dict) else {}
        self._every_n.set_upstream(identity.get("every_n_frames"))

    def _run_id_for(self, recording) -> str | None:
        """Resolve the selected tracking spec to this recording's own tracking run id."""
        spec = self._tracking.currentData()
        if not isinstance(spec, dict):
            return None
        return run_spec.resolve_spec_for_recording(recording, "tracking", spec)

    def _update_available_tracks(self) -> None:
        if not isinstance(self._tracking.currentData(), dict) or not self._recordings:
            self._available.setText("")
            return
        parts = []
        for recording in self._recordings:
            run_id = self._run_id_for(recording)
            # Prefer the ids resolve_recordings read off the UI thread; fall back to the
            # (mtime-cached) read so a spec chosen after resolve still resolves.
            tracks = self._track_cache.get((recording.id, run_id))
            if tracks is None:
                tracks = recording.track_ids(run_id) if run_id else []
            parts.append(
                f"{recording.id}: " + (", ".join(str(t) for t in tracks) or "—")
            )
        self._available.setText("Available tracks — " + "   ".join(parts))

    def _sync_variants(self) -> None:
        model_cls = EMBEDDING_MODELS.get(selected_model_id(self._model))
        variants = model_cls.variants() if model_cls is not None else []
        self._variant.clear()
        if variants:
            # The model's declared default, not dict order. Landing on the first entry
            # meant a form nobody touched ran a size nobody installed.
            fill_variants(
                self._variant,
                selected_model_id(self._model),
                variants,
                model_cls.default_variant(),
                keep_default=True,
            )
            self._variant.setEnabled(True)
        else:
            self._variant.addItem("(default)", None)
            self._variant.setEnabled(False)

    def _sync_readouts(self) -> None:
        model_cls = EMBEDDING_MODELS.get(selected_model_id(self._model))
        readouts = list(getattr(model_cls, "READOUTS", ()) or [])
        self._readout.clear()
        if readouts:
            for name in readouts:
                self._readout.addItem(name, name)
            self._readout.setEnabled(True)
        else:
            self._readout.addItem("(default)", None)
            self._readout.setEnabled(False)

    def on_experiment_set(self, experiment) -> None:
        # A run record stores a model *id*; a picker has to show the name (§11). O(models),
        # not O(recordings), so it stays inside the UI-thread rule.
        self._model_names = experiment.model_names() if experiment is not None else {}
        self._experiment = experiment
        self._heading.set_experiment(experiment)

    def resolve_recordings(self, recordings: list) -> dict:
        """Off the UI thread (CLAUDE.md): the disk reads — spec groups plus every
        selected recording's track ids (one tracks.zarr open each, ~1 s over 60
        recordings), and the keypoint track groups with each one's skeleton (§14).
        Returns a Qt-free payload ``apply_recordings`` installs."""
        groups = spec_groups(recordings, "tracking", self._model_names)
        tracks: dict[tuple[str, str], list[int]] = {}
        for recording in recordings:
            for group in groups:
                run_id = run_spec.resolve_spec_for_recording(
                    recording, "tracking", group["spec"]
                )
                if run_id is not None and (recording.id, run_id) not in tracks:
                    tracks[(recording.id, run_id)] = recording.track_ids(run_id)
        keypoint_groups = self._keypoint_picker.resolve(recordings, self._model_names)
        skeletons: dict = {}
        for group in keypoint_groups:
            for recording in recordings:
                run_id = run_spec.resolve_spec_for_recording(
                    recording, "keypoint_track", group["spec"]
                )
                if run_id is None:
                    continue
                try:
                    names = tuple(recording.keypoint_track_run(run_id).skeleton.names)
                except (FileNotFoundError, ValueError, KeyError):
                    continue
                skeletons[run_spec.freeze_spec(group["spec"])] = names
                break
        return {
            "recordings": recordings,
            "groups": groups,
            "tracks": tracks,
            "keypoint_groups": keypoint_groups,
            "skeletons": skeletons,
        }

    def apply_recordings(self, recordings: list, resolved: dict) -> None:
        """UI thread: install the resolved spec groups + track ids and repopulate."""
        self._recordings = resolved["recordings"]
        self._groups = resolved["groups"]
        self._track_cache = resolved["tracks"]
        repopulate_spec_combo(
            self._tracking,
            spec_items(self._groups, len(self._recordings))
            or [("(no tracking run in the selected recordings)", None)],
        )
        self._skeleton_by_spec = dict(resolved.get("skeletons") or {})
        self._keypoint_picker.apply(
            resolved.get("keypoint_groups") or [], len(self._recordings)
        )
        if self._keypoint.count() == 0:
            self._keypoint.addItem(
                "(no keypoint track run in the selected recordings)", None
            )
        self._sync_heading_skeleton()
        self._every_n.set_recordings(self._recordings)
        self._sync_upstream_every_n()
        self._update_available_tracks()

    def on_recordings_changed(self, recordings: list) -> None:
        """Synchronous path for direct callers (the launch dialog uses the off-thread
        resolve→apply split)."""
        self.apply_recordings(recordings, self.resolve_recordings(recordings))

    def prefer_upstream_spec(self, module_id: str, spec: dict) -> None:
        """A next-step launch from a keypoint track run (§14): open the keypoint picker on
        that run's settings. Align stays as the user sets it — rotate or mirror depends on
        the camera (Tenet 9)."""
        if module_id == "keypoint_track":
            self._keypoint_picker.prefer(spec)

    def covered_recording_ids(self) -> set[str] | None:
        covered = covered_ids(self._groups, self._tracking.currentData())
        if self._align_mode() == "none":
            return covered
        # An aligned run also needs the keypoint track run in every recording it covers.
        by_keypoints = self._keypoint_picker.covered()
        if covered is None:
            return by_keypoints
        return covered if by_keypoints is None else covered & by_keypoints

    def _effective_track_ids(self) -> list[int]:
        if not self._all_tracks.isChecked():
            return self._parse_track_ids() or []
        if not self._recordings:
            return []
        run_id = self._run_id_for(self._recordings[0])
        return self._recordings[0].track_ids(run_id) if run_id else []

    def _open_scrubber(self) -> None:
        if not self._recordings:
            QMessageBox.information(
                self, "Input image preview", "Check the recordings to preview on first."
            )
            return
        tracking_spec = self._tracking.currentData()
        track_ids = self._effective_track_ids()
        if not isinstance(tracking_spec, dict) or not track_ids:
            QMessageBox.information(
                self,
                "Input image preview",
                "Select a tracking run and at least one track id (or All tracks) first.",
            )
            return
        run_ids = {
            recording.id: self._run_id_for(recording) for recording in self._recordings
        }
        alignment = None
        keypoint_run_ids: dict = {}
        if self._align_mode() != "none":
            error = self.validation_error()
            if error is not None:
                QMessageBox.information(self, "Input image preview", error)
                return
            definition = self._experiment.get_heading_definition(self._heading.name())
            alignment = {
                "align": self._align_mode(),
                "heading_definition": {
                    "front": list(definition["front"]),
                    "back": list(definition["back"]),
                },
                "max_gap_s": self._max_gap.value(),
                "mirror_hold_s": self._mirror_hold.value(),
                "facing_camera_ratio": self._facing_ratio.value(),
            }
            spec = self._keypoint_picker.spec()
            keypoint_run_ids = {
                recording.id: run_spec.resolve_spec_for_recording(
                    recording, "keypoint_track", spec
                )
                for recording in self._recordings
            }
        InputImageScrubber(
            self._recordings,
            run_ids,
            track_ids,
            self._constructor.currentData(),
            model_id=selected_model_id(self._model),
            parent=self,
            alignment=alignment,
            keypoint_run_ids=keypoint_run_ids,
        ).exec()

    def _parse_track_ids(self) -> list[int] | None:
        text = self._tracks.text().strip()
        try:
            return [int(x) for x in text.split(",") if x.strip()]
        except ValueError:
            return None

    def load_config(self, config: dict) -> None:
        """Restore a previous run's embedding settings. ``tracking_spec`` is a *selection* —
        the dialog re-derives which tracking runs the current recordings share — and ``compute``
        is a property of this host, not of the run being copied (see ``detection``)."""
        for combo, key in (
            (self._model, "model_id"),
            (self._variant, "model_variant"),
            (self._readout, "readout"),
            (self._constructor, "constructor"),
            (self._canvas_scope, "canvas_scope"),
        ):
            index = combo.findData(config.get(key))
            if index >= 0:
                combo.setCurrentIndex(index)
        set_optional_value(self._input_size, config.get("input_size"))
        if "all_tracks" in config:
            self._all_tracks.setChecked(bool(config["all_tracks"]))
        track_ids = config.get("track_ids") or []
        if track_ids:
            self._tracks.setText(", ".join(str(int(t)) for t in track_ids))
        if config.get("batch_size") is not None:
            self._batch.setValue(int(config["batch_size"]))
        self._every_n.set_value(config.get("every_n_frames"))
        # §14: the mode, the heading and the settings are settings and are restored; the
        # keypoint track run is a selection (a spec), re-derived from the recordings.
        index = self._align.findData(config.get("align") or "none")
        self._align.setCurrentIndex(index if index >= 0 else 0)
        self._heading.set_name(config.get("heading"))
        for name, widget in (
            ("max_gap_s", self._max_gap),
            ("mirror_hold_s", self._mirror_hold),
            ("facing_camera_ratio", self._facing_ratio),
        ):
            if config.get(name) is not None:
                widget.setValue(float(config[name]))

    def config(self) -> dict:
        compute = self._compute.compute_target()
        all_tracks = self._all_tracks.isChecked()
        align = self._align_mode()
        return {
            "tracking_spec": self._tracking.currentData(),
            "all_tracks": all_tracks,
            "track_ids": [] if all_tracks else (self._parse_track_ids() or []),
            "model_id": selected_model_id(self._model),
            "model_variant": self._variant.currentData(),
            "readout": self._readout.currentData(),
            "input_size": optional_value(self._input_size),
            "constructor": self._constructor.currentData(),
            "canvas_scope": self._canvas_scope.currentData(),
            "batch_size": self._batch.value(),
            "every_n_frames": self._every_n.value(),
            "compute": compute,
            "align": align,
            "keypoint_track_spec": (
                self._keypoint_picker.spec() if align != "none" else None
            ),
            "heading": self._heading.name() if align != "none" else None,
            "max_gap_s": self._max_gap.value(),
            "mirror_hold_s": self._mirror_hold.value(),
            "facing_camera_ratio": self._facing_ratio.value(),
        }

    def validation_error(self) -> str | None:
        if self._unavailable:  # nothing here can run — say so before the run does
            return self._unavailable
        not_installed = self._not_installed()
        if not_installed is not None:
            return not_installed
        if self._tracking.currentData() is None:
            return "No tracking run in the selected recordings — run tracking first."
        if not self._all_tracks.isChecked() and not self._parse_track_ids():
            return (
                "Enter at least one track id (comma-separated integers), or "
                "check All tracks."
            )
        if self._align_mode() != "none":
            if self._keypoint_picker.spec() is None:
                return (
                    "Alignment needs a keypoint track run in the selected recordings — "
                    "run keypoint detect and keypoint track first, or set Align to none."
                )
            if self._heading.name() is None:
                return "Choose a heading definition, or define one, to align on."
            reason = self._heading.reason()
            if reason is not None:
                return (
                    f"The heading definition does not fit this run's skeleton: {reason}"
                )
            from detail.transforms.constructors import ALIGNABLE

            if self._constructor.currentData() not in ALIGNABLE:
                return (
                    "Alignment needs an input image that crops to the subject; choose a "
                    "cropped, true-scale constructor."
                )
        return self._every_n.warning()


@register_module_ui
class EmbeddingModuleUI(ModuleUI):
    module_id = "embedding"

    def config_widget(self) -> ConfigWidget:
        return _EmbeddingConfig()

    def preview_widget(self, recording, run_id: str) -> QWidget | None:
        from detail.gui.run_preview import EmbeddingRunPreview

        return EmbeddingRunPreview(recording, run_id)
