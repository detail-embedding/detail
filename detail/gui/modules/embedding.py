"""Embedding module UI (§14): config form + result view.

Per-recording upstream (a tracking run + its tracks, or "All tracks"), an embedding
model, a named input-image **constructor** (transforms.constructors), and compute. Optionally
**Align to body axis** (§26): a keypoints run on the same tracking run gives each frame's axis,
and the subject is turned (rotate) or flipped (mirror) before the model sees it; the keypoints
row and the alignment settings show only when alignment is on. **Pose** (§26) puts the keypoints'
pose into the embedding too: drawn onto the mask image, or added as numbers after the mask's vector.
The "Preview input images…" button opens the input-image scrubber; the result
view (EmbeddingRunPreview) shows coverage + a PCA point cloud. Runs need a model
backend (DINOv3) — install it with `pip install -e '.[torch]'`.
"""

from PyQt6.QtCore import Qt
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

from detail.api import models as api_models
from detail.api import run_spec
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.embedding import POSE_CHOICES, EmbeddingConfig
from detail.core.registries import EMBEDDING_MODELS
from detail.core.utils import resolve_track_id
from detail.executor.runners import import_runner_modules
from detail.gui.compute_target import ComputeTargetRow
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label
from detail.gui.modules._common import (
    UpstreamSpecPicker,
    covered_ids,
    group_runs,
    lineage_fits,
    repopulate_spec_combo,
    spec_groups,
    spec_items,
    spec_reads,
    widest_group,
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
from detail.storage import guidance
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
        self._tracking.currentIndexChanged.connect(self._sync_track_selector_mode)
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
        self._tracks_label = field_label("embedding", "all_tracks", "Track ids:")
        form.addRow(self._tracks_label, tracks_row)
        self._tracks_row = tracks_row
        # A track-merge run's object_id is a position, not an animal (§13) — this row
        # replaces the raw int field with one checkbox per subject name, resolved to
        # the right object_id *per recording* (EmbeddingModule.validate) rather than
        # assuming the same integer means the same animal in every recording.
        self._subjects_row = QWidget()
        self._subjects_layout = QHBoxLayout(self._subjects_row)
        self._subjects_layout.setContentsMargins(0, 0, 0, 0)
        self._subject_checks: list[QCheckBox] = []
        self._subjects_label = field_label("embedding", "all_tracks", "Subjects:")
        form.addRow(self._subjects_label, self._subjects_row)
        self._subjects_row.setVisible(False)
        self._subjects_label.setVisible(False)
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
        self._unavailable = apply_availability(
            self._model, "embedding", allow_remote=True
        )
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
        # {frozen spec: {recording_id: tracking run id}} each keypoint track row assigned to.
        self._keypoint_reads: dict = {}
        # A next-step launch's keypoint track run names its tracking run (``aux``).
        self._preferred_tracking: dict | None = None
        self._tracking.currentIndexChanged.connect(self._sync_keypoint_lineage)
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
        # §14 pose in the embedding: the same keypoint track run, read a second way; the
        # choices worded once (POSE_CHOICES) for this form, --help and the docs.
        self._pose = QComboBox()
        self._pose.setToolTip(field_meta(EmbeddingConfig, "pose")["help"])
        for value, (label, explanation) in POSE_CHOICES.items():
            self._pose.addItem(label, value)
            self._pose.setItemData(
                self._pose.count() - 1, explanation, Qt.ItemDataRole.ToolTipRole
            )
        self._pose.currentIndexChanged.connect(self._sync_alignment)
        form.addRow(field_label("embedding", "pose", "Pose:"), self._pose)
        self._pose_weight = field_widget(EmbeddingConfig, "pose_weight")
        form.addRow(
            field_label("embedding", "pose_weight", "Pose weight:"), self._pose_weight
        )
        self._pose_min_score = optional_spin(
            EmbeddingConfig, "pose_min_score", "(the track run's cut-off)"
        )
        form.addRow(
            field_label("embedding", "pose_min_score", "Pose min score:"),
            self._pose_min_score,
            advanced=True,
        )
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
        self._compute = ComputeTargetRow(allow_remote=True)
        self._compute.add_to(form.section())
        self._model.currentIndexChanged.connect(self._sync_backends)
        self._sync_backends()
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

    # -- §14 alignment ------------------------------------------------------------ #
    def _align_mode(self) -> str:
        return str(self._align.currentData() or "none")

    def _pose_mode(self) -> str:
        return str(self._pose.currentData() or "none")

    def _reads_keypoints(self) -> bool:
        """Whether this run reads the keypoint track run: to align, or for the pose."""
        return self._align_mode() != "none" or self._pose_mode() != "none"

    def _needs_heading(self) -> bool:
        """The heading turns the image (align) and the pose numbers (append); a drawn pose
        is painted as placed and needs none."""
        return self._align_mode() != "none" or self._pose_mode() == "append"

    def _sync_alignment(self, *_args) -> None:
        """The keypoint run, heading and settings matter only when something reads them;
        greyed otherwise, never hidden, and the note says what the mode does."""
        mode = self._align_mode()
        on = mode != "none"
        self._keypoint.setEnabled(self._reads_keypoints())
        for widget in (self._heading, self._max_gap, self._facing_ratio):
            widget.setEnabled(self._needs_heading())
        self._mirror_hold.setEnabled(mode == "mirror")
        self._pose_weight.setEnabled(self._pose_mode() == "append")
        self._pose_min_score.setEnabled(self._pose_mode() != "none")
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

    def _sync_tracks_enabled(self) -> None:
        self._tracks.setEnabled(not self._all_tracks.isChecked())

    def _subject_names_for_current_spec(self) -> list[str] | None:
        """The ordered subject names a track-merge run's own spec names (§13) — its
        ``class_name`` is that run's composition, e.g. ``"black mouse+white mouse"``
        (``track_merge.py``'s ``extra_record_fields``) — or ``None`` when the picked
        spec is not a merge, so the raw integer field applies instead."""
        spec = self._tracking.currentData()
        if not isinstance(spec, dict):
            return None
        identity = spec.get("identity", {})
        if identity.get("tracker_id") != "track_merge":
            return None
        names = [n for n in (identity.get("class_name") or "").split("+") if n]
        return names or None

    def _sync_track_selector_mode(self, *_args) -> None:
        """Swap the raw integer field for one checkbox per subject name when the
        picked tracking run is a merge — an object_id is a position there, not an
        animal, so picking "track 0" would silently mean a different animal in a
        recording whose merge happened to order its sources the other way."""
        names = self._subject_names_for_current_spec()
        in_name_mode = names is not None
        self._tracks_row.setVisible(not in_name_mode)
        self._tracks_label.setVisible(not in_name_mode)
        self._subjects_row.setVisible(in_name_mode)
        self._subjects_label.setVisible(in_name_mode)
        if not in_name_mode:
            return
        current = {c.text(): c.isChecked() for c in self._subject_checks}
        for check in self._subject_checks:
            check.setParent(None)
            check.deleteLater()
        self._subject_checks = []
        for name in names:
            check = QCheckBox(name)
            check.setChecked(current.get(name, True))  # default: embed every subject
            self._subjects_layout.insertWidget(len(self._subject_checks), check)
            self._subject_checks.append(check)

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

    def _sync_backends(self) -> None:
        model_id = selected_model_id(self._model)
        self._compute.set_backends(
            api_models.supported_backends(model_id) if model_id else []
        )

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

    def on_recordings_changed(self, recordings: list) -> None:
        """Cheap: no disk I/O of its own (the tracking-spec options and track-id cache
        are resolved separately, over every *eligible* recording, by
        ``resolve_eligible_recordings`` below — see its docstring for why)."""
        self._recordings = recordings
        self._every_n.set_recordings(self._recordings)
        self._sync_upstream_every_n()
        self._update_available_tracks()

    def resolve_eligible_recordings(self, recordings: list) -> dict:
        """Off the UI thread (CLAUDE.md): the disk reads — spec groups plus every
        *eligible* recording's track ids (one tracks.zarr open each, ~1 s over 60
        recordings), not just the checked ones ``on_recordings_changed`` gets.

        Checking recordings one at a time otherwise locks the tracking-spec picker to
        whichever recording happened to be checked first — its own tracking runs are the
        only ones ``spec_groups`` would ever see, so every other recording's would show
        as "not in the selected run" even when it actually has a matching one. Resolved
        once per module pick rather than once per checkbox click, so the track-id cache
        this builds only grows more complete over the eligible pool, never stale for a
        newly-checked recording."""
        groups = spec_groups(recordings, "tracking", self._model_names)
        tracks: dict[tuple[str, str], list[int]] = {}
        for recording in recordings:
            for group in groups:
                run_id = run_spec.resolve_spec_for_recording(
                    recording, "tracking", group["spec"]
                )
                if run_id is not None and (recording.id, run_id) not in tracks:
                    tracks[(recording.id, run_id)] = recording.track_ids(run_id)
        # Alignment reads keypoint track runs (§14) from the same eligible pool: a picker
        # narrowed to the checked recordings is the bug the eligible pair exists to avoid.
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
            "groups": groups,
            "tracks": tracks,
            "total": len(recordings),
            "keypoint_groups": keypoint_groups,
            "keypoint_reads": spec_reads(
                recordings, "keypoint_track", keypoint_groups, "tracking_run_id"
            ),
            "skeletons": skeletons,
        }

    def apply_eligible_recordings(self, recordings: list, resolved: dict) -> None:
        """UI thread: install the resolved spec groups + track ids and repopulate."""
        self._groups = resolved["groups"]
        self._track_cache = resolved["tracks"]
        # Before the tracking combo refills: its change re-ties the keypoint rows.
        self._keypoint_reads = dict(resolved.get("keypoint_reads") or {})
        repopulate_spec_combo(
            self._tracking,
            spec_items(self._groups, resolved["total"])
            or [("(no tracking run in this experiment)", None)],
            fallback=widest_group(self._groups)["spec"] if self._groups else None,
        )
        self._select_preferred_tracking()
        self._sync_upstream_every_n()
        self._update_available_tracks()
        self._skeleton_by_spec = dict(resolved.get("skeletons") or {})
        self._keypoint_picker.apply(
            resolved.get("keypoint_groups") or [], resolved["total"]
        )
        if self._keypoint.count() == 0:
            self._keypoint.addItem("(no keypoint track run in this experiment)", None)
        self._sync_keypoint_lineage()
        self._sync_heading_skeleton()
        self._sync_track_selector_mode()

    def _sync_keypoint_lineage(self, *_args) -> None:
        """Offer the keypoint track runs whose poses went to the chosen tracking run, recording
        by recording: the stage refuses any other (§14 *Lineage*), since its points must come
        from the same tracks."""
        spec = self._tracking.currentData()
        if not isinstance(spec, dict):
            self._keypoint_picker.restrict(None)
            return
        self._keypoint_picker.restrict(
            lineage_fits(self._keypoint_reads, group_runs(self._groups, spec)),
            "its poses went to another tracking run",
        )

    def _missing_keypoints(self, needs: str, otherwise: str) -> str:
        """Why no keypoint track run is chosen: none on this tracking run, or none at all."""
        if self._keypoint_picker.none_fit():
            return (
                f"{needs} needs a keypoint track run whose poses went to this tracking run; "
                f"none of the selected recordings has one. Run keypoint track with this "
                f"tracking run, or {otherwise}."
            )
        return (
            f"{needs} needs a keypoint track run in the selected recordings: "
            f"{guidance.missing_keypoints_hint()}; or {otherwise}."
        )

    def prefer_upstream_spec(self, module_id: str, spec: dict) -> None:
        """A next-step launch from a keypoint track run (§14): open the keypoint picker on
        that run's settings. Align stays as the user sets it — rotate or mirror depends on
        the camera (Tenet 9)."""
        if module_id == "keypoint_track":
            # Its tracking run first: the keypoint row is offered only beside it.
            self._preferred_tracking = (spec.get("aux") or {}).get("tracking")
            self._select_preferred_tracking()
            self._keypoint_picker.prefer(spec)

    def _select_preferred_tracking(self) -> None:
        """Open the tracking row on the preferred keypoint track run's own tracking run,
        when it is on offer — the pair a next-step launch from that run means."""
        if self._preferred_tracking is None:
            return
        key = run_spec.freeze_spec(self._preferred_tracking)
        for index in range(self._tracking.count()):
            data = self._tracking.itemData(index)
            if isinstance(data, dict) and run_spec.freeze_spec(data) == key:
                self._tracking.setCurrentIndex(index)
                return

    def covered_recording_ids(self) -> set[str] | None:
        covered = covered_ids(self._groups, self._tracking.currentData())
        if not self._reads_keypoints():
            return covered
        # An aligned or posed run also needs the keypoint track run in every recording.
        by_keypoints = self._keypoint_picker.covered()
        if covered is None:
            return by_keypoints
        return covered if by_keypoints is None else covered & by_keypoints

    def _effective_track_ids(self) -> list[int]:
        """The preview's own track list: always concrete integers, resolved off the
        *first* checked recording — the scrubber's track picker is one combo shared
        across every recording (`InputImageScrubber`), the same "first recording"
        convention "All tracks" already uses below."""
        if not self._subjects_row.isHidden():
            names = [c.text() for c in self._subject_checks if c.isChecked()]
            if not names or not self._recordings:
                return []
            run_id = self._run_id_for(self._recordings[0])
            if run_id is None:
                return []
            subject_labels = self._recordings[0].subject_labels(run_id)
            return [resolve_track_id(n, subject_labels) for n in names]
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
        pose = None
        if self._pose_mode() == "draw":
            error = self.validation_error()
            if error is not None:
                QMessageBox.information(self, "Input image preview", error)
                return
            pose = {
                "pose": "draw",
                "pose_min_score": optional_value(self._pose_min_score),
            }
            spec = self._keypoint_picker.spec()
            keypoint_run_ids = {
                recording.id: run_spec.resolve_spec_for_recording(
                    recording, "keypoint_track", spec
                )
                for recording in self._recordings
            }
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
            pose=pose,
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
        # A merge-run's own selector is subject names, not ints (§13) — a bare int(t)
        # here would crash restoring any such saved config; the raw-int field only
        # applies when the ids really are ints.
        if track_ids and not any(isinstance(t, str) for t in track_ids):
            self._tracks.setText(", ".join(str(int(t)) for t in track_ids))
        # The subject checkboxes themselves are rebuilt from the *current* tracking
        # selection, not this stored config, since a name selector only applies when the
        # picked spec is still a merge.
        if config.get("batch_size") is not None:
            self._batch.setValue(int(config["batch_size"]))
        self._every_n.set_value(config.get("every_n_frames"))
        # §14: the mode, the heading and the settings are settings and are restored; the
        # keypoint track run is a selection (a spec), re-derived from the recordings.
        index = self._align.findData(config.get("align") or "none")
        self._align.setCurrentIndex(index if index >= 0 else 0)
        index = self._pose.findData(config.get("pose") or "none")
        self._pose.setCurrentIndex(index if index >= 0 else 0)
        if config.get("pose_weight") is not None:
            self._pose_weight.setValue(float(config["pose_weight"]))
        set_optional_value(self._pose_min_score, config.get("pose_min_score"))
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
        align = self._align_mode()
        if not self._subjects_row.isHidden():
            all_tracks = False
            track_ids = [c.text() for c in self._subject_checks if c.isChecked()]
        else:
            all_tracks = self._all_tracks.isChecked()
            track_ids = [] if all_tracks else (self._parse_track_ids() or [])
        return {
            "tracking_spec": self._tracking.currentData(),
            "all_tracks": all_tracks,
            "track_ids": track_ids,
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
                self._keypoint_picker.spec() if self._reads_keypoints() else None
            ),
            "heading": self._heading.name() if self._needs_heading() else None,
            "pose": self._pose_mode(),
            "pose_weight": self._pose_weight.value(),
            "pose_min_score": optional_value(self._pose_min_score),
            "max_gap_s": self._max_gap.value(),
            "mirror_hold_s": self._mirror_hold.value(),
            "facing_camera_ratio": self._facing_ratio.value(),
        }

    def validation_error(self) -> str | None:
        if self._unavailable:  # nothing here can run — say so before the run does
            return self._unavailable
        # A remote run loads the weights on the cluster, which installs them there; this
        # machine's disk is not what decides whether it can run.
        remote = self._compute.current_backend() == "remote"
        not_installed = None if remote else self._not_installed()
        if not_installed is not None:
            return not_installed
        if self._tracking.currentData() is None:
            return "No tracking run in the selected recordings — run tracking first."
        if not self._subjects_row.isHidden():
            if not any(c.isChecked() for c in self._subject_checks):
                return "Pick at least one subject to embed."
        elif not self._all_tracks.isChecked() and not self._parse_track_ids():
            return (
                "Enter at least one track id (comma-separated integers), or "
                "check All tracks."
            )
        if self._align_mode() != "none":
            if self._keypoint_picker.spec() is None:
                return self._missing_keypoints("Alignment", "set Align to none")
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
        if self._pose_mode() != "none":
            if self._keypoint_picker.spec() is None:
                return self._missing_keypoints("Pose", "set Pose to Mask only")
            if self._pose_mode() == "append" and self._heading.name() is None:
                return (
                    "Pose numbers are turned so the heading points right: choose a "
                    "heading definition, or define one."
                )
        return self._every_n.warning()


@register_module_ui
class EmbeddingModuleUI(ModuleUI):
    module_id = "embedding"

    def config_widget(self) -> ConfigWidget:
        return _EmbeddingConfig()

    def preview_widget(
        self, recording, run_id: str, *, experiment_dir, parent_run_id
    ) -> QWidget | None:
        from detail.gui.run_preview import EmbeddingRunPreview

        # An embedding run's results are vectors and a scatter; the export of a run's vectors
        # is the top-bar Export…, not a button on a frame viewer, so the context goes unused.
        return EmbeddingRunPreview(recording, run_id)
