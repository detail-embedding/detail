"""Fit/train launch dialogs (§12): generate a dataset, train a model.

Experiment-scoped counterparts to ``gui/launch.py`` (which launches per-recording
processing runs). Each dialog validates, calls the experiment API
(``generate_dataset`` / ``train_model``), and holds the returned run handle in
``run_handle`` for the shell to show live progress.
"""

from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.api.experiment import Experiment
from detail.api.models import prompt_capable_models, supported_backends
from detail.configs.distill import DEFAULT_N_FRAMES, DistillConfig
from detail.configs.labeling import LabelingConfig
from detail.core.dataclasses import LabelingClass
from detail.gui.compute_target import ComputeTargetRow
from detail.gui.dialog import Dialog
from detail.gui.sizing import set_available
from detail.gui.status_color import ERROR, MUTED, READY, set_label_color
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label
from detail.gui._pydantic_form import (
    build_fields,
    field_meta,
    field_widget,
)
from detail.gui.spin import IntSpin
from detail.gui.preview import PromptTestDialog
from detail.core.registries import MODELS, TRAINERS
from detail.executor.runners import import_runner_modules
from detail.gui.recording_filter import with_filter
from detail.gui.recording_items import fill_recording_list, selected_recording_ids

# The trainer's knobs, a group inside Advanced Settings, rebuilt whenever the trainer
# changes (see _rebuild_hyperparams).
DISTILL_FIELDS = DistillConfig.model_fields
_TRAINING_KNOBS = "Training parameters"

_BASE_MODELS = ["yolo11s-seg.pt", "yolo11n-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt"]


def _colored(label: QLabel, ok: bool) -> None:
    palette = label.palette()
    palette.setColor(QPalette.ColorRole.WindowText, READY if ok else ERROR)
    label.setPalette(palette)


class GenerateDatasetDialog(Dialog):
    """Export a YOLO dataset from a completed detection/tracking run (§12)."""

    def __init__(self, experiment, source_run_id: str | None = None, parent=None):
        super().__init__(parent)
        self._experiment = experiment
        self.run_handle = None
        self.setWindowTitle("Generate a training dataset")

        layout = self.body()
        form = ConfigForm(layout, LabelingConfig)
        self._name = QLineEdit()
        form.addRow("Name:", self._name)
        self._source = QComboBox()
        self._runs = [
            r
            for r in experiment.processing_runs
            if r.get("module_id") in ("detection", "tracking")
            and r.get("status") == "complete"
        ]
        for record in self._runs:
            self._source.addItem(
                f"{record.get('module_id')}  ·  {record['id']}", record["id"]
            )
        if not self._runs:
            self._source.addItem("(no complete detection/tracking runs)", None)
        if source_run_id is not None:
            index = self._source.findData(source_run_id)
            if index >= 0:
                self._source.setCurrentIndex(index)
        form.addRow("Source run:", self._source)

        self._mode = QComboBox()
        self._mode.addItem("All frames", "all")
        self._mode.addItem("Every N frames", "every_n")
        self._mode.currentIndexChanged.connect(self._sync_every_n)
        form.addRow("Frames:", self._mode)
        self._every_n = IntSpin()  # dataset-export's own sampling; no config model
        self._every_n.setRange(2, 100000)
        self._every_n.setValue(10)
        form.addRow("N:", self._every_n)
        # Same two parameters the labeling runner takes, so they read their bounds and
        # help from its model rather than repeating them here. Their labels are plain
        # strings, so each says where it belongs instead of being routed from the model
        # the way a ``field_label`` row is.
        self._val_split = field_widget(LabelingConfig, "val_split")
        self._val_split.setValue(0.15)
        form.addRow("Validation split:", self._val_split, advanced=True)
        self._min_area = field_widget(LabelingConfig, "min_contour_area")
        self._min_area.setValue(100)
        form.addRow("Min contour area (px):", self._min_area, advanced=True)
        form.finish()
        layout.addStretch()
        self._sync_every_n()

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Generate")
        box.accepted.connect(self._run)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

    def _sync_every_n(self) -> None:
        self._every_n.setEnabled(self._mode.currentData() == "every_n")

    def _frame_selection(self) -> dict:
        if self._mode.currentData() == "every_n":
            return {"mode": "every_n", "n": self._every_n.value()}
        return {"mode": "all"}

    def _run(self) -> None:
        name = self._name.text().strip()
        source = self._source.currentData()
        if not name:
            QMessageBox.warning(self, "Generate dataset", "Name is required.")
            return
        if source is None:
            QMessageBox.warning(
                self,
                "Generate dataset",
                "No complete detection/tracking run to export.",
            )
            return
        try:
            self.run_handle = self._experiment.generate_dataset(
                name,
                source,
                frame_selection=self._frame_selection(),
                val_split=self._val_split.value(),
                min_contour_area=self._min_area.value(),
            )
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Generate dataset", str(exc))
            return
        self.accept()


class GenerateLabelingDialog(Dialog):
    """Label a random frame sample with a prompt-capable model into a YOLO dataset (§12) — no run
    needed."""

    def __init__(self, experiment, parent=None):
        super().__init__(parent)
        self._experiment = experiment
        self.run_handle = None
        self.setWindowTitle("Label a dataset")

        layout = self.body()
        form = ConfigForm(layout, LabelingConfig)
        self._name = QLineEdit()
        form.addRow("Name:", self._name)
        self._recordings = QListWidget()
        self._recordings.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        fill_recording_list(self._recordings, experiment.recordings)
        form.addRow(
            "Recordings (none = all):",
            with_filter(self._recordings, experiment, select_buttons=True),
        )
        # Model + size/variant, mirroring the detection form: only prompt-capable models can
        # label. A no-variant model (SAM3) shows "(default)" → data None (a non-None variant on a
        # no-variant model raises).
        model_meta = field_meta(LabelingConfig, "model_id")
        self._model = QComboBox()
        self._model.addItems(prompt_capable_models() or ["(no model registered)"])
        self._model.setCurrentText(
            LabelingConfig.model_fields["model_id"].default  # default to sam3
        )
        self._model.setToolTip(model_meta["help"])
        self._model.currentTextChanged.connect(self._sync_variants)
        self._model.currentTextChanged.connect(self._sync_backends)
        form.addRow(
            field_label("labeling", "model_id", "Teacher (labels the frames):"),
            self._model,
        )
        variant_meta = field_meta(LabelingConfig, "model_variant")
        self._variant = QComboBox()
        self._variant.setToolTip(variant_meta["help"])
        form.addRow(
            field_label("labeling", "model_variant", "Teacher size:"),
            self._variant,
        )
        self._sync_variants()
        # Scalar rows are built from LabelingConfig so their range/help match the CLI + validator.
        fields = build_fields(LabelingConfig)
        nf_label, self._n_frames = fields["n_frames"]
        self._n_frames.setValue(
            500
        )  # dialog context default (the model requires n_frames)
        form.addRow(field_label("labeling", "n_frames", nf_label + ":"), self._n_frames)
        samp = field_meta(LabelingConfig, "sampling")  # a Literal → hand-built combo
        self._sampling = QComboBox()
        self._sampling.addItems(["stratified", "uniform"])
        self._sampling.setToolTip(samp["help"])
        form.addRow(
            field_label("labeling", "sampling", samp["label"] + ":"), self._sampling
        )
        lt_label, self._logit = fields["logit_threshold"]
        self._logit.setValue(0.5)
        form.addRow(
            field_label("labeling", "logit_threshold", lt_label + ":"), self._logit
        )
        vs_label, self._val_split = fields["val_split"]
        self._val_split.setRange(0.0, 0.9)  # practical cap; the model bound is < 1.0
        self._val_split.setValue(0.15)
        form.addRow(
            field_label("labeling", "val_split", vs_label + ":"), self._val_split
        )
        bg_label, self._background = fields["background_share"]
        form.addRow(
            field_label("labeling", "background_share", bg_label + ":"),
            self._background,
        )
        ma_label, self._min_area = fields["min_contour_area"]
        self._min_area.setValue(100)
        form.addRow(
            field_label("labeling", "min_contour_area", ma_label + ":"), self._min_area
        )
        self._compute = ComputeTargetRow(allow_remote=True)
        self._compute.add_to(form.section())
        self._sync_backends()
        form.finish()

        layout.addWidget(QLabel("Classes (name + text prompt):"))
        self._classes_box = QVBoxLayout()
        layout.addLayout(self._classes_box)
        self._class_rows: list[tuple[QLineEdit, QLineEdit, QWidget]] = []
        add_btn = QPushButton("Add class")
        add_btn.clicked.connect(self._add_class_row)
        layout.addWidget(add_btn)
        self._add_class_row()  # start with one row
        test_btn = QPushButton("Test prompts")
        test_btn.clicked.connect(self._open_prompt_test)
        layout.addWidget(test_btn)
        layout.addStretch()

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Label")
        box.accepted.connect(self._run)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

    def _sync_variants(self) -> None:
        """Populate the size/variant combo from the selected model (mirrors the detection form).
        A no-variant model (SAM3) shows a disabled '(default)' whose data is None."""
        model_cls = MODELS.get(self._model.currentText())
        variants = model_cls.variants() if model_cls is not None else []
        self._variant.clear()
        if variants:
            for name in variants:
                self._variant.addItem(name, name)
            self._variant.setEnabled(True)
        else:
            self._variant.addItem("(default)", None)
            self._variant.setEnabled(False)

    def _sync_backends(self) -> None:
        self._compute.set_backends(supported_backends(self._model.currentText()))

    def _add_class_row(self) -> None:
        row = QWidget()
        hbox = QHBoxLayout(row)
        hbox.setContentsMargins(0, 0, 0, 0)
        name = QLineEdit()
        name.setPlaceholderText("class name")
        prompt = QLineEdit()
        prompt.setPlaceholderText("text prompt")
        remove = QPushButton("−")
        remove.setFixedWidth(28)
        remove.clicked.connect(lambda: self._remove_class_row(row))
        hbox.addWidget(name, 1)
        hbox.addWidget(prompt, 2)
        hbox.addWidget(remove)
        self._classes_box.addWidget(row)
        self._class_rows.append((name, prompt, row))

    def _remove_class_row(self, row: QWidget) -> None:
        self._class_rows = [r for r in self._class_rows if r[2] is not row]
        row.setParent(None)

    def _classes(self) -> list[LabelingClass]:
        classes = []
        for i, (name, prompt, _) in enumerate(self._class_rows):
            n, p = name.text().strip(), prompt.text().strip()
            if n and p:
                classes.append(LabelingClass(class_id=i, name=n, prompt=p))
        return classes

    def _open_prompt_test(self) -> None:
        dialog = self._build_prompt_test()
        if dialog is not None:
            dialog.exec()

    def _build_prompt_test(self) -> PromptTestDialog | None:
        """The labeling equivalent of detection's *Test prompts…* — preview what the
        per-class prompts detect on sampled frames (with the selected model) before a
        full labeling run. Returns the dialog (or ``None`` when there's nothing to test)
        so it can be exercised without the blocking modal ``exec``."""
        classes = self._classes()
        if not classes:
            QMessageBox.information(
                self, "Test prompts", "Add at least one class with a text prompt."
            )
            return None
        selected = selected_recording_ids(self._recordings)
        recordings = [self._experiment.get_recording(sid) for sid in selected] or list(
            self._experiment.recordings
        )
        if not recordings:
            QMessageBox.information(self, "Test prompts", "Add a recording to test on.")
            return None
        return PromptTestDialog(
            recordings,
            model_id=self._model.currentText(),
            prompt_text="\n".join(c.prompt for c in classes),
            compute=self._compute.compute_target(),
            parent=self,
            model_variant=self._variant.currentData(),
        )

    def _run(self) -> None:
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(self, "Label dataset", "Name is required.")
            return
        classes = self._classes()
        if not classes:
            QMessageBox.warning(
                self, "Label dataset", "Add at least one class (name + prompt)."
            )
            return
        selected = selected_recording_ids(self._recordings)
        try:
            self.run_handle = self._experiment.label_dataset(
                name,
                classes,
                n_frames=self._n_frames.value(),
                source_recording_ids=selected or None,
                logit_threshold=self._logit.value(),
                min_contour_area=self._min_area.value(),
                val_split=self._val_split.value(),
                background_share=self._background.value(),
                compute=self._compute.compute_target(),
                sampling=self._sampling.currentText(),
                model_id=self._model.currentText(),
                model_variant=self._variant.currentData(),
            )
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Label dataset", str(exc))
            return
        self.accept()


class TrainModelDialog(Dialog):
    """Train a custom detection model from a completed dataset (§12)."""

    def __init__(self, experiment, dataset_id: str | None = None, parent=None):
        super().__init__(parent)
        self._experiment = experiment
        self.run_handle = None
        self.setWindowTitle("Train a model")
        import_runner_modules(PIPELINE_RUNNERS)  # populate the TRAINERS registry

        layout = self.body()
        form = ConfigForm(layout)
        self._name = QLineEdit()
        form.addRow("Name:", self._name)
        self._dataset = QComboBox()
        self._datasets = [d for d in experiment.datasets if d.status == "complete"]
        for ref in self._datasets:
            self._dataset.addItem(
                f"{ref.name or ref.id}  ·  {ref.id}  ({ref.n_train}/{ref.n_val})",
                ref.id,
            )
        if not self._datasets:
            self._dataset.addItem("(no complete datasets)", None)
        if dataset_id is not None:
            index = self._dataset.findData(dataset_id)
            if index >= 0:
                self._dataset.setCurrentIndex(index)
        form.addRow("Dataset:", self._dataset)

        self._trainer = QComboBox()
        for trainer_id in sorted(TRAINERS):
            self._trainer.addItem(trainer_id, trainer_id)
        if not TRAINERS:
            self._trainer.addItem("(no trainer registered)", None)
        self._trainer.currentIndexChanged.connect(self._sync_trainer)
        self._trainer.hide()  # one training framework today; it's a framework, not a model — kept for the config, not shown
        self._readiness = QLabel("")
        form.addRow("", self._readiness)

        self._base_model = QComboBox()
        self._base_model.setEditable(True)
        self._base_model.addItems(_BASE_MODELS)
        form.addRow("Student (the fast model):", self._base_model)

        self._compute = ComputeTargetRow(allow_remote=True)
        self._compute.add_to(form.section())

        # The trainer's knobs get their own section: they are rebuilt wholesale when the
        # trainer changes, and clearing a layout shared with the compute rows would take
        # those with them.
        self._form = form
        self._hparams: dict[str, QWidget] = {}
        form.finish()
        layout.addStretch()
        self._sync_trainer()  # fills readiness, backends, hyperparams for the first trainer

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Train")
        box.accepted.connect(self._run)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

    def _trainer_instance(self):
        trainer_id = self._trainer.currentData()
        cls = TRAINERS.get(trainer_id) if trainer_id else None
        return cls() if cls is not None else None

    def _sync_trainer(self) -> None:
        trainer = self._trainer_instance()
        ready = trainer is not None and trainer.is_ready()
        self._readiness.setText(
            "✓ ready to train"
            if ready
            else "✗ training framework not ready: install its backend"
        )
        _colored(self._readiness, ready)
        self._compute.set_backends(trainer.supported_backends if trainer else [])
        self._rebuild_hyperparams(trainer)

    def _rebuild_hyperparams(self, trainer) -> None:
        self._form.advanced().clear(_TRAINING_KNOBS)
        knobs = self._form.section(_TRAINING_KNOBS)
        self._hparams = {}
        model_cls = getattr(trainer, "config_model", None) if trainer else None
        if model_cls is None:
            self._form.finish()
            return
        # The pydantic model is the single source for each hyperparam's range/help
        # (build_fields reads them); defaults stay per-surface, so set them here.
        for name, (label, widget) in build_fields(model_cls).items():
            default = model_cls.model_fields[name].default
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(default))
            else:
                widget.setValue(default)
            self._hparams[name] = widget
            knobs.addRow(f"{label}:", widget)
        self._form.finish()

    def _run(self) -> None:
        name = self._name.text().strip()
        dataset_id = self._dataset.currentData()
        trainer_id = self._trainer.currentData()
        if not name:
            QMessageBox.warning(self, "Train model", "Name is required.")
            return
        if dataset_id is None:
            QMessageBox.warning(self, "Train model", "No complete dataset to train on.")
            return
        compute = self._compute.compute_target()
        hyperparams = {
            key: (w.isChecked() if isinstance(w, QCheckBox) else w.value())
            for key, w in self._hparams.items()
        }
        try:
            self.run_handle = self._experiment.train_model(
                name,
                dataset_id,
                trainer_id,
                self._base_model.currentText().strip(),
                hyperparams=hyperparams,
                compute=compute,
            )
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Train model", str(exc))
            return
        self.accept()


#: The kinds of model **Distill new model** makes, in the order the choice lists them.
MODEL_TYPES = (("detector", "Detector (boxes, masks)"), ("keypoints", "Keypoints"))


class DistillDialog(Dialog):
    """**Distill new model** (§12): one durable run that labels frames with a teacher and
    trains a fast YOLO model on them, for either kind of model.

    *Model type* chooses the kind. **Detector**: a prompt-capable model labels a frame sample
    and a YOLO detector is trained on it (with a *Test prompts…* preview); launches
    ``Experiment.distill``. **Keypoints**: finished keypoint track runs label the frames and a
    YOLO-pose student is trained and registered (``gui/keypoint_distill.py``'s form); launches
    ``Experiment.distill_keypoints``. Name and experiment are shared. The returned
    ``AttachedDistillRun`` is held in ``run_handle`` for the shell to stream. Compute is per
    stage (SAM3 labeling needs CUDA)."""

    def __init__(self, experiment=None, parent=None, *, model_type: str = "detector"):
        super().__init__(parent)
        self.run_handle = None
        self.run_experiment = None  # the experiment the launched run belongs to
        self.setWindowTitle("Distill new model")
        import_runner_modules(PIPELINE_RUNNERS)  # populate the TRAINERS registry

        # Distilling is machine-global, so you pick the experiment (and its recordings) here
        # rather than opening one first. Put the passed (already-open) instance first,
        # replacing any workspace duplicate by dir, so a caller holding it — the app, a test
        # that monkeypatches its ``.distill`` — acts on the *same* object, not a fresh
        # Experiment for the same directory.
        self._experiments = Experiment.list_all()
        if experiment is not None:
            self._experiments = [experiment] + [
                e for e in self._experiments if e.dir != experiment.dir
            ]
        self._experiment = None

        outer = self.body()
        head = QFormLayout()
        self._type = QComboBox()
        for value, label in MODEL_TYPES:
            self._type.addItem(label, value)
        self._type.setCurrentIndex(max(self._type.findData(model_type), 0))
        head.addRow("Model type:", self._type)
        self._name = QLineEdit()
        head.addRow("Name:", self._name)
        self._exp_combo = QComboBox()
        for exp in self._experiments:
            self._exp_combo.addItem(exp.name, exp)
        if not self._experiments:
            self._exp_combo.addItem("(no experiments — create one first)", None)
            self._exp_combo.setEnabled(False)
        elif experiment is not None:
            for i, exp in enumerate(self._experiments):
                if exp.dir == experiment.dir:
                    self._exp_combo.setCurrentIndex(i)
                    break
        self._experiment = self._exp_combo.currentData()
        self._exp_combo.currentIndexChanged.connect(self._sync_experiment)
        head.addRow("Experiment:", self._exp_combo)
        outer.addLayout(head)

        # The detector's fields, then the keypoint model's; Model type shows one of them.
        self._pages = QStackedWidget()
        detector_page = QWidget()
        layout = QVBoxLayout(detector_page)
        layout.setContentsMargins(0, 0, 0, 0)
        form = ConfigForm(layout, LabelingConfig)
        self._recordings = QListWidget()
        self._recordings.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        picker = with_filter(self._recordings, self._experiment, select_buttons=True)
        self._recording_filter = picker.filter
        form.addRow("Recordings (none = all):", picker)
        self._sync_recordings()

        # --- Labeling stage (mirrors GenerateLabelingDialog) ---
        model_meta = field_meta(LabelingConfig, "model_id")
        self._model = QComboBox()
        self._model.addItems(prompt_capable_models() or ["(no model registered)"])
        self._model.setCurrentText(LabelingConfig.model_fields["model_id"].default)
        self._model.setToolTip(model_meta["help"])
        self._model.currentTextChanged.connect(self._sync_variants)
        self._model.currentTextChanged.connect(self._sync_label_backends)
        form.addRow(
            field_label("labeling", "model_id", "Teacher (labels the frames):"),
            self._model,
        )
        self._variant = QComboBox()
        form.addRow(
            field_label("labeling", "model_variant", "Teacher size:"), self._variant
        )
        self._sync_variants()
        fields = build_fields(LabelingConfig)
        nf_label, self._n_frames = fields["n_frames"]
        self._n_frames.setValue(DEFAULT_N_FRAMES)
        form.addRow(field_label("labeling", "n_frames", nf_label + ":"), self._n_frames)
        self._sampling = QComboBox()
        self._sampling.addItems(["stratified", "uniform"])
        form.addRow(
            field_label("labeling", "sampling", "Frame sampling:"), self._sampling
        )
        _, self._logit = fields["logit_threshold"]
        self._logit.setValue(0.5)
        form.addRow(
            field_label("labeling", "logit_threshold", "Confidence threshold:"),
            self._logit,
        )
        _, self._val_split = fields["val_split"]
        self._val_split.setRange(0.0, 0.9)
        self._val_split.setValue(0.15)
        form.addRow(
            field_label("labeling", "val_split", "Validation split:"), self._val_split
        )
        bg_label, self._background = fields["background_share"]
        form.addRow(
            field_label("labeling", "background_share", bg_label + ":"),
            self._background,
        )
        _, self._min_area = fields["min_contour_area"]
        self._min_area.setValue(100)
        form.addRow(
            field_label("labeling", "min_contour_area", "Min contour area (px):"),
            self._min_area,
        )
        self._compute = ComputeTargetRow(
            allow_remote=True
        )  # labeling (also drives Test prompts)
        self._compute.add_to(form.section(), prefix="Label")
        self._sync_label_backends()

        # --- Training stage (mirrors TrainModelDialog) ---
        self._trainer = QComboBox()
        for trainer_id in sorted(TRAINERS):
            self._trainer.addItem(trainer_id, trainer_id)
        if not TRAINERS:
            self._trainer.addItem("(no trainer registered)", None)
        self._trainer.currentIndexChanged.connect(self._sync_trainer)
        self._trainer.hide()  # one training framework today; it's a framework, not a model — kept for the config, not shown
        self._readiness = QLabel("")
        form.addRow("", self._readiness)
        self._base_model = QComboBox()
        self._base_model.setEditable(True)
        self._base_model.addItems(_BASE_MODELS)
        form.addRow("Student (the fast model):", self._base_model)
        self._train_compute = ComputeTargetRow(allow_remote=True)
        self._train_compute.add_to(form.section(), prefix="Train")
        self._stage_videos = QCheckBox()
        self._stage_videos.setToolTip(DISTILL_FIELDS["stage_videos"].description)
        form.addRow(
            f"{DISTILL_FIELDS['stage_videos'].json_schema_extra['label']}:",
            self._stage_videos,
        )
        for row in (self._compute, self._train_compute):
            row.backend.currentTextChanged.connect(self._sync_stage_videos)
        self._sync_stage_videos()
        self._form = form
        self._hparams: dict[str, QWidget] = {}
        form.finish()
        self._sync_trainer()  # readiness + backends + hyperparams for the first trainer

        layout.addWidget(QLabel("Classes (name + text prompt):"))
        self._classes_box = QVBoxLayout()
        layout.addLayout(self._classes_box)
        self._class_rows: list[tuple[QLineEdit, QLineEdit, QWidget]] = []
        add_btn = QPushButton("Add class")
        add_btn.clicked.connect(self._add_class_row)
        layout.addWidget(add_btn)
        self._add_class_row()
        test_btn = QPushButton("Test prompts")
        test_btn.clicked.connect(self._open_prompt_test)
        layout.addWidget(test_btn)
        layout.addStretch()
        self._pages.addWidget(detector_page)

        from detail.gui.keypoint_distill import KeypointDistillForm

        self._keypoints = KeypointDistillForm(self._experiment)
        self._keypoints.readiness_changed.connect(self._sync_ok)
        self._pages.addWidget(self._keypoints)
        outer.addWidget(self._pages, 1)
        self._type.currentIndexChanged.connect(self._sync_type)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok = box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setText("Distill")
        box.accepted.connect(self._run)
        box.rejected.connect(self.reject)
        # Why Distill is greyed, on screen beside it: a tooltip alone went unseen (a greyed
        # button with no visible reason was reported as broken). Inside the button box, so it
        # never scrolls away from the button it explains.
        self._why = QLabel("")
        self._why.setWordWrap(True)
        set_label_color(self._why, MUTED)
        box.layout().insertWidget(0, self._why, 1)
        self.set_buttons(box)
        self._sync_type()

    def model_type(self) -> str:
        return str(self._type.currentData())

    def keypoint_form(self):
        return self._keypoints

    def _sync_type(self, *_args) -> None:
        self._pages.setCurrentIndex(0 if self.model_type() == "detector" else 1)
        self._sync_ok()

    def _sync_ok(self) -> None:
        reason = self._keypoints.blocker() if self.model_type() == "keypoints" else None
        set_available(self._ok, reason is None, reason or "")
        self._why.setText(reason or "")
        self._why.setVisible(reason is not None)

    def why_text(self) -> str:
        """The visible reason Distill is greyed, or empty."""
        return self._why.text() if self._why.isVisibleTo(self) else ""

    def _sync_variants(self) -> None:
        model_cls = MODELS.get(self._model.currentText())
        variants = model_cls.variants() if model_cls is not None else []
        self._variant.clear()
        if variants:
            for name in variants:
                self._variant.addItem(name, name)
            self._variant.setEnabled(True)
        else:
            self._variant.addItem("(default)", None)
            self._variant.setEnabled(False)

    def _sync_label_backends(self) -> None:
        self._compute.set_backends(supported_backends(self._model.currentText()))
        self._sync_stage_videos()

    def _sync_experiment(self) -> None:
        self._experiment = self._exp_combo.currentData()
        self._sync_recordings()
        self._keypoints.set_experiment(self._experiment)

    def _sync_recordings(self) -> None:
        self._recordings.clear()
        self._recording_filter.set_experiment(self._experiment)
        if self._experiment is None:
            return
        fill_recording_list(self._recordings, self._experiment.recordings)

    def _sync_trainer(self) -> None:
        trainer_id = self._trainer.currentData()
        cls = TRAINERS.get(trainer_id) if trainer_id else None
        trainer = cls() if cls is not None else None
        ready = trainer is not None and trainer.is_ready()
        self._readiness.setText(
            "✓ ready to train"
            if ready
            else "✗ training framework not ready: install its backend"
        )
        _colored(self._readiness, ready)
        self._train_compute.set_backends(trainer.supported_backends if trainer else [])
        self._sync_stage_videos()
        self._rebuild_hyperparams(trainer)

    def _rebuild_hyperparams(self, trainer) -> None:
        self._form.advanced().clear(_TRAINING_KNOBS)
        knobs = self._form.section(_TRAINING_KNOBS)
        self._hparams = {}
        model_cls = getattr(trainer, "config_model", None) if trainer else None
        if model_cls is None:
            self._form.finish()
            return
        for name, (label, widget) in build_fields(model_cls).items():
            default = model_cls.model_fields[name].default
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(default))
            else:
                widget.setValue(default)
            self._hparams[name] = widget
            knobs.addRow(f"{label}:", widget)
        self._form.finish()

    def _add_class_row(self) -> None:
        row = QWidget()
        hbox = QHBoxLayout(row)
        hbox.setContentsMargins(0, 0, 0, 0)
        name = QLineEdit()
        name.setPlaceholderText("class name")
        prompt = QLineEdit()
        prompt.setPlaceholderText("text prompt")
        remove = QPushButton("−")
        remove.setFixedWidth(28)
        remove.clicked.connect(lambda: self._remove_class_row(row))
        hbox.addWidget(name, 1)
        hbox.addWidget(prompt, 2)
        hbox.addWidget(remove)
        self._classes_box.addWidget(row)
        self._class_rows.append((name, prompt, row))

    def _remove_class_row(self, row: QWidget) -> None:
        self._class_rows = [r for r in self._class_rows if r[2] is not row]
        row.setParent(None)

    def _classes(self) -> list[LabelingClass]:
        classes = []
        for i, (name, prompt, _) in enumerate(self._class_rows):
            n, p = name.text().strip(), prompt.text().strip()
            if n and p:
                classes.append(LabelingClass(class_id=i, name=n, prompt=p))
        return classes

    def _open_prompt_test(self) -> None:
        classes = self._classes()
        if not classes:
            QMessageBox.information(
                self, "Test prompts", "Add at least one class with a text prompt."
            )
            return
        selected = selected_recording_ids(self._recordings)
        recordings = [self._experiment.get_recording(sid) for sid in selected] or list(
            self._experiment.recordings
        )
        if not recordings:
            QMessageBox.information(self, "Test prompts", "Add a recording to test on.")
            return
        PromptTestDialog(
            recordings,
            model_id=self._model.currentText(),
            prompt_text="\n".join(c.prompt for c in classes),
            compute=self._compute.compute_target(),
            parent=self,
            model_variant=self._variant.currentData(),
        ).exec()

    def _sync_stage_videos(self) -> None:
        """The background upload only exists for a remote run: greyed (not hidden, per the GUI
        style rule) and cleared otherwise, so a stale tick never reaches ``distill``."""
        if not hasattr(self, "_stage_videos"):
            return  # the first backend sync runs while the form is still being built
        remote = "remote" in (
            self._compute.current_backend(),
            self._train_compute.current_backend(),
        )
        self._stage_videos.setEnabled(remote)
        if not remote:
            self._stage_videos.setChecked(False)

    def _run(self) -> None:
        if self._experiment is None:
            QMessageBox.warning(self, "Distill", "Create an experiment first.")
            return
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(self, "Distill", "Name is required.")
            return
        if self.model_type() == "keypoints":
            try:
                self.run_handle = self._keypoints.launch(name)
            except ValueError as exc:
                QMessageBox.warning(self, "Distill", str(exc))
                return
            self.run_experiment = self._experiment
            self.accept()
            return
        classes = self._classes()
        if not classes:
            QMessageBox.warning(
                self, "Distill", "Add at least one class (name + prompt)."
            )
            return
        trainer_id = self._trainer.currentData()
        if trainer_id is None:
            QMessageBox.warning(self, "Distill", "No trainer available.")
            return
        selected = selected_recording_ids(self._recordings)
        hyperparams = {
            key: (w.isChecked() if isinstance(w, QCheckBox) else w.value())
            for key, w in self._hparams.items()
        }
        try:
            self.run_handle = self._experiment.distill(
                name,
                classes=classes,
                n_frames=self._n_frames.value(),
                source_recording_ids=selected or None,
                sampling=self._sampling.currentText(),
                logit_threshold=self._logit.value(),
                min_contour_area=self._min_area.value(),
                val_split=self._val_split.value(),
                background_share=self._background.value(),
                label_model_id=self._model.currentText(),
                label_model_variant=self._variant.currentData(),
                label_compute=self._compute.compute_target(),
                base_model=self._base_model.currentText().strip(),
                trainer_id=trainer_id,
                hyperparams=hyperparams,
                train_compute=self._train_compute.compute_target(),
                stage_videos=self._stage_videos.isChecked(),
            )
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Distill", str(exc))
            return
        self.run_experiment = self._experiment
        self.accept()


_POSE_BASE_MODELS = [
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
]
