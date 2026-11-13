"""**Set up keypoints…** (§26 *Keypoint setup*, #717): one dialog that teaches a keypoint model
for the whole experiment and distills a fast student from it.

One teacher recording per acquisition setup (``Experiment.keypoint_teacher_plan``, read off the
UI thread because it reads every recording), its first minutes only, detected with a distilled
detector and tracked; then the patch method learns the animal from an example set and labels
the sampled frames, and a YOLO-pose student trains on them. Without an example set the run
stops once the teacher is tracked, and its page (``pages.DistillRunDetail``) offers the
marking and **Continue setup**; with one already marked it goes straight through.

The dataset fields are read off ``KeypointDatasetConfig`` and the training knobs off the
trainer's config model (§6.5), as :class:`~detail.gui.keypoint_distill.KeypointDistillForm`
reads them; ``Experiment.start_keypoint_setup`` does the launching.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.keypoint_dataset import KeypointDatasetConfig
from detail.core import class_names
from detail.core.protocols import ModelNotReadyError
from detail.core.registries import KEYPOINT_MODELS, TRAINERS
from detail.executor.runners import import_runner_modules
from detail.gui._pydantic_form import (
    build_fields,
    choice_combo,
    optional_spin,
    optional_value,
)
from detail.gui.busy import run_busy
from detail.gui.class_example_sets import ClassExampleSets
from detail.gui.compute_target import ComputeTargetRow
from detail.gui.dialog import Dialog
from detail.gui.example_set_picker import ExampleSetPicker
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label
from detail.gui.keypoint_distill import _POSE_BASES, patch_methods
from detail.gui.model_availability import fill_variants
from detail.gui.sizing import set_available
from detail.gui.spin import IntSpin
from detail.gui.status_color import ERROR, set_label_color

_MODULE = "keypoint_dataset"
_TRAINING_KNOBS = "Training parameters"
#: Set separately: the cut-off with its "the method's default" state, the input as a choice.
_OWN_ROWS = ("score_cutoff", "student_input")
#: The frames a setup labels by default: the field has none (it is the distill's one required
#: input), and a setup is the quick path, so it opens on a size that trains in minutes.
_DEFAULT_N_FRAMES = 300


def closed_set_detectors(experiment) -> list[dict]:
    """The detectors a setup can find the animal with: closed-set ones that name classes
    (``detection_model_entries``), the experiment's own trained ones last."""
    from detail.gui.modules.detection import detection_model_entries

    return [
        e
        for e in detection_model_entries(experiment)
        if not e["prompt_capable"] and e["classes"]
    ]


def method_default_cutoff(experiment, model_id: str) -> float | None:
    """The score cut-off ``model_id`` teaches with when none is given (its
    ``score_spec.default_cutoff``), or None when the method sets none."""
    from detail.models import resolve as models_resolve

    cls = models_resolve.keypoint_method_class(
        getattr(experiment, "dir", None), model_id
    )
    cutoff = getattr(getattr(cls, "score_spec", None), "default_cutoff", None)
    return None if cutoff is None else float(cutoff)


class KeypointSetupDialog(Dialog):
    """Set up keypoints for the experiment; on Start, ``run_handle`` holds the
    ``AttachedDistillRun`` for the shell to stream (as the distill dialog's does)."""

    def __init__(self, experiment, parent=None):
        super().__init__(parent)
        self.run_handle = None
        self.run_experiment = experiment
        self._experiment = experiment
        self._plan_read = False
        self.setWindowTitle("Set up keypoints")
        import_runner_modules(PIPELINE_RUNNERS)  # populate the TRAINERS registry

        layout = self.body()
        intro = QLabel(
            "Teach a keypoint model from the first minutes of one recording per camera or "
            "box, then train a fast model on what it learns. Without an example set the run "
            "stops once those recordings are tracked, so you can mark one on them."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self._plan_error = QLabel("")
        self._plan_error.setWordWrap(True)
        set_label_color(self._plan_error, ERROR)
        self._plan_error.hide()
        layout.addWidget(self._plan_error)

        form = ConfigForm(layout, KeypointDatasetConfig)
        self._form = form
        self._name = QLineEdit()
        # Named after the animal until the person types their own.
        self._name_edited = False
        self._name.textEdited.connect(self._on_name_edited)
        form.addRow("Name:", self._name)

        # --- the animal: a distilled detector and its class ---
        self._detector = QComboBox()
        newest_trained = 0
        for entry in closed_set_detectors(experiment):
            self._detector.addItem(entry["label"], entry)
            if entry["kind"] == "trained":
                newest_trained = self._detector.count() - 1
        if self._detector.count() == 0:
            self._detector.addItem("(no distilled detector yet — distill one first)")
            self._detector.setEnabled(False)
        # Trained models are listed oldest first: the last is the one just distilled.
        self._detector.setCurrentIndex(newest_trained)
        self._class = QComboBox()
        self._detector.currentIndexChanged.connect(lambda _i: self._sync_classes())
        self._class.currentIndexChanged.connect(lambda _i: self._sync_name())
        form.addRow("Detector:", self._detector)
        form.addRow("Animal (detector class):", self._class)
        # Two or more animals told apart as classes (§26 *One example set per detection
        # class*): each tracked on its own, the tracks merged, one example set per class.
        self._several = QCheckBox("Two or more animals (each tracked, then merged)")
        self._several.setToolTip(
            "For animals the detector tells apart as classes, such as a resident and an "
            "intruder: each class is tracked on its own, the tracks are merged, and each "
            "class gets its own example set, so animals that look different are each "
            "matched against their own clicks. One fast model is trained for all of them."
        )
        self._several.toggled.connect(lambda _on: self._sync_animals())
        form.addRow("", self._several)
        self._animals_box = QWidget()
        self._animals_layout = QVBoxLayout(self._animals_box)
        self._animals_layout.setContentsMargins(0, 0, 0, 0)
        self._animal_ticks: dict[str, QCheckBox] = {}
        form.addRow("Animals:", self._animals_box)

        # --- the teacher recordings ---
        self._minutes = IntSpin()
        self._minutes.setRange(1, 60)
        self._minutes.setValue(5)
        self._minutes.setSuffix(" min")
        self._minutes.setToolTip(
            "How much of each teacher recording to use, from its start. A shorter "
            "recording is used whole."
        )
        self._minutes.valueChanged.connect(lambda _v: self._sync_spans())
        form.addRow("Use the first:", self._minutes)
        self._teachers: dict[str, QComboBox] = {}
        self._spans: dict[str, QLabel] = {}
        self._durations: dict[str, float] = {}
        # Filled when the plan arrives, off the UI thread.
        self._teacher_box = QWidget()
        self._teacher_rows = QFormLayout(self._teacher_box)
        self._teacher_rows.setContentsMargins(0, 0, 0, 0)
        form.addRow(QLabel("Teacher recordings, one per camera or box:"))
        form.addRow(self._teacher_box)

        # --- the teacher method and its example set ---
        self._method = QComboBox()
        for model_id in patch_methods():
            self._method.addItem(model_id, model_id)
        self._method.setCurrentIndex(max(0, self._method.findData("dinov3_match")))
        self._method.currentIndexChanged.connect(lambda _i: self._sync_method())
        form.addRow("Teacher method:", self._method)
        self._variant = QComboBox()
        form.addRow("Size:", self._variant)
        self._have_examples = QCheckBox(
            "I already have an example set for this animal (no pause)"
        )
        self._have_examples.setToolTip(
            "An example set marked on any recordings of this animal lets the run go "
            "straight through. Otherwise it waits, once the teacher recordings are tracked, "
            "for you to mark one on them."
        )
        self._have_examples.toggled.connect(lambda _on: self._sync())
        form.addRow("", self._have_examples)
        self._examples = ExampleSetPicker(experiment)
        self._examples.changed.connect(self._sync)
        form.addRow("Examples:", self._examples)
        self._class_sets = ClassExampleSets(experiment=experiment, checkable=False)
        self._class_sets.changed.connect(self._sync)
        form.addRow("Examples:", self._class_sets)
        self._score_cutoff = optional_spin(
            KeypointDatasetConfig, "score_cutoff", "the method's default"
        )
        self._score_cutoff.valueChanged.connect(lambda _v: self._sync())
        form.addRow(
            field_label(_MODULE, "score_cutoff", "Score cut-off:"), self._score_cutoff
        )

        # --- where each part runs ---
        self._detect_compute = ComputeTargetRow(allow_remote=True)
        self._detect_compute.add_to(form.section(), prefix="Detect")
        # The patch method labels the sampled frames here, where the videos are.
        self._keypoints_compute = ComputeTargetRow()
        self._keypoints_compute.add_to(form.section(), prefix="Label")

        # --- the dataset, off its config model ---
        self._fields: dict[str, QWidget] = {}
        for name, (label, widget) in build_fields(KeypointDatasetConfig).items():
            if name in _OWN_ROWS:
                continue
            self._fields[name] = widget
            form.addRow(field_label(_MODULE, name, label + ":"), widget)
        self._fields["n_frames"].setValue(_DEFAULT_N_FRAMES)
        self._student_input = choice_combo(KeypointDatasetConfig, "student_input")
        self._student_input.currentIndexChanged.connect(
            lambda _i: self._sync_student_input()
        )
        form.addRow(
            field_label(_MODULE, "student_input", "Student input:"), self._student_input
        )

        # --- the student ---
        self._base_model = QComboBox()
        self._base_model.setEditable(True)
        self._base_model.addItems(_POSE_BASES)
        form.addRow("Student (the fast pose model):", self._base_model)
        self._trainer_id = next(iter(sorted(TRAINERS)), None)
        trainer = TRAINERS[self._trainer_id]() if self._trainer_id else None
        self._train_compute = ComputeTargetRow(allow_remote=True)
        self._train_compute.set_backends(trainer.supported_backends if trainer else [])
        self._train_compute.add_to(form.section(), prefix="Train")
        self._hparams: dict[str, QWidget] = {}
        model_cls = getattr(trainer, "config_model", None)
        if model_cls is not None:
            knobs = form.section(_TRAINING_KNOBS)
            for name, (label, widget) in build_fields(model_cls).items():
                self._hparams[name] = widget
                knobs.addRow(f"{label}:", widget)
        form.finish()
        layout.addStretch()

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._start = box.button(QDialogButtonBox.StandardButton.Ok)
        self._start.setText("Start")
        box.accepted.connect(self._run)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

        self._sync_classes()
        self._sync_method()
        self._sync_detect_backends()
        self._detector.currentIndexChanged.connect(
            lambda _i: self._sync_detect_backends()
        )
        self._sync_student_input()
        self._sync()
        run_busy(
            self,
            experiment.keypoint_teacher_plan,
            self._on_plan,
            label="Reading the recordings…",
            on_error=self._on_plan_error,
        )

    # -- the plan ---------------------------------------------------------------- #
    def _on_plan(self, plan: list[dict]) -> None:
        names = self._experiment.recording_names()
        for row in plan:
            setup = row["acquisition_setup"]
            combo = QComboBox()
            for recording_id in row["recordings"]:
                combo.addItem(names.get(recording_id, recording_id), recording_id)
            combo.setCurrentIndex(max(0, combo.findData(row["recording_id"])))
            combo.currentIndexChanged.connect(lambda _i: self._sync_spans())
            span = QLabel("")
            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.addWidget(combo, 1)
            cell_layout.addWidget(span)
            self._teacher_rows.addRow(f"{setup}:", cell)
            self._teachers[setup] = combo
            self._spans[setup] = span
            self._durations.update(row.get("durations") or {})
        self._plan_read = True
        self._sync_spans()
        self._sync()

    def _on_plan_error(self, exc: BaseException) -> None:
        """A blank acquisition setup, or no video: said where the person will look."""
        self._plan_error.setText(str(exc).strip("'"))
        self._plan_error.show()
        self._plan_read = True
        self._sync()

    # -- syncing ----------------------------------------------------------------- #
    def _sync_classes(self) -> None:
        entry = self._detector.currentData()
        names = list(entry.get("classes", [])) if isinstance(entry, dict) else []
        labels = self._experiment.renamed_classes()
        self._class.clear()
        for name in names:
            self._class.addItem(class_names.label(name, labels), name)
        for tick in self._animal_ticks.values():
            tick.deleteLater()
        self._animal_ticks = {}
        for name in names:
            tick = QCheckBox(class_names.label(name, labels))
            tick.setChecked(True)
            tick.toggled.connect(lambda _on: self._sync_animals())
            self._animals_layout.addWidget(tick)
            self._animal_ticks[name] = tick
        several = len(names) >= 2
        if not several and self._several.isChecked():
            self._several.setChecked(False)
        set_available(
            self._several,
            several,
            "" if several else "The detector finds fewer than two classes",
        )
        self._sync_animals()

    def animals(self) -> list[str]:
        """The animal classes the setup is for: the one chosen, or each ticked."""
        if self._several.isChecked():
            return [n for n, tick in self._animal_ticks.items() if tick.isChecked()]
        return [self._class.currentData()] if self._class.currentData() else []

    def _sync_animals(self) -> None:
        several = self._several.isChecked()
        self._have_examples.setText(
            "I already have an example set for each animal (no pause)"
            if several
            else "I already have an example set for this animal (no pause)"
        )
        self._form.main.setRowVisible(self._class, not several)
        self._form.main.setRowVisible(self._animals_box, several)
        self._class_sets.set_classes(
            self.animals() if several else [],
            labels=self._experiment.renamed_classes(),
        )
        self._sync_name()
        self._sync()

    def _on_name_edited(self, text: str) -> None:
        self._name_edited = bool(text.strip())

    def _sync_name(self) -> None:
        animals = self.animals()
        if not self._name_edited and animals:
            self._name.setText(f"{' and '.join(animals)} keypoints")

    def _sync_spans(self) -> None:
        window_s = self._minutes.value() * 60
        for setup, combo in self._teachers.items():
            duration = self._durations.get(combo.currentData(), 0.0)
            self._spans[setup].setText(
                f"first {self._minutes.value()} of {duration / 60:.0f} min"
                if duration > window_s
                else f"all {duration / 60:.1f} min"
            )

    def _sync_method(self) -> None:
        model_id = self._method.currentData() or ""
        cls = KEYPOINT_MODELS.get(model_id)
        fill_variants(
            self._variant,
            model_id,
            cls.variants() if cls is not None else [],
            cls.default_variant() if cls is not None else None,
        )
        self._keypoints_compute.set_backends(
            list(getattr(cls, "supported_backends", [])) if cls else []
        )
        self._sync()

    def _sync_detect_backends(self) -> None:
        from detail.api.models import supported_backends as model_backends

        entry = self._detector.currentData()
        if not isinstance(entry, dict):
            self._detect_compute.set_backends([])
        elif entry["kind"] == "built-in":
            self._detect_compute.set_backends(model_backends(entry["id"]))
        else:
            # A trained or imported YOLO detector runs on any backend this machine has.
            self._detect_compute.set_backends(["cpu", "cuda", "mps"])

    def _sync_student_input(self) -> None:
        self._fields["student_crop_px"].setEnabled(
            self._student_input.currentData() == "crop"
        )

    def blocker(self) -> str | None:
        """Why Start is held, or None."""
        if not self._detector.isEnabled():
            return "Distill a detector first: keypoint setup finds the animal with it"
        if not patch_methods():
            return "No keypoint method that learns from an example set is available"
        if not self._plan_read:
            return "Reading the recordings…"
        if not self._teachers:
            return self._plan_error.text() or "No recording to teach from"
        several = self._several.isChecked()
        if several and len(self.animals()) < 2:
            return "Tick two animals or more, or untick Two or more animals"
        if self._have_examples.isChecked():
            if several:
                missing = [
                    name
                    for name in self.animals()
                    if name not in self._class_sets.value()
                ]
                if missing:
                    return (
                        f"Choose an example set for {', '.join(missing)}, or untick it "
                        "to mark them after tracking"
                    )
            elif self._examples.set_id() is None:
                return "Choose an example set, or untick it to mark one after tracking"
            if (
                method_default_cutoff(
                    self._experiment, self._method.currentData() or ""
                )
                is None
                and optional_value(self._score_cutoff) is None
            ):
                return (
                    f"Set a score cut-off: {self._method.currentData()} has no default"
                )
        return None

    def _sync(self) -> None:
        have = self._have_examples.isChecked()
        several = self._several.isChecked()
        self._form.main.setRowVisible(self._examples, have and not several)
        self._form.main.setRowVisible(self._class_sets, have and several)
        reason = self.blocker()
        set_available(self._start, reason is None, reason or "")

    # -- launch ------------------------------------------------------------------ #
    def _run(self) -> None:
        title = "Set up keypoints"
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(self, title, "Name is required.")
            return
        entry = self._detector.currentData()
        hyperparams = {
            key: (w.isChecked() if isinstance(w, QCheckBox) else w.value())
            for key, w in self._hparams.items()
        }
        several = self._several.isChecked()
        have = self._have_examples.isChecked()
        animals = (
            {"class_names": self.animals()}
            if several
            else {"class_name": self._class.currentData()}
        )
        example_set = self._examples.set_id() if have and not several else None
        class_example_sets = self._class_sets.value() if have and several else None
        try:
            self.run_handle = self._experiment.start_keypoint_setup(
                name,
                entry["id"],
                **animals,
                class_example_sets=class_example_sets or None,
                teachers={
                    setup: combo.currentData()
                    for setup, combo in self._teachers.items()
                },
                window_s=float(self._minutes.value() * 60),
                detect_compute=self._detect_compute.compute_target(),
                keypoints_model_id=self._method.currentData(),
                keypoints_model_variant=self._variant.currentData(),
                keypoints_compute=self._keypoints_compute.compute_target(),
                example_set=example_set,
                score_cutoff=optional_value(self._score_cutoff),
                n_frames=int(self._fields["n_frames"].value()),
                val_split=float(self._fields["val_split"].value()),
                exclude_audit_s=float(self._fields["exclude_audit_s"].value()),
                every_n_frames=int(self._fields["every_n_frames"].value()),
                student_input=self._student_input.currentData(),
                student_crop_px=int(self._fields["student_crop_px"].value()),
                base_model=self._base_model.currentText().strip(),
                trainer_id=self._trainer_id or "ultralytics_yolo",
                hyperparams=hyperparams or None,
                train_compute=self._train_compute.compute_target(),
            )
        except (ValueError, FileNotFoundError, KeyError, ModelNotReadyError) as exc:
            QMessageBox.warning(self, title, str(exc).strip("'"))
            return
        self.accept()
