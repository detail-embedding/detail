"""The keypoint half of **Distill new model** (§12 *Distill keypoints*): a teacher → a fast
YOLO-pose student, as one durable distill run (label → train → register).

**Teacher**, one of two: the **patch method** itself (``dinov3_match``, the default) with an
example set (the shared ``ExampleSetPicker``) and a tracking run (``UpstreamSpecPicker``: every
recording that has the chosen tracking is labelled), run on the sampled frames on its own
compute; or **finished keypoint detect runs**, one ticked per recording. Either teacher's poses
are unassigned (§26, revised 2026-10-21) and are given to a tracking run's subjects by §26
*Assignment*'s function: a detect-run teacher's tracking run is the one complete tracking run on
its detection run, which each row shows; a row with none or several is refused before launch.

A keypoint model is a model like any other, so it is made from the same dialog as a detector
(``gui/fit.py::DistillDialog``, *Model type: Keypoints*), which embeds :class:`KeypointDistillForm`.
The label stage's fields are read off ``KeypointDatasetConfig`` (§6.5, never typed here); the
training half mirrors ``TrainModelDialog`` (the trainer's knobs under *Training parameters*, a
``-pose`` base, its own compute). The teacher runs are a **selection** (one finished detect run per
recording), listed off the UI thread through ``run_busy``, because reading every recording's
record is O(recordings) disk I/O. ``Experiment.distill_keypoints`` does the launching.
"""

from __future__ import annotations

from collections import Counter

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.api.keypoint_distill import DEFAULT_POSE_BASE
from detail.configs.distill import DEFAULT_N_FRAMES
from detail.configs.keypoint_dataset import KeypointDatasetConfig
from detail.core.registries import KEYPOINT_MODELS, TRAINERS
from detail.executor.runners import import_runner_modules
from detail.gui import sizing
from detail.gui._pydantic_form import build_fields, optional_spin, optional_value
from detail.gui.busy import run_busy
from detail.gui.compute_target import ComputeTargetRow
from detail.gui.example_set_picker import ExampleSetPicker
from detail.gui.modules._common import UpstreamSpecPicker, spec_runs
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label
from detail.gui.model_availability import (
    fill_variants,
    unavailable_reason,
    variant_available,
)
from detail.gui.sizing import set_available
from detail.gui.status_color import ERROR, READY
from detail.storage import read_cache
from detail.storage import recording as recording_store

_MODULE = "keypoint_dataset"
_TRAINING_KNOBS = "Training parameters"
#: The pose checkpoints offered for the student, the default first (editable for another).
_POSE_BASES = [
    DEFAULT_POSE_BASE,
    "yolo11n-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
]
#: The two teachers, the patch method first: it needs no keypoint runs to exist.
TEACHERS = (
    ("model", "Patch method + example set"),
    ("keypoint_detect", "Finished keypoint detect run"),
)


def patch_methods() -> list[str]:
    """The keypoint methods that learn from an example set: what can teach directly."""
    from detail import surfaces

    return sorted(
        m
        for m, cls in KEYPOINT_MODELS.items()
        if getattr(cls, "needs_examples", False) and surfaces.is_surfaced(m)
    )


def _read_teachers(experiment, tracking: UpstreamSpecPicker) -> tuple:
    """Off the UI thread: the finished detect runs, the tracking-run groups over every
    recording (what the patch teacher can be given), and each group's own run per recording
    (a run id is a per-recording counter, §6.5), so launching reads nothing."""
    recordings = list(experiment.recordings)  # off-thread
    groups = tracking.resolve(recordings)
    runs = spec_runs(recordings, "tracking", groups)
    return finished_detect_runs(experiment), groups, len(recordings), runs


def finished_detect_runs(experiment) -> list[dict]:
    """Every complete ``keypoint_detect`` run of ``experiment``, recording order: one row per
    run as ``{recording_id, recording_name, run_id, method, teacher}``, ``teacher`` being
    ``api.keypoint_distill.teacher_info`` (the detect method, its score, its example set, its
    detection run and the tracking runs on it its poses can be given to).
    Reads each recording's record once, pinned, and two small records per run, which is why
    it runs off the UI thread."""
    from detail.api.keypoint_distill import teacher_info

    names = experiment.recording_names()
    rows: list[dict] = []
    with read_cache.pinned():
        for recording in experiment.recordings:  # off-thread
            data = recording_store.read_recording(recording.dir)
            for run in data.get("keypoint_detect_runs", []):
                if str(run.get("status", "complete")) != "complete":
                    continue
                rows.append(
                    {
                        "recording_id": recording.id,
                        "recording_name": names.get(recording.id, recording.id),
                        "run_id": str(run["id"]),
                        "method": str(run.get("method") or run.get("model_id") or ""),
                        "teacher": teacher_info(
                            experiment, recording.id, str(run["id"])
                        ),
                    }
                )
    return rows


def _subjects_text(teacher: dict) -> str:
    """Where a detect-run teacher's poses go, for its row: the tracking run on its
    detection run, or why there is not exactly one."""
    on = list(teacher.get("tracking_run_ids") or [])
    detection = teacher.get("detection_run_id") or "?"
    if len(on) == 1:
        return f"subjects from {on[0]}"
    if not on:
        return f"no tracking run on {detection}"
    return f"{len(on)} tracking runs on {detection}"


def _tracking_blocker(teacher: dict) -> str | None:
    """Why a ticked detect-run teacher cannot be given to subjects, or None: its poses go
    to the one complete tracking run on its detection run (§26 *Assignment*)."""
    on = list(teacher.get("tracking_run_ids") or [])
    detection = teacher.get("detection_run_id") or "?"
    if not on:
        return (
            f"A ticked run's detection run {detection} has no finished tracking run: its "
            "poses are given to tracked animals, so run tracking on it first"
        )
    if len(on) > 1:
        return (
            f"A ticked run's detection run {detection} has {len(on)} tracking runs "
            f"({', '.join(on)}); its poses can be given to only one. Name one with "
            "`detail keypoints distill --tracking-run`"
        )
    return None


class KeypointDistillForm(QWidget):
    """The keypoint fields of the distill dialog: the teacher runs, the dataset's size and the
    student's base. :meth:`blocker` says why it cannot launch yet (``readiness_changed`` fires
    when that may have changed); :meth:`launch` starts ``Experiment.distill_keypoints``.
    """

    readiness_changed = pyqtSignal()

    def __init__(self, experiment=None, parent=None):
        super().__init__(parent)
        self._experiment = None
        self._rows: list[dict] = []
        self._reading = False
        self._blocker: str | None = "Reading the teachers…"
        import_runner_modules(PIPELINE_RUNNERS)  # populate the TRAINERS registry

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = ConfigForm(layout, KeypointDatasetConfig)
        self._form = form

        # --- which teacher ---
        self._kind = QComboBox()
        for value, label in TEACHERS:
            self._kind.addItem(label, value)
        self._kind.setToolTip(
            "Patch method: the method runs on the sampled frames itself, from an example set "
            "and a tracking run. Finished detect run: its stored placements are the labels."
        )
        self._kind.currentIndexChanged.connect(lambda _i: self._sync_kind())
        form.addRow("Teacher:", self._kind)

        # --- the patch-method teacher ---
        self._method = QComboBox()
        for model_id in patch_methods():
            self._method.addItem(model_id, model_id)
        self._method.currentIndexChanged.connect(lambda _i: self._sync_method())
        form.addRow("Method:", self._method)
        self._variant = QComboBox()
        self._variant.currentIndexChanged.connect(lambda _i: self._sync())
        form.addRow("Size:", self._variant)
        self._examples = ExampleSetPicker()
        self._examples.changed.connect(self._sync)
        form.addRow("Examples:", self._examples)
        self._tracking = UpstreamSpecPicker("tracking")
        self._tracking.combo.setToolTip(
            "The tracking whose boxes and masks the method is given. Every recording with "
            "this tracking is labelled."
        )
        self._tracking.combo.currentIndexChanged.connect(lambda _i: self._sync())
        self._tracking.combo.setPlaceholderText(
            "(no tracking run in this experiment yet)"
        )
        self._recording_total = 0
        self._tracking_runs: dict = {}  # frozen spec -> {recording_id: tracking run id}
        form.addRow("Tracking:", self._tracking.combo)
        self._patch_rows = (
            self._method,
            self._variant,
            self._examples,
            self._tracking.combo,
        )

        # --- the teacher: one finished keypoint detect run per recording ---
        # Sized to its rows (up to six, then it scrolls); the form's slack goes below
        # everything, not into a mostly empty list.
        self._runs = QListWidget()
        self._runs.itemChanged.connect(lambda _item: self._sync())
        self._runs.setToolTip(
            "The finished keypoint detect runs of this experiment. Tick one per recording: "
            "its stored placements, unsmoothed (and the example set's own clicks), are the "
            "labels."
        )
        form.addRow("Teacher runs:", self._runs)
        self._runs_note = QLabel("Reading keypoint detect runs…")
        self._runs_note.setWordWrap(True)
        form.addRow("", self._runs_note)

        # --- the label stage, off the config model (advanced rows route themselves) ---
        self._fields: dict[str, QWidget] = {}
        for name, (label, widget) in build_fields(KeypointDatasetConfig).items():
            self._fields[name] = widget
            form.addRow(field_label(_MODULE, name, label + ":"), widget)
        self._fields["n_frames"].setValue(DEFAULT_N_FRAMES)
        self._score_cutoff = optional_spin(
            KeypointDatasetConfig, "score_cutoff", "the teacher's default"
        )
        self._score_cutoff.valueChanged.connect(lambda _v: self._sync())
        self._test = QPushButton("Test cut-off…")
        self._test.setToolTip(
            "See the teacher's points on a sample of frames drawn the way the distill draws "
            "them, slide the cut-off, and check it against your example clicks in pixels."
        )
        self._test.clicked.connect(self._test_cutoff)
        cutoff_row = QWidget()
        cutoff_layout = QHBoxLayout(cutoff_row)
        cutoff_layout.setContentsMargins(0, 0, 0, 0)
        cutoff_layout.addWidget(self._score_cutoff, 1)
        cutoff_layout.addWidget(self._test)
        form.addRow(field_label(_MODULE, "score_cutoff", "Score cut-off:"), cutoff_row)
        self._cutoff_note = QLabel("")
        self._cutoff_note.setWordWrap(True)
        form.addRow("", self._cutoff_note)

        # --- the training stage (mirrors TrainModelDialog) ---
        self._trainer = QComboBox()
        for trainer_id in sorted(TRAINERS):
            self._trainer.addItem(trainer_id, trainer_id)
        if not TRAINERS:
            self._trainer.addItem("(no trainer registered)", None)
        self._trainer.currentIndexChanged.connect(self._sync_trainer)
        self._trainer.hide()  # one training framework today; kept for the config, not shown
        self._readiness = QLabel("")
        form.addRow("", self._readiness)
        self._base_model = QComboBox()
        self._base_model.setEditable(True)
        self._base_model.addItems(_POSE_BASES)
        self._base_model.setToolTip(
            "A -pose checkpoint the trainer fine-tunes; the student learns the skeleton's "
            "points from it."
        )
        form.addRow("Student (the fast pose model):", self._base_model)
        self._label_compute = ComputeTargetRow()  # where the patch method runs
        self._label_compute.add_to(form.section(), prefix="Label")
        self._train_compute = ComputeTargetRow()
        self._train_compute.add_to(form.section(), prefix="Train")
        self._hparams: dict[str, QWidget] = {}
        form.finish()
        layout.addStretch()
        self._sync_trainer()
        self._sync_method()
        self._fit_list()
        if not patch_methods():
            self._kind.setCurrentIndex(self._kind.findData("keypoint_detect"))
        self._sync_kind()
        self.set_experiment(experiment)

    def blocker(self) -> str | None:
        """Why the keypoint distill cannot launch yet, or None."""
        return self._blocker

    def teacher_kind(self) -> str:
        return str(self._kind.currentData())

    def set_experiment(self, experiment) -> None:
        """Read ``experiment``'s teachers off the UI thread: its finished keypoint detect runs,
        and the tracking runs a patch teacher can be given."""
        self._experiment = experiment
        self._examples.set_experiment(experiment)
        self._runs.clear()
        self._rows = []
        self._fit_list()
        self._cutoff_note.setText("")
        if experiment is None:
            self._runs_note.setText("Open an experiment to distill its keypoint runs.")
            self._set_blocker("Open an experiment first")
            return
        self._runs_note.setText("Reading keypoint detect runs…")
        self._reading = True
        self._sync()
        tracking = self._tracking
        run_busy(
            self,
            lambda: _read_teachers(experiment, tracking),
            self._show_teachers,
            label="Reading the teachers…",
            # The experiment combo is in the fit dialog, outside this overlay: a second
            # experiment can be picked while the first is read, and the older read lands last.
            supersedes="teachers",
        )

    def _show_teachers(self, result) -> None:
        rows, groups, total, runs = result
        self._reading = False
        self._recording_total = total
        self._tracking_runs = runs
        self._tracking.apply(groups, total)
        self._show_runs(rows)

    def _set_blocker(self, reason: str | None) -> None:
        self._blocker = reason
        self.readiness_changed.emit()

    # -- the teacher list -------------------------------------------------------- #
    def _show_runs(self, rows: list[dict]) -> None:
        self._rows = list(rows)
        self._runs.clear()
        per_recording = Counter(r["recording_id"] for r in rows)
        for row in rows:
            text = f"{row['recording_name']} · {row['run_id']}"
            if row["method"]:
                text += f" · {row['method']}"
            examples = (row.get("teacher") or {}).get("example_set")
            if examples:
                text += f' · examples: {examples["id"]} "{examples["name"]}"'
            text += " · " + _subjects_text(row.get("teacher") or {})
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # A recording with one finished run is pre-ticked; one with several asks.
            item.setCheckState(
                Qt.CheckState.Checked
                if per_recording[row["recording_id"]] == 1
                else Qt.CheckState.Unchecked
            )
            item.setData(Qt.ItemDataRole.UserRole, (row["recording_id"], row["run_id"]))
            self._runs.addItem(item)
        if not rows:
            self._runs_note.setText(
                "No finished keypoint detect run in this experiment. Run keypoint detect on "
                "a recording first, or use the patch method directly."
            )
        else:
            self._runs_note.setText(
                f"{len(rows)} finished run(s) across {len(per_recording)} recording(s). "
                "Where a recording has several, tick the one that teaches."
            )
        self._fit_list()
        self._sync()

    def _fit_list(self) -> None:
        rows = max(1, min(self._runs.count(), 6))
        row = self._runs.sizeHintForRow(0) if self._runs.count() else sizing.lines(1)
        self._runs.setFixedHeight(rows * max(row, 1) + 2 * self._runs.frameWidth() + 2)

    def _sync_kind(self) -> None:
        by_model = self.teacher_kind() == "model"
        main = self._form.main
        for widget in self._patch_rows:
            main.setRowVisible(widget, by_model)
        for widget in (self._runs, self._runs_note):
            main.setRowVisible(widget, not by_model)
        self._sync()

    def _sync_method(self) -> None:
        cls = KEYPOINT_MODELS.get(self._method.currentData() or "")
        fill_variants(
            self._variant,
            self._method.currentData() or "",
            cls.variants() if cls is not None else [],
            cls.default_variant() if cls is not None else None,
        )
        self._label_compute.set_backends(cls.supported_backends if cls else [])
        self._sync()

    def _method_teacher(self) -> list[dict]:
        cls = KEYPOINT_MODELS.get(self._method.currentData() or "")
        if cls is None:
            return []
        spec = cls.score_spec
        return [
            {
                "method": self._method.currentData(),
                "default_cutoff": spec.default_cutoff,
                "meaning": spec.meaning,
                "low": spec.low,
                "high": spec.high,
            }
        ]

    def _ticked_teachers(self) -> list[dict]:
        out = []
        for i in range(self._runs.count()):
            item = self._runs.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                key = item.data(Qt.ItemDataRole.UserRole)
                out.extend(
                    r.get("teacher") or {}
                    for r in self._rows
                    if (r["recording_id"], r["run_id"]) == tuple(key)
                )
        return out

    def _patch_blocker(self) -> str | None:
        if not patch_methods():
            return "No method that learns from an example set is available"
        # Refused here, before Test cut-off… or Distill, rather than inside the run: the
        # same check `resolve_method` makes, read off the greyed size row and the model.
        model_id = self._method.currentData() or ""
        reason = unavailable_reason(model_id)
        if reason is not None:
            return f"{model_id} {reason}: install it in the Model manager"
        if not variant_available(self._variant):
            size = self._variant.currentData()
            return (
                f"{model_id} {size} is not installed: choose an installed size, or "
                "install it in the Model manager"
            )
        if self._examples.set_id() is None:
            return "Choose an example set, or make one with New example set…"
        covered = self._tracking.covered()
        if not covered:
            return (
                "Needs a tracking run: run detection and tracking on a recording first"
            )
        return None

    def _teacher_state(self) -> tuple[str | None, list[dict]]:
        """Why the chosen teacher is not complete yet (or None), and its methods' facts."""
        if self._experiment is None:
            return "Open an experiment first", []
        if self._reading:
            return "Reading the teachers…", []
        if self.teacher_kind() == "model":
            reason = self._patch_blocker()
            if reason is not None:
                return reason, []
            return None, self._method_teacher()
        if not self._rows:
            return "Needs a finished keypoint detect run", []
        teachers = self._ticked_teachers()
        if not teachers:
            return "Tick at least one keypoint detect run", []
        for teacher in teachers:
            reason = _tracking_blocker(teacher)
            if reason is not None:
                return reason, []
        return None, teachers

    def _sync(self) -> None:
        """Hold Distill, with the reason, until the chosen teacher is complete; then say what
        the cut-off means for it, and hold Distill until one is set when the teacher's method
        has no default (Tenet 9: never guess it). *Test cut-off…* follows the teacher alone:
        it is how a cut-off gets chosen."""
        if self._experiment is None:
            return
        reason, teachers = self._teacher_state()
        set_available(self._test, reason is None, reason or "")
        if reason is not None:
            self._score_cutoff.setSpecialValueText("the teacher's default")
            if not self._reading:
                self._cutoff_note.setText("")
            self._set_blocker(reason)
            return
        first = teachers[0]
        scale = (
            f"{first['meaning']}, {first['low']:g} to {first['high']:g}"
            if first.get("meaning")
            else "the teacher's score"
        )
        undefaulted = sorted(
            {t["method"] for t in teachers if t.get("default_cutoff") is None}
        )
        chosen = optional_value(self._score_cutoff)
        self._score_cutoff.setSpecialValueText(
            "required: set a value"
            if undefaulted
            else f"the teacher's default ({first['default_cutoff']:g})"
        )
        note = f"{scale}. A point scored below the cut-off is taught as not visible."
        if undefaulted and chosen is None:
            methods = ", ".join(undefaulted)
            self._cutoff_note.setText(f"{note} {methods} has no default, so set one.")
            self._set_blocker(f"Set a score cut-off: {methods} has no default")
            return
        if chosen is None:
            note += f" Left as the teacher's default: {first['default_cutoff']:g}."
        self._cutoff_note.setText(note)
        self._set_blocker(None)

    def selected_tracking_runs(self) -> dict[str, str]:
        """``{recording_id: tracking run id}`` for every recording the chosen tracking
        covers, each resolved to its own run (a run id is a per-recording counter, §6.5).
        """
        from detail.api import run_spec

        spec = self._tracking.spec()
        if spec is None:
            return {}
        return dict(self._tracking_runs.get(run_spec.freeze_spec(spec), {}))

    def selected_detect_runs(self) -> dict[str, str]:
        """``{recording_id: run_id}`` for the ticked rows.

        Raises:
            ValueError: Two runs ticked for one recording.
        """
        out: dict[str, str] = {}
        for i in range(self._runs.count()):
            item = self._runs.item(i)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            recording_id, run_id = item.data(Qt.ItemDataRole.UserRole)
            if recording_id in out:
                name = next(
                    r["recording_name"]
                    for r in self._rows
                    if r["recording_id"] == recording_id
                )
                raise ValueError(
                    f"{name} has two runs ticked ({out[recording_id]}, {run_id}); a "
                    "recording teaches from one"
                )
            out[recording_id] = run_id
        return out

    # -- the trainer drives its knobs ------------------------------------------- #
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
        palette = self._readiness.palette()
        palette.setColor(QPalette.ColorRole.WindowText, READY if ready else ERROR)
        self._readiness.setPalette(palette)
        self._train_compute.set_backends(trainer.supported_backends if trainer else [])
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

    # -- launch ------------------------------------------------------------------ #
    def teacher_keywords(self) -> dict:
        """The chosen teacher as ``Experiment.distill_keypoints`` / ``test_keypoints`` take it.

        Raises:
            ValueError: The teacher is not complete, or two runs ticked for one recording.
        """
        if self.teacher_kind() == "model":
            reason = self._patch_blocker()
            if reason is not None:
                raise ValueError(reason + ".")
            return {
                "example_set": self._examples.set_id(),
                "tracking": self.selected_tracking_runs(),
                "model_id": self._method.currentData(),
                "model_variant": self._variant.currentData(),
                "label_compute": self._label_compute.compute_target(),
            }
        detect_runs = self.selected_detect_runs()
        if not detect_runs:
            raise ValueError("Tick at least one keypoint detect run.")
        return {"detect_runs": detect_runs}

    def test_teacher(self) -> dict:
        """What *Test cut-off…* hands ``test_keypoints``: the teacher, drawn the way this
        distill would draw (its frame stride and audit clearance)."""
        return {
            **self.teacher_keywords(),
            "every_n_frames": int(self._fields["every_n_frames"].value()),
            "exclude_audit_s": float(self._fields["exclude_audit_s"].value()),
        }

    def _test_cutoff(self) -> None:
        """Open *Test cut-off…* on the chosen teacher; *Use this cut-off* writes the value
        back to the field."""
        from detail.gui.keypoint_test import KeypointTestDialog

        try:
            teacher = self.test_teacher()
        except ValueError as exc:
            QMessageBox.warning(self, "Test cut-off", str(exc))
            return
        _reason, teachers = self._teacher_state()
        cutoff = optional_value(self._score_cutoff)
        if cutoff is None and teachers:
            cutoff = teachers[0].get("default_cutoff")
        dialog = KeypointTestDialog(
            self._experiment, teacher, cutoff=cutoff, parent=self
        )
        if dialog.exec() and dialog.chosen_cutoff() is not None:
            self._score_cutoff.setValue(dialog.chosen_cutoff())

    def launch(self, name: str):
        """Start the keypoint distill; the ``AttachedDistillRun``.

        Raises:
            ValueError: No name, no ticked run, two runs ticked for one recording, or what
                ``Experiment.distill_keypoints`` refuses.
        """
        if not name:
            raise ValueError("Give the model a name.")
        teacher = self.teacher_keywords()
        hyperparams = {
            key: (w.isChecked() if isinstance(w, QCheckBox) else w.value())
            for key, w in self._hparams.items()
        }
        try:
            return self._experiment.distill_keypoints(
                name,
                n_frames=int(self._fields["n_frames"].value()),
                **teacher,
                val_split=float(self._fields["val_split"].value()),
                score_cutoff=optional_value(self._score_cutoff),
                exclude_audit_s=float(self._fields["exclude_audit_s"].value()),
                every_n_frames=int(self._fields["every_n_frames"].value()),
                base_model=self._base_model.currentText().strip(),
                trainer_id=self._trainer.currentData(),
                hyperparams=hyperparams,
                train_compute=self._train_compute.compute_target(),
            )
        except (FileNotFoundError, KeyError) as exc:
            raise ValueError(str(exc)) from exc
