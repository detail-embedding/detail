"""The ``keypoint_detect`` launch form and its run viewer registration (§26 "Surfaces").

Copies the tracking form's shape: the **detection run** through an ``UpstreamSpecPicker`` (one
option per set of settings across the selection; §26 revised 2026-10-21 — a detect run places
poses on every animal and keeps them unassigned, so it asks for no tracking run and no tracks),
the method and its size from the ``KEYPOINT_MODELS`` registry plus the experiment's trained
students and the model catalog's keypoint models, and every knob read off ``KeypointDetectConfig`` (§6.5) — the two ``Literal | None``
knobs render through ``any_field_widget`` with an *auto* row, which is what "let the method
resolve it" means. ``min_pose_confidence`` is shown only for a method that
``reports_confidence`` (the student), the one kind ``validate`` accepts it for; ``class_name``
only for a method that works per detection, the one kind ``validate`` accepts it for. The class
row offers the classes the chosen detection run actually detected, read off its parts on a
background thread whenever the detection run changes — tracking's class row, the same read.

The detection run and the example set are **selections** (they name artifacts this experiment
mints), so *Start from* never restores them; the method and its settings are.

**Example set per class** (§26 *One example set per detection class*): for animals told apart
at detection that look different, a tick replaces *Class* and *Examples* with one row per class
the detection run detected — a tick for each class to place poses in, and that class's own
example-set picker. It is offered for a per-detection method that learns from examples, and
enabled once the detection run has two classes or more. *Test keypoints* tests the same map.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QComboBox, QPushButton, QWidget

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.keypoint_detect import (
    DEFAULT_CROP_PX,
    DEFAULT_MIN_POSE_CONFIDENCE,
    KeypointDetectConfig,
)
from detail.core.keypoints import student_input_of
from detail.core.registries import KEYPOINT_MODELS
from detail.executor.runners import import_runner_modules
from detail.gui._pydantic_form import (
    any_field_value,
    any_field_widget,
    field_meta,
    field_widget,
    optional_spin,
    optional_value,
    set_optional_value,
)
from detail.gui.compute_target import ComputeTargetRow
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label, help_button, label_with
from detail.gui.model_availability import (
    apply_availability,
    availability_note,
    fill_variants,
    selected_model_id,
    show_note,
    variant_available,
)
from detail.gui.class_example_sets import ClassExampleSets
from detail.gui.example_set_picker import ExampleSetPicker
from detail.gui.modules._common import (
    UpstreamSpecPicker,
    arena_covered_ids,
    arena_items,
    detected_class_groups,
    repopulate_combo,
    spec_runs,
)
from detail.gui.modules._every_n import EveryNRow, source_rate
from detail.gui.sizing import set_available
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    register_module_ui,
    surfaced,
)
from detail.storage import read_cache

_MODULE = "keypoint_detect"
_NO_CLASS = "(every class)"
#: How the method picker says where a pose model comes from (``keypoint_students``' source).
_SOURCE_LABELS = {
    "trained": "trained in this experiment",
    "imported": "from the model catalog",
}


class _KeypointDetectConfig(ConfigWidget):
    config_model = KeypointDetectConfig

    def __init__(self):
        super().__init__()
        import_runner_modules(PIPELINE_RUNNERS)  # populate KEYPOINT_MODELS
        form = ConfigForm(self, KeypointDetectConfig)
        # The Test keypoints button and what its state reads exist before any row whose
        # selection changes it: marking the methods' availability below can move the method,
        # which re-syncs the button.
        self._test = QPushButton("Test keypoints")
        self._experiment = None
        self._recordings: list = []
        self._detection_runs: dict = (
            {}
        )  # frozen spec -> {recording_id: detection run id}
        # (recording_id, detection run id) -> the complete tracking runs on it: what Test
        # keypoints places points on (the patch teacher is handed tracked frames, §12).
        self._tracking_on: dict = {}
        self._unavailable = None
        self._picker = UpstreamSpecPicker("detection")
        self._detection = self._picker.combo
        self._detection.setToolTip(
            field_meta(KeypointDetectConfig, "detection_run_id")["help"]
        )
        self._detection.currentIndexChanged.connect(
            lambda _index: self.coverage_changed.emit()
        )
        self._detection.currentIndexChanged.connect(self._sync_upstream_every_n)
        form.addRow(
            field_label(_MODULE, "detection_run_id", "Detection run:"),
            self._detection,
        )
        # Built before the method row: choosing a method shows or hides it.
        self._class = QComboBox()
        self._class.setToolTip(field_meta(KeypointDetectConfig, "class_name")["help"])
        self._class.addItem(_NO_CLASS, None)
        self._class_label = field_label(_MODULE, "class_name", "Class:")
        form.addRow(self._class_label, self._class)
        self._class_groups: list[dict] = []
        self._class_resolve_gen = 0
        self._class_call = None
        # A class a restored config named before the detection run's classes were read.
        self._pending_class: str | None = None
        self._class.currentIndexChanged.connect(
            lambda _index: self.coverage_changed.emit()
        )
        self._detection.currentIndexChanged.connect(self._refresh_class_groups)
        # {student id: works per detection}: a crop-input student does, a whole-frame one
        # does not, and its class (`TrainedYoloPose`) cannot say which.
        self._student_per_detection: dict[str, bool] = {}
        self._method = QComboBox()
        self._method.setToolTip(field_meta(KeypointDetectConfig, "model_id")["help"])
        for model_id in surfaced(KEYPOINT_MODELS):
            self._method.addItem(model_id, model_id)
        # Imported ONNX models are machine-global (§26): offered in every experiment, after
        # the registered methods and before an experiment's own trained students.
        from detail.models.resolve import imported_keypoint_models

        self._imported_ids = set(imported_keypoint_models())
        for model_id in sorted(self._imported_ids):
            self._method.addItem(f"{model_id} · from the catalog", model_id)
        self._method.currentIndexChanged.connect(self._sync_method)
        form.addRow(field_label(_MODULE, "model_id", "Method:"), self._method)
        self._no_methods = availability_note()
        form.addRow("", self._no_methods)
        self._variant = QComboBox()
        self._variant.setToolTip(
            field_meta(KeypointDetectConfig, "model_variant")["help"]
        )
        form.addRow(
            field_label(_MODULE, "model_variant", "Size / variant:"), self._variant
        )
        # PR 2's example-set picker (`gui/example_set_picker.py`): skeleton, set, and its
        # New / Mark frames / Delete buttons, shown only when the method learns from examples
        # (§26 *Example sets follow the model that needs them*).
        self._example_picker = ExampleSetPicker()
        self._example_picker.setToolTip(
            field_meta(KeypointDetectConfig, "example_set_id")["help"]
        )
        self._example_picker.changed.connect(lambda: self.coverage_changed.emit())
        self._examples_label = field_label(_MODULE, "example_set_id", "Examples:")
        form.addRow(self._examples_label, self._example_picker)
        # One set per detection class, in place of Class and Examples (§26 *One example set
        # per detection class*); enabled once the detection run has two classes.
        self._per_class = QCheckBox("One example set per class")
        self._per_class.setToolTip(
            field_meta(KeypointDetectConfig, "class_example_sets")["help"]
        )
        self._per_class_label = field_label(
            _MODULE, "class_example_sets", "Example set per class:"
        )
        form.addRow(self._per_class_label, self._per_class)
        self._class_sets = ClassExampleSets()
        form.addRow("", self._class_sets)
        self._per_class.toggled.connect(self._sync_method)
        self._per_class.toggled.connect(lambda _on: self.coverage_changed.emit())
        self._class_sets.changed.connect(lambda: self.coverage_changed.emit())
        self._class_sets.changed.connect(self._sync_test)
        set_available(
            self._per_class, False, "The detection run has fewer than two classes"
        )
        self._student_ids: set[str] = set()  # pose models offered in `_method`
        # Built before availability marking moves the method, which re-syncs its visibility.
        self._min_conf = optional_spin(
            KeypointDetectConfig,
            "min_pose_confidence",
            f"(method default, {DEFAULT_MIN_POSE_CONFIDENCE})",
            step=0.05,
        )
        self._min_conf_label = field_label(
            _MODULE, "min_pose_confidence", "Minimum pose confidence:"
        )
        form.addRow(self._min_conf_label, self._min_conf)
        # After the rows its selection drives exist: marking may move the selection to a
        # runnable method, which fires _sync_method.
        self._unavailable = apply_availability(self._method, "keypoint")
        if self._unavailable:
            # Say what each method needs: dinov3_match has no Model manager row of its own,
            # so the generic "install models" sentence names nothing anyone can find.
            from detail.gui.model_availability import unavailable_reason

            reasons = [
                f"{model_id} {unavailable_reason(model_id)}"
                for model_id in surfaced(KEYPOINT_MODELS)
                if unavailable_reason(model_id)
            ]
            if reasons:
                self._unavailable = (
                    "No keypoint method can run here: "
                    + "; ".join(reasons)
                    + (
                        ". Install it on the Model manager."
                        if len(reasons) == 1
                        else ". Install one on the Model manager."
                    )
                )
        # What the registered methods alone say; a pose model offered later can still run
        # where they cannot (`_refresh_students`).
        self._methods_unavailable = self._unavailable
        show_note(self._no_methods, self._unavailable)
        self._decode = any_field_widget(KeypointDetectConfig, "decode")
        form.addRow(
            field_label(_MODULE, "decode", "Decode:"), self._decode, advanced=True
        )
        self._side_rule = any_field_widget(KeypointDetectConfig, "side_rule")
        form.addRow(
            field_label(_MODULE, "side_rule", "Left/right rule:"),
            self._side_rule,
            advanced=True,
        )
        self._crop_px = optional_spin(
            KeypointDetectConfig, "crop_px", f"(method default, {DEFAULT_CROP_PX})"
        )
        form.addRow(
            field_label(_MODULE, "crop_px", "Crop size (px):"),
            self._crop_px,
            advanced=True,
        )
        every_n_meta = field_meta(KeypointDetectConfig, "every_n_frames")
        self._every_n = EveryNRow(KeypointDetectConfig, upstream_noun="detected")
        form.addRow(
            field_label(_MODULE, "every_n_frames", every_n_meta["label"] + ":"),
            self._every_n,
        )
        self._batch = field_widget(KeypointDetectConfig, "batch_size")
        form.addRow(field_label(_MODULE, "batch_size", "Batch size:"), self._batch)
        self._compute = ComputeTargetRow(allow_remote=True)
        self._compute.add_to(form.section())
        # The keypoints counterpart of detection's Test prompts, in the same place: the last
        # row, opening `KeypointDetectTestDialog` on this form's method, size, example set,
        # settings and the tracked frames of the chosen detection run. It has no "Use this cut-off": a detect run stores raw
        # points, and the cut-off is chosen where they are viewed or exported.
        # Which tracking Test keypoints draws its frames from, asked only when a selected
        # recording has several on the chosen detection run — the social flow always does
        # (each class's track and their merge), and picking one would be a guess (Tenet 9).
        self._test_tracking = UpstreamSpecPicker(
            "tracking", auto_label="(choose the tracking to test on)"
        )
        self._test_tracking.combo.setToolTip(
            "The tracking whose frames Test keypoints draws from, for recordings with "
            "several on this detection run: the merged run tests every animal at once. "
            "The detect run itself uses no tracking."
        )
        self._test_tracking.combo.currentIndexChanged.connect(self._sync_test)
        self._test_tracking_groups: list[dict] = []
        self._test_tracking_runs: dict = {}  # frozen spec -> {recording_id: run id}
        self._test_tracking_label = label_with(
            "Test on tracking:", help_button("test-on-tracking")
        )
        form.addRow(self._test_tracking_label, self._test_tracking.combo)
        self._test_tracking_label.setVisible(False)
        self._test_tracking.combo.setVisible(False)
        self._test.clicked.connect(self._open_test)
        form.addRow("", self._test)
        # the knobs the base load_config restores by field name (scope: setting)
        self._fields = {
            "decode": self._decode,
            "side_rule": self._side_rule,
            "crop_px": self._crop_px,
            "batch_size": self._batch,
        }
        self._rates: tuple[float, float] | None = None
        self._detection.currentIndexChanged.connect(self._sync_test_tracking)
        self._detection.currentIndexChanged.connect(self._sync_test)
        self._variant.currentIndexChanged.connect(self._sync_test)
        self._example_picker.changed.connect(self._sync_test)
        self._sync_method()
        form.finish()

    # -- the method drives the variant row and whether an example set is asked for -- #
    def _method_class(self):
        """The chosen method's class through the one lookup (a registered method, or
        ``TrainedYoloPose`` for a trained student), for its class-level facts."""
        from detail.models.resolve import keypoint_method_class

        experiment_dir = None if self._experiment is None else self._experiment.dir
        return keypoint_method_class(experiment_dir, selected_model_id(self._method))

    def _sync_method(self, *_args) -> None:
        model_id = selected_model_id(self._method)
        cls = self._method_class()
        fill_variants(
            self._variant,
            model_id,
            cls.variants() if cls else [],
            cls.default_variant() if cls else None,
        )
        needs = bool(getattr(cls, "needs_examples", False))
        per_detection = self._per_detection(model_id, cls)
        # A set per class is for a per-detection method that learns from examples; ticked,
        # it stands in for both Class and Examples.
        offer = needs and per_detection
        self._per_class.setVisible(offer)
        self._per_class_label.setVisible(offer)
        per_class = offer and self._per_class.isChecked()
        self._class_sets.setVisible(per_class)
        # Hidden rather than greyed: an example set is an input of this method, not a
        # setting every method has (§26).
        self._example_picker.setVisible(needs and not per_class)
        self._examples_label.setVisible(needs and not per_class)
        # Hidden for the same reason: only a method that reports a pose confidence takes the
        # threshold, and `validate` refuses it for any other (§26 *Visibility*).
        reports = bool(getattr(cls, "reports_confidence", False))
        self._min_conf.setVisible(reports)
        self._min_conf_label.setVisible(reports)
        # Hidden for a whole-frame method: it is handed no detections, so there is no class
        # to choose among, and `validate` refuses the setting there.
        self._class.setVisible(per_detection and not per_class)
        self._class_label.setVisible(per_detection and not per_class)
        self._sync_test()

    def _per_class_active(self) -> bool:
        return self._class_sets.isVisibleTo(self)

    def _per_detection(self, model_id: str | None, cls) -> bool:
        """Whether the chosen method works per detection: a student by how it was trained,
        anything else by its class."""
        if model_id in self._student_per_detection:
            return self._student_per_detection[model_id]
        return bool(getattr(cls, "per_detection", True))

    # -- the class row: the chosen detection run's own classes ------------------ #
    def _refresh_class_groups(self, *_args) -> None:
        """Off the UI thread: which classes the chosen detection run actually detected on
        each selected recording (``detected_class_groups``, tracking's read). Re-read on
        every detection-run change, generation-guarded so an earlier read finishing late
        cannot overwrite a newer one."""
        from detail.gui.workers import BackgroundCall

        spec = self._picker.spec()
        self._class_resolve_gen += 1
        generation = self._class_resolve_gen
        recordings = list(self._recordings)
        call = BackgroundCall(
            lambda: detected_class_groups(
                recordings, spec if isinstance(spec, dict) else None
            ),
            receiver=self,
        )
        call.done.connect(
            lambda groups, g=generation: self._apply_class_groups(g, groups)
        )
        call.failed.connect(lambda _exc: None)  # the class is optional
        self._class_call = call
        call.start()

    def _apply_class_groups(self, generation: int, groups: list[dict]) -> None:
        if generation != self._class_resolve_gen:
            return  # a later detection-run selection has superseded this read
        self._class_groups = groups
        repopulate_combo(
            self._class,
            [(_NO_CLASS, None)] + arena_items(groups, len(self._recordings)),
        )
        names = [group["name"] for group in groups]
        self._class_sets.set_classes(
            names,
            labels={g["name"]: g.get("label", g["name"]) for g in groups},
        )
        several = len(names) >= 2
        if not several and self._per_class.isChecked():
            self._per_class.setChecked(False)
        set_available(
            self._per_class,
            several,
            "" if several else "The detection run has fewer than two classes",
        )
        if self._pending_class is not None:
            index = self._class.findData(self._pending_class)
            if index >= 0:
                self._class.setCurrentIndex(index)
                self._pending_class = None

    # -- Test keypoints --------------------------------------------------------- #
    def _test_blocker(self) -> str | None:
        """Why *Test keypoints* cannot run yet, or None. The test runs a method that learns
        from an example set on tracked frames of the chosen detection run, so it needs all
        three."""
        if self._experiment is None:
            return "Open an experiment first"
        if self._unavailable:
            return self._unavailable
        model_id = selected_model_id(self._method)
        cls = KEYPOINT_MODELS.get(model_id) if model_id else None
        if cls is None or not getattr(cls, "needs_examples", False):
            return (
                "Test keypoints tries a method that learns from an example set; run a "
                "trained student on a recording to see its points"
            )
        if not variant_available(self._variant):
            return (
                f"{model_id} {self._variant.currentData()} is not installed: choose an "
                "installed size, or install it in the Model manager"
            )
        if self._per_class_active():
            problem = self._class_sets.problem()
            if problem:
                return problem
        elif self._example_picker.set_id() is None:
            return "Choose an example set, or make one with New example set…"
        if not self.selected_detection_runs():
            return "Choose a detection run first"
        several = self._ambiguous_tracking()
        if several and self._test_tracking.spec() is None:
            return (
                f"{several} has more than one tracking run on this detection run: choose "
                "which one Test keypoints draws its frames from, in Test on tracking"
            )
        if not self.selected_tracking_runs():
            return (
                "Test keypoints draws its frames from a tracking run on the chosen "
                "detection run: run tracking on it first"
            )
        return None

    def _sync_test(self, *_args) -> None:
        reason = self._test_blocker()
        set_available(self._test, reason is None, reason or "")
        if reason is None:
            self._test.setToolTip(
                "Place this method's points on about a dozen tracked frames of the chosen "
                "detection run across the selected recordings, with a score cut-off to look through. Creates no run "
                "and writes nothing."
            )

    def selected_detection_runs(self) -> dict[str, str]:
        """``{recording_id: detection run id}`` for the selected recordings the chosen
        detection settings cover, each resolved to its own run."""
        from detail.api import run_spec

        spec = self._picker.spec()
        if spec is None:
            return {}
        runs = self._detection_runs.get(run_spec.freeze_spec(spec), {})
        selected = {r.id for r in self._recordings}
        return {rec: run for rec, run in runs.items() if rec in selected}

    def selected_tracking_runs(self) -> dict[str, str]:
        """``{recording_id: tracking run id}`` *Test keypoints* draws its frames from: the
        chosen *Test on tracking*'s run on each selected recording, when one is chosen;
        otherwise the one complete tracking run on each recording's chosen detection run. A
        recording with none is left out; one with several and no choice is
        :meth:`_ambiguous_tracking`.
        """
        from detail.api import run_spec

        chosen = self._test_tracking.spec()
        out = {}
        for rec, run in self.selected_detection_runs().items():
            on = self._tracking_on.get((rec, run), [])
            if chosen is not None:
                pick = self._test_tracking_runs.get(run_spec.freeze_spec(chosen), {})
                if (
                    pick.get(rec) in on
                ):  # that run, and on this recording's detection run
                    out[rec] = pick[rec]
            elif len(on) == 1:
                out[rec] = on[0]
        return out

    def _sync_test_tracking(self, *_args) -> None:
        """Offer *Test on tracking* only when a selected recording has several tracking runs
        on its chosen detection run, and only the tracking settings that are among them.
        """
        from detail.api import run_spec

        on_chosen = {
            (rec, tracking)
            for rec, run in self.selected_detection_runs().items()
            for tracking in self._tracking_on.get((rec, run), [])
        }
        offered = [
            group
            for group in self._test_tracking_groups
            if any(
                (rec, tracking) in on_chosen
                for rec, tracking in self._test_tracking_runs.get(
                    run_spec.freeze_spec(group["spec"]), {}
                ).items()
            )
        ]
        self._test_tracking.apply(offered, len(self._recordings))
        needed = self._ambiguous_tracking() is not None
        self._test_tracking_label.setVisible(needed)
        self._test_tracking.combo.setVisible(needed)

    def _ambiguous_tracking(self) -> str | None:
        """The name of a selected recording with several tracking runs on its chosen
        detection run, or None: picking one would be a guess (Tenet 9)."""
        names = {r.id: r.name for r in self._recordings}
        for rec, run in self.selected_detection_runs().items():
            if len(self._tracking_on.get((rec, run), [])) > 1:
                return names.get(rec, rec)
        return None

    def test_teacher(self) -> dict:
        """What *Test keypoints* hands ``Experiment.test_keypoints``: this form's method,
        size, example set, method settings, the tracking on the chosen detection run and
        compute, drawn at its frame step, so the test places points the way the run will.
        """
        settings = {
            "decode": any_field_value(KeypointDetectConfig, "decode", self._decode),
            "side_rule": any_field_value(
                KeypointDetectConfig, "side_rule", self._side_rule
            ),
            "crop_px": optional_value(self._crop_px),
        }
        sets = (
            {"class_example_sets": self._class_sets.value()}
            if self._per_class_active()
            else {"example_set": self._example_picker.set_id()}
        )
        return {
            **sets,
            "tracking": self.selected_tracking_runs(),
            "model_id": selected_model_id(self._method),
            "model_variant": self._variant.currentData(),
            "label_compute": self._compute.compute_target(),
            "settings": {k: v for k, v in settings.items() if v is not None},
            "every_n_frames": self._every_n.value() or 1,
        }

    def _open_test(self) -> None:
        from detail.gui.keypoint_test import KeypointDetectTestDialog

        if self._test_blocker() is not None:
            return
        cls = KEYPOINT_MODELS.get(selected_model_id(self._method) or "")
        spec = getattr(cls, "score_spec", None)
        teacher = self.test_teacher()
        names = {r.id: r.name for r in self._recordings}
        KeypointDetectTestDialog(
            self._experiment,
            teacher,
            {rec: names.get(rec, rec) for rec in teacher["tracking"]},
            cutoff=None if spec is None else spec.default_cutoff,
            parent=self,
        ).exec()

    def _sync_upstream_every_n(self, *_args) -> None:
        """Adopt the picked detection run's cadence: it is the floor this stage can use."""
        spec = self._picker.spec()
        identity = spec.get("identity", {}) if isinstance(spec, dict) else {}
        self._every_n.set_upstream(identity.get("every_n_frames"))

    # -- the launch dialog's hooks --------------------------------------------- #
    def on_experiment_set(self, experiment) -> None:
        self._experiment = experiment
        self._example_picker.set_experiment(experiment)
        self._class_sets.set_experiment(experiment)
        self._refresh_students()
        self._sync_test()

    def _refresh_students(self) -> None:
        """Offer the pose models after the registered methods: this experiment's trained
        students (§12 *Distill keypoints*), then the model catalog's keypoint models (§12
        *Keypoint models in the catalog*), each labelled by where it comes from. A pose model
        is a method too, chosen by its model id; the lookup tries the experiment first, so
        a catalog model sharing a student's id is not offered (``keypoint_students``).
        """
        from detail.models.resolve import keypoint_students

        current = self._method.currentData()
        for student_id in list(self._student_ids):
            index = self._method.findData(student_id)
            if index >= 0:
                self._method.removeItem(index)
        self._student_ids = set()
        self._student_per_detection = {}
        if self._experiment is None:
            return
        for record in keypoint_students(self._experiment.dir):
            self._student_per_detection[record["id"]] = (
                student_input_of(record)["student_input"] == "crop"
            )
            where = _SOURCE_LABELS.get(record.get("source"), "trained student")
            self._method.addItem(
                f"{record.get('name') or record['id']} ({record['id']}) · {where}",
                record["id"],
            )
            self._student_ids.add(record["id"])
        if current is not None and self._method.findData(current) >= 0:
            self._method.setCurrentIndex(self._method.findData(current))
        self._offer_pose_models_when_no_method_runs()
        self._sync_method()

    def _offer_pose_models_when_no_method_runs(self) -> None:
        """A pose model needs no foundation model, so on a machine where no registered
        method can run (no DINOv3 weights), the form must still launch one: drop the
        *no model available* row and open on the first pose model. A catalog model is often
        restored on exactly such a machine."""
        if not self._student_ids:
            self._unavailable = self._methods_unavailable
        else:
            self._unavailable = None
            for index in reversed(range(self._method.count())):
                if self._method.itemData(index) is None:  # the placeholder row
                    self._method.removeItem(index)
            item = self._method.model().item(self._method.currentIndex())
            if item is None or not item.isEnabled():
                for index in range(self._method.count()):
                    if self._method.itemData(index) in self._student_ids:
                        self._method.setCurrentIndex(index)
                        break
        show_note(self._no_methods, self._unavailable)

    def resolve_recordings(self, recordings: list) -> dict:
        """Off the UI thread: every disk read this selection needs, once."""
        with read_cache.pinned():
            model_names = self._experiment.model_names() if self._experiment else {}
            groups = self._picker.resolve(recordings, model_names)
            runs = spec_runs(recordings, "detection", groups)
            tracking_groups = self._test_tracking.resolve(recordings, model_names)
            return {
                "recordings": recordings,
                "groups": groups,
                "runs": runs,
                "tracking_on": _tracking_on(recordings, runs),
                "tracking_groups": tracking_groups,
                "tracking_runs": spec_runs(recordings, "tracking", tracking_groups),
                "rates": source_rate(recordings),
            }

    def apply_recordings(self, recordings: list, resolved: dict) -> None:
        self._recordings = resolved["recordings"]
        self._detection_runs = resolved.get("runs", {})
        self._tracking_on = resolved.get("tracking_on", {})
        self._test_tracking_groups = list(resolved.get("tracking_groups") or [])
        self._test_tracking_runs = dict(resolved.get("tracking_runs") or {})
        self._picker.apply(resolved["groups"], len(self._recordings))
        self._sync_test_tracking()
        self._rates = resolved["rates"]
        self._every_n.set_rates(self._rates)
        self._sync_upstream_every_n()
        self._refresh_class_groups()
        self._sync_test()

    def on_recordings_changed(self, recordings: list) -> None:
        self.apply_recordings(recordings, self.resolve_recordings(recordings))

    def covered_recording_ids(self) -> set[str] | None:
        """The recordings with the chosen detection run and, when a class is chosen, a
        detection of that class on it: a recording without one would be refused by
        ``validate`` (an unknown class), so it is not offered."""
        detection = self._picker.covered()
        if self._per_class_active():
            # Every ticked class must be on a recording (``validate`` checks each one).
            covered = detection
            for name in self._class_sets.value():
                ids = arena_covered_ids(self._class_groups, name) or set()
                covered = ids if covered is None else covered & ids
            return covered
        if not self._class.isVisibleTo(self):
            return detection
        cls = arena_covered_ids(self._class_groups, self._class.currentData())
        if cls is None:
            return detection
        return cls if detection is None else detection & cls

    def load_config(self, config: dict) -> None:
        """Restore a previous run's method and settings. ``detection_spec`` and
        ``example_set_id`` are selections and are re-asked; ``compute`` is this host's.
        """
        index = self._method.findData(config.get("model_id"))
        if index >= 0:
            self._method.setCurrentIndex(index)
        index = self._variant.findData(config.get("model_variant"))
        if index >= 0:
            self._variant.setCurrentIndex(index)
        set_optional_value(self._min_conf, config.get("min_pose_confidence"))
        # A setting (a class name means the same animal on the next recording); kept until
        # the detection run's classes are read if they have not been yet.
        class_name = config.get("class_name")
        index = self._class.findData(class_name) if class_name else 0
        self._class.setCurrentIndex(max(index, 0))
        self._pending_class = class_name if class_name and index < 0 else None
        # Which classes got their own set is a setting; the sets themselves are re-asked.
        by_class = config.get("class_example_sets") or {}
        self._per_class.setChecked(bool(by_class))
        if by_class:
            self._class_sets.load_classes(by_class)
        self._every_n.set_value(config.get("every_n_frames"))
        super().load_config(config)  # decode, side_rule, crop_px, batch_size

    def config(self) -> dict:
        per_class = self._per_class_active()
        return {
            "detection_spec": self._picker.spec(),
            "class_name": (
                self._class.currentData() if self._class.isVisibleTo(self) else None
            ),
            "model_id": selected_model_id(self._method),
            "model_variant": self._variant.currentData(),
            "example_set_id": (
                self._example_picker.set_id()
                if self._example_picker.isVisibleTo(self)
                else None
            ),
            "class_example_sets": self._class_sets.value() if per_class else None,
            "min_pose_confidence": (
                optional_value(self._min_conf)
                if self._min_conf.isVisibleTo(self)
                else None
            ),
            "decode": any_field_value(KeypointDetectConfig, "decode", self._decode),
            "side_rule": any_field_value(
                KeypointDetectConfig, "side_rule", self._side_rule
            ),
            "crop_px": optional_value(self._crop_px),
            "every_n_frames": self._every_n.value(),
            "batch_size": self._batch.value(),
            "compute": self._compute.compute_target(),
        }

    def validation_error(self) -> str | None:
        if self._unavailable:
            return self._unavailable
        if self._picker.spec() is None:
            return "No detection run in the selected recordings — run detection first."
        model_id = selected_model_id(self._method)
        if model_id in self._student_ids or model_id in self._imported_ids:
            return (
                self._every_n.warning()
            )  # a trained student or an imported model: no example set to choose
        cls = KEYPOINT_MODELS.get(model_id) if model_id else None
        if cls is None:
            return "No keypoint method is registered in this build."
        if self._per_class_active():
            return self._class_sets.problem() or self._every_n.warning()
        if cls.needs_examples and self._example_picker.set_id() is None:
            return (
                "Choose the example set this method learns from, or make one with New "
                "example set… and mark the skeleton on a few frames."
            )
        return self._every_n.warning()


def _tracking_on(recordings: list, runs: dict) -> dict:
    """Off the UI thread: ``{(recording_id, detection run id): [complete tracking run
    ids on it]}`` for every detection run ``runs`` offers."""
    by_id = {r.id: r for r in recordings}
    wanted = {(rec, run) for per in runs.values() for rec, run in per.items()}
    out: dict = {}
    for rec, run in sorted(wanted):
        out[(rec, run)] = sorted(
            str(t["id"])
            for t in by_id[rec].tracking_runs  # off-thread
            if str(t.get("status", "complete")) == "complete"
            and str(t.get("detection_run_id")) == str(run)
        )
    return out


@register_module_ui
class KeypointDetectModuleUI(ModuleUI):
    module_id = "keypoint_detect"

    def config_widget(self) -> ConfigWidget:
        return _KeypointDetectConfig()

    def preview_widget(
        self, recording, run_id: str, *, experiment_dir, parent_run_id
    ) -> QWidget | None:
        """The viewer takes the export context like every per-recording viewer, and offers no
        frame / video export of its own: a keypoint run exports through the run page's one
        Export control, the five keypoint targets (§26 *Analysis and export*; #693 left out).
        """
        from detail.gui.keypoint_run_preview import KeypointDetectRunPreview

        del experiment_dir, parent_run_id
        return KeypointDetectRunPreview(recording, run_id)
