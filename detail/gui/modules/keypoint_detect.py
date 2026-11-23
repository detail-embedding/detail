"""The ``keypoint_detect`` launch form and its run viewer registration (§26 "Surfaces").

Copies the tracking form's shape: the **detection run** through an ``UpstreamSpecPicker`` (one
option per set of settings across the selection; §26 revised 2026-10-21 — a detect run places
poses on every animal and keeps them unassigned, so it asks for no tracking run and no tracks),
the method and its size from the ``KEYPOINT_MODELS`` registry plus the experiment's trained
students and the model catalog's keypoint models, and every knob read off ``KeypointDetectConfig`` (§6.5) — the two ``Literal | None``
knobs render through ``any_field_widget`` with an *auto* row, which is what "let the method
resolve it" means. ``min_pose_confidence`` is shown only for a method that
``reports_confidence`` (the student), the one kind ``validate`` accepts it for.

The detection run and the example set are **selections** (they name artifacts this experiment
mints), so *Start from* never restores them; the method and its settings are.
"""

from __future__ import annotations

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.keypoint_detect import (
    DEFAULT_CROP_PX,
    DEFAULT_MIN_POSE_CONFIDENCE,
    KeypointDetectConfig,
)
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
from detail.gui.help_ui import field_label
from detail.gui.model_availability import (
    apply_availability,
    availability_note,
    fill_variants,
    selected_model_id,
    show_note,
    variant_available,
)
from detail.gui.example_set_picker import ExampleSetPicker
from detail.gui.modules._common import UpstreamSpecPicker, spec_runs
from detail.gui.modules._every_n import EveryNRow, source_rate
from detail.gui.qt import QComboBox, QPushButton, QWidget
from detail.gui.sizing import set_available
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    register_module_ui,
    surfaced,
)
from detail.storage import read_cache

_MODULE = "keypoint_detect"
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
        self._method = QComboBox()
        self._method.setToolTip(field_meta(KeypointDetectConfig, "model_id")["help"])
        for model_id in surfaced(KEYPOINT_MODELS):
            self._method.addItem(model_id, model_id)
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
                    + ". Install it on the Model manager."
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
        self._compute = ComputeTargetRow()
        self._compute.add_to(form.section())
        # The keypoints counterpart of detection's Test prompts, in the same place: the last
        # row, opening `KeypointDetectTestDialog` on this form's method, size, example set,
        # settings and the tracked frames of the chosen detection run. It has no "Use this cut-off": a detect run stores raw
        # points, and the cut-off is chosen where they are viewed or exported.
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
        # Hidden rather than greyed: an example set is an input of this method, not a
        # setting every method has (§26).
        self._example_picker.setVisible(needs)
        self._examples_label.setVisible(needs)
        # Hidden for the same reason: only a method that reports a pose confidence takes the
        # threshold, and `validate` refuses it for any other (§26 *Visibility*).
        reports = bool(getattr(cls, "reports_confidence", False))
        self._min_conf.setVisible(reports)
        self._min_conf_label.setVisible(reports)
        self._sync_test()

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
        if self._example_picker.set_id() is None:
            return "Choose an example set, or make one with New example set…"
        if not self.selected_detection_runs():
            return "Choose a detection run first"
        several = self._ambiguous_tracking()
        if several:
            return (
                f"{several} has more than one tracking run on this detection run; Test "
                "keypoints draws from one recording's tracked frames, so leave it out"
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
        """``{recording_id: tracking run id}``: the one complete tracking run on each
        selected recording's chosen detection run — the frames *Test keypoints* draws from.
        A recording with none is left out; one with several is :meth:`_ambiguous_tracking`.
        """
        out = {}
        for rec, run in self.selected_detection_runs().items():
            on = self._tracking_on.get((rec, run), [])
            if len(on) == 1:
                out[rec] = on[0]
        return out

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
        return {
            "example_set": self._example_picker.set_id(),
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
        if self._experiment is None:
            return
        for record in keypoint_students(self._experiment.dir):
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
            groups = self._picker.resolve(recordings)
            runs = spec_runs(recordings, "detection", groups)
            return {
                "recordings": recordings,
                "groups": groups,
                "runs": runs,
                "tracking_on": _tracking_on(recordings, runs),
                "rates": source_rate(recordings),
            }

    def apply_recordings(self, recordings: list, resolved: dict) -> None:
        self._recordings = resolved["recordings"]
        self._detection_runs = resolved.get("runs", {})
        self._tracking_on = resolved.get("tracking_on", {})
        self._picker.apply(resolved["groups"], len(self._recordings))
        self._rates = resolved["rates"]
        self._every_n.set_rates(self._rates)
        self._sync_upstream_every_n()
        self._sync_test()

    def on_recordings_changed(self, recordings: list) -> None:
        self.apply_recordings(recordings, self.resolve_recordings(recordings))

    def covered_recording_ids(self) -> set[str] | None:
        return self._picker.covered()

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
        self._every_n.set_value(config.get("every_n_frames"))
        super().load_config(config)  # decode, side_rule, crop_px, batch_size

    def config(self) -> dict:
        return {
            "detection_spec": self._picker.spec(),
            "model_id": selected_model_id(self._method),
            "model_variant": self._variant.currentData(),
            "example_set_id": (
                self._example_picker.set_id()
                if self._example_picker.isVisibleTo(self)
                else None
            ),
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
        if model_id in self._student_ids:
            return (
                self._every_n.warning()
            )  # a trained student: no example set to choose
        cls = KEYPOINT_MODELS.get(model_id) if model_id else None
        if cls is None:
            return "No keypoint method is registered in this build."
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

    def preview_widget(self, recording, run_id: str) -> QWidget | None:
        from detail.gui.keypoint_run_preview import KeypointDetectRunPreview

        return KeypointDetectRunPreview(recording, run_id)
