"""The ``keypoint_track`` launch form and its run viewer registration (§26 "Surfaces").

The detect run comes through an ``UpstreamSpecPicker`` (one option per set of detect settings
across the selection), and so does the **tracking run** whose subjects the poses are given to
(§26 *Assignment*): its first row lets the stage resolve the one tracking run on the detect
run's detection run — ``validate`` refuses when there are several, and the launch shows that
message — and the other rows name settings explicitly. A row whose run is on another detection
run than the chosen detect run's is greyed, saying so, and the rows follow the detect run as it
changes: ``validate`` refuses that pair, so the form does not offer it. The two match settings
are read off
``KeypointTrackConfig``. The method is a choice between the registered track methods or *the
detect run's method's default*, which is what the stage resolves when nothing is picked; the
default is looked up through ``keypoint_method_class``, the one lookup (a trained student is
not in ``KEYPOINT_MODELS``). The
chosen method's knobs are read off **its** config model (§6.5) and rebuilt when the method
changes — the way the tracking form rebuilds a tracker's hyperparameters — so a knob is never
typed into this file. Every knob is identity for the run (§26 "Identity").
"""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QWidget

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.describe import describe_config
from detail.configs.keypoint_track import (
    METHOD_CONFIGS,
    TUNE_METHOD,
    KeypointTrackConfig,
    TemporalDecodeConfig,
)
from detail.core.registries import KEYPOINT_MODELS, KEYPOINT_TRACK_METHODS
from detail.executor.runners import import_runner_modules
from detail.gui._pydantic_form import (
    any_field_value,
    any_field_widget,
    field_meta,
    set_any_field_value,
)
from detail.gui.compute_target import ComputeTargetRow
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label
from detail.gui.modules._common import (
    UpstreamSpecPicker,
    lineage_fits,
    spec_reads,
    widest_group,
)
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    register_module_ui,
    surfaced,
)
from detail.gui.sizing import set_available
from detail.gui.skeleton_points import SkeletonPointsPicker
from detail.storage import read_cache

_MODULE = "keypoint_track"
_DEFAULT_SUFFIX = "  (default)"
_KNOBS = "Method settings"
#: The ``ui`` hint a config field declares to be shown as a skeleton's points (``track_points``).
_POINTS_UI = "skeleton_points"
_AUTO_TRACKING = "The one on the detect run's detection run"
_OTHER_DETECTION = "on another detection run than the keypoint detect run"
#: The match settings, read off the config model (§26 *Assignment*; policy, not measured).
_MATCH_FIELDS = ("match_min_inside", "match_mask_grow")


class _KeypointTrackConfig(ConfigWidget):
    config_model = KeypointTrackConfig

    def __init__(self):
        super().__init__()
        import_runner_modules(PIPELINE_RUNNERS)  # populate KEYPOINT_TRACK_METHODS
        self._config_form = ConfigForm(self, KeypointTrackConfig)
        form = self._config_form
        self._picker = UpstreamSpecPicker("keypoint_detect")
        self._detect = self._picker.combo
        self._detect.setToolTip(
            field_meta(KeypointTrackConfig, "detect_run_id")["help"]
        )
        self._detect.currentIndexChanged.connect(
            lambda _index: self.coverage_changed.emit()
        )
        self._detect.currentIndexChanged.connect(self._sync_default)
        form.addRow(
            field_label(_MODULE, "detect_run_id", "Keypoint detect run:"), self._detect
        )
        self._tracking_picker = UpstreamSpecPicker(
            "tracking", auto_label=_AUTO_TRACKING
        )
        self._tracking = self._tracking_picker.combo
        self._tracking.setToolTip(
            field_meta(KeypointTrackConfig, "tracking_run_id")["help"]
        )
        self._tracking.currentIndexChanged.connect(
            lambda _index: self.coverage_changed.emit()
        )
        form.addRow(
            field_label(_MODULE, "tracking_run_id", "Tracking run:"), self._tracking
        )
        # {frozen spec: {recording_id: detection run id}} each detect / tracking row read.
        self._detect_reads: dict = {}
        self._tracking_reads: dict = {}
        # A next-step launch's tracking run names its detection run (``parent``).
        self._preferred_detection: dict | None = None
        self._detect.currentIndexChanged.connect(self._sync_tracking)
        self._match: dict[str, QWidget] = {}
        for name in _MATCH_FIELDS:
            widget = any_field_widget(KeypointTrackConfig, name)
            self._match[name] = widget
            form.addRow(
                field_label(
                    _MODULE, name, field_meta(KeypointTrackConfig, name)["label"] + ":"
                ),
                widget,
                advanced=True,
            )
        self._experiment = None
        self._method = QComboBox()
        self._method.setToolTip(field_meta(KeypointTrackConfig, "method")["help"])
        # The methods only; the detect run's own default is selected and marked
        # "(default)" in the list rather than offered as a row of its own (in-app review).
        for method_id in surfaced(KEYPOINT_TRACK_METHODS):
            self._method.addItem(method_id, method_id)
        # Once the user picks a method, a change of detect run no longer moves it.
        self._user_chose = False
        self._method.activated.connect(self._mark_user_choice)
        form.addRow(field_label(_MODULE, "method", "Method:"), self._method)
        # The points the method may move (`track_points`, both methods): the chosen detect
        # run's own skeleton, whatever it is, listed once it is read off the UI thread.
        self._points = SkeletonPointsPicker()
        self._points.setToolTip(
            field_meta(TemporalDecodeConfig, "track_points")["help"]
        )
        self._skeletons: dict = {}  # frozen detect spec -> (names, pairs)
        # The ? opens temporal_decode's parameter page: both methods declare the field
        # identically, and the row is shared across them.
        form.addRow(
            field_label("temporal_decode", "track_points", "Points to track:"),
            self._points,
        )
        self._points.changed.connect(self.coverage_changed.emit)
        self._detect.currentIndexChanged.connect(self._sync_points)
        self._knobs: dict[str, QWidget] = {}
        self._recordings: list = []
        self._recording_names: dict[str, str] = {}
        self._experiment = None
        self._method.currentIndexChanged.connect(self._rebuild_knobs)
        # Where it runs: this machine's CPU or a registered cluster, which takes a batch's
        # recordings side by side (§6); there is no model, so no GPU to choose.
        self._compute = ComputeTargetRow(allow_remote=True, cpu_only=True)
        self._compute.add_to(form.section())
        # Tune tracking: the method's settings scored against an example set's labels
        # (§26 *Tuning a track method*); the last row, like detect's Test keypoints.
        self._tune = QPushButton("Tune tracking…")
        self._tune.clicked.connect(self._open_tune)
        # Preview tracking: the settings side by side over a few seconds of a few recordings,
        # no labels needed (§26 *Previewing a track method*); beside Tune on the last row.
        self._preview = QPushButton("Preview tracking…")
        self._preview.clicked.connect(self._open_preview)
        tools = QWidget()
        row = QHBoxLayout(tools)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._preview)
        row.addWidget(self._tune)
        row.addStretch(1)
        form.addRow("", tools)
        self._detect.currentIndexChanged.connect(self._sync_tune)
        self._method.currentIndexChanged.connect(self._sync_tune)
        self._tracking.currentIndexChanged.connect(self._sync_tune)
        self._sync_default()
        self._rebuild_knobs()
        self._sync_tune()
        form.finish()

    # -- the detect run's default method ------------------------------------------ #
    def _default_method(self) -> str | None:
        """The chosen detect run's method's own track method (`default_track_method`):
        `temporal_decode` for `dinov3_match`, `despike_smooth` for a trained student —
        through `keypoint_method_class`, the one lookup, with no table or fallback id here
        (§26 *Track methods*). None until a detect run is chosen, or for a method the lookup
        does not know; the stage then refuses rather than guess."""
        from detail.models.resolve import keypoint_method_class

        spec = self._picker.spec()
        if spec is None:
            return None
        model_id = (spec.get("identity") or {}).get("model_id")
        experiment_dir = None if self._experiment is None else self._experiment.dir
        cls = keypoint_method_class(experiment_dir, model_id)
        default = getattr(cls, "default_track_method", None)
        if default is None or self._method.findData(default) < 0:
            return None
        return default

    def _sync_default(self, *_args) -> None:
        """Mark the default in the list and, unless the user picked a method, select it."""
        default = self._default_method()
        for index in range(self._method.count()):
            method_id = self._method.itemData(index)
            label = method_id + (_DEFAULT_SUFFIX if method_id == default else "")
            self._method.setItemText(index, label)
        if not self._user_chose and default is not None:
            self._method.setCurrentIndex(self._method.findData(default))

    def _mark_user_choice(self, *_args) -> None:
        self._user_chose = True

    # -- the method drives its knobs -------------------------------------------- #
    def _rebuild_knobs(self, *_args) -> None:
        method = self._method.currentData()
        self._config_form.advanced().clear(_KNOBS)
        self._knobs = {}
        if method is not None:
            model = METHOD_CONFIGS[method]
            for field in describe_config(model):
                if field["ui"] == _POINTS_UI:
                    # The detect run's points, on the main form (one row, kept across method
                    # changes) rather than under Advanced: it is a choice, not a constant.
                    self._knobs[field["name"]] = self._points
                    continue
                widget = any_field_widget(model, field["name"])
                self._knobs[field["name"]] = widget
                self._config_form.addRow(
                    field_label(method, field["name"], field["label"] + ":"),
                    widget,
                    advanced=_KNOBS,
                )
        # The section was hidden at construction (no method, no rows) and finish() is what
        # shows or hides it, so a rebuild must call it again or the settings stay unreachable
        # behind a hidden Advanced (reported from the app).
        self._config_form.finish()

    # -- Tune tracking ------------------------------------------------------------- #
    def _tune_blocker(self) -> str | None:
        """Why *Tune tracking* cannot run yet, or None."""
        if self._experiment is None:
            return "Open an experiment first"
        if self._picker.spec() is None:
            return "Choose a keypoint detect run first"
        if self._method.currentData() != TUNE_METHOD:
            return f"Tune tracking tries {TUNE_METHOD} settings; choose {TUNE_METHOD}"
        return None

    def _preview_blocker(self) -> str | None:
        """Why *Preview tracking* cannot run yet, or None."""
        if self._experiment is None:
            return "Open an experiment first"
        if self._picker.spec() is None:
            return "Choose a keypoint detect run first"
        if self._method.currentData() is None:
            return "Choose a track method first"
        if not self._preview_recordings():
            return "No selected recording has this keypoint detect run and tracking run"
        return None

    def _preview_recordings(self) -> list[tuple[str, str]]:
        """The selected recordings the form's settings cover, as ``(id, name)``."""
        covered = self.covered_recording_ids()
        return [
            (r.id, self._recording_names.get(r.id, r.id))
            for r in self._recordings
            if covered is None or r.id in covered
        ]

    def _open_preview(self) -> None:
        from detail.gui.keypoint_track_preview import KeypointTrackPreviewDialog

        if self._preview_blocker() is not None:
            return
        current = self.config()
        dialog = KeypointTrackPreviewDialog(
            self._experiment, current, self._preview_recordings(), self
        )
        if dialog.exec() and dialog.chosen_config is not None:
            self.load_config(
                {"method": current["method"], "config": dialog.chosen_config}
            )

    def _sync_tune(self, *_args) -> None:
        reason = self._preview_blocker()
        set_available(self._preview, reason is None, reason or "")
        if reason is None:
            self._preview.setToolTip(
                "Run tracking on a few seconds of a few recordings for your current settings "
                "and two variants, and watch them side by side with detection alone. No "
                "labels needed; creates no run and writes nothing."
            )
        reason = self._tune_blocker()
        set_available(self._tune, reason is None, reason or "")
        if reason is None:
            self._tune.setToolTip(
                "Try tracking settings against frames you labelled in an example set the "
                "detect run did not learn from, and pick the one that keeps the most points "
                "on target. Creates no run and writes nothing."
            )

    def _open_tune(self) -> None:
        from detail.gui.keypoint_track_tune import KeypointTrackTuneDialog

        if self._tune_blocker() is not None:
            return
        current = self.config()
        dialog = KeypointTrackTuneDialog(
            self._experiment,
            current["keypoint_detect_spec"],
            current["config"],
            self,
            tracking_spec=current.get("tracking_spec"),
            match={
                "min_inside": current["match_min_inside"],
                "mask_grow": current["match_mask_grow"],
            },
        )
        if dialog.exec() and dialog.chosen_config is not None:
            self.load_config({"method": TUNE_METHOD, "config": dialog.chosen_config})

    # -- the launch dialog's hooks --------------------------------------------- #
    def on_experiment_set(self, experiment) -> None:
        """A trained student's default is found through the experiment's models."""
        self._experiment = experiment
        self._sync_default()
        self._sync_tune()

    def resolve_recordings(self, recordings: list) -> dict:
        """Off the UI thread: the detect-run groups, and each group's skeleton (its names and
        pairs, read from one of its runs) for the points picker."""
        from detail.gui.modules._common import spec_runs
        from detail.storage.keypoint_runs import KeypointDetectRun

        model_names = self._experiment.model_names() if self._experiment else {}
        with read_cache.pinned():
            groups = self._picker.resolve(recordings, model_names)
            tracking_groups = self._tracking_picker.resolve(recordings, model_names)
            by_id = {r.id: r for r in recordings}
            skeletons = {}
            for key, runs in spec_runs(recordings, "keypoint_detect", groups).items():
                for recording_id, run_id in runs.items():
                    try:
                        skeleton = KeypointDetectRun(
                            by_id[recording_id].dir, run_id
                        ).skeleton
                    except (FileNotFoundError, KeyError):
                        continue
                    skeletons[key] = (skeleton.names, skeleton.pairs)
                    break
            return {
                "recordings": recordings,
                "groups": groups,
                "skeletons": skeletons,
                "tracking_groups": tracking_groups,
                "detect_reads": spec_reads(
                    recordings, "keypoint_detect", groups, "detection_run_id"
                ),
                "tracking_reads": spec_reads(
                    recordings, "tracking", tracking_groups, "detection_run_id"
                ),
                # Preview tracking's recording list names them; read here, off the UI thread.
                "names": {r.id: r.name for r in recordings},
            }

    def apply_recordings(self, recordings: list, resolved: dict) -> None:
        self._recordings = resolved["recordings"]
        self._recording_names = resolved.get("names", {})
        self._skeletons = resolved.get("skeletons", {})
        self._detect_reads = resolved.get("detect_reads", {})
        self._tracking_reads = resolved.get("tracking_reads", {})
        self._prefer_detect_on(self._preferred_detection, resolved["groups"])
        self._picker.apply(resolved["groups"], len(self._recordings))
        self._sync_points()
        self._tracking_picker.apply(
            resolved.get("tracking_groups") or [], len(self._recordings)
        )
        self._sync_tracking()
        self._sync_tune()

    def _sync_tracking(self, *_args) -> None:
        """Offer the tracking runs on the chosen detect run's detection run, recording by
        recording: the stage gives the poses only to tracks of the detections they were
        placed on (§26 *Assignment*) and refuses any other pair."""
        from detail.api import run_spec

        spec = self._picker.spec()
        if spec is None:
            self._tracking_picker.restrict(None)
            return
        wanted = self._detect_reads.get(run_spec.freeze_spec(spec), {})
        self._tracking_picker.restrict(
            lineage_fits(self._tracking_reads, wanted), _OTHER_DETECTION
        )

    def prefer_upstream_spec(self, module_id: str, spec: dict) -> None:
        """A next-step launch from a tracking run (it is this stage's second input): open
        the tracking picker on that run's settings rather than on the automatic row, and the
        detect picker on a detect run of that run's detection run, beside which it is
        offered."""
        if module_id == "tracking":
            self._preferred_detection = spec.get("parent")
            self._prefer_detect_on(self._preferred_detection, self._picker.groups)
            self._tracking_picker.prefer(spec)

    def _prefer_detect_on(self, detection: dict | None, groups: list[dict]) -> None:
        """Prefer the widest (newest on a tie) detect run on ``detection``, if any."""
        from detail.api import run_spec

        if detection is None:
            return
        key = run_spec.freeze_spec(detection)
        on = [g for g in groups if run_spec.freeze_spec(g["spec"]["parent"]) == key]
        if on:
            self._picker.prefer(widest_group(on)["spec"])

    def on_recordings_changed(self, recordings: list) -> None:
        self.apply_recordings(recordings, self.resolve_recordings(recordings))

    def covered_recording_ids(self) -> set[str] | None:
        covered = self._picker.covered()
        by_tracking = self._tracking_picker.covered()
        if covered is None:
            return by_tracking
        return covered if by_tracking is None else covered & by_tracking

    def load_config(self, config: dict) -> None:
        """Restore the method, the match and the method's settings; the detect and
        tracking runs are selections and are re-asked, and ``compute`` is this host's.
        """
        for name, widget in self._match.items():
            if config.get(name) is not None:
                set_any_field_value(KeypointTrackConfig, name, widget, config[name])
        index = self._method.findData(config.get("method"))
        if index >= 0:
            self._user_chose = True
            self._method.setCurrentIndex(index)
        else:  # no method recorded: the detect run's default, as on a fresh form
            self._user_chose = False
            self._sync_default()
        method = self._method.currentData()
        if method is None:
            return
        model = METHOD_CONFIGS[method]
        for name, value in (config.get("config") or {}).items():
            widget = self._knobs.get(name)
            if widget is self._points:
                self._points.set_value(value)
            elif widget is not None:
                set_any_field_value(model, name, widget, value)

    # -- the detect run's points -------------------------------------------------- #
    def _sync_points(self, *_args) -> None:
        """Show the chosen detect run's skeleton in the points picker."""
        from detail.api import run_spec

        spec = self._picker.spec()
        found = (
            None if spec is None else self._skeletons.get(run_spec.freeze_spec(spec))
        )
        if found is not None:
            self._points.set_skeleton(*found)

    def nested_config_model(self, key: str, config: dict):
        """The model that describes a nested part of :meth:`config`: ``config`` is the chosen
        method's settings (`METHOD_CONFIGS`). Read by the round-trip guard so it varies them
        within their bounds."""
        if key == "config":
            return METHOD_CONFIGS.get(config.get("method") or "")
        return None

    def config(self) -> dict:
        method = self._method.currentData()
        knobs: dict = {}
        if method is not None:
            model = METHOD_CONFIGS[method]
            knobs = {
                name: (
                    widget.value()
                    if widget is self._points
                    else any_field_value(model, name, widget)
                )
                for name, widget in self._knobs.items()
            }
        tracking = self._tracking_picker.spec()
        return {
            "keypoint_detect_spec": self._picker.spec(),
            # Omitted on the automatic row, so validate resolves the one tracking run on the
            # detect run's detection run (and refuses several, naming them).
            **({} if tracking is None else {"tracking_spec": tracking}),
            **{
                name: any_field_value(KeypointTrackConfig, name, widget)
                for name, widget in self._match.items()
            },
            "method": method,
            "config": knobs,
            "compute": self._compute.compute_target(),
        }

    def validation_error(self) -> str | None:
        if self._picker.spec() is None:
            return (
                "No keypoint detect run in the selected recordings — run keypoint detect "
                "first."
            )
        if self._points.value() == []:
            return "Tick at least one point to track (all of them tracks every point)."
        return None


@register_module_ui
class KeypointTrackModuleUI(ModuleUI):
    module_id = "keypoint_track"

    def config_widget(self) -> ConfigWidget:
        return _KeypointTrackConfig()

    def preview_widget(
        self, recording, run_id: str, *, experiment_dir, parent_run_id
    ) -> QWidget | None:
        """The viewer takes the export context like every per-recording viewer, and offers no
        frame / video export of its own: a keypoint run exports through the run page's one
        Export control, the five keypoint targets (§26 *Analysis and export*; #693 left out).
        """
        from detail.gui.keypoint_run_preview import KeypointTrackRunPreview

        del experiment_dir, parent_run_id
        return KeypointTrackRunPreview(recording, run_id)
