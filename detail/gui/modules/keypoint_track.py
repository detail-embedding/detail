"""The ``keypoint_track`` launch form and its run viewer registration (§26 "Surfaces").

The detect run comes through an ``UpstreamSpecPicker`` (one option per set of detect settings
across the selection), and so does the **tracking run** whose subjects the poses are given to
(§26 *Assignment*): its first row lets the stage resolve the one tracking run on the detect
run's detection run — ``validate`` refuses when there are several, and the launch shows that
message — and the other rows name settings explicitly. The two match settings are read off
``KeypointTrackConfig``. The method is a choice between the registered track methods or *the
detect run's method's default*, which is what the stage resolves when nothing is picked; the
default is looked up through ``keypoint_method_class``, the one lookup (a trained student is
not in ``KEYPOINT_MODELS``). The
chosen method's knobs are read off **its** config model (§6.5) and rebuilt when the method
changes — the way the tracking form rebuilds a tracker's hyperparameters — so a knob is never
typed into this file. Every knob is identity for the run (§26 "Identity").
"""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QWidget

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.describe import describe_config
from detail.configs.keypoint_track import METHOD_CONFIGS, KeypointTrackConfig
from detail.core.registries import KEYPOINT_TRACK_METHODS
from detail.executor.runners import import_runner_modules
from detail.gui._pydantic_form import (
    any_field_value,
    any_field_widget,
    field_meta,
    set_any_field_value,
)
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label
from detail.gui.modules._common import UpstreamSpecPicker
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    register_module_ui,
    surfaced,
)
from detail.storage import read_cache

_MODULE = "keypoint_track"
_DEFAULT_SUFFIX = "  (default)"
_KNOBS = "Method settings"
_AUTO_TRACKING = "The one on the detect run's detection run"
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
        self._knobs: dict[str, QWidget] = {}
        self._recordings: list = []
        self._method.currentIndexChanged.connect(self._rebuild_knobs)
        self._sync_default()
        self._rebuild_knobs()
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

    # -- the launch dialog's hooks --------------------------------------------- #
    def on_experiment_set(self, experiment) -> None:
        """A trained student's default is found through the experiment's models."""
        self._experiment = experiment
        self._sync_default()

    def resolve_recordings(self, recordings: list) -> dict:
        with read_cache.pinned():
            return {
                "recordings": recordings,
                "groups": self._picker.resolve(recordings),
                "tracking_groups": self._tracking_picker.resolve(recordings),
            }

    def apply_recordings(self, recordings: list, resolved: dict) -> None:
        self._recordings = resolved["recordings"]
        self._picker.apply(resolved["groups"], len(self._recordings))
        self._tracking_picker.apply(
            resolved.get("tracking_groups") or [], len(self._recordings)
        )

    def prefer_upstream_spec(self, module_id: str, spec: dict) -> None:
        """A next-step launch from a tracking run (it is this stage's second input): open
        the tracking picker on that run's settings rather than on the automatic row."""
        if module_id == "tracking":
            self._tracking_picker.prefer(spec)

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
        tracking runs are selections and are re-asked."""
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
            if widget is not None:
                set_any_field_value(model, name, widget, value)

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
                name: any_field_value(model, name, widget)
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
        }

    def validation_error(self) -> str | None:
        if self._picker.spec() is None:
            return (
                "No keypoint detect run in the selected recordings — run keypoint detect "
                "first."
            )
        return None


@register_module_ui
class KeypointTrackModuleUI(ModuleUI):
    module_id = "keypoint_track"

    def config_widget(self) -> ConfigWidget:
        return _KeypointTrackConfig()

    def preview_widget(self, recording, run_id: str) -> QWidget | None:
        from detail.gui.keypoint_run_preview import KeypointTrackRunPreview

        return KeypointTrackRunPreview(recording, run_id)
