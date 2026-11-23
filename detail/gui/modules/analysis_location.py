"""Location-export analysis module UI (§16).

Config: a tracking-run picker offering tracking **specs** (§6.5) — a run's full-lineage
kind — found across the selected recordings, annotated with ``N/M`` coverage, several of which
may be ticked (``PooledSpecPicker``); ``config()`` emits ``tracking_spec`` or
``tracking_specs`` and the run resolves them to each recording's own tracking run. Below it,
the measurement options (rows per second), each row built from
``LocationExportConfig`` so its range, default and help are the model's. Result: the
per-recording CSV artifacts the run produced, read from its ``analysis.yaml`` manifest.
"""

from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)

from detail.configs.location_export import LocationExportConfig
from detail.configs.location_export import option_names
from detail.gui._pydantic_form import any_field_value
from detail.gui._pydantic_form import any_field_widget
from detail.gui._pydantic_form import field_meta
from detail.gui._pydantic_form import set_any_field_value
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label
from detail.gui.modules._common import PooledSpecPicker
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    register_analysis_module_ui,
)

#: What the rows-per-second box reads when it is left unset.
EVERY_FRAME_TEXT = "every frame"


class _LocationExportConfig(ConfigWidget):
    def __init__(self) -> None:
        super().__init__()
        self._model_names: dict = {}  # {model_id: name}; filled by on_experiment_set
        form = ConfigForm(self, LocationExportConfig)
        self._tracking = PooledSpecPicker("tracking")
        form.addRow(
            field_label("location_export", "tracking_run_id", "Tracking run:"),
            self._tracking.widget,
        )
        self._tracking.changed.connect(lambda _item: self.coverage_changed.emit())
        # One row per measurement option, enumerated from the model: an option added to it
        # gets a row, a value in `config()` and a restore in `load_config` without being
        # named here. `_options` rather than `_fields`, because the base `load_config` cannot
        # set an optional box and this form restores every option the same way itself.
        self._options: dict[str, QWidget] = {}
        for name in option_names():
            widget = any_field_widget(LocationExportConfig, name)
            label = field_meta(LocationExportConfig, name)["label"]
            form.addRow(field_label("location_export", name, f"{label}:"), widget)
            self._options[name] = widget
        self._options["sample_fps"].setSpecialValueText(EVERY_FRAME_TEXT)
        form.finish()

    def on_experiment_set(self, experiment) -> None:
        # A run record stores a model *id*; a picker has to show the name (§11). O(models),
        # not O(recordings), so it stays inside the UI-thread rule.
        self._model_names = experiment.model_names() if experiment is not None else {}

    def resolve_recordings(self, recordings: list) -> dict:
        """Off the UI thread (CLAUDE.md). ``spec_groups`` walks all four processing modules
        per recording — several ``recording.yaml`` round trips per recording,
        synchronously on every checkbox click. Pinned, so the cache's per-property
        ``stat`` is paid once per recording rather than once per field asked for.
        """
        return {"groups": self._tracking.resolve(recordings, self._model_names)}

    def apply_recordings(self, recordings: list, resolved: dict) -> None:
        """UI thread: install the resolved groups and repopulate. No disk."""
        self._tracking.apply(resolved["groups"], len(recordings))

    def on_recordings_changed(self, recordings: list) -> None:
        """Synchronous path for direct callers (the launch dialog uses resolve→apply)."""
        self.apply_recordings(recordings, self.resolve_recordings(recordings))

    def covered_recording_ids(self) -> set[str] | None:
        return self._tracking.covered()

    def config(self) -> dict:
        options = {
            name: any_field_value(LocationExportConfig, name, widget)
            for name, widget in self._options.items()
        }
        return {**self._tracking.selection(), **options}

    def load_config(self, config: dict) -> None:
        """Restore the measurement options of a previous run; the tracking selection is
        re-asked, because it names runs this selection of recordings may not have."""
        for name, widget in self._options.items():
            if name in config:
                set_any_field_value(LocationExportConfig, name, widget, config[name])

    def validation_error(self) -> str | None:
        if not self._tracking.specs():
            return "No tracking run in the selected recordings — run tracking first."
        return self._tracking.validation_error()


@register_analysis_module_ui
class LocationExportModuleUI(ModuleUI):
    module_id = "location_export"
    purpose = "The location tables this run wrote, per recording."

    def config_widget(self) -> ConfigWidget:
        return _LocationExportConfig()

    def result_widget(self, experiment, run_record: dict) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        run_id = run_record.get("id", "")
        layout.addWidget(
            QLabel(
                f"Per-track ROI/POI spatial features, written under "
                f"analysis/{run_id}/<recording>/:"
            )
        )
        artifacts = run_record.get("artifacts", [])
        if not artifacts:
            layout.addWidget(QLabel("(no output files recorded)"))
            return widget
        from detail.gui.pages import _table  # local: avoid an import cycle

        rows = [(a.get("path", ""), a.get("kind", "")) for a in artifacts]
        layout.addWidget(_table(["File", "Kind"], rows))
        return widget
