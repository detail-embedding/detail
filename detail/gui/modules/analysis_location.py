"""Location-export analysis module UI (§16).

Config: a tracking-run picker offering tracking **specs** (§6.5) — a run's full-lineage
kind — found across the selected recordings, annotated with ``N/M`` coverage, several of which
may be ticked (``PooledSpecPicker``); ``config()`` emits ``tracking_spec`` or
``tracking_specs`` and the run resolves them to each recording's own tracking run. Below it,
the measurement options (rows per second, thin parts), each row built from
``LocationExportConfig`` so its range, default and help are the model's. Result: two views,
**Plots** (``gui/location_plots.py``, the run drawn from its own CSVs) and **Files** (the
per-recording CSV artifacts the run produced, read from its ``analysis.yaml`` manifest).
"""

from PyQt6.QtWidgets import (
    QLabel,
    QTabWidget,
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
from detail.gui.result_views import MultiViewWorkspace

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
        # The scale means something only while thin parts are being removed.
        thin = self._options["remove_thin_parts"]
        thin.toggled.connect(self._options["thin_part_scale"].setEnabled)
        self._options["thin_part_scale"].setEnabled(thin.isChecked())
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


#: The run page's two views of a location export, and what each is for.
PLOTS_VIEW = (
    "Plots",
    "The animal's path, where it spent its time, its speed and its body.",
)
FILES_VIEW = ("Files", "The location tables this run wrote, per recording.")


def _files_widget(run_record: dict) -> QWidget:
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


class LocationResults(MultiViewWorkspace):
    """A location export's result: its Plots, drawn from the run's own CSVs, and its Files.

    Both views always, so the cards a running export shows are the views it ends with: a run
    that wrote no tables says so on its Plots view when opened."""

    def __init__(self, experiment, run_record: dict) -> None:
        super().__init__()
        from detail.gui.location_plots import LocationPlotsPanel

        tabs = QTabWidget()
        per_recording = run_record.get("per_recording") or {}
        complete = sum(
            1 for state in per_recording.values() if state.get("status") == "complete"
        )
        self.plots = LocationPlotsPanel(experiment, run_record.get("id", ""))
        tabs.addTab(self.plots, PLOTS_VIEW[0])
        specs = [(*PLOTS_VIEW, self.plots, lambda: f"{complete} recordings to plot")]
        files = _files_widget(run_record)
        tabs.addTab(files, FILES_VIEW[0])
        n_files = len(run_record.get("artifacts") or [])
        specs.append((*FILES_VIEW, files, lambda: f"{n_files} files"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs)
        self._init_views(tabs, specs)


@register_analysis_module_ui
class LocationExportModuleUI(ModuleUI):
    module_id = "location_export"
    purpose = FILES_VIEW[1]

    def config_widget(self) -> ConfigWidget:
        return _LocationExportConfig()

    def planned_views(self) -> list[tuple[str, str]]:
        return [PLOTS_VIEW, FILES_VIEW]

    def result_widget(self, experiment, run_record: dict) -> QWidget:
        return LocationResults(experiment, run_record)
