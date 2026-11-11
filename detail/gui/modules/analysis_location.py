"""Location-export analysis module UI (§16).

Config: a tracking-run picker offering tracking **specs** (§6.5) — a run's full-lineage
kind — found across the selected recordings, annotated with ``N/M`` coverage, several of which
may be ticked (``PooledSpecPicker``); ``config()`` emits ``tracking_spec`` or
``tracking_specs`` and the run resolves them to each recording's own tracking run. Result: the
per-recording CSV artifacts the run produced, read from its ``analysis.yaml`` manifest.
"""

from PyQt6.QtWidgets import (
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from detail.gui.help_ui import field_label
from detail.gui.modules._common import PooledSpecPicker
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    register_analysis_module_ui,
)


class _LocationExportConfig(ConfigWidget):
    def __init__(self) -> None:
        super().__init__()
        self._model_names: dict = {}  # {model_id: name}; filled by on_experiment_set
        form = QFormLayout(self)
        self._tracking = PooledSpecPicker("tracking")
        form.addRow(
            field_label("location_export", "tracking_run_id", "Tracking run:"),
            self._tracking.widget,
        )
        self._tracking.changed.connect(lambda _item: self.coverage_changed.emit())

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
        return self._tracking.selection()

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
