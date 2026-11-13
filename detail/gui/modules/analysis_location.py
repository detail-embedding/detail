"""Location-export analysis module UI (§16).

Config: a tracking-run picker offering tracking **specs** (§6.5) — a run's full-lineage
kind — found across every recording eligible for this module, annotated with ``N/M``
coverage, several of which may be ticked (``PooledSpecPicker``); ``config()`` emits
``tracking_spec`` or ``tracking_specs`` and the run resolves them to each recording's own
tracking run; its *Per recording…* button names a recording's run outright
(``tracking_overrides``). An optional keypoint track picker (first row: none) emits
``keypoint_track_spec``, narrowed to runs whose poses went to the ticked tracking run, with its
own *Per recording…* row (``keypoint_track_overrides``); a tick
asks for the mask-edge distance between tracks. Result: a **Plots** tab (``gui/location_plots.LocationPlotsPanel``, §16
"Location plots") and a **Files** tab listing the per-recording CSV artifacts the run produced,
read from its ``analysis.yaml`` manifest.
"""

from PyQt6.QtWidgets import (
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from detail.api import run_spec
from detail.configs.location_export import LocationExportConfig
from detail.gui._pydantic_form import field_meta, field_widget
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label
from detail.gui.modules._common import (
    PooledSpecPicker,
    RunOverrides,
    UpstreamSpecPicker,
    group_runs,
    lineage_fits,
    recording_names,
    spec_reads,
)
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    register_analysis_module_ui,
)

#: The keypoint picker's first row: export no keypoints.
NO_KEYPOINTS = "(none)"


class _LocationExportConfig(ConfigWidget):
    config_model = LocationExportConfig

    def __init__(self) -> None:
        super().__init__()
        self._model_names: dict = {}  # {model_id: name}; filled by on_experiment_set
        # ConfigForm routes a field the model marks advanced (the edge distance) under Advanced.
        form = ConfigForm(self, LocationExportConfig)
        self._tracking = PooledSpecPicker("tracking")
        form.addRow(
            field_label("location_export", "tracking_run_id", "Tracking run:"),
            self._tracking.panel,
        )
        self._tracking.changed.connect(lambda _item: self._sync_keypoint_lineage())
        self._tracking.changed.connect(lambda _item: self.coverage_changed.emit())
        # Optional: the first row is "no keypoints". Narrowed to keypoint track runs whose poses
        # went to the chosen tracking run, recording by recording — the run refuses any other.
        self._keypoints = UpstreamSpecPicker("keypoint_track", auto_label=NO_KEYPOINTS)
        self._keypoints.combo.setToolTip(
            field_meta(LocationExportConfig, "keypoint_run_id")["help"]
        )
        self._keypoints.combo.currentIndexChanged.connect(
            lambda _index: self.coverage_changed.emit()
        )
        form.addRow(
            field_label("location_export", "keypoint_run_id", "Keypoint track run:"),
            self._keypoints.combo,
        )
        # A recording whose tracking run is chosen per recording needs the keypoint track run
        # made on *that* run, which the keypoint settings above may not reach (§6.5).
        self._keypoint_overrides = RunOverrides("keypoint_track")
        self._keypoint_overrides.changed.connect(self.coverage_changed.emit)
        form.addRow("", self._keypoint_overrides.widget)
        self._tracking_groups: list[dict] = []
        self._keypoint_groups: list[dict] = []
        self._keypoint_names: dict[str, str] = {}
        self._keypoint_reads: dict = {}
        self._fields = {
            "pair_edge_distance": field_widget(
                LocationExportConfig, "pair_edge_distance"
            )
        }
        form.addRow(
            field_label(
                "location_export", "pair_edge_distance", "Distance between mask edges:"
            ),
            self._fields["pair_edge_distance"],
        )
        form.finish()

    def on_experiment_set(self, experiment) -> None:
        # A run record stores a model *id*; a picker has to show the name (§11). O(models),
        # not O(recordings), so it stays inside the UI-thread rule.
        self._model_names = experiment.model_names() if experiment is not None else {}

    def resolve_eligible_recordings(self, recordings: list) -> dict:
        """Off the UI thread (CLAUDE.md): every recording eligible for tracking, not just
        the checked ones — checking recordings one at a time otherwise locks this picker
        to whichever recording happened to be checked first (§6.5), since its own
        tracking runs are the only ones ``spec_groups`` would ever see. Resolved once per
        module pick rather than once per checkbox click.

        ``spec_groups`` walks all four processing modules per recording — several
        ``recording.yaml`` round trips per recording. Pinned, so the cache's per-property
        ``stat`` is paid once per recording rather than once per field asked for.
        """
        keypoint_groups = self._keypoints.resolve(recordings, self._model_names)
        return {
            "groups": self._tracking.resolve(recordings, self._model_names),
            "total": len(recordings),
            "names": recording_names(recordings),
            "keypoint_groups": keypoint_groups,
            "keypoint_reads": spec_reads(
                recordings, "keypoint_track", keypoint_groups, "tracking_run_id"
            ),
        }

    def apply_eligible_recordings(self, recordings: list, resolved: dict) -> None:
        """UI thread: install the resolved groups and repopulate. No disk."""
        names = resolved.get("names", {})
        self._tracking_groups = resolved["groups"]
        self._tracking.apply(resolved["groups"], resolved["total"], names=names)
        self._keypoint_reads = dict(resolved.get("keypoint_reads") or {})
        self._keypoint_groups = resolved.get("keypoint_groups") or []
        self._keypoints.apply(self._keypoint_groups, resolved["total"])
        self._keypoint_names = names
        self._sync_keypoint_lineage()

    def _sync_keypoint_lineage(self) -> None:
        """Offer each keypoint track setting only on the recordings where its run assigned its
        poses to the ticked tracking run there."""
        wanted: dict[str, str] = {}
        for spec in self._tracking.specs():
            wanted.update(group_runs(self._tracking_groups, spec))
        wanted.update(self._tracking.overrides.chosen())
        fits = lineage_fits(self._keypoint_reads, wanted) if wanted else None
        self._keypoints.restrict(fits, "its poses went to another tracking run")
        # The per-recording keypoint choice offers, on each recording, only the runs whose
        # poses went to the tracking run read there — the run refuses any other.
        groups = [
            {
                **group,
                "runs": {
                    sid: run
                    for sid, run in group["runs"].items()
                    if fits is None
                    or sid in fits.get(run_spec.freeze_spec(group["spec"]), set())
                },
            }
            for group in self._keypoint_groups
        ]
        self._keypoint_overrides.set_options(groups, self._keypoint_names)

    def covered_recording_ids(self) -> set[str] | None:
        tracked = self._tracking.covered()
        with_points = self._keypoints.covered()
        if tracked is None or with_points is None:
            return tracked
        return tracked & (with_points | set(self._keypoint_overrides.chosen()))

    def config(self) -> dict:
        config = dict(self._tracking.selection())
        spec = self._keypoints.spec()
        if spec is not None:
            config["keypoint_track_spec"] = spec
        config.update(self._keypoint_overrides.selection())
        config["pair_edge_distance"] = self._fields["pair_edge_distance"].isChecked()
        return config

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
        artifacts = run_record.get("artifacts", [])
        files = QWidget()
        files_layout = QVBoxLayout(files)
        files_layout.addWidget(
            QLabel(
                f"Where each animal is on every frame, written under "
                f"analysis/{run_id}/<recording>/:"
            )
        )
        if not artifacts:
            files_layout.addWidget(QLabel("(no output files recorded)"))
            layout.addWidget(files)
            return widget
        from detail.gui.location_plots import LocationPlotsPanel
        from detail.gui.pages import _table  # local: avoid an import cycle

        rows = [(a.get("path", ""), a.get("kind", "")) for a in artifacts]
        files_layout.addWidget(_table(["File", "Kind"], rows))
        # The tab is as tall as the Plots tab; the spare height goes below the list.
        files_layout.addStretch(1)
        tabs = QTabWidget()
        tabs.addTab(LocationPlotsPanel(experiment, run_id), "Plots")
        tabs.addTab(files, "Files")
        layout.addWidget(tabs)
        return widget
