"""Temporal-clustering ("Behavior clustering") module UI (§22) — the launch form.

Config: which recordings to pool a behavior vocabulary from, which embedding run to read, and
which method to fit with. Everything else is advanced and comes from the config models, so a bound
written once in ``configs/temporal_clustering.py`` and ``configs/temporal_dtw_finch.py`` reaches
the form, the CLI's ``--help`` and the parameter reference together (§6.5).

The **result** view is the shared interpretation workspace (``gui/cluster_workspace.py``), which
this module only has to hand a run and its vocabulary's nouns. Choosing a granularity, looking at
representatives and naming several clusters at once is the same job whether a cluster is a posture
or a behavior, so it lives once rather than once per module.
"""

from PyQt6.QtWidgets import QComboBox, QLabel, QWidget

from detail.configs.temporal_clustering import TemporalClusteringConfig
from detail.configs.temporal_codebook import TemporalCodebookConfig
from detail.configs.temporal_dtw_finch import TemporalDtwFinchConfig
from detail.gui._pydantic_form import (
    build_fields,
    optional_spin,
    optional_value,
    set_any_field_value,
)
from detail.gui.cluster_workspace import (
    TEMPORAL_TERMS,
    ClusterWorkspace,
    cluster_view_specs,
)
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label
from detail.gui.modules._common import PooledSpecPicker, pooled_settings_note
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    register_analysis_module_ui,
)

# The knobs each method surfaces, in the order they read. Keyed by method id, because the two
# methods have *disjoint* knob sets: one segments and compares bouts, the other builds a codebook
# and decodes. A single flat list would show a user the detector and the DTW band while they are
# configuring a method that has neither.
METHOD_FIELDS: dict[str, tuple[str, ...]] = {
    "dtw_finch": (
        "detector",
        "min_len_s",
        "band_s",
        "window_s",
        "pool",
        "feat_stride",
        "penalty",
        "percentile",
        "novelty_half_s",
        "abd_smooth_half_s",
        "abd_nms_window_s",
        "min_size_s",
        "max_gap_s",
        "max_segments",
    ),
    "codebook_viterbi": (
        "fit_share",
        "codewords",
        "min_run_s",
        "window_s",
        "max_gap_s",
        "seed",
    ),
}

METHOD_MODELS = {
    "dtw_finch": TemporalDtwFinchConfig,
    "codebook_viterbi": TemporalCodebookConfig,
}


def _advanced_flag(model_cls, name: str) -> bool | str | None:
    """The field's own ``advanced`` declaration, or ``None`` if it makes none."""
    extra = model_cls.model_fields[name].json_schema_extra or {}
    return extra.get("advanced")


class _TemporalClusteringConfig(ConfigWidget):
    """The launch form: recordings come from the dialog's selection, the rest from the models."""

    config_model = TemporalClusteringConfig

    def __init__(self) -> None:
        super().__init__()
        self._experiment = None
        self._recordings: list[str] = []

        form = ConfigForm(self, TemporalClusteringConfig)
        self._form = form
        module_rows = build_fields(TemporalClusteringConfig)
        self._fields = {name: widget for name, (_label, widget) in module_rows.items()}

        self._model_names: dict = {}  # {model_id: name}; filled by on_experiment_set
        self._embed = PooledSpecPicker("embedding")
        form.addRow(
            field_label("temporal_clustering", "embed_run_id", "Embedding run:"),
            self._embed.widget,
        )
        self._embed.changed.connect(self._on_embedding_ticked)
        form.addRow(
            field_label("temporal_clustering", "track_id", "Track:"),
            self._fields["track_id"],
        )

        self._method = QComboBox()
        self._method.addItems(self._method_ids())
        # The configured default, not whichever id sorts first: index 0 once launched a method
        # the CLI and API did not default to.
        self._method.setCurrentIndex(
            max(
                0,
                self._method.findText(
                    TemporalClusteringConfig.model_fields["method"].default
                ),
            )
        )
        form.addRow(
            field_label("temporal_clustering", "method", "Method:"), self._method
        )

        # Every method's rows are built once and the unselected one's are hidden, rather than the
        # form being rebuilt on each change: a rebuild would discard whatever the user had already
        # typed into the rows the two share.
        self._widgets: dict[str, dict] = {}
        self._rows: dict[str, list] = {}
        for method_id in self._method_ids():
            self._widgets[method_id] = self._build_method_fields(method_id)
            self._rows[method_id] = self._add_method_rows(form, method_id)

        self._method.currentTextChanged.connect(self._show_method_rows)
        self._show_method_rows(self._method.currentText())

        self._advice = QLabel("")
        self._advice.setWordWrap(True)
        form.addRow(self._advice)

    @staticmethod
    def _build_method_fields(method_id: str) -> dict:
        """Every widget one method offers, generated from its model where it can be.

        ``build_fields`` covers plain scalars; the enums and the ``X | None`` numbers are built
        here. Not optional extras: the CLI exposes them, and a knob reachable from one surface and
        not the other is the parity gap CLAUDE.md calls the bug.
        """
        model_cls = METHOD_MODELS[method_id]
        rows = build_fields(model_cls)
        widgets = {name: widget for name, (_label, widget) in rows.items()}
        labels = {name: label for name, (label, _w) in rows.items()}
        fields = model_cls.model_fields

        if "detector" in fields:
            detector = QComboBox()
            detector.addItems(["binseg", "novelty", "speed", "abd"])
            detector.setCurrentText(fields["detector"].default)
            detector.setToolTip(fields["detector"].description or "")
            widgets["detector"] = detector
            labels["detector"] = "Detector"
        if "pool" in fields:
            pool = QComboBox()
            pool.addItems(["mean", "max", "flatten"])
            pool.setCurrentText(fields["pool"].default)
            pool.setToolTip(fields["pool"].description or "")
            widgets["pool"] = pool
            labels["pool"] = "Window pooling"
        # "unset" means compare every bout / derive the gap from the cadence, which is what the
        # model's `None` means — so the affordance says that rather than showing a number the user
        # would have to know to ignore.
        if "max_segments" in fields:
            widgets["max_segments"] = optional_spin(
                model_cls, "max_segments", "every bout"
            )
            labels["max_segments"] = "Max bouts"
        if "max_gap_s" in fields:
            widgets["max_gap_s"] = optional_spin(
                model_cls, "max_gap_s", "from the embed cadence"
            )
            labels["max_gap_s"] = "Max tracking gap"
        return {"widgets": widgets, "labels": labels}

    def _add_method_rows(self, form, method_id: str) -> list:
        """Add one method's rows and return them, so they can be shown and hidden as a set.

        The row *index* is what `QFormLayout.setRowVisible` takes, and it is recorded per layout
        because a form has two — the main body and the advanced section — and a knob can be in
        either.
        """
        built = self._widgets[method_id]
        model_cls = METHOD_MODELS[method_id]
        added = []
        for name in METHOD_FIELDS[method_id]:
            if name not in built["widgets"]:
                continue
            advanced = _advanced_flag(model_cls, name)
            layout = (
                form.section(advanced if isinstance(advanced, str) else None)
                if advanced
                else form.main
            )
            before = layout.rowCount()
            form.addRow(
                field_label(method_id, name, f"{built['labels'][name]}:"),
                built["widgets"][name],
                advanced=advanced,
            )
            if layout.rowCount() > before:
                added.append((layout, layout.rowCount() - 1))
        return added

    def _show_method_rows(self, method_id: str) -> None:
        """Show the selected method's knobs and hide every other method's."""
        for other, rows in self._rows.items():
            for layout, index in rows:
                layout.setRowVisible(index, other == method_id)

    @staticmethod
    def _method_ids() -> list[str]:
        from detail.cluster_methods import import_temporal_cluster_methods
        from detail.core.registries import TEMPORAL_CLUSTER_METHODS

        import_temporal_cluster_methods()
        return sorted(TEMPORAL_CLUSTER_METHODS)

    def on_experiment_set(self, experiment) -> None:
        self._experiment = experiment
        # A run record stores a model *id*; a picker has to show the name (§11).
        self._model_names = experiment.model_names() if experiment is not None else {}

    def resolve_recordings(self, recordings: list) -> dict:
        """Off the UI thread: the embedding settings across the selection (§6.5)."""
        return {"groups": self._embed.resolve(recordings, self._model_names)}

    def apply_recordings(self, recordings: list, resolved: dict) -> None:
        """Offer each set of embedding settings, with how many recordings have a run of it.

        Settings, not the run ids every selected recording shares: an id is a per-recording
        counter, so a recording that was retried or embedded later holds the same settings under
        a different id — and was refused, with every setting identical (#547).
        """
        self._recordings = [r.id for r in recordings]
        self._embed.apply(resolved["groups"], len(recordings))
        self._refresh_advice()

    def on_recordings_changed(self, recordings: list) -> None:
        """Synchronous path for direct callers (the launch dialog uses resolve→apply)."""
        self.apply_recordings(recordings, self.resolve_recordings(recordings))

    def covered_recording_ids(self) -> set[str] | None:
        return self._embed.covered()

    def _covered_recordings(self) -> list[str]:
        covered = self._embed.covered()
        return [sid for sid in self._recordings if covered is None or sid in covered]

    def _on_embedding_ticked(self, *_args) -> None:
        """A tick changes which recordings are covered, so the dialog re-narrows its list."""
        self._refresh_advice()
        self.coverage_changed.emit()

    def _refresh_advice(self) -> None:
        if not self._recordings:
            self._advice.setText("")
        elif not self._embed.specs():
            self._advice.setText(
                "These recordings have no completed embedding run. Embed them first."
            )
        else:
            self._advice.setText(
                f"One vocabulary pooled across {len(self._covered_recordings())} "
                f"recording(s){pooled_settings_note(len(self._embed.specs()))}. "
                "Comparing every pair of bouts is the cost of a fit and it grows with the "
                "square of their number — set Max bouts under Advanced to bound it, and the "
                "run will say what it left out."
            )

    def load_config(self, config: dict) -> None:
        """Restore a previous fit's settings — for Refine and the settings picker.

        The method is selected *first*: the two methods have disjoint knob sets, so writing a
        parameter before its method is showing would write it into the other method's form or
        into nothing at all.

        ``source_recording_ids`` and ``embedding_spec(s)`` are *selections* the dialog re-asks
        for.
        """
        method_id = config.get("method")
        if method_id in self._widgets:
            self._method.setCurrentText(method_id)
        super().load_config(config)  # track_id, from the model
        widgets = self._widgets[self._method.currentText()]["widgets"]
        model_cls = METHOD_MODELS[self._method.currentText()]
        for name, value in (config.get("method_params") or {}).items():
            widget = widgets.get(name)
            if widget is None:
                continue  # a knob this method no longer declares
            if isinstance(widget, QComboBox):
                if widget.findText(str(value)) >= 0:
                    widget.setCurrentText(str(value))
            else:
                set_any_field_value(model_cls, name, widget, value)

    def config(self) -> dict:
        method_id = (
            self._method.currentText()
            or TemporalClusteringConfig.model_fields["method"].default
        )
        widgets = self._widgets[method_id]["widgets"]
        return {
            # Narrowed to the recordings the chosen settings cover, as the launch dialog
            # narrows its own list: this key wins over the dialog's when the two merge.
            "source_recording_ids": self._covered_recordings(),
            **self._embed.selection(),
            "track_id": self._fields["track_id"].value(),
            "method": method_id,
            # Only the selected method's knobs. Sending another method's would be sending
            # parameters its config model has never heard of, which `validate` would reject.
            "method_params": {
                name: self._value_of(widgets, name)
                for name in METHOD_FIELDS[method_id]
                if name in widgets
            },
        }

    @staticmethod
    def _value_of(widgets: dict, name: str):
        """One method widget's value.

        Combos carry text. The two `X | None` boxes report their "unset" position as ``None``,
        which is passed through rather than replaced by a number — ``None`` is what tells a method
        to compare every bout or to derive the gap, so substituting the floor would silently
        change the run.
        """
        widget = widgets[name]
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if name in ("max_segments", "max_gap_s"):
            return optional_value(widget)
        return widget.value()

    def validation_error(self) -> str | None:
        if not self._recordings:
            return "Select at least one recording to fit the vocabulary on."
        return self._embed.validation_error()


@register_analysis_module_ui
class TemporalClusteringModuleUI(ModuleUI):
    module_id = "temporal_clustering"
    mutable = True

    def config_widget(self) -> ConfigWidget:
        return _TemporalClusteringConfig()

    def planned_views(self) -> list[tuple[str, str]]:
        return cluster_view_specs(TEMPORAL_TERMS)

    def result_widget(self, experiment, run_record: dict) -> QWidget:
        return ClusterWorkspace(
            experiment,
            experiment.temporal_clustering_run(run_record.get("id", "")),
            TEMPORAL_TERMS,
        )

    def snapshot_widget(
        self, experiment, run_record: dict, snapshot_id: str
    ) -> QWidget:
        """The same workspace, driven by the run as it was (§21).

        One argument different from ``result_widget``, and deliberately not a second widget: the
        workspace asks a run handle for reads and refuses to write when it says it is not editable,
        so a frozen interpretation renders through exactly the code that renders a live one. A
        parallel read-only screen is what would let the two drift while both look right.
        """
        run = experiment.temporal_clustering_run(run_record.get("id", ""))
        return ClusterWorkspace(experiment, run.view(snapshot_id), TEMPORAL_TERMS)
