"""Find look-alike clusters — the analysis module UI (§16, §22 "Look-alike clusters").

Config: the clustering run to read (a picker of the experiment's temporal-clustering
runs, filled off the UI thread), and every other setting from ``ClusterLookalikesConfig``. It
reads a run rather than a recording selection, so the launch dialog hides its recording picker
(``selects_recordings``). Opened from the New analysis dialog, and from a clustering run's
Interpret tab (*Look-alike clusters…* ▸ *Find look-alikes…*) with that run already chosen.

*Pairs to compare* is every pair (the default) or only labelled clusters against the rest. Run
with every pair asks first (``launch_question``): how many pairs, about how long, and how many
the labelled mode would compare — counted off the UI thread, since it reads the node cache.

*Run on* sends the job to a registered remote target instead (§6 "Remote analysis"), which keeps
this computer free for the hours a run's pairs can take; the comparisons are placed beside the
run here once they are copied back.

Result: one row per pair — its status, how well body measures and place alone tell it apart,
and whether it looks alike. The groups are the clustering run's, shown in its Interpret tab.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from detail.configs.cluster_lookalikes import PAIRS_ALL, ClusterLookalikesConfig
from detail.gui._pydantic_form import (
    any_field_value,
    any_field_widget,
    field_meta,
    set_any_field_value,
)
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label
from detail.gui.modules._common import (
    add_run_on_row,
    repopulate_combo,
    run_on_compute,
    select_run_on,
)
from detail.gui.modules.analysis_profile_describe import clustering_runs
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    register_analysis_module_ui,
)
from detail.gui.space_prompt import LaunchQuestion
from detail.gui.workers import BackgroundCall

MODULE_ID = "cluster_lookalikes"
#: Fields built by hand below; every other field of the model is built from its metadata.
_BY_HAND = ("source_run_id",)


class _ClusterLookalikesConfig(ConfigWidget):
    config_model = ClusterLookalikesConfig

    def __init__(self) -> None:
        super().__init__()
        self._experiment = None
        self._preferred: str | None = None
        self._lister: BackgroundCall | None = None
        form = ConfigForm(self, ClusterLookalikesConfig)

        self._run = QComboBox()
        self._run.setToolTip(
            field_meta(ClusterLookalikesConfig, "source_run_id")["help"]
        )
        self._run.addItem("(finding clustering runs…)", None)
        form.addRow(
            field_label(MODULE_ID, "source_run_id", "Clustering run:"), self._run
        )
        self._where = add_run_on_row(
            form,
            "Where the job runs. A remote target compares the pairs there, so this "
            "computer is free for the many hours the pairs can take; the comparisons come back "
            "and are placed beside the run here.",
        )
        self._fields: dict[str, QWidget] = {}
        for name in ClusterLookalikesConfig.model_fields:
            if name in _BY_HAND:
                continue
            widget = any_field_widget(ClusterLookalikesConfig, name)
            if name == "n_per_side":  # unset is a choice of its own, not "work it out"
                widget.setSpecialValueText("time budget")
            self._fields[name] = widget
            label = field_meta(ClusterLookalikesConfig, name)["label"]
            form.addRow(field_label(MODULE_ID, name, f"{label}:"), widget)
        form.finish()

    # -- launch-dialog hooks ------------------------------------------------------------ #
    def on_experiment_set(self, experiment) -> None:
        self._experiment = experiment
        if experiment is None:
            return
        self._lister = BackgroundCall(
            lambda: clustering_runs(experiment), receiver=self
        )  # off-thread: a scan of analysis/
        self._lister.done.connect(self._show_runs)
        self._lister.start()

    def _show_runs(self, runs: list) -> None:
        repopulate_combo(
            self._run, runs or [("(no clustering run — fit one first)", None)]
        )
        if self._preferred is not None:
            index = self._run.findData(self._preferred)
            if index >= 0:
                self._run.setCurrentIndex(index)

    def prefer_selection(self, selection: dict) -> None:
        """The clustering run the opener is showing (its Interpret tab)."""
        run_id = selection.get("source_run_id")
        if run_id is None:
            return
        self._preferred = str(run_id)
        index = self._run.findData(self._preferred)
        if index >= 0:
            self._run.setCurrentIndex(index)

    # -- the config ------------------------------------------------------------------- #
    def config(self) -> dict:
        out = {
            name: any_field_value(ClusterLookalikesConfig, name, widget)
            for name, widget in self._fields.items()
        }
        out["source_run_id"] = self._run.currentData()
        compute = run_on_compute(self._where)
        if compute is not None:
            out["compute"] = compute
        return out

    def load_config(self, config: dict) -> None:
        for name, widget in self._fields.items():
            if name in config:
                set_any_field_value(ClusterLookalikesConfig, name, widget, config[name])
        select_run_on(self._where, config)

    def validation_error(self) -> str | None:
        if self._run.currentData() is None:
            return "Choose a clustering run whose clusters to compare."
        return None

    def launch_question(self, experiment, config: dict):
        """Every pair is compute intensive: say how many pairs and about how long, and what
        the labelled mode would compare, before it starts. Labelled asks nothing."""
        if config.get("pairs") != PAIRS_ALL or experiment is None:
            return None
        run_id = config["source_run_id"]
        return lambda: every_pair_question(experiment, run_id, config)


def every_pair_question(experiment, run_id: str, config: dict) -> LaunchQuestion | None:
    """The question before comparing every pair of ``run_id`` (off the UI thread: it reads
    the node cache), or ``None`` when none is left to compare: no pair at all, which the job's
    own refusal explains, or every pair already compared."""
    counts = experiment.count_lookalike_pairs(run_id, config)  # off-thread (run_busy)
    if not counts.to_compare(PAIRS_ALL):
        return None
    return LaunchQuestion(f"{counts.warning()}\n\nRun every pair?", "Run every pair")


def _percent(value) -> str:
    return f"{value:.0%}" if isinstance(value, float) and value == value else ""


@register_analysis_module_ui
class ClusterLookalikesModuleUI(ModuleUI):
    module_id = MODULE_ID
    selects_recordings = False

    def config_widget(self) -> ConfigWidget:
        return _ClusterLookalikesConfig()

    def result_widget(self, experiment, run_record: dict) -> QWidget:
        from detail.gui.pages import _table  # local: avoid an import cycle

        widget = QWidget()
        layout = QVBoxLayout(widget)
        source = run_record.get("source_run_id") or (
            run_record.get("params") or {}
        ).get("source_run_id", "")
        counts = run_record.get("counts") or {}
        summary = ", ".join(f"{n} {status}" for status, n in counts.items())
        groups = run_record.get("groups") or []
        grouped = (
            "; ".join(
                f"{len(g.get('nodes') or [])} clusters"
                + (f" “{g['name']}”" if g.get("name") else "")
                for g in groups
            )
            or "none"
        )
        text = QLabel(
            f"Pairs of {source}: {summary or 'none yet'}. "
            f"{run_record.get('looks_alike', 0)} compared here look alike. Groups when it "
            f"ended: {grouped}. Each comparison is saved beside {source}; its Interpret tab "
            "shows the groups under Look-alike clusters."
        )
        text.setWordWrap(True)
        layout.addWidget(text)
        rows = []
        for p in run_record.get("pairs") or []:
            alike = p.get("looks_alike")
            rows.append(
                (
                    f"#{p.get('a', '')}",
                    f"#{p.get('b', '')}",
                    p.get("status", ""),
                    _percent(p.get("accuracy")),
                    _percent(p.get("place_accuracy")),
                    "" if alike is None else ("yes" if alike else "no"),
                    p.get("error") or "",
                )
            )
        if rows:
            layout.addWidget(
                _table(
                    [
                        "Cluster",
                        "Against",
                        "Status",
                        "Body measures",
                        "Place alone",
                        "Look alike",
                        "Why",
                    ],
                    rows,
                )
            )
        return widget
