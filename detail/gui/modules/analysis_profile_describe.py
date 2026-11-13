"""Profile clusters — the analysis module UI (§16, §22). Its module id stays
``profile_describe``, so runs made before it lost its Describe step still open.

Config: the clustering run to read (a picker of the experiment's temporal-clustering
runs, filled off the UI thread), and every other setting from ``ProfileDescribeConfig`` — the
clusters and the profile's own settings — with the time the job should take on this machine,
estimated off the UI thread from the profile budget. It reads a run rather than a recording
selection, so the launch dialog hides its recording picker (``selects_recordings``). Opened from
the New analysis dialog, and from a clustering run's Interpret tab with that run already chosen.

*Run on* sends the job to a registered remote target instead (§6 "Remote analysis"); the
profiles come back and are placed beside the run here.

Result: one row per cluster — its status, how well its measures separate it, or why it stopped.
Each cluster's profile is the clustering run's, opened from its Interpret tab. A run recorded
before the job lost its Describe step carries a ``describe`` part per cluster; it is not shown.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.configs.profile_describe import ProfileDescribeConfig
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
from detail.gui.modules.registry import (
    ConfigWidget,
    ModuleUI,
    register_analysis_module_ui,
)
from detail.gui.workers import BackgroundCall

MODULE_ID = "profile_describe"
#: Fields built by hand below; every other field of the model is built from its metadata.
_BY_HAND = ("source_run_id", "node_ids")
#: One cluster's setting the batch inherits and never uses: its `compare` has no 'node'.
_NOT_A_BATCH_SETTING = ("against",)
_CLUSTERING_MODULES = ("temporal_clustering",)


def clustering_runs(experiment) -> list[tuple[str, str]]:
    """``(label, run id)`` for every temporal-clustering run — a scan of
    ``analysis/``, so it is called off the UI thread."""
    from detail.configs.describe import module_display_name

    out = []
    for ref in experiment.analysis_runs:  # off-thread (see the docstring)
        if ref.module_id not in _CLUSTERING_MODULES:
            continue
        name = f" — {ref.name}" if getattr(ref, "name", "") else ""
        out.append((f"{module_display_name(ref.module_id)} · {ref.id}{name}", ref.id))
    return out


class _ProfileDescribeConfig(ConfigWidget):
    config_model = ProfileDescribeConfig

    def __init__(self) -> None:
        super().__init__()
        self._experiment = None
        self._preferred: str | None = None
        self._lister: BackgroundCall | None = None
        self._estimator: BackgroundCall | None = None
        form = ConfigForm(self, ProfileDescribeConfig)

        self._run = QComboBox()
        self._run.setToolTip(field_meta(ProfileDescribeConfig, "source_run_id")["help"])
        self._run.addItem("(finding clustering runs…)", None)
        self._run.currentIndexChanged.connect(lambda _i: self._clear_estimate())
        form.addRow(
            field_label(MODULE_ID, "source_run_id", "Clustering run:"), self._run
        )
        self._where = add_run_on_row(
            form,
            "Where the job runs. A remote target profiles there, so this computer is free "
            "while it runs; the profiles come back and are placed beside the run here.",
        )

        self._fields: dict[str, QWidget] = {}
        for name in ProfileDescribeConfig.model_fields:
            if name in _BY_HAND or name in _NOT_A_BATCH_SETTING:
                continue
            widget = any_field_widget(ProfileDescribeConfig, name)
            if name == "n_per_side":  # unset is a choice of its own, not "work it out"
                widget.setSpecialValueText("time budget")
            self._fields[name] = widget
            label = field_meta(ProfileDescribeConfig, name)["label"]
            form.addRow(field_label(MODULE_ID, name, f"{label}:"), widget)

        self._node_ids = QLineEdit()
        self._node_ids.setPlaceholderText("e.g. 877, 962")
        self._node_ids.setToolTip(field_meta(ProfileDescribeConfig, "node_ids")["help"])
        form.addRow(
            field_label(
                MODULE_ID,
                "node_ids",
                f"{field_meta(ProfileDescribeConfig, 'node_ids')['label']}:",
            ),
            self._node_ids,
        )
        estimate_row = QWidget()
        row = QHBoxLayout(estimate_row)
        row.setContentsMargins(0, 0, 0, 0)
        self._estimate = QLabel("")
        self._estimate.setWordWrap(True)
        row.addWidget(self._estimate, 1)
        self._estimate_button = QPushButton("Estimate time")
        self._estimate_button.setToolTip(
            "How long this should take on this computer: each profile at its budget, or "
            "as long as this run's own profiles took."
        )
        self._estimate_button.clicked.connect(self.estimate)
        row.addWidget(self._estimate_button)
        form.addRow(estimate_row)
        form.finish()
        self._where.currentIndexChanged.connect(lambda _i: self._clear_estimate())

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
    def _node_list(self) -> list[int] | None:
        text = self._node_ids.text().replace(",", " ").split()
        try:
            return [int(t) for t in text]
        except ValueError:
            return None

    def config(self) -> dict:
        out = {
            name: any_field_value(ProfileDescribeConfig, name, widget)
            for name, widget in self._fields.items()
        }
        out["source_run_id"] = self._run.currentData()
        out["node_ids"] = self._node_list() or []
        compute = run_on_compute(self._where)
        if compute is not None:
            out["compute"] = compute
        return out

    def load_config(self, config: dict) -> None:
        for name, widget in self._fields.items():
            if name in config:
                set_any_field_value(ProfileDescribeConfig, name, widget, config[name])
        if "node_ids" in config:
            self._node_ids.setText(", ".join(str(n) for n in config["node_ids"] or []))
        select_run_on(self._where, config)

    def validation_error(self) -> str | None:
        if self._run.currentData() is None:
            return "Choose a clustering run to profile."
        if self._node_list() is None:
            return "Node ids are whole numbers, separated by commas."
        if self._fields["clusters"].currentData() == "ids" and not self._node_list():
            return "Clusters is 'ids': list at least one node id."
        return None

    # -- the estimate ----------------------------------------------------------------- #
    def _clear_estimate(self) -> None:
        self._estimate.setText("")

    def estimate(self) -> None:
        """Estimate the time off the UI thread (it reads the run's node cache)."""
        run_id = self._run.currentData()
        if self._experiment is None or run_id is None or self.validation_error():
            self._estimate.setText(self.validation_error() or "")
            return
        config = {k: v for k, v in self.config().items() if k != "source_run_id"}
        experiment = self._experiment
        self._estimate.setText("Estimating…")
        self._estimator = BackgroundCall(
            lambda: experiment.estimate_profile_describe(run_id, config),
            receiver=self,
        )  # off-thread
        self._estimator.done.connect(lambda guess: self._estimate.setText(guess.text()))
        self._estimator.failed.connect(lambda exc: self._estimate.setText(str(exc)))
        self._estimator.start()


@register_analysis_module_ui
class ProfileDescribeModuleUI(ModuleUI):
    module_id = MODULE_ID
    selects_recordings = False

    def config_widget(self) -> ConfigWidget:
        return _ProfileDescribeConfig()

    def result_widget(self, experiment, run_record: dict) -> QWidget:
        from detail.gui.pages import _table  # local: avoid an import cycle

        widget = QWidget()
        layout = QVBoxLayout(widget)
        source = run_record.get("source_run_id") or (
            run_record.get("params") or {}
        ).get("source_run_id", "")
        counts = run_record.get("counts") or {}
        summary = ", ".join(f"{n} {status}" for status, n in counts.items())
        text = QLabel(
            f"Clusters of {source}: {summary or 'none yet'}. Each cluster's profile is "
            f"saved beside {source}; open it from its Interpret tab."
        )
        text.setWordWrap(True)
        layout.addWidget(text)
        rows = []
        for c in run_record.get("clusters") or []:
            profile = c.get("profile") or {}
            accuracy = profile.get("accuracy")
            separated = (
                f"{accuracy:.0%}" + (" ✓" if profile.get("separated") else "")
                if isinstance(accuracy, float) and accuracy == accuracy
                else ""
            )
            rows.append(
                (
                    str(c.get("node", "")),
                    c.get("name") or "",
                    c.get("status", ""),
                    separated,
                    c.get("error") or "",
                )
            )
        if rows:
            layout.addWidget(
                _table(["Node", "Label", "Status", "Separated", "Error"], rows)
            )
        return widget
