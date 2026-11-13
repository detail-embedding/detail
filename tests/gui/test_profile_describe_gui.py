"""Profile clusters and Duplicate analysis, reached the way a user reaches them.

- **A module that reads a run, not recordings.** Its launch dialog hides the recording picker
  and launches without a recording selection, the form naming the clustering run; a dialog
  that still asked for recordings would refuse to run ("Select at least one recording").
- **The way in from a clustering run.** The Interpret tab's button opens that dialog with the
  run already chosen.
- **A form that opens on the model's defaults.** The batch's fixed count per side is 50; a
  form that opened on "time budget" would launch the other behaviour with nothing to say so.
- **Duplicate from the run page** copies the run off the UI thread and opens the copy.
- **The result view** lists every cluster from the run's record, a record written before the
  job lost its Describe step included (its ``describe`` parts are not shown).
"""

from __future__ import annotations

import pytest

from tests.gui.conftest import drain_until


@pytest.fixture
def clustered(fitted_temporal):
    experiment, run_id = fitted_temporal
    experiment.temporal_clustering_run(run_id).rebuild_node_cache()
    return experiment, run_id


def _dialog(experiment, run_id):
    from detail.gui.launch import LaunchAnalysisDialog

    return LaunchAnalysisDialog(
        experiment,
        seed={"module_id": "profile_describe", "params": {}},
        selection={"source_run_id": run_id},
    )


def test_the_dialog_hides_recordings_and_launches_on_the_chosen_run(
    qapp, clustered, monkeypatch
):
    experiment, run_id = clustered
    dialog = _dialog(experiment, run_id)
    dialog.show()
    form = dialog.config_widget()
    assert drain_until(qapp, lambda: form.config()["source_run_id"] == run_id)
    assert dialog._recording_picker.isHidden()
    config = form.config()
    assert config["n_per_side"] == 50, "the form did not open on the model's default"
    assert "n_clips" not in config and "backend" not in config
    launched: list = []

    def fake_run(module_id, cfg):
        launched.append((module_id, cfg))
        raise ValueError("stop here")

    monkeypatch.setattr(experiment, "run_analysis", fake_run)
    monkeypatch.setattr(
        "detail.gui.launch.QMessageBox.critical", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        "detail.gui.launch.QMessageBox.warning", lambda *a, **k: None, raising=False
    )
    dialog._run()
    assert drain_until(qapp, lambda: bool(launched))
    module_id, cfg = launched[0]
    assert module_id == "profile_describe"
    assert cfg["source_run_id"] == run_id and "source_recording_ids" not in cfg
    dialog.close()


def test_a_module_that_reads_recordings_still_shows_the_picker(qapp, clustered):
    experiment, run_id = clustered
    dialog = _dialog(experiment, run_id)
    index = dialog._module_combo.findData("location_export")
    dialog._module_combo.setCurrentIndex(index)
    assert not dialog._recording_picker.isHidden()
    dialog.close()


def test_the_interpret_tab_opens_the_dialog_on_its_run(qapp, clustered, monkeypatch):
    from detail.gui.cluster_workspace import POSE_TERMS, ClusterWorkspace
    from detail.gui.launch import LaunchAnalysisDialog

    experiment, run_id = clustered
    run = experiment.temporal_clustering_run(run_id)
    workspace = ClusterWorkspace(experiment, run, POSE_TERMS)
    tab = workspace._interpret
    assert drain_until(qapp, lambda: tab._view is not None)
    (batch,) = [
        a for a in tab.cluster_tools_menu().actions() if a.text() == "Profile clusters…"
    ]
    assert batch.isEnabled(), batch.toolTip()
    opened: list = []

    def fake_exec(self):
        opened.append(self.config_widget())
        return 0

    monkeypatch.setattr(LaunchAnalysisDialog, "exec", fake_exec)
    batch.trigger()
    (form,) = opened
    assert drain_until(qapp, lambda: form.config()["source_run_id"] == run_id)


def test_duplicate_on_the_run_page_copies_and_opens_the_copy(qapp, clustered):
    from detail.gui.pages import AnalysisRunDetailPage

    experiment, run_id = clustered
    opened: list = []
    page = AnalysisRunDetailPage(experiment, run_id, on_open=opened.append)
    page.show()
    page._duplicate(name="trial")
    assert drain_until(qapp, lambda: bool(opened), timeout_ms=30_000)
    record = experiment.analysis_record(opened[0])
    assert record["name"] == "trial" and record["duplicated_from"] == run_id


def test_the_result_view_lists_every_cluster_of_an_old_record_too(qapp, clustered):
    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES
    from PyQt6.QtWidgets import QTableView, QTableWidget

    experiment, run_id = clustered
    record = {
        "id": "an_09",
        "module_id": "profile_describe",
        "source_run_id": run_id,
        "counts": {"complete": 1, "failed": 1},
        "clusters": [
            {"node": 5, "name": "rear", "status": "complete",
             "profile": {"accuracy": 0.8, "separated": True},
             "describe": {"names": ["rearing"]}},
            {"node": 6, "name": "", "status": "failed", "error": "describe: stalled",
             "profile": {}, "describe": {}},
        ],
    }  # fmt: skip
    widget = ANALYSIS_GUI_MODULES["profile_describe"].result_widget(experiment, record)
    tables = widget.findChildren(QTableView) + widget.findChildren(QTableWidget)
    assert tables
    model = tables[0].model()
    assert model.rowCount() == 2
    cells = {
        model.index(r, c).data()
        for r in range(model.rowCount())
        for c in range(model.columnCount())
    }
    assert {"describe: stalled", "80% ✓"} <= cells
    assert "rearing" not in cells, "an old record's suggested names are not shown"
