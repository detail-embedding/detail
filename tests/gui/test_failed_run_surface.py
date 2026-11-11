"""A failed analysis run offers nothing, and a stale one offers the rebuild instead.

**The class.** *A surface that reads a run's own optimistic record instead of the evidence.*
An authored run (§21) writes ``status: open`` before projecting its artifact, so a run whose
artifact write died went on saying ``open`` — and every surface reads the record. The GUI
listed it beside healthy runs, opened its interpretation workspace, and the workspace's first
act on a run with no granularity is to **commit** one (``_InterpretTab._apply_loaded``), which
reprojects, which re-ran the sixty-recording artifact write that had just killed the run —
unattended, on the share it died on. Opening the run to look at it left it worse than it was.

**How it hides.** The page builds, the workspace builds, and every widget is correct for the
record it was handed. Nothing is wrong at the line where any of it is written.

**The rule.** The status is reconciled against ``job.json`` on the way in
(``core/dataclasses.py::reconciled_status``), and a failed run gets the error and *Delete*,
not results.

Asserted as the property, not as "the widget constructed". The positive control is the same
run with a healthy envelope, which must still open its workspace — without it "shows nothing"
passes for any reason at all. The other half, that opening a failed run writes nothing at all,
lives in ``test_cluster_workspace.py``: it needs the real fit, because the auto-commit that
did the damage only runs once the workspace has finished loading one.

The last case here is the sibling defect: a run whose *reprojection* died, whose settings and
results therefore describe different things. Its results are as misleading as a failed run's —
plausible numbers under the wrong heading — so the page offers the regeneration in their place.
"""

import json

import pytest

from detail.gui.cluster_workspace import ClusterWorkspace
from detail.gui.pages import AnalysisRunDetailPage
from detail.storage import analysis_runs as analysis_store


def _uninterpreted(cohort):
    """The shape a fit that died before choosing a granularity leaves — ``an_07``'s."""
    run_dir = cohort.run_dir()
    state = cohort.read_state()
    state.granularity = None
    cohort.write_state(state)
    return run_dir


def _envelope(run_dir, status: str) -> None:
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "job": {"job_id": run_dir.name, "config": {}},
                "progress": None,
                "error": (
                    "PermissionError: [WinError 32] The process cannot access the file"
                    if status == "failed"
                    else None
                ),
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def failed_run(authored_cohort):
    run_dir = _uninterpreted(authored_cohort)
    _envelope(run_dir, "failed")
    return authored_cohort, run_dir


def test_the_page_reports_the_failure_the_job_recorded(qapp, failed_run):
    cohort, run_dir = failed_run
    AnalysisRunDetailPage(cohort.experiment, "an_01")
    record = analysis_store.read_analysis_yaml(run_dir)
    assert record["status"] == "failed"
    assert "WinError 32" in str(record.get("error"))


def test_a_failed_run_does_not_offer_its_results(qapp, failed_run):
    cohort, _run_dir = failed_run
    page = AnalysisRunDetailPage(cohort.experiment, "an_01")
    assert not page.findChildren(ClusterWorkspace), (
        "the interpretation workspace opened on a run whose job failed — and its first act "
        "is to commit a granularity, which re-runs the write that killed it"
    )
    from PyQt6.QtWidgets import QPushButton

    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert "Open output folder" not in labels
    assert "Delete analysis" in labels


def test_the_same_run_with_a_healthy_job_still_opens_its_workspace(
    qapp, authored_cohort
):
    """The positive control. Without it, "no workspace" passes for any reason at all."""
    run_dir = authored_cohort.run_dir()
    _envelope(run_dir, "complete")
    page = AnalysisRunDetailPage(authored_cohort.experiment, "an_01")
    assert page.findChildren(ClusterWorkspace)
    assert analysis_store.read_analysis_yaml(run_dir)["status"] == "open"


def test_a_run_whose_results_are_out_of_date_offers_the_rebuild_instead(
    qapp, authored_cohort
):
    """The stale half: settings and results describe different things, so neither is shown.

    Distinct from the failed case in what it offers — there the run is over and the answer is
    *Delete*; here the fit is intact and one pass over the outputs puts the run right.
    """
    from PyQt6.QtWidgets import QPushButton

    run_dir = authored_cohort.run_dir()
    _envelope(run_dir, "complete")
    analysis_store.set_artifact_stale(run_dir, True)

    page = AnalysisRunDetailPage(authored_cohort.experiment, "an_01")

    assert not page.findChildren(
        ClusterWorkspace
    ), "the workspace opened on results the run itself says do not match its settings"
    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert "Rebuild results" in labels
