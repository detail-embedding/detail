"""A result page's data is reachable through some export target (#336).

**The class.** *A page grows a panel and the data behind it has no way out.* The rule is
"any analysis you show should be exportable", and the failure it prevents is a user who gets a
striking result and then cannot do anything with it except look at it inside the tool. Nothing
fails when a page drifts out of that
rule: the page renders, the numbers are right, and the export menu simply does not list them.

**Two drifts, two mechanisms, and only one of them needs a test.**

*The menu falling behind the catalogue* is structural — the shell builds the control from the
catalogue and no page hand-lists anything — so it cannot happen. *A page growing a panel with no
export* can, and this is that sweep, in the shape this repo already uses for discovering surfaces
rather than listing them (`test_empty_states.py` walks every `IndexTable`; `test_dialog_space.py`
discovers dialogs).

**The rule is deliberately the loose one:** the data *behind* a panel must be reachable, not that
each panel gets its own file. A cluster scatter of 40k points and `instances.csv` are the same
export. The strict reading generates exemptions faster than exports, and an allowlist nobody
believes is worse than no allowlist.
"""

import pytest

from detail.api import export_targets  # noqa: F401,E402  (registers the catalogue)
from detail.api import exporters
from detail.api.exporters import ExportSubject
from detail.gui import export as gui_export
from detail.gui.qt import QTableView, QTableWidget

pytestmark = pytest.mark.guard

#: Navigation kinds whose pages show derived results a user would want out, mapped to the
#: opener that shows them. A kind here with no export target fails below.
_RESULT_KINDS = ("recording", "run", "analysis_run", "experiment", "comparison")

#: A kind whose pages legitimately export nothing, with the reason. Kept short on purpose: an
#: allowlist that grows faster than the catalogue is how the rule stops meaning anything.
_EXEMPT: dict[str, str] = {}


def _panels(widget) -> list:
    """Tables and plots on a page — the things that show data a user might want out."""
    found = list(widget.findChildren(QTableView)) + list(
        widget.findChildren(QTableWidget)
    )
    try:
        from pyqtgraph import PlotWidget

        found += list(widget.findChildren(PlotWidget))
    except (
        ImportError
    ):  # pragma: no cover - pyqtgraph is a hard dep, this is belt and braces
        pass
    return found


@pytest.mark.parametrize("kind", _RESULT_KINDS)
def test_every_result_kind_has_an_export_target(kind):
    """A kind with no target is a page whose Export control can never appear.

    Checked against the catalogue rather than against a built page, so it holds for a kind whose
    page is expensive to construct — and it is the check that would have caught the reported
    bug, which was a whole subject kind with no way out.
    """
    if kind in _EXEMPT:
        pytest.skip(_EXEMPT[kind])
    have = [t for t in exporters.all_targets() if t.subject_kind == kind]
    assert have, (
        f"navigation kind {kind!r} declares no export target, so a user on one of its pages "
        f"has no way to get its data out. Add one to detail/api/export_targets.py, or an "
        f"_EXEMPT entry saying why its data needs no export."
    )


def test_the_analysis_result_pages_show_panels_and_can_export_them(
    qapp, authored_cohort
):
    """The reported bug, end to end: a page full of results and no way out.

    Builds the real workspace, counts what it draws, and asserts the catalogue answers for the
    subject that page is showing — rather than asserting the widget was constructed, which was
    true the whole time the capability was missing.
    """
    from detail.gui.cluster_workspace import ClusterWorkspace

    experiment = authored_cohort.experiment
    workspace = ClusterWorkspace(experiment, experiment.authored_run("an_01"))
    workspace.resize(1000, 800)
    workspace.show()
    qapp.processEvents()
    try:
        panels = _panels(workspace)
        assert panels, "the fixture draws no panels, so this asserts nothing"
        targets = exporters.targets_for(
            ExportSubject("analysis_run", experiment.dir, "an_01")
        )
        assert (
            targets
        ), f"the workspace draws {len(panels)} panel(s) and its subject exports nothing"
    finally:
        workspace.close()


def test_the_sweep_would_catch_a_kind_with_nothing_to_export():
    """Guard the guard: run the sweep's own assertion against a kind with no targets.

    Not "a fake kind has no targets", which is true by construction and proves nothing about
    the check — this drives the body of `test_every_result_kind_has_an_export_target` and
    requires it to raise.
    """
    with pytest.raises(AssertionError, match="declares no export target"):
        test_every_result_kind_has_an_export_target("a_kind_with_no_targets")


# --------------------------------------------------------------------------- #
# the shell places it — through the real door
# --------------------------------------------------------------------------- #
def test_opening_an_analysis_run_shows_the_export_control(qapp, authored_cohort):
    """Through the real opener, not by constructing the button.

    A control that works in a private harness and never appears where the user is standing is
    the exact failure this whole change exists to fix, so the test drives the shell.
    """
    from detail.gui.app import MainWindow

    window = MainWindow()
    window._open_experiment(authored_cohort.experiment)
    assert (
        not window._export.isHidden()
    ), "an experiment's overview exports its bundle and its metadata"
    window._open_analysis_run("an_01")
    assert (
        not window._export.isHidden()
    ), "the Export control did not appear on an analysis run's page"
    window.close()


def test_opening_a_distill_run_shows_the_export_control(qapp, tmp_path, monkeypatch):
    """A distill is the slowest thing many users start, and its page had no Export control —
    so its Performance report, which says how long labeling and training took and on what,
    was reachable from the CLI only. Driven through the real opener, like the test above.
    """
    from detail.api.experiment import Experiment
    from detail.gui.app import MainWindow
    from detail.storage import activity, distill_runs, locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = Experiment.create("p", tmp_path)
    distill_runs.write_run(
        exp.dir,
        "distill_001",
        {
            "id": "distill_001",
            "status": "complete",
            "stages": {
                "label": {"status": "complete", "dataset_id": "dataset_001"},
                "train": {"status": "complete", "model_id": "model_001"},
            },
        },
    )
    log = activity.ActivityLog(distill_runs.run_dir(exp.dir, "distill_001"))
    log.sample(activity.make_row(elapsed_s=0.0, unit="labeling", phase="labeling"))

    window = MainWindow()
    window._open_experiment(exp)
    window._open_distill_run("distill_001")
    assert (
        not window._export.isHidden()
    ), "the Export control did not appear on a distill run's page"
    window.close()


def test_leaving_a_page_clears_the_export_control(qapp, authored_cohort):
    """Fail closed: a page that sets no subject must not inherit the last one's.

    Otherwise a screen showing nothing exportable offers to export whatever was open before —
    which is worse than a missing button, because it writes the wrong thing without complaint.
    """
    from detail.gui.app import MainWindow

    window = MainWindow()
    window._open_experiment(authored_cohort.experiment)
    window._open_analysis_run("an_01")
    assert not window._export.isHidden()

    window._show_experiments_home()
    assert (
        window._export.isHidden()
    ), "the control survived a navigation to a page with no export subject"
    window.close()


def test_the_control_carries_the_page_on_screen(qapp, authored_cohort):
    """So the menu can ask it for a scope. A stale page means a stale selection."""
    from detail.gui.app import MainWindow

    window = MainWindow()
    window._open_experiment(authored_cohort.experiment)
    window._open_analysis_run("an_01")
    assert window._export._page is window._content.widget(0)
    assert (
        gui_export.scope_of(window._export._page) is None
    ), "nothing is selected yet, so the scope is everything"
    window.close()
