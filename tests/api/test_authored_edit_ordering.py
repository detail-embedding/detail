"""An authored edit never leaves a durable state its reprojection has not backed.

**The class.** *A durable commit ordered before the operation that makes it true.*
``AuthoredRun._edit`` wrote the state, logged the history entry, and only then reprojected
the §19 artifact. A failure at the third step left the first two standing: the run's state
said ``granularity: 138`` while its artifact said ``granularity: null`` with an empty
vocabulary, and the only trace of the attempt was a history line asserting it had worked.

**How it hides.** Nothing raises where the ordering is written, and every file involved
stays well-formed — a reader that trusts either one is simply wrong about the other, and no
surface compares them. It reached a real experiment: a pose-clustering run whose artifact
write had died on a network share was opened in the GUI, the workspace auto-committed an
opening granularity, the reprojection wrote *nothing at all*, and the run came out of it
strictly worse than it went in.

**The rule.** §19 calls the frame view a materialized view and says regenerate rather than
reconcile — so the disagreement does not have to be prevented, but it does have to be
*recorded*. The stale mark goes down before the state is written and comes up only when the
reprojection lands, which also covers the kill an ``except`` cannot see;
``AuthoredRun.reproject`` (``detail analysis rebuild``, the GUI's *Rebuild results*) is the
way back.

The tests assert the **files**, never that the call raised: "it raised" is a regression test
for the exception, and the defect is what is left on disk afterwards.

**The second half (#24): a recorded disagreement that nothing downstream reads.** The mark went
down and stayed down, but only the next edit, ``analysis status`` and the run page read it. A
query run's cutoff moved to 0.6, its rebuild died, and ``export-query`` then pinned a snapshot
recording cutoff 0.6 over instances cut at 0.5; on a clustering run ``measures export``,
``export-temporal-clustering`` and ``--finalize`` all succeeded on the old instances. The error
itself ("no embedded rows for track 0 …") named no recording and did not say the change had been
saved, and INTERPRET titled it "Label not saved" over a label that was on disk.

**The rule.** Every reader of a run's output refuses through one check,
``analysis_runs.refuse_if_stale``, reached through ``AuthoredRun.bring_frame_views_current`` (which
a pin — and so finalize, both exports and a comparison — and measures already call first), and
called directly by the reader that does not read frame views: a clip-review census (a new
one-by-one audit batch was the other, until that audit was removed). The edit's own failure is a ``RebuildFailedError`` that says the change was
saved, names the rebuild, and keeps the real failure as its cause. The refusal tests below
assert that nothing was written (no snapshot, no export, no census), not only that the call
raised.
"""

import pytest

from detail.api.temporal_clustering import TemporalClusteringRun
from detail.storage import analysis_runs as analysis_store
from detail.storage import temporal_runs


# A label on a node: it funnels through `_edit` and reprojects, and the ordering under test is
# `_edit`'s, which is shared.
#
# The node is one of the root's children, read off the run's own tree, so the label covers real
# members and the assertions below compare a resolution that moved rather than two empty maps.
def _an_edit(run) -> dict:
    tree = run.hierarchy()
    return {str(int(tree.children[tree.n_nodes - 1 - tree.n_leaves][0])): "rearing"}


class _Boom(RuntimeError):
    """The reprojection dying — a transient share write, in one exception."""


def _break_projection(monkeypatch):
    """Patched on the *concrete* class, since that is the one that overrides ``_project``.

    Patching the base passes the assignment and changes nothing — the subclass's own method
    still wins, and the test then measures the real projection failing for its own reason.
    """

    def explode(self, *, progress=None, reuse=False):
        raise _Boom("the artifact write died")

    monkeypatch.setattr(TemporalClusteringRun, "_project", explode)


#: What an edit raises when its reprojection dies: the change is saved, so the error says so and
#: names the rebuild (#24); the real failure is its ``__cause__``.
_EditBoom = analysis_store.RebuildFailedError


def _silence_projection(monkeypatch):
    monkeypatch.setattr(TemporalClusteringRun, "_project", lambda self, **_: None)


def test_a_failed_reprojection_leaves_the_disagreement_recorded(
    authored_cohort, monkeypatch
):
    """The state moves — it has to, the edit is what the user asked for — but the run says so."""
    cohort = authored_cohort
    run = cohort.run()
    assert not run.artifact_stale

    edit = _an_edit(run)
    _break_projection(monkeypatch)
    with pytest.raises(_EditBoom):
        run.name_nodes(edit)

    written = cohort.read_state().node_names
    assert written, "the state must have moved, or the rest of this asserts nothing"
    assert written == {str(k): v for k, v in edit.items()}
    assert "rearing" not in (
        analysis_store.read_analysis_yaml(cohort.run_dir()).get("vocabulary") or []
    )
    assert run.artifact_stale, (
        "the state and the artifact disagree and nothing on disk says so — a reader that "
        "trusts either file is wrong about the other"
    )


def test_the_error_of_a_failed_reprojection_says_the_edit_was_saved(
    authored_cohort, monkeypatch
):
    """The bare cause ("no embedded rows for track 0") read as the edit having failed, and the
    GUI titled it "Label not saved" — while the label was on disk (#24)."""
    cohort = authored_cohort
    run = cohort.run()
    _break_projection(monkeypatch)
    with pytest.raises(analysis_store.RebuildFailedError) as info:
        run.name_nodes(_an_edit(run))
    message = str(info.value)
    assert "the change was saved" in message
    assert f"detail analysis rebuild {run.id}" in message
    assert "the artifact write died" in message, "the real cause is still named"
    assert isinstance(info.value.__cause__, _Boom)


def test_the_mark_is_down_before_the_state_is_written(authored_cohort, monkeypatch):
    """Not in an ``except``: a SIGKILL between the two writes leaves the same disagreement.

    Observed at the moment the state is rewritten, which is the only window a kill could
    land in.
    """
    cohort = authored_cohort
    seen = {}

    real = temporal_runs.write_temporal_state

    def watch(run_dir, state):
        seen["stale_at_state_write"] = analysis_store.artifact_stale(run_dir)
        return real(run_dir, state)

    monkeypatch.setattr(temporal_runs, "write_temporal_state", watch)
    _break_projection(monkeypatch)
    with pytest.raises(_EditBoom):
        cohort.run().name_nodes(_an_edit(cohort.run()))

    assert seen["stale_at_state_write"] is True


def test_a_successful_edit_leaves_nothing_marked(authored_cohort, monkeypatch):
    """The mark means something only if it is not always on."""
    cohort = authored_cohort
    _silence_projection(monkeypatch)
    cohort.run().name_nodes(_an_edit(cohort.run()))
    assert not cohort.run().artifact_stale


def test_rebuilding_clears_the_mark(authored_cohort, monkeypatch):
    """The way back, through the call the CLI and the GUI both drive."""
    cohort = authored_cohort
    _break_projection(monkeypatch)
    with pytest.raises(_EditBoom):
        cohort.run().name_nodes(_an_edit(cohort.run()))
    assert cohort.run().artifact_stale

    _silence_projection(monkeypatch)
    cohort.run().reproject()
    assert not cohort.run().artifact_stale


def test_a_failed_rebuild_stays_marked(authored_cohort, monkeypatch):
    """A repair that does not land must not report itself as one."""
    cohort = authored_cohort
    _break_projection(monkeypatch)
    with pytest.raises(_Boom):
        cohort.run().reproject()
    assert cohort.run().artifact_stale


# --- what the mark licenses -------------------------------------------------------------------
#
# A clear mark is also what lets an edit take the cheap path (#325): carry positions by instance
# id and defer the frame views to their first reader. A run left stale by a failed edit gets the
# full rebuild instead, since nothing it left behind is trusted. So the mark decides `reuse`, read
# *before* the edit puts its own mark down: after that it is always set, and reading it there would
# never take the cheap path.


def _record_reuse(monkeypatch) -> list:
    seen: list = []

    def record(self, *, progress=None, reuse=False):
        seen.append(reuse)

    monkeypatch.setattr(TemporalClusteringRun, "_project", record)
    return seen


def test_an_edit_on_an_agreeing_artifact_keeps_what_it_did_not_change(
    authored_cohort, monkeypatch
):
    cohort = authored_cohort
    seen = _record_reuse(monkeypatch)
    cohort.run().name_nodes(_an_edit(cohort.run()))
    assert seen == [True]


def test_an_edit_after_a_failed_one_rebuilds_everything(authored_cohort, monkeypatch):
    """A killed reprojection may have left any view half-written, so none of them is trusted."""
    cohort = authored_cohort
    _break_projection(monkeypatch)
    with pytest.raises(_EditBoom):
        cohort.run().name_nodes(_an_edit(cohort.run()))
    assert cohort.run().artifact_stale

    seen = _record_reuse(monkeypatch)
    cohort.run().name_nodes(_an_edit(cohort.run()))
    assert seen == [False]
    assert not cohort.run().artifact_stale


def test_rebuilding_keeps_nothing(authored_cohort, monkeypatch):
    """The repair must not trust what it repairs, stale or not."""
    cohort = authored_cohort
    seen = _record_reuse(monkeypatch)
    cohort.run().reproject()
    assert seen == [False]


# --- what the mark refuses (#24) --------------------------------------------------------------
#
# The mark went down and stayed down, and nothing but the next edit, `analysis status` and the run
# page read it. Every reader of the run's output — a pin, and so finalize and both exports and a
# comparison; measures; a clip-review census — took the stale artifact as
# current: an export pinned a snapshot recording cutoff 0.6 over instances cut at 0.5. Each reader
# below must refuse, naming the command that rebuilds.

_REBUILD = "detail analysis rebuild"


def _left_stale(cohort, monkeypatch):
    """A run whose last edit was saved and whose reprojection died. Returns the run."""
    _break_projection(monkeypatch)
    with pytest.raises(_EditBoom):
        cohort.run().name_nodes(_an_edit(cohort.run()))
    monkeypatch.undo()
    run = cohort.run()
    assert (
        run.artifact_stale
    ), "the fixture must leave the run stale, or nothing is tested"
    return run


def _refuses(call, run_id: str) -> None:
    with pytest.raises(analysis_store.ResultsOutOfDateError) as info:
        call()
    message = str(info.value)
    assert "was saved" in message
    assert f"{_REBUILD} {run_id}" in message


def test_a_pin_of_a_stale_run_refuses_and_writes_no_snapshot(
    authored_cohort, monkeypatch
):
    from detail.storage import snapshots as snapshots_store

    cohort = authored_cohort
    run = _left_stale(cohort, monkeypatch)
    before = snapshots_store.discover_snapshots(cohort.run_dir())
    _refuses(lambda: run.pin(reason="export"), run.id)
    assert snapshots_store.discover_snapshots(cohort.run_dir()) == before


def test_finalizing_a_stale_run_refuses_and_leaves_it_open(
    authored_cohort, monkeypatch
):
    cohort = authored_cohort
    run = _left_stale(cohort, monkeypatch)
    _refuses(run.finalize, run.id)
    assert analysis_store.read_analysis_yaml(cohort.run_dir()).get("status") != (
        "complete"
    )


def test_exporting_a_stale_clustering_run_refuses(
    authored_cohort, monkeypatch, tmp_path
):
    cohort = authored_cohort
    run = _left_stale(cohort, monkeypatch)
    out = tmp_path / "export"
    _refuses(lambda: run.export(out), run.id)
    assert not any(out.glob("*.csv")) if out.exists() else True


def test_measuring_a_stale_run_refuses(authored_cohort, monkeypatch, tmp_path):
    cohort = authored_cohort
    run = _left_stale(cohort, monkeypatch)
    measures = cohort.experiment.measures(run.id)
    _refuses(measures.table, run.id)
    _refuses(lambda: measures.export(tmp_path / "m"), run.id)


def test_a_comparison_of_a_stale_run_refuses(authored_cohort, monkeypatch):
    from detail.api import comparison

    cohort = authored_cohort
    run = _left_stale(cohort, monkeypatch)
    _refuses(
        lambda: comparison._resolve_artifact(cohort.experiment, run.id, None), run.id
    )


def test_exporting_a_stale_query_run_refuses(tmp_path):
    """The case the issue reproduced: a cutoff saved, the rebuild dead, the export pinning it."""
    from tests.api.test_query import _searched

    exp, run = _searched(tmp_path)
    run.set_cutoff("q01", 0.5)
    analysis_store.set_artifact_stale(run._run_dir(), True)
    _refuses(lambda: run.export(tmp_path / "out"), run.id)

    # Through the CLI: an error naming the rebuild, not a traceback and not an export; and the
    # state the calibrate command echoes says its counts predate the saved cutoff.
    from click.testing import CliRunner

    from detail.cli import main

    exported = CliRunner().invoke(
        main,
        ["analysis", "export-query", "--experiment", str(exp.dir), "--run", run.id,
         "--out", str(tmp_path / "cli_out")],
    )  # fmt: skip
    assert exported.exit_code == 1, exported.output
    assert f"{_REBUILD} {run.id}" in exported.output
    assert not (tmp_path / "cli_out").exists() or not any(
        (tmp_path / "cli_out").iterdir()
    )
    shown = CliRunner().invoke(
        main,
        ["analysis", "query-calibrate", "--experiment", str(exp.dir), "--run", run.id],
    )
    assert "results: out of date" in shown.output, shown.output


def test_a_clip_review_census_of_a_stale_run_refuses_and_writes_nothing(
    fitted_codebook,
):
    """A census records the cluster's membership as it is on disk, which a stale run has wrong."""
    from detail.storage import clip_review as review_store

    experiment, run_id = fitted_codebook
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    climb, _rear = view.children(view.root)
    run.name_nodes({climb: "climb"})
    analysis_store.set_artifact_stale(run._run_dir(), True)
    _refuses(lambda: run.finish_review(climb), run_id)
    assert review_store.read_review(run._run_dir()) == []
