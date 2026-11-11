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
    with pytest.raises(_Boom):
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
    with pytest.raises(_Boom):
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
    with pytest.raises(_Boom):
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
    with pytest.raises(_Boom):
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
