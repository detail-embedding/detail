"""A clustering run cannot be finished before anything is labelled (§21, §22).

**The class.** A run finished before the decision its result needs was made. A fit is a tree
nobody has read until a node carries a label: until then no instance in the output carries one,
so AUDIT has nothing to review and BROWSE nothing to show. Finished that way, AUDIT said "label a node on
INTERPRET" while INTERPRET said "this run is finished, reopen it": two instructions at once, and
the second undoes the first. The behavior scan had already learned this (a behavior with no
cutoff, ``tests/api/test_query_review.py``); the clustering run had not, because the rule lived
in ``QueryRun.finalize`` rather than in the base class.

**The rule.** ``AuthoredRun.finish_blocker`` is the hook, and each kind words its own reason.
``AuthoredRun.finalize`` refuses as read-only, then as incomplete, then with the blocker, before
anything is pinned; the CLI prints it as one ``Error:`` line; INTERPRET greys **Finish** with it
in words beside the button (``tests/gui/test_cluster_workspace.py``). That every kind has decided
what finishing needs is ``tests/api/test_run_lifecycle_parity.py``.
"""

import pytest
import yaml
from click.testing import CliRunner

from detail.api.authored_run import ReadOnlyRunError
from detail.cli import main
from detail.core.errors import UserInputError
from detail.storage import analysis_runs

_REASON_START = "No behavior is labelled yet, so this run cannot be finished."


def _a_node(run) -> int:
    """An internal node of the run's tree, the kind a first label goes on."""
    tree = run.hierarchy()
    return int(tree.children[tree.n_nodes - 1 - tree.n_leaves][0])


def _history_ops(cohort) -> list[str]:
    path = cohort.history_path()
    if not path.is_file():
        return []
    return [
        yaml.safe_load(line)["op"]
        for line in path.read_text(encoding="utf-8-sig").splitlines()
    ]


def test_finalize_is_refused_while_nothing_is_labelled_naming_the_fix(
    authored_cohort,
):
    run = authored_cohort.run()
    with pytest.raises(UserInputError) as refused:
        run.finalize()
    message = str(refused.value)
    assert message.startswith(_REASON_START)
    assert "Label one on Interpret" in message
    assert (
        f"`detail analysis temporal-interpret --run {run.id} --name-node NODE=NAME`"
        in message
    ), "the CLI way names the real command and this run"
    assert run.status == "open"
    assert not run.snapshots(), "pinned a snapshot on the way to refusing"
    assert "finalize" not in _history_ops(authored_cohort)


def test_finish_blocker_is_the_reason_finalize_gives(authored_cohort):
    """The read a surface greys Finish with and the refusal are one sentence, not two copies."""
    run = authored_cohort.run()
    reason = run.finish_blocker()
    assert reason is not None and reason.startswith(_REASON_START)
    with pytest.raises(UserInputError) as refused:
        run.finalize()
    assert str(refused.value) == reason


def test_one_label_lets_the_run_finish_and_removing_it_refuses_again(fitted_temporal):
    """Through the label edits a user makes, on a real fit: ``authored_cohort`` cannot
    reproject, and a label is an edit that does."""
    experiment, run_id = fitted_temporal
    run = experiment.temporal_clustering_run(run_id)
    node = _a_node(run)
    run.name_nodes({node: "climb"})
    assert run.finish_blocker() is None
    run.finalize()
    assert experiment.temporal_clustering_run(run_id).status == "complete"

    run.reopen()
    run.name_nodes({node: ""})
    reason = run.finish_blocker() or ""
    assert reason.startswith(_REASON_START), "the last label went; no refusal came back"


def test_a_label_with_no_cluster_does_not_count(fitted_temporal):
    """A label made and never given a cluster yields no instance: still nothing to audit."""
    experiment, run_id = fitted_temporal
    run = experiment.temporal_clustering_run(run_id)
    run.create_label("climb")
    assert run.finish_blocker() is not None


def test_a_snapshot_refuses_to_finalize_as_read_only_before_asking_about_labels(
    authored_cohort,
):
    """A view of an unlabelled run: the refusal is the snapshot's own, which names the way back
    (the live run), not "label one", which a snapshot cannot do."""
    run = authored_cohort.run()
    view = run.view(run.pin(reason="export"))
    assert view.finish_blocker() is not None
    with pytest.raises(ReadOnlyRunError, match="will not finalize"):
        view.finalize()


def test_an_unfinished_compute_is_refused_as_that_before_asking_about_labels(
    authored_cohort,
):
    """The deeper cause first: no label fixes a fit that did not finish."""
    run = authored_cohort.run()
    analysis_runs.set_analysis_status(authored_cohort.run_dir(), "partial")
    with pytest.raises(ValueError) as refused:
        run.finalize()
    assert not isinstance(refused.value, UserInputError), str(refused.value)
    assert "compute did not finish" in str(refused.value)


def test_the_cli_refuses_to_finish_an_unlabelled_run_without_a_traceback(
    fitted_temporal,
):
    experiment, run_id = fitted_temporal
    run = experiment.temporal_clustering_run(run_id)
    base = [
        "analysis",
        "temporal-interpret",
        "--experiment",
        str(experiment.dir),
        "--run",
        run_id,
    ]
    refused = CliRunner().invoke(main, base + ["--finalize"])
    assert refused.exit_code == 1, refused.output
    assert "Traceback" not in refused.output
    assert isinstance(refused.exception, SystemExit), refused.exception
    errors = [l for l in refused.output.splitlines() if l.startswith("Error: ")]
    assert errors == [f"Error: {run.finish_blocker()}"], refused.output
    assert "run finalized" not in refused.output
    assert experiment.temporal_clustering_run(run_id).status == "open"

    # The fix the message names, in the same command: the label lands, then the finish.
    done = CliRunner().invoke(
        main, base + ["--name-node", f"{_a_node(run)}=climb", "--finalize"]
    )
    assert done.exit_code == 0, done.output
    assert "run finalized" in done.output
    assert experiment.temporal_clustering_run(run_id).status == "complete"


def test_the_finalize_help_says_when_it_is_refused():
    result = CliRunner().invoke(main, ["analysis", "temporal-interpret", "--help"])
    assert result.exit_code == 0, result.output
    assert "Refused while nothing is labelled" in " ".join(result.output.split())
