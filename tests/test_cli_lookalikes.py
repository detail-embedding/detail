"""``analysis lookalikes`` and ``analysis find-lookalikes``: the CLI's half of INTERPRET's
Look-alike clusters window (§22).

Surface parity with the window: list each group with its members, its proposed name, why the
members are grouped and what each member's label would sweep in; give one group one label after a
summary of what it does — the members the window ticks, unless told otherwise — leaving members
out, adding one or naming it otherwise, against the same API (``Lookalikes``), so a label given
here is the label given there. ``find-lookalikes`` refuses a run with nothing to compare, with why,
and sends ``--remote-target`` to the job (§6 "Remote analysis"). With every pair (``--pairs all``,
the default) it prints how many pairs each mode compares and asks before starting, as the launch
dialog does; ``--yes`` skips the question, ``--pairs labelled`` needs none.
"""

import pytest
from click.testing import CliRunner

from detail.api.experiment import Experiment
from detail.cli import main
from tests.api.test_cluster_lookalikes import save_comparison


@pytest.fixture(autouse=True)
def _a_floor_the_fixture_clears(monkeypatch):
    """The naming floor a cluster needs (``NodeView.min_cluster_s``) is ``NAMING_FLOOR_S``,
    30 s, for a cache built now: more than this whole fixture, so it would leave no cluster to
    compare. Lowered to what these tests ran under when the cache took a floor
    (``rebuild_node_cache(0.05)``)."""
    from detail.api import node_view

    monkeypatch.setattr(node_view, "NAMING_FLOOR_S", 0.05)


@pytest.fixture
def grouped(fitted_temporal, monkeypatch):
    """Three clusters from both halves of a real fit's tree, chained by two look-alike pairs,
    and a pair the body measures told apart."""
    from detail.api.temporal_clustering import TemporalClusteringRun

    monkeypatch.setattr(TemporalClusteringRun, "_project", lambda self, **_: None)
    experiment, run_id = fitted_temporal
    run = experiment.temporal_clustering_run(run_id)
    run.rebuild_node_cache()
    view = run.node_view()
    left, right = list(view.children(view.root))[:2]
    a1, a2 = list(view.children(left))[:2]
    b1 = list(view.children(right))[0]
    run_dir = experiment.analysis_run_dir(str(run.id))
    save_comparison(run_dir, a1, b1, 0.52)
    save_comparison(run_dir, a2, b1, 0.55)
    save_comparison(run_dir, a1, right, 0.95)
    return experiment, run, (a1, a2, b1)


def _cli(experiment, run, command, *args, input=None):
    return CliRunner().invoke(
        main,
        ["analysis", command, "--experiment", str(experiment.dir)]
        + ["--run", str(run.id), *args],
        input=input,
    )


def test_lists_each_group_its_members_and_its_links(grouped):
    experiment, run, nodes = grouped
    result = _cli(experiment, run, "lookalikes")
    assert result.exit_code == 0, result.output
    assert "group 1: 3 clusters" in result.output
    for node in nodes:
        assert f"#{node}" in result.output
    assert (
        result.output.count("body measures") == 2
    ), "the links, not the pair told apart"
    assert "position and facing alone 90%" in result.output


def test_a_member_whose_label_would_sweep_others_in_is_marked_and_not_labelled(grouped):
    # The fit has no labels, so a member with clusters inside it sweeps them in and is not
    # chosen by default, while a leaf member sweeps nothing and is.
    experiment, run, _nodes = grouped
    group = run.lookalike_groups()[0]
    assert group.sweeps and group.ticked
    swept = next(iter(group.sweeps))
    assert swept not in group.ticked
    listing = _cli(experiment, run, "lookalikes")
    assert f"[ ] #{swept}" in listing.output
    assert "would also label" in listing.output and "of the run)" in listing.output
    included = _cli(
        experiment, run, "lookalikes", "--label-group", "1", "--name", "rearing",
        "--include", str(swept), input="n\n",
    )  # fmt: skip
    assert f"#{swept} — also labels" in included.output
    assert "Label them first" in included.output
    assert not run.state().node_names
    result = _cli(
        experiment,
        run,
        "lookalikes",
        "--label-group",
        "1",
        "--name",
        "rearing",
        "--yes",
    )
    assert result.exit_code == 0, result.output
    labelled = run.state().node_names
    assert str(swept) not in labelled
    assert all(str(n) in labelled for n in group.ticked)


def test_a_group_is_labelled_after_its_summary_leaving_one_out(grouped):
    experiment, run, nodes = grouped
    left_out = nodes[0]
    every = [arg for node in nodes for arg in ("--include", str(node))]
    declined = _cli(
        experiment, run, "lookalikes", "--label-group", "1", "--name", "rearing",
        *every, "--leave-out", str(left_out), input="n\n",
    )  # fmt: skip
    assert "Will label 2 clusters “rearing”" in declined.output
    assert not run.state().node_names, "declined: nothing written"

    result = _cli(
        experiment, run, "lookalikes", "--label-group", "1", "--name", "rearing",
        *every, "--leave-out", str(left_out), "--yes",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    names = run.state().node_names
    assert {names[str(n)] for n in nodes if n != left_out} == {"rearing"}
    assert str(left_out) not in names
    source = run.state().node_label_sources[str(nodes[1])]
    assert source["from"] == "lookalikes"


def test_a_group_with_no_name_asks_for_one(grouped):
    experiment, run, _nodes = grouped
    result = _cli(experiment, run, "lookalikes", "--label-group", "1", "--yes")
    assert result.exit_code != 0
    assert "give one with --name" in result.output


def test_a_group_number_or_member_that_is_not_there_is_refused(grouped):
    experiment, run, _nodes = grouped
    missing = _cli(experiment, run, "lookalikes", "--label-group", "4", "--name", "x")
    assert missing.exit_code != 0 and "there is no group 4" in missing.output
    stranger = _cli(
        experiment, run, "lookalikes", "--label-group", "1", "--name", "x",
        "--leave-out", "99999",
    )  # fmt: skip
    assert stranger.exit_code != 0 and "not in group 1" in stranger.output


def test_a_run_with_none_says_how_to_make_them(fitted_temporal):
    experiment, run_id = fitted_temporal
    run = experiment.temporal_clustering_run(run_id)
    run.rebuild_node_cache()
    result = _cli(experiment, run, "lookalikes")
    assert result.exit_code == 0, result.output
    assert "no look-alike groups" in result.output
    assert "find-lookalikes" in result.output


def test_find_lookalikes_with_nothing_to_compare_says_why(fitted_temporal, monkeypatch):
    from detail.api import node_view

    experiment, run_id = fitted_temporal
    run = Experiment(experiment.dir).temporal_clustering_run(run_id)
    run.rebuild_node_cache()
    monkeypatch.setattr(
        node_view, "NAMING_FLOOR_S", 1e9
    )  # no cluster reaches the floor
    result = _cli(experiment, run, "find-lookalikes")
    assert result.exit_code != 0
    assert "has no pair of clusters to compare" in result.output
    assert "branch tops" in result.output


def test_find_lookalikes_sends_the_remote_target_to_the_job(fitted_temporal):
    """--remote-target reaches the job's validation: a target this machine has not registered
    is refused by name, before the pairs are even resolved."""
    experiment, run_id = fitted_temporal
    run = Experiment(experiment.dir).temporal_clustering_run(run_id)
    run.rebuild_node_cache()
    result = _cli(
        experiment, run, "find-lookalikes", "--remote-target", "nowhere", "--yes"
    )
    assert result.exit_code != 0
    assert "no remote target 'nowhere' is registered" in result.output


@pytest.fixture
def submitted(fitted_temporal, monkeypatch):
    """A fit with branch tops to compare, and the config ``find_lookalikes`` is called with
    (the job itself is not run: refused with a marker once the config is in hand)."""
    experiment, run_id = fitted_temporal
    run = Experiment(experiment.dir).temporal_clustering_run(run_id)
    run.rebuild_node_cache()
    sent: list[dict] = []

    def find_lookalikes(self, run_id, config=None):
        sent.append(dict(config or {}))
        raise ValueError("stopped before the job")

    monkeypatch.setattr(Experiment, "find_lookalikes", find_lookalikes)
    return experiment, run, sent


def test_every_pair_warns_and_asks_before_starting(submitted):
    experiment, run, sent = submitted
    counts = Experiment(experiment.dir).count_lookalike_pairs(str(run.id))
    assert counts.n_all, "the fixture has pairs to compare"
    result = _cli(experiment, run, "find-lookalikes", input="n\n")
    assert result.exit_code != 0
    assert "compute intensive" in result.output
    assert f"{counts.n_all:,} pair" in result.output
    assert "no cluster is labelled yet" in result.output
    assert "Compare every pair?" in result.output
    assert not sent, "a declined question started the job"


def test_every_pair_starts_once_the_question_is_answered(submitted):
    experiment, run, sent = submitted
    result = _cli(experiment, run, "find-lookalikes", input="y\n")
    assert "stopped before the job" in result.output
    assert [c["pairs"] for c in sent] == ["all"]


def test_yes_skips_the_question_but_not_the_warning(submitted):
    experiment, run, sent = submitted
    result = _cli(experiment, run, "find-lookalikes", "--yes")
    assert "compute intensive" in result.output
    assert "Compare every pair?" not in result.output
    assert [c["pairs"] for c in sent] == ["all"]


def test_labelled_reaches_the_job_without_a_question(submitted):
    experiment, run, sent = submitted
    result = _cli(experiment, run, "find-lookalikes", "--pairs", "labelled")
    assert "compute intensive" not in result.output
    assert "Compare every pair?" not in result.output
    assert [c["pairs"] for c in sent] == ["labelled"]


def test_labelled_with_no_label_is_refused_by_the_job(fitted_temporal):
    experiment, run_id = fitted_temporal
    run = Experiment(experiment.dir).temporal_clustering_run(run_id)
    run.rebuild_node_cache()
    result = _cli(experiment, run, "find-lookalikes", "--pairs", "labelled")
    assert result.exit_code != 0
    assert "label at least one cluster first" in result.output
