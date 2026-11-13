"""Find look-alike clusters (``pipeline/cluster_lookalikes.py``): many pairs, one recorded job.

The classes guarded here:

- **A batch that redoes finished work.** A job over a hundred pairs is stopped or crashes;
  running it again must do only what is left. A pair compared with the same settings — by this
  job, the other way round, or by hand in the Cluster profile window — is skipped, and nothing
  is measured for it.
- **A cancel that throws away what finished**, and **one failure ending the run**: the pairs
  done before a Cancel keep their comparisons, a pair that fails is a row with its reason, and
  the record says which finished and which were not reached.
- **A bar that sits still, or a phase label frozen.** The bar folds the running pair's share
  into the pairs done and moves within a pair; the messages name the declared phases in order.
- **A job that compares what the run says now, not what it was launched for** (Tenet 9): the
  pairs are resolved at launch and written into the job and its ``params``.
- **A job with nothing to do that says nothing.** No pair to compare is refused before anything
  runs, with why — for ``pairs: labelled``, that no cluster is labelled yet.
- **A mode the record does not name.** ``pairs: labelled`` resolves to the pairs with a labelled
  side at launch, and ``params`` says which mode; an old ``pairs: names`` loads as every pair.

The jobs run their runner in-process (the executor's own entry, with the stood-in tracks).
"""

from __future__ import annotations

import threading

import pytest

import detail.api.cluster_lookalikes as lookalikes
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.executor.runners import ANALYSIS_MODULES, import_runner_modules, job_params
from detail.storage import analysis_runs as analysis_store
from tests.pipeline.phase_messages import assert_messages_name_their_phases
from tests.pipeline.test_profile_describe_module import _stand_in_tracks

import_runner_modules(PIPELINE_RUNNERS)
MODULE = ANALYSIS_MODULES["cluster_lookalikes"]

#: Small and fast: a fixed count and a quick explainer for each pair.
SETTINGS = {"explainer": "logistic_regression", "n_per_side": 20}


@pytest.fixture(autouse=True)
def _a_floor_the_fixture_clears(monkeypatch):
    """The naming floor a cluster needs (``NodeView.min_cluster_s``) is ``NAMING_FLOOR_S``,
    30 s, for a cache built now: more than this whole fixture, so it would leave no cluster to
    compare. Lowered to what these tests ran under when the cache took a floor
    (``rebuild_node_cache(0.05)``)."""
    from detail.api import node_view

    monkeypatch.setattr(node_view, "NAMING_FLOOR_S", 0.05)


def _apart(view, n: int) -> list[int]:
    """``n`` branch tops with members enough to profile, none inside another."""
    chosen: list[int] = []
    # Smallest first: a large top tends to hold the others.
    for node in sorted(
        lookalikes.branch_tops(view), key=lambda t: view.summary(t).members
    ):
        if len(view.sample_runs(node, n=10_000, seed=0)) < 10:
            continue
        if any(c in view.ancestors(node) or node in view.ancestors(c) for c in chosen):
            continue
        chosen.append(node)
        if len(chosen) == n:
            return chosen
    raise AssertionError(f"the fixture has fewer than {n} unrelated branch tops")


@pytest.fixture
def clustered(fitted_profile, monkeypatch):
    experiment, run_id = fitted_profile
    run = experiment.temporal_clustering_run(run_id)
    run.rebuild_node_cache()
    view = run.node_view()
    _stand_in_tracks(monkeypatch, view)
    return experiment, str(run.id), view


@pytest.fixture
def named(clustered, monkeypatch):
    """Three unrelated branch tops, and the job's pairs narrowed to the three pairs between
    them, so a job compares exactly those (every pair of the fixture's tops would be slow).
    """
    experiment, run_id, view = clustered
    nodes = _apart(view, 3)
    a, b, c = sorted(nodes)
    monkeypatch.setattr(
        lookalikes, "candidate_pairs", lambda _v: [(a, b), (a, c), (b, c)]
    )
    return experiment, run_id, view, nodes


def _launch(experiment, run_id: str, **config):
    """What ``Experiment.run_analysis`` does before it submits: validate, reserve an id,
    write the initial record, build the job. The runner is then called in-process."""
    config = MODULE.validate(
        experiment.dir, {**SETTINGS, **config, "source_run_id": run_id}
    )
    job_id = analysis_store.reserve_analysis_id(experiment.dir)
    run_dir = analysis_store.analysis_dir(experiment.dir, job_id)
    analysis_store.write_analysis_yaml(
        run_dir, {"id": job_id, "module_id": MODULE.module_id, "status": "running"}
    )
    return MODULE.build_job(job_id, config), run_dir


def _run(experiment, run_id, *, on_event=None, **config):
    job, run_dir = _launch(experiment, run_id, **config)
    events: list = []
    stop = threading.Event()

    def progress(event):
        events.append(event)
        if on_event is not None:
            on_event(event, stop)

    MODULE.runner(job, run_dir, progress, stop)
    return job, analysis_store.read_analysis_yaml(run_dir), events


def test_each_pair_is_compared_saved_beside_the_run_and_listed(named):
    experiment, run_id, _view, nodes = named
    job, record, _events = _run(experiment, run_id)
    pairs = {(p["a"], p["b"]) for p in record["pairs"]}
    a, b, c = sorted(nodes)
    assert pairs == {(a, b), (a, c), (b, c)}
    assert all(p["status"] == "complete" for p in record["pairs"])
    assert record["status"] == "complete"
    source = experiment.analysis_run_dir(run_id)
    for p in record["pairs"]:
        assert (source / p["file"]).is_file(), "saved beside the clustering run"
        assert "place_accuracy" in p
    evidence = lookalikes.read_evidence(source)
    assert {item.pair for item in evidence} == pairs, "a comparison the groups read"
    assert record["params"] == job_params(job)
    assert record["params"]["pair_list"] == [list(p) for p in job.pair_list]
    assert isinstance(record["groups"], list)


def test_the_pairs_are_resolved_at_launch(named, monkeypatch):
    """The pairs the job was launched with are its pairs, whatever the run's tree offers by the
    time it runs (Tenet 9)."""
    experiment, run_id, _view, nodes = named
    job, _run_dir = _launch(experiment, run_id)
    monkeypatch.setattr(lookalikes, "candidate_pairs", lambda _v: [])
    a, b, c = sorted(nodes)
    assert [tuple(p) for p in job.pair_list] == [(a, b), (a, c), (b, c)]


def test_a_job_written_before_the_cut_reads_its_pairs(named):
    """A job's pairs once carried a third item (the names' overlap); it is ignored."""
    experiment, run_id, _view, nodes = named
    job, run_dir = _launch(experiment, run_id)
    job.pair_list = [[a, b, 0.5] for a, b in job.pair_list]
    MODULE.runner(job, run_dir, lambda _e: None, threading.Event())
    record = analysis_store.read_analysis_yaml(run_dir)
    assert [p["status"] for p in record["pairs"]] == ["complete"] * 3
    assert all("name_overlap" not in p for p in record["pairs"])


def test_a_second_run_compares_nothing_again(named, monkeypatch):
    experiment, run_id, _view, _nodes = named
    _run(experiment, run_id)

    def measured(*_a, **_k):
        raise AssertionError("a compared pair was measured again")

    monkeypatch.setattr(type(experiment), "profile_cluster", measured)
    _job, record, _events = _run(experiment, run_id)
    assert [p["status"] for p in record["pairs"]] == ["skipped"] * 3
    assert record["status"] == "complete"


def test_a_pair_compared_by_hand_the_other_way_round_is_skipped(named):
    experiment, run_id, _view, nodes = named
    a, b, _c = sorted(nodes)
    experiment.profile_cluster(
        run_id,
        b,
        {**SETTINGS, "compare": "node", "against": a, "leave_out_place": True},
    )
    _job, record, _events = _run(experiment, run_id)
    status = {(p["a"], p["b"]): p["status"] for p in record["pairs"]}
    assert status[(a, b)] == "skipped"
    assert sorted(status.values()) == ["complete", "complete", "skipped"]


def test_redo_compares_every_pair_again(named):
    experiment, run_id, _view, _nodes = named
    _run(experiment, run_id)
    _job, record, _events = _run(experiment, run_id, skip_existing=False)
    assert [p["status"] for p in record["pairs"]] == ["complete"] * 3


def test_a_cancel_keeps_the_finished_pairs(named):
    experiment, run_id, _view, _nodes = named

    def cancel_at_the_second(event, stop):
        if event.message and "pair 2 of 3" in event.message:
            stop.set()

    _job, record, _events = _run(experiment, run_id, on_event=cancel_at_the_second)
    statuses = [p["status"] for p in record["pairs"]]
    assert statuses == ["complete", "cancelled", "not run"]
    assert record["status"] == "partial"
    source = experiment.analysis_run_dir(run_id)
    assert (source / record["pairs"][0]["file"]).is_file()
    assert len(lookalikes.read_evidence(source)) == 1


def test_a_failed_pair_does_not_stop_the_job(named, monkeypatch):
    experiment, run_id, _view, nodes = named
    first = min(nodes)
    real = type(experiment).profile_cluster

    def fails_for_the_first(self, run, node, config=None, **kw):
        if node == first and config["against"] == sorted(nodes)[1]:
            raise RuntimeError("no track for these members")
        return real(self, run, node, config, **kw)

    monkeypatch.setattr(type(experiment), "profile_cluster", fails_for_the_first)
    _job, record, _events = _run(experiment, run_id)
    statuses = [p["status"] for p in record["pairs"]]
    assert sorted(statuses) == ["complete", "complete", "failed"]
    failed = next(p for p in record["pairs"] if p["status"] == "failed")
    assert "no track for these members" in failed["error"]
    assert record["status"] == "partial"


def test_the_bar_folds_each_pair_and_the_phases_advance(named, monkeypatch):
    from detail.core import progress as progress_mod

    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    experiment, run_id, _view, _nodes = named
    _job, _record, events = _run(experiment, run_id)
    assert {e.recordings_total for e in events} == {3}
    assert all(e.unit_label == "pair" for e in events)
    inside = [
        e
        for e in events
        if e.recordings_done == 0 and 0 < e.frames_done < e.frames_total
    ]
    assert inside, "the bar never moved inside a pair"
    assert any(e.eta_s is not None for e in inside), "no time left mid-pair"
    folded = [
        (e.recordings_done + (e.frames_done / e.frames_total if e.frames_total else 0))
        for e in events
    ]
    assert folded == sorted(folded), "the folded bar went backwards"
    messages = [e.message for e in events]
    assert_messages_name_their_phases(messages, MODULE.phases)
    heads = [m.split(" · ")[0] for m in messages if "pair 1 of 3" in m]
    assert list(dict.fromkeys(heads)) == list(MODULE.phases)


def test_every_pair_of_branch_tops_is_resolved(clustered):
    experiment, run_id, view = clustered
    config = MODULE.validate(experiment.dir, {**SETTINGS, "source_run_id": run_id})
    tops = lookalikes.branch_tops(view)
    assert config["pair_list"] == [list(p) for p in lookalikes.candidate_pairs(view)]
    for a, b in config["pair_list"]:
        assert a in tops and b in tops and a < b
        assert a not in view.ancestors(b) and b not in view.ancestors(a)


def test_a_saved_config_with_the_retired_pair_settings_still_launches(clustered):
    """An older run's settings (``pairs: names``, ``min_name_overlap``) from "start from"
    validate: ``names`` reads as the default, every pair, and the overlap is dropped."""
    experiment, run_id, view = clustered
    config = MODULE.validate(
        experiment.dir,
        {
            **SETTINGS,
            "source_run_id": run_id,
            "pairs": "names",
            "min_name_overlap": 0.33,
        },
    )
    assert config["pairs"] == "all" and "min_name_overlap" not in config
    assert config["pair_list"] == [list(p) for p in lookalikes.candidate_pairs(view)]


def _label(experiment, run_id: str, names: dict, monkeypatch) -> dict:
    """Label clusters (no re-projection: these tests read only the labels); the run's
    ``{node: label id}``."""
    from detail.api.temporal_clustering import TemporalClusteringRun

    monkeypatch.setattr(TemporalClusteringRun, "_project", lambda self, **_: None)
    run = experiment.temporal_clustering_run(run_id)
    run.name_nodes(names)
    return dict(run.state().node_labels)


def test_labelled_compares_each_labelled_top_against_every_other(
    clustered, monkeypatch
):
    """The pairs resolved at launch are the labelled ones, and the record says which mode it
    was (Tenet 9)."""
    experiment, run_id, view = clustered
    top = _apart(view, 1)[0]
    node_labels = _label(experiment, run_id, {top: "rearing"}, monkeypatch)
    job, run_dir = _launch(experiment, run_id, pairs="labelled")
    expected = lookalikes.labelled_pairs(view, node_labels)
    assert [tuple(p) for p in job.pair_list] == expected
    assert 0 < len(expected) < len(lookalikes.candidate_pairs(view))
    covered = set(lookalikes.labelled_tops(view, node_labels))
    assert top in covered
    for a, b in expected:
        assert a in covered or b in covered, "a pair with no labelled side"
    assert job_params(job)["pairs"] == "labelled"


def test_labelled_with_no_label_says_to_label_one_first(clustered):
    experiment, run_id, _view = clustered
    with pytest.raises(ValueError, match="label at least one cluster first") as caught:
        MODULE.validate(
            experiment.dir, {**SETTINGS, "source_run_id": run_id, "pairs": "labelled"}
        )
    assert "every pair" in str(caught.value)


def test_the_count_api_matches_what_each_mode_resolves(clustered, monkeypatch):
    """``Experiment.count_lookalike_pairs`` counts what the job would resolve, both modes."""
    experiment, run_id, view = clustered
    top = _apart(view, 1)[0]
    _label(experiment, run_id, {top: "rearing"}, monkeypatch)
    counts = experiment.count_lookalike_pairs(run_id, SETTINGS)
    every = MODULE.validate(experiment.dir, {**SETTINGS, "source_run_id": run_id})
    labelled = MODULE.validate(
        experiment.dir, {**SETTINGS, "source_run_id": run_id, "pairs": "labelled"}
    )
    assert counts.counts == {
        "all": len(every["pair_list"]),
        "labelled": len(labelled["pair_list"]),
    }
    assert counts.n_labelled_tops >= 1
