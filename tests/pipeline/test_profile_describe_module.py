"""Profile clusters (``pipeline/profile_describe.py``): many clusters, one recorded job.

The module id is ``profile_describe``, the id the job had while it also ran a model's
description of each cluster; runs and saved settings from then must still open and launch.

The classes guarded here:

- **A batch that redoes finished work.** A job over 159 clusters is stopped or crashes;
  running it again must do only what is left. So a cluster whose profile answers the settings is
  skipped and nothing is measured for it.
- **A cancel that throws away what finished.** The clusters done before a Cancel keep their
  profiles, and the record says which finished and which were not reached.
- **One failure ending the night.** A cluster whose profile fails is a row with its reason; the
  next cluster runs.
- **A bar that sits still, or a phase label frozen.** The bar folds the running cluster's
  share into clusters done, moves within one cluster, carries a time left, and the messages
  name the declared phases in order (``phase_messages``).
- **A record that cannot say what made it** (Tenet 9): ``params`` is the job's.
- **An old run's settings that no longer launch.** A config saved with the description step's
  settings (backend, clips, model, ...) still validates, and those settings are dropped.

Seated on the profile fit (``fitted_profile``: a temporal fit with enough bouts per node to
profile several clusters), plus the workspace's own temporal fit.
The jobs run their runner in-process (the executor's own entry, with the stood-in tracks);
one test runs it through the executor.
"""

from __future__ import annotations

import json
import threading

import pytest

from detail.api import profile_describe as batch
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.profile_describe import RETIRED_FIELDS
from detail.executor.runners import ANALYSIS_MODULES, import_runner_modules, job_params
from detail.storage import analysis_runs as analysis_store
from tests.pipeline.phase_messages import assert_messages_name_their_phases

import_runner_modules(PIPELINE_RUNNERS)
MODULE = ANALYSIS_MODULES["profile_describe"]

#: Small and fast: a fixed count and a quick explainer for each profile.
SETTINGS = {"explainer": "logistic_regression", "n_per_side": 40}


def _stand_in_tracks(monkeypatch, view):
    """Every member of the run on a stood-in track (the fixture's embedding names a tracking
    run it never wrote), so any node can be profiled."""
    import detail.api.cluster_profile as cluster_profile
    from tests.api.test_cluster_profile import _track

    by_unit: dict = {}
    for m in view.sample_runs(view.root, n=100_000, seed=0):
        by_unit.setdefault((m["recording_id"], int(m["subject"])), []).append(m)
    monkeypatch.setattr(
        cluster_profile,
        "_trajectory",
        lambda _e, key, _r, _ts: _track(by_unit, key, fast=True),
    )


def _clusters(view, n: int) -> list[int]:
    """``n`` nodes with enough members to profile, the largest first."""
    nodes = [
        node
        for node in range(view.tree.n_nodes)
        if node != view.root and len(view.sample_runs(node, n=10_000, seed=0)) >= 12
    ]
    assert len(nodes) >= n, f"the fixture has only {len(nodes)} profilable nodes"
    return sorted(nodes, key=lambda node: -view.summary(node).time_ns)[:n]


@pytest.fixture
def clustered(fitted_profile, monkeypatch):
    experiment, run_id = fitted_profile
    run = experiment.temporal_clustering_run(run_id)
    run.rebuild_node_cache()
    view = run.node_view()
    _stand_in_tracks(monkeypatch, view)
    return experiment, str(run.id), view


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


def _run(experiment, run_id, *, on_event=None, stop=None, **config):
    job, run_dir = _launch(experiment, run_id, **config)
    events: list = []
    stop = stop or threading.Event()

    def progress(event):
        events.append(event)
        if on_event is not None:
            on_event(event, stop)

    MODULE.runner(job, run_dir, progress, stop)
    return job, analysis_store.read_analysis_yaml(run_dir), events


# --- the job ---------------------------------------------------------------------------- #
def test_each_cluster_is_profiled_and_listed(clustered):
    experiment, run_id, view = clustered
    nodes = _clusters(view, 2)
    job, record, _events = _run(experiment, run_id, clusters="ids", node_ids=nodes)
    assert record["status"] == "complete"
    assert [c["node"] for c in record["clusters"]] == nodes
    assert all(c["status"] == "complete" for c in record["clusters"])
    for cluster in record["clusters"]:
        assert experiment.cluster_profile(run_id, cluster["node"]) is not None
        assert cluster["profile"]["status"] == "done"
        assert "describe" not in cluster
    assert record["counts"] == {"complete": 2}
    assert record["source_run_id"] == run_id


def test_the_job_writes_its_own_dir_and_placing_again_places_nothing(clustered):
    """The job's results are its own (what a remote copy-back returns) and are placed beside
    the run; placing them again — after a copy-back, or a crash between two files — changes
    nothing."""
    experiment, run_id, view = clustered
    (node,) = _clusters(view, 1)
    job, _record, _events = _run(experiment, run_id, clusters="ids", node_ids=[node])
    own = experiment.analysis_run_dir(job.job_id) / "profiles" / f"node_{node}"
    assert (own / "profile.yaml").is_file()
    source = experiment.analysis_run_dir(run_id)
    before = {
        p.relative_to(source).as_posix(): p.read_bytes()
        for p in (source / "profiles").rglob("*")
        if p.is_file()
    }
    placed = batch.place(experiment, job.job_id)
    assert set(placed.values()) <= set(before)
    after = {
        p.relative_to(source).as_posix(): p.read_bytes()
        for p in (source / "profiles").rglob("*")
        if p.is_file()
    }
    assert after == before


def test_the_record_params_are_the_jobs(clustered):
    experiment, run_id, view = clustered
    nodes = _clusters(view, 1)
    job, record, _events = _run(experiment, run_id, clusters="ids", node_ids=nodes)
    assert job.nodes == nodes
    assert record["params"] == json.loads(json.dumps(job_params(job)))
    assert record["params"]["nodes"] == nodes
    assert record["params"]["compute"] is None
    assert not set(RETIRED_FIELDS) & set(record["params"])


def test_an_old_runs_settings_still_launch_without_the_description_step(clustered):
    """A config saved when the job also described each cluster ("start from" an old run)
    validates; its description settings are dropped, never refused or carried."""
    experiment, run_id, view = clustered
    old = {
        "backend": "ollama_vision",
        "n_clips": 50,
        "batch_size": 2,
        "blind": False,
        "use_setup": True,
        "use_behaviours": False,
        "allow_external": False,
        "model": "qwen2.5vl:7b",
        "vision_model": None,
        "num_ctx": 8192,
        "served_on": None,
    }
    assert set(old) - {"served_on"} == set(RETIRED_FIELDS)
    config = MODULE.validate(
        experiment.dir,
        {
            **SETTINGS,
            **old,
            "source_run_id": run_id,
            "clusters": "ids",
            "node_ids": [1],
        },
    )
    assert not set(old) & set(config)
    job = MODULE.build_job("an_99", {**config, **old})
    assert not hasattr(job, "backend") and not hasattr(job, "served_on")


def test_a_job_json_written_before_the_cut_still_reads(clustered):
    """A worker reading an old ``job.json`` (its description settings and ``served_on``
    included) builds the job from the fields it knows."""
    experiment, run_id, view = clustered
    config = MODULE.validate(
        experiment.dir,
        {**SETTINGS, "source_run_id": run_id, "clusters": "ids", "node_ids": [1]},
    )
    written = MODULE.build_job("an_99", config).to_dict()
    old = {**written, "backend": "ollama", "n_clips": 50, "served_on": None}
    read = MODULE.job_class.from_dict(old)
    assert read.to_dict() == written


def test_a_second_run_skips_the_finished_clusters(clustered, monkeypatch):
    experiment, run_id, view = clustered
    first, second, third = _clusters(view, 3)
    _run(experiment, run_id, clusters="ids", node_ids=[first, second])
    real = type(experiment).profile_cluster
    measured = []

    def counting(self, run, node, *a, **kw):
        measured.append(node)
        return real(self, run, node, *a, **kw)

    monkeypatch.setattr(type(experiment), "profile_cluster", counting)
    _job, record, _events = _run(
        experiment, run_id, clusters="ids", node_ids=[first, second, third]
    )
    statuses = [c["status"] for c in record["clusters"]]
    assert statuses == ["skipped", "skipped", "complete"]
    assert measured == [third], "only the new cluster was measured"


def test_a_changed_profile_setting_redoes_the_profile(clustered):
    experiment, run_id, view = clustered
    (node,) = _clusters(view, 1)
    _run(experiment, run_id, clusters="ids", node_ids=[node])
    _job, record, _events = _run(
        experiment, run_id, clusters="ids", node_ids=[node], seed=1
    )
    (cluster,) = record["clusters"]
    assert cluster["profile"]["status"] == "done"
    assert experiment.cluster_profile(run_id, node).config["seed"] == 1


def test_skip_existing_off_does_everything_again(clustered):
    experiment, run_id, view = clustered
    (node,) = _clusters(view, 1)
    _run(experiment, run_id, clusters="ids", node_ids=[node])
    _job, record, _events = _run(
        experiment, run_id, clusters="ids", node_ids=[node], skip_existing=False
    )
    assert record["clusters"][0]["profile"]["status"] == "done"


def test_a_cancel_keeps_the_finished_clusters(clustered):
    experiment, run_id, view = clustered
    nodes = _clusters(view, 3)

    def cancel_at_the_second(event, stop):
        if event.message and "cluster 2 of 3" in event.message:
            stop.set()

    _job, record, _events = _run(
        experiment,
        run_id,
        clusters="ids",
        node_ids=nodes,
        on_event=cancel_at_the_second,
    )
    statuses = [c["status"] for c in record["clusters"]]
    assert statuses == ["complete", "cancelled", "not run"]
    assert record["status"] == "partial"
    assert experiment.cluster_profile(run_id, nodes[0]) is not None
    assert experiment.cluster_profile(run_id, nodes[2]) is None


def test_a_failed_cluster_does_not_stop_the_job(clustered, monkeypatch):
    experiment, run_id, view = clustered
    nodes = _clusters(view, 3)
    real = type(experiment).profile_cluster

    def fails_for_the_second(self, run, node, *a, **kw):
        if node == nodes[1]:
            raise RuntimeError("no track for these members")
        return real(self, run, node, *a, **kw)

    monkeypatch.setattr(type(experiment), "profile_cluster", fails_for_the_second)
    _job, record, _events = _run(experiment, run_id, clusters="ids", node_ids=nodes)
    first, second, third = record["clusters"]
    assert (first["status"], second["status"], third["status"]) == (
        "complete",
        "failed",
        "complete",
    )
    assert second["profile"]["status"] == "failed"
    assert "no track for these members" in second["error"]
    assert record["status"] == "partial"


def test_the_bar_folds_each_cluster_and_the_phases_advance(clustered, monkeypatch):
    from detail.core import progress as progress_mod

    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    experiment, run_id, view = clustered
    nodes = _clusters(view, 2)
    _job, _record, events = _run(experiment, run_id, clusters="ids", node_ids=nodes)
    assert {e.recordings_total for e in events} == {2}
    assert all(e.unit_label == "cluster" for e in events)
    # Within the first cluster, not only at its boundary: the inner pair moves.
    inside = [
        e
        for e in events
        if e.recordings_done == 0 and 0 < e.frames_done < e.frames_total
    ]
    assert inside, "the bar never moved inside a cluster"
    assert any(e.eta_s is not None for e in inside), "no time left mid-cluster"
    folded = [
        (e.recordings_done + (e.frames_done / e.frames_total if e.frames_total else 0))
        for e in events
    ]
    assert folded == sorted(folded), "the folded bar went backwards"
    messages = [e.message for e in events]
    assert_messages_name_their_phases(messages, MODULE.phases)
    heads = [m.split(" · ")[0] for m in messages if "cluster 1 of 2" in m]
    order = [p for p in dict.fromkeys(heads)]
    assert order == list(MODULE.phases), order


def test_the_root_and_unknown_nodes_are_refused(clustered):
    experiment, run_id, view = clustered
    with pytest.raises(ValueError, match="root"):
        batch.select_nodes(view, "ids", [view.root])
    with pytest.raises(ValueError, match="not in this run's tree"):
        batch.select_nodes(view, "ids", [view.tree.n_nodes + 5])


def test_the_floor_is_the_runs_minimum_cluster_time_without_the_root(
    clustered, monkeypatch
):
    """The floor is ``NodeView.min_cluster_s``: the run's own where its cache recorded one,
    else ``NAMING_FLOOR_S`` — 30 s, more than this whole fixture, so it is lowered here to
    leave some clusters above it and some below."""
    from detail.api import node_view

    monkeypatch.setattr(node_view, "NAMING_FLOOR_S", 0.5)
    experiment, run_id, view = clustered
    floor = batch.select_nodes(view, "floor")
    floor_ns = view.min_cluster_s * 1e9
    assert view.root not in floor
    expected = {
        n
        for n in range(view.tree.n_nodes)
        if n != view.root and view.summary(n).time_ns >= floor_ns
    }
    assert floor and set(floor) == expected
    everything = batch.select_nodes(view, "all")
    assert set(floor) <= set(everything)
    leaves = batch.select_nodes(view, "leaves")
    assert all(n < view.tree.n_leaves for n in leaves)
    sizes = [view.summary(n).time_ns for n in floor]
    assert sizes == sorted(sizes, reverse=True), "largest first"


def test_the_estimate_counts_what_is_left(clustered):
    experiment, run_id, view = clustered
    first, second = _clusters(view, 2)
    config = {**SETTINGS, "clusters": "ids", "node_ids": [first, second]}
    before = experiment.estimate_profile_describe(run_id, config)
    assert (before.n_done, before.n_profile) == (0, 2)
    assert before.seconds == before.profile_s > 0
    _run(experiment, run_id, clusters="ids", node_ids=[first])
    after = experiment.estimate_profile_describe(run_id, config)
    assert (after.n_done, after.n_profile) == (1, 1)
    assert "already done" in after.text()


# --- a temporal run ------------------------------------------------------------------------ #
def test_a_temporal_run_is_profiled(
    experiment_dir, recording_dir, make_embedding_run, monkeypatch
):
    from tests.gui.test_temporal_workspace import _fitted

    experiment, run_id = _fitted(experiment_dir, recording_dir, make_embedding_run)
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    _stand_in_tracks(monkeypatch, view)
    nodes = [
        n
        for n in batch.select_nodes(view, "all")
        if len(view.sample_runs(n, n=10_000, seed=0)) >= 4
    ][:1]
    assert nodes, "the temporal fit has no node to profile"
    _job, record, _events = _run(
        experiment, run_id, clusters="ids", node_ids=nodes, n_per_side=None,
        time_budget_s=5.0,
    )  # fmt: skip
    (cluster,) = record["clusters"]
    assert cluster["status"] == "complete", cluster
    assert experiment.cluster_profile(run_id, nodes[0]).unit == "behavior"


# --- through the executor ------------------------------------------------------------------ #
def test_the_job_runs_through_the_executor_and_lists_its_clusters(
    clustered, tmp_path, monkeypatch
):
    """The one real executor run: launched the way every surface launches it, in a worker
    process. The worker has no stood-in tracks, so each profile measures nothing; the job still
    runs every cluster, records why each one stopped where it did, and completes."""
    experiment, run_id, view = clustered
    monkeypatch.setenv("DETAIL_CONFIG_DIR", str(tmp_path))
    nodes = _clusters(view, 2)
    handle = experiment.profile_describe_clusters(
        run_id, {**SETTINGS, "clusters": "ids", "node_ids": nodes}
    )
    handle.wait()
    record = experiment.analysis_record(handle.id)
    assert record["module_id"] == "profile_describe"
    assert [c["node"] for c in record["clusters"]] == nodes
    assert all(c["status"] in ("complete", "failed") for c in record["clusters"])
    assert record["params"]["source_run_id"] == run_id
    assert (
        record["recording_ids"] == experiment.analysis_record(run_id)["recording_ids"]
    )
