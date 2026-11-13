"""Profile clusters on a remote target (§6 "Remote analysis").

The job sent to a cluster profiles there, which frees this machine for the hours a large run
takes. The classes guarded here:

- **A job whose results stay on the cluster.** It names the clustering run it reads (re-sent
  exactly before it starts) and places its results beside that run here once they are back.
- **A node sized for something else.** The work is CPU: the job asks for one GPU, the fewest
  Tillicum takes, and the node's CPUs and memory scale with it.
- **An envelope the cluster cannot read.** The job the shepherd writes there is JSON, and the
  worker reads back what it needs.
"""

from __future__ import annotations

import pytest

from detail.api import remote as api_remote
from detail.configs.remote import RemoteTargetConfig
from detail.executor import remote_shepherd
from detail.storage import analysis_runs as analysis_store
from tests.pipeline.test_profile_describe_module import (  # noqa: F401
    MODULE,
    SETTINGS,
    _clusters,
    _run,
    clustered,
)

REMOTE = {
    "backend": "remote",
    "device_indices": None,
    "cloud_provider": None,
    "cloud_instance": None,
    "remote_target": "tillicum",
}


@pytest.fixture
def tillicum(monkeypatch):
    monkeypatch.setattr(
        api_remote,
        "get_remote_target",
        lambda name: RemoteTargetConfig(name=name, host=name, remote_workdir="/x"),
    )


def _validated(experiment, run_id, **config):
    return MODULE.validate(
        experiment.dir, {**SETTINGS, **config, "source_run_id": run_id}
    )


def test_a_remote_job_records_where_it_runs_and_what_it_reads(clustered, tillicum):
    experiment, run_id, view = clustered
    config = _validated(
        experiment, run_id, clusters="ids", node_ids=_clusters(view, 1), compute=REMOTE
    )
    assert config["compute"]["remote_target"] == "tillicum"
    job = MODULE.build_job("an_99", config)
    assert MODULE.remote_resources(job) == {"gpus": 1}
    assert MODULE.refreshed_inputs(experiment.dir, job) == [
        analysis_store.analysis_dir(experiment.dir, run_id)
    ]
    assert remote_shepherd._model_readiness_probe(job, "/x/exp", "") is None


def test_after_the_copy_back_the_results_are_placed_beside_the_run(clustered, tillicum):
    """What the cluster's copy of the job placed never comes back; the job's own folder does,
    and placing from it puts every file where the Interpret tab reads it."""
    experiment, run_id, view = clustered
    (node,) = _clusters(view, 1)
    job, _record, _events = _run(
        experiment, run_id, clusters="ids", node_ids=[node], compute=REMOTE
    )
    placed = experiment.analysis_run_dir(run_id) / "profiles" / f"node_{node}"
    for path in sorted(placed.rglob("*"), reverse=True):  # as on this machine before
        path.unlink() if path.is_file() else path.rmdir()
    placed.rmdir()

    MODULE.after_copy_back(experiment.dir, job, experiment.analysis_run_dir(job.job_id))

    assert experiment.cluster_profile(run_id, node) is not None


def test_the_shepherd_sizes_the_node_to_one_gpu(clustered, tillicum):
    experiment, run_id, view = clustered
    config = _validated(
        experiment, run_id, clusters="ids", node_ids=_clusters(view, 1), compute=REMOTE
    )
    job = MODULE.build_job("an_99", config)
    target = RemoteTargetConfig(
        name="tillicum",
        host="t",
        remote_workdir="/x",
        gpus_per_job=4,
        cpus_per_task=16,
        mem_gb=128,
    )
    sized = remote_shepherd._sized_for(target, job)
    assert (sized.gpus_per_job, sized.cpus_per_task, sized.mem_gb) == (1, 4, 32)


def test_the_cluster_envelope_the_shepherd_writes_is_json(clustered, tillicum):
    """The shepherd sets every job's ``compute`` to the cluster's ``ComputeTarget`` and writes
    the job as ``job.json`` there. This job holds ``compute`` as a dict, and its ``to_dict``
    passed the ``ComputeTarget`` through as it was: the first real remote run failed at
    "submitting the job" with ``ComputeTarget is not JSON serializable``. The same replace and
    write the shepherd does, on a real job; and the worker reads back what it needs."""
    import dataclasses
    import json

    from detail.core.dataclasses import job_from_dict, job_to_dict

    experiment, run_id, view = clustered
    nodes = _clusters(view, 1)
    config = _validated(
        experiment, run_id, clusters="ids", node_ids=nodes, compute=REMOTE
    )
    job = MODULE.build_job("an_99", config)
    target = RemoteTargetConfig(name="tillicum", host="t", remote_workdir="/x")
    on_cluster = dataclasses.replace(
        job, compute=remote_shepherd._effective_compute(target)
    )

    written = json.loads(json.dumps(job_to_dict(on_cluster)))

    assert written["compute"]["backend"] == "cuda"
    read = job_from_dict(written)
    assert read.nodes == nodes and read.source_run_id == run_id
