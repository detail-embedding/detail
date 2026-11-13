"""Find look-alike clusters on a remote target (§6 "Remote analysis", §22 "Look-alike clusters").

``pairs: all`` is hundreds of pairs and many hours on CPU, so the job can run on a cluster. The
classes guarded here:

- **A job whose results stay on the cluster.** The comparisons belong beside the clustering
  run, which a copy-back (the job's own run directory) never returns. The job writes each one
  into its own run directory and places it from there, and ``after_copy_back`` places them all
  here — idempotently, since a re-run calls it again.
- **A skip check that reads a stale copy.** The job names the clustering run it reads, so the
  pairs already placed here are re-sent and skipped there.
- **A remote target that quietly becomes this machine**, or one not registered here: the target
  is validated and carried into the job, or refused before anything is sent.
- **An envelope the shepherd cannot write.** It sets ``compute`` to a ``ComputeTarget`` before
  writing ``job.json``; the job must write that as its dict.

The jobs run their runner in-process, as the module tests do.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import yaml

import detail.api.cluster_lookalikes as lookalikes
from detail.api import profile_describe
from detail.api import remote as api_remote
from detail.configs.remote import RemoteTargetConfig
from detail.core.dataclasses import job_from_dict, job_to_dict
from detail.executor import remote_shepherd
from detail.storage import analysis_runs as analysis_store
from tests.pipeline.test_cluster_lookalikes_module import (  # noqa: F401 — fixtures
    MODULE,
    SETTINGS,
    _a_floor_the_fixture_clears,
    _launch,
    _run,
    named,
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


def _comparisons(run_dir) -> set[str]:
    return {
        p.relative_to(run_dir).as_posix()
        for p in (run_dir / "profiles").rglob("against_*/*")
        if p.is_file()
    }


def test_a_remote_job_carries_its_target_and_names_the_run_it_reads(named, tillicum):
    experiment, run_id, _view, _nodes = named
    job, _run_dir = _launch(experiment, run_id, compute=REMOTE)
    assert job.compute["backend"] == "remote"
    assert job.compute["remote_target"] == "tillicum"
    assert MODULE.remote_capable
    assert MODULE.refreshed_inputs(experiment.dir, job) == [
        analysis_store.analysis_dir(experiment.dir, run_id)
    ]


def test_a_remote_job_holds_one_gpu_and_the_node_shrinks_with_it(named, tillicum):
    """CPU work on a target sized for a model run: the job asks for the one GPU Tillicum
    insists on, not the target's eight, and its CPUs and memory scale down with it."""
    from detail.executor.schedulers.slurm import render_sbatch_script

    experiment, run_id, _view, _nodes = named
    job, _run_dir = _launch(experiment, run_id, compute=REMOTE)
    target = RemoteTargetConfig(
        name="tillicum",
        host="t",
        remote_workdir="/x",
        gpus_per_job=8,
        cpus_per_task=64,
        mem_gb=1600,
    )

    sized = remote_shepherd._sized_for(target, job)

    assert (sized.gpus_per_job, sized.cpus_per_task, sized.mem_gb) == (1, 8, 200)
    script = render_sbatch_script(sized, "an_01", "/x/run", "detail.pipeline")
    assert "#SBATCH --gres=gpu:1\n" in script


def test_a_local_job_records_no_target(named):
    experiment, run_id, _view, _nodes = named
    job, record, _events = _run(experiment, run_id)
    assert job.compute is None
    assert record["params"]["compute"] is None


def test_an_unregistered_target_is_refused(named):
    experiment, run_id, _view, _nodes = named
    with pytest.raises(ValueError, match="no remote target 'tillicum' is registered"):
        _launch(experiment, run_id, compute=REMOTE)


def test_a_cloud_target_is_refused(named):
    experiment, run_id, _view, _nodes = named
    with pytest.raises(ValueError, match="find look-alike clusters on a cloud target"):
        _launch(experiment, run_id, compute={**REMOTE, "backend": "cloud"})


def test_each_comparison_is_the_jobs_own_and_placed_beside_the_run(named):
    """What a copy-back returns is the job's run directory, so that is where the job writes;
    the run's copy is placed from it."""
    experiment, run_id, _view, _nodes = named
    job, record, _events = _run(experiment, run_id)
    own = _comparisons(experiment.analysis_run_dir(job.job_id))
    placed = _comparisons(experiment.analysis_run_dir(run_id))
    assert len(own) == 6, "profile.yaml and measures.csv for each of the three pairs"
    assert own == placed
    assert {p["file"] for p in record["pairs"]} <= placed
    assert record["artifacts"] == MODULE.output_manifest(job.job_id, {})


def test_after_the_copy_back_the_comparisons_are_placed_beside_the_run(named, tillicum):
    """What the cluster's copy of the job placed never comes back; the job's own folder does,
    and placing from it puts every comparison where the groups read it — twice is once.
    """
    experiment, run_id, _view, nodes = named
    job, _record, _events = _run(experiment, run_id, compute=REMOTE)
    source = experiment.analysis_run_dir(run_id)
    for folder in sorted((source / "profiles").rglob("against_*")):
        for path in folder.iterdir():  # as on this machine before the copy-back
            path.unlink()
        folder.rmdir()
    assert lookalikes.read_evidence(source) == []

    run_dir = experiment.analysis_run_dir(job.job_id)
    MODULE.after_copy_back(experiment.dir, job, run_dir)
    first = {p: (source / p).read_bytes() for p in _comparisons(source)}
    MODULE.after_copy_back(experiment.dir, job, run_dir)

    assert len(lookalikes.read_evidence(source)) == 3
    assert {p: (source / p).read_bytes() for p in _comparisons(source)} == first


def test_a_newer_comparison_beside_the_run_is_not_replaced(named):
    """A pair compared by hand after the job ran is the run's newest word on it; placing the
    job's older copy again leaves it alone."""
    experiment, run_id, _view, nodes = named
    job, record, _events = _run(experiment, run_id)
    a, b = record["pairs"][0]["a"], record["pairs"][0]["b"]
    experiment.profile_cluster(
        run_id,
        a,
        {**SETTINGS, "compare": "node", "against": b, "leave_out_place": True},
    )
    # Timestamps are to the second; say "later" outright rather than racing the clock.
    path = experiment.analysis_run_dir(run_id) / record["pairs"][0]["file"]
    saved = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    saved["created_at"] = "2999-01-01T00:00:00Z"
    path.write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")

    profile_describe.place(experiment, job.job_id)

    assert experiment.cluster_profile(run_id, a, b).created_at == saved["created_at"]


def test_the_cluster_envelope_the_shepherd_writes_is_json(named, tillicum):
    experiment, run_id, _view, _nodes = named
    job, _run_dir = _launch(experiment, run_id, compute=REMOTE)
    target = RemoteTargetConfig(name="tillicum", host="t", remote_workdir="/x")
    on_cluster = dataclasses.replace(
        job, compute=remote_shepherd._effective_compute(target)
    )

    written = json.loads(json.dumps(job_to_dict(on_cluster)))

    assert written["compute"]["backend"] != "remote"
    read = job_from_dict(written)
    assert read.pair_list == job.pair_list and read.source_run_id == run_id


def test_a_job_written_before_compute_existed_reads_as_local(named):
    experiment, run_id, _view, _nodes = named
    job, _run_dir = _launch(experiment, run_id)
    old = {k: v for k, v in job.to_dict().items() if k != "compute"}
    assert type(job).from_dict(old).compute is None
