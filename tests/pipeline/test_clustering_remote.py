"""Pose and temporal clustering on a remote target (§6 "Remote analysis", §22).

A behavior-clustering fit over a large selection takes hours here, so it can run on a cluster.
Run there, it has to come back as the run it would have been here. The classes guarded:

- **A module that cannot run remotely and says so only at the end.** ``run_analysis`` refuses
  a remote ``compute`` for a module that does not declare ``remote_capable``; both clustering
  modules declare it, carry the target into the job, and record it in ``analysis.yaml``.
- **A remote run that differs from a local one.** The whole path runs here, end to end: the
  shepherd uploads to a folder standing in for the cluster, its scheduler runs the real worker
  in that folder, and the run directory is copied back. A write the job makes outside its own
  run directory never comes back, so the copy's outputs are compared with a local run's, file by
  file, and the cluster's experiment outside the run directory must be left as it was uploaded.
- **A fit that reads a stale copy of its input.** The upload never rewrites a run directory the
  cluster already has, so each embedding run the fit reads is sent again over it
  (``refreshed_inputs``); an embedding run that went up half-written is replaced before the fit.
- **An envelope the shepherd cannot write.** It sets ``compute`` to a ``ComputeTarget`` before it
  writes the cluster's ``job.json``; the job writes that as its dict.

Each module's local run is one real executor run; the remote one runs in this process (the
shepherd, then the worker inside the stand-in cluster), which is what lets it be compared.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from detail.api import remote as api_remote
from detail.api.experiment import Experiment
from detail.configs.remote import RemoteTargetConfig
from detail.core.dataclasses import JobStatus, job_from_dict, job_to_dict
from detail.executor import remote as remote_module
from detail.executor import remote_shepherd, rundir, ssh_transport, worker
from detail.executor.runners import ANALYSIS_MODULES
from detail.executor.schedulers.base import JobState
from detail.pipeline import clustering_remote
from detail.storage import recording as recording_store
from detail.storage import remote_targets as remote_targets_store
from detail.storage import run_facts
from tests.experiments import (
    build_experiment_dir,
    build_recording_dir,
    write_embedding_run,
)

WORKDIR = "/scratch/detail_runs"
REMOTE = {
    "backend": "remote",
    "device_indices": None,
    "cloud_provider": None,
    "cloud_instance": None,
    "remote_target": "tillicum",
}
STEP = 17_000_000
VIDEO_TAIL = 60
MODULES = ("temporal_clustering",)

#: Files a run directory holds that say *how* it ran rather than what it found: the envelope,
#: the logs, the progress sampled along the way, and the machine it ran on with the memory it
#: used (``storage/run_facts``). Everything else must match a local run. ``machine.yaml`` differs
#: here even on one box: the remote job's worker runs in this process, so it reports torch
#: whenever an earlier test in the same pytest worker imported it, and the local subprocess
#: does not.
_HOW_IT_RAN = frozenset(
    {
        "job.json",
        "shepherd.log",
        "worker.log",
        "activity.csv",
        run_facts.MACHINE_FILENAME,
        run_facts.RESOURCES_FILENAME,
    }
)


def _config(module_id: str) -> dict:
    """A small fit each module finishes in seconds, on two recordings."""
    del module_id
    method = {
        "method": "dtw_finch",
        "method_params": {
            "window_s": 0.05,
            "min_len_s": 0.1,
            "band_s": 0.05,
            "penalty": 1.0,
            "min_size_s": 0.05,
        },
    }
    return {
        "source_recording_ids": ["recording_001", "recording_002"],
        "embed_run_id": "emb_01",
        "track_id": 0,
        **method,
    }


def _experiment(root: Path) -> Path:
    """Two recordings, each with an embedding run of repeating kinds the fits can find."""
    root.mkdir(parents=True, exist_ok=True)
    experiment_dir = build_experiment_dir(root)
    first = build_recording_dir(experiment_dir)
    second = experiment_dir / "recordings" / "recording_002"
    shutil.copytree(first, second)
    data = recording_store.read_recording(second)
    recording_store.write_recording(
        second, {**data, "id": "recording_002", "name": "recording_002"}
    )
    rng = np.random.default_rng(3)
    dim, kinds, rows = 8, 3, 30
    anchors = rng.normal(size=(kinds, 2, dim))
    for index, recording_dir in enumerate((first, second)):
        parts = []
        for bout in range(12):
            ramp = np.linspace(0.0, 1.0, rows)[:, None]
            a, b = anchors[(bout + index) % kinds]
            parts.append(
                a * (1 - ramp) + b * ramp + 0.05 * rng.normal(size=(rows, dim))
            )
        features = np.vstack(parts).astype(np.float32)
        frame_idx = np.arange(len(features), dtype=np.int64)
        recording_store.write_timestamps_csv(
            recording_dir / "video/timestamps/generated/part_001.csv",
            np.arange(len(features) + VIDEO_TAIL, dtype=np.int64) * STEP,
        )
        write_embedding_run(
            recording_dir, features, frame_idx, frame_idx * STEP, embed_run_id="emb_01"
        )
    return experiment_dir


class _Cluster:
    """A folder standing in for the cluster.

    ``ssh_transport``'s calls act on it as rsync and ssh would (the upload leaves alone what a
    run directory the cluster has already holds; a refresh is an exact mirror; the copy-back is
    the run directory), and its scheduler runs the real worker inside it, in this process, at
    submission.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.submitted: list[str] = []

    def path(self, remote: str) -> Path:
        return self.root / remote.removeprefix(WORKDIR).lstrip("/")

    # -- ssh_transport ---------------------------------------------------------------- #
    def copy_to_remote(
        self, host, local_dir, remote_dir, *, never_overwrite=(), never_send=(), **_k
    ):
        local_dir = Path(local_dir)
        held = [Path(p).resolve() for p in never_overwrite]
        for source in local_dir.rglob("*"):
            if not source.is_file():
                continue
            dest = self.path(remote_dir) / source.relative_to(local_dir)
            in_held = any(source.resolve().is_relative_to(h) for h in held)
            if in_held and dest.exists():
                continue  # --ignore-existing in the held directories' own pass
            if source.name in never_send and not in_held:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)

    def mirror_to_remote(self, host, local_dir, remote_dir, *, keep_remote=(), **_k):
        dest = self.path(remote_dir)
        kept = {
            name: (dest / name).read_bytes()
            for name in keep_remote
            if (dest / name).exists()
        }
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(local_dir, dest, ignore=shutil.ignore_patterns(*keep_remote))
        for name, data in kept.items():
            (dest / name).write_bytes(data)

    def copy_from_remote(self, host, remote_dir, local_dir, *, keep_local=(), **_k):
        shutil.copytree(
            self.path(remote_dir),
            local_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*keep_local),
        )

    def write_remote_file(self, host, path, content, **_k):
        dest = self.path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    def read_remote_file(self, host, path, **_k):
        source = self.path(path)
        if not source.exists():
            raise ssh_transport.RemoteCommandError(["cat", path], 1, "", "no such file")
        return source.read_text(encoding="utf-8-sig")

    def run_remote(self, host, command, **_k):
        words = command.split()
        stdout = ""
        if words[0] == "cat":
            target = self.path(words[1].strip("'"))
            stdout = target.read_text(encoding="utf-8-sig") if target.exists() else ""
        elif words[:2] == ["rm", "-rf"]:
            shutil.rmtree(self.path(words[2].strip("'")), ignore_errors=True)
        elif words[:2] == ["mkdir", "-p"]:
            self.path(words[2].strip("'")).mkdir(parents=True, exist_ok=True)
        else:
            raise AssertionError(f"unexpected remote command: {command}")
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    # -- the scheduler ---------------------------------------------------------------- #
    def submit_script(self, target, script, remote_run_dir):
        self.submitted.append(remote_run_dir)
        worker.run_job(self.path(remote_run_dir))
        return str(len(self.submitted))

    def poll_state(self, target, remote_job_id):
        return JobState.COMPLETED

    def cancel(self, target, remote_job_id):
        return True


class _InProcessRemote(remote_module.RemoteExecutor):
    """The executor ``run_analysis`` picks for a remote ``compute``, running its shepherd here
    rather than in a subprocess, so the stand-in cluster is in reach."""

    def submit(self, job):
        spec = remote_module.JOB_RUNNERS[remote_module.job_type_tag(job)]
        run_dir = Path(spec.run_dir_for(self._experiment_root, job))
        run_dir.mkdir(parents=True, exist_ok=True)
        rundir.init_run_dir(run_dir, job)
        self._runs[job.job_id] = run_dir
        remote_shepherd.run(run_dir, self._target, Path(self._experiment_root))
        return job.job_id


_TRANSPORT = (
    "copy_to_remote",
    "mirror_to_remote",
    "copy_from_remote",
    "write_remote_file",
    "read_remote_file",
    "run_remote",
)


def _tillicum(mp) -> None:
    target = RemoteTargetConfig(
        name="tillicum", host="tillicum", remote_workdir=WORKDIR
    )
    mp.setattr(api_remote, "get_remote_target", lambda name: target)
    mp.setattr(remote_targets_store, "get_remote_target", lambda name: target)


def _use(mp, cluster: _Cluster) -> None:
    for name in _TRANSPORT:
        mp.setattr(ssh_transport, name, getattr(cluster, name))
    mp.setattr(ssh_transport, "is_reachable", lambda host, **_k: True)
    mp.setitem(remote_shepherd._SCHEDULERS, "slurm", cluster)


def _launch(experiment, module_id: str, compute: dict | None) -> str:
    config = _config(module_id)
    if compute is not None:
        config["compute"] = compute
    run = experiment.run_analysis(module_id, config)
    run.wait()
    return run.id


def _tree(directory: Path) -> dict[str, bytes]:
    return {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


@dataclasses.dataclass
class _Pair:
    experiment: Experiment
    local: str
    remote: str
    cluster: _Cluster
    #: The cluster's experiment outside the remote run, as uploaded and after the job.
    uploaded: dict[str, bytes]
    after: dict[str, bytes]


@pytest.fixture(scope="module")
def pairs(tmp_path_factory) -> dict[str, _Pair]:
    """Per module, one fit run here and the same fit run on the stand-in cluster."""
    out: dict[str, _Pair] = {}
    with pytest.MonkeyPatch.context() as mp:
        _tillicum(mp)
        mp.setattr(remote_shepherd.time, "sleep", lambda *_: None)
        mp.setattr(remote_module, "RemoteExecutor", _InProcessRemote)
        for module_id in MODULES:
            experiment = Experiment(_experiment(tmp_path_factory.mktemp(module_id)))
            cluster = _Cluster(tmp_path_factory.mktemp(f"{module_id}_cluster"))
            _use(mp, cluster)
            snapshot: dict[str, bytes] = {}
            real_submit = cluster.submit_script

            def submit(target, script, remote_run_dir, _real=real_submit, _c=cluster):
                root = _c.path(remote_run_dir).parent.parent
                snapshot.update(_outside(root, _c.path(remote_run_dir)))
                return _real(target, script, remote_run_dir)

            mp.setattr(cluster, "submit_script", submit)
            remote = _launch(experiment, module_id, REMOTE)
            # Here second: a stray write by the first would otherwise be uploaded, and the
            # cluster's copy of it would read as the upload's.
            local = _launch(experiment, module_id, None)
            root = cluster.path(cluster.submitted[0]).parent.parent
            out[module_id] = _Pair(
                experiment,
                local,
                remote,
                cluster,
                uploaded=snapshot,
                after=_outside(root, cluster.path(cluster.submitted[0])),
            )
    return out


def _outside(experiment_root: Path, run_dir: Path) -> dict[str, bytes]:
    """Every file of the experiment except the run directory's own."""
    return {
        rel: data
        for rel, data in _tree(experiment_root).items()
        if not (experiment_root / rel).is_relative_to(run_dir)
    }


@pytest.mark.parametrize("module_id", MODULES)
def test_both_clustering_modules_may_run_remotely(module_id):
    module = ANALYSIS_MODULES[module_id]
    assert module.remote_capable
    assert module.remote_resources(None) == {"gpus": 1}


@pytest.mark.parametrize("module_id", MODULES)
def test_a_remote_run_completes_and_records_its_target(pairs, module_id):
    pair = pairs[module_id]
    run_dir = pair.experiment.analysis_run_dir(pair.remote)
    assert rundir.read_status(run_dir) == JobStatus.COMPLETE, rundir.read_envelope(
        run_dir
    ).get("error")
    record = pair.experiment.analysis_record(pair.remote)
    assert record["compute"]["remote_target"] == "tillicum"
    # Where it ran is the record's `compute`; its params are the job the worker ran, which on
    # the cluster names the cluster's own backend.
    assert record["params"]["compute"]["backend"] == "cuda"
    # The fit leaves an authored run open, on the cluster as here (§21).
    assert record["status"] == "open"
    local = pair.experiment.analysis_record(pair.local)
    assert "compute" not in local and local["params"]["compute"] is None
    # The envelope here keeps the job as submitted; the cluster's names its own backend.
    assert rundir.read_job(run_dir).compute["backend"] == "remote"
    on_cluster = json.loads(
        pair.cluster.path(pair.cluster.submitted[0] + "/job.json").read_text(
            encoding="utf-8-sig"
        )
    )
    assert on_cluster["job"]["compute"]["backend"] == "cuda"


@pytest.mark.parametrize("module_id", MODULES)
def test_the_remote_run_found_what_the_local_run_found(pairs, module_id):
    """Every output, file by file, after the copy-back: the fit, the §19 artifact, the frame
    views and the node cache. Only the record of how it ran may differ — where (``compute``),
    the run's own id, and the instants it was written at."""
    pair = pairs[module_id]
    local = _tree(pair.experiment.analysis_run_dir(pair.local))
    remote = _tree(pair.experiment.analysis_run_dir(pair.remote))
    outputs = sorted(set(local) - _HOW_IT_RAN)
    assert "analysis.yaml" in outputs and any("node_cache" in rel for rel in outputs)
    assert sorted(set(remote) - _HOW_IT_RAN) == outputs
    differ = [
        rel
        for rel in outputs
        if _normalized(rel, local[rel], pair.local)
        != _normalized(rel, remote[rel], pair.remote)
    ]
    assert (
        not differ
    ), f"{module_id}: the remote run's {differ} differ from a local run's"


def _normalized(rel: str, data: bytes, run_id: str) -> bytes:
    """A file with where it ran, its run's id and the instants it was written at taken out.

    The fit's modification time is one of those instants: the node cache records it and never
    compares it (``NodeView.stale_reason``), so a copied fit does not read as a stale cache.
    """
    if rel == "analysis.yaml":
        record = yaml.safe_load(data)
        record.pop("compute", None)
        record.get("params", {}).pop("compute", None)
        data = yaml.safe_dump(record, sort_keys=False).encode("utf-8")
    text = data.decode("utf-8", errors="surrogateescape").replace(run_id, "<run>")
    text = re.sub(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z?", "<at>", text)
    text = re.sub(r'"fit_mtime_ns": \d+', '"fit_mtime_ns": <at>', text)
    return text.encode("utf-8", errors="surrogateescape")


@pytest.mark.parametrize("module_id", MODULES)
def test_the_job_writes_nothing_on_the_cluster_outside_its_run_directory(
    pairs, module_id
):
    """A write anywhere else never comes back: the copy-back is the run directory."""
    pair = pairs[module_id]
    assert pair.uploaded
    changed = sorted(
        rel
        for rel in set(pair.uploaded) | set(pair.after)
        if pair.uploaded.get(rel) != pair.after.get(rel)
    )
    assert not changed, f"{module_id} wrote outside its run directory: {changed}"


@pytest.mark.parametrize("module_id", MODULES)
def test_each_embedding_run_the_fit_reads_is_refreshed(pairs, module_id):
    pair = pairs[module_id]
    job = rundir.read_job(pair.experiment.analysis_run_dir(pair.remote))
    root = pair.experiment.dir
    assert ANALYSIS_MODULES[module_id].refreshed_inputs(root, job) == [
        root / "recordings" / rid / "embeddings" / "emb_01"
        for rid in ("recording_001", "recording_002")
    ]


@pytest.mark.parametrize("module_id", MODULES)
def test_a_half_uploaded_embedding_run_is_replaced_before_the_fit(
    tmp_path, monkeypatch, module_id
):
    """The upload never rewrites a run directory the cluster has, so an embedding run that went
    up while it was still being written keeps its first copy there; the refresh replaces it.
    """
    experiment = Experiment(_experiment(tmp_path / "exp"))
    cluster = _Cluster(tmp_path / "cluster")
    _tillicum(monkeypatch)
    _use(monkeypatch, cluster)
    store = experiment.dir / "recordings" / "recording_001" / "embeddings" / "emb_01"
    stale = cluster.path(f"{WORKDIR}/{experiment.dir.name}") / store.relative_to(
        experiment.dir
    )
    shutil.copytree(store, stale)
    for path in stale.rglob("zarr.json"):
        path.write_text("{}", encoding="utf-8")  # the first, shorter store
    (stale / "left_over_by_a_cancelled_job").write_text("x", encoding="utf-8")
    module = ANALYSIS_MODULES[module_id]
    job = module.build_job(
        "an_09",
        module.validate(experiment.dir, {**_config(module_id), "compute": REMOTE}),
    )
    transfer = types.SimpleNamespace(phase=lambda message: None)

    remote_shepherd._refresh_inputs(
        RemoteTargetConfig(name="tillicum", host="tillicum", remote_workdir=WORKDIR),
        experiment.dir,
        f"{WORKDIR}/{experiment.dir.name}",
        job,
        transfer,
    )

    def outputs(directory):
        return {rel: d for rel, d in _tree(directory).items() if rel != "job.json"}

    assert outputs(stale) == outputs(store)


@pytest.mark.parametrize("module_id", MODULES)
def test_the_cluster_envelope_the_shepherd_writes_is_json(
    tmp_path, monkeypatch, module_id
):
    experiment = Experiment(_experiment(tmp_path / "exp"))
    _tillicum(monkeypatch)
    module = ANALYSIS_MODULES[module_id]
    job = module.build_job(
        "an_09",
        module.validate(experiment.dir, {**_config(module_id), "compute": REMOTE}),
    )
    on_cluster = dataclasses.replace(
        job,
        compute=remote_shepherd._effective_compute(
            RemoteTargetConfig(name="tillicum", host="t", remote_workdir=WORKDIR)
        ),
    )

    written = json.loads(json.dumps(job_to_dict(on_cluster)))

    assert written["compute"]["backend"] == "cuda"
    assert job_from_dict(written).source_recording_ids == job.source_recording_ids


@pytest.mark.parametrize("module_id", MODULES)
def test_the_shepherds_phases_are_the_modules_own_event(module_id):
    """Uploading, refreshing and queueing are published as the module's own progress event,
    which every surface already renders, before the worker's first event."""
    module = ANALYSIS_MODULES[module_id]
    job = module.job_class.from_dict(
        {"job_id": "an_09", "method": "finch", "compute": REMOTE}
    )
    event = remote_shepherd._phase_event(
        job, message="uploading experiment files", elapsed_s=1.0, eta_s=None
    )
    assert isinstance(event, module.progress_event_class)
    assert event.message == "uploading experiment files"
    assert (event.recordings_done, event.frames_done) == (0, 0)


@pytest.mark.parametrize("module_id", MODULES)
def test_an_unregistered_target_is_refused_before_anything_is_read(tmp_path, module_id):
    experiment = Experiment(_experiment(tmp_path / "exp"))
    with pytest.raises(ValueError, match="no remote target 'tillicum' is registered"):
        ANALYSIS_MODULES[module_id].validate(
            experiment.dir, {**_config(module_id), "compute": REMOTE}
        )


@pytest.mark.parametrize("module_id", MODULES)
def test_a_job_written_before_compute_existed_reads_as_local(module_id):
    job_class = ANALYSIS_MODULES[module_id].job_class
    old = {"job_id": "an_03", "method": "finch", "source_recording_ids": ["r"]}
    assert job_class.from_dict(old).compute is None
    assert clustering_remote.compute_dict(None) is None
