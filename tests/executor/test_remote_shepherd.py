"""remote_shepherd.run() — the local orchestration loop, exercised in-process against a
mocked ssh_transport and a fake scheduler. No real cluster, no real subprocess: this is
the "tests against a mocked transport" slice of the remote-executor design.

**The class guarded at the bottom (#644): a long phase that emits nothing until a boundary reads as
hung.** Before the remote worker wrote its first event, the shepherd uploaded the experiment, then
every referenced video (80 recordings was ~116 GB, about an hour), checked the model, submitted and
waited in the queue, and published nothing for any of it. The distill page sat on "starting
labeling" and the person watching could not tell a healthy run from a frozen one. It hid because the
upload worked perfectly the whole time: the defect is only visible in what the run *published*, so
those tests read `rundir.set_progress`. The cadence test runs on a **single-file** library (the
case the rule exists for: with one video the message has only a 0/1 and a 1/1 and everything between
is silent), asserts an intermediate event with a populated ETA, and asserts the stage's own
fraction is never touched (an invented one would run the bar to 100% and drop it back at labeling).

**The class guarded at the end of the file: being unable to ask was recorded as an answer.** The
shepherd asks Slurm about the job every few seconds over ssh. `SlurmScheduler.poll_state` fell
through to "neither squeue nor sacct know this job, so it failed" whenever both came back empty,
and an ssh that never connected is empty: a dropped or expired session (the laptop slept, the
network changed) recorded a healthy multi-hour job as `failed` while it carried on on the cluster.
The sync back at the end had the same shape: it failed quietly with the connection down, and the
run was recorded `complete` with no results on disk. It hid because every test gave the fake
scheduler a *definite* answer; none made the scheduler unable to answer. The rule, which
`executor/local.py::_is_our_worker` already states for its own `ps` ("every way of failing to get
an answer has to give the same answer, and it has to be the conservative one"): an ask that did
not happen is `UNREACHABLE`, never a terminal state. The run waits, visibly and with a backoff, and
carries on when the person logs in again.

**The class guarded after that: a mirror run in both directions treats the far side's snapshot as
the truth.** The shepherd uploaded the whole experiment at the start and, when the job ended,
copied the whole *remote* experiment back down over the local one. By then the local experiment had
moved on (a four-hour job is a long time to leave an app open): a recording renamed meanwhile, the
live `job.json` of another run, a run deleted from the Runs page. The copy down put the cluster's
snapshot from the start of the job over all of it, so the rename was reverted, the other run read as
never started, and the deleted run came back. Nothing failed, and with one job at a time it only
showed as edits that quietly did not stick. It hid because every test's fake `copy_from_remote`
was a stub; none had a filesystem on both sides that the job's life happened to. The rule: a job
writes only inside its run folder, so the shepherd brings back that folder and nothing else. The
tests below give both sides a real directory, change the local one while the "job" runs, and read
what is on this machine afterwards.
"""

import contextlib
import io
import itertools
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from detail.configs.remote import RemoteTargetConfig
from detail.core.dataclasses import (
    ComputeTarget,
    DetectionJob,
    DetectionProgressEvent,
    EmbeddingJob,
    EmbeddingProgressEvent,
    JobStatus,
    LabelingClass,
    LabelingJob,
    LabelingProgressEvent,
    ModelReadiness,
    Prompt,
    PromptTarget,
    TextCue,
    TrainingJob,
    TrainingProgressEvent,
    job_to_dict,
)
from detail.core.protocols import EmbeddingModel
from detail.core.registries import EMBEDDING_MODELS, register_embedding_model
from detail.executor import remote_shepherd, rundir, ssh_transport
from detail.executor.schedulers.base import JobState
from detail.storage import locations
from detail.storage import recording as recording_store
from detail.storage import remote_targets as remote_targets_store
from detail.storage import video_roots


def _target(**overrides):
    fields = dict(
        name="tillicum",
        host="tillicum",
        remote_workdir="/gscratch/scrubbed/uwnetid/detail_runs",
        gpus_per_job=2,
    )
    fields.update(overrides)
    return RemoteTargetConfig(**fields)


def _job(job_id="det_01"):
    return DetectionJob(
        job_id=job_id,
        recording_id="recording_001",
        model_id="sam3",
        prompt=Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        conf_threshold=0.5,
        compute=ComputeTarget("remote", None, None, None),
        created_at="2026-01-01T00:00:00Z",
    )


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _ready_run_remote(*a, **k):
    """Default stand-in for ``ssh_transport.run_remote`` — reports the job's model
    as installed, so tests not exercising the readiness probe itself aren't broken
    by it."""
    payload = json.dumps(
        {
            "ready": True,
            "reason": "ok",
            "message": "",
            "action_url": None,
            "repo_id": None,
        }
    )
    return _FakeCompletedProcess(returncode=0, stdout=payload)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    experiment_root = tmp_path / "exp1"
    recording_dir = experiment_root / "recordings" / "recording_001"
    run_dir = recording_dir / "runs" / "detect" / "det_01"
    run_dir.mkdir(parents=True)
    rundir.init_run_dir(run_dir, _job())
    # No video sources — these tests aren't exercising the video-root sync, so an
    # empty file list keeps it a no-op (see tests below for that behavior itself).
    recording_store.write_recording(
        recording_dir, {"id": "recording_001", "video": {"files": []}}
    )

    monkeypatch.setattr(remote_shepherd.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        remote_targets_store, "get_remote_target", lambda name: _target()
    )
    monkeypatch.setattr(ssh_transport, "copy_to_remote", lambda *a, **k: None)
    monkeypatch.setattr(ssh_transport, "copy_from_remote", lambda *a, **k: None)
    monkeypatch.setattr(ssh_transport, "write_remote_file", lambda *a, **k: None)
    monkeypatch.setattr(ssh_transport, "run_remote", _ready_run_remote)
    return experiment_root, run_dir


class _FakeScheduler:
    def __init__(self, states, job_id="999"):
        self._states = list(states)
        self._job_id = job_id
        self.cancel_calls = []
        self.submitted = []

    def submit_script(self, target, script, remote_run_dir):
        self.submitted.append((target, script, remote_run_dir))
        return self._job_id

    def poll_state(self, target, remote_job_id):
        # Falls back to RUNNING if a test's state list is shorter than the number of
        # polls it actually takes — a loud test failure (an assertion on the wrong
        # status) rather than a silent AttributeError, and never terminal by accident.
        return self._states.pop(0) if self._states else JobState.RUNNING

    def cancel(self, target, remote_job_id):
        self.cancel_calls.append(remote_job_id)


def _install_scheduler(monkeypatch, scheduler):
    monkeypatch.setitem(remote_shepherd._SCHEDULERS, "slurm", scheduler)


def test_effective_compute_uses_target_gpu_count():
    compute = remote_shepherd._effective_compute(_target(gpus_per_job=3))
    assert compute.backend == "cuda"
    assert compute.device_indices == [0, 1, 2]


def test_remote_paths_mirror_relative_layout(tmp_path):
    experiment_root = tmp_path / "exp1"
    run_dir = experiment_root / "recordings" / "r1" / "runs" / "detect" / "det_01"
    target = _target(remote_workdir="/scratch/x/")
    exp_root, run_root = remote_shepherd._remote_paths(target, experiment_root, run_dir)
    assert exp_root == "/scratch/x/exp1"
    assert run_root == "/scratch/x/exp1/recordings/r1/runs/detect/det_01"


def test_run_forwards_runner_modules_into_the_submitted_script(monkeypatch, experiment):
    """RemoteExecutor.submit() sets DETAIL_RUNNER_MODULES in the shepherd's own
    environment (executor/remote.py); the compute node's sbatch shell inherits none
    of it, so the shepherd must read it back and hand it to render_sbatch_script.
    Missing this is the class of bug behind "no runner registered for job type
    'detection'" on a real cluster: the worker started, conda activated fine, and its
    job-type registry was simply empty."""
    experiment_root, run_dir = experiment
    scheduler = _FakeScheduler([JobState.RUNNING])
    _install_scheduler(monkeypatch, scheduler)
    monkeypatch.setenv(
        "DETAIL_RUNNER_MODULES", "detail.pipeline.detection,detail.pipeline.tracking"
    )
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "complete", "progress": None}),
    )

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    submitted_script = scheduler.submitted[0][1]
    assert (
        "export DETAIL_RUNNER_MODULES=detail.pipeline.detection,detail.pipeline.tracking"
        in submitted_script
    )


def test_run_success_mirrors_progress_and_completes(monkeypatch, experiment):
    experiment_root, run_dir = experiment
    scheduler = _FakeScheduler([JobState.RUNNING])
    _install_scheduler(monkeypatch, scheduler)

    envelopes = [
        {"status": "running", "progress": {"type": "detection", "parts_done": 1}},
        {
            "status": "complete",
            "progress": {"type": "detection", "parts_done": 2},
            "error": None,
        },
    ]

    def fake_read(host, path, **k):
        return json.dumps(envelopes.pop(0) if len(envelopes) > 1 else envelopes[0])

    monkeypatch.setattr(ssh_transport, "read_remote_file", fake_read)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    assert rundir.read_status(run_dir) == JobStatus.COMPLETE
    assert rundir.read_envelope(run_dir)["remote_job_id"] == "999"
    # the GPUs it held, for its cost: detection takes the target's own (2)
    assert rundir.read_envelope(run_dir)["remote_gpus"] == 2
    assert scheduler.submitted  # sbatch was submitted
    # The shepherd's own RUNNING/terminal transitions (submission, then completion)
    # stamp local timing (§ run location, duration, cost) — submission-to-completion,
    # including any time the job sat queued on Slurm (decided with the user).
    envelope = rundir.read_envelope(run_dir)
    assert envelope["started_at"] and envelope["completed_at"]


def test_the_workers_envelope_never_replaces_this_machines_compute_or_start(
    monkeypatch, experiment
):
    """The remote worker's own job.json is written under _effective_compute
    (``backend="cuda"``, never ``"remote"``) and carries no local ``started_at``. When the copy
    back laid it over the local one, the family's own ``set_*_status`` read that envelope, so a
    completed remote run's record permanently forgot it ever left this machine: no "remote:
    tillicum" in the Active-runs "where" column, no duration, no cost. Found live against a real
    Tillicum distill run, where a training run and its labeling run both showed "local" and no
    duration once finished, despite showing "remote: tillicum" correctly while still running.

    The copy back no longer brings the envelope at all (#742), and the worker's fields are
    merged in after it (`_merge_worker_envelope`), minus what this machine owns. This pins that
    the merge, like the old restore, never lets the worker's compute or clock stand in for ours.
    """
    experiment_root, run_dir = experiment
    scheduler = _FakeScheduler([JobState.RUNNING])
    _install_scheduler(monkeypatch, scheduler)
    cluster_job = job_to_dict(_job())
    cluster_job["compute"] = {
        "backend": "cuda",
        "device_indices": [0, 1],
        "cloud_provider": None,
        "cloud_instance": None,
        "remote_target": None,
    }
    worker_envelope = {
        "schema_version": 1,
        "status": "complete",
        "job": cluster_job,
        "progress": None,
        "error": None,
        "completed_at": "2026-01-01T00:05:00Z",
    }
    monkeypatch.setattr(
        ssh_transport, "read_remote_file", lambda *a, **k: json.dumps(worker_envelope)
    )

    def _fake_copy_from_remote(host, remote_root, local_root, *, keep_local=(), **k):
        # A real copy back: the cluster's envelope comes too unless it is kept local.
        if rundir.JOB_FILENAME not in keep_local:
            (run_dir / "job.json").write_text(
                json.dumps(worker_envelope), encoding="utf-8"
            )

    monkeypatch.setattr(ssh_transport, "copy_from_remote", _fake_copy_from_remote)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    envelope = rundir.read_envelope(run_dir)
    assert (
        envelope["job"]["compute"]["backend"] == "remote"
    ), "the worker's on-cluster compute leaked into the final record"
    assert envelope[
        "started_at"
    ], "the local started_at was lost -- duration_s reads null"
    assert envelope["worker_completed_at"] == "2026-01-01T00:05:00Z"


@pytest.mark.parametrize(
    "cluster_job, replaced",
    [
        # A deleted run's id handed out again: the cluster still holds the old run's output.
        ({"job_id": "det_01", "created_at": "2025-06-01T00:00:00Z"}, True),
        # A previous attempt at this same job (resume, restarted shepherd): its work is kept.
        ({"job_id": "det_01", "created_at": "2026-01-01T00:00:00Z"}, False),
        # Nothing there, or nothing readable: nothing proves it is another job's.
        (None, False),
    ],
)
def test_a_cluster_run_dir_left_by_another_job_is_replaced_before_the_upload(
    monkeypatch, experiment, cluster_job, replaced
):
    """The upload never overwrites a run directory the cluster already has, assuming this job's
    own is not there yet. A reused run id broke that: a 12-keypoint run on tillicum ran inside a
    deleted run's directory and opened its 4-keypoint zarr."""
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "complete", "progress": None}),
    )
    remote_run_dir = remote_shepherd._remote_paths(_target(), experiment_root, run_dir)[
        1
    ]
    events = []

    def fake_run_remote(host, command, **k):
        if command.startswith("cat ") and "job.json" in command:
            events.append("read")
            text = json.dumps({"job": cluster_job}) if cluster_job else ""
            return _FakeCompletedProcess(stdout=text)
        if command.startswith("rm -rf "):
            assert remote_run_dir in command
            events.append("remove")
            return _FakeCompletedProcess()
        return _ready_run_remote(host, command, **k)

    monkeypatch.setattr(ssh_transport, "run_remote", fake_run_remote)
    monkeypatch.setattr(
        ssh_transport, "copy_to_remote", lambda *a, **k: events.append("upload")
    )

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    expected = ["read", "remove", "upload"] if replaced else ["read", "upload"]
    assert events[: len(expected)] == expected


def test_run_records_remote_error_on_failure(monkeypatch, experiment):
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "failed", "error": "CUDA OOM"}),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    assert "CUDA OOM" in rundir.read_envelope(run_dir)["error"]


def test_run_scheduler_terminal_without_worker_status_falls_back_to_failed(
    monkeypatch, experiment
):
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.FAILED]))
    # The remote job.json is never readable (crashed before ever writing) — worker
    # never got the chance to record its own status.
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: (_ for _ in ()).throw(
            ssh_transport.RemoteCommandError(["ssh"], 1, "", "no such file")
        ),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    assert "without a terminal status" in rundir.read_envelope(run_dir)["error"]


def test_run_scheduler_cancelled_without_worker_status_maps_to_cancelled(
    monkeypatch, experiment
):
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.CANCELLED]))
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: (_ for _ in ()).throw(
            ssh_transport.RemoteCommandError(["ssh"], 1, "", "no such file")
        ),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.CANCELLED


def test_run_cancel_requested_calls_scheduler_cancel(monkeypatch, experiment):
    experiment_root, run_dir = experiment

    class _CancelledOnceQueued(_FakeScheduler):
        def submit_script(self, target, script, remote_run_dir):
            rundir.request_cancel(run_dir)  # the cancel lands while the job is queued
            return super().submit_script(target, script, remote_run_dir)

    scheduler = _CancelledOnceQueued([JobState.RUNNING, JobState.CANCELLED])
    _install_scheduler(monkeypatch, scheduler)
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: (_ for _ in ()).throw(
            ssh_transport.RemoteCommandError(["ssh"], 1, "", "no such file")
        ),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert scheduler.cancel_calls == ["999"]
    assert status == JobStatus.CANCELLED


def test_run_submission_failure_records_local_failed(monkeypatch, experiment):
    experiment_root, run_dir = experiment
    monkeypatch.setattr(
        remote_targets_store,
        "get_remote_target",
        lambda name: (_ for _ in ()).throw(KeyError("nope")),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    assert (
        "nope" in rundir.read_envelope(run_dir)["error"]
        or "KeyError" in rundir.read_envelope(run_dir)["error"]
    )


def test_run_sync_back_failure_is_a_failure_not_a_complete_with_no_results(
    monkeypatch, experiment
):
    """It used to keep the job's own status, so a run whose results never arrived read
    **complete** with nothing on disk. A job that finished on the cluster but could not be copied
    back is a failure that says where its results are (the status of one that had *already*
    failed or been cancelled is kept: tests/executor/test_settled_end.py)."""
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "complete", "progress": None}),
    )
    monkeypatch.setattr(
        ssh_transport,
        "copy_from_remote",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network blip")),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    assert rundir.read_status(run_dir) == JobStatus.FAILED
    assert "could not be copied back" in rundir.read_envelope(run_dir)["error"]


# --- video-root sync (raw video lives outside the experiment directory) ---- #
def _register_local_video_root(monkeypatch, tmp_path, alias, video_dir):
    config_dir = tmp_path / "config" / "detail"
    monkeypatch.setattr(locations, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(locations, "STORAGE_PATH", config_dir / "storage.yaml")
    video_roots.register_video_root(alias, video_dir)


def _write_recording_with_video(recording_dir, files):
    recording_dir.mkdir(parents=True, exist_ok=True)
    recording_store.write_recording(
        recording_dir, {"id": recording_dir.name, "video": {"files": files}}
    )


def _training_job():
    return TrainingJob(
        job_id="model_001",
        model_id="model_001",
        dataset_id="dataset_001",
        trainer_id="ultralytics_yolo",
        base_model="yolo11n.pt",
        hyperparams={},
        compute=ComputeTarget("remote", None, None, None),
        created_at="2026-01-01T00:00:00Z",
    )


def _labeling_job(source_recording_ids):
    return LabelingJob(
        job_id="dataset_001",
        dataset_id="dataset_001",
        name="d",
        source_recording_ids=source_recording_ids,
        classes=[LabelingClass(0, "mouse", "a mouse")],
        n_frames=4,
        logit_threshold=0.5,
        min_contour_area=100,
        val_split=0.15,
        compute=ComputeTarget("remote", None, None, None),
        created_at="2026-01-01T00:00:00Z",
    )


def test_job_recording_ids_by_job_type():
    assert remote_shepherd._job_recording_ids(_job()) == ["recording_001"]  # detection
    assert remote_shepherd._job_recording_ids(_embedding_job()) == ["recording_001"]
    assert remote_shepherd._job_recording_ids(_labeling_job(["r1", "r2"])) == [
        "r1",
        "r2",
    ]
    # training works from an already-labeled dataset — no raw video at all
    assert remote_shepherd._job_recording_ids(_training_job()) == []


def test_referenced_video_sources_skips_copied_mode_files(tmp_path):
    experiment_root = tmp_path / "exp1"
    recording_dir = experiment_root / "recordings" / "recording_001"
    _write_recording_with_video(
        recording_dir,
        [
            {"index": 0, "source": {"root": "Cropped Videos", "rel": "a.mp4"}},
            {"index": 1, "file": "video/part_002.mp4"},  # copied mode: no "source"
        ],
    )
    sources = remote_shepherd._referenced_video_sources(
        experiment_root, ["recording_001"]
    )
    assert sources == [{"root": "Cropped Videos", "rel": "a.mp4"}]


def test_sync_video_roots_copies_the_file_and_registers_the_alias_remotely(
    tmp_path, monkeypatch
):
    """Found live: a real distill job's labeling stage failed on the cluster with
    MissingVideoRootError — the video root is registered on *this* machine, never on
    the one that actually runs the job, because raw video lives outside the
    experiment directory the whole-tree mirror sweeps up."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_file = video_dir / "a.mp4"
    video_file.write_bytes(b"fake video bytes")
    _register_local_video_root(monkeypatch, tmp_path, "Cropped Videos", video_dir)

    experiment_root = tmp_path / "exp1"
    recording_dir = experiment_root / "recordings" / "recording_001"
    _write_recording_with_video(
        recording_dir,
        [{"index": 0, "source": {"root": "Cropped Videos", "rel": "a.mp4"}}],
    )

    copied = []
    monkeypatch.setattr(
        ssh_transport,
        "copy_file_to_remote",
        lambda host, local_path, remote_path, **k: copied.append(
            (host, local_path, remote_path)
        ),
    )
    registered = []

    def fake_run_remote(host, command, *, stdin_text=None, check=True, **k):
        registered.append((host, stdin_text))
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(ssh_transport, "run_remote", fake_run_remote)

    target = _target(remote_workdir="/scratch/x")
    remote_shepherd._sync_video_roots(target, experiment_root, _job())

    assert copied == [
        ("tillicum", video_file, "/scratch/x/_video_roots/Cropped Videos/a.mp4")
    ]
    assert len(registered) == 1
    assert (
        "video_roots.register_video_root('Cropped Videos', "
        "'/scratch/x/_video_roots/Cropped Videos')" in registered[0][1]
    )


def test_sync_video_roots_is_a_noop_for_a_job_with_no_video_dependency(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        ssh_transport, "copy_file_to_remote", lambda *a, **k: calls.append(a)
    )
    monkeypatch.setattr(ssh_transport, "run_remote", lambda *a, **k: calls.append(a))
    remote_shepherd._sync_video_roots(_target(), tmp_path / "exp1", _training_job())
    assert calls == []  # no ssh calls at all — an empty source list costs nothing


def test_sync_video_roots_raises_when_remote_registration_fails(tmp_path, monkeypatch):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "a.mp4").write_bytes(b"x")
    _register_local_video_root(monkeypatch, tmp_path, "Cropped Videos", video_dir)
    experiment_root = tmp_path / "exp1"
    recording_dir = experiment_root / "recordings" / "recording_001"
    _write_recording_with_video(
        recording_dir,
        [{"index": 0, "source": {"root": "Cropped Videos", "rel": "a.mp4"}}],
    )
    monkeypatch.setattr(ssh_transport, "copy_file_to_remote", lambda *a, **k: None)
    monkeypatch.setattr(
        ssh_transport,
        "run_remote",
        lambda *a, **k: _FakeCompletedProcess(
            returncode=1, stderr="disk quota exceeded"
        ),
    )
    with pytest.raises(RuntimeError, match="disk quota exceeded"):
        remote_shepherd._sync_video_roots(_target(), experiment_root, _job())


def test_run_syncs_referenced_video_before_submission(
    monkeypatch, experiment, tmp_path
):
    experiment_root, run_dir = experiment
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "a.mp4").write_bytes(b"x")
    _register_local_video_root(monkeypatch, tmp_path, "Cropped Videos", video_dir)
    recording_dir = experiment_root / "recordings" / "recording_001"
    _write_recording_with_video(
        recording_dir,
        [{"index": 0, "source": {"root": "Cropped Videos", "rel": "a.mp4"}}],
    )
    scheduler = _FakeScheduler([JobState.RUNNING])
    _install_scheduler(monkeypatch, scheduler)
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "complete", "progress": None}),
    )
    copied = []
    monkeypatch.setattr(
        ssh_transport,
        "copy_file_to_remote",
        lambda host, local_path, remote_path, **k: copied.append(remote_path),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    assert copied == [
        "/gscratch/scrubbed/uwnetid/detail_runs/_video_roots/Cropped Videos/a.mp4"
    ]


# --- remote-side model readiness preflight --------------------------------- #
def test_model_readiness_probe_none_for_a_job_with_no_model():
    class _NoModelJob:
        job_id = "raw_01"

    assert (
        remote_shepherd._model_readiness_probe(_NoModelJob(), "/scratch/x/exp1", "")
        is None
    )


def test_model_readiness_probe_imports_the_runner_modules_before_resolving():
    """resolve_detection_model looks up a trained/imported model's family in
    TRAINED_LOADERS, populated only by importing the module that registers it (e.g.
    detail.trainers.ultralytics_yolo for "yolo") -- the same gap
    executor/worker.py closes via import_runner_modules before it does anything
    else. Found live against a real cluster: a genuinely-installed distilled model
    failed readiness with "no trained-model loader for family 'yolo'; available:
    []" because this probe called resolve_detection_model with nothing imported."""
    script = remote_shepherd._model_readiness_probe(
        _job(),
        "/scratch/x/exp1",
        "detail.pipeline.detection,detail.trainers.ultralytics_yolo",
    )
    assert "import detail.pipeline.detection" in script
    assert "import detail.trainers.ultralytics_yolo" in script
    # both imports must run before the resolver is even imported, let alone called
    resolve_import_at = script.index("from detail.models.resolve import")
    assert script.index("import detail.pipeline.detection") < resolve_import_at
    assert script.index("import detail.trainers.ultralytics_yolo") < resolve_import_at


def test_model_readiness_probe_with_no_runner_modules_still_resolves():
    """An empty runner_modules string (e.g. a test building the probe directly) must
    not inject a syntactically broken import line."""
    script = remote_shepherd._model_readiness_probe(_job(), "/scratch/x/exp1", "")
    assert "resolve_detection_model" in script
    compile(script, "<probe>", "exec")  # raises SyntaxError if the template is broken


def _embedding_job(model_id="fake_embedder"):
    return EmbeddingJob(
        job_id="emb_01",
        recording_id="recording_001",
        embedding_run_id="emb_01",
        tracking_run_id="track_01",
        track_ids=[0],
        model_id=model_id,
        renderer_spec=[],
        batch_size=32,
        num_workers=0,
        compute=ComputeTarget("remote", None, None, None),
        created_at="2026-01-01T00:00:00Z",
    )


class _FakeEmbedderForProbe(EmbeddingModel):
    model_id = "fake_embedder"
    output_dim = 4
    input_size = 32
    supported_backends = ["cpu", "cuda"]

    def load(self, device):
        pass

    def is_ready(self):
        return True

    def embed(self, image):
        raise NotImplementedError

    def embed_batch(self, images):
        raise NotImplementedError

    def readiness(self):
        return ModelReadiness(
            ready=False,
            reason="not_installed",
            message="fake_embedder weights are not installed on this machine",
            action_url=None,
            repo_id=None,
        )


def test_model_readiness_probe_resolves_an_embedding_job_through_embedding_models():
    """An embedding job's model_id lives in EMBEDDING_MODELS, a disjoint namespace
    from the detection registries resolve_detection_model reads -- probing it
    through the detection resolver (the bug: this probe was written for detection
    only and never branched for embedding once embedding was wired to remote,
    #457) raises "unknown model_id" for *every* real embedding model, before
    .readiness() is ever called. Executes the generated script for real (not just
    compiles it) against a registered fake embedding model, since a compile-only
    check cannot see a wrong-registry lookup -- that fails at runtime, not at
    parse time, and the original bug's script compiled just fine."""
    register_embedding_model("fake_embedder")(_FakeEmbedderForProbe)
    try:
        script = remote_shepherd._model_readiness_probe(
            _embedding_job(), "/scratch/x/exp1", ""
        )
        assert "resolve_detection_model" not in script
        assert "EMBEDDING_MODELS" in script
        namespace: dict = {}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(script, "<probe>", "exec"), namespace)
        payload = json.loads(buf.getvalue().strip())
        assert payload["ready"] is False
        assert payload["reason"] == "not_installed"
    finally:
        EMBEDDING_MODELS.pop("fake_embedder", None)


def test_run_fails_before_submission_when_model_not_installed_remotely(
    monkeypatch, experiment
):
    experiment_root, run_dir = experiment
    scheduler = _FakeScheduler([JobState.RUNNING])
    _install_scheduler(monkeypatch, scheduler)
    not_ready = json.dumps(
        {
            "ready": False,
            "reason": "not_installed",
            "message": "sam3 weights are not installed on this host",
            "action_url": None,
            "repo_id": None,
        }
    )
    monkeypatch.setattr(
        ssh_transport,
        "run_remote",
        lambda *a, **k: _FakeCompletedProcess(stdout=not_ready),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    assert "not installed on this host" in rundir.read_envelope(run_dir)["error"]
    assert not scheduler.submitted  # failed before an sbatch job was ever queued


def test_run_fails_when_the_readiness_probe_itself_cannot_run(monkeypatch, experiment):
    experiment_root, run_dir = experiment
    scheduler = _FakeScheduler([JobState.RUNNING])
    _install_scheduler(monkeypatch, scheduler)
    monkeypatch.setattr(
        ssh_transport,
        "run_remote",
        lambda *a, **k: _FakeCompletedProcess(
            returncode=1, stdout="", stderr="conda: command not found"
        ),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    assert "conda: command not found" in rundir.read_envelope(run_dir)["error"]
    assert not scheduler.submitted


def test_run_succeeds_when_the_remote_model_is_ready(monkeypatch, experiment):
    # The default fixture stand-in reports ready — this pins that the ordinary
    # (installed) case still submits and completes normally.
    experiment_root, run_dir = experiment
    scheduler = _FakeScheduler([JobState.RUNNING])
    _install_scheduler(monkeypatch, scheduler)
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "complete", "progress": None}),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    assert scheduler.submitted


# --- progress before the remote worker exists (#644) ------------------------------------ #
# **The class: a long phase that emits nothing until a boundary reads as hung.** Before the
# remote worker wrote its first event, the shepherd uploaded the experiment, then every
# referenced video (80 recordings was ~116 GB, about an hour), checked the model, submitted and
# waited in the queue, and published nothing for any of it. The distill page sat on "starting
# labeling" and the person watching could not tell a healthy run from a frozen one.
#
# Every test here reads the events the shepherd *publishes* (`rundir.set_progress`), because the
# class is invisible at the line where it is written: the upload worked perfectly the whole time.
@pytest.fixture
def published(monkeypatch):
    """Every event the shepherd publishes, in order, with the cadence throttle off."""
    from detail.core import progress as core_progress

    monkeypatch.setattr(core_progress, "PROGRESS_INTERVAL_S", 0.0)
    events = []
    monkeypatch.setattr(
        remote_shepherd.rundir,
        "set_progress",
        lambda run_dir, event: events.append(event),
    )
    return events


def _video_library(monkeypatch, tmp_path, sizes):
    """An experiment whose recording references ``len(sizes)`` videos of the given byte sizes."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    files = []
    for i, size in enumerate(sizes):
        (video_dir / f"v{i}.mp4").write_bytes(b"x" * size)
        files.append({"index": i, "source": {"root": "Lib", "rel": f"v{i}.mp4"}})
    _register_local_video_root(monkeypatch, tmp_path, "Lib", video_dir)
    experiment_root = tmp_path / "exp1"
    _write_recording_with_video(experiment_root / "recordings" / "recording_001", files)
    return experiment_root


def _ten_seconds_a_tick(monkeypatch):
    """The shepherd's clock, advancing 10 s per read, so an ETA is deterministic."""
    ticks = itertools.count(0, 10)
    monkeypatch.setattr(
        remote_shepherd,
        "time",
        SimpleNamespace(monotonic=lambda: float(next(ticks)), sleep=lambda *_: None),
    )


def test_one_big_video_still_shows_motion_and_an_eta_while_it_uploads(
    monkeypatch, tmp_path, published
):
    """A single file is the case that matters: the message moves per *finished* file, so with one
    there are two of them (0/1 and 1/1) and the whole upload between sat silent. The ETA is what
    moves inside it, fed by the bytes rsync reports."""
    experiment_root = _video_library(monkeypatch, tmp_path, [1000])
    _ten_seconds_a_tick(monkeypatch)
    job = _labeling_job(["recording_001"])

    def copy_half(host, local_path, remote_path, *, on_bytes=None, **k):
        on_bytes(500)

    monkeypatch.setattr(ssh_transport, "copy_file_to_remote", copy_half)
    monkeypatch.setattr(
        ssh_transport, "run_remote", lambda *a, **k: _FakeCompletedProcess()
    )
    transfer = remote_shepherd._TransferProgress(tmp_path / "run", job)

    remote_shepherd._sync_video_roots(_target(), experiment_root, job, transfer)

    assert len(published) == 3, [e.message for e in published]
    first, mid, last = published
    assert all(isinstance(e, LabelingProgressEvent) for e in published)
    assert first.eta_s is None and last.eta_s is None
    assert mid.eta_s is not None and mid.eta_s > 0, "no ETA through the in-flight file"
    assert mid.message == first.message, "the message must not change inside one file"
    assert first.message.startswith("uploading videos · 0/1 files")
    assert last.message.startswith("uploading videos · 1/1 files")
    # The stage's own fraction is untouched: nothing has been labeled, and an invented one would
    # run the bar to 100% and drop it back to 0 when labeling starts.
    assert {(e.frames_done, e.frames_total) for e in published} == {(0, 0)}


def test_the_message_advances_per_file_and_each_step_reaches_the_log(
    monkeypatch, tmp_path, published, capsys
):
    experiment_root = _video_library(monkeypatch, tmp_path, [100, 300])
    job = _labeling_job(["recording_001"])
    monkeypatch.setattr(
        ssh_transport,
        "copy_file_to_remote",
        lambda host, local_path, remote_path, *, on_bytes=None, **k: on_bytes(50),
    )
    monkeypatch.setattr(
        ssh_transport, "run_remote", lambda *a, **k: _FakeCompletedProcess()
    )
    transfer = remote_shepherd._TransferProgress(tmp_path / "run", job)

    remote_shepherd._sync_video_roots(_target(), experiment_root, job, transfer)

    steps = [e.message.split(" · ")[1] for e in published]
    assert [s for i, s in enumerate(steps) if i == 0 or s != steps[i - 1]] == [
        "0/2 files",
        "1/2 files",
        "2/2 files",
    ]
    logged = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert [line.split(" ", 1)[1].split(" · ")[1] for line in logged] == [
        "0/2 files",
        "1/2 files",
        "2/2 files",
    ], "shepherd.log is where a run that stopped reporting says what it was last doing"


def _messages(events):
    return [e.message for e in events]


def test_a_run_names_every_phase_before_the_worker_reports(
    monkeypatch, experiment, published
):
    experiment_root, run_dir = experiment
    _install_scheduler(
        monkeypatch, _FakeScheduler([JobState.PENDING, JobState.RUNNING])
    )
    reads = iter(
        [{"status": "running", "progress": None}] * 2
        + [{"status": "complete", "progress": None}] * 2
    )
    monkeypatch.setattr(
        ssh_transport, "read_remote_file", lambda *a, **k: json.dumps(next(reads))
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    assert _messages(published) == [
        "uploading experiment files",
        "checking the model on the cluster",
        "submitting the job",
        "waiting in the tillicum queue",
        "starting on tillicum (loading the model)",
        "syncing results back",
    ]
    assert all(isinstance(e, DetectionProgressEvent) for e in published)


def test_the_workers_own_progress_is_never_overwritten_by_a_queue_message(
    monkeypatch, experiment, published
):
    """Once the remote worker writes an event the shepherd forwards it; a later "starting on…"
    would replace a real frame count with a phase name."""
    experiment_root, run_dir = experiment
    _install_scheduler(
        monkeypatch, _FakeScheduler([JobState.RUNNING, JobState.RUNNING])
    )
    worker = {
        "type": "detection",
        "parts_done": 0,
        "parts_total": 1,
        "frames_done": 5,
        "frames_total": 10,
        "elapsed_s": 1.0,
        "eta_s": 2.0,
        "message": None,
    }
    reads = iter(
        [{"status": "running", "progress": worker}] * 2
        + [{"status": "complete", "progress": None}]
    )
    monkeypatch.setattr(
        ssh_transport, "read_remote_file", lambda *a, **k: json.dumps(next(reads))
    )

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert _messages(published) == [
        "uploading experiment files",
        "checking the model on the cluster",
        "submitting the job",
        "syncing results back",
    ]


def test_a_job_with_no_model_is_not_told_its_model_is_being_checked(
    monkeypatch, experiment, published
):
    experiment_root, _ = experiment
    run_dir = experiment_root / "models" / "model_001"
    run_dir.mkdir(parents=True)
    rundir.init_run_dir(run_dir, _training_job())
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.PENDING]))
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "complete", "progress": None}),
    )

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert "checking the model on the cluster" not in _messages(published)
    assert all(isinstance(e, TrainingProgressEvent) for e in published)


@pytest.mark.parametrize(
    "make_job, family",
    [
        (_job, DetectionProgressEvent),
        (lambda: _labeling_job(["recording_001"]), LabelingProgressEvent),
        (_training_job, TrainingProgressEvent),
        (lambda: _embedding_job(), EmbeddingProgressEvent),
    ],
)
def test_each_job_family_gets_its_own_event_and_it_survives_the_envelope(
    tmp_path, make_job, family
):
    """The event is read back by `progress_event_from_dict` in whichever process attaches, so it
    must be one the job's own surfaces already know, and must round-trip with its message.
    """
    job = make_job()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rundir.init_run_dir(run_dir, job)
    event = remote_shepherd._phase_event(
        job,
        message="uploading videos · 1/80 files · 1.4/115.8 GB",
        elapsed_s=30.0,
        eta_s=2400.0,
    )
    assert isinstance(event, family)

    rundir.set_progress(run_dir, event)

    assert rundir.read_progress(run_dir) == event
    assert event.eta_s == 2400.0 and event.message.startswith("uploading videos")


def test_a_job_the_shepherd_has_no_event_for_publishes_nothing_rather_than_failing():
    job = SimpleNamespace(job_id="x")  # not a known Job type
    assert (
        remote_shepherd._phase_event(job, message="m", elapsed_s=0.0, eta_s=None)
        is None
    )


# --- a labeling job is sent its sampled frames, not the video library -------------------- #
# **The class: shipping the whole input to do work on a tiny sample of it.** Labeling reads ~1,000
# frames chosen from the timestamp maps and the dataset id alone, yet a remote run uploaded every
# referenced video first (80 recordings was ~116 GB, about an hour). The shepherd now extracts the
# frames where the video is and the mirror carries only them. The extraction itself, and that
# labeling from it is byte-identical to labeling from the video, is pinned in
# `tests/pipeline/test_labeling.py`; these tests pin the orchestration around it.
def _labeling_run(monkeypatch, experiment, tmp_path):
    """A labeling run dir inside the fixture's experiment, over one *referenced* video, with the
    shepherd's extraction replaced by a recorder (the real one needs a decodable video).
    """
    from detail.pipeline import labeling as labeling_module

    experiment_root, _ = experiment
    _video_library(monkeypatch, tmp_path, [1000])
    run_dir = experiment_root / "datasets" / "dataset_001"
    run_dir.mkdir(parents=True)
    rundir.init_run_dir(run_dir, _labeling_job(["recording_001"]))

    seen = SimpleNamespace(
        order=[],
        frames_at_mirror=[],
        uploaded_videos=[],
        envelopes={},
    )

    def extract(root, job, out_dir, progress_cb=None, stop_event=None):
        seen.order.append("extract")
        out_dir.mkdir(parents=True)
        (out_dir / "f.png").write_bytes(b"png")
        (out_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return 1

    def mirror(host, local_dir, remote_dir, **k):
        seen.order.append("mirror")
        seen.frames_at_mirror.append((run_dir / "source_frames" / "f.png").is_file())

    monkeypatch.setattr(labeling_module, "extract_source_frames", extract)
    monkeypatch.setattr(ssh_transport, "copy_to_remote", mirror)
    monkeypatch.setattr(
        ssh_transport,
        "copy_file_to_remote",
        lambda *a, **k: seen.uploaded_videos.append(a),
    )
    monkeypatch.setattr(
        ssh_transport,
        "write_remote_file",
        lambda host, path, content, **k: seen.envelopes.__setitem__(path, content),
    )
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    return experiment_root, run_dir, seen


def _remote_reports(monkeypatch, envelope):
    monkeypatch.setattr(
        ssh_transport, "read_remote_file", lambda *a, **k: json.dumps(envelope)
    )


def test_a_remote_labeling_job_is_sent_its_frames_and_no_video(
    monkeypatch, experiment, tmp_path
):
    experiment_root, run_dir, seen = _labeling_run(monkeypatch, experiment, tmp_path)
    _remote_reports(monkeypatch, {"status": "complete", "progress": None})

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    # Extracted before the mirror, so the mirror carries the frames.
    assert seen.order == ["extract", "mirror"] and seen.frames_at_mirror == [True]
    assert seen.uploaded_videos == [], "no video should cross for a labeling job"
    remote_job = json.loads(
        next(c for p, c in seen.envelopes.items() if p.endswith("/job.json"))
    )["job"]
    assert remote_job["source_frames"] == "source_frames"
    # The record on this machine still says what was submitted.
    assert rundir.read_job(run_dir).source_frames is None
    # The scratch frames are gone once the dataset exists.
    assert not (run_dir / "source_frames").exists()


def test_the_extracted_frames_are_kept_when_the_run_fails_so_a_retry_is_cheap(
    monkeypatch, experiment, tmp_path
):
    experiment_root, run_dir, _ = _labeling_run(monkeypatch, experiment, tmp_path)
    _remote_reports(
        monkeypatch, {"status": "failed", "error": "boom", "progress": None}
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    assert (run_dir / "source_frames" / "manifest.json").is_file()


def test_a_job_that_reads_extracted_frames_has_no_video_to_upload(
    monkeypatch, tmp_path
):
    import dataclasses

    _video_library(monkeypatch, tmp_path, [1000])
    job = _labeling_job(["recording_001"])
    assert remote_shepherd._job_recording_ids(job) == ["recording_001"]
    framed = dataclasses.replace(job, source_frames="source_frames")
    assert remote_shepherd._job_recording_ids(framed) == []


def test_frame_extraction_is_a_named_phase_with_an_eta_and_no_fraction(
    monkeypatch, tmp_path, published, capsys
):
    _ten_seconds_a_tick(monkeypatch)
    transfer = remote_shepherd._TransferProgress(
        tmp_path / "run", _labeling_job(["recording_001"])
    )

    for done in (0, 5, 10):
        transfer.frames(done, 10)

    assert set(_messages(published)) == {"extracting 10 frames from the videos"}
    assert [e.eta_s is None for e in published] == [True, False, True]
    assert {(e.frames_done, e.frames_total) for e in published} == {(0, 0)}
    logged = capsys.readouterr().out
    assert "extracting 10 frames from the videos" in logged
    assert "extracted 5/10 frames" in logged


# --- waiting on the opt-in background video upload (executor/video_staging.py) ------------- #
def _staged_run(monkeypatch, experiment, tmp_path, updates):
    """A detection run over one referenced video, with a live background upload covering it.

    ``updates`` are applied to the recorded staging state one per poll of the wait. Returns the
    ordered list of ``wait`` / ``copy`` events the run produced.
    """
    from detail.executor import local, video_staging

    experiment_root, run_dir = experiment
    _video_library(monkeypatch, tmp_path, [100])
    monkeypatch.setattr(local, "coordinator_alive", lambda *_a: True)
    state = video_staging._initial_state("tillicum", ["recording_001"])
    state.update(pid=4242, files_total=1, message="uploading videos · 0/1 files")
    video_staging._write_state(experiment_root, "tillicum", state)
    events, queue = [], list(updates)

    def poll(_s):
        events.append("wait")
        current = video_staging.read_state(experiment_root, "tillicum")
        video_staging._write_state(
            experiment_root, "tillicum", {**current, **queue.pop(0)}
        )

    # The module's own `time`, not `time.sleep` itself: that is the one global every thread shares,
    # and a GUI worker still polling from an earlier test in this process would call this hook.
    monkeypatch.setattr(
        video_staging, "time", SimpleNamespace(sleep=poll, monotonic=time.monotonic)
    )
    monkeypatch.setattr(
        ssh_transport,
        "copy_file_to_remote",
        lambda *a, **k: events.append("copy"),
    )
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "complete", "progress": None}),
    )
    return events


def test_a_run_waits_for_the_background_upload_before_it_uploads_anything(
    monkeypatch, experiment, tmp_path, published
):
    """Two uploads of the same files at once would copy the bytes twice and fight over the file.
    The run waits for the staging process, shows *its* progress as the phase, and only then runs
    its own upload — which finds every file already in the cluster's cache and sends none.
    """
    experiment_root, run_dir = experiment
    events = _staged_run(
        monkeypatch,
        experiment,
        tmp_path,
        [
            {"message": "uploading videos · 1/2 files", "eta_s": 30.0},
            {"status": "complete", "message": "the videos are on the cluster"},
        ],
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    assert events == ["wait", "wait", "copy"], "the run's own upload came first"
    waiting = [m for m in _messages(published) if m.startswith("waiting for the")]
    assert waiting == [
        "waiting for the background video upload · uploading videos · 0/1 files",
        "waiting for the background video upload · uploading videos · 1/2 files",
    ]
    mid = next(e for e in published if "1/2 files" in e.message)
    assert mid.eta_s == 30.0, "the staging run's measured ETA is this phase's ETA"
    assert isinstance(mid, DetectionProgressEvent)


def test_a_failed_background_upload_costs_time_never_correctness(
    monkeypatch, experiment, tmp_path, capsys
):
    experiment_root, run_dir = experiment
    events = _staged_run(
        monkeypatch,
        experiment,
        tmp_path,
        [{"status": "failed", "error": "RuntimeError: disk quota exceeded"}],
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    assert events == ["wait", "copy"], "the run uploads what the staging run did not"
    assert (
        "background video upload ended (failed): RuntimeError: disk quota exceeded"
        in capsys.readouterr().out
    )


def test_a_cancelled_run_does_not_sit_behind_the_upload(
    monkeypatch, experiment, tmp_path
):
    experiment_root, run_dir = experiment
    events = _staged_run(monkeypatch, experiment, tmp_path, [])
    rundir.request_cancel(run_dir)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.CANCELLED
    assert events == [], "neither waited on the upload nor started its own"
    assert rundir.read_status(run_dir) == JobStatus.CANCELLED


def test_a_job_that_reads_no_video_never_waits_on_the_upload(
    monkeypatch, experiment, tmp_path
):
    """A labeling job reads extracted frames and a training job reads a dataset: neither needs
    the videos, so a distill's own label stage must not stall behind the upload it started.
    """
    import dataclasses

    from detail.executor import video_staging

    experiment_root, run_dir = experiment
    _staged_run(monkeypatch, experiment, tmp_path, [])
    monkeypatch.setattr(
        video_staging,
        "wait_for_active",
        lambda *a, **k: pytest.fail("waited on an upload the job does not need"),
    )
    transfer = remote_shepherd._TransferProgress(run_dir, _training_job())
    framed = dataclasses.replace(
        _labeling_job(["recording_001"]), source_frames="source_frames"
    )

    for job in (framed, _training_job()):
        assert not remote_shepherd._wait_for_staging(
            _target(), experiment_root, run_dir, job, transfer
        )


# --- an ssh connection that drops mid-run (the class in the module docstring) ---------------- #
_NO_SESSION = ssh_transport.RemoteCommandError(
    ["ssh"], 255, "", "Permission denied (keyboard-interactive)"
)


def _reads(monkeypatch, items):
    """``read_remote_file`` answering ``items`` in order; ``None`` is ssh failing to connect."""
    queue = iter(items)

    def read(*_a, **_k):
        item = next(queue)
        if item is None:
            raise _NO_SESSION
        return json.dumps(item)

    monkeypatch.setattr(ssh_transport, "read_remote_file", read)


def _naps(monkeypatch):
    """Every ``time.sleep`` the shepherd asks for, without sleeping."""
    naps = []
    monkeypatch.setattr(
        remote_shepherd,
        "time",
        SimpleNamespace(monotonic=time.monotonic, sleep=naps.append),
    )
    return naps


_COMPLETE = {"status": "complete", "progress": None}
_UNREACHABLE = JobState.UNREACHABLE


def test_a_dropped_connection_waits_and_the_run_still_completes(
    monkeypatch, experiment, published
):
    """The run this class failed: ssh gone for three polls, then back. It must finish, not be
    recorded failed because nobody could say whether the job was alive."""
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([_UNREACHABLE] * 3))
    _reads(monkeypatch, [None, None, None, _COMPLETE, _COMPLETE])
    naps = _naps(monkeypatch)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    assert rundir.read_status(run_dir) == JobStatus.COMPLETE
    assert naps == [10, 20, 40], "backs off while it cannot ask, rather than every 5 s"
    waiting = {m for m in _messages(published) if m.startswith("cannot reach")}
    assert len(waiting) == 1, "one stable message, or the CLI prints a line per poll"
    (message,) = waiting
    assert "tillicum over ssh since" in message and "log in again" in message


def test_a_dropped_connection_keeps_the_workers_numbers_and_changes_only_the_words(
    monkeypatch, experiment, published
):
    """The bar must not fall to zero because the network did: the worker had reported 5/10."""
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING, _UNREACHABLE]))
    worker = {
        "type": "detection",
        "parts_done": 0,
        "parts_total": 1,
        "frames_done": 5,
        "frames_total": 10,
        "elapsed_s": 1.0,
        "eta_s": 2.0,
        "message": None,
    }
    _reads(monkeypatch, [{"status": "running", "progress": worker}, None, _COMPLETE])
    _naps(monkeypatch)
    payloads = []
    real = rundir.update_payload
    monkeypatch.setattr(
        remote_shepherd.rundir,
        "update_payload",
        lambda rd, **fields: payloads.append(fields) or real(rd, **fields),
    )

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    held = [
        p["progress"]
        for p in payloads
        if "progress" in p
        and str(p["progress"].get("message")).startswith("cannot reach")
    ]
    assert len(held) == 1
    assert (held[0]["frames_done"], held[0]["frames_total"]) == (5, 10)
    assert held[0]["eta_s"] == 2.0
    assert not [
        m for m in _messages(published) if m.startswith("cannot reach")
    ], "a zero-work event would replace the worker's frame count with a phase name"


def test_the_upload_never_overwrites_a_run_the_cluster_already_has(
    monkeypatch, experiment
):
    """Every run directory in the experiment is handed to the mirror as one to send only when it
    is absent (`tests/executor/test_scoped_upload.py` has the real-rsync behaviour); a second job's
    upload otherwise resets the first's `job.json` and records a finished job as failed.
    """
    experiment_root, run_dir = experiment
    other = (
        experiment_root / "recordings" / "recording_001" / "runs" / "detect" / "det_02"
    )
    other.mkdir(parents=True)
    rundir.init_run_dir(other, _job("det_02"))
    sent = []
    monkeypatch.setattr(ssh_transport, "copy_to_remote", lambda *a, **k: sent.append(k))
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    _reads(monkeypatch, [_COMPLETE])

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    (kwargs,) = sent
    assert set(kwargs["never_overwrite"]) == {run_dir, other}
    # A run made after that list is taken is covered by name: its envelope is never sent by the
    # ordinary pass (the real-rsync behaviour is in test_scoped_upload.py).
    assert list(kwargs["never_send"]) == [rundir.JOB_FILENAME]


def test_the_copy_back_keeps_the_shepherds_own_log(monkeypatch, experiment):
    """The shepherd's log is its stdout; the copy back must not replace it with the cluster's
    stale copy (`tests/executor/test_copy_back_keeps_local.py` has the real-rsync behaviour).
    """
    experiment_root, run_dir = experiment
    back = []
    monkeypatch.setattr(
        ssh_transport, "copy_from_remote", lambda *a, **k: back.append(k)
    )
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    _reads(monkeypatch, [_COMPLETE])

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    (kwargs,) = back
    # The log, and the envelope (#742: `tests/executor/test_status_after_data.py`).
    assert set(kwargs["keep_local"]) == {
        rundir.SHEPHERD_LOG_FILENAME,
        rundir.JOB_FILENAME,
    }


def test_a_cancel_that_could_not_be_delivered_is_sent_again_not_assumed_sent(
    monkeypatch, experiment
):
    """Cancelling with the connection down used to mark the cancel delivered; the job then ran
    to the end of its walltime on a GPU nobody wanted."""
    experiment_root, run_dir = experiment

    class _Scheduler(_FakeScheduler):
        def submit_script(self, target, script, remote_run_dir):
            rundir.request_cancel(run_dir)  # the cancel lands once the job is queued
            return super().submit_script(target, script, remote_run_dir)

        def cancel(self, target, remote_job_id):
            self.cancel_calls.append(remote_job_id)
            return len(self.cancel_calls) > 2  # undeliverable twice, then it goes

    scheduler = _Scheduler([_UNREACHABLE, _UNREACHABLE, JobState.CANCELLED])
    _install_scheduler(monkeypatch, scheduler)
    _reads(monkeypatch, [None] * 8)  # the two polls, then the end-of-job read
    _naps(monkeypatch)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.CANCELLED
    assert scheduler.cancel_calls == ["999"] * 3, "asked again until it was delivered"


def test_a_job_the_scheduler_has_no_record_of_still_fails(monkeypatch, experiment):
    """The control: only an ask that did not happen is waited on. A job Slurm answers about as
    gone must still end the run, or a vanished job would be waited on for a week."""
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.FAILED]))
    _reads(monkeypatch, [{"status": "running", "progress": None}])
    naps = _naps(monkeypatch)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    assert naps == []


def test_it_gives_up_after_the_longest_it_will_wait_and_says_the_job_may_still_run(
    monkeypatch, experiment
):
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([_UNREACHABLE] * 50))
    _reads(monkeypatch, [None] * 50)
    _naps(monkeypatch)
    monkeypatch.setattr(remote_shepherd, "MAX_UNREACHABLE_S", -1)
    asked = []
    monkeypatch.setattr(
        ssh_transport, "is_reachable", lambda *a, **k: asked.append("ask") or False
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    error = rundir.read_envelope(run_dir)["error"]
    assert "unreachable over ssh for" in error
    assert "the remote job may still be running" in error
    assert not [
        c for c in asked if c == "ask"
    ], "having given up on a dead connection it waited out a second one before the sync"


def test_the_sync_back_waits_for_the_connection_instead_of_recording_a_run_with_no_results(
    monkeypatch, experiment, published
):
    """The job finished while the connection was down. The results are on the cluster; copying
    with ssh dead used to fail quietly and leave `complete` over an empty directory."""
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([]))
    _reads(monkeypatch, [_COMPLETE, _COMPLETE])
    _naps(monkeypatch)
    events = []
    answers = iter([False, False, True])
    monkeypatch.setattr(
        ssh_transport,
        "is_reachable",
        lambda host, **k: events.append("ask") or next(answers),
    )
    monkeypatch.setattr(
        ssh_transport, "copy_from_remote", lambda *a, **k: events.append("copy")
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    assert events == [
        "ask",
        "ask",
        "ask",
        "copy",
    ], "copied before the connection was back"
    assert [m for m in _messages(published) if m.startswith("cannot reach")]


def test_results_that_stay_out_of_reach_fail_the_run_rather_than_calling_it_complete(
    monkeypatch, experiment
):
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([]))
    _reads(monkeypatch, [_COMPLETE])
    _naps(monkeypatch)
    monkeypatch.setattr(remote_shepherd, "MAX_UNREACHABLE_S", -1)
    monkeypatch.setattr(ssh_transport, "is_reachable", lambda *a, **k: False)
    monkeypatch.setattr(
        ssh_transport,
        "copy_from_remote",
        lambda *a, **k: pytest.fail("copied over a connection known to be down"),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    assert "results could not be copied back" in rundir.read_envelope(run_dir)["error"]


def test_an_envelope_read_that_hangs_is_read_as_not_there_yet(monkeypatch):
    """The read is the other ask in the loop. It has a deadline, and a hang is a retry."""
    asked = {}

    def hang(host, path, **k):
        asked.update(k)
        raise ssh_transport.subprocess.TimeoutExpired(cmd="ssh", timeout=30)

    monkeypatch.setattr(ssh_transport, "read_remote_file", hang)

    assert remote_shepherd._read_remote_envelope(_target(), "/scratch/run") is None
    assert asked == {
        "timeout_s": ssh_transport.ASK_TIMEOUT_S,
        # a poll's ask gives a full connection only a short wait: the loop has its own for it
        "refused_retries": ssh_transport.ASK_REFUSED_RETRIES,
    }


# --- the copy back is the job's run folder, never the experiment ------------------------- #
def _cluster(monkeypatch, tmp_path):
    """A folder standing in for the cluster, with ``copy_to_remote`` / ``copy_from_remote`` copying
    like ``rsync -a`` (create what is missing, overwrite what differs, delete nothing) between it
    and this machine. Returns the function mapping a remote path to its folder."""
    cluster = tmp_path / "cluster"
    workdir = _target().remote_workdir

    def on_cluster(remote_path: str) -> Path:
        return cluster / remote_path.removeprefix(workdir).lstrip("/")

    def up(host, local_dir, remote_dir, **k):
        shutil.copytree(local_dir, on_cluster(remote_dir), dirs_exist_ok=True)

    def down(host, remote_dir, local_dir, **k):
        shutil.copytree(on_cluster(remote_dir), local_dir, dirs_exist_ok=True)

    monkeypatch.setattr(ssh_transport, "copy_to_remote", up)
    monkeypatch.setattr(ssh_transport, "copy_from_remote", down)
    return on_cluster


def _live_run(root, recording_id, run_id, **envelope):
    """Another run's folder, as the coordinator that owns it keeps it."""
    run = root / "recordings" / recording_id / "runs" / "detect" / run_id
    run.mkdir(parents=True)
    (run / "job.json").write_text(
        json.dumps({"status": "running", **envelope}), encoding="utf-8"
    )
    return run


def test_a_job_finishing_does_not_put_the_clusters_snapshot_over_the_local_experiment(
    monkeypatch, experiment, tmp_path
):
    """What the person did on this machine while the job ran must still be there afterwards."""
    experiment_root, run_dir = experiment
    on_cluster = _cluster(monkeypatch, tmp_path)
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    other = _live_run(experiment_root, "recording_002", "det_02", pid=None)
    deleted = _live_run(experiment_root, "recording_003", "det_03")
    recording_dir = experiment_root / "recordings" / "recording_001"
    remote_run = on_cluster(
        remote_shepherd._remote_paths(_target(), experiment_root, run_dir)[1]
    )
    lived = []

    def meanwhile(*_a, **_k):
        """The job is running: the cluster writes its output, and life goes on here."""
        if not lived:
            lived.append(True)
            (remote_run / "result.txt").write_text("output", encoding="utf-8")
            (other / "job.json").write_text(
                json.dumps({"status": "running", "pid": 4242}), encoding="utf-8"
            )
            recording_store.write_recording(
                recording_dir,
                {"id": "recording_001", "name": "renamed", "video": {"files": []}},
            )
            shutil.rmtree(deleted.parent.parent.parent)
        return json.dumps({"status": "complete", "progress": None})

    monkeypatch.setattr(ssh_transport, "read_remote_file", meanwhile)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    assert (run_dir / "result.txt").read_text(
        encoding="utf-8-sig"
    ) == "output", "not brought back"
    assert (
        json.loads((other / "job.json").read_text(encoding="utf-8-sig"))["pid"] == 4242
    ), "another run's live job.json was replaced by its stale copy from the cluster"
    assert (
        recording_store.read_recording(recording_dir)["name"] == "renamed"
    ), "a rename made while the job ran was reverted"
    assert not deleted.exists(), "a run deleted while the job ran came back"


@pytest.mark.parametrize("remote_status", ["complete", "failed"])
def test_the_sync_back_is_exactly_the_jobs_run_folder(
    monkeypatch, experiment, remote_status
):
    """Pinned as the call itself, because the filesystem test above cannot tell a run folder
    from a slightly larger folder that happens to contain nothing else yet. A failed job is
    synced too: its `worker.log` is the only account of why."""
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": remote_status, "progress": None}),
    )
    calls = []
    monkeypatch.setattr(
        ssh_transport,
        "copy_from_remote",
        lambda host, remote, local, **k: calls.append((host, remote, Path(local))),
    )

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    _, remote_run_dir = remote_shepherd._remote_paths(
        _target(), experiment_root, run_dir
    )
    assert calls == [("tillicum", remote_run_dir, run_dir)]


# --------------------------------------------------------------------------- #
# Remote analysis (§6 "Remote analysis"): the inputs a job re-sends and the step it runs
# on this machine once its results are back.
# --------------------------------------------------------------------------- #
def _with_hooks(monkeypatch, *, refreshed=(), after=None):
    from detail.executor.runners import JOB_RUNNERS, RunnerSpec

    monkeypatch.setitem(
        JOB_RUNNERS,
        "detection",
        RunnerSpec(
            run=lambda *a: None,  # the shepherd never runs the job itself
            run_dir_for=lambda root, job: None,
            refreshed_inputs=lambda root, job: [root / rel for rel in refreshed],
            after_copy_back=after or (lambda root, job, run_dir: None),
        ),
    )


def _completes(monkeypatch):
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "complete", "progress": None}),
    )


def test_a_run_the_job_reads_is_sent_again_over_the_clusters_copy(
    monkeypatch, experiment
):
    """The upload leaves alone every run directory the cluster already has, so a clustering
    run whose node cache was rebuilt here would be read stale there. The runs a job names are
    sent again after the upload, exactly, their envelope aside."""
    experiment_root, run_dir = experiment
    _with_hooks(monkeypatch, refreshed=["analysis/an_10"])
    _completes(monkeypatch)
    uploads = []
    monkeypatch.setattr(
        ssh_transport,
        "copy_to_remote",
        lambda host, local, remote, **k: uploads.append(
            ("copy", Path(local), remote, k)
        ),
    )
    monkeypatch.setattr(
        ssh_transport,
        "mirror_to_remote",
        lambda host, local, remote, **k: uploads.append(
            ("mirror", Path(local), remote, k)
        ),
    )

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert uploads[0][:2] == ("copy", experiment_root)  # the whole experiment first
    how, local, remote, kwargs = uploads[1]
    assert how == "mirror", "exact, so what a cancelled job left there cannot stay"
    assert local == experiment_root / "analysis" / "an_10"
    assert remote == "/gscratch/scrubbed/uwnetid/detail_runs/exp1/analysis/an_10"
    assert kwargs == {"keep_remote": [rundir.JOB_FILENAME]}


def test_the_after_copy_back_step_runs_once_the_results_are_here(
    monkeypatch, experiment
):
    """A job whose results also belong outside its run folder puts them there on this machine,
    after the copy back (the cluster's copy of that place never comes back) and before the
    final status, so a reader that sees ``complete`` finds them in place."""
    experiment_root, run_dir = experiment
    seen = []

    def after(root, job, folder):
        seen.append(
            (root, job.job_id, folder, (folder / "result.txt").exists(),
             rundir.read_status(folder))
        )  # fmt: skip

    _with_hooks(monkeypatch, after=after)
    _completes(monkeypatch)
    monkeypatch.setattr(
        ssh_transport,
        "copy_from_remote",
        lambda host, remote, local, **k: (Path(local) / "result.txt").write_text(
            "x", encoding="utf-8"
        ),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    assert seen == [(experiment_root, "det_01", run_dir, True, JobStatus.RUNNING)]


def test_a_failed_after_copy_back_step_fails_the_run_and_says_where_results_are(
    monkeypatch, experiment
):
    experiment_root, run_dir = experiment

    def after(root, job, folder):
        raise OSError("disk full")

    _with_hooks(monkeypatch, after=after)
    _completes(monkeypatch)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    error = rundir.read_envelope(run_dir)["error"]
    assert "could not be put in place (OSError: disk full)" in error
    assert str(run_dir) in error


def test_a_run_cancelled_during_the_upload_is_never_submitted(monkeypatch, experiment):
    experiment_root, run_dir = experiment
    scheduler = _FakeScheduler([JobState.RUNNING])
    _install_scheduler(monkeypatch, scheduler)
    monkeypatch.setattr(
        ssh_transport,
        "copy_to_remote",
        lambda *a, **k: rundir.request_cancel(run_dir),
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.CANCELLED
    assert not scheduler.submitted


# --- a training run's curve, live from the cluster -------------------------------------------- #
_RESULTS = "run/results.csv"


def _epoch(n):
    return {"type": "training", "epoch": n, "total_epochs": 3, "metrics": {}}


def _training_cluster(monkeypatch, envelopes, results_by_poll):
    """``read_remote_file`` for a training job: the envelope from ``envelopes`` one poll at a
    time, and ``results.csv`` as the cluster has it at that poll. Returns the paths read.
    """
    polls = iter(envelopes)
    state = {"poll": -1}
    reads = []

    def read(host, path, **_k):
        reads.append(path)
        if path.endswith("job.json"):
            state["poll"] += 1
            return json.dumps(next(polls))
        rows = results_by_poll[state["poll"]]
        if rows is None:
            raise ssh_transport.RemoteCommandError(["ssh"], 1, "", "No such file")
        return "epoch,metrics/mAP50(B)\n" + "".join(f"{e},0.{e}\n" for e in rows)

    monkeypatch.setattr(ssh_transport, "read_remote_file", read)
    return reads


def test_live_files_of_a_training_job_are_its_trainers(tmp_path):
    assert remote_shepherd._live_files(_training_job()) == (_RESULTS,)
    assert remote_shepherd._live_files(_job()) == ()


def test_a_training_curve_reaches_this_machine_before_its_epoch_is_announced(
    monkeypatch, experiment
):
    """The page redraws once per new epoch and remembers which epoch it drew. Published
    first, the epoch would be drawn from the file as it stood — one row short — and the
    redraw it was owed would never come; for a remote run, the file was never there at all.
    """
    _experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING] * 3))
    _training_cluster(
        monkeypatch,
        [
            {"status": "running", "progress": _epoch(1)},
            {"status": "running", "progress": _epoch(2)},
            {"status": "complete", "progress": _epoch(3)},
        ],
        [[1], [1, 2], [1, 2, 3]],
    )
    rows_when_published = []
    real_update = rundir.update_payload

    def update(path, **fields):
        if "progress" in fields:
            local = run_dir / _RESULTS
            text = local.read_text(encoding="utf-8-sig") if local.is_file() else ""
            rows_when_published.append(
                (fields["progress"]["epoch"], len(text.splitlines()) - 1)
            )
        return real_update(path, **fields)

    monkeypatch.setattr(rundir, "update_payload", update)

    status, _ = remote_shepherd._poll_until_terminal(
        run_dir,
        _target(),
        _FakeScheduler([]),
        "999",
        "/remote/run",
        live_files=(_RESULTS,),
    )

    assert status == JobStatus.COMPLETE
    assert rows_when_published == [(1, 1), (2, 2), (3, 3)]


def test_the_curve_is_fetched_once_per_new_event_not_every_poll(
    monkeypatch, experiment
):
    """An extra ssh ask every 5 s through an epoch that takes minutes is a cost with nothing
    to show for it: the file only changes when an epoch ends, which is when progress moves.
    """
    _experiment_root, run_dir = experiment
    same = {"status": "running", "progress": _epoch(1)}
    reads = _training_cluster(
        monkeypatch,
        [same, same, same, {"status": "complete", "progress": _epoch(2)}],
        [[1], [1], [1], [1, 2]],
    )

    remote_shepherd._poll_until_terminal(
        run_dir,
        _target(),
        _FakeScheduler([]),
        "999",
        "/remote/run",
        live_files=(_RESULTS,),
    )

    assert [p for p in reads if p.endswith("results.csv")] == [
        "/remote/run/run/results.csv"
    ] * 2


def test_a_curve_not_yet_written_does_not_disturb_the_poll(monkeypatch, experiment):
    _experiment_root, run_dir = experiment
    _training_cluster(
        monkeypatch,
        [
            {"status": "running", "progress": _epoch(0)},  # loading the base model
            {"status": "complete", "progress": _epoch(1)},
        ],
        [None, [1]],
    )

    status, _ = remote_shepherd._poll_until_terminal(
        run_dir,
        _target(),
        _FakeScheduler([]),
        "999",
        "/remote/run",
        live_files=(_RESULTS,),
    )

    assert status == JobStatus.COMPLETE
    assert (run_dir / _RESULTS).is_file()
