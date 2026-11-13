"""RemoteExecutor — submit()/cancel() are thin overrides of LocalExecutor; the actual
remote orchestration is exercised in test_remote_shepherd.py. Here we only need to
confirm submit() spawns the shepherd (not the ordinary worker) with the right argv, and
that cancel() reaches the scheduler directly rather than only the shepherd process."""

import contextlib
import subprocess
import sys
from pathlib import Path

import pytest

from detail.configs.remote import RemoteTargetConfig
from detail.core.dataclasses import (
    ComputeTarget,
    DetectionJob,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.executor import local as local_module
from detail.executor import remote as remote_module
from detail.executor import rundir
from detail.executor.remote import RemoteExecutor
from detail.executor.runners import JOB_RUNNERS, register_runner
from detail.storage import remote_targets as remote_targets_store


def _run_dir_for(experiment_root, job):
    return Path(experiment_root) / "runs" / job.job_id


@contextlib.contextmanager
def _registered():
    prior = JOB_RUNNERS.get("detection")
    register_runner("detection", run_dir_for=_run_dir_for)(lambda *a, **k: None)
    try:
        yield
    finally:
        if prior is not None:
            JOB_RUNNERS["detection"] = prior
        else:
            JOB_RUNNERS.pop("detection", None)


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


class _FakePopen:
    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.pid = 424242

    def poll(self):
        return None


def test_submit_spawns_the_shepherd_not_the_worker(tmp_path, monkeypatch):
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        return _FakePopen(argv, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    with _registered():
        ex = RemoteExecutor(tmp_path, target="tillicum")
        job_id = ex.submit(_job())

    assert job_id == "det_01"
    assert captured["argv"][:3] == [
        sys.executable,
        "-m",
        "detail.executor.remote_shepherd",
    ]
    run_dir = Path(captured["argv"][3])
    assert captured["argv"][4] == "tillicum"
    assert captured["argv"][5] == str(tmp_path)
    assert rundir.read_pid(run_dir) == 424242
    assert rundir.read_job(run_dir).job_id == "det_01"


def test_cancel_goes_through_cancel_run_dir_which_sends_the_scancel(
    tmp_path, monkeypatch
):
    """One path for every cancel: ``cancel_run_dir`` sends the scancel itself, so a run
    cancelled by its directory alone (after a restart, with no handle) reaches the cluster too
    (tests/executor/test_remote_cancel.py). The handle's cancel is that same call."""
    scheduler_calls = []

    class _FakeScheduler:
        def cancel(self, target, remote_job_id):
            scheduler_calls.append(remote_job_id)

    monkeypatch.setitem(remote_module._SCHEDULERS, "slurm", _FakeScheduler())
    monkeypatch.setattr(
        remote_targets_store,
        "get_remote_target",
        lambda name: RemoteTargetConfig(
            name=name, host="tillicum", remote_workdir="/scratch/x"
        ),
    )

    local_cancel_calls = []
    monkeypatch.setattr(
        local_module,
        "cancel_run_dir",
        lambda run_dir: local_cancel_calls.append(run_dir),
    )

    with _registered():
        ex = RemoteExecutor(tmp_path, target="tillicum")
        run_dir = tmp_path / "runs" / "det_01"
        run_dir.mkdir(parents=True)
        rundir.init_run_dir(run_dir, _job())
        rundir.update_payload(run_dir, remote_job_id="999")
        ex._runs["det_01"] = run_dir

        ex.cancel("det_01")

    assert local_cancel_calls == [run_dir]
    assert scheduler_calls == [], "not a second scancel beside cancel_run_dir's"


def test_cancel_without_a_remote_job_id_still_falls_back_to_local_kill(
    tmp_path, monkeypatch
):
    """submit() failed before the shepherd ever recorded a remote_job_id — cancel must
    still reach the (still-running) shepherd process rather than raising."""
    local_cancel_calls = []
    monkeypatch.setattr(
        local_module,
        "cancel_run_dir",
        lambda run_dir: local_cancel_calls.append(run_dir),
    )
    with _registered():
        ex = RemoteExecutor(tmp_path, target="tillicum")
        run_dir = tmp_path / "runs" / "det_01"
        run_dir.mkdir(parents=True)
        rundir.init_run_dir(run_dir, _job())
        ex._runs["det_01"] = run_dir

        ex.cancel("det_01")

    assert local_cancel_calls == [run_dir]
