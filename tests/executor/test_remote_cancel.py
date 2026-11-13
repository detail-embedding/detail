"""Cancelling a remote run by its directory reaches the cluster (``local.cancel_run_dir``).

**The class: a cancel that stops the messenger and not the job.** A remote run's process on this
machine is its shepherd, which sends ``scancel`` only when a poll reads the cancel flag — every few
seconds, after an ssh round trip. ``cancel_run_dir`` gave it three seconds and then killed it, so
the poll that would have asked never came: the run read ``cancelled`` here while its Slurm job ran
on to its walltime, holding its GPUs, with nothing left that knew it existed. Every cancel path
that works from a run directory alone (the Runs page, ``cancel_analysis``, ``cancel_run`` after a
restart, a distill stage) went through it; only ``RemoteExecutor.cancel``, which needs a live
handle, sent the scancel itself.

**The rule.** The run directory says the job is remote (``remote_job_id`` and the job's
``remote_target``), so ``cancel_run_dir`` sends the scancel before it stops the shepherd. A
scancel that could not be delivered leaves the shepherd alive instead, because it is the one
process that asks again at every poll.
"""

import json
import subprocess
import sys

import pytest

from detail.configs.remote import RemoteTargetConfig
from detail.executor import local as local_mod
from detail.executor import remote as remote_mod
from detail.executor import rundir
from detail.executor.local import cancel_run_dir
from detail.storage import remote_targets as remote_targets_store


class _Scheduler:
    def __init__(self):
        self.delivered = True
        self.calls = []

    def cancel(self, target, remote_job_id):
        self.calls.append((target.name, remote_job_id))
        return self.delivered


def _envelope(run_dir, **extra):
    payload = {"schema_version": 1, "status": "running", "error": None, **extra}
    (run_dir / "job.json").write_text(json.dumps(payload), encoding="utf-8")


def _remote_envelope(run_dir, pid=None, remote_job_id="4242"):
    _envelope(
        run_dir,
        pid=pid,
        remote_job_id=remote_job_id,
        job={"compute": {"backend": "remote", "remote_target": "tillicum"}},
    )


@pytest.fixture
def scheduler(monkeypatch):
    fake = _Scheduler()
    monkeypatch.setitem(remote_mod._SCHEDULERS, "slurm", fake)
    monkeypatch.setattr(
        remote_targets_store,
        "get_remote_target",
        lambda name: RemoteTargetConfig(
            name=name, host=name, remote_workdir="/scratch/x"
        ),
    )
    return fake


@pytest.fixture
def shepherd(tmp_path):
    """A live stand-in for the shepherd: it carries the run dir in its argv, so the PID-reuse
    guard recognizes it as this run's process."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", str(tmp_path)],
        **local_mod.new_process_group_kwargs(),
    )
    yield proc
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)


def test_a_remote_runs_job_is_cancelled_on_the_cluster_then_its_shepherd_stopped(
    tmp_path, scheduler, shepherd
):
    _remote_envelope(tmp_path, pid=shepherd.pid)

    cancel_run_dir(tmp_path, grace_s=0.0)

    assert scheduler.calls == [("tillicum", "4242")]
    shepherd.wait(timeout=5)
    assert rundir.read_status(tmp_path).value == "cancelled"


def test_an_undelivered_scancel_leaves_the_shepherd_to_ask_again(
    tmp_path, scheduler, shepherd
):
    scheduler.delivered = False
    _remote_envelope(tmp_path, pid=shepherd.pid)

    cancel_run_dir(tmp_path, grace_s=0.0)

    assert scheduler.calls == [("tillicum", "4242")]
    assert shepherd.poll() is None
    assert rundir.cancel_requested(tmp_path)
    assert rundir.read_status(tmp_path).value == "running"  # the shepherd's to record


def test_a_remote_run_whose_shepherd_is_gone_is_still_cancelled_on_the_cluster(
    tmp_path, scheduler
):
    _remote_envelope(tmp_path, pid=None)

    cancel_run_dir(tmp_path, grace_s=0.0)

    assert scheduler.calls == [("tillicum", "4242")]
    assert rundir.read_status(tmp_path).value == "cancelled"


def test_a_remote_run_not_yet_submitted_asks_no_scheduler(
    tmp_path, scheduler, shepherd
):
    _remote_envelope(tmp_path, pid=shepherd.pid, remote_job_id=None)

    cancel_run_dir(tmp_path, grace_s=0.0)

    assert scheduler.calls == []
    shepherd.wait(timeout=5)
    assert rundir.read_status(tmp_path).value == "cancelled"


def test_a_local_run_never_asks_a_scheduler(tmp_path, scheduler):
    _envelope(tmp_path, job={"compute": {"backend": "cuda", "remote_target": None}})

    cancel_run_dir(tmp_path, grace_s=0.0)

    assert scheduler.calls == []
    assert rundir.read_status(tmp_path).value == "cancelled"


def test_a_target_no_longer_registered_leaves_the_cancel_to_the_shepherd(
    tmp_path, monkeypatch, shepherd
):
    def missing(name):
        raise KeyError(name)

    monkeypatch.setattr(remote_targets_store, "get_remote_target", missing)
    _remote_envelope(tmp_path, pid=shepherd.pid)

    cancel_run_dir(tmp_path, grace_s=0.0)

    assert shepherd.poll() is None
    assert rundir.cancel_requested(tmp_path)
