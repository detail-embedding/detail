"""A remote run's terminal status reaches this machine only after its data does (#742).

**The class: a terminal status visible before the output it vouches for is in place.** (The same
shape as an authored run's ``open`` written before its artifact, ``docs/recorded_classes.md``; this
is its remote copy-back instance.) When a remote job ended, the shepherd copied the job's run
folder back with ``rsync``, ``job.json`` included. The cluster's ``job.json`` already said
``complete`` and sorts before the data, so it landed first. The coordinator polls exactly that
file (``LocalExecutor.status``), sees a terminal status, and reaps: it waits ten seconds
(``_REAP_TIMEOUT_S``) and then SIGKILLs the shepherd's process group, ``rsync`` with it. Every log
ended at "syncing results back" with one rsync temp file stamped ten seconds later, and 153 of 153
embedding runs on one experiment were recorded complete with hundreds of chunks still on the
cluster, which zarr reads back as zeros without a word.

**How it hid.** Every shepherd test faked ``copy_from_remote`` as an instant write, so nothing
ever observed the run *during* the copy; and the copy that wrote the cluster's envelope over the
local one was the tested behaviour (the bookkeeping was put back afterwards).

**The rule.** The copy back leaves the local envelope alone (``keep_local`` names it; rsync
excludes it, and the scp fallback never moves it, ``test_copy_back_keeps_local.py``). The worker's
own fields are merged in after the data (``_merge_worker_envelope``: everything but what this
machine owns, so a stage's results like detection's ``mask_resolution`` still arrive), and the
terminal status is the shepherd's last write. A shepherd killed during the copy leaves the run
non-terminal, which the reaper's liveness check then fails, instead of complete and empty.

**The fake copy here is faithful on the one point that matters:** like rsync it delivers the
cluster's ``job.json`` *before* the data, unless the caller keeps it local.
"""

import json

import pytest

from detail.core.dataclasses import JobStatus
from detail.executor import remote_shepherd, rundir, ssh_transport
from detail.executor.schedulers.base import JobState
from tests.executor.test_remote_shepherd import (  # noqa: F401
    _FakeScheduler,
    _install_scheduler,
    experiment,
)
from tests.executor.test_settled_end import _reads

_TERMINAL = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED}

#: The cluster's envelope at the end: terminal, with a stage's own result fields and its clock.
CLUSTER = {
    "schema_version": 1,
    "status": "complete",
    "error": None,
    "progress": {"type": "detection", "frames_done": 9, "frames_total": 9},
    "mask_resolution": [480, 640],
    "parts": {"part_001": "complete"},
    "started_at": "2026-10-09T03:33:13Z",
    "completed_at": "2026-10-09T03:48:03Z",
    "job": {"compute": {"backend": "cuda"}},
}


def _copy_like_rsync(run_dir, seen: dict, *, die_midway: bool = False):
    """A copy back that writes ``job.json`` first (it sorts first), unless kept local, then the
    data — and records the run's status as the data is arriving."""

    def copy(host, remote_dir, local_dir, *, keep_local=(), **kwargs):
        if rundir.JOB_FILENAME not in keep_local:
            (run_dir / rundir.JOB_FILENAME).write_text(
                json.dumps(CLUSTER), encoding="utf-8"
            )
        seen["status_while_copying"] = rundir.read_status(run_dir)
        if die_midway:
            raise SystemExit("killed")  # SIGKILL: no handler of the shepherd's runs
        (run_dir / "part_001.zarr").mkdir(exist_ok=True)

    return copy


@pytest.fixture
def ending(monkeypatch, experiment):
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    _reads(monkeypatch, json.dumps(CLUSTER))
    return experiment_root, run_dir


def test_the_status_is_not_terminal_while_the_data_is_still_arriving(
    monkeypatch, ending
):
    experiment_root, run_dir = ending
    seen: dict = {}
    monkeypatch.setattr(
        ssh_transport, "copy_from_remote", _copy_like_rsync(run_dir, seen)
    )

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert (
        seen["status_while_copying"] not in _TERMINAL
    ), "a reader polling the run saw it finished while its results were still copying back"
    assert status == JobStatus.COMPLETE
    assert rundir.read_status(run_dir) == JobStatus.COMPLETE  # the last write
    assert (run_dir / "part_001.zarr").is_dir()


def test_a_copy_killed_midway_leaves_the_run_unfinished_not_complete(
    monkeypatch, ending
):
    """What the reaper did, ten seconds in. The run must not read complete afterwards."""
    experiment_root, run_dir = ending
    seen: dict = {}
    monkeypatch.setattr(
        ssh_transport,
        "copy_from_remote",
        _copy_like_rsync(run_dir, seen, die_midway=True),
    )

    with pytest.raises(SystemExit):
        remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert rundir.read_status(run_dir) not in _TERMINAL


def test_a_stages_own_results_still_arrive_and_this_machines_fields_survive(
    monkeypatch, ending
):
    """Leaving the envelope out of the copy must not drop what a stage writes into it:
    detection's ``mask_resolution`` is how every mask is decoded."""
    experiment_root, run_dir = ending
    submitted = rundir.read_envelope(run_dir)
    monkeypatch.setattr(
        ssh_transport, "copy_from_remote", _copy_like_rsync(run_dir, {})
    )

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    envelope = rundir.read_envelope(run_dir)
    assert envelope["mask_resolution"] == [480, 640]
    assert envelope["parts"] == {"part_001": "complete"}
    assert envelope["progress"]["frames_done"] == 9
    assert envelope["worker_started_at"] == "2026-10-09T03:33:13Z"
    assert envelope["worker_completed_at"] == "2026-10-09T03:48:03Z"
    # ... and what only this machine knows is still this machine's
    assert envelope["job"] == submitted["job"]
    assert envelope["job"]["compute"]["backend"] == "remote"
    assert envelope["started_at"] != "2026-10-09T03:33:13Z"


def test_an_unreadable_cluster_envelope_after_the_copy_fails_the_run(
    monkeypatch, ending
):
    """The data is here but the stage's own fields are not: a detection run without
    ``mask_resolution`` cannot be read, so it is not complete. It says where the results are.
    """
    experiment_root, run_dir = ending
    seen: dict = {}
    copy = _copy_like_rsync(run_dir, seen)
    monkeypatch.setattr(ssh_transport, "copy_from_remote", copy)
    read = remote_shepherd._read_remote_envelope
    # Only the read *after* the copy fails; the poll before it reads normally.
    monkeypatch.setattr(
        remote_shepherd,
        "_read_remote_envelope",
        lambda *a: None if "status_while_copying" in seen else read(*a),
    )

    assert remote_shepherd.run(run_dir, "tillicum", experiment_root) == JobStatus.FAILED
    assert "could not be copied back" in rundir.read_envelope(run_dir)["error"]
