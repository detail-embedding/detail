"""What the shepherd concludes when a job ends, and what it keeps from the cluster (§6).

**The class: a decision taken on an answer that was never obtained.** Two of them, in the
place the shepherd decides how a remote job *ended*.

1. The poll reads the worker's envelope, *then* asks the scheduler. A job that finishes between
   the two has written its terminal status after the read the poll is still holding, so "the
   scheduler says completed, the envelope I read says running" was read as "the worker died without
   a status" and a finished detection was recorded as failed. And when the re-read was *refused*
   (a full ssh connection), "could not ask" was read as "nothing written", same verdict, a real
   run lost (recording_038 of an 80-recording detection). The poll's ask had learned
   (`_Reconnect`) that being unable to ask is no answer; the decision made once the job is over
   had not.
2. The copy back was wrapped so that a failure would not hide the job's real status, and then
   the status it kept was **complete** — with the results never copied, a run that reads complete
   and has nothing on disk.

**It hid because every test gave the fake scheduler and the fake ``cat`` a definite answer
at the right moment.** None made the envelope lag the scheduler by one read, and none made the
copy back fail. These do, and they check the things that must not change: a worker that really
died with no status is still a failure, and a job that failed on the cluster keeps its own error.

The last tests cover what the sync keeps: the worker's own start and finish, which the cost
estimate bills (a job that waits five hours in the queue and runs for twenty minutes costs twenty
minutes). They arrive by the read of the cluster's envelope after the data, not by the copy,
which leaves the local envelope alone (#742, ``test_status_after_data.py``).
"""

import json

import pytest

from detail.core.dataclasses import JobStatus
from detail.executor import remote_shepherd, rundir, ssh_transport
from detail.executor.schedulers.base import JobState

# Fixtures and helpers of the shepherd tests: the same experiment, fake scheduler and transport.
from tests.executor.test_remote_shepherd import (  # noqa: F401
    _FakeScheduler,
    _install_scheduler,
    _job,
    experiment,
)

RUNNING = json.dumps({"status": "running", "progress": None})
COMPLETE = json.dumps({"status": "complete", "progress": None, "error": None})
REFUSED = ssh_transport.RemoteCommandError(
    ["ssh", "tillicum", "cat"],
    255,
    "",
    "mux_client_request_session: session request failed: Session open refused by peer",
)


def _reads(monkeypatch, *answers):
    """``read_remote_file`` answers in turn (the last repeats); an exception is raised."""
    queue = list(answers)
    calls = {"n": 0}

    def fake(host, path, **kwargs):
        calls["n"] += 1
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(ssh_transport, "read_remote_file", fake)
    return calls


def test_a_job_that_finishes_between_the_read_and_the_ask_is_complete(
    monkeypatch, experiment
):
    """read: running → ask: COMPLETED → re-read: still running (the write has not landed yet) →
    re-read: complete. Read once and believed, this was "completed without a terminal status";
    one re-read is not enough either, since the write can lag the scheduler by a moment.
    """
    experiment_root, run_dir = experiment
    _install_scheduler(
        monkeypatch, _FakeScheduler([JobState.RUNNING, JobState.COMPLETED])
    )
    _reads(monkeypatch, RUNNING, RUNNING, RUNNING, COMPLETE)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    assert rundir.read_envelope(run_dir).get("error") is None


def test_a_refused_read_after_the_job_ended_waits_instead_of_failing_it(
    monkeypatch, experiment
):
    """The incident: the ask for the result file is refused (a full connection) just as the job
    ends. That says nothing about the file, so the run waits for the connection and reads it.
    """
    experiment_root, run_dir = experiment
    scheduler = _FakeScheduler([JobState.RUNNING] + [JobState.COMPLETED] * 5)
    _install_scheduler(monkeypatch, scheduler)
    _reads(monkeypatch, RUNNING, RUNNING, REFUSED, REFUSED, COMPLETE)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE


def test_a_worker_that_really_died_without_a_status_is_still_a_failure(
    monkeypatch, experiment
):
    """The fallback the re-read must not remove: the scheduler says the job ended, the worker's
    envelope was readable on every try, and it never says it finished (OOM-killed, preempted).
    """
    experiment_root, run_dir = experiment
    _install_scheduler(
        monkeypatch, _FakeScheduler([JobState.RUNNING] + [JobState.COMPLETED] * 5)
    )
    calls = _reads(monkeypatch, RUNNING)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    assert "without a terminal status" in rundir.read_envelope(run_dir)["error"]
    assert (
        calls["n"] >= remote_shepherd.SETTLE_READS
    )  # it asked again before believing it


def test_a_job_that_failed_on_the_cluster_keeps_its_own_error(monkeypatch, experiment):
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    _reads(monkeypatch, json.dumps({"status": "failed", "error": "CUDA OOM"}))

    assert remote_shepherd.run(run_dir, "tillicum", experiment_root) == JobStatus.FAILED
    assert "CUDA OOM" in rundir.read_envelope(run_dir)["error"]


def test_a_copy_back_that_fails_does_not_leave_a_complete_run_with_no_results(
    monkeypatch, experiment
):
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    _reads(monkeypatch, COMPLETE)

    def refused(host, remote_dir, local_dir, **kwargs):
        raise REFUSED

    monkeypatch.setattr(ssh_transport, "copy_from_remote", refused)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    error = rundir.read_envelope(run_dir)["error"]
    assert "could not be copied back" in error
    assert "/gpfs/scrubbed/uwnetid/detail_runs/exp1" in error or "detail_runs" in error
    assert "tillicum" in error  # says where the results are


def test_a_failed_copy_back_does_not_hide_a_failure_that_was_already_one(
    monkeypatch, experiment
):
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    _reads(monkeypatch, json.dumps({"status": "failed", "error": "CUDA OOM"}))
    monkeypatch.setattr(
        ssh_transport,
        "copy_from_remote",
        lambda *a, **k: (_ for _ in ()).throw(REFUSED),
    )

    assert remote_shepherd.run(run_dir, "tillicum", experiment_root) == JobStatus.FAILED
    assert "CUDA OOM" in rundir.read_envelope(run_dir)["error"]


# --------------------------------------------------------------------------- #
# what the sync keeps: the worker's own clock
# --------------------------------------------------------------------------- #
def _sync_writing(run_dir, **stamps):
    """A ``copy_from_remote`` like the real one: it brings back the run folder, and the worker's
    envelope (with ``stamps``) only if the caller did not keep it local — which the shepherd
    does (#742), so the clock reaches it through the read of the cluster's envelope instead.
    """

    def sync(host, remote_dir, local_dir, **kwargs):
        (run_dir / "result.txt").write_text("from the worker", encoding="utf-8")
        if rundir.JOB_FILENAME in kwargs.get("keep_local", ()):
            return
        envelope = {"schema_version": 1, "status": "complete", "error": None}
        envelope.update(stamps)
        (run_dir / "job.json").write_text(json.dumps(envelope), encoding="utf-8")

    return sync


def _complete_with(**stamps) -> str:
    return json.dumps({"status": "complete", "progress": None, "error": None, **stamps})


def test_the_workers_own_start_and_finish_are_kept_beside_the_submission_time(
    monkeypatch, experiment
):
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    _reads(
        monkeypatch,
        _complete_with(
            started_at="2026-10-06T16:03:25Z", completed_at="2026-10-06T16:24:59Z"
        ),
    )
    monkeypatch.setattr(ssh_transport, "copy_from_remote", _sync_writing(run_dir))

    assert (
        remote_shepherd.run(run_dir, "tillicum", experiment_root) == JobStatus.COMPLETE
    )

    envelope = rundir.read_envelope(run_dir)
    assert envelope["worker_started_at"] == "2026-10-06T16:03:25Z"
    assert envelope["worker_completed_at"] == "2026-10-06T16:24:59Z"
    # ... and `started_at` is still when this machine submitted it, not the worker's
    assert envelope["started_at"] != "2026-10-06T16:03:25Z"


def test_a_job_that_never_started_on_a_gpu_records_that_it_did_not(
    monkeypatch, experiment
):
    """Cancelled while still queued: the cluster's envelope has no ``started_at``. The keys are
    still written (as ``None``), because their *presence* is what tells the cost estimate the
    record knows its running time, and that it was zero."""
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    _reads(monkeypatch, json.dumps({"status": "cancelled", "error": None}))
    monkeypatch.setattr(
        ssh_transport,
        "copy_from_remote",
        _sync_writing(run_dir, status="cancelled"),
    )

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    envelope = rundir.read_envelope(run_dir)
    assert "worker_started_at" in envelope and envelope["worker_started_at"] is None


def test_without_a_sync_no_worker_clock_is_claimed(monkeypatch, experiment):
    """An unreachable cluster brought nothing back, so the local envelope's own stamps are the
    submission's; recording them as the worker's would bill the queue as running time.
    """
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    _reads(monkeypatch, COMPLETE)

    def refused(host, remote_dir, local_dir, **kwargs):
        raise REFUSED

    monkeypatch.setattr(ssh_transport, "copy_from_remote", refused)

    remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert "worker_started_at" not in rundir.read_envelope(run_dir)
