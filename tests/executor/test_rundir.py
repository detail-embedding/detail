import pytest

from detail.core.dataclasses import (
    ComputeTarget,
    DetectionJob,
    DetectionProgressEvent,
    JobStatus,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.executor import rundir


def _detection_job(job_id="det_01"):
    return DetectionJob(
        job_id=job_id,
        recording_id="recording_001",
        model_id="sam3",
        prompt=Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        conf_threshold=0.5,
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-01-01T00:00:00Z",
    )


def test_envelope_round_trip(tmp_path):
    job = _detection_job()
    rundir.init_run_dir(tmp_path, job)
    assert rundir.read_status(tmp_path) == JobStatus.PENDING
    assert rundir.read_job(tmp_path) == job
    assert rundir.read_progress(tmp_path) is None
    assert rundir.read_envelope(tmp_path)["schema_version"] == 1


def test_set_status_and_error(tmp_path):
    rundir.init_run_dir(tmp_path, _detection_job())
    rundir.set_status(tmp_path, JobStatus.FAILED, error="oops")
    env = rundir.read_envelope(tmp_path)
    assert env["status"] == "failed"
    assert env["error"] == "oops"


def test_set_status_stamps_started_at_on_first_running(tmp_path):
    rundir.init_run_dir(tmp_path, _detection_job())
    assert "started_at" not in rundir.read_envelope(tmp_path)
    rundir.set_status(tmp_path, JobStatus.RUNNING)
    env = rundir.read_envelope(tmp_path)
    assert env["started_at"]
    assert "completed_at" not in env


def test_set_status_stamps_completed_at_on_first_terminal(tmp_path):
    rundir.init_run_dir(tmp_path, _detection_job())
    rundir.set_status(tmp_path, JobStatus.RUNNING)
    rundir.set_status(tmp_path, JobStatus.COMPLETE)
    env = rundir.read_envelope(tmp_path)
    assert env["started_at"]
    assert env["completed_at"]
    assert env["completed_at"] >= env["started_at"]


def test_set_status_completed_at_is_idempotent_within_one_attempt(tmp_path):
    """A duplicate terminal write (e.g. local.py's cancel backstop firing after the
    worker already recorded its own terminal status) must not move the finish
    stamp — the first one is the real one."""
    rundir.init_run_dir(tmp_path, _detection_job())
    rundir.set_status(tmp_path, JobStatus.RUNNING)
    rundir.set_status(tmp_path, JobStatus.COMPLETE)
    first_completed_at = rundir.read_envelope(tmp_path)["completed_at"]
    rundir.set_status(tmp_path, JobStatus.COMPLETE)
    assert rundir.read_envelope(tmp_path)["completed_at"] == first_completed_at


def test_set_status_a_running_reaffirmation_leaves_started_at_alone(tmp_path):
    rundir.init_run_dir(tmp_path, _detection_job())
    rundir.set_status(tmp_path, JobStatus.RUNNING)
    first_started_at = rundir.read_envelope(tmp_path)["started_at"]
    rundir.set_status(tmp_path, JobStatus.RUNNING)
    assert rundir.read_envelope(tmp_path)["started_at"] == first_started_at


def test_set_status_resets_timing_on_resume_after_a_terminal_status(tmp_path):
    """A resumed run (re-entering RUNNING after FAILED/CANCELLED, possibly days
    later) gets a fresh started_at and loses the stale completed_at from the
    earlier attempt — duration/cost must reflect only the latest attempt, not a
    multi-day gap plus this attempt's own time (decided with the user)."""
    rundir.init_run_dir(tmp_path, _detection_job())
    rundir.set_status(tmp_path, JobStatus.RUNNING)
    rundir.set_status(tmp_path, JobStatus.FAILED, error="crashed")
    first_attempt = rundir.read_envelope(tmp_path)
    assert first_attempt["started_at"] and first_attempt["completed_at"]

    rundir.set_status(tmp_path, JobStatus.RUNNING)  # the resume
    resumed = rundir.read_envelope(tmp_path)
    assert "completed_at" not in resumed
    assert resumed["started_at"] >= first_attempt["started_at"]

    rundir.set_status(tmp_path, JobStatus.COMPLETE)
    final = rundir.read_envelope(tmp_path)
    assert final["started_at"] == resumed["started_at"]
    assert final["completed_at"]


def test_set_status_tolerates_an_envelope_with_no_prior_status(tmp_path):
    """A hand-built envelope with no "status" key at all (the shape a legacy
    envelope predating this feature would still satisfy, since it never had
    started_at/completed_at either) must not choke set_status's own
    previous_status lookup."""
    rundir._atomic_write(
        rundir._job_path(tmp_path),
        {"schema_version": 1, "job": None, "progress": None, "error": None},
    )
    rundir.set_status(tmp_path, JobStatus.RUNNING)
    rundir.set_status(tmp_path, JobStatus.COMPLETE)
    final = rundir.read_envelope(tmp_path)
    assert final["started_at"] and final["completed_at"]


def test_set_and_read_progress(tmp_path):
    rundir.init_run_dir(tmp_path, _detection_job())
    rundir.set_progress(tmp_path, DetectionProgressEvent(1, 2, 3, 4, 5.0, None))
    event = rundir.read_progress(tmp_path)
    assert isinstance(event, DetectionProgressEvent)
    assert event.parts_done == 1
    assert event.eta_s is None


def test_read_envelope_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rundir.read_envelope(tmp_path)


def test_cancel_flag(tmp_path):
    assert not rundir.cancel_requested(tmp_path)
    rundir.request_cancel(tmp_path)
    assert rundir.cancel_requested(tmp_path)


def test_atomic_write_leaves_no_tmp(tmp_path):
    rundir.init_run_dir(tmp_path, _detection_job())
    rundir.set_status(tmp_path, JobStatus.RUNNING)
    assert (tmp_path / "job.json").exists()
    assert not (tmp_path / "job.json.tmp").exists()


# --- worker.log: the spawned process's own output (#279) --------------------- #
def test_open_log_appends_so_a_reused_run_dir_keeps_the_failed_attempt(tmp_path):
    """A run dir is reused across attempts, so the log must append, never truncate.

    ``resume_training`` re-submits into the same ``models/<id>/``. Truncating on the
    resume would destroy the log of the attempt that *failed* — which is the only one
    anyone opens the file to read. The header line is what keeps two attempts apart.
    """
    with rundir.open_log(tmp_path) as handle:
        handle.write(b"first attempt output\n")
    with rundir.open_log(tmp_path) as handle:
        handle.write(b"second attempt output\n")

    text = rundir.log_path(tmp_path).read_text(encoding="utf-8-sig")
    assert "first attempt output" in text  # the failed attempt survived the resume
    assert "second attempt output" in text
    assert text.count("attempt started") == 2  # each attempt is delimited


def test_open_log_keeps_the_coordinator_separate_from_the_worker(tmp_path):
    """A durable run has both a coordinator and workers; their output must not interleave
    into one unreadable file."""
    with rundir.open_log(tmp_path, rundir.COORDINATOR_LOG_FILENAME) as handle:
        handle.write(b"coordinator says\n")
    with rundir.open_log(tmp_path) as handle:
        handle.write(b"worker says\n")

    assert "coordinator says" in rundir.log_path(
        tmp_path, rundir.COORDINATOR_LOG_FILENAME
    ).read_text(encoding="utf-8-sig")
    worker_log = rundir.log_path(tmp_path).read_text(encoding="utf-8-sig")
    assert "worker says" in worker_log
    assert "coordinator says" not in worker_log
