"""Run-directory IPC: the job.json envelope shared by executor and worker (Section 6).

job.json carries a generic envelope the executor reads (status, the tagged job,
the latest tagged progress event, an error message) plus, in practice,
stage-specific payload keys the runner manages (e.g. detection's ``parts``) that
the executor never touches. The worker is the only writer once running and the
executor only reads, so a single atomic write (tmp + os.replace) keeps the
polling reader from ever seeing a torn file — no locking needed.
"""

import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import BinaryIO

from detail.core import atomic
from detail.core import progress as core_progress
from detail.core.dataclasses import (
    Job,
    JobStatus,
    ProgressEvent,
    job_from_dict,
    job_to_dict,
    progress_event_from_dict,
    progress_event_to_dict,
)

SCHEMA_VERSION = 1
JOB_FILENAME = "job.json"
CANCEL_FILENAME = "cancel"
#: Stop-and-keep's marker — its own file, because the worker has to know *which* was asked.
#: Both stop the runner; only a cancel makes the run `partial`. One shared marker is what made
#: a user's chosen finish read as an interruption, and offered Resume on a finalized model.
STOP_FILENAME = "stop"
LOG_FILENAME = "worker.log"
COORDINATOR_LOG_FILENAME = "coordinator.log"


def _job_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / JOB_FILENAME


def _atomic_write(path: Path, envelope: dict) -> None:
    # Unique temp name per write: two processes legitimately write the envelope
    # during a run's startup — the executor records the worker pid right after
    # spawn (LocalExecutor.submit) while the worker writes status=running. A shared
    # "<name>.tmp" lets one writer's os.replace consume the other's temp, crashing
    # it with FileNotFoundError. With a unique temp each writer renames its own; the
    # last os.replace wins atomically and a reader never sees a torn file.
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(envelope, f)
        atomic.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def init_run_dir(run_dir: Path | str, job: Job) -> None:
    """Write the initial job.json envelope with status ``pending``."""
    _atomic_write(
        _job_path(run_dir),
        {
            "schema_version": SCHEMA_VERSION,
            "status": JobStatus.PENDING.value,
            "job": job_to_dict(job),
            "progress": None,
            "error": None,
        },
    )


def log_path(run_dir: Path | str, filename: str = LOG_FILENAME) -> Path:
    """Where a spawned process's stdout + stderr land for this run."""
    return Path(run_dir) / filename


def open_log(run_dir: Path | str, filename: str = LOG_FILENAME) -> BinaryIO:
    """Open a run's log for a process about to be spawned into it.

    A spawned worker inherits whatever stdout the launcher had, so without this its
    output goes to the launching console — or nowhere, under a windowed GUI — and the
    backend's own log, the one artifact that names where a stalled stage stopped, exists
    only in a terminal scrollback that dies with the window. That is what left #279
    undiagnosable.

    **Append, never truncate.** A run dir is reused across attempts (a resumed training
    re-submits into the same ``models/<id>/``), so truncating would destroy the log of
    the attempt that failed — the only one anyone wants to read. A timestamp header
    separates one attempt from the last.

    Deliberately a file and not a pipe: a ``PIPE`` nobody drains blocks the child on a
    full buffer once it has written ~64 KB, which is the same class of silent hang this
    exists to diagnose. The caller closes its own handle right after ``Popen``; the child
    holds a duplicate.
    """
    handle = open(log_path(run_dir, filename), "ab")
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    handle.write(f"\n===== attempt started {stamp} =====\n".encode())
    handle.flush()
    return handle


def read_envelope(run_dir: Path | str) -> dict:
    return json.loads(atomic.read_text(_job_path(run_dir)))


def read_status(run_dir: Path | str) -> JobStatus:
    return JobStatus(read_envelope(run_dir)["status"])


def read_job(run_dir: Path | str) -> Job:
    return job_from_dict(read_envelope(run_dir)["job"])


def read_progress(run_dir: Path | str) -> ProgressEvent | None:
    progress = read_envelope(run_dir).get("progress")
    return progress_event_from_dict(progress) if progress is not None else None


def set_status(
    run_dir: Path | str, status: JobStatus, error: str | None = None
) -> None:
    envelope = read_envelope(run_dir)
    envelope["status"] = status.value
    if error is not None:
        envelope["error"] = error
    _atomic_write(_job_path(run_dir), envelope)


def set_pid(run_dir: Path | str, pid: int) -> None:
    """Record the worker's process id in the envelope so a later cancel — even
    from a fresh process after the GUI restarted — can find and kill it."""
    envelope = read_envelope(run_dir)
    envelope["pid"] = int(pid)
    _atomic_write(_job_path(run_dir), envelope)


def read_pid(run_dir: Path | str) -> int | None:
    return read_envelope(run_dir).get("pid")


@core_progress.best_effort("job progress")
def set_progress(run_dir: Path | str, event: ProgressEvent) -> None:
    """Publish a worker's latest progress into its envelope.

    **Cannot raise**, unlike its neighbours: ``set_status`` / ``set_pid`` / ``update_payload``
    share ``_atomic_write`` but are sources of truth, and a caller that cannot record a terminal
    status must hear about it. Progress is the one field here that is only ever read by a bar,
    so a lost write costs a frame of animation — where a raised one reaches a runner's isolation
    boundary and is recorded as the *work* failing (#289).
    """
    envelope = read_envelope(run_dir)
    envelope["progress"] = progress_event_to_dict(event)
    _atomic_write(_job_path(run_dir), envelope)


def update_payload(run_dir: Path | str, **fields) -> None:
    """Merge stage-specific payload keys (e.g. parts, mask_resolution) into job.json.

    The executor envelope (status/job/progress/error) is untouched; these keys are
    opaque to the executor and managed by the stage's runner (Section 6/7)."""
    envelope = read_envelope(run_dir)
    envelope.update(fields)
    _atomic_write(_job_path(run_dir), envelope)


def request_cancel(run_dir: Path | str) -> None:
    (Path(run_dir) / CANCEL_FILENAME).write_text("", encoding="utf-8")


def clear_cancel(run_dir: Path | str) -> None:
    """Remove stale cancel and stop markers before reusing a run dir for a new job
    (idempotent). Only needed where a run dir is reused across jobs — e.g. a clip-annotation
    run whose every iteration submits under the same run id. Both go: either one left behind
    stops the next worker on its first poll."""
    (Path(run_dir) / CANCEL_FILENAME).unlink(missing_ok=True)
    (Path(run_dir) / STOP_FILENAME).unlink(missing_ok=True)


def cancel_requested(run_dir: Path | str) -> bool:
    return (Path(run_dir) / CANCEL_FILENAME).exists()


def request_stop(run_dir: Path | str) -> None:
    """Ask the run to finish early and keep what it has (never a kill — `local.request_stop`)."""
    (Path(run_dir) / STOP_FILENAME).write_text("", encoding="utf-8")


def stop_requested(run_dir: Path | str) -> bool:
    return (Path(run_dir) / STOP_FILENAME).exists()
