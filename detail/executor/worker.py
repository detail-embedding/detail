"""Subprocess entry point that runs one job (Section 6).

Invoked as ``python -m detail.executor.worker <run_dir>``. Reads job.json,
dispatches to the registered runner, and is the isolation boundary: it catches
all runner exceptions and records them as a ``failed`` status before exiting.
"""

import os
import sys
import threading
import time
import traceback

from detail.core import resource_peaks
from detail.core.dataclasses import JobStatus
from detail.executor import rundir
from detail.storage import activity
from detail.storage import run_facts
from detail.executor.runners import (
    JOB_RUNNERS,
    RUNNER_MODULES_ENV,
    import_runner_modules,
)

_CANCEL_POLL_S = 0.2


def _start_cancel_watcher(run_dir, stop_event: threading.Event) -> None:
    """Set ``stop_event`` when a cancel or a stop-and-keep is requested, for cooperative
    shutdown. The runner sees one event; which marker set it decides the terminal status.
    """

    def watch() -> None:
        while not stop_event.wait(_CANCEL_POLL_S):
            if rundir.cancel_requested(run_dir) or rundir.stop_requested(run_dir):
                stop_event.set()
                return

    threading.Thread(target=watch, daemon=True).start()


def run_job(run_dir) -> JobStatus:
    """Run the job described by ``run_dir/job.json`` and return its terminal status."""
    job_type = rundir.read_envelope(run_dir)["job"]["type"]
    spec = JOB_RUNNERS.get(job_type)
    if spec is None:
        rundir.set_status(
            run_dir,
            JobStatus.FAILED,
            error=f"no runner registered for job type {job_type!r}",
        )
        return JobStatus.FAILED

    job = rundir.read_job(run_dir)
    stop_event = threading.Event()
    _start_cancel_watcher(run_dir, stop_event)
    rundir.set_status(run_dir, JobStatus.RUNNING)
    # A single-job run has one unit, so `unit` stays empty and the phase carries the detail —
    # the same log a multi-recording run writes, so one reader serves every run kind rather than
    # a second shape for the runs that happen not to loop over recordings.
    log = activity.ActivityLog(run_dir)
    start = time.monotonic()

    def publish(event) -> None:
        rundir.set_progress(run_dir, event)
        log.sample(
            activity.make_row(
                elapsed_s=time.monotonic() - start,
                phase=getattr(event, "message", "") or "",
                done=getattr(event, "frames_done", 0) or 0,
                total=getattr(event, "frames_total", 0) or 0,
                eta_unit_s=getattr(event, "eta_s", None),
                eta_run_s=getattr(event, "eta_s", None),
            )
        )

    peaks = resource_peaks.PeakSampler().start()
    try:
        spec.run(job, run_dir, publish, stop_event)
    except KeyboardInterrupt:
        # Ctrl-C reaches the worker directly (shared process group). Record a
        # terminal status before exiting so the run isn't left stuck "running"
        # and blocking the next run (the parent's stream_progress guard can't —
        # Ctrl-C kills it too).
        rundir.set_status(run_dir, JobStatus.CANCELLED, error="interrupted (SIGINT)")
        return JobStatus.CANCELLED
    except Exception as exc:  # isolation boundary (CLAUDE.md): record and exit
        # The envelope records only `Type: message`, so without this the traceback — the
        # part that says *where* it failed — is discarded at the one moment it is wanted.
        # It goes to stderr, which `LocalExecutor.submit` points at run_dir/worker.log.
        traceback.print_exc()
        rundir.set_status(
            run_dir, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}"
        )
        return JobStatus.FAILED
    finally:
        stop_event.set()
        run_facts.record(run_dir, peaks)

    # A stop-and-keep alone is a finish the user chose, so it lands COMPLETE; only a cancel is
    # an interruption. (A cancel wins if both were asked.)
    status = (
        JobStatus.PARTIAL if rundir.cancel_requested(run_dir) else JobStatus.COMPLETE
    )
    rundir.set_status(run_dir, status)
    return status


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: python -m detail.executor.worker <run_dir>")
    modules = [m for m in os.environ.get(RUNNER_MODULES_ENV, "").split(",") if m]
    import_runner_modules(modules)
    status = run_job(argv[0])
    return 0 if status == JobStatus.COMPLETE else 1


if __name__ == "__main__":
    sys.exit(main())
