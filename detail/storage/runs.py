"""Experiment-level processing-run records: ``runs/<run_id>/run.yaml`` (§15).

A *processing run* applies one processing module over a selection of recordings
(`Experiment.run_processing`). Its record is experiment-level (it spans recordings),
stored under ``runs/`` and discovered by scanning — the same pattern as recordings
under ``recordings/``. Per-recording outputs still live in each recording (Tenet 5);
this is only the cross-recording index + status.
"""

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from detail.core import atomic
from detail.core import progress
from detail.storage import dirscan
from detail.storage import read_cache

SCHEMA_VERSION = 1
RUN_FILENAME = "run.yaml"
CANCEL_FILENAME = "cancel"
PROGRESS_FILENAME = "progress.json"
COORDINATOR_PID_FILENAME = "coordinator.pid"


def runs_dir(experiment_dir: Path | str) -> Path:
    return Path(experiment_dir) / "runs"


def run_dir(experiment_dir: Path | str, run_id: str) -> Path:
    return runs_dir(experiment_dir) / run_id


@contextlib.contextmanager
def runs_lock(experiment_dir: Path | str):
    """Serialize ``next_run_id`` + the reserving ``write_run`` across processes.

    The same read-modify-write hazard ``recording.recording_lock`` exists for, one level up
    and unprotected until it bit: ``next_run_id`` scans ``runs/`` and returns max+1, so two
    processing runs launched at the same moment both get ``run_NNN``, both write
    ``run.yaml.tmp`` to that one path, and the loser's ``os.replace`` fails with
    ``FileNotFoundError`` — its tmp already consumed by the winner. Measured with six
    embedding runs starting together.

    Advisory lock on a sibling ``runs/.lock``. Readers do not take it: ``write_run`` is
    atomic, so a read always sees a whole file.
    """
    directory = runs_dir(experiment_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with atomic.file_lock(directory / ".lock"):
        yield


def next_run_id(experiment_dir: Path | str) -> str:
    """Experiment-monotonic ``run_NNN`` id, one past the highest existing.

    Racy on its own — call it inside :func:`runs_lock`, holding the lock until the record is
    written, or two concurrent launches get the same id.
    """
    nums = []
    for rid in discover_runs(experiment_dir):
        try:
            nums.append(int(rid.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return f"run_{max(nums, default=0) + 1:03d}"


def write_run(experiment_dir: Path | str, run_id: str, record: dict) -> None:
    """Write ``runs/<run_id>/run.yaml`` with ``schema_version`` as the first key (Tenet 7).

    Atomic (tmp + ``os.replace``) because the orchestrator thread and a concurrent
    cancel can both write this record; a reader never sees a torn file."""
    rdir = run_dir(experiment_dir, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    ordered = {"schema_version": SCHEMA_VERSION}
    for key, value in record.items():
        if key != "schema_version":
            ordered[key] = value
    path = rdir / RUN_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(ordered, f, sort_keys=False)
    atomic.replace(tmp, path)


def read_run(experiment_dir: Path | str, run_id: str) -> dict:
    """Read one run record.

    Raises:
        FileNotFoundError: If the run record is absent.
    """
    path = run_dir(experiment_dir, run_id) / RUN_FILENAME
    try:
        return read_cache.cached_document(
            path, lambda: yaml.safe_load(atomic.read_text(path))
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"no {RUN_FILENAME} for run {run_id!r} in {experiment_dir}"
        ) from None


def delete_run(experiment_dir: Path | str, run_id: str) -> None:
    """Remove the experiment-level ``runs/<run_id>/`` record directory (idempotent)."""
    rdir = run_dir(experiment_dir, run_id)
    if rdir.is_dir():
        shutil.rmtree(rdir)


def discover_runs(experiment_dir: Path | str) -> list[str]:
    """Sorted ids of runs (subdirs of ``runs/`` holding a run.yaml)."""
    return dirscan.discover_ids(runs_dir(experiment_dir), RUN_FILENAME)


def list_runs(experiment_dir: Path | str) -> list[dict]:
    """All run records, ordered by run id."""
    return dirscan.read_each(
        discover_runs(experiment_dir), lambda rid: read_run(experiment_dir, rid)
    )


def request_cancel(experiment_dir: Path | str, run_id: str) -> None:
    """Touch ``runs/<run_id>/cancel`` — a read-only signal the run's detached
    coordinator polls so it stops between recordings (mirrors the per-job cancel flag
    in ``executor.rundir``). Separate file, never rewritten by the coordinator's own
    record write, so it can't be clobbered."""
    rdir = run_dir(experiment_dir, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / CANCEL_FILENAME).write_text("", encoding="utf-8")


def clear_cancel(experiment_dir: Path | str, run_id: str) -> None:
    """Remove a stale run-level cancel marker before relaunching the run (idempotent).

    A cancelled run's sentinel outlives the coordinator that honoured it. Without this,
    a retry's fresh coordinator sees it on its very first recording, stops, and re-records
    the run ``cancelled`` — the run-level twin of ``executor.rundir.clear_cancel``.
    """
    (run_dir(experiment_dir, run_id) / CANCEL_FILENAME).unlink(missing_ok=True)


def cancel_requested(experiment_dir: Path | str, run_id: str) -> bool:
    return (run_dir(experiment_dir, run_id) / CANCEL_FILENAME).exists()


# --- run-level aggregate progress (durable, poll-read across processes) ------ #
@progress.best_effort("run progress")
def write_progress(experiment_dir: Path | str, run_id: str, data: dict) -> None:
    """Atomically write ``runs/<run_id>/progress.json`` — the aggregate progress a
    detached run coordinator publishes and any reader (GUI reopen / CLI attach) polls.
    A unique temp per write (the parallel orchestrator emits from several recording
    threads at once), then atomic replace so a poller never sees a torn file (Windows
    sharing-violation retried by ``atomic``).

    **Cannot raise.** This exact write is the one that failed in #289, and its caller is the
    coordinator's ``emit`` — a statement about the run, sitting inside the handler that decides
    whether a *recording* failed. See ``core.progress.best_effort`` for why publishing is the
    one thing here allowed to give up quietly.
    """
    rdir = run_dir(experiment_dir, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / PROGRESS_FILENAME
    fd, tmp = tempfile.mkstemp(dir=rdir, prefix=".progress.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        atomic.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read_progress(experiment_dir: Path | str, run_id: str) -> dict | None:
    """Latest aggregate progress dict, or ``None`` if not written yet."""
    path = run_dir(experiment_dir, run_id) / PROGRESS_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(atomic.read_text(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def set_coordinator_pid(experiment_dir: Path | str, run_id: str, pid: int) -> None:
    """Record the detached coordinator's pid (the parent writes this right after
    spawn) so reconcile/attach can tell a live coordinator from a dead one."""
    rdir = run_dir(experiment_dir, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / COORDINATOR_PID_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(str(pid), encoding="utf-8")
    atomic.replace(tmp, path)


def read_coordinator_pid(experiment_dir: Path | str, run_id: str) -> int | None:
    """The recorded coordinator pid, or ``None`` if absent/unreadable."""
    path = run_dir(experiment_dir, run_id) / COORDINATOR_PID_FILENAME
    if not path.is_file():
        return None
    try:
        return int(atomic.read_text(path).strip())
    except (ValueError, FileNotFoundError):
        return None
