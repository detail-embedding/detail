"""Experiment-level distill-run records: ``distill/<run_id>/distill.yaml`` (§12).

A *distill run* is the unified label → train workflow driven as one durable, stage-keyed
run (the run-level twin of a processing run, §15). Its record is experiment-level and
stored under ``distill/``, discovered by scanning — the same primitives as ``runs/``, but a
parallel store because a distill run is stage-keyed (``label``/``train``), not
``module_id`` + ``per_recording``, and must not appear in the processing-runs index (the
same precedent as ``analysis_runs.py`` / ``tool_runs.py``). The produced dataset and trained
model keep their own ``datasets/<id>/`` / ``models/<id>/`` records; this only indexes and
sequences the run.
"""

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from detail.core import atomic
from detail.storage import dirscan
from detail.storage import id_allocator

SCHEMA_VERSION = 1
DISTILL_FILENAME = "distill.yaml"
CANCEL_FILENAME = "cancel"
PROGRESS_FILENAME = "progress.json"
COORDINATOR_PID_FILENAME = "coordinator.pid"


def distill_dir(experiment_dir: Path | str) -> Path:
    return Path(experiment_dir) / "distill"


def run_dir(experiment_dir: Path | str, run_id: str) -> Path:
    return distill_dir(experiment_dir) / run_id


def next_distill_id(experiment_dir: Path | str) -> str:
    """Claim the next ``distill_NNN`` id (creating its directory); a deleted run's id is never
    handed out again (``storage/id_allocator.py``)."""
    return id_allocator.claim_dir(distill_dir(experiment_dir), "distill", digits=3)


def write_run(experiment_dir: Path | str, run_id: str, record: dict) -> None:
    """Write ``distill/<run_id>/distill.yaml`` with ``schema_version`` as the first key (Tenet 7).

    Atomic (tmp + ``os.replace``) because the coordinator and a concurrent cancel can both
    write this record; a reader never sees a torn file."""
    rdir = run_dir(experiment_dir, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    ordered = {"schema_version": SCHEMA_VERSION}
    for key, value in record.items():
        if key != "schema_version":
            ordered[key] = value
    path = rdir / DISTILL_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(ordered, f, sort_keys=False)
    atomic.replace(tmp, path)


def read_run(experiment_dir: Path | str, run_id: str) -> dict:
    """Read one distill-run record.

    Raises:
        FileNotFoundError: If the run record is absent.
    """
    path = run_dir(experiment_dir, run_id) / DISTILL_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no {DISTILL_FILENAME} for distill run {run_id!r} in {experiment_dir}"
        )
    return yaml.safe_load(atomic.read_text(path))


def delete_run(experiment_dir: Path | str, run_id: str) -> None:
    """Remove the experiment-level ``distill/<run_id>/`` record directory (idempotent).

    Prunes only the index — never the produced dataset, trained model, or a promoted
    global model."""
    rdir = run_dir(experiment_dir, run_id)
    if rdir.is_dir():
        shutil.rmtree(rdir)


def discover_runs(experiment_dir: Path | str) -> list[str]:
    """Sorted ids of distill runs (subdirs of ``distill/`` holding a distill.yaml)."""
    return dirscan.discover_ids(distill_dir(experiment_dir), DISTILL_FILENAME)


def list_runs(experiment_dir: Path | str) -> list[dict]:
    """All distill-run records, ordered by run id."""
    return dirscan.read_each(
        discover_runs(experiment_dir), lambda rid: read_run(experiment_dir, rid)
    )


def request_cancel(experiment_dir: Path | str, run_id: str) -> None:
    """Touch ``distill/<run_id>/cancel`` — a read-only signal the run's detached
    coordinator polls between stages so it stops. Separate file, never rewritten by the
    coordinator's own record write, so it can't be clobbered."""
    rdir = run_dir(experiment_dir, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / CANCEL_FILENAME).write_text("", encoding="utf-8")


def clear_cancel(experiment_dir: Path | str, run_id: str) -> None:
    """Remove a stale cancel marker before relaunching the run (idempotent).

    A cancelled run's sentinel outlives the coordinator that honoured it; without this a
    resume's fresh coordinator sees it on its first stage and re-records ``cancelled``.
    """
    (run_dir(experiment_dir, run_id) / CANCEL_FILENAME).unlink(missing_ok=True)


def cancel_requested(experiment_dir: Path | str, run_id: str) -> bool:
    return (run_dir(experiment_dir, run_id) / CANCEL_FILENAME).exists()


# --- run-level folded progress (durable, poll-read across processes) --------- #
def write_progress(experiment_dir: Path | str, run_id: str, data: dict) -> None:
    """Atomically write ``distill/<run_id>/progress.json`` — the folded ``DistillProgress``
    a detached distill coordinator publishes and any reader (GUI reopen / CLI attach)
    polls. A unique temp per write, then atomic replace so a poller never sees a torn
    file (Windows sharing-violation retried by ``atomic``)."""
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
    """Latest folded progress dict, or ``None`` if not written yet."""
    path = run_dir(experiment_dir, run_id) / PROGRESS_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(atomic.read_text(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def set_coordinator_pid(experiment_dir: Path | str, run_id: str, pid: int) -> None:
    """Record the detached coordinator's pid (the parent writes this right after spawn)
    so reconcile/attach can tell a live coordinator from a dead one."""
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
