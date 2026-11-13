"""Upload referenced video to a remote target in the background, ahead of the job that needs it.

A remote labeling job no longer uploads video (it is sent its sampled frames), but the detection run
that follows a distill reads *every* recording, and its first act used to be to upload them all:
80 recordings was ~116 GB, about an hour, with the run waiting. Staging moves that copy off the
critical path: it is the same per-file upload the shepherd does (``remote_shepherd._sync_video_roots``,
into the same persistent cache on the cluster), run in a **detached process** that starts when a
distill asks for it and keeps going while labeling and training run, however long they take.

It is **opt-in** (``Experiment.distill(stage_videos=True)``): it copies every referenced video to the
cluster's storage, which can be 100+ GB the user did not necessarily mean to spend.

A staging run is one process per ``(experiment, target)``, recorded in
``<experiment>/staging/<target>.json``: status, pid, files and bytes done, a measured ETA. That file
is how anything else sees it, and how a later remote job avoids uploading the same files a second time
over the first: its shepherd waits for a live staging run that covers any of its recordings
(:func:`wait_for_active`), publishing the staging run's progress as its own phase, then finds the
files already there. If the staging run failed, the shepherd carries on and uploads what is left
itself, so a failed background copy costs time and never correctness.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

from detail.core import atomic
from detail.core import progress as core_progress
from detail.core.progress import PhaseClock, ThrottledProgress
from detail.core.utils import utc_timestamp
from detail.executor import local, rundir
from detail.storage import locations

STAGING_DIRNAME = "staging"
POLL_INTERVAL_S = 5.0

_SPAWNED: list[subprocess.Popen] = []


def staging_dir(experiment_root) -> Path:
    """Where this experiment's staging state and log live: **outside** the experiment tree.

    The shepherd mirrors the whole experiment directory to the cluster. A state file inside it would
    be uploaded mid-run, a stale copy of a file that is rewritten every half second. It used to be
    worse: the job's end copied the whole remote experiment back over the local one, rewinding a
    running upload's progress (and its pid) to whatever it said at submission. That copy back is now
    the job's own run folder, so this is no longer the thing keeping the file safe, and it stays
    out of the tree as the second line of defence. It sits beside the other machine-global config,
    in a folder named for the experiment's resolved path.
    """
    resolved = str(Path(experiment_root).resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    return locations.CONFIG_DIR / STAGING_DIRNAME / f"{Path(resolved).name}-{digest}"


def state_path(experiment_root, target_name: str) -> Path:
    return staging_dir(experiment_root) / f"{target_name}.json"


def read_state(experiment_root, target_name: str) -> dict | None:
    """The staging run's recorded state for ``target_name``, or ``None`` when there is none."""
    try:
        return json.loads(atomic.read_text(state_path(experiment_root, target_name)))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def all_states(experiment_root) -> list[dict]:
    """Every staging run recorded for this experiment, one per target."""
    states = []
    for path in sorted(staging_dir(experiment_root).glob("*.json")):
        state = read_state(experiment_root, path.stem)
        if state is not None:
            states.append(state)
    return states


def _write_state(experiment_root, target_name: str, state: dict) -> None:
    """Atomic: other processes read this file while it is being rewritten."""
    path = state_path(experiment_root, target_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    atomic.replace(temp, path)


@core_progress.best_effort("video staging progress")
def _publish_state(experiment_root, target_name: str, state: dict) -> None:
    """A progress sample: losing one is invisible, so it can never fail the upload it describes."""
    _write_state(experiment_root, target_name, state)


def is_active(experiment_root, state: dict | None) -> bool:
    """Whether ``state`` describes a staging process that is still running (pid-reuse guarded)."""
    return (
        bool(state)
        and state.get("status") == "running"
        and local.coordinator_alive(Path(experiment_root).resolve(), state.get("pid"))
    )


def _initial_state(target_name: str, recording_ids: list[str]) -> dict:
    return {
        "schema_version": 1,
        "status": "running",
        "target": target_name,
        "recording_ids": list(recording_ids),
        "pid": None,
        "started_at": utc_timestamp(),
        "updated_at": utc_timestamp(),
        "files_done": 0,
        "files_total": 0,
        "bytes_done": 0,
        "bytes_current": 0,
        "bytes_total": 0,
        "eta_s": None,
        "message": "starting",
        "error": None,
    }


def start(experiment_root, target_name: str, recording_ids: list[str]) -> dict:
    """Begin staging ``recording_ids``' videos onto ``target_name`` in a detached process.

    Idempotent: when a staging run for this target is already alive, it is returned as it is rather
    than a second one started over it. Returns the recorded state.

    Raises:
        OSError: the process could not be started.
    """
    experiment_root = Path(experiment_root).resolve()
    existing = read_state(experiment_root, target_name)
    if is_active(experiment_root, existing):
        return existing
    state = _initial_state(target_name, recording_ids)
    _write_state(experiment_root, target_name, state)
    with rundir.open_log(staging_dir(experiment_root), f"{target_name}.log") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "detail.executor.video_staging",
                str(experiment_root),
                target_name,
                *recording_ids,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **local.new_process_group_kwargs(),
        )
    _SPAWNED[:] = [p for p in _SPAWNED if p.poll() is None]
    _SPAWNED.append(proc)
    state["pid"] = proc.pid
    _write_state(experiment_root, target_name, state)
    return state


def stop(experiment_root, target_name: str) -> bool:
    """Stop a running staging process and record it as cancelled. ``True`` if one was running.

    Files already on the cluster stay there: the upload is a cache, and a later run reuses them.
    """
    state = read_state(experiment_root, target_name)
    if not is_active(experiment_root, state):
        return False
    local.kill_coordinator(Path(experiment_root).resolve(), state.get("pid"))
    state.update(
        status="cancelled",
        updated_at=utc_timestamp(),
        eta_s=None,
        message="stopped",
    )
    _write_state(experiment_root, target_name, state)
    return True


def wait_for_active(
    experiment_root,
    target_name: str,
    recording_ids: list[str],
    on_wait: Callable[[dict], None],
    sleep: Callable[[float], None] | None = None,
    poll_s: float = POLL_INTERVAL_S,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict | None:
    """Block while a staging run for ``target_name`` covering any of ``recording_ids`` is alive.

    ``on_wait(state)`` is called with each state seen, so the waiter can show it. Returns the last
    state it waited on (re-read once the run ended, so its final status is visible), or ``None``
    when there was nothing to wait for. A run that does not cover the caller's recordings is not
    waited on, and the staging process never waits on itself. ``cancelled()`` ends the wait early,
    returning the live run's state: a cancelled job must not sit behind an upload it no longer
    needs. It only matters while there *is* a live upload to wait on, so a cancel with none
    changes nothing here.
    """
    waited = False
    while True:
        state = read_state(experiment_root, target_name)
        if (
            not is_active(experiment_root, state)
            or state.get("pid") == os.getpid()
            or not set(state.get("recording_ids") or ()) & set(recording_ids)
        ):
            return state if waited else None
        if cancelled():
            return state
        waited = True
        on_wait(state)
        (sleep or time.sleep)(poll_s)


class _StagingProgress:
    """The progress sink the shepherd's upload loop drives, writing the staging state file.

    Same ``videos`` call the shepherd's own ``_TransferProgress`` takes, so the one upload loop
    serves both. The ETA is measured over the upload's own clock, weighed by bytes.
    """

    def __init__(self, experiment_root, target_name: str, state: dict) -> None:
        self._root = experiment_root
        self._target = target_name
        self._state = state
        self._clock = PhaseClock(time.monotonic())
        self._throttle = ThrottledProgress(self._publish)

    def videos(
        self,
        files_done: int,
        files_total: int,
        bytes_done: int,
        bytes_total: int,
        current_bytes: int = 0,
        *,
        force: bool = False,
    ) -> None:
        fraction = (bytes_done + current_bytes) / bytes_total if bytes_total else 0.0
        self._throttle(
            {
                "files_done": files_done,
                "files_total": files_total,
                "bytes_done": bytes_done,
                "bytes_current": current_bytes,
                "bytes_total": bytes_total,
                "eta_s": self._clock.eta(
                    "videos", min(fraction, 1.0), time.monotonic()
                ),
                "message": (
                    f"uploading videos · {files_done}/{files_total} files · "
                    f"{bytes_done / 1e9:.1f}/{bytes_total / 1e9:.1f} GB"
                ),
            },
            force=force,
        )

    def _publish(self, fields: dict) -> None:
        self._state.update(fields, updated_at=utc_timestamp())
        _publish_state(self._root, self._target, self._state)

    def finish(self, status: str, error: str | None = None) -> None:
        """The terminal write. Unlike progress it is a source of truth and is allowed to raise."""
        self._state.update(
            status=status,
            error=error,
            updated_at=utc_timestamp(),
            eta_s=None,
            bytes_current=0,
            message="the videos are on the cluster" if status == "complete" else status,
        )
        _write_state(self._root, self._target, self._state)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3:
        raise SystemExit(
            "usage: python -m detail.executor.video_staging <experiment_root> "
            "<target_name> <recording_id> [<recording_id> ...]"
        )
    experiment_root, target_name, recording_ids = Path(argv[0]), argv[1], list(argv[2:])
    # Imported here: the shepherd imports this module to wait on it, and this one imports the
    # shepherd for the upload loop, so neither can import the other at the top.
    from detail.executor import remote_shepherd
    from detail.storage import remote_targets as remote_targets_store

    state = read_state(experiment_root, target_name) or _initial_state(
        target_name, recording_ids
    )
    state["pid"] = os.getpid()
    progress = _StagingProgress(experiment_root, target_name, state)
    try:
        target = remote_targets_store.get_remote_target(target_name)
        remote_shepherd._sync_video_roots(
            target,
            experiment_root,
            SimpleNamespace(source_recording_ids=recording_ids),
            progress,
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 — isolation boundary: recorded as failed, not raised
        traceback.print_exc()
        progress.finish("failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    progress.finish("complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
