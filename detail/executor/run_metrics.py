"""Assemble a just-finished job's timing/compute/cost fields for its family's own run
record (§ run location, duration, cost).

``rundir.set_status`` (the single seam every job's status change funnels through)
stamps ``started_at``/``completed_at`` into ``job.json`` for free; this module is the
one place that turns those into what a run record actually stores — ``duration_s``,
and, for a remote job whose target has a configured rate, ``estimated_cost_usd``. Every
family's ``set_*_status`` (datasets/models/analysis/export/clusterers/recording) calls
:func:`completion_fields` and merges its result into the record it is about to write, so
this logic exists exactly once rather than once per family.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from detail.core import run_cost
from detail.executor import rundir
from detail.storage import remote_targets as remote_targets_store

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _parse(timestamp: str) -> datetime:
    return datetime.strptime(timestamp, _TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


def _duration_s(started_at: str | None, completed_at: str | None) -> float | None:
    if not started_at or not completed_at:
        return None
    try:
        return (_parse(completed_at) - _parse(started_at)).total_seconds()
    except ValueError:
        return None  # a malformed stamp is data, not a crash — same posture as below


def _running_s(envelope: dict, duration_s: float | None) -> float | None:
    """How long a remote job **ran** — what a GPU is billed for — or ``duration_s`` for a record
    that does not say.

    ``duration_s`` is submission to completion: it includes every minute the job sat in the
    Slurm queue, which on a busy cluster is hours of nothing. The cluster worker stamps its own
    start and finish (``rundir.set_status``), and the shepherd keeps them beside the local ones
    (``worker_started_at`` / ``worker_completed_at``). The presence of the *key* marks a record
    written with that knowledge: a job that never reached a GPU has it as ``None`` and ran for
    zero seconds, where a record from before it existed (no key) can only be billed on
    ``duration_s``, as it always was. A worker killed after it started has no finish stamp, and
    runs until the job ended here.
    """
    if "worker_started_at" not in envelope:
        return duration_s
    started = envelope.get("worker_started_at")
    if not started:
        return 0.0
    ended = envelope.get("worker_completed_at") or envelope.get("completed_at")
    return _duration_s(started, ended)


def completion_fields(run_dir: Path | str, compute: dict | None) -> dict:
    """The fields a run family's record should merge in once its job is terminal.

    ``compute`` is the job's own :class:`ComputeTarget`-shaped dict (``None`` for a
    family that never records one — always local) — echoed back into the result
    unchanged, so a caller whose record-writer overwrites the whole record mid-run
    (labeling/training's ``_finalize``) still ends up with it once this merges in,
    without needing its own separate write. Never raises: a bookkeeping read must
    not block the status write it accompanies (the isolation-boundary rule) — a run
    dir with a missing or corrupt envelope, or a remote target that has since been
    renamed/removed, yields fewer fields rather than an exception.
    """
    try:
        envelope = rundir.read_envelope(run_dir)
    except (OSError, json.JSONDecodeError):
        return {}
    started_at = envelope.get("started_at")
    completed_at = envelope.get("completed_at")
    duration_s = _duration_s(started_at, completed_at)
    out: dict = {
        "compute": compute,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_s": duration_s,
    }
    if compute and compute.get("backend") == "remote" and duration_s is not None:
        try:
            target = remote_targets_store.get_remote_target(compute["remote_target"])
        except KeyError:
            target = None
        if target is not None:
            running_s = _running_s(envelope, duration_s)
            if "worker_started_at" in envelope and running_s is not None:
                out["runtime_s"] = running_s
            if running_s is not None:
                # The GPUs the shepherd asked for (a module may size below the target); a
                # record from before it said so held the target's.
                gpus = envelope.get("remote_gpus") or target.gpus_per_job
                cost = run_cost.estimate_cost_usd(
                    gpus, running_s, target.cost_per_gpu_hour
                )
                if cost is not None:
                    out["estimated_cost_usd"] = cost
    return out
