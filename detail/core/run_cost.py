"""Where a run executed, and (for a remote one) what it cost — §6/§12.

Pure formatting/arithmetic, no I/O, so the CLI, the GUI, and the executor layer's own
record-writers (`executor/run_metrics.py`, which *does* do I/O) can all share the one
implementation, instead of a hand-rolled label string in each — the class of bug this
repo's own history warns about (four independent `created_at=""` copies fixed
separately, per `docs/recorded_classes.md`).
"""


def estimate_cost_usd(
    gpus_per_job: int, elapsed_s: float, cost_per_gpu_hour: float | None
) -> float | None:
    """``gpus_per_job`` GPUs for ``elapsed_s`` seconds at ``cost_per_gpu_hour`` — or
    ``None`` when no rate is configured for the target ("no rate set = no cost
    shown", decided with the user rather than guessing a number)."""
    if cost_per_gpu_hour is None:
        return None
    return gpus_per_job * (elapsed_s / 3600.0) * cost_per_gpu_hour


def compute_label(compute: dict | None) -> str:
    """ "local", "local (cuda)", or "remote: tillicum" — the one label every GUI
    surface renders for "where did this run".

    ``compute`` is a :class:`ComputeTarget`-shaped mapping (``to_dict()``'s shape),
    or ``None`` for a run family that never records one at all (tracking, analysis,
    export, clusterer today) — which always means local, since none of those
    families has a remote-capable backend yet.
    """
    backend = (compute or {}).get("backend")
    if backend == "remote":
        return f"remote: {compute.get('remote_target') or '?'}"
    return f"local ({backend})" if backend else "local"
