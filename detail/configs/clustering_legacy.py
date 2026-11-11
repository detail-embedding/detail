"""Keys a clustering run once recorded and no longer means anything by (§22).

The node cache used to picture only nodes over a size floor: ``min_cluster_s`` in a run's launch
config and params, ``cache_floor_s`` in its state. The floor is gone (every node lists every
clip), but the runs written with it still open, and a run started from one still launches. This
is the one place those keys are dropped on read, so a legacy value never reaches a job, a form,
a state written back or a run started from it.
"""

#: Every key the size floor left behind, wherever it was written.
LEGACY_FLOOR_KEYS = ("min_cluster_s", "cache_floor_s")


def drop_legacy_floor(mapping: dict | None) -> dict:
    """``mapping`` without the size floor's keys (a copy; ``None`` reads as empty)."""
    return {k: v for k, v in (mapping or {}).items() if k not in LEGACY_FLOOR_KEYS}
