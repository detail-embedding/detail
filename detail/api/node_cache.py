"""Build a clustering run's node cache: the evidence every node reads, computed once (§22).

The orchestration half of ``storage/node_cache.py`` (the store) and ``analysis/node_cache.py``
(the pure leaf tables). It is run at the end of a fit (the ``caching`` phase) and again by a
rebuild, and it reports progress through the ``(stage, fraction, detail)`` callback every long
operation here uses, naming each step as it goes so the bar never sits on one label.

**Every node, every clip.** There is no size floor and nothing is decoded: the cache keeps per-leaf
figures and per-leaf-run sums, from which any node's clips and their typicality are a pass over its
runs (``NodeView.clips``). The method says how clips are cut and scored (``clip_scoring``).
"""

import os
from collections.abc import Callable, Sequence
from pathlib import Path

from detail.analysis import node_cache as tables_lib
from detail.storage import experiment as experiment_store
from detail.storage import node_cache as store

Progress = Callable[[str, float, str], None] | None


def fit_digest(fit_path: Path | str) -> dict:
    """What a cache was built from: the fit shard's name, size and modification time."""
    stat = os.stat(fit_path)
    return {
        "fit": Path(fit_path).name,
        "fit_size": int(stat.st_size),
        "fit_mtime_ns": int(stat.st_mtime_ns),
    }


def build_node_cache(
    experiment_dir,
    *,
    method,
    fit,
    params: dict,
    embed_run_for: Callable[[str], str],
    built_from: dict,
    trajectories: Sequence | None = None,
    progress: Progress = None,
) -> store.NodeCache:
    """Compute a whole node cache for ``fit``.

    ``params`` are the run's method params and ``trajectories`` the fit's own inputs, both handed
    to ``method.clip_scoring``; a method whose typicality needs every member's vector (pose)
    needs the trajectories, and a caller that has none loads them first.
    """

    def report(detail: str, fraction: float) -> None:
        if progress is not None:
            progress("caching", float(min(max(fraction, 0.0), 1.0)), detail)

    tree = method.hierarchy_of(fit)
    report("member time", 0.0)
    places = method.member_places(fit)
    leaves = method.member_leaves(fit)
    report("positions", 0.05)
    kinematics = _kinematics(
        experiment_dir,
        places,
        embed_run_for,
        lambda done, total: report(
            f"positions ({done:,} of {total:,} animals)", 0.05 + 0.45 * done / total
        ),
    )
    positions = tables_lib.member_positions(places, kinematics) if kinematics else None
    report("split separations", 0.5)
    separations = method.node_separations(fit)
    report("clips and typicality", 0.7)
    scoring = method.clip_scoring(fit, dict(params or {}), trajectories)
    tables = tables_lib.leaf_tables(
        places,
        leaves,
        tree.n_leaves,
        separations,
        positions=positions,
        scoring=scoring,
    )
    report("done", 1.0)
    return store.NodeCache(
        leaves=tables,
        built_from={**dict(built_from), store.RULES_KEY: store.RULES},
        clips={
            "kind": scoring.kind,
            "min_members": int(scoring.min_members),
            "join": bool(scoring.join),
        },
    )


def _kinematics(experiment_dir, places, embed_run_for, step=None) -> dict:
    """``{(recording_id, subject): {t_ns, cx_norm, cy_norm}}`` for every unit that has them.

    Empty when no recording has kinematics at all; the build then stores no positions.
    """
    from detail.api.run_preview import load_track_kinematics

    units = set(
        zip(
            (places.recording_ids[int(r)] for r in places.recording_index),
            (int(s) for s in places.subject),
        )
    )
    kinematics: dict[tuple[str, int], dict] = {}
    units = sorted(units)
    for done, (recording_id, subject) in enumerate(units):
        if step is not None:
            step(done, len(units))
        recording_dir = experiment_store.recording_path(experiment_dir, recording_id)
        found = load_track_kinematics(
            recording_dir, embed_run_for(recording_id), subject
        )
        if found is not None:
            kinematics[(recording_id, subject)] = found
    return kinematics


def write_for_run(
    experiment_dir,
    run_dir,
    *,
    method,
    fit,
    fit_path,
    params: dict,
    embed_run_for: Callable[[str], str],
    trajectories: Sequence | None = None,
    progress: Progress = None,
) -> Path:
    """Build a run's node cache and write it."""
    cache = build_node_cache(
        experiment_dir,
        method=method,
        fit=fit,
        params=params,
        embed_run_for=embed_run_for,
        built_from=fit_digest(fit_path),
        trajectories=trajectories,
        progress=progress,
    )
    return store.write_node_cache(run_dir, cache)
