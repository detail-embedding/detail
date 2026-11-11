"""``<run>/node_cache.zarr`` — the precomputed evidence behind a clustering run's nodes (§22).

Every array is **leaf-additive** (``analysis/node_cache.LeafTables``): a figure kept per leaf or per
leaf run, so any node is answered exactly by a sum over its leaves, and any node's clips by a pass
over its leaf runs. The layout and dtypes are tabled in ``DESIGN_PLAN.md`` §22, *The node cache*.

Written whole, staged beside the store and swapped in (the frame view's discipline, because a
workspace may hold the cache open while a rebuild replaces it), through ``zarr_store`` so every
chunk write rides out a network share's transient replace. ``schema_version`` is the first attr;
``built_from`` says what the cache was computed from (the fit, and the ``RULES`` it was computed
under), and a reader compares it rather than trusting a stale cache; ``clips`` says how the run's clips are cut and scored.

**Older caches still read.** A cache written while the cache held per-node member lists and
thumbnails (``list_*`` / ``thumb_*`` arrays, a ``min_cluster_s`` floor and a ``pictured`` attr)
reads with those ignored, and without the per-run sums its clips need (``run_rows`` …): the
reader reports it as needing a rebuild rather than guessing (``NodeView.stale_reason``).
"""

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detail.analysis.node_cache import LeafTables
from detail.core import atomic
from detail.storage import zarr_store

SCHEMA_VERSION = 1
DIRNAME = "node_cache.zarr"
BUILT_FROM_ATTR = "built_from"
#: The ``built_from`` key naming the rules the cache's figures were computed under, and the
#: current rules. Bumped when a rule changes what an existing array holds (a correction, not an
#: added column), so a cache computed under the old rule reads as stale and is rebuilt rather
#: than trusted (``NodeView.stale_reason``). A cache without the key was built under rule 1.
#:
#: - 1: the first rules.
#: - 2: a leaf run ends where its last member's covered time ends, ``start + time``. Rule 1 added
#:   the time to the member's *end*, which put a bout's run end a whole span too late (a bout
#:   method's clusters overlapped their neighbours and ran past the recording's last frame).
RULES_KEY = "rules"
RULES = 2
#: ``{"kind", "min_members", "join"}``: the method's ``ClipScoring`` without its arrays.
CLIPS_ATTR = "clips"

_RUN_CHUNK = 65_536
_LEAF_CHUNK = 4_096


@dataclass(frozen=True)
class NodeCache:
    """A whole node cache, in memory."""

    leaves: LeafTables
    built_from: dict
    #: How clips are cut and scored (``CLIPS_ATTR``); empty for a cache from before clips.
    clips: dict = dataclasses.field(default_factory=dict)


def _positions_or_nan(values, length: int) -> np.ndarray:
    """A position column as stored: ``float32``, all-NaN where the builder had none."""
    if values is None:
        return np.full(length, np.nan, dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def cache_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / DIRNAME


def has_node_cache(run_dir: Path | str) -> bool:
    return (cache_path(run_dir) / "zarr.json").exists()


def write_node_cache(run_dir: Path | str, cache: NodeCache) -> Path:
    """Write the whole cache, staged beside it and swapped in."""
    path = cache_path(run_dir)
    staged = path.with_name(path.name + ".tmp")
    shutil.rmtree(staged, ignore_errors=True)
    root = zarr_store.open_group(staged, mode="w")
    root.attrs["schema_version"] = SCHEMA_VERSION  # Tenet 7: before any array
    root.attrs[BUILT_FROM_ATTR] = dict(cache.built_from)
    root.attrs[CLIPS_ATTR] = dict(cache.clips)

    t = cache.leaves
    leaves, recordings = t.leaf_rec_time_ns.shape
    ids = root.create_array(
        "recording_ids", shape=(recordings,), chunks=(max(recordings, 1),), dtype="str"
    )
    if recordings:
        ids[:] = np.array(t.recording_ids, dtype=object)
    whole = {
        "leaf_time_ns": t.leaf_time_ns,
        "leaf_members": t.leaf_members,
        "leaf_rec_time_ns": t.leaf_rec_time_ns,
        "separation": t.separation,
    }
    for name, data in whole.items():
        root.create_array(name, data=data, chunks=tuple(max(s, 1) for s in data.shape))
    root.create_array(
        "leaf_rec_pos",
        data=t.leaf_rec_pos,
        chunks=(min(64, max(leaves, 1)), max(recordings, 1), *t.leaf_rec_pos.shape[2:]),
    )
    runs = len(t.run_leaf)
    for name in (
        "run_recording",
        "run_subject",
        "run_start_ns",
        "run_end_ns",
        "run_leaf",
        "run_cx",
        "run_cy",
    ):
        data = getattr(t, name)
        if name in ("run_cx", "run_cy"):
            data = _positions_or_nan(data, runs)
        root.create_array(name, data=data, chunks=(_RUN_CHUNK,))
    if t.run_rows is not None:
        root.create_array(
            "run_rows",
            data=np.asarray(t.run_rows, dtype=np.int32),
            chunks=(_RUN_CHUNK,),
        )
    if t.run_last_ns is not None:
        root.create_array(
            "run_last_ns",
            data=np.asarray(t.run_last_ns, dtype=np.int64),
            chunks=(_RUN_CHUNK,),
        )
    if t.run_dist_sum is not None:
        root.create_array(
            "run_dist_sum",
            data=np.asarray(t.run_dist_sum, dtype=np.float64),
            chunks=(_RUN_CHUNK,),
        )
    if t.run_vec_sum is not None:
        vec = np.asarray(t.run_vec_sum, dtype=np.float32)
        root.create_array(
            "run_vec_sum", data=vec, chunks=(_RUN_CHUNK, max(vec.shape[1], 1))
        )
        leaf_vec = np.asarray(t.leaf_vec_sum, dtype=np.float64)
        root.create_array(
            "leaf_vec_sum",
            data=leaf_vec,
            chunks=(_LEAF_CHUNK, max(leaf_vec.shape[1], 1)),
        )

    displaced = path.with_name(path.name + ".old")
    shutil.rmtree(displaced, ignore_errors=True)
    if path.exists():
        atomic.replace(path, displaced)
    atomic.replace(staged, path)
    shutil.rmtree(displaced, ignore_errors=True)
    return path


def read_node_cache(run_dir: Path | str) -> NodeCache:
    """Read the whole cache into memory.

    Raises:
        FileNotFoundError: the run has no node cache (it predates §22's, or its build failed).
        ValueError: the store was written under another schema version.
    """
    path = cache_path(run_dir)
    if not has_node_cache(run_dir):
        raise FileNotFoundError(f"no node cache at {path}")
    root = zarr_store.open_group(path, mode="r")
    version = root.attrs.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"node cache at {path} has schema_version {version!r}; this build reads "
            f"{SCHEMA_VERSION}. Rebuild the node cache."
        )

    def read(name: str) -> np.ndarray:
        return np.asarray(root[name][:])

    def optional(name: str) -> np.ndarray | None:
        """An array added after the first caches were written, or ``None`` (additive)."""
        return np.asarray(root[name][:]) if name in root else None

    def read_positions(name: str, length: int) -> np.ndarray:
        """A position column, or all-NaN for a cache written before it existed (additive)."""
        if name in root:
            return np.asarray(root[name][:], dtype=np.float32)
        return np.full(length, np.nan, dtype=np.float32)

    leaves = LeafTables(
        recording_ids=tuple(str(v) for v in read("recording_ids")),
        leaf_time_ns=read("leaf_time_ns"),
        leaf_members=read("leaf_members"),
        leaf_rec_time_ns=read("leaf_rec_time_ns"),
        leaf_rec_pos=read("leaf_rec_pos"),
        run_recording=read("run_recording"),
        run_subject=read("run_subject"),
        run_start_ns=read("run_start_ns"),
        run_end_ns=read("run_end_ns"),
        run_leaf=read("run_leaf"),
        separation=read("separation"),
    )
    runs = len(leaves.run_leaf)
    leaves = dataclasses.replace(
        leaves,
        run_cx=read_positions("run_cx", runs),
        run_cy=read_positions("run_cy", runs),
        run_rows=optional("run_rows"),
        run_last_ns=optional("run_last_ns"),
        run_dist_sum=optional("run_dist_sum"),
        run_vec_sum=optional("run_vec_sum"),
        leaf_vec_sum=optional("leaf_vec_sum"),
    )
    return NodeCache(
        leaves=leaves,
        built_from=dict(root.attrs.get(BUILT_FROM_ATTR) or {}),
        clips=dict(root.attrs.get(CLIPS_ATTR) or {}),
    )
