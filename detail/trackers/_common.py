"""Shared pure primitives + resume plumbing for the trackers (Section 13).

Every registered tracker (``anchor_tracker`` and ``iou_single_tracker``) links/gates detections by the same bbox
geometry, frame grouping, arena test, contiguity check, and hyperparam
validation, and they all resume at part boundaries the same way. Those live here
once so a fix or a tighter check lands in every tracker at the same time (e.g.
the bool-vs-int hyperparam guard).

Pure compute over the detection columns (Tenet 7); the arena test and contiguity
gate use cv2. The resume helpers own the ``part_handoff.json`` format shared by
every tracker (Section 7): the tracker supplies only its own opaque ``state``.
"""

import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml

from detail.core import atomic
from detail.core.utils import bbox_iou, polygon_at_frame

Bbox = tuple[float, float, float, float]
SUBJECT_ID = 0  # single-subject trackers: the one track is always object_id 0

SCHEMA_VERSION = 1
HANDOFF_FILENAME = "part_handoff.json"


def iou(a: Bbox, b: Bbox) -> float:
    """IoU of two ``(x1, y1, x2, y2)`` boxes; 0 when they don't overlap."""
    return bbox_iou(a, b)


def group_by_frame(
    frame_idx: np.ndarray, every_n_frames: int = 1
) -> dict[int, list[int]]:
    """``{frame_value: row_indices}`` in frame order (rows are written in frame
    order, so the dict's insertion order is ascending frame).

    ``every_n_frames`` thins to this run's cadence, keeping frames ``0, N, 2N, …``. The
    predicate is on the *source* frame index, the same rule every stage uses, so it
    composes with whatever stride detection already ran at (§4).
    """
    groups: dict[int, list[int]] = {}
    for i, f in enumerate(frame_idx):
        frame = int(f)
        if frame % every_n_frames:
            continue
        groups.setdefault(frame, []).append(i)
    return groups


def frame_walk(groups: dict[int, list[int]], every_n_frames: int = 1) -> range:
    """The frames a tracker steps through: every frame **this run processed**, including
    those with no detections (a track must age while its subject is absent, §9).

    Stepping by ``every_n_frames`` is what keeps that true at any cadence. Walking every
    integer instead would count frames the run never looked at, so a ``max_missed`` of 5
    would expire after 5 *source* frames — at stride 8, before a single processed frame
    had passed, killing every track.
    """
    if not groups:
        return range(0)
    return range(min(groups), max(groups) + 1, max(1, every_n_frames))


def processed_frames_between(
    first: int,
    last: int,
    every_n_frames: int,
    part_frame_offsets: Sequence[int] | None,
) -> int:
    """How many frames a run at ``every_n_frames`` processed in the global span
    ``[first, last]``, both ends included.

    A part's processed frames are its own ``0, N, 2N, ...`` (§4), so each part restarts the
    grid at its offset: every Nth *global* frame is wrong as soon as a part's length is not a
    multiple of N. ``part_frame_offsets`` is each part's first global frame, as
    ``pipeline/tracking.py`` passes it; ``None`` (a direct caller without offsets) reads the
    span as one part. This is what a track's ``n_gaps`` is measured against (#32): a frame the
    run never looked at was not missed.
    """
    step = max(1, int(every_n_frames))
    first, last = int(first), int(last)
    if last < first:
        return 0
    offsets = [int(o) for o in part_frame_offsets] if part_frame_offsets else [0]
    total = 0
    for index, start in enumerate(offsets):
        end = offsets[index + 1] - 1 if index + 1 < len(offsets) else last
        lo, hi = max(first, start) - start, min(last, end) - start
        if lo > hi:
            continue
        total += max(hi // step - (-(-lo // step)) + 1, 0)
    return total


def filter_by_arena(rows, bbox, arena_polygon, global_frame: int) -> list[int]:
    """Keep rows whose normalized bbox center lies inside-or-on the polygon at
    this frame (the arena is interpolated per frame, Section 10)."""
    verts = polygon_at_frame(arena_polygon, global_frame).astype(np.float32)
    kept = []
    for i in rows:
        x1, y1, x2, y2 = (float(v) for v in bbox[i])
        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        if cv2.pointPolygonTest(verts, center, False) >= 0:
            kept.append(i)
    return kept


def is_contiguous(mask: np.ndarray) -> bool:
    """True iff the binary mask is a single connected component (an empty mask is
    not contiguous). ``connectedComponents`` counts the background as label 0, so
    exactly one foreground component yields 2 labels."""
    n_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
    return n_labels == 2


def zero_tracks_error(n_detections: int, params: dict) -> ValueError:
    """The error for a tracking run that produced no track, naming the real cause.

    With no detections at all, the tracker's settings are not the cause, and saying they "may be
    too strict" sent the user to tune the wrong stage: the detection run is where to look. Every
    tracker raises this one message (#33).
    """
    if n_detections == 0:
        return ValueError(
            "zero tracks produced: the detection run found no detections on this recording, "
            "so there was nothing to track. Run detection again with a lower --conf or a "
            "different prompt, then track"
        )
    return ValueError(
        f"zero tracks produced from {n_detections} detection(s); the tracker's settings may "
        f"be too strict (conf_threshold={params['conf_threshold']}, "
        f"require_contiguous={params['require_contiguous']})"
    )


def validate_hyperparams(schema: dict, hyperparams: dict) -> dict:
    """Fill defaults, reject unknown/wrong-type keys, enforce the secondary floor.

    ``schema`` maps key -> ``(type, default)`` with ``type`` in ``{bool, int,
    float}``. ``bool`` is an ``int`` subclass, so it is checked *before* the
    int->float coercion — otherwise a ``True`` typo silently becomes ``1.0``. A
    fragment floor above the subject floor leaves the secondary tier empty, which
    is almost always a mistake, so fail loud (``secondary_conf == conf_threshold``
    is allowed and simply disables merging). Schemas without those keys skip that
    check.
    """
    unknown = set(hyperparams) - set(schema)
    if unknown:
        raise ValueError(
            f"unknown hyperparams: {sorted(unknown)}; allowed: {sorted(schema)}"
        )
    result: dict = {}
    for key, (typ, default) in schema.items():
        if key not in hyperparams:
            result[key] = default
            continue
        value = hyperparams[key]
        if typ is bool:
            if not isinstance(value, bool):
                raise ValueError(
                    f"hyperparam {key!r} must be bool, got {type(value).__name__}"
                )
        elif typ is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if typ is not bool and not isinstance(value, typ):
            raise ValueError(
                f"hyperparam {key!r} must be {typ.__name__}, "
                f"got {type(value).__name__}"
            )
        result[key] = value
    if result.get("secondary_conf", 0.0) > result.get("conf_threshold", 1.0):
        raise ValueError(
            "secondary_conf must be <= conf_threshold (got "
            f"secondary_conf={result['secondary_conf']}, "
            f"conf_threshold={result['conf_threshold']})"
        )
    return result


# --------------------------------------------------------------------------- #
# Resume: part-boundary handoff (Section 7)
# --------------------------------------------------------------------------- #
def read_handoff(output_dir: Path | str) -> tuple[int, dict | None]:
    """``(parts_completed, state)`` from ``part_handoff.json``, or ``(0, None)``
    if there is none — the fresh-run case. ``state`` is the tracker's own opaque
    cross-part snapshot as of the last committed part boundary."""
    path = Path(output_dir) / HANDOFF_FILENAME
    if not path.exists():
        return 0, None
    with open(path, encoding="utf-8-sig") as f:
        envelope = json.load(f)
    return int(envelope["parts_completed"]), envelope.get("state")


def write_handoff(output_dir: Path | str, parts_completed: int, state: dict) -> None:
    """Atomically commit the part boundary: ``schema_version`` first, then the
    completed-part count and the tracker's opaque ``state`` (Tenets 6/7 — no
    absolute paths; the tracker must keep paths out of ``state``). The handoff
    write is the commit point, so it always follows the part's assignment write."""
    path = Path(output_dir) / HANDOFF_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "parts_completed": int(parts_completed),
                "state": state,
            },
            f,
        )
    atomic.replace(tmp, path)


def write_track_yaml(path: Path | str, data: dict) -> None:
    """Write a tracking run's ``track.yaml`` atomically (tmp + ``atomic.replace``).

    Every tracker writes this once, at run completion, from a subprocess the GUI polls
    for "has this run finished" — so a plain truncating write lets a concurrent reader
    observe a torn file (the atomic-write hard rule, CLAUDE.md). Lifted here, like
    ``write_handoff`` above, so the three trackers' identical write step can't drift the
    way the surrounding record-building already has.
    """
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
    atomic.replace(tmp, path)


def clear_zarr(path: Path | str) -> None:
    """Remove a zarr store if present, so a reprocessed part or a re-emitted
    summary is written fresh — the part writers open ``mode="a"`` and would
    otherwise append duplicate rows onto a stale store (Tenet 12 resume)."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
