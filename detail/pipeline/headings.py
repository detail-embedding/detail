"""Camera-relative headings for the ``heading`` example-frame sampler (§26 *Headings*).

Each tracked frame's heading relative to the camera, from the frames alone — no clicks, no labels:
the third version a study tried. Three readings, then a fill-in:

1. **Walking.** While the animal walks steadily its direction of travel is its heading. Sideways
   travel is read in image pixels against the body's size; travel toward or away from the camera
   needs the floor ellipse and counts only when the tail-free body grows or shrinks the same way.
   The body's shape must agree (side-on for sideways, end-on for toward/away).
2. **Tail.** The head points away from the tail base: its side gives left/right, its height
   front/back — the latter only for a leftover that reaches well past the body (paws do not).
   A walking heading the tail contradicts is dropped, and so is toward/away walking the tail does
   not confirm: walking away and turning away look alike.
3. **Rearing**: a mask much taller than the recording's walking median.
4. **Similar frames** fill in the rest at sampling time, from the caches of *other* recordings
   (:func:`heading_pool`) — which is why nothing that depends on another recording is cached.

The per-recording cache (``recordings/<id>/headings/<tracking_run>/``) is written by
:func:`compute_headings` and read by :func:`load_headings`; its layout is the spec's.

``torch`` is imported inside the functions that embed (Tenet 13).
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from detail.core import atomic
from detail.core import yaml_load
from detail.core.examples import HEADING_BINS, HEADING_BUCKETS
from detail.core.utils import decode_mask
from detail.pipeline import subject_frames
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage import zarr_store
from detail.storage.zarr_io import DetectionPartReader

SCHEMA_VERSION = 1
METHOD_VERSION = 3
HEADINGS_DIR = "headings"
META_FILENAME = "headings.yaml"
ROW_CHUNK, FEATURE_CHUNK = 10000, 1000

# The study's constants (its third version), recorded on every cache (Tenet 9).
CROP_PX, POOL, PROJ_DIM, PROJ_SEED = 224, 4, 64, 20261019
EMBED_EVERY, EMBED_BATCH = 2, 32
FOCAL_FRAC = (
    0.8  # assumed focal length as a fraction of image width: the bearing correction
)
SMOOTH_SIGMA, LATERAL_HALF, RADIAL_HALF = 2.0, 3, 6  # frames
SPEED_MIN = 0.35  # floor semi-axes per second
DX_MIN_BODY, DY_MIN_BODY, DLOGA_MIN = 0.15, 0.06, 0.05
SIDE_ON, END_ON = (
    0.35,
    0.45,
)  # side-on-ness (elongation x |cos axis|) a walking heading needs
TAIL_SIDE_ON, TAIL_END_ON = 0.35, 0.15
TAIL_DX_MIN, TAIL_DY_MIN, TAIL_MIN_FRAC, TAIL_REACH_MIN = 0.25, 0.12, 0.03, 0.6
REAR_RATIO = 1.45
HANG_TOP, HANG_RATIO = (
    0.2,
    2.6,
)  # held by the tail: box top in the frame's top fifth, or this tall
AREA_RANGE = subject_frames.AREA_RANGE
FLOOR_ASPECT_TOLERANCE = 0.20
KNN_K, KNN_TAU, KNN_MIN_AGREEMENT = 20, 0.02, 0.55

CONSTANTS = {
    "crop_px": CROP_PX,
    "pool": POOL,
    "embed_every": EMBED_EVERY,
    "focal_frac": FOCAL_FRAC,
    "smooth_sigma": SMOOTH_SIGMA,
    "lateral_half": LATERAL_HALF,
    "radial_half": RADIAL_HALF,
    "speed_min": SPEED_MIN,
    "dx_min_body": DX_MIN_BODY,
    "dy_min_body": DY_MIN_BODY,
    "dloga_min": DLOGA_MIN,
    "side_on": SIDE_ON,
    "end_on": END_ON,
    "tail_side_on": TAIL_SIDE_ON,
    "tail_end_on": TAIL_END_ON,
    "tail_dx_min": TAIL_DX_MIN,
    "tail_dy_min": TAIL_DY_MIN,
    "tail_min_frac": TAIL_MIN_FRAC,
    "tail_reach_min": TAIL_REACH_MIN,
    "rear_ratio": REAR_RATIO,
    "hang_top": HANG_TOP,
    "hang_ratio": HANG_RATIO,
    "area_range": list(AREA_RANGE),
}

# The source of a frame's heading, best first: what a bucket's candidates are ranked by.
SOURCES = ("walking+tail", "walking", "tail", "similar")


def _report(progress, stage: str, fraction: float, detail: str) -> None:
    if progress is not None:
        progress(stage, fraction, detail)


def angle_diff(a, b):
    """``a - b`` in degrees, wrapped to [-180, 180)."""
    return (
        np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0
    ) % 360.0 - 180.0


def bin_of(phi):
    """The heading bin index (``HEADING_BINS``) of a heading in degrees."""
    return np.round(np.asarray(phi, dtype=float) / 45.0).astype(int) % 8


def _smooth(x: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    radius = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    return np.convolve(np.pad(x, radius, mode="edge"), k / k.sum(), mode="valid")


def _runs(frames: np.ndarray, min_len: int) -> list[np.ndarray]:
    """Index runs of one track's rows with no gap over two frames, at least ``min_len`` long."""
    breaks = np.flatnonzero(np.diff(frames) > 2) + 1
    return [r for r in np.split(np.arange(len(frames)), breaks) if len(r) >= min_len]


# --------------------------------------------------------------------------- #
# The floor
# --------------------------------------------------------------------------- #
def floor_fit(
    recording_dir: Path, floor_class: str, prefer: str | None = None
) -> dict | None:
    """``{cx, cy, a, b, detection_run_id}`` in frame pixels: the floor class's masks, voted per
    pixel over up to 60 frames, fitted on the far rim (the top boundary) so a floor's front face
    does not inflate it. ``prefer`` is the detection run to try first. None when no run has it.
    """
    import cv2

    detections = Path(recording_dir) / "detections"
    runs = sorted(p.name for p in detections.iterdir()) if detections.is_dir() else []
    if prefer in runs:
        runs.remove(prefer)
        runs.insert(0, prefer)
    data = recording_store.read_recording(recording_dir)
    width, height = (int(v) for v in data["video"]["resolution"])
    for run_id in runs:
        part = detections / run_id / "part_001.zarr"
        size = subject_frames._mask_size(detections / run_id)
        if not part.is_dir() or size is None:
            continue
        reader = DetectionPartReader(part)
        if "class_name" not in reader.column_names:
            continue
        names = np.array([str(c) for c in reader.column("class_name")[:]])
        rows = np.flatnonzero(names == floor_class)
        if rows.size < 10:
            continue
        conf = np.asarray(reader.column("confidence")[:])
        frame = np.asarray(reader.column("frame_idx")[:])
        best: dict[int, int] = {}
        for r in rows:
            if int(frame[r]) not in best or conf[r] > conf[best[int(frame[r])]]:
                best[int(frame[r])] = int(r)
        chosen = sorted(best.values())
        take = np.linspace(0, len(chosen) - 1, min(60, len(chosen))).astype(int)
        rle = reader.column("mask_rle")
        vote = np.zeros(size, dtype=np.float32)
        for i in take:
            vote += decode_mask(str(rle[chosen[i]]), size)
        floor = vote >= 0.5 * len(take)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(floor.astype(np.uint8))
        if n < 2:
            continue
        floor = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cols = np.flatnonzero(floor.any(axis=0))
        cx, a = (cols[0] + cols[-1]) / 2.0, (cols[-1] - cols[0]) / 2.0
        xs = np.arange(int(cols[0]), int(cols[-1]) + 1)
        top = np.array([np.argmax(floor[:, x]) for x in xs], dtype=float)
        t = (xs - cx) / a
        use = np.abs(t) < 0.85
        s = np.sqrt(np.clip(1 - t[use] ** 2, 0, None))
        (cy, b), *_ = np.linalg.lstsq(
            np.stack([np.ones_like(s), -s], 1), top[use], rcond=None
        )
        sx, sy = width / size[1], height / size[0]
        return {
            "cx": float(cx * sx),
            "cy": float(cy * sy),
            "a": float(a * sx),
            "b": float(b * sy),
            "detection_run_id": run_id,
        }
    return None


def resolve_floors(
    experiment_dir: Path, sources: dict[str, str], floor_class: str, groups: dict | None
) -> dict[str, dict | None]:
    """Each recording's floor: its own fit, or its group's median when the fit is missing or
    its aspect is more than ``FLOOR_ASPECT_TOLERANCE`` from the group's; None when neither.
    """
    fits = {}
    for recording_id, tracking_run_id in sources.items():
        rec_dir = experiment_store.recording_path(experiment_dir, recording_id)
        meta = yaml_load.safe_load(
            atomic.read_text(rec_dir / "tracking" / tracking_run_id / "track.yaml")
        )
        fits[recording_id] = floor_fit(
            rec_dir, floor_class, meta.get("detection_run_id")
        )
    groups = groups or {}
    by_group: dict[str, list[dict]] = {}
    for recording_id, fit in fits.items():
        if fit is not None:
            by_group.setdefault(groups.get(recording_id, ""), []).append(fit)
    medians = {
        g: {k: float(np.median([f[k] for f in fs])) for k in ("cx", "cy", "a", "b")}
        for g, fs in by_group.items()
        if len(fs) >= 3
    }
    out = {}
    for recording_id, fit in fits.items():
        group = groups.get(recording_id, "")
        median = medians.get(group)
        if fit is not None and (
            median is None
            or abs((fit["b"] / fit["a"]) / (median["b"] / median["a"]) - 1)
            <= FLOOR_ASPECT_TOLERANCE
        ):
            out[recording_id] = {**fit, "class": floor_class, "source": "own"}
        elif median is not None:
            out[recording_id] = {
                **median,
                "class": floor_class,
                "detection_run_id": None,
                "source": f"group_median:{group}",
            }
        else:
            out[recording_id] = None
    if fits and all(fit is None for fit in fits.values()):
        raise ValueError(
            f"no chosen recording has a {floor_class!r} detection to fit the floor to; "
            "name the detection class that is the floor, or leave floor_class out"
        )
    return out


# --------------------------------------------------------------------------- #
# Body and tail
# --------------------------------------------------------------------------- #
def body_and_tail(mask: np.ndarray, bbox_px) -> dict:
    """The tail-free body's shape (``e`` elongation, ``axis`` deg, ``area``, centre) and the
    tail's base offset from the body's centre, its pixels and its reach in body sizes.
    """
    import cv2

    h, w = mask.shape
    x0, y0, x1, y1 = (int(v) for v in bbox_px)
    x0, y0, x1, y1 = max(x0 - 6, 0), max(y0 - 6, 0), min(x1 + 6, w), min(y1 + 6, h)
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    out = {
        "e": np.nan,
        "axis": np.nan,
        "area": float(m.sum()),
        "tail_dx": np.nan,
        "tail_dy": np.nan,
        "tail_px": 0.0,
        "tail_reach": np.nan,
    }
    if out["area"] < 50:
        return out
    r = max(3, int(0.09 * math.sqrt(out["area"])))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    body = cv2.morphologyEx(m, cv2.MORPH_OPEN, disc)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(body)
    if n < 2:
        return out
    body = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    mo = cv2.moments(body, binaryImage=True)
    area = float(mo["m00"])
    cov = np.array([[mo["mu20"], mo["mu11"]], [mo["mu11"], mo["mu02"]]]) / area
    lo, hi = np.linalg.eigvalsh(cov)
    cx, cy = mo["m10"] / area, mo["m01"] / area
    out.update(
        e=1.0 - math.sqrt(max(lo, 0.0) / hi) if hi > 0 else 0.0,
        axis=0.5 * math.degrees(math.atan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"])),
        area=area,
    )
    rest = (m > 0) & (cv2.dilate(body, np.ones((3, 3), np.uint8)) == 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8))
    if n < 2:
        return out
    tail = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    dist = cv2.distanceTransform((1 - body).astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.nonzero(tail)
    near = dist[ys, xs] <= 0.35 * math.sqrt(area)
    if tail.sum() < 0.01 * area or near.sum() < 5:
        return out
    out.update(
        tail_dx=float(xs[near].mean() - cx),
        tail_dy=float(ys[near].mean() - cy),
        tail_px=float(tail.sum()),
        tail_reach=float(dist[ys, xs].max() / math.sqrt(area)),
    )
    return out


# --------------------------------------------------------------------------- #
# One track's readings
# --------------------------------------------------------------------------- #
@dataclass
class TrackRows:
    """One track's per-frame measurements, in frame order, in frame pixels."""

    frame_idx: np.ndarray
    x: np.ndarray  # mask centroid
    y_floor: np.ndarray  # 90th-percentile mask row: the floor point
    height: np.ndarray
    area: np.ndarray  # whole mask
    top: np.ndarray  # box top, normalized
    body: list[dict]  # body_and_tail per frame


def walking_headings(
    rows: TrackRows, floor: dict | None, fps: float, width: int, pickable: np.ndarray
) -> dict[int, tuple[float, str]]:
    """``{row: (heading deg, "lateral" | "radial" | "diag")}`` for steady walking."""
    frames = rows.frame_idx
    beta = np.degrees(np.arctan((rows.x - width / 2) / (FOCAL_FRAC * width)))
    area = np.array([b["area"] for b in rows.body], dtype=float)
    side_on = np.array(
        [b["e"] * abs(math.cos(math.radians(b["axis"]))) for b in rows.body]
    )
    a_px = floor["a"] if floor else None
    b_px = floor["b"] if floor else None
    out = {}
    for run in _runs(frames, 2 * RADIAL_HALF + 3):
        good = np.where(area[run] > 0, area[run], np.nanmedian(area[run]))
        xs, ys, la = (
            _smooth(rows.x[run]),
            _smooth(rows.y_floor[run]),
            _smooth(np.log(good)),
        )
        for j in range(RADIAL_HALF, len(run) - RADIAL_HALF):
            i = run[j]
            if not pickable[i] or not area[i] > 0:
                continue
            body = math.sqrt(area[i])
            dx = xs[j + LATERAL_HALF] - xs[j - LATERAL_HALF]
            dt_l = (frames[run[j + LATERAL_HALF]] - frames[run[j - LATERAL_HALF]]) / fps
            dy, dla = (
                ys[j + RADIAL_HALF] - ys[j - RADIAL_HALF],
                la[j + RADIAL_HALF] - la[j - RADIAL_HALF],
            )
            dt_r = (frames[run[j + RADIAL_HALF]] - frames[run[j - RADIAL_HALF]]) / fps
            radial = (
                floor is not None
                and abs(dy) >= DY_MIN_BODY * body
                and abs(dla) >= DLOGA_MIN
                and np.sign(dy) == np.sign(dla)
            )
            lateral = abs(dx) >= DX_MIN_BODY * body
            if not (lateral or radial):
                continue
            if floor is not None:
                du, dv = dx / a_px / dt_l, (dy / b_px / dt_r) if radial else 0.0
                if math.hypot(du, dv) < SPEED_MIN:
                    continue
            else:
                du, dv = dx, 0.0
            theta = math.degrees(math.atan2(dv, du if lateral else 0.0))
            phi = float(angle_diff(theta, beta[i]))
            c, s = abs(math.cos(math.radians(phi))), side_on[i]
            if (c >= 0.85 and not s >= SIDE_ON) or (c <= 0.4 and not s <= END_ON):
                continue
            out[i] = (
                phi,
                "lateral" if c >= 0.85 else ("radial" if c <= 0.4 else "diag"),
            )
    return out


def tail_headings(rows: TrackRows, pickable: np.ndarray) -> dict[int, float]:
    """``{row: heading deg}`` from the tail base, as the module docstring says."""
    found = []
    for i, b in enumerate(rows.body):
        if not pickable[i] or not math.isfinite(b["tail_dx"]) or b["area"] <= 0:
            continue
        if b["tail_px"] < TAIL_MIN_FRAC * b["area"]:
            continue
        body = math.sqrt(b["area"])
        s = b["e"] * abs(math.cos(math.radians(b["axis"])))
        found.append((i, b["tail_dx"] / body, b["tail_dy"] / body, s, b["tail_reach"]))
    # the rump sits below a side-on body's centre whatever its facing: take that offset off
    side = [
        dy
        for _i, dx, dy, s, _r in found
        if s >= TAIL_SIDE_ON and abs(dx) >= TAIL_DX_MIN
    ]
    bias = float(np.median(side)) if len(side) >= 10 else 0.0
    out = {}
    for i, dx, dy, s, reach in found:
        dy -= bias
        if s >= TAIL_SIDE_ON:
            if abs(dx) >= TAIL_DX_MIN:
                out[i] = 180.0 if dx > 0 else 0.0
        elif not reach >= TAIL_REACH_MIN:
            continue
        elif s <= TAIL_END_ON:
            if abs(dy) >= TAIL_DY_MIN:
                out[i] = 90.0 if dy < 0 else -90.0
        elif abs(dx) >= TAIL_DX_MIN and abs(dy) >= TAIL_DY_MIN:
            out[i] = math.degrees(
                math.atan2(1.0 if dy < 0 else -1.0, -1.0 if dx > 0 else 1.0)
            )
    return out


def fuse(walking: dict, tails: dict) -> tuple[dict[int, float], set[int]]:
    """Walking headings the tail does not contradict, and the rows where it did."""
    kept, conflicts = {}, set()
    for i, (phi, kind) in walking.items():
        t = tails.get(i)
        if t is not None:
            if abs(float(angle_diff(phi, t))) < 90.0:
                kept[i] = phi
            else:
                conflicts.add(i)
        elif kind != "radial":
            kept[i] = phi
    return kept, conflicts


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #
def headings_dir(recording_dir: Path, tracking_run_id: str) -> Path:
    return Path(recording_dir) / HEADINGS_DIR / tracking_run_id


def cache_matches(
    recording_dir: Path, tracking_run_id: str, floor: dict | None
) -> bool:
    """Whether the cache on disk was written by this method version, constants and floor."""
    meta_path = headings_dir(recording_dir, tracking_run_id) / META_FILENAME
    if not meta_path.is_file():
        return False
    meta = yaml_load.safe_load(atomic.read_text(meta_path))
    return (
        meta.get("method") == {"version": METHOD_VERSION, "constants": CONSTANTS}
        and meta.get("floor") == floor
    )


def _projection(channels: int) -> np.ndarray:
    """The fixed seeded Gaussian projection (no data seen), so a cache needs no other
    recording: ``(PROJ_DIM, channels)``."""
    rng = np.random.default_rng(PROJ_SEED)
    return rng.standard_normal((PROJ_DIM, channels)).astype(np.float32) / math.sqrt(
        channels
    )


def _embed(model, crops: list[np.ndarray]) -> np.ndarray:
    """``(N, POOL*POOL*PROJ_DIM)`` float16: 4x4-pooled patch features, projected per cell and
    each cell L2-normalised."""
    import torch
    import torch.nn.functional as F

    out, proj = [], None
    for start in range(0, len(crops), EMBED_BATCH):
        grids = model.patch_grids(crops[start : start + EMBED_BATCH])
        pooled = F.adaptive_avg_pool2d(torch.stack([g.float() for g in grids]), POOL)
        cells = pooled.flatten(2).transpose(1, 2).cpu().numpy()  # (B, cells, C)
        if proj is None:
            proj = _projection(cells.shape[2])
        z = cells @ proj.T
        z /= np.linalg.norm(z, axis=2, keepdims=True) + 1e-8
        out.append(z.reshape(len(z), -1).astype(np.float16))
    return (
        np.concatenate(out)
        if out
        else np.zeros((0, POOL * POOL * PROJ_DIM), np.float16)
    )


def compute_headings(
    experiment_dir: Path | str,
    recording_id: str,
    tracking_run_id: str,
    floor: dict | None,
    model,
    progress=None,
) -> Path:
    """Write one recording's headings cache and return its directory.

    ``model`` is a loaded DINOv3 ViT (``patch_grids``). ``progress(stage, fraction, detail)``
    hears every phase — reading masks, decoding, embedding — as it advances (§6).

    Raises:
        FileNotFoundError: The recording's video is not on this machine.
    """
    import cv2

    experiment_dir = Path(experiment_dir)
    rec_dir = experiment_store.recording_path(experiment_dir, recording_id)
    data = recording_store.read_recording(rec_dir)
    width, height = (int(v) for v in data["video"]["resolution"])
    fps = float(data["video"]["fps_actual"])
    rows, mask_size = subject_frames._tracked_rows(rec_dir, tracking_run_id, None)
    if mask_size is None:
        raise ValueError(
            f"{recording_id}: tracking run {tracking_run_id!r} has no masks; the heading "
            "sampler reads each subject's mask"
        )
    subject_frames._require_videos(rec_dir, {r["part_id"] for r in rows})
    subject_frames._prefetch_masks(rows)
    sx, sy = width / mask_size[1], height / mask_size[0]
    by_track: dict[tuple[str, int], list[dict]] = {}
    for n, row in enumerate(rows):
        if n % 200 == 0:
            _report(
                progress,
                "reading masks",
                n / max(len(rows), 1),
                f"{n}/{len(rows)} frames",
            )
        mask = subject_frames.subject_mask(row["det"], row["rows"], mask_size)
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            continue
        x0, y0, x1, y1 = row["bbox"]
        bb = (
            x0 * mask_size[1],
            y0 * mask_size[0],
            x1 * mask_size[1],
            y1 * mask_size[0],
        )
        body = body_and_tail(mask, bb)
        for key in ("area", "tail_px"):
            body[key] *= sx * sy
        body["tail_dx"] *= sx
        body["tail_dy"] *= sy
        by_track.setdefault((row["part_id"], int(row["track_id"])), []).append(
            {
                **row,
                "x": float(xs.mean()) * sx,
                "y_floor": float(np.percentile(ys, 90)) * sy,
                "height": float(np.percentile(ys, 98) - np.percentile(ys, 2)) * sy,
                "mask_area": float(len(xs)) * sx * sy,
                "body": body,
            }
        )
    _report(progress, "reading masks", 1.0, f"{len(rows)}/{len(rows)} frames")

    table: dict[str, list] = {
        k: []
        for k in (
            "part_id",
            "frame_idx",
            "t_ns",
            "track_id",
            "walking_deg",
            "tail_deg",
            "conflict",
            "rearing",
            "pickable",
            "bbox",
        )
    }
    for (part_id, track_id), items in sorted(by_track.items()):
        items.sort(key=lambda r: r["frame_idx"])
        tr = TrackRows(
            frame_idx=np.array([r["frame_idx"] for r in items]),
            x=np.array([r["x"] for r in items]),
            y_floor=np.array([r["y_floor"] for r in items]),
            height=np.array([r["height"] for r in items]),
            area=np.array([r["mask_area"] for r in items]),
            top=np.array([r["bbox"][1] for r in items]),
            body=[r["body"] for r in items],
        )
        med_h, med_area = float(np.median(tr.height)), float(np.median(tr.area))
        on_floor = np.ones(len(items), bool)
        if floor is not None:
            u = (tr.x - floor["cx"]) / floor["a"]
            v = (tr.y_floor - floor["cy"]) / floor["b"]
            on_floor = u * u + v * v <= 1.3
        hanging = (tr.top < HANG_TOP) | (tr.height / med_h > HANG_RATIO)
        in_view = np.array(
            [
                min(r["bbox"][:2]) >= subject_frames.EDGE_MARGIN
                and max(r["bbox"][2:]) <= 1 - subject_frames.EDGE_MARGIN
                for r in items
            ]
        )
        ratio = tr.area / med_area
        pickable = (
            on_floor
            & ~hanging
            & in_view
            & (ratio >= AREA_RANGE[0])
            & (ratio <= AREA_RANGE[1])
        )
        walking = walking_headings(tr, floor, fps, width, pickable)
        tails = tail_headings(tr, pickable)
        kept, conflicts = fuse(walking, tails)
        walking_rows = [i for i, (_p, k) in walking.items() if k == "lateral"]
        walk_h = (
            np.median(tr.height[walking_rows]) if len(walking_rows) >= 10 else med_h
        )
        rearing = (tr.height / walk_h >= REAR_RATIO) & pickable
        for i, r in enumerate(items):
            table["part_id"].append(part_id)
            table["frame_idx"].append(int(r["frame_idx"]))
            table["t_ns"].append(int(r["t_ns"]))
            table["track_id"].append(track_id)
            table["walking_deg"].append(kept.get(i, np.nan))
            table["tail_deg"].append(tails.get(i, np.nan))
            table["conflict"].append(i in conflicts)
            table["rearing"].append(bool(rearing[i]) and i not in kept)
            table["pickable"].append(bool(pickable[i]))
            table["bbox"].append(tuple(float(v) for v in r["bbox"]))

    embed_rows = [
        i
        for i, (f, ok) in enumerate(zip(table["frame_idx"], table["pickable"]))
        if ok and f % EMBED_EVERY == 0
    ]
    # Decode sequentially (Tenet 11) and embed in fixed batches as the crops come, so memory
    # stays one batch of pictures however long the recording is; only features are kept.
    order: list[int] = []
    feats, mirrors = [], []
    batch: list[tuple[int, np.ndarray]] = []
    done = 0

    def flush() -> None:
        nonlocal done
        if not batch:
            return
        pictures = [c for _i, c in batch]
        feats.append(_embed(model, pictures))
        mirrors.append(
            _embed(model, [np.ascontiguousarray(c[:, ::-1]) for c in pictures])
        )
        order.extend(i for i, _c in batch)
        done += len(batch)
        batch.clear()
        _report(
            progress,
            "decoding and embedding",
            done / max(len(embed_rows), 1),
            f"{done}/{len(embed_rows)} frames",
        )

    for part_id in sorted(set(table["part_id"])):
        want: dict[int, list[int]] = {}
        for i in embed_rows:
            if table["part_id"][i] == part_id:
                want.setdefault(table["frame_idx"][i], []).append(i)
        if not want:
            continue
        part = recording_store.video_files(data)[
            subject_frames._part_index(data, part_id)
        ]
        cap = cv2.VideoCapture(
            str(recording_store.resolve_part_video_path(rec_dir, part))
        )
        last, frame = max(want), 0
        while frame <= last:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame in want:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                for i in want[frame]:
                    crop = subject_frames.subject_crop(rgb, table["bbox"][i])
                    batch.append(
                        (
                            i,
                            cv2.resize(
                                crop, (CROP_PX, CROP_PX), interpolation=cv2.INTER_AREA
                            ),
                        )
                    )
                if len(batch) >= EMBED_BATCH * 4:
                    flush()
            frame += 1
        cap.release()
    flush()
    feature_row = np.full(len(table["frame_idx"]), -1, np.int32)
    feature_row[order] = np.arange(len(order), dtype=np.int32)
    _write_cache(
        rec_dir,
        tracking_run_id,
        table,
        feature_row,
        (
            np.concatenate(feats)
            if feats
            else np.zeros((0, POOL * POOL * PROJ_DIM), np.float16)
        ),
        (
            np.concatenate(mirrors)
            if mirrors
            else np.zeros((0, POOL * POOL * PROJ_DIM), np.float16)
        ),
        floor,
        model,
    )
    return headings_dir(rec_dir, tracking_run_id)


def _write_cache(
    rec_dir, tracking_run_id, table, feature_row, feats, mirrors, floor, model
):
    out = headings_dir(rec_dir, tracking_run_id)
    out.mkdir(parents=True, exist_ok=True)
    parts = sorted(set(table["part_id"]))
    part_col = np.array(table["part_id"])
    for part_id in parts:
        sel = np.flatnonzero(part_col == part_id)
        root = zarr_store.open_group(out / f"{part_id}.zarr", mode="w")
        root.attrs["schema_version"] = SCHEMA_VERSION
        columns = {
            "frame_idx": ("int32", np.array(table["frame_idx"])[sel]),
            "t_ns": ("int64", np.array(table["t_ns"])[sel]),
            "track_id": ("int32", np.array(table["track_id"])[sel]),
            "walking_deg": (
                "float32",
                np.array(table["walking_deg"], dtype=float)[sel],
            ),
            "tail_deg": ("float32", np.array(table["tail_deg"], dtype=float)[sel]),
            "conflict": ("bool", np.array(table["conflict"])[sel]),
            "rearing": ("bool", np.array(table["rearing"])[sel]),
            "pickable": ("bool", np.array(table["pickable"])[sel]),
            "bbox": ("float32", np.array(table["bbox"], dtype=float)[sel]),
        }
        rows_here = feature_row[sel]
        keep = rows_here >= 0
        local = np.full(len(sel), -1, np.int32)
        local[keep] = np.arange(int(keep.sum()), dtype=np.int32)
        columns["feature_row"] = ("int32", local)
        for name, (dtype, values) in columns.items():
            chunks = (ROW_CHUNK, *values.shape[1:])
            arr = root.create_array(
                name, shape=values.shape, chunks=chunks, dtype=dtype
            )
            arr[:] = values.astype(dtype)
        for name, values in (
            ("features", feats[rows_here[keep]]),
            ("features_mirror", mirrors[rows_here[keep]]),
        ):
            arr = root.create_array(
                name,
                shape=values.shape,
                chunks=(FEATURE_CHUNK, values.shape[1]),
                dtype="float16",
            )
            arr[:] = values
    meta = {
        "schema_version": SCHEMA_VERSION,
        "tracking_run_id": tracking_run_id,
        "floor": floor,
        "backbone": {
            "model_id": "dinov3_vit",
            "variant": getattr(model, "weight", None),
            "crop_px": CROP_PX,
            "pool": POOL,
            "projection": {"kind": "random", "seed": PROJ_SEED, "dim": PROJ_DIM},
        },
        "embed_every": EMBED_EVERY,
        "method": {"version": METHOD_VERSION, "constants": CONSTANTS},
        "parts": parts,
        "created_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    tmp = out / (META_FILENAME + ".tmp")
    tmp.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    atomic.replace(tmp, out / META_FILENAME)


@dataclass
class Headings:
    """One recording's cache, read whole (it is small): parallel arrays over its rows."""

    recording_id: str
    tracking_run_id: str
    part_id: np.ndarray
    frame_idx: np.ndarray
    t_ns: np.ndarray
    track_id: np.ndarray
    walking_deg: np.ndarray
    tail_deg: np.ndarray
    conflict: np.ndarray
    rearing: np.ndarray
    pickable: np.ndarray
    bbox: np.ndarray
    feature_row: np.ndarray  # into ``features`` (recording-wide), -1 when not embedded
    features: np.ndarray
    features_mirror: np.ndarray

    def key(self, i: int) -> tuple[str, str, int, int]:
        return (
            self.recording_id,
            str(self.part_id[i]),
            int(self.t_ns[i]),
            int(self.track_id[i]),
        )


def load_headings(
    experiment_dir: Path | str, recording_id: str, tracking_run_id: str
) -> Headings:
    """Read a recording's cache. Raises ``FileNotFoundError`` when it was never written."""
    rec_dir = experiment_store.recording_path(Path(experiment_dir), recording_id)
    out = headings_dir(rec_dir, tracking_run_id)
    meta = yaml_load.safe_load(atomic.read_text(out / META_FILENAME))
    cols: dict[str, list] = {}
    feats, mirrors, offset = [], [], 0
    for part_id in meta["parts"]:
        root = zarr_store.open_group(out / f"{part_id}.zarr", mode="r")
        n = root["frame_idx"].shape[0]
        cols.setdefault("part_id", []).append(np.full(n, part_id))
        for name in (
            "frame_idx",
            "t_ns",
            "track_id",
            "walking_deg",
            "tail_deg",
            "conflict",
            "rearing",
            "pickable",
            "bbox",
        ):
            cols.setdefault(name, []).append(np.asarray(root[name][:]))
        fr = np.asarray(root["feature_row"][:])
        cols.setdefault("feature_row", []).append(np.where(fr >= 0, fr + offset, -1))
        feats.append(np.asarray(root["features"][:]))
        mirrors.append(np.asarray(root["features_mirror"][:]))
        offset += feats[-1].shape[0]
    joined = {k: np.concatenate(v) for k, v in cols.items()}
    return Headings(
        recording_id=recording_id,
        tracking_run_id=tracking_run_id,
        features=np.concatenate(feats).astype(np.float32),
        features_mirror=np.concatenate(mirrors).astype(np.float32),
        **joined,
    )


# --------------------------------------------------------------------------- #
# The pool: every frame's bucket, from the caches of the chosen recordings
# --------------------------------------------------------------------------- #
@dataclass
class Pool:
    """Every pickable embedded frame of the chosen recordings, with its bucket and how its
    heading was known. Parallel arrays; ``recording`` indexes ``caches``."""

    caches: list[Headings]
    recording: np.ndarray
    row: np.ndarray  # into that cache
    heading: np.ndarray  # degrees, NaN for rearing / unclear
    bucket: np.ndarray  # into HEADING_BUCKETS
    source: np.ndarray  # into SOURCES, -1 for rearing / unclear
    confidence: np.ndarray
    features: np.ndarray

    def key(self, j: int) -> tuple[str, str, int, int]:
        return self.caches[int(self.recording[j])].key(int(self.row[j]))


def _features(h: Headings, rows: np.ndarray) -> np.ndarray:
    f = h.features[h.feature_row[rows]]
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


def heading_pool(caches: list[Headings]) -> Pool:
    """Each embedded pickable frame's bucket: its own walking or tail heading when it has one,
    else the similarity-weighted circular mean of its ``KNN_K`` nearest labelled frames from the
    *other* recordings (walking frames, their mirror images, tail frames), ``unclear`` below
    ``KNN_MIN_AGREEMENT``; rearing frames are their own bucket."""
    ref_z, ref_phi, ref_rec = [], [], []
    q_rec, q_row = [], []
    for r, h in enumerate(caches):
        embedded = (h.feature_row >= 0) & h.pickable
        walking = embedded & np.isfinite(h.walking_deg)
        tail = embedded & np.isfinite(h.tail_deg) & ~h.conflict & ~walking
        for rows, phi, mirror in (
            (np.flatnonzero(walking), h.walking_deg, True),
            (np.flatnonzero(tail), h.tail_deg, False),
        ):
            if not len(rows):
                continue
            ref_z.append(_features(h, rows))
            ref_phi.append(phi[rows])
            ref_rec.append(np.full(len(rows), r))
            if mirror:
                fm = h.features_mirror[h.feature_row[rows]]
                ref_z.append(fm / (np.linalg.norm(fm, axis=1, keepdims=True) + 1e-8))
                ref_phi.append(angle_diff(180.0, phi[rows]))
                ref_rec.append(np.full(len(rows), r))
        rows = np.flatnonzero(embedded)
        q_rec.append(np.full(len(rows), r))
        q_row.append(rows)
    q_rec, q_row = np.concatenate(q_rec), np.concatenate(q_row)
    feats = (
        np.concatenate(
            [_features(caches[r], q_row[q_rec == r]) for r in range(len(caches))]
        )
        if len(q_row)
        else np.zeros((0, 1))
    )
    # the order above is by recording, which is q_rec's order
    heading = np.full(len(q_row), np.nan)
    source = np.full(len(q_row), -1)
    confidence = np.zeros(len(q_row))
    for j in range(len(q_row)):
        h, i = caches[q_rec[j]], q_row[j]
        if np.isfinite(h.walking_deg[i]):
            heading[j], confidence[j] = h.walking_deg[i], 1.0
            source[j] = 0 if np.isfinite(h.tail_deg[i]) else 1
        elif np.isfinite(h.tail_deg[i]) and not h.conflict[i]:
            heading[j], source[j], confidence[j] = h.tail_deg[i], 2, 1.0
    if ref_z:
        ref_z, ref_phi, ref_rec = (
            np.concatenate(ref_z),
            np.concatenate(ref_phi),
            np.concatenate(ref_rec),
        )
        need = np.flatnonzero(source < 0)
        rad = np.radians(ref_phi)
        k = min(KNN_K, len(ref_phi))
        for start in range(0, len(need), 1024):
            js = need[start : start + 1024]
            sim = feats[js] @ ref_z.T
            sim[q_rec[js, None] == ref_rec[None, :]] = -np.inf
            if not np.isfinite(sim).any():
                continue
            idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            w = np.take_along_axis(sim, idx, axis=1)
            w = np.where(
                np.isfinite(w), np.exp((w - w.max(1, keepdims=True)) / KNN_TAU), 0.0
            )
            c = (w * np.cos(rad[idx])).sum(1) / np.maximum(w.sum(1), 1e-12)
            s = (w * np.sin(rad[idx])).sum(1) / np.maximum(w.sum(1), 1e-12)
            agree = np.hypot(c, s)
            ok = agree >= KNN_MIN_AGREEMENT
            heading[js[ok]] = np.degrees(np.arctan2(s[ok], c[ok]))
            source[js[ok]] = 3
            confidence[js] = agree
    bucket = np.where(
        np.isfinite(heading),
        bin_of(np.nan_to_num(heading)),
        HEADING_BUCKETS.index("unclear"),
    )
    rearing = np.array(
        [caches[q_rec[j]].rearing[q_row[j]] for j in range(len(q_row))], bool
    )
    bucket = np.where(
        rearing & (source != 0) & (source != 1),
        HEADING_BUCKETS.index("rearing"),
        bucket,
    )
    return Pool(caches, q_rec, q_row, heading, bucket, source, confidence, feats)


def rank(
    pool: Pool, bucket: str, exclude: set, anchors: np.ndarray, n: int
) -> list[int]:
    """Up to ``n`` of ``bucket``'s pool indices, best first: source, then confidence, then each
    next one the farthest (cosine) from ``anchors`` and the picks so far among the next best
    few, so turning one down never hands back its near-twin."""
    b = HEADING_BUCKETS.index(bucket)
    members = [
        j for j in np.flatnonzero(pool.bucket == b) if pool.key(j) not in exclude
    ]
    order = sorted(
        members,
        key=lambda j: (
            pool.source[j] if pool.source[j] >= 0 else 9,
            -pool.confidence[j],
        ),
    )
    chosen: list[int] = []
    taken = [a for a in anchors] if len(anchors) else []
    window = 25
    while order and len(chosen) < n:
        head = order[:window]
        if taken:
            ref = np.stack(taken)
            score = [float(np.max(pool.features[j] @ ref.T)) for j in head]
            best = head[int(np.argmin(score))]
        else:
            best = head[0]
        chosen.append(best)
        taken.append(pool.features[best])
        order.remove(best)
    return chosen


# --------------------------------------------------------------------------- #
# Orchestration: what the strategy and the review window call
# --------------------------------------------------------------------------- #
def build_pool(
    experiment_dir: Path | str,
    sources: dict[str, str],
    *,
    floor_class: str | None = None,
    groups: dict[str, str] | None = None,
    device: str | None = None,
    progress=None,
) -> Pool:
    """The pool over ``sources`` (recording id -> tracking run id), computing every cache that
    is missing or was written by another method version or floor. ``progress`` hears each
    recording's phases folded into the whole: ``(done + inner) / total`` (§6).

    Raises:
        ValueError: No sources, no DINOv3 weights, or a tracking run without masks.
        FileNotFoundError: A recording to compute has no video on this machine.
    """
    experiment_dir = Path(experiment_dir)
    if not sources:
        raise ValueError("choose at least one recording to sample frames from")
    floors = (
        resolve_floors(experiment_dir, sources, floor_class, groups)
        if floor_class
        else {r: None for r in sources}
    )
    todo = [
        r
        for r in sorted(sources)
        if not cache_matches(
            experiment_store.recording_path(experiment_dir, r), sources[r], floors[r]
        )
    ]
    model = None
    for done, recording_id in enumerate(todo):
        if model is None:
            _report(progress, "loading the backbone", 0.0, "DINOv3 ViT")
            model = _load_backbone(device)

        def inner(stage, fraction, detail, done=done, recording_id=recording_id):
            _report(
                progress,
                stage,
                (done + fraction) / len(todo),
                f"{recording_id} ({done + 1}/{len(todo)}) · {detail}",
            )

        compute_headings(
            experiment_dir,
            recording_id,
            sources[recording_id],
            floors[recording_id],
            model,
            inner,
        )
    if model is not None:
        _release(model)
    _report(progress, "matching similar frames", 0.0, f"{len(sources)} recordings")
    caches = [load_headings(experiment_dir, r, sources[r]) for r in sorted(sources)]
    pool = heading_pool(caches)
    _report(progress, "matching similar frames", 1.0, f"{len(pool.row)} frames")
    return pool


def _load_backbone(device: str | None):
    from detail.core.utils import default_compute, resolve_device
    from detail.models.dinov3 import DINOv3ViT

    variant = subject_frames.installed_backbone()
    if variant is None:
        raise ValueError(
            "the heading sampler needs DINOv3 ViT weights; install them in the Model "
            "manager (embedding · patch features)"
        )
    model = DINOv3ViT(variant=variant)
    backends = model.supported_backends
    model.load(device or resolve_device(default_compute(backends), backends))
    return model


def _release(model) -> None:
    import torch

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def anchors_for(pool: Pool, example_set, bucket: str) -> np.ndarray:
    """Features of the set's frames already in ``bucket``: what the next pick must differ
    from."""
    index = {pool.key(j): j for j in range(len(pool.row))}
    b = HEADING_BUCKETS.index(bucket)
    rows = [
        index[f.key()]
        for f in example_set.frames
        if f.key() in index and pool.bucket[index[f.key()]] == b
    ]
    return pool.features[rows] if rows else np.zeros((0, pool.features.shape[1]))


def candidates(pool: Pool, example_set, bucket: str, n: int) -> list[dict]:
    """``bucket``'s next ``n`` candidates, best first, skipping the set's frames and those
    rejected for the bucket: ``{key, recording_id, tracking_run_id, part_id, frame_idx, t_ns,
    track_id, bbox, source, confidence, heading_deg}``."""
    exclude = {f.key() for f in example_set.frames}
    exclude |= {r.key() for r in example_set.rejected if r.bucket in (None, bucket)}
    out = []
    for j in rank(pool, bucket, exclude, anchors_for(pool, example_set, bucket), n):
        h, i = pool.caches[int(pool.recording[j])], int(pool.row[j])
        out.append(
            {
                "key": h.key(i),
                "recording_id": h.recording_id,
                "tracking_run_id": h.tracking_run_id,
                "part_id": str(h.part_id[i]),
                "frame_idx": int(h.frame_idx[i]),
                "t_ns": int(h.t_ns[i]),
                "track_id": int(h.track_id[i]),
                "bbox": tuple(float(v) for v in h.bbox[i]),
                "source": SOURCES[pool.source[j]] if pool.source[j] >= 0 else None,
                "confidence": float(pool.confidence[j]),
                "heading_deg": float(pool.heading[j]),
            }
        )
    return out


def coverage(pool: Pool | None, example_set) -> dict[str, dict[str, int]]:
    """Per bucket, the set's frames in it — ``{bucket: {"in_set": n, "marked": m}}`` — plus
    ``hand-picked`` and ``other`` (sampled another way, and not in the pool): what the compass
    draws. A frame's bucket is its ``sampled_by`` when the heading sampler chose it, else the
    pool's when the pool holds it."""
    names = HEADING_BUCKETS + ("hand-picked", "other")
    out = {b: {"in_set": 0, "marked": 0} for b in names}
    for i, bucket in enumerate(frame_buckets(pool, example_set)):
        out[bucket]["in_set"] += 1
        out[bucket]["marked"] += int(example_set.labeled(i))
    return out


def frame_buckets(pool: Pool | None, example_set) -> list[str]:
    """Each of the set's frames' bucket, in frame order: its ``sampled_by`` when the heading
    sampler chose it, else the pool's when the pool holds it, else ``hand-picked`` or
    ``other``. The one rule the compass and the review window both use."""
    index = {pool.key(j): j for j in range(len(pool.row))} if pool is not None else {}
    out = []
    for frame in example_set.frames:
        bucket = None
        if frame.sampled_by and frame.sampled_by.startswith("heading:"):
            bucket = frame.sampled_by.split(":")[1]
        elif frame.key() in index:
            bucket = HEADING_BUCKETS[int(pool.bucket[index[frame.key()]])]
        if bucket not in HEADING_BUCKETS:
            bucket = "hand-picked" if frame.sampled_by is None else "other"
        out.append(bucket)
    return out
