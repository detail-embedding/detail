"""Per-recording ROI/POI spatial-features export.

Synchronous, read-only analysis over a completed tracking run: for each
``(frame, track)``, the track's bbox / centroids / areas and its position
relative to the recording's ROIs and points of interest, as three tidy tables keyed
by ``t_ns`` (so they join with streams, §3).

**The body.** Beside the whole mask's centre and area, ``tracks`` carries the *body*: its
centre, area, length, width and the direction of its long axis. The body is the mask, or,
with ``remove_thin_parts``, the largest piece of it left after a morphological opening that
removes a tail, a limb or an antenna. Length and width are four standard deviations of the
body's area along and across its principal axis: the full axes of an ellipse with the same
spread, which a ragged outline does not move the way it moves a bounding box.

Three things that are decided here because each went wrong once on real recordings
(detail#75):

- **The opening is sized from the animal, never from the calibration**: its diameter is
  ``thin_part_scale`` times the square root of the subject's own median mask area. A wrong
  millimetre calibration then cannot change a body's shape.
- **One opening per subject per recording**, from masks spread over the whole recording, so a
  body is measured the same way from its first frame to its last.
- **Sampling never invents a row.** ``sample_fps`` keeps the tracked frame nearest each tick of
  a clock; a tick with no tracked frame near it has no row, in any of the three tables.

This is **not** a §16 ``AnalysisModule`` (those are experiment-scoped, cross-recording)
— it is the per-recording substrate such a module would aggregate, exposed as a
synchronous export (no executor, no run record). The relation math is pure numpy /
cv2 over arrays; only ``compute_location_tables`` reads the on-disk zarrs.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from detail.core import atomic
from detail.analysis.geometry import mask_extent
from detail.analysis.geometry import window_centre_norm
from detail.configs.location_export import LocationExportConfig
from detail.core.utils import active_polygon_at_frame, decode_mask
from detail.executor import rundir
from detail.storage import recording as recording_store
from detail.storage import spatial as spatial_store
from detail.storage.zarr_io import PartZarrReader

SCHEMA_VERSION = 1

#: The body's columns, after every column an older export has, so a reader that takes
#: columns by position still finds what it found before.
BODY_COLUMNS = [
    "body_cx_norm", "body_cy_norm", "body_cx_px", "body_cy_px",
    "body_area_norm", "body_area_px",
    "body_length_px", "body_width_px", "body_angle_deg",
    "body_cx_mm", "body_cy_mm", "body_area_mm2",
    "body_length_mm", "body_width_mm",
]  # fmt: skip
NS_PER_S = 1_000_000_000
#: Masks read per subject to size its opening, spread evenly over its tracked frames.
OPENING_PROBE_FRAMES = 40

TRACKS_COLUMNS = [
    "frame_idx", "t_ns", "object_id",
    "bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm",
    "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
    "bbox_cx_norm", "bbox_cy_norm", "bbox_cx_px", "bbox_cy_px",
    "mask_cx_norm", "mask_cy_norm", "mask_cx_px", "mask_cy_px",
    "bbox_area_norm", "bbox_area_px", "mask_area_norm", "mask_area_px",
    "bbox_cx_mm", "bbox_cy_mm", "mask_cx_mm", "mask_cy_mm",
    "bbox_area_mm2", "mask_area_mm2",
    *BODY_COLUMNS,
]  # fmt: skip
ROI_COLUMNS = [
    "frame_idx", "t_ns", "object_id", "roi_id", "roi_name",
    "centroid_in_roi", "mask_centroid_in_roi",
    "bbox_frac_in_roi", "mask_frac_in_roi",
    "bbox_fully_in_roi", "mask_fully_in_roi",
    "signed_dist_norm", "signed_dist_px", "signed_dist_mm",
]  # fmt: skip
POINT_COLUMNS = [
    "frame_idx", "t_ns", "object_id", "point_id", "point_name",
    "centroid_dist_norm", "centroid_dist_px",
    "mask_centroid_dist_norm", "mask_centroid_dist_px",
    "point_in_bbox", "point_in_mask",
    "centroid_dist_mm", "mask_centroid_dist_mm",
]  # fmt: skip


@dataclass
class LocationTables:
    """The three tidy tables as ``(columns, rows)`` (CSV-ready), plus lineage.

    All keyed by ``t_ns`` (int64, from the detection zarr — Tenet 8) so they join
    with the streams CSV. ROI/POI rows appear only for frames where the definition
    is active (§10).
    """

    tracks: tuple[list[str], list[tuple]]
    roi_relations: tuple[list[str], list[tuple]]
    point_relations: tuple[list[str], list[tuple]]
    source: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Pure relation math (arrays in, plain values out)
# --------------------------------------------------------------------------- #
def _bbox_features(box, fw: float, fh: float) -> dict:
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    return {
        "x1_norm": x1, "y1_norm": y1, "x2_norm": x2, "y2_norm": y2,
        "x1_px": x1 * fw, "y1_px": y1 * fh, "x2_px": x2 * fw, "y2_px": y2 * fh,
        "cx_norm": cx, "cy_norm": cy, "cx_px": cx * fw, "cy_px": cy * fh,
        "area_norm": area, "area_px": area * fw * fh,
    }  # fmt: skip


def _mask_features(
    window, x0: int, y0: int, mw: int, mh: int, fw: float, fh: float, fallback
) -> dict:
    """Mask centroid (normalized + px) and area, of a mask given as its ``window`` of the
    ``(mh, mw)`` grid at ``(x0, y0)``. ``fallback`` (bbox centroid norm) is used when the
    mask is empty."""
    # The kinematics centre (cell centres, ``+ 0.5``), not the mean cell index: that was half
    # a mask cell up and left of the animal (#32).
    cx_n, cy_n = window_centre_norm(window, x0, y0, mw, mh)
    if not np.isfinite(cx_n):
        cx_n, cy_n = fallback
    area_norm = float(window.sum()) / (mw * mh)
    return {
        "cx_norm": cx_n, "cy_norm": cy_n, "cx_px": cx_n * fw, "cy_px": cy_n * fh,
        "area_norm": area_norm, "area_px": area_norm * fw * fh,
    }  # fmt: skip


def sampled_indices(t_ns: np.ndarray, sample_fps: float) -> np.ndarray:
    """Which of one subject's tracked frames to keep at ``sample_fps``, as indices into
    ``t_ns`` in ascending order.

    A clock ticks every ``1 / sample_fps`` seconds from the start of the recording. Each frame
    belongs to its nearest tick, and each tick keeps the frame nearest it (the earlier one on a
    tie). A tick no frame is nearest to keeps nothing, so a gap in tracking stays a gap; at a
    rate above the video's every frame is kept.
    """
    t_ns = np.asarray(t_ns, dtype=np.int64)
    period = NS_PER_S / float(sample_fps)
    tick = np.rint(t_ns / period).astype(np.int64)
    distance = np.abs(t_ns - tick * period)
    order = np.lexsort((np.arange(len(t_ns)), distance, tick))
    ticks = tick[order]
    first = np.r_[True, ticks[1:] != ticks[:-1]] if len(ticks) else np.zeros(0, bool)
    return np.sort(order[first])


def thin_part_kernel(
    median_area_px: float, scale: float, cell_w_px: float, cell_h_px: float
) -> tuple[np.ndarray, float]:
    """The opening that removes a subject's thin parts, and its diameter in video pixels.

    The diameter is ``scale`` times the square root of the subject's median mask area, so it
    follows the animal's size in the image and not the distance calibration. The kernel is an
    ellipse on the mask grid, wider or taller in cells where the grid is finer than the video
    in that direction.

    Raises:
        ValueError: The masks are too coarse for an opening that small: it would span less
            than one and a half cells both ways, and would remove nothing.
    """
    diameter_px = float(scale) * float(np.sqrt(max(median_area_px, 0.0)))
    cells = (diameter_px / cell_w_px, diameter_px / cell_h_px)
    if max(cells) < 1.5:
        raise ValueError(
            f"thin parts cannot be removed from these masks: the opening would be "
            f"{diameter_px:.1f} px across, less than one and a half mask cells "
            f"({cell_w_px:.1f} x {cell_h_px:.1f} px each). Raise thin_part_scale, or leave "
            "remove_thin_parts off."
        )
    kx, ky = (max(1, int(round(c)) | 1) for c in cells)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky)), diameter_px


def body_window(
    mask: np.ndarray, extent: tuple[int, int, int, int], kernel: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """The body without its thin parts, as ``(window, x0, y0)``: the largest connected piece
    of ``mask`` left by an opening with ``kernel``, cut out of the grid at ``(x0, y0)``.

    ``extent`` is the mask's own (:func:`geometry.mask_extent`). The opening runs on that
    window grown by the kernel's size, taken from the grid itself: where the animal touches
    the edge of the frame the window stops there too, and the edge does not erode it. A mask
    the opening removes entirely (a sliver) comes back whole.
    """
    y0, y1, x0, x1 = extent
    ky, kx = kernel.shape
    top, left = max(y0 - ky, 0), max(x0 - kx, 0)
    grown = mask[top : min(y1 + ky, mask.shape[0]), left : min(x1 + kx, mask.shape[1])]
    opened = cv2.morphologyEx(
        np.ascontiguousarray(grown, dtype=np.uint8), cv2.MORPH_OPEN, kernel
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if count <= 1:
        return mask[y0:y1, x0:x1], x0, y0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest, left, top


def body_axes(body: np.ndarray, cell_w_px: float, cell_h_px: float) -> dict:
    """Length, width (video pixels) and long-axis direction (degrees, 0 to 180) of a body,
    given as the whole grid or as any window of it that holds the body.

    Four standard deviations of the body's area along and across its principal axis. The
    spread is that of the cells' area, not of their centres, so each cell adds its own
    ``side ** 2 / 12``: without it the same animal measures shorter on a coarser mask grid.
    An empty mask has no axes.
    """
    ys, xs = np.nonzero(body)
    if len(xs) == 0:
        return {"length_px": None, "width_px": None, "angle_deg": None}
    x = (xs + 0.5) * cell_w_px
    y = (ys + 0.5) * cell_h_px
    cov = np.cov(np.vstack([x, y]), bias=True) if len(xs) > 1 else np.zeros((2, 2))
    cov = cov + np.diag([cell_w_px**2 / 12.0, cell_h_px**2 / 12.0])
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, 1]
    return {
        "length_px": 4.0 * float(np.sqrt(max(evals[1], 0.0))),
        "width_px": 4.0 * float(np.sqrt(max(evals[0], 0.0))),
        "angle_deg": float(np.degrees(np.arctan2(major[1], major[0])) % 180.0),
    }


def _rasterize_poly(poly_norm, mw: int, mh: int) -> np.ndarray:
    grid = np.zeros((mh, mw), dtype=np.uint8)
    pts = np.round(np.asarray(poly_norm, dtype=np.float64) * (mw, mh)).astype(np.int32)
    cv2.fillPoly(grid, [pts], 1)
    return grid.astype(bool)


def _roi_geometry(poly_norm, mw, mh, fw, fh) -> dict:
    """Polygon-derived artifacts that do not depend on the track: the vertices
    (normalized + pixel) and the rasterized ROI grid. Resolved once per (ROI,
    frame) and reused across every track in that frame — the polygon is identical
    for all of them."""
    verts_n = np.asarray(poly_norm, dtype=np.float32)
    return {
        "verts_n": verts_n,
        "verts_px": (verts_n * (fw, fh)).astype(np.float32),
        "grid": _rasterize_poly(poly_norm, mw, mh),
    }


def _cells(norm: float, n: int, rounding) -> int:
    """A normalized box edge on an ``n``-cell grid, ``rounding`` outward (``np.floor`` for a
    low edge, ``np.ceil`` for a high one), with an edge within 1e-9 of a cell boundary taken
    as on it."""
    value = float(norm) * n
    nearest = round(value)
    if abs(value - nearest) < 1e-9:
        return int(nearest)
    return int(rounding(value))


def _roi_relation(geom, feat, mask, mw, mh) -> dict:
    verts_n, verts_px, roi_grid = geom["verts_n"], geom["verts_px"], geom["grid"]
    centroid_in = cv2.pointPolygonTest(
        verts_n, (feat["cx_norm"], feat["cy_norm"]), False
    )
    mask_cen_in = cv2.pointPolygonTest(
        verts_n, (feat["mcx_norm"], feat["mcy_norm"]), False
    )
    # Every cell the box touches: the low corner floored, the high one ceiled. Rounding both
    # left a box under one cell wide covering no cell at all, so it read 0% inside a region
    # it sat in (#32). `_cells` keeps a corner on a cell edge (0.3 x 40 is 12.000000000000002)
    # from taking in the next cell.
    bx1, by1 = _cells(feat["x1_norm"], mw, np.floor), _cells(
        feat["y1_norm"], mh, np.floor
    )
    bx2, by2 = _cells(feat["x2_norm"], mw, np.ceil), _cells(
        feat["y2_norm"], mh, np.ceil
    )
    bbox_grid = np.zeros((mh, mw), dtype=bool)
    bbox_grid[max(by1, 0) : by2, max(bx1, 0) : bx2] = True
    bbox_n, mask_n = int(bbox_grid.sum()), int(mask.sum())
    return {
        "centroid_in_roi": int(centroid_in >= 0),
        "mask_centroid_in_roi": int(mask_cen_in >= 0),
        "bbox_frac_in_roi": (bbox_grid & roi_grid).sum() / bbox_n if bbox_n else 0.0,
        "mask_frac_in_roi": (mask & roi_grid).sum() / mask_n if mask_n else 0.0,
        "bbox_fully_in_roi": int(bbox_n > 0 and not (bbox_grid & ~roi_grid).any()),
        "mask_fully_in_roi": int(mask_n > 0 and not (mask & ~roi_grid).any()),
        "signed_dist_norm": cv2.pointPolygonTest(
            verts_n, (feat["cx_norm"], feat["cy_norm"]), True
        ),
        "signed_dist_px": cv2.pointPolygonTest(
            verts_px, (feat["cx_px"], feat["cy_px"]), True
        ),
    }  # fmt: skip


def _point_relation(p_norm, feat, mask, mw, mh, fw, fh) -> dict:
    px_n, py_n = float(p_norm[0][0]), float(p_norm[0][1])
    gx, gy = int(px_n * mw), int(py_n * mh)
    in_mask = 0 <= gy < mh and 0 <= gx < mw and bool(mask[gy, gx])
    return {
        "centroid_dist_norm": float(np.hypot(feat["cx_norm"] - px_n, feat["cy_norm"] - py_n)),
        "centroid_dist_px": float(np.hypot((feat["cx_norm"] - px_n) * fw, (feat["cy_norm"] - py_n) * fh)),
        "mask_centroid_dist_norm": float(np.hypot(feat["mcx_norm"] - px_n, feat["mcy_norm"] - py_n)),
        "mask_centroid_dist_px": float(np.hypot((feat["mcx_norm"] - px_n) * fw, (feat["mcy_norm"] - py_n) * fh)),
        "point_in_bbox": int(feat["x1_norm"] <= px_n <= feat["x2_norm"] and feat["y1_norm"] <= py_n <= feat["y2_norm"]),
        "point_in_mask": int(in_mask),
    }  # fmt: skip


def _r(v, nd):
    return round(float(v), nd)


# --------------------------------------------------------------------------- #
# Orchestrator (reads zarrs; the only filesystem-touching code)
# --------------------------------------------------------------------------- #
def _spatial_def(recording_dir, kind: str, entry: dict) -> tuple:
    """``(id, name, PolygonKeyframes, active-interval)`` for one ROI/POI, reading
    its keyframes once so the per-frame loop interpolates in memory rather than
    re-reading recording.yaml + the keyframe CSV on every frame."""
    pk = spatial_store.read_keyframes(
        recording_dir, kind, entry["id"], entry["n_vertices"]
    )
    active = entry.get("active")
    interval = (active["start"], active["end"]) if active else None
    return entry["id"], entry["name"], pk, interval


def _plan_groups(parts, tracking_dir: Path, offsets: list[int]) -> list[tuple]:
    """Per detection part: ``(part, {(frame, track): [row, ...]}, {(frame, track): t_ns})``.

    The groups a row of each table is made from, read from the three small columns that name
    them, so what is sampled and what is counted for progress are settled before a mask is
    decoded. Frames are recording-wide, so a key is unique across parts.
    """
    planned = []
    for part_index, det_part in parts:
        reader = PartZarrReader(det_part)
        assign = PartZarrReader(tracking_dir / "assignments" / det_part.name)
        object_id = np.asarray(assign.column("object_id"))
        frame_idx = np.asarray(reader.column("frame_idx"))
        t_ns = np.asarray(reader.column("t_ns"))
        offset = offsets[part_index] if part_index < len(offsets) else offsets[-1]
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in np.where(object_id >= 0)[0]:
            groups[(int(frame_idx[i]) + offset, int(object_id[i]))].append(int(i))
        times = {key: int(t_ns[rows[0]]) for key, rows in groups.items()}
        planned.append((det_part, dict(groups), times))
    return planned


def _sampled_keys(planned: list[tuple], sample_fps: float) -> set[tuple[int, int]]:
    """The ``(frame, track)`` groups kept at ``sample_fps``, each track on its own frames."""
    by_track: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for _part, _groups, times in planned:
        for (frame, oid), t in times.items():
            by_track[oid].append((t, frame))
    kept: set[tuple[int, int]] = set()
    for oid, seen in by_track.items():
        seen.sort()
        t_ns = np.array([t for t, _frame in seen], dtype=np.int64)
        kept.update((seen[i][1], oid) for i in sampled_indices(t_ns, sample_fps))
    return kept


def _group_mask(mask_rle, rows: list[int], shape: tuple[int, int]) -> np.ndarray:
    """One track's mask on one frame: its detections OR-merged, like embedding (§14)."""
    mask = decode_mask(str(mask_rle[rows[0]]), shape).astype(bool)
    for j in rows[1:]:
        mask = mask | decode_mask(str(mask_rle[j]), shape).astype(bool)
    return mask


def _track_openings(
    planned: list[tuple],
    kept: set[tuple[int, int]] | None,
    shape: tuple[int, int],
    cell_px: tuple[float, float],
    scale: float,
) -> dict[int, tuple[np.ndarray, float]]:
    """``{track: (kernel, diameter in px)}``: one opening per track for the whole recording,
    sized from the median area of masks spread evenly over the frames it will be measured on.
    """
    where: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for index, (_part, groups, _times) in enumerate(planned):
        for frame, oid in groups:
            if kept is None or (frame, oid) in kept:
                where[oid].append((frame, index, oid))
    openings = {}
    for oid, frames in where.items():
        frames.sort()
        step = max(1, len(frames) // OPENING_PROBE_FRAMES)
        probe = frames[::step][:OPENING_PROBE_FRAMES]
        columns: dict[int, object] = {}
        areas = []
        for frame, index, _oid in probe:
            det_part, groups, _times = planned[index]
            if index not in columns:
                columns[index] = PartZarrReader(det_part).column("mask_rle")
            rows = groups[(frame, oid)]
            lo, hi = min(rows), max(rows) + 1
            block = np.asarray(columns[index][lo:hi])
            mask = _group_mask(block, [r - lo for r in rows], shape)
            areas.append(float(mask.sum()) * cell_px[0] * cell_px[1])
        openings[oid] = thin_part_kernel(float(np.median(areas)), scale, *cell_px)
    return openings


def compute_location_tables(
    recording,
    tracking_run_id: str,
    on_progress: Callable[[int, int], None] | None = None,
    options: LocationExportConfig | None = None,
) -> LocationTables:
    """Compute the three tables for a completed tracking run of ``recording``.

    ``options`` carries the choices about the measurement (``sample_fps``,
    ``remove_thin_parts``, ``thin_part_scale``); only those fields are read, and ``None`` is
    the model's defaults: every tracked frame, the whole mask as the body.

    Reads the tracking assignments + its detection zarr + the recording's ROIs/POIs;
    OR-merges each track's detections per frame (bbox + mask, like embedding §14);
    numbers frames recording-global via ``recording_store.part_frame_offsets`` in the
    detection run's frame space.

    ``on_progress(rows_done, rows_total)`` is called as detection rows are consumed
    (the caller throttles), so a long single-recording run shows a live bar. The
    denominator is the count of *assigned* detection rows — the only rows that
    become groups — so the bar reaches 100% even when an arena filter leaves some
    detections unlinked.
    """
    options = options if options is not None else LocationExportConfig()
    recording_dir = Path(recording.dir)
    tracking_dir = recording_dir / "tracking" / tracking_run_id
    track_meta = yaml.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    detection_run_id = track_meta["detection_run_id"]
    detection_dir = recording_dir / "detections" / detection_run_id
    det_env = rundir.read_envelope(detection_dir)
    mask_h, mask_w = (int(v) for v in det_env["mask_resolution"])
    recording_store.raise_on_legacy_downsampling(det_env.get("job"))
    every_n_frames = int((det_env.get("job") or {}).get("every_n_frames") or 1)
    offsets = recording_store.part_frame_offsets(recording_dir) or [0]
    res = recording.video_resolution
    fw, fh = (float(res[0]), float(res[1])) if res else (1.0, 1.0)
    # pixels_per_mm at the video resolution → real-world (_mm) sibling columns; blank when
    # the recording is uncalibrated. inv converts a pixel length to mm, inv2 an area to mm².
    ppm = recording.pixels_per_mm
    inv = (1.0 / ppm) if ppm else None
    inv2 = (inv * inv) if inv is not None else None

    def _mm(px: float | None):
        return _r(px * inv, 3) if inv is not None and px is not None else ""

    def _px(value: float | None):
        return _r(value, 2) if value is not None else ""

    def _mm2(px: float):
        return _r(px * inv2, 3) if inv2 is not None else ""

    # Resolve each ROI/POI's keyframes + active interval ONCE here: roi_at_frame /
    # point_at_frame re-read recording.yaml + the keyframe CSV on every call, which in
    # a per-frame loop is hundreds of thousands of redundant disk reads. With the
    # keyframes in hand, active_polygon_at_frame interpolates in memory.
    roi_defs = [_spatial_def(recording_dir, "roi", r) for r in recording.rois]
    point_defs = [_spatial_def(recording_dir, "point", p) for p in recording.points]

    track_rows: list[tuple] = []
    roi_rows: list[tuple] = []
    point_rows: list[tuple] = []

    # Play order, with each part's play index: after a part move the sorted folder listing is
    # not the order the parts play in, and the offsets are indexed by play position.
    parts = recording_store.parts_in_play_order(recording_dir, detection_dir)
    planned = _plan_groups(parts, tracking_dir, offsets)
    kept = (
        _sampled_keys(planned, options.sample_fps)
        if options.sample_fps is not None
        else None
    )
    # A mask cell's size in video pixels: the mask grid need not have the video's shape.
    cell_px = (fw / mask_w, fh / mask_h)
    openings = (
        _track_openings(
            planned, kept, (mask_h, mask_w), cell_px, options.thin_part_scale
        )
        if options.remove_thin_parts
        else {}
    )
    # The denominator must match what rows_done accrues: only assigned rows
    # (object_id >= 0) become groups, and with sampling only the groups kept, so
    # counting every detection row would leave the bar short.
    rows_total = sum(
        len(rows)
        for _part, groups, _times in planned
        for key, rows in groups.items()
        if kept is None or key in kept
    )
    rows_done = 0

    for det_part, groups, _times in planned:
        if kept is not None:
            groups = {key: rows for key, rows in groups.items() if key in kept}
            if not groups:
                continue
        det = PartZarrReader(det_part).read()
        t_ns = np.asarray(det["t_ns"])
        bbox = np.asarray(det["bbox"])
        mask_rle = det["mask_rle"]

        # ROI geometry (rasterized grid + vertices) and the POI point are identical
        # for every track in a frame, so resolve them once per frame and reuse.
        # Groups are processed in frame order, so the cache holds only the current
        # frame's geometry (bounded — no accumulation across frames).
        cache_frame: int | None = None
        roi_geom: dict[str, dict | None] = {}
        point_geom: dict[str, np.ndarray | None] = {}

        for gframe, oid in sorted(groups):
            idxs = groups[(gframe, oid)]
            if gframe != cache_frame:
                roi_geom, point_geom = {}, {}
                for rid, _name, pk, interval in roi_defs:
                    poly = active_polygon_at_frame(pk, interval, gframe)
                    roi_geom[rid] = (
                        _roi_geometry(poly, mask_w, mask_h, fw, fh)
                        if poly is not None
                        else None
                    )
                for pid, _name, pk, interval in point_defs:
                    point_geom[pid] = active_polygon_at_frame(pk, interval, gframe)
                cache_frame = gframe

            t = int(t_ns[idxs[0]])
            boxes = bbox[idxs]
            box = (
                boxes[:, 0].min(), boxes[:, 1].min(),
                boxes[:, 2].max(), boxes[:, 3].max(),
            )  # fmt: skip
            mask = _group_mask(mask_rle, idxs, (mask_h, mask_w))

            bf = _bbox_features(box, fw, fh)
            # Everything about the mask is measured on its own window of the grid: an
            # animal fills a few percent of a full-resolution grid, and scanning the rest of
            # it on every frame was most of this loop's time.
            extent = mask_extent(mask)
            y0, y1, x0, x1 = extent if extent is not None else (0, 0, 0, 0)
            window = mask[y0:y1, x0:x1]
            mf = _mask_features(
                window, x0, y0, mask_w, mask_h, fw, fh, (bf["cx_norm"], bf["cy_norm"])
            )
            # The body: the mask itself, or what an opening leaves of it. Measured by the
            # same function as the mask, so with thin parts kept the two centres and areas
            # are the same numbers by construction.
            opening = openings.get(oid)
            if opening is not None and extent is not None:
                body, bx0, by0 = body_window(mask, extent, opening[0])
                bd = _mask_features(
                    body, bx0, by0, mask_w, mask_h, fw, fh,
                    (mf["cx_norm"], mf["cy_norm"]),
                )  # fmt: skip
            else:
                body, bd = window, mf
            axes = body_axes(body, *cell_px)
            feat = {
                **bf,
                "mcx_norm": mf["cx_norm"], "mcy_norm": mf["cy_norm"],
                "mcx_px": mf["cx_px"], "mcy_px": mf["cy_px"],
            }  # fmt: skip
            track_rows.append((
                gframe, t, oid,
                _r(bf["x1_norm"], 6), _r(bf["y1_norm"], 6), _r(bf["x2_norm"], 6), _r(bf["y2_norm"], 6),
                _r(bf["x1_px"], 2), _r(bf["y1_px"], 2), _r(bf["x2_px"], 2), _r(bf["y2_px"], 2),
                _r(bf["cx_norm"], 6), _r(bf["cy_norm"], 6), _r(bf["cx_px"], 2), _r(bf["cy_px"], 2),
                _r(mf["cx_norm"], 6), _r(mf["cy_norm"], 6), _r(mf["cx_px"], 2), _r(mf["cy_px"], 2),
                _r(bf["area_norm"], 8), _r(bf["area_px"], 2), _r(mf["area_norm"], 8), _r(mf["area_px"], 2),
                _mm(bf["cx_px"]), _mm(bf["cy_px"]), _mm(mf["cx_px"]), _mm(mf["cy_px"]),
                _mm2(bf["area_px"]), _mm2(mf["area_px"]),
                _r(bd["cx_norm"], 6), _r(bd["cy_norm"], 6), _r(bd["cx_px"], 2), _r(bd["cy_px"], 2),
                _r(bd["area_norm"], 8), _r(bd["area_px"], 2),
                _px(axes["length_px"]), _px(axes["width_px"]), _px(axes["angle_deg"]),
                _mm(bd["cx_px"]), _mm(bd["cy_px"]), _mm2(bd["area_px"]),
                _mm(axes["length_px"]), _mm(axes["width_px"]),
            ))  # fmt: skip

            for rid, name, _pk, _interval in roi_defs:
                geom = roi_geom[rid]
                if geom is None:
                    continue
                r = _roi_relation(geom, feat, mask, mask_w, mask_h)
                roi_rows.append((
                    gframe, t, oid, rid, name,
                    r["centroid_in_roi"], r["mask_centroid_in_roi"],
                    _r(r["bbox_frac_in_roi"], 6), _r(r["mask_frac_in_roi"], 6),
                    r["bbox_fully_in_roi"], r["mask_fully_in_roi"],
                    _r(r["signed_dist_norm"], 6), _r(r["signed_dist_px"], 2),
                    _mm(r["signed_dist_px"]),
                ))  # fmt: skip

            for pid, name, _pk, _interval in point_defs:
                p = point_geom[pid]
                if p is None:
                    continue
                r = _point_relation(p, feat, mask, mask_w, mask_h, fw, fh)
                point_rows.append((
                    gframe, t, oid, pid, name,
                    _r(r["centroid_dist_norm"], 6), _r(r["centroid_dist_px"], 2),
                    _r(r["mask_centroid_dist_norm"], 6), _r(r["mask_centroid_dist_px"], 2),
                    r["point_in_bbox"], r["point_in_mask"],
                    _mm(r["centroid_dist_px"]), _mm(r["mask_centroid_dist_px"]),
                ))  # fmt: skip

            if on_progress is not None:
                rows_done += len(idxs)
                on_progress(rows_done, rows_total)

    track_rows.sort(key=lambda row: (row[0], row[2]))
    roi_rows.sort(key=lambda row: (row[0], row[2], row[3]))
    point_rows.sort(key=lambda row: (row[0], row[2], row[3]))
    return LocationTables(
        tracks=(TRACKS_COLUMNS, track_rows),
        roi_relations=(ROI_COLUMNS, roi_rows),
        point_relations=(POINT_COLUMNS, point_rows),
        source={
            "schema_version": SCHEMA_VERSION,
            "tracking_run_id": tracking_run_id,
            "detection_run_id": detection_run_id,
            "every_n_frames": every_n_frames,
            "mask_resolution": [mask_h, mask_w],
            "video_resolution": list(res) if res else None,
            "pixels_per_mm": ppm,
            # What the body columns were measured from, so a reader can tell a run that has
            # them from one that does not, and how the tail was treated, without parsing rows.
            "sample_fps": options.sample_fps,
            "remove_thin_parts": bool(options.remove_thin_parts),
            "thin_part_scale": float(options.thin_part_scale),
            "thin_part_opening_px": {
                str(oid): round(diameter, 2)
                for oid, (_kernel, diameter) in sorted(openings.items())
            },
            "body_axis": "shape",
            "roi_ids": [d[0] for d in roi_defs],
            "point_ids": [d[0] for d in point_defs],
        },
    )
