"""Per-recording ROI/POI spatial-features export.

Synchronous, read-only analysis over a completed tracking run: for each
``(frame, track)``, the track's bbox / centroids / areas and its position
relative to the recording's ROIs and points of interest, as three tidy tables keyed
by ``t_ns`` (so they join with streams, §3).

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

from detail.core import atomic
from detail.core import yaml_load
from detail.core.utils import active_polygon_at_frame, decode_mask
from detail.executor import rundir
from detail.storage import recording as recording_store
from detail.storage import spatial as spatial_store
from detail.storage.zarr_io import PartZarrReader, resolve_assignment_rows

SCHEMA_VERSION = 1

TRACKS_COLUMNS = [
    "frame_idx", "t_ns", "object_id",
    "bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm",
    "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
    "bbox_cx_norm", "bbox_cy_norm", "bbox_cx_px", "bbox_cy_px",
    "mask_cx_norm", "mask_cy_norm", "mask_cx_px", "mask_cy_px",
    "bbox_area_norm", "bbox_area_px", "mask_area_norm", "mask_area_px",
    "bbox_cx_mm", "bbox_cy_mm", "mask_cx_mm", "mask_cy_mm",
    "bbox_area_mm2", "mask_area_mm2",
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
#: One row per frame per unordered pair of subjects both tracked on it (§16 *Keypoints and pairs
#: in the export*); the edge columns are blank unless the run measured them.
PAIR_COLUMNS = [
    "frame_idx", "t_ns", "object_id_a", "object_id_b",
    "mask_centroid_dist_norm", "mask_centroid_dist_px", "mask_centroid_dist_mm",
    "bbox_centroid_dist_norm", "bbox_centroid_dist_px", "bbox_centroid_dist_mm",
    "mask_edge_dist_px", "mask_edge_dist_mm",
]  # fmt: skip
#: One row per frame x subject x skeleton point of a keypoint track run; a point the run did not
#: place has blank coordinates and score, and ``visible`` 0.
KEYPOINT_COLUMNS = [
    "frame_idx", "t_ns", "object_id", "keypoint",
    "x_norm", "y_norm", "x_px", "y_px", "x_mm", "y_mm", "score", "visible",
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
    pair_relations: tuple[list[str], list[tuple]] = field(
        default_factory=lambda: (PAIR_COLUMNS, [])
    )
    #: ``None`` when the export was given no keypoint track run.
    keypoints: tuple[list[str], list[tuple]] | None = None
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


def _mask_features(mask, mw: int, mh: int, fw: float, fh: float, fallback) -> dict:
    """Mask centroid (normalized + px) and area. ``fallback`` (bbox centroid norm)
    is used when the mask is empty."""
    ys, xs = np.where(mask)
    if len(xs):
        cx_n, cy_n = float(xs.mean()) / mw, float(ys.mean()) / mh
    else:
        cx_n, cy_n = fallback
    area_norm = float(mask.sum()) / (mw * mh)
    return {
        "cx_norm": cx_n, "cy_norm": cy_n, "cx_px": cx_n * fw, "cy_px": cy_n * fh,
        "area_norm": area_norm, "area_px": area_norm * fw * fh,
    }  # fmt: skip


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


def _roi_relation(geom, feat, mask, mw, mh) -> dict:
    verts_n, verts_px, roi_grid = geom["verts_n"], geom["verts_px"], geom["grid"]
    centroid_in = cv2.pointPolygonTest(
        verts_n, (feat["cx_norm"], feat["cy_norm"]), False
    )
    mask_cen_in = cv2.pointPolygonTest(
        verts_n, (feat["mcx_norm"], feat["mcy_norm"]), False
    )
    bx1, by1 = int(round(feat["x1_norm"] * mw)), int(round(feat["y1_norm"] * mh))
    bx2, by2 = int(round(feat["x2_norm"] * mw)), int(round(feat["y2_norm"] * mh))
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


def mask_edge_gap(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    """The gap between two masks' nearest pixels, in mask pixels: ``0`` when they touch (share
    an edge or a corner) or overlap. ``None`` when either mask is empty.

    The distance transform of "not A" gives every pixel's distance to A's nearest pixel centre;
    the nearest of B's pixels is the centre-to-centre distance, and one pixel of that is the two
    pixels' own half-widths, so the gap between their edges is one less.
    """
    if not mask_a.any() or not mask_b.any():
        return None
    if (mask_a & mask_b).any():
        return 0.0
    to_a = cv2.distanceTransform(
        (~mask_a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    return max(float(to_a[mask_b].min()) - 1.0, 0.0)


def _pair_row(frame: int, t: int, a: dict, b: dict, fw: float, fh: float, mm, edge_px):
    """One ``pair_relations`` row from two subjects' rounded centroids (the values
    ``tracks.csv`` holds, so the distance is reproducible from it)."""

    def dist(key_x: str, key_y: str) -> tuple[float, float]:
        dx, dy = a[key_x] - b[key_x], a[key_y] - b[key_y]
        return float(np.hypot(dx, dy)), float(np.hypot(dx * fw, dy * fh))

    mask_n, mask_px = dist("mcx", "mcy")
    box_n, box_px = dist("cx", "cy")
    return (
        frame, t, a["oid"], b["oid"],
        _r(mask_n, 6), _r(mask_px, 2), mm(mask_px),
        _r(box_n, 6), _r(box_px, 2), mm(box_px),
        "" if edge_px is None else _r(edge_px, 2),
        "" if edge_px is None else mm(edge_px),
    )  # fmt: skip


def keypoint_table(
    recording,
    keypoint_run_id: str,
    tracking_run_id: str,
    frame_wh: tuple[float, float],
    mm,
) -> tuple[tuple[list[str], list[tuple]], dict]:
    """``keypoints.csv`` for one keypoint track run, and what the manifest records of it.

    Read through ``api.keypoint_export.PointSource`` (the track run, the detect method's display
    cut-off, coordinates kept under it) so this table and every keypoint export agree. Frames
    are lifted to recording-global numbering with the offsets ``tracks.csv`` uses.

    Raises:
        ValueError: The keypoint track run assigned its poses to another tracking run, whose
            ``object_id``s are not this export's.
    """
    from detail.api.keypoint_export import PointSource

    source = PointSource(recording.dir, keypoint_run_id)
    if source.tracking_run_id != tracking_run_id:
        raise ValueError(
            f"keypoint track run {keypoint_run_id!r} assigned its poses to tracking run "
            f"{source.tracking_run_id!r}, not {tracking_run_id!r}; its subjects are not this "
            "export's, so its points would land on the wrong animal"
        )
    offsets = recording_store.part_frame_offsets(Path(recording.dir)) or [0]
    stems = source.part_stems()
    names = list(source.skeleton.names)
    fw, fh = frame_wh
    rows: list[tuple] = []
    for part_id in source.part_ids():
        index = stems.index(part_id) if part_id in stems else 0
        offset = offsets[index] if index < len(offsets) else offsets[-1]
        for object_id, part in source.rows_in(part_id).items():
            for i in range(len(part.frame_idx)):
                frame, t = int(part.frame_idx[i]) + offset, int(part.t_ns[i])
                for k, name in enumerate(names):
                    x, y = float(part.points[i, k, 0]), float(part.points[i, k, 1])
                    score = float(part.scores[i, k])
                    if np.isfinite(x) and np.isfinite(y):
                        place = (
                            _r(x, 6), _r(y, 6), _r(x * fw, 2), _r(y * fh, 2),
                            mm(x * fw), mm(y * fh),
                        )  # fmt: skip
                    else:
                        place = ("",) * 6
                    rows.append((
                        frame, t, int(object_id), name, *place,
                        _r(score, 6) if np.isfinite(score) else "",
                        int(bool(part.visible[i, k])),
                    ))  # fmt: skip
    rows.sort(key=lambda row: (row[0], row[2]))  # stable: points stay in skeleton order
    meta = {
        "keypoint_run_id": keypoint_run_id,
        "keypoint_cutoff": source.cutoff,
        "keypoint_names": names,
    }
    return (KEYPOINT_COLUMNS, rows), meta


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


def compute_location_tables(
    recording,
    tracking_run_id: str,
    on_progress: Callable[[int, int], None] | None = None,
    *,
    pair_edge_distance: bool = False,
    keypoint_run_id: str | None = None,
) -> LocationTables:
    """Compute the three tables for a completed tracking run of ``recording``.

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
    recording_dir = Path(recording.dir)
    tracking_dir = recording_dir / "tracking" / tracking_run_id
    track_meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    detection_run_id = track_meta["detection_run_id"]
    class_name = track_meta.get("class_name")
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

    def _mm(px: float):
        return _r(px * inv, 3) if inv is not None else ""

    def _mm2(px: float):
        return _r(px * inv2, 3) if inv2 is not None else ""

    # Resolve each ROI/POI's keyframes + active interval ONCE here: roi_at_frame /
    # point_at_frame re-read recording.yaml + the keyframe CSV on every call, which in
    # a per-frame loop is hundreds of thousands of redundant disk reads. With the
    # keyframes in hand, active_polygon_at_frame interpolates in memory.
    roi_defs = [_spatial_def(recording_dir, "roi", r) for r in recording.rois]
    point_defs = [_spatial_def(recording_dir, "point", p) for p in recording.points]

    track_rows: list[tuple] = []
    pair_rows: list[tuple] = []
    # The subjects of the frame being walked, flushed into pair rows when the frame changes:
    # groups arrive in frame order, so this holds one frame, never the recording.
    frame_subjects: list[dict] = []
    frame_key: tuple[int, int] | None = None

    def flush_pairs() -> None:
        subjects = sorted(frame_subjects, key=lambda sub: sub["oid"])
        for i, a in enumerate(subjects):
            for b in subjects[i + 1 :]:
                edge = None
                if pair_edge_distance:
                    gap = mask_edge_gap(a["mask"], b["mask"])
                    edge = None if gap is None else gap * fw / mask_w
                pair_rows.append(
                    _pair_row(frame_key[0], frame_key[1], a, b, fw, fh, _mm, edge)
                )
        frame_subjects.clear()

    roi_rows: list[tuple] = []
    point_rows: list[tuple] = []

    parts = sorted(detection_dir.glob("part_*.zarr"))
    assign_dir = tracking_dir / "assignments"
    # The denominator must match what rows_done accrues: only assigned rows
    # (object_id >= 0) become groups, so counting every detection row would leave
    # the bar short whenever an arena filter or unlinked detections drop rows. One
    # small column read per part (the loop opens each assignment part anyway).
    rows_total = sum(
        int(
            (
                np.asarray(PartZarrReader(assign_dir / p.name).column("object_id")) >= 0
            ).sum()
        )
        for p in parts
    )
    rows_done = 0

    for part_index, det_part in enumerate(parts):
        det_reader = PartZarrReader(det_part)
        assign_reader = PartZarrReader(tracking_dir / "assignments" / det_part.name)
        # object_id is parallel to the detection rows the tracker actually kept
        # when it ran (§13) — every row when this run had no class filter, only
        # one class's otherwise (anchor_tracker et al. build object_ids from
        # filter_by_class'd data). raw_rows recovers which reader row each
        # assignment row really is.
        raw_rows = resolve_assignment_rows(det_reader, assign_reader, class_name)
        n = len(raw_rows)
        det = {k: np.asarray(v)[raw_rows] for k, v in det_reader.read().items()}
        assign = assign_reader.read()
        object_id = np.asarray(assign["object_id"][:n])
        frame_idx = np.asarray(det["frame_idx"])
        t_ns = np.asarray(det["t_ns"])
        bbox = np.asarray(det["bbox"])
        mask_rle = det["mask_rle"]
        offset = offsets[part_index] if part_index < len(offsets) else offsets[-1]

        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in np.where(object_id >= 0)[0]:
            groups[(int(frame_idx[i]) + offset, int(object_id[i]))].append(int(i))

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
            mask = decode_mask(str(mask_rle[idxs[0]]), (mask_h, mask_w)).astype(bool)
            for j in idxs[1:]:
                mask = mask | decode_mask(str(mask_rle[j]), (mask_h, mask_w)).astype(
                    bool
                )

            bf = _bbox_features(box, fw, fh)
            mf = _mask_features(
                mask, mask_w, mask_h, fw, fh, (bf["cx_norm"], bf["cy_norm"])
            )
            if frame_key is not None and frame_key[0] != gframe:
                flush_pairs()
            frame_key = (gframe, t)
            frame_subjects.append(
                {
                    "oid": oid,
                    "mcx": _r(mf["cx_norm"], 6), "mcy": _r(mf["cy_norm"], 6),
                    "cx": _r(bf["cx_norm"], 6), "cy": _r(bf["cy_norm"], 6),
                    "mask": mask if pair_edge_distance else None,
                }
            )  # fmt: skip
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

    if frame_subjects:
        flush_pairs()
    track_rows.sort(key=lambda row: (row[0], row[2]))
    roi_rows.sort(key=lambda row: (row[0], row[2], row[3]))
    point_rows.sort(key=lambda row: (row[0], row[2], row[3]))
    pair_rows.sort(key=lambda row: (row[0], row[2], row[3]))
    keypoints, keypoint_meta = None, {"keypoint_run_id": None}
    if keypoint_run_id is not None:
        keypoints, keypoint_meta = keypoint_table(
            recording, keypoint_run_id, tracking_run_id, (fw, fh), _mm
        )
    return LocationTables(
        tracks=(TRACKS_COLUMNS, track_rows),
        roi_relations=(ROI_COLUMNS, roi_rows),
        point_relations=(POINT_COLUMNS, point_rows),
        pair_relations=(PAIR_COLUMNS, pair_rows),
        keypoints=keypoints,
        source={
            "schema_version": SCHEMA_VERSION,
            "tracking_run_id": tracking_run_id,
            # §13 ``{object_id: label}`` (a track merge names its animals), so the export
            # carries who each ``object_id`` is even after its tracking run is gone.
            "subject_labels": track_meta.get("subject_labels"),
            "detection_run_id": detection_run_id,
            "every_n_frames": every_n_frames,
            "mask_resolution": [mask_h, mask_w],
            "video_resolution": list(res) if res else None,
            "pixels_per_mm": ppm,
            "roi_ids": [d[0] for d in roi_defs],
            "point_ids": [d[0] for d in point_defs],
            "pair_edge_distance": bool(pair_edge_distance),
            **keypoint_meta,
        },
    )
