"""Embedding stage runner — coordinator over one or more (track, part) units (Section 14).

Two execution paths share one per-unit function (``_embedding_worker.embed_one``):

- **Inline** (single device): load model + renderer once in this process, run
  each ``(track_id, part_index)`` unit sequentially. Zero subprocess overhead.
- **Multi-GPU** (``compute.backend == "cuda"`` and ``len(device_indices) > 1``):
  ``ProcessPoolExecutor(spawn)`` sized to ``len(device_indices)``; each worker
  is pinned to one GPU via ``CUDA_VISIBLE_DEVICES`` before any torch import,
  loads the model + renderer once at startup, then chews through units as
  they're scheduled round-robin per §14.

Per-unit zarr outputs (``track_<tid>/part_<NNN>.zarr``) are independent stores
so workers never contend on the same file. Cross-process channel is file-based:
workers write per-unit progress JSONs under ``<run_dir>/parts/track_<tid>_<part_id>.json``;
the coordinator polls these between ``EmbeddingProgressEvent`` emissions.
Stop = the coordinator touches ``<run_dir>/stop.flag``; workers poll it between
batches and exit early with ``status="failed"``. **Embedding treats stop as
failure** (not partial) — embedding has no resume support, so partial would be
misleading.
"""

import math
import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml
from detail.analysis import geometry
from detail.core import yaml_load
from detail.core.visualize import bgr_to_rgb
from detail.configs.embedding import EmbeddingConfig
from detail.core import atomic, disk_space
from detail.core.dataclasses import (
    EmbeddingJob,
    EmbeddingProgressEvent,
    RenderContext,
)
from detail.core.progress import PROGRESS_INTERVAL_S, remaining_seconds
from detail.core.skeleton import Skeleton
from detail.core.protocols import ModelNotReadyError
from detail.core.registries import (
    EMBEDDING_MODELS,
    TRANSFORMS,
    construct_embedding_model,
)
from detail.core.utils import (
    check_cuda_indices,
    decode_mask,
    effective_backend,
    mask_boxes,
    resolve_device,
    resolve_downstream_every_n,
    resolve_track_id,
    utc_timestamp,
)
from detail.executor import rundir
from detail.executor.runners import ProcessingModule, register_processing_module
from detail.pipeline import _embedding_worker, teacher_window
from detail.storage import experiment, guidance, keypoint_runs, read_cache
from detail.storage import recording as recording_store
from detail.storage import remote_targets
from detail.storage import video_integrity
from detail.storage.zarr_io import (  # noqa: F401
    DetectionPartReader,
    PartZarrReader,
    kinematics_part_path,
    object_ids_on_detection_rows,
    resolve_assignment_rows,
)
from detail.transforms import basic as _basic  # noqa: F401 — registers transforms
from detail.transforms.constructors import (
    ALIGNABLE,
    CONSTRUCTORS,
    alignment_step,
    check_pose_step,
    with_alignment,
    with_pose,
)
from detail.transforms.renderer import MaskRenderer

SCHEMA_VERSION = 1
# Embedding runs a multiprocess pool and emits by *polling* the worker progress
# files at the shared live-bar cadence (§6) — it doesn't use the in-loop
# ThrottledProgress helper (a worker.join(timeout) loop is already incremental).
_PROGRESS_POLL_INTERVAL_S = PROGRESS_INTERVAL_S
# The max-contrast background is resolved once per run (``resolve_shared_config``), never per
# recording. Still recorded on every run: runs made before that say ``per_recording``, and a
# cross-run consumer must not pool the two procedures as one representation.
BG_COLOR_SCOPE = "shared"
# The channel order of every image the renderer is handed and the model receives. Recorded on
# every run because runs made before it was fixed rendered and embedded BGR frames (a red
# subject reached the model in the blue channel); an absent key reads as "bgr", so the two are
# never compared as one representation (``api/run_preview._RECIPE_KEYS``).
CHANNEL_ORDER = "rgb"

#: What a row carries besides its vector: frame_idx, t_ns and det_row (int64 each), the
#: alignment angle, and the kinematics sibling (t_ns, centre, area, bbox). Rounded up.
EMBEDDING_ROW_EXTRA_BYTES = 96

#: Read-outs that concatenate two pooled features, so the vector is twice the model's width.
_DOUBLED_READOUTS = frozenset({"gap_gmp", "cls_patch"})


def embedding_dim(config: dict) -> int:
    """The width of one stored vector, from the model's declared per-variant width (no
    weights loaded): doubled by a concatenating read-out. Falls back to 1024 for a model that declares no width, the size the
    measured runs had, so an unknown model is estimated, not skipped."""
    model_cls = EMBEDDING_MODELS.get(config.get("model_id"))
    if model_cls is None:
        from detail.api._runner_modules import PIPELINE_RUNNERS
        from detail.executor.runners import import_runner_modules

        import_runner_modules(PIPELINE_RUNNERS)
        model_cls = EMBEDDING_MODELS.get(config.get("model_id"))
    per_variant = getattr(model_cls, "VARIANT_OUTPUT_DIMS", None) or {}
    dim = per_variant.get(config.get("model_variant")) or getattr(
        model_cls, "output_dim", 0
    )
    dim = int(dim) or 1024
    if config.get("readout") in _DOUBLED_READOUTS:
        dim *= 2
    return dim


def embedding_run_dir(experiment_root, job: EmbeddingJob) -> Path:
    """Run directory for an embedding job: ``recordings/<id>/embeddings/<run_id>/``."""
    return (
        experiment.recording_path(experiment_root, job.recording_id)
        / "embeddings"
        / job.embedding_run_id
    )


def largest_mask_px(
    recording_dir,
    tracking_run_id: str,
    track_ids: list[int],
    *,
    diagonal: bool = False,
) -> int:
    """Largest mask dimension (px) across the given tracks' detections.

    The "centered, true-scale" constructors size their shared canvas to this so
    every mask keeps its relative size. Measured at the detection's mask
    resolution (the same pixel scale the renderer's frame is decoded at).

    ``diagonal`` measures each subject box's diagonal instead of its longer side: the canvas
    a body-axis ``rotate`` needs (§26), because a turned subject's extent along either axis
    is at most its box's diagonal (``align_rotate`` crops to the turned subject).

    A completed tracking + detection run is immutable, so the result is cached
    against the tracking run's ``track.yaml`` mtime (keyed on the track set) — a
    batch launch computes it once per recording; the launch dialog runs that off the
    UI thread, and this makes a re-launch instant. A re-created run rewrites
    ``track.yaml`` (new mtime) and recomputes.

    Raises:
        ValueError: If the tracks have no detections in the run.
    """
    recording_dir = Path(recording_dir)
    key = (
        "largest_mask_px",
        str(recording_dir),
        tracking_run_id,
        tuple(sorted({int(t) for t in track_ids})),
    )
    stamp = recording_dir / "tracking" / tracking_run_id / "track.yaml"
    side, diag = read_cache.cached_read(
        key, stamp, lambda: _largest_mask_px(recording_dir, tracking_run_id, track_ids)
    )
    return diag if diagonal else side


def _largest_mask_px(
    recording_dir: Path, tracking_run_id: str, track_ids: list[int]
) -> tuple[int, int]:
    tracking_dir = recording_dir / "tracking" / tracking_run_id
    track_meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    detection_dir = recording_dir / "detections" / track_meta["detection_run_id"]
    frame_h, frame_w = rundir.read_envelope(detection_dir)["mask_resolution"]
    wanted = np.asarray(sorted({int(t) for t in track_ids}), dtype=np.int64)
    class_name = track_meta.get("class_name")
    largest = 0.0
    largest_diag = 0.0
    for part_path in sorted(detection_dir.glob("part_*.zarr")):
        reader = PartZarrReader(part_path)
        assignment = PartZarrReader(tracking_dir / "assignments" / part_path.name)
        raw_rows = resolve_assignment_rows(reader, assignment, class_name)
        n = len(raw_rows)
        bbox = np.asarray(reader.column("bbox")[: reader.n_rows])[raw_rows]
        frame_idx = np.asarray(reader.column("frame_idx")[: reader.n_rows])[raw_rows]
        object_id = assignment.column("object_id")[:n]
        sel = np.isin(object_id.astype(np.int64), wanted)
        if not sel.any():
            continue
        # Detections sharing a (track, frame) are OR-merged downstream, so size
        # the canvas to the union bbox per group — not each detection alone.
        unions: dict[tuple[int, int], list[float]] = {}
        for oid, fr, box in zip(object_id[sel], frame_idx[sel], bbox[sel]):
            key = (int(oid), int(fr))
            cur = unions.get(key)
            if cur is None:
                unions[key] = [
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ]
            else:
                cur[0], cur[1] = min(cur[0], float(box[0])), min(cur[1], float(box[1]))
                cur[2], cur[3] = max(cur[2], float(box[2])), max(cur[3], float(box[3]))
        for x1, y1, x2, y2 in unions.values():
            width, height = (x2 - x1) * frame_w, (y2 - y1) * frame_h
            largest = max(largest, width, height)
            largest_diag = max(largest_diag, math.hypot(width, height))
    if largest <= 0:
        raise ValueError(
            f"no detections for track_ids {sorted(set(int(t) for t in track_ids))} "
            f"in tracking run {tracking_run_id!r}"
        )
    # round before ceil so float32 bbox noise can't inflate the canvas by a pixel. The
    # diagonal gets 2 px more: a turned mask is rasterized nearest-neighbour, so its edge can
    # land a pixel past the exact rotated extent on either side.
    return (
        int(math.ceil(round(largest, 3))),
        int(math.ceil(round(largest_diag, 3))) + 2,
    )


# --------------------------------------------------------------------------- #
# §14 Egocentric alignment: the heading per unit, and the turned-mask canvas
# --------------------------------------------------------------------------- #
def _alignment_settings(config_or_job) -> dict:
    """The alignment fields as one dict, from a validated config or an `EmbeddingJob`."""
    get = (
        config_or_job.get
        if isinstance(config_or_job, dict)
        else lambda k, d=None: getattr(config_or_job, k, d)
    )
    return {
        "align": str(get("align") or "none"),
        "keypoint_run_id": get("keypoint_run_id"),
        "heading_definition": get("heading_definition"),
        "heading_cutoff": get("heading_cutoff"),
        "max_gap_s": float(get("max_gap_s", 0.5) or 0.0),
        "mirror_hold_s": float(get("mirror_hold_s", 1.0) or 0.0),
        "facing_camera_ratio": float(get("facing_camera_ratio", 0.5) or 0.0),
        "pose": str(get("pose") or "none"),
        "pose_weight": get("pose_weight"),
        "pose_min_score": get("pose_min_score"),
    }


def track_frames_at_cadence(
    det_part_path: Path,
    assignment_path: Path,
    track_id: int,
    every_n_frames: int,
    class_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(frame_idx, t_ns)`` of the frames one track occupies in a part at the run's
    cadence — the same rule `embed_one` selects rows by, so the heading computed here is
    for exactly the frames the worker embeds."""
    reader = DetectionPartReader(det_part_path)
    object_id = object_ids_on_detection_rows(
        reader, PartZarrReader(assignment_path), class_name
    )
    frame_idx = np.asarray(reader.column("frame_idx")[: reader.n_rows]).astype(np.int64)
    t_ns = np.asarray(reader.column("t_ns")[: reader.n_rows]).astype(np.int64)
    sel = object_id.astype(np.int64) == int(track_id)
    frames, first = np.unique(frame_idx[sel], return_index=True)
    keep = frames % int(every_n_frames) == 0
    return frames[keep], t_ns[sel][first][keep]


def headings_for_run(
    recording_dir: Path,
    tracking_run_id: str,
    track_ids: list[int],
    every_n_frames: int,
    settings: dict,
    mask_size: tuple[int, int],
) -> dict[tuple[int, str], dict]:
    """Per ``(track_id, part_id)``: ``{frame_idx, t_ns, deg, source, mirrored}`` arrays over
    the track's frames at the cadence (§14). The one computation the units builder, the
    canvas pre-pass and the launch form's coverage count share."""
    from detail.analysis import body_axis
    from detail.storage.keypoint_runs import KeypointTrackRun

    recording_dir = Path(recording_dir)
    tracking_dir = recording_dir / "tracking" / tracking_run_id
    track_meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    detection_dir = recording_dir / "detections" / track_meta["detection_run_id"]
    run = KeypointTrackRun(recording_dir, settings["keypoint_run_id"])
    frame_wh = (int(mask_size[1]), int(mask_size[0]))
    out: dict[tuple[int, str], dict] = {}
    for track_id in track_ids:
        series = body_axis.track_series(
            run,
            int(track_id),
            settings["heading_definition"],
            frame_wh,
            cutoff=settings["heading_cutoff"],
            facing_camera_ratio=settings["facing_camera_ratio"],
        )
        for det_part in sorted(detection_dir.glob("part_*.zarr")):
            part_id = det_part.stem
            frames, t_ns = track_frames_at_cadence(
                det_part,
                tracking_dir / "assignments" / det_part.name,
                int(track_id),
                every_n_frames,
                track_meta.get("class_name"),
            )
            part_series = series.get(part_id)
            if part_series is None:
                deg = np.full(frames.shape, np.nan, dtype=np.float32)
                source = np.full(frames.shape, body_axis.SOURCE_NONE, dtype=np.int8)
            else:
                deg, source = body_axis.heading_at(
                    part_series, t_ns, settings["max_gap_s"]
                )
            mirrored = (
                body_axis.mirror_flags(deg, t_ns, settings["mirror_hold_s"])
                if settings["align"] == "mirror"
                else np.zeros(frames.shape, dtype=bool)
            )
            out[(int(track_id), part_id)] = {
                "frame_idx": frames,
                "t_ns": t_ns,
                "deg": deg,
                "source": source,
                "mirrored": mirrored,
            }
    return out


def poses_for_run(
    recording_dir: Path,
    tracking_run_id: str,
    track_ids: list[int],
    every_n_frames: int,
    settings: dict,
    mask_size: tuple[int, int],
    headings: dict[tuple[int, str], dict] | None,
) -> dict[tuple[int, str], dict]:
    """Per ``(track_id, part_id)``: ``{frame_idx, xy, score, vec, n_visible}`` over the track's
    frames at the cadence (§14 *Pose in the embedding*), the points joined from the keypoint
    track run by frame. ``vec`` is the body-relative pose block (``pose: append``), None for
    ``draw``. ``headings`` is ``headings_for_run``'s answer for the same frames, which
    ``append`` needs to turn the numbers exactly as the image is turned.

    Read once per run, here, like the heading: the body length is the whole track's, which a
    unit holding one part could not compute.
    """
    from detail.analysis import body_axis, pose_features
    from detail.storage.keypoint_runs import KeypointTrackRun

    recording_dir = Path(recording_dir)
    pose = settings["pose"]
    tracking_dir = recording_dir / "tracking" / tracking_run_id
    track_meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    detection_dir = recording_dir / "detections" / track_meta["detection_run_id"]
    run = KeypointTrackRun(recording_dir, settings["keypoint_run_id"])
    k = len(run.skeleton.names)
    frame_wh = (int(mask_size[1]), int(mask_size[0]))
    out: dict[tuple[int, str], dict] = {}
    for track_id in track_ids:
        body_length = None
        if pose == "append":
            series = body_axis.track_series(
                run,
                int(track_id),
                settings["heading_definition"],
                frame_wh,
                cutoff=settings["heading_cutoff"],
                facing_camera_ratio=settings["facing_camera_ratio"],
            )
            body_length = next((h.typical_length_px for h in series.values()), None)
        have_parts = set(run.part_ids(int(track_id)))
        for det_part in sorted(detection_dir.glob("part_*.zarr")):
            part_id = det_part.stem
            heading = (headings or {}).get((int(track_id), part_id))
            if heading is not None:
                frames = np.asarray(heading["frame_idx"], dtype=np.int64)
            else:
                frames, _t_ns = track_frames_at_cadence(
                    det_part,
                    tracking_dir / "assignments" / det_part.name,
                    int(track_id),
                    every_n_frames,
                    track_meta.get("class_name"),
                )
            xy = np.full((frames.size, k, 2), np.nan, dtype=np.float32)
            score = np.full((frames.size, k), np.nan, dtype=np.float32)
            if part_id in have_parts and frames.size:
                rows = run.rows(int(track_id), part_id)
                at = {int(f): i for i, f in enumerate(np.asarray(rows["frame_idx"]))}
                for j, f in enumerate(frames.tolist()):
                    i = at.get(int(f))
                    if i is not None:
                        xy[j] = rows["points"][i]
                        score[j] = rows["point_scores"][i]
            vec = None
            if pose == "append":
                vec, n_visible = pose_features.pose_block(
                    xy,
                    score,
                    deg=np.asarray(heading["deg"], dtype=np.float64),
                    mirrored=np.asarray(heading["mirrored"], dtype=bool),
                    body_length_px=body_length,
                    frame_wh=frame_wh,
                    min_score=settings["pose_min_score"],
                    frame_mode="mirror" if settings["align"] == "mirror" else "rotate",
                )
            else:
                n_visible = (
                    pose_features.visible_points(xy, score, settings["pose_min_score"])
                    .sum(axis=1)
                    .astype(np.uint8)
                )
            out[(int(track_id), part_id)] = {
                "frame_idx": frames,
                "xy": xy,
                "score": score,
                "vec": vec,
                "n_visible": n_visible,
            }
    return out


def heading_coverage(headings: dict[tuple[int, str], dict]) -> dict:
    """``{track_<id>: {part_id: {frames, own, interpolated, dropped, longest_gap_s}}}``."""
    from detail.analysis import body_axis

    out: dict = {}
    for (track_id, part_id), h in sorted(headings.items()):
        counts = body_axis.coverage_counts(h["source"])
        counts["longest_gap_s"] = round(
            body_axis.longest_gap_s(h["t_ns"], h["source"]), 3
        )
        out.setdefault(f"track_{track_id}", {})[part_id] = counts
    return out


def largest_turned_mask_px(
    recording_dir, tracking_run_id: str, track_ids: list[int], settings: dict
) -> int:
    """The shared canvas for an aligned run: the largest **turned** mask box + 2 px over
    the stored masks (§14 *Canvas*) — the unaligned rule applied after turning, so aligned
    and unaligned frames draw the animal at about the same size. CPU only, no video decode.
    Cached against the tracking run, keyed on the keypoint run and the alignment settings.
    """
    recording_dir = Path(recording_dir)
    key = (
        "largest_turned_mask_px",
        str(recording_dir),
        tracking_run_id,
        tuple(sorted({int(t) for t in track_ids})),
        settings["keypoint_run_id"],
        repr(sorted((settings.get("heading_definition") or {}).items())),
        settings["heading_cutoff"],
        settings["facing_camera_ratio"],
    )
    stamp = recording_dir / "tracking" / tracking_run_id / "track.yaml"
    return read_cache.cached_read(
        key,
        stamp,
        lambda: _largest_turned_mask_px(
            recording_dir, tracking_run_id, track_ids, settings
        ),
    )


def _largest_turned_mask_px(
    recording_dir: Path, tracking_run_id: str, track_ids: list[int], settings: dict
) -> int:
    from detail.transforms.basic import turned_mask_extent

    tracking_dir = recording_dir / "tracking" / tracking_run_id
    track_meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    detection_dir = recording_dir / "detections" / track_meta["detection_run_id"]
    frame_h, frame_w = rundir.read_envelope(detection_dir)["mask_resolution"]
    # Every frame that could ever receive a heading, whatever gap a run fills: measured
    # with an unbounded gap, so the canvas is never short for a run with a longer
    # ``max_gap_s`` and the cache key need not carry it.
    unbounded = {**settings, "max_gap_s": float("inf"), "align": "rotate"}
    headings = headings_for_run(
        recording_dir, tracking_run_id, track_ids, 1, unbounded, (frame_h, frame_w)
    )
    largest = 0
    for det_part in sorted(detection_dir.glob("part_*.zarr")):
        reader = PartZarrReader(det_part)
        rles = np.asarray(reader.column("mask_rle")[:])
        candidates = _turned_box_bounds(
            headings,
            det_part.stem,
            track_ids,
            np.asarray(reader.column("frame_idx")[:]).astype(np.int64),
            np.asarray(
                PartZarrReader(tracking_dir / "assignments" / det_part.name).column(
                    "object_id"
                )[:]
            ).astype(np.int64),
            rles,
            (frame_h, frame_w),
        )
        # Largest bound first, so the pass stops at the first frame that cannot beat the
        # largest box so far. Decoding every mask instead cost a minute or more per
        # recording, every launch.
        candidates.sort(key=lambda c: c[0], reverse=True)
        for bound, deg, rows in candidates:
            if bound <= largest:
                break
            mask = decode_mask(str(rles[rows[0]]), (frame_h, frame_w))
            for extra in rows[1:]:
                mask = mask | decode_mask(str(rles[extra]), (frame_h, frame_w))
            largest = max(largest, *turned_mask_extent(mask, deg))
    if largest <= 0:
        raise ValueError(
            f"no frame of track_ids {sorted(set(int(t) for t in track_ids))} in tracking "
            f"run {tracking_run_id!r} has a heading; nothing to align"
        )
    return int(largest) + 2


def _turned_box_bounds(
    headings: dict[tuple[int, str], dict],
    part_id: str,
    track_ids: list[int],
    frame_col: np.ndarray,
    object_id: np.ndarray,
    rles: np.ndarray,
    size: tuple[int, int],
) -> list[tuple[int, float, list[int]]]:
    """``(bound, deg, rows)`` for each frame of one part where a track has a heading and a
    mask: the track's detection rows on that frame, the frame's heading, and a bound no
    side of the turned box can exceed. Turned by θ, the pixel centres of a ``w × h`` box
    span at most ``(w - 1)|cos θ| + (h - 1)|sin θ|`` along one axis and the same with the
    two swapped along the other; ``floor(span) + 2`` is at least ``turned_mask_extent``'s
    ``ceil(span) + 1``, float error included."""
    from detail.analysis import body_axis

    out: list[tuple[int, float, list[int]]] = []
    for track_id in track_ids:
        h = headings.get((int(track_id), part_id))
        if h is None:
            continue
        defined = np.asarray(h["source"]) != body_axis.SOURCE_NONE
        frames = np.asarray(h["frame_idx"], dtype=np.int64)[defined]  # sorted, unique
        degs = np.asarray(h["deg"], dtype=np.float64)[defined]
        rows = np.flatnonzero(object_id == int(track_id))
        rows = rows[rles[rows] != ""]  # no mask: an empty box, never the largest
        at = np.searchsorted(frames, frame_col[rows])
        hit = at < frames.size
        hit[hit] = frames[at[hit]] == frame_col[rows[hit]]
        rows, at = rows[hit], at[hit]
        if rows.size == 0:
            continue
        boxes = mask_boxes(rles[rows], size)
        heading_at, group = np.unique(at, return_inverse=True)
        x0 = np.full(heading_at.size, np.inf)
        y0 = np.full(heading_at.size, np.inf)
        x1 = np.zeros(heading_at.size)
        y1 = np.zeros(heading_at.size)
        np.minimum.at(x0, group, boxes[:, 0])
        np.minimum.at(y0, group, boxes[:, 1])
        np.maximum.at(x1, group, boxes[:, 0] + boxes[:, 2])
        np.maximum.at(y1, group, boxes[:, 1] + boxes[:, 3])
        across, down = x1 - x0 - 1, y1 - y0 - 1
        turn = np.radians(degs[heading_at])
        cos, sin = np.abs(np.cos(turn)), np.abs(np.sin(turn))
        span = np.maximum(across * cos + down * sin, across * sin + down * cos)
        bounds = np.floor(span).astype(np.int64) + 2
        order = np.argsort(group, kind="stable")
        starts = np.searchsorted(group[order], np.arange(heading_at.size + 1))
        for k, frame_at in enumerate(heading_at.tolist()):
            out.append(
                (
                    int(bounds[k]),
                    float(degs[frame_at]),
                    rows[order[starts[k] : starts[k + 1]]].tolist(),
                )
            )
    return out


def _max_contrast_rgb(mean_rgb: tuple[int, int, int]) -> list[int]:
    """In-gamut ``[R, G, B]`` maximizing CIELAB deltaE-76 from ``mean_rgb`` (a grid search).

    RGB because the renderer paints it onto RGB frames (``DetectionMaskDataset`` converts each
    decoded frame once), and the model reads RGB.

    cv2's LAB is uint8-packed (L on 0-255, a/b offset by 128); rescale to real CIELAB units
    (L on 0-100, a/b centered) before the Euclidean distance, or lightness dominates and the
    result skews bright instead of most-contrasting.
    """

    def true_lab(rgb) -> np.ndarray:
        px = np.array([[rgb]], dtype=np.uint8)
        lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float64)
        return np.array([lab[0] * 100.0 / 255.0, lab[1] - 128.0, lab[2] - 128.0])

    ref = true_lab(mean_rgb)
    levels = [int(round(v)) for v in np.linspace(0, 255, 9)]
    best, best_d = [0, 0, 0], -1.0
    for r in levels:
        for g in levels:
            for b in levels:
                d = float(np.linalg.norm(true_lab((r, g, b)) - ref))
                if d > best_d:
                    best_d, best = d, [r, g, b]
    return best


def contrast_bg_color(
    recording_dir,
    tracking_run_id: str,
    track_ids: list[int],
    *,
    n_sample: int = 30,
    seed: int = 0,
) -> list[int]:
    """Max-contrast background ``[R, G, B]`` for a run: the color farthest (CIELAB) from the
    animal's mean masked-pixel color.

    Cached against the tracking run's immutable ``track.yaml`` (same idiom as
    :func:`largest_mask_px`) — the color-sampling pass decodes frames, so this pays that cost
    once per (run, track set).
    """
    recording_dir = Path(recording_dir)
    # ``_rgb``: the cached value was BGR before the channel-order fix; a new key so a process
    # that cached one can never hand it to an RGB renderer.
    key = (
        "contrast_bg_color_rgb",
        str(recording_dir),
        tracking_run_id,
        tuple(sorted({int(t) for t in track_ids})),
    )
    stamp = recording_dir / "tracking" / tracking_run_id / "track.yaml"
    return read_cache.cached_read(
        key,
        stamp,
        lambda: _max_contrast_rgb(
            _mean_mask_color_rgb(
                recording_dir, tracking_run_id, track_ids, n_sample, seed
            )
        ),
    )


def _mean_mask_color_rgb(
    recording_dir: Path,
    tracking_run_id: str,
    track_ids: list[int],
    n_sample: int,
    seed: int,
) -> tuple[int, int, int]:
    # Reuse render-preview's candidate enumeration + frame/mask decode (preview-path
    # seeks are fine for a one-time cached statistic).
    from detail.pipeline.render_preview import _collect_candidates, _sample
    from detail.storage.video import read_frame_at

    data = recording_store.read_recording(recording_dir)
    tracking_dir = recording_dir / "tracking" / tracking_run_id
    track_meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    detection_dir = recording_dir / "detections" / track_meta["detection_run_id"]
    env = rundir.read_envelope(detection_dir)
    frame_h, frame_w = env["mask_resolution"]
    recording_store.raise_on_legacy_downsampling(env.get("job"))
    files = recording_store.video_files(data)
    requested = {int(t) for t in track_ids}
    candidates = _collect_candidates(
        detection_dir, tracking_dir, requested, track_meta.get("class_name")
    )
    if not candidates:
        raise ValueError(
            f"no detections for track_ids {sorted(requested)} in "
            f"tracking run {tracking_run_id!r}"
        )
    total = np.zeros(3, dtype=np.float64)
    count = 0
    for cand in _sample(candidates, n_sample, seed):
        tsmap = recording_store.load_active_timestamps_cached(
            recording_dir, data, cand.part_index
        )
        decoded = read_frame_at(
            recording_store.resolve_part_video_path(
                recording_dir, files[cand.part_index]
            ),
            cand.frame_idx,
            timestamps=tsmap.values,
        )
        if decoded is None:
            continue
        actual, frame_bgr = decoded
        if actual != cand.frame_idx:
            continue  # never sample a picture through another frame's mask
        mask = decode_mask(cand.mask_rles[0], (frame_h, frame_w))
        for extra in cand.mask_rles[1:]:
            mask = mask | decode_mask(extra, (frame_h, frame_w))
        # The same conversion the run's frames get, so the mean is in the order it is used in.
        pixels = bgr_to_rgb(frame_bgr)[mask]
        if pixels.size:
            total += pixels.sum(axis=0)
            count += pixels.shape[0]
    if count == 0:
        raise ValueError(
            f"could not decode any masked pixels for tracking run {tracking_run_id!r}"
        )
    mean = total / count
    return (int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))


def resolve_renderer_spec(
    recording_dir,
    tracking_run_id: str,
    track_ids: list[int],
    constructor: str,
    model_id: str | None = None,
    resize: int | None = None,
    input_size: int | None = None,
    canvas_px: int | None = None,
    bg_color: list[int] | None = None,
) -> list[dict]:
    """Resolve a named constructor (``transforms.constructors``) to a concrete
    ``renderer_spec``, computing the shared canvas from the data when needed.

    ``canvas_px`` / ``bg_color`` override the values this would otherwise derive from
    *this* recording alone. A multi-recording processing run resolves them once across
    the whole selection (``EmbeddingModule.resolve_shared_config``) and passes them in,
    so every recording gets an identical spec and the runs stay comparable. Omitted
    (the single-recording path), they are derived from this recording as before.

    The final square size the renderer emits follows this precedence: an explicit
    ``resize`` (advanced renderer-only override), else the run's ``input_size`` (the
    model-resolution knob, which also sets the model's processor size), else the
    embedding model's own ``input_size`` (via ``model_id``) — so by default the renderer
    hands the model exactly its input size instead of a large canvas the processor
    re-downscales every frame (§14). This keeps the GUI (which sends no ``resize``) on
    the same fast path as the CLI (which defaults to 224).

    Raises:
        ValueError: For an unknown constructor name.
    """
    recipe = CONSTRUCTORS.get(constructor)
    if recipe is None:
        raise ValueError(
            f"unknown constructor {constructor!r}; available: {sorted(CONSTRUCTORS)}"
        )
    effective_resize = resize if resize is not None else input_size
    if effective_resize is None and model_id is not None:
        model_cls = EMBEDDING_MODELS.get(model_id)
        if model_cls is None:
            # A caller (e.g. the CLI) may resolve the spec before the model modules are
            # imported, leaving EMBEDDING_MODELS empty. Silently dropping the resize step
            # here is harmless for a fixed-canvas constructor but yields variable-size
            # frames that break the batch collate for a variable-size one,
            # so populate the registry before reading input_size.
            from detail.api._runner_modules import PIPELINE_RUNNERS
            from detail.executor.runners import import_runner_modules

            import_runner_modules(PIPELINE_RUNNERS)
            model_cls = EMBEDDING_MODELS.get(model_id)
        if model_cls is not None:
            effective_resize = model_cls.input_size
    if recipe.needs_canvas and canvas_px is None:
        canvas_px = largest_mask_px(recording_dir, tracking_run_id, track_ids)
    elif not recipe.needs_canvas:
        canvas_px = None
    build_kwargs: dict = {"canvas_px": canvas_px, "resize": effective_resize}
    if recipe.needs_bg_color:
        build_kwargs["bg_color"] = (
            list(bg_color)
            if bg_color is not None
            else contrast_bg_color(recording_dir, tracking_run_id, track_ids)
        )
    return recipe.build(**build_kwargs)


def _detection_run_entry(recording_data: dict, detection_run_id: str) -> dict:
    """The detection run's ``recording.yaml`` record — the source of its cadence.

    Embedding reads the same source frames detection saw, so its own ``every_n_frames``
    is validated against this run's (§4). Refuses a run from the removed downsampling
    stage, whose rows index a video that no longer exists.
    """
    for run in recording_data.get("detection_runs", []):
        if run["id"] == detection_run_id:
            recording_store.raise_on_legacy_downsampling(run)
            return run
    raise ValueError(
        f"detection_run_id {detection_run_id!r} not found in recording.yaml"
    )


def _subject_labels(recording_dir, tracking_run_id: str) -> dict | None:
    """A tracking run's own ``subject_labels`` (§13: ``{track_id: animal_id}``), or
    ``None`` when the run never set one (every tracker but ``track_merge`` writes
    ``null``) — ``resolve_track_id``'s input, so a string track selector ("black
    mouse") resolves to *this recording's own* track id rather than an integer
    that only means "whichever the tracker numbered first" (Section 13)."""
    tracking_dir = Path(recording_dir) / "tracking" / tracking_run_id
    track_meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    return track_meta.get("subject_labels")


class DetectionMaskDataset:
    """Iterable of rendered images for one ``(track_id, part)``.

    When the renderer reads the video's pixels (``renderer.needs_frame`` — the
    masked-color constructors), each iteration opens one ``cv2.VideoCapture`` and
    walks it forward (``walk_frames``) — no seeking after open (Tenet 11). Frames
    between the track's detection rows are grabbed and discarded so the cursor stays
    sequential; the alternative (random seek per row) is forbidden outside preview
    paths.

    When the renderer reads no pixels (``needs_frame`` is False — the silhouette
    constructors build from ``ctx.mask`` alone), **no video is opened**: rows render
    straight from the detection masks. Decoding a video whose frames the renderer
    never touches is pure waste (§14).

    One rendered image per **frame**: a track may hold more than one detection in
    a frame (e.g. ``anchor_tracker`` secondaries — the detector split the subject
    across detections), so the masks sharing a frame are OR-merged into the
    subject silhouette. ``mask_rles_per_frame[i]`` is the list of RLE strings for
    ``frame_indices[i]``; a single-detection track has length-1 lists (the merge
    is then a no-op).

    Plain class (no ``torch`` base) so it's importable and testable without the
    torch backend. Iterating it renders serially; the worker instead runs
    ``walk_frames`` on one thread and ``_render_row`` on a pool (§14 *Execution*).
    """

    def __init__(
        self,
        video_path: Path,
        frame_indices: np.ndarray,
        t_ns: np.ndarray,
        det_rows: np.ndarray,
        mask_rles_per_frame: list[list[str]],
        mask_size: tuple[int, int],
        renderer: MaskRenderer,
        track_id: int,
        bboxes: list[tuple[float, float, float, float]] | None = None,
        aux_per_frame: list[dict] | None = None,
    ) -> None:
        order = np.argsort(frame_indices, kind="stable")
        self.video_path = Path(video_path)
        self.frame_indices = np.asarray(frame_indices)[order].astype(np.int64)
        self.t_ns = np.asarray(t_ns)[order].astype(np.int64)
        self.det_rows = np.asarray(det_rows)[order].astype(np.int64)
        self.mask_rles_per_frame = [mask_rles_per_frame[i] for i in order]
        # §14 alignment: per-frame ``heading_deg`` / ``align_mirrored`` the align transforms
        # read from ``ctx.aux``; empty dicts for an unaligned run.
        self.aux_per_frame = (
            [dict(aux_per_frame[i]) for i in order]
            if aux_per_frame is not None
            else [{} for _ in order]
        )
        # Per-frame union bbox (normalized xyxy) so crop_to_bbox crops the subject;
        # None per frame when unset (old callers) keeps the prior center-crop behavior.
        self.bboxes = (
            [bboxes[i] for i in order] if bboxes is not None else [None] * len(order)
        )
        self.mask_size = (int(mask_size[0]), int(mask_size[1]))
        self.renderer = renderer
        self.track_id = int(track_id)
        # §14 `kinematics/`: where the subject was on each frame, filled as rows render.
        # Collected here rather than recomputed later because this is the one place the
        # OR-merged mask exists — reconstructing it downstream from `det_row` resolves the
        # wrong part on a multi-part video and describes the largest *fragment* rather than
        # the animal. Keyed by frame index (unique: one row per frame), so it does not depend
        # on the order rows come back in, and both render paths write distinct keys from a
        # single process (the worker's render pool, or a serial iteration).
        self.geometry: dict[int, geometry.FrameGeometry] = {}

    def __len__(self) -> int:
        return int(len(self.frame_indices))

    def __iter__(self):
        if len(self.frame_indices) == 0:
            return
        if getattr(self.renderer, "needs_frame", True):
            yield from self._iter_with_frames()
        else:
            yield from self._iter_masks_only()

    def _iter_masks_only(self):
        """Frame-independent renderer: render from the mask, no video opened."""
        for i, target in enumerate(self.frame_indices.tolist()):
            yield self._render_row(i, int(target), frame=None)

    def _iter_with_frames(self):
        """Frame-dependent renderer, serially: each frame of the walk rendered as it is read.
        The reference the worker's pipelined path (`_embedding_worker._render_batches`) is
        held byte-identical to."""
        for i, target, frame in self.walk_frames():
            yield self._render_row(i, target, frame=frame)

    def walk_frames(self):
        """Yield ``(i, frame_idx, rgb_frame)`` for every row, from one forward cv2 walk
        (Tenet 11). The only reader of the video: the worker runs this on its own thread and
        renders the frames elsewhere.

        Frames between rows are skipped with ``grab()``: the decoder still decodes them (an
        H.264 frame can be a reference for the next), but they are never converted to BGR —
        the same walk detection and keypoint detection make.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            cur = 0
            for i, target in enumerate(self.frame_indices.tolist()):
                while cur < target:
                    if not cap.grab():
                        raise RuntimeError(
                            f"video {self.video_path} ended at frame {cur} "
                            f"before reaching target frame {target}"
                        )
                    cur += 1
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"failed to read frame {target} from {self.video_path}"
                    )
                cur += 1
                # cv2 decodes BGR; `RenderContext.frame` and every model read RGB (§14). The
                # one conversion, here, covers the worker and the in-process path alike.
                yield i, int(target), bgr_to_rgb(frame)
        finally:
            cap.release()

    def _render_row(self, i: int, target: int, frame) -> dict:
        rles = self.mask_rles_per_frame[i]
        mask = decode_mask(rles[0], self.mask_size)
        for extra in rles[1:]:  # OR-merge a frame's detections (anchor secondaries)
            mask = mask | decode_mask(extra, self.mask_size)
        box = self.bboxes[i]
        self.geometry[target] = geometry.frame_geometry(
            mask, [] if box is None else [box], self.mask_size
        )
        ctx = RenderContext(
            frame=frame,
            mask=mask,
            bbox=self.bboxes[i],
            image=None,
            frame_idx=target,
            t_ns=int(self.t_ns[i]),
            class_name="",
            confidence=0.0,
            object_id=self.track_id,
            aux=dict(self.aux_per_frame[i]),
        )
        return {
            "image": self.renderer.render(ctx),
            "frame_idx": target,
            "t_ns": int(self.t_ns[i]),
            "det_row": int(self.det_rows[i]),
        }


# --------------------------------------------------------------------------- #
# Top-level runner
# --------------------------------------------------------------------------- #
def run(job: EmbeddingJob, run_dir, progress_cb, stop_event) -> None:
    run_dir = Path(run_dir)
    recording_dir = run_dir.parent.parent
    recording_data = recording_store.read_recording(recording_dir)
    tracking_dir = recording_dir / "tracking" / job.tracking_run_id
    track_meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    detection_run_id = track_meta["detection_run_id"]
    detection_dir = recording_dir / "detections" / detection_run_id
    frame_h, frame_w = rundir.read_envelope(detection_dir)["mask_resolution"]

    _detection_run_entry(recording_data, detection_run_id)  # refuses a legacy ds run
    source_files = recording_store.video_files(recording_data)

    model_cls = EMBEDDING_MODELS.get(job.model_id)
    if model_cls is None:
        raise ValueError(
            f"unknown model_id {job.model_id!r}; available: {sorted(EMBEDDING_MODELS)}"
        )

    progress_dir = run_dir / _embedding_worker.PROGRESS_DIRNAME
    progress_dir.mkdir(parents=True, exist_ok=True)
    stop_flag = run_dir / _embedding_worker.STOP_FLAG_NAME

    det_parts = sorted(detection_dir.glob("part_*.zarr"))
    units = _build_units(
        job=job,
        det_parts=det_parts,
        tracking_dir=tracking_dir,
        recording_dir=recording_dir,
        source_files=source_files,
        run_dir=run_dir,
        progress_dir=progress_dir,
        stop_flag=stop_flag,
        mask_size=(frame_h, frame_w),
    )

    device_indices = _resolve_device_indices(job)
    if _is_multi_gpu(job, device_indices):
        results, interrupted = _run_multi_gpu(
            job=job,
            units=units,
            device_indices=device_indices,
            stop_flag=stop_flag,
            progress_cb=progress_cb,
            stop_event=stop_event,
        )
    else:
        results, interrupted = _run_inline(
            job=job,
            units=units,
            stop_flag=stop_flag,
            progress_cb=progress_cb,
            stop_event=stop_event,
        )

    # output_dim is authoritative from the worker (model.output_dim is set
    # by load(), which only runs in the worker for the multi-GPU path).
    output_dim = _first_output_dim(results)
    coverage = None
    if job.align != "none":
        coverage = heading_coverage(
            {
                (u["track_id"], u["part_id"]): {
                    "t_ns": np.asarray(u["heading"]["t_ns"], dtype=np.int64),
                    "source": np.asarray(u["heading"]["source"], dtype=np.int8),
                }
                for u in units
                if "heading" in u and "t_ns" in u["heading"]
            }
        )
    _write_embedding_yaml(
        run_dir / "embedding.yaml",
        job,
        output_dim,
        "partial" if interrupted else "complete",
        results=results,
        heading_coverage=coverage,
    )
    _raise_if_any_failed(results)


# --------------------------------------------------------------------------- #
# Work-unit construction
# --------------------------------------------------------------------------- #
def _build_units(
    *,
    job: EmbeddingJob,
    det_parts: list[Path],
    tracking_dir: Path,
    recording_dir: Path,
    source_files: list[dict],
    run_dir: Path,
    progress_dir: Path,
    stop_flag: Path,
    mask_size: tuple[int, int],
) -> list[dict]:
    """One unit per (track_id, part_index). Flat list — ordered so the inline path stays
    sequential per series (matching the existing progress contract) and the multi-GPU
    dispatcher can round-robin freely."""
    track_meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    class_name = track_meta.get("class_name")
    settings = _alignment_settings(job)
    # The heading is read for alignment, and for the pose numbers (``append`` turns them by it);
    # only an aligned run hands it to the worker, which drops a frame with none.
    headings = (
        headings_for_run(
            recording_dir,
            job.tracking_run_id,
            list(job.track_ids),
            int(job.every_n_frames or 1),
            settings,
            mask_size,
        )
        if settings["align"] != "none" or settings["pose"] == "append"
        else None
    )
    poses = (
        poses_for_run(
            recording_dir,
            job.tracking_run_id,
            list(job.track_ids),
            int(job.every_n_frames or 1),
            settings,
            mask_size,
            headings,
        )
        if settings["pose"] != "none"
        else None
    )
    units: list[dict] = []
    for track_id in job.track_ids:
        for part_index, part_path in enumerate(det_parts):
            part_id = part_path.stem
            assignment_path = tracking_dir / "assignments" / part_path.name
            video_path = recording_store.resolve_part_video_path(
                recording_dir, source_files[part_index]
            )
            out_zarr = run_dir / f"track_{track_id}" / part_path.name
            progress_json = progress_dir / _embedding_worker.progress_json_name(
                track_id, part_id
            )
            # Unique frames this track occupies in the part = the worker's
            # frames_total for the unit (detections sharing a frame OR-merge to
            # one image). Pre-counted here so the bar has a real run-wide frame
            # denominator from the first event, not a per-unit sentinel.
            assignment_reader = PartZarrReader(assignment_path)
            det_reader = PartZarrReader(part_path)
            raw_rows = resolve_assignment_rows(
                det_reader, assignment_reader, class_name
            )
            n = len(raw_rows)
            object_id = assignment_reader.column("object_id")[:n]
            frame_idx = np.asarray(det_reader.column("frame_idx")[: det_reader.n_rows])[
                raw_rows
            ]
            sel = np.asarray(object_id).astype(np.int64) == int(track_id)
            n_frames = int(np.unique(np.asarray(frame_idx)[sel]).size)
            unit = {
                "track_id": int(track_id),
                "part_index": int(part_index),
                "part_id": part_id,
                "video_path": video_path,
                "det_zarr_path": part_path,
                "assignment_zarr_path": assignment_path,
                "mask_size": tuple(mask_size),
                "out_zarr_path": out_zarr,
                "progress_json_path": progress_json,
                "stop_flag_path": stop_flag,
                "n_frames": n_frames,
                "class_name": class_name,
            }
            if poses is not None:
                p = poses[(int(track_id), part_id)]
                unit["pose"] = (
                    {  # lists, like the heading: the unit rides a process pool
                        "mode": settings["pose"],
                        "weight": settings["pose_weight"],
                        "frame_idx": p["frame_idx"].tolist(),
                        "xy": p["xy"].tolist(),
                        "score": p["score"].tolist(),
                        "vec": None if p["vec"] is None else p["vec"].tolist(),
                        "n_visible": [int(v) for v in p["n_visible"]],
                    }
                )
            if headings is not None and settings["align"] != "none":
                from detail.analysis import body_axis

                h = headings[(int(track_id), part_id)]
                unit["heading"] = {  # lists: the unit rides a process pool as a dict
                    "frame_idx": h["frame_idx"].tolist(),
                    "t_ns": h["t_ns"].tolist(),
                    "deg": [float(v) for v in h["deg"]],
                    "source": [int(v) for v in h["source"]],
                    "mirrored": [bool(v) for v in h["mirrored"]],
                }
                unit["n_frames"] = int((h["source"] != body_axis.SOURCE_NONE).sum())
            units.append(unit)
    return units


# --------------------------------------------------------------------------- #
# Inline (single-device) path
# --------------------------------------------------------------------------- #
def _run_inline(
    *,
    job: EmbeddingJob,
    units: list[dict],
    stop_flag: Path,
    progress_cb,
    stop_event,
) -> tuple[list[dict], bool]:
    parts_total = _parts_total(units)
    units_total = len(units)
    frames_total = _frames_total(units)
    start = time.monotonic()

    def _emit(track_id, results, *, live_frames=0, message=None):
        if progress_cb is not None:
            progress_cb(
                _aggregate_event_for_track(
                    track_id=track_id,
                    results=results,
                    parts_total=parts_total,
                    units_total=units_total,
                    frames_total=frames_total,
                    start=start,
                    live_frames=live_frames,
                    message=message,
                )
            )

    if units:  # bar at 0% with "loading model" during the (slow) weight load
        _emit(units[0]["track_id"], [], message="loading model")

    model_cls = EMBEDDING_MODELS[job.model_id]
    model = construct_embedding_model(
        model_cls, job.model_variant, readout=job.readout, input_size=job.input_size
    )
    model.load(resolve_device(job.compute, model.supported_backends))
    renderer = MaskRenderer.from_spec(job.renderer_spec)
    output_dim = int(model.output_dim)

    results: list[dict] = []
    interrupted = False
    for unit in units:
        if stop_event is not None and stop_event.is_set():
            interrupted = True
            break
        # Run the unit in a thread so we can poll its per-batch JSON snapshot and
        # stream a live frame count — otherwise a single track/part shows nothing
        # until it finishes.
        result_holder: dict[str, dict] = {}

        def _work(unit=unit):
            result_holder["result"] = _embedding_worker.embed_one(
                track_id=unit["track_id"],
                part_index=unit["part_index"],
                part_id=unit["part_id"],
                video_path=unit["video_path"],
                det_zarr_path=unit["det_zarr_path"],
                assignment_zarr_path=unit["assignment_zarr_path"],
                mask_size=unit["mask_size"],
                out_zarr_path=unit["out_zarr_path"],
                progress_json_path=unit["progress_json_path"],
                stop_flag_path=unit["stop_flag_path"],
                model=model,
                renderer=renderer,
                output_dim=output_dim,
                batch_size=int(job.batch_size),
                every_n_frames=int(getattr(job, "every_n_frames", 1) or 1),
                gpu_idx=None,
                class_name=unit.get("class_name"),
                heading=unit.get("heading"),
                pose=unit.get("pose"),
            )

        worker = threading.Thread(target=_work)
        worker.start()
        while worker.is_alive():
            worker.join(_PROGRESS_POLL_INTERVAL_S)
            snapshot = _embedding_worker.read_progress_json(
                Path(unit["progress_json_path"])
            )
            live_frames = int(snapshot.get("frames_done", 0)) if snapshot else 0
            _emit(unit["track_id"], results, live_frames=live_frames)
        results.append(result_holder["result"])
        _emit(unit["track_id"], results)
    return results, interrupted


# --------------------------------------------------------------------------- #
# Multi-GPU pool path
# --------------------------------------------------------------------------- #
def _run_multi_gpu(
    *,
    job: EmbeddingJob,
    units: list[dict],
    device_indices: list[int],
    stop_flag: Path,
    progress_cb,
    stop_event,
) -> tuple[list[dict], bool]:
    if EMBEDDING_MODELS.get(job.model_id) is None:
        raise ValueError(
            f"unknown model_id {job.model_id!r}; available: {sorted(EMBEDDING_MODELS)}"
        )

    from detail.api._runner_modules import PIPELINE_RUNNERS

    n_workers = len(device_indices)
    mp_ctx = mp.get_context("spawn")
    parts_total = _parts_total(units)
    frames_total = _frames_total(units)
    results: list[dict] = []
    start = time.monotonic()

    # One pool per GPU (max_workers=1) — same reason as detection: a single
    # initializer can't pin different workers to different GPUs.
    pools: list[ProcessPoolExecutor] = []
    try:
        for gpu_idx in device_indices:
            pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp_ctx,
                initializer=_embedding_worker.pool_initializer,
                initargs=(
                    int(gpu_idx),
                    job.model_id,
                    job.model_variant,
                    list(job.renderer_spec),
                    PIPELINE_RUNNERS,
                    job.readout,
                    job.input_size,
                    n_workers,  # the render pools share this machine's cores
                ),
            )
            pools.append(pool)

        futures = {}
        for unit_index, unit in enumerate(units):
            assigned = unit_index % n_workers
            args = {
                "track_id": unit["track_id"],
                "part_index": unit["part_index"],
                "part_id": unit["part_id"],
                "video_path": str(unit["video_path"]),
                "det_zarr_path": str(unit["det_zarr_path"]),
                "assignment_zarr_path": str(unit["assignment_zarr_path"]),
                "mask_size": list(unit["mask_size"]),
                "out_zarr_path": str(unit["out_zarr_path"]),
                "progress_json_path": str(unit["progress_json_path"]),
                "stop_flag_path": str(unit["stop_flag_path"]),
                "batch_size": int(job.batch_size),
                # Must travel in the args dict: the pool worker rebuilds the call from it,
                # so a job field that stops here is silently dropped on the multi-GPU path
                # only — which is exactly what happened to the retired temporal stack, and
                # to this unit's own class_name before it was added here.
                "every_n_frames": int(getattr(job, "every_n_frames", 1) or 1),
                "class_name": unit.get("class_name"),
                "heading": unit.get("heading"),  # the same: rebuilt from this dict
                "pose": unit.get("pose"),  # and this one
            }
            fut = pools[assigned].submit(_embedding_worker.pool_run_one, args)
            futures[fut] = unit_index

        pending = set(futures)
        if progress_cb is not None and units:
            progress_cb(
                _aggregate_event_for_track(
                    track_id=units[0]["track_id"],
                    results=results,
                    parts_total=parts_total,
                    units_total=len(units),
                    frames_total=frames_total,
                    start=start,
                    message="loading model",
                )
            )
        while pending:
            if (
                stop_event is not None
                and stop_event.is_set()
                and not stop_flag.exists()
            ):
                stop_flag.touch()
            done, pending = _wait_for_completion(pending)
            for fut in done:
                result = fut.result()
                results.append(result)
                if progress_cb is not None:
                    progress_cb(
                        _aggregate_event_for_track(
                            track_id=result["track_id"],
                            results=results,
                            parts_total=parts_total,
                            units_total=len(units),
                            frames_total=frames_total,
                            start=start,
                        )
                    )
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)

    interrupted = stop_event is not None and stop_event.is_set()
    results.sort(key=lambda r: (r["track_id"], r["part_index"]))
    return results, interrupted


def _wait_for_completion(pending):
    from concurrent.futures import wait, FIRST_COMPLETED

    return wait(pending, timeout=_PROGRESS_POLL_INTERVAL_S, return_when=FIRST_COMPLETED)


# --------------------------------------------------------------------------- #
# Decisions + helpers
# --------------------------------------------------------------------------- #
def _resolve_device_indices(job: EmbeddingJob) -> list[int]:
    if job.compute.backend != "cuda":
        return []
    indices = job.compute.device_indices
    if indices is None or len(indices) == 0:
        return [0]
    return list(indices)


def _is_multi_gpu(job: EmbeddingJob, device_indices: list[int]) -> bool:
    return job.compute.backend == "cuda" and len(device_indices) > 1


def _parts_total(units: list[dict]) -> dict[int, int]:
    """parts_total per track_id — invariant across the run."""
    out: dict[int, int] = {}
    for u in units:
        out[u["track_id"]] = out.get(u["track_id"], 0) + 1
    return out


def _frames_total(units: list[dict]) -> int:
    """Run-wide unique-frame count (the bar's real denominator) = sum of each
    unit's pre-counted ``n_frames``."""
    return sum(int(u.get("n_frames", 0)) for u in units)


def _aggregate_event_for_track(
    *,
    track_id: int,
    results: list[dict],
    parts_total: dict[int, int],
    units_total: int,
    frames_total: int,
    start: float,
    live_frames: int = 0,
    message: str | None = None,
) -> EmbeddingProgressEvent:
    track_results = [r for r in results if r["track_id"] == track_id]
    parts_done = sum(1 for r in track_results if r.get("status") == "complete")
    elapsed = time.monotonic() - start
    # Run-wide frame progress against the pre-counted total — completed units'
    # frames plus the in-flight unit's live count — so the bar shows a real
    # %/ETA even for a single (track, part) unit. track_id / parts_* stay
    # per-track for the postfix's "where am I" context.
    frames_done = sum(int(r.get("frames_done", 0)) for r in results) + live_frames
    eta_s = (
        remaining_seconds(frames_done / frames_total, elapsed)
        if frames_total > 0
        else None
    )
    return EmbeddingProgressEvent(
        track_id=track_id,
        parts_done=parts_done,
        parts_total=int(parts_total.get(track_id, 0)),
        frames_done=frames_done,
        frames_total=int(frames_total),
        elapsed_s=elapsed,
        eta_s=eta_s,
        units_done=len(results),
        units_total=units_total,
        message=message,
    )


def _first_output_dim(results: list[dict]) -> int:
    """Workers carry the loaded model's ``output_dim`` in each result dict —
    the multi-GPU coordinator never loads the model itself, so the worker is
    the only source of truth."""
    for r in results:
        d = int(r.get("output_dim", 0))
        if d > 0:
            return d
    return 0


def _any_failed(results: list[dict]) -> bool:
    return any(r.get("status") == "failed" for r in results)


def _raise_if_any_failed(results: list[dict]) -> None:
    """Embedding treats any failed unit (stop, OOM, model exception) as a
    failed run — surfaces to the executor envelope as ``status="failed"``."""
    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        names = ", ".join(
            f"track_{r['track_id']}/{r['part_id']}({r.get('error')})" for r in failed
        )
        raise RuntimeError(f"embedding failed on {len(failed)} unit(s): {names}")


# --------------------------------------------------------------------------- #
# embedding.yaml
# --------------------------------------------------------------------------- #
def _pose_dim(job: EmbeddingJob, results: list[dict] | None) -> int | None:
    """The width of the appended pose block (2 per keypoint), or None without ``append``."""
    if job.pose != "append":
        return None
    for r in results or []:
        if int(r.get("pose_dim", 0)) > 0:
            return int(r["pose_dim"])
    return None


def _mask_dim(job: EmbeddingJob, results: list[dict] | None) -> int | None:
    """The width of the mask's own part of an appended vector, or None without ``append``."""
    pose_dim = _pose_dim(job, results)
    if pose_dim is None:
        return None
    return _first_output_dim(results or []) - pose_dim


def _write_embedding_yaml(
    path,
    job: EmbeddingJob,
    output_dim: int,
    status: str,
    results: list[dict] | None = None,
    heading_coverage: dict | None = None,
) -> None:
    """Written once, at run completion — but the executor subprocess and the GUI's
    poll for "has this run finished" are different processes, so a plain truncating
    write is still visible to a reader mid-write. tmp + ``atomic.replace``, per the
    atomic-write hard rule (CLAUDE.md)."""
    data = {
        "schema_version": SCHEMA_VERSION,
        "id": job.embedding_run_id,
        "created_at": job.created_at,
        "tracking_run_id": job.tracking_run_id,
        "model_id": job.model_id,
        "model_variant": job.model_variant,
        "constructor": job.constructor,
        "readout": job.readout,
        "input_size": job.input_size,
        "output_dim": output_dim,
        "track_ids": list(job.track_ids),
        "canvas_scope": job.canvas_scope,
        "bg_color_scope": job.bg_color_scope,
        "channel_order": CHANNEL_ORDER,
        "renderer_spec": job.renderer_spec,
        "batch_size": job.batch_size,
        "compute": job.compute.to_dict(),
        # §14 alignment: always written, so an unaligned run says so rather than being
        # read as "written before alignment existed".
        "align": job.align,
        "keypoint_run_id": job.keypoint_run_id,
        "heading_definition": job.heading_definition,
        "heading_cutoff": job.heading_cutoff,
        "max_gap_s": job.max_gap_s,
        "mirror_hold_s": job.mirror_hold_s,
        "facing_camera_ratio": job.facing_camera_ratio,
        "heading_coverage": heading_coverage or {},
        # §14 pose in the embedding. For ``append`` the vector is the mask block followed by
        # the pose block: ``mask_dim`` + ``pose_dim`` == ``output_dim``, so a reader can split it.
        "pose": job.pose,
        "pose_weight": job.pose_weight,
        "pose_min_score": job.pose_min_score,
        "mask_dim": _mask_dim(job, results),
        "pose_dim": _pose_dim(job, results),
        "status": status,
    }
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
    atomic.replace(tmp, path)


def _shared_inputs(
    experiment_dir,
    recording_id: str,
    module,
    config: dict,
    want_canvas: bool,
    want_mean: bool,
) -> tuple[int | None, tuple[int, int, int] | None]:
    """One recording's contribution to the parameters resolved across the selection: its
    largest mask (when the canvas is ``shared``) and its mean masked color (when a
    max-contrast background is being pooled).

    Resolves the recording's own tracking run + tracks exactly as ``validate`` does, so
    a batch launched by ``--tracking-ref`` (a different run id per recording) and
    ``all_tracks`` both resolve to the same tracks the run will embed.
    """
    recording_dir = experiment.recording_path(experiment_dir, recording_id)
    data = recording_store.read_recording(recording_dir)
    resolved = module.resolve_upstream_spec(recording_dir, data, dict(config))
    tracking_run_id = resolved["tracking_run_id"]
    if resolved.get("all_tracks"):
        tracks_path = Path(recording_dir) / "tracking" / tracking_run_id / "tracks.zarr"
        track_ids = sorted(
            {int(t) for t in PartZarrReader(tracks_path).read()["track_id"]}
        )
    else:
        # A selector may still be a name (§13/§14) at this point — resolve_shared_config
        # runs before validate() resolves names to this recording's own object ids, so a
        # name-based selection must be resolved here too, the same way, or this crashes
        # the moment a shared-scoped canvas meets a name selector.
        selectors = list(resolved["track_ids"])
        subject_labels = (
            _subject_labels(recording_dir, tracking_run_id)
            if any(isinstance(t, str) for t in selectors)
            else None
        )
        track_ids = [int(resolve_track_id(t, subject_labels)) for t in selectors]
    if not want_canvas:
        px = None
    elif (resolved.get("align") or "none") != "none":
        # The aligned canvas is sized from the turned masks (§14), with the same resolution
        # of the heading definition and cut-off `validate` performs for the run itself.
        settings = _alignment_settings(
            _resolve_alignment(recording_dir, data, dict(resolved), track_ids)
        )
        px = largest_turned_mask_px(recording_dir, tracking_run_id, track_ids, settings)
    else:
        px = largest_mask_px(recording_dir, tracking_run_id, track_ids)
    mean = (
        _mean_mask_color_rgb(Path(recording_dir), tracking_run_id, track_ids, 30, 0)
        if want_mean
        else None
    )
    return px, mean


_ALIGNMENT_DEFAULTS = {
    "keypoint_run_id": None,
    "heading": None,
    "heading_definition": None,
    "heading_cutoff": None,
    "keypoint_track_identity": None,
    "max_gap_s": 0.5,
    "mirror_hold_s": 1.0,
    "facing_camera_ratio": 0.5,
}


def _draw_pose_step(recording_dir, config: dict) -> dict | None:
    """The ``draw_pose`` step for a ``pose: draw`` run, else None: the keypoint track run's own
    skeleton and the resolved floor, so the run records exactly what it drew (§14)."""
    if config.get("pose") != "draw":
        return None
    from detail.storage.keypoint_runs import KeypointTrackRun

    skeleton = KeypointTrackRun(
        Path(recording_dir), config["keypoint_run_id"]
    ).skeleton.to_dict()
    skeleton.pop("layout", None)  # how the editor drew it, not what the points mean
    step: dict = {"type": "draw_pose", "skeleton": skeleton}
    if config.get("pose_min_score") is not None:
        step["min_score"] = float(config["pose_min_score"])
    return step


def point_cutoff(track_run, detect) -> float | None:
    """The score under which a point does not move the heading mean (§14: "at or above the
    track run's decode cut-off"). A track method that chooses among candidates carries
    ``decode_cutoff`` in its config, ``None`` included (dinov3_match's scores are its own,
    and a cut-off of none is a faithful reading). A method with no such field — a student's
    ``despike_smooth`` — falls back to the detect method's display cut-off
    (``score_spec.default_cutoff``), so a point at confidence 0.05 does not pull the axis.
    """
    track_config = track_run.meta.get("config") or {}
    if "decode_cutoff" in track_config:
        value = track_config["decode_cutoff"]
        return None if value is None else float(value)
    from detail.models.resolve import keypoint_method_class

    model_cls = keypoint_method_class(
        detect.recording_dir.parent.parent, detect.meta.get("model_id")
    )
    spec = getattr(model_cls, "score_spec", None)
    cutoff = getattr(spec, "default_cutoff", None)
    return None if cutoff is None else float(cutoff)


#: A run without pose records these, so two runs that drew nothing never differ in a setting
#: that did nothing (the alignment fields' rule, `_ALIGNMENT_DEFAULTS`).
_POSE_DEFAULTS = {"pose": "none", "pose_weight": None, "pose_min_score": None}
POSE_MODES = ("none", "draw", "append")


def _resolve_alignment(
    recording_dir, recording_data: dict, config: dict, track_ids: list[int]
) -> dict:
    """§14's pre-run checks for what an embedding reads from a keypoint track run — the
    heading it aligns to and/or the pose it puts into the vector — resolving what the run
    records: the heading definition copied in, the track run's decode cut-off, the pose floor
    and weight, and the keypoint track run's identity. With ``align`` and ``pose`` both
    ``none`` every such field is cleared instead, so two runs that read nothing never differ
    in a setting that did nothing.

    The heading is needed to align and for the pose *numbers* (``append``: they are turned so
    the heading points right); ``draw`` paints the points as placed and needs none. Pose needs
    a row on every frame the run embeds, so the detect run's cadence must divide this run's: a
    frame it never placed points on would be embedded with no pose, quietly, among frames that
    have it.

    Raises:
        ValueError: naming the missing or mismatched input (the table in §14).
    """
    from detail.api import run_spec
    from detail.analysis import body_axis
    from detail.storage import heading_definitions
    from detail.storage.keypoint_runs import KeypointDetectRun, KeypointTrackRun

    recording_dir = Path(recording_dir)
    align = config.get("align") or "none"
    pose = config.get("pose") or "none"
    if align not in body_axis.ALIGN_MODES:
        raise ValueError(f"align must be one of {body_axis.ALIGN_MODES}, got {align!r}")
    if pose not in POSE_MODES:
        raise ValueError(f"pose must be one of {POSE_MODES}, got {pose!r}")
    if align == "none" and pose == "none":
        if config.get("keypoint_run_id") or config.get("heading"):
            raise ValueError(
                "a keypoint track run or a heading definition was given with align and "
                "pose both 'none'; set --align rotate or mirror, or --pose draw or append, "
                "or drop them"
            )
        config["align"] = "none"
        config.update(_ALIGNMENT_DEFAULTS)
        config.update(_POSE_DEFAULTS)
        return config
    what = " and ".join(
        part
        for part in (
            f"align {align!r}" if align != "none" else "",
            f"pose {pose!r}" if pose != "none" else "",
        )
        if part
    )
    needs_heading = align != "none" or pose == "append"
    keypoint_run_id = config.get("keypoint_run_id")
    heading_name = config.get("heading")
    if not keypoint_run_id:
        raise ValueError(
            f"{what} reads a keypoint track run; give keypoint_run_id (a kt_NN on this "
            "embedding's tracking run)"
        )
    if needs_heading and not heading_name:
        raise ValueError(
            f"{what} needs a heading definition (heading): "
            + (
                "the image is turned to it"
                if align != "none"
                else "the pose numbers are turned so it points right"
            )
        )
    if heading_name and not needs_heading:
        raise ValueError(
            f"heading {heading_name!r} is given but nothing reads it: pose 'draw' paints "
            "the points as placed; drop the heading, or set --align or --pose append"
        )
    entry = {r["id"]: r for r in recording_data.get("keypoint_track_runs", [])}.get(
        keypoint_run_id
    )
    if entry is None:
        raise ValueError(
            f"keypoint track run {keypoint_run_id!r} not found in recording "
            f"{recording_dir.name!r}"
        )
    if entry.get("status") != "complete":
        raise ValueError(
            f"keypoint track run {keypoint_run_id!r} is {entry.get('status')!r}, not "
            "complete"
        )
    teacher_window.refuse_teacher_run(
        entry,
        consumer=f"an embedding with {what}",
        upstream=f"keypoint track run {keypoint_run_id!r}",
    )
    track_run = KeypointTrackRun(recording_dir, keypoint_run_id)
    detect = KeypointDetectRun(recording_dir, track_run.meta["detect_run_id"])
    # §14 *Lineage*: a keypoint track run's subjects are the tracking run it assigned its
    # instances to (§26 *Assignment*, recorded in its track.yaml since 2026-10-21); the detect
    # run under it knows no subject.
    if str(track_run.meta["tracking_run_id"]) != str(config["tracking_run_id"]):
        raise ValueError(
            f"keypoint track run {keypoint_run_id!r} assigned its poses to tracking run "
            f"{track_run.meta['tracking_run_id']!r}, not this embedding's "
            f"{config['tracking_run_id']!r}; its points must come from the same tracks"
        )
    missing_tracks = sorted(set(int(t) for t in track_ids) - set(track_run.track_ids))
    if missing_tracks:
        raise ValueError(
            f"keypoint track run {keypoint_run_id!r} has no rows for track_ids "
            f"{missing_tracks}; it covers {track_run.track_ids}"
        )
    if needs_heading:
        experiment_dir = recording_dir.parent.parent
        definition = heading_definitions.get_definition(
            experiment_dir, str(heading_name)
        )
        checked = body_axis.validate_definition(
            definition, tuple(track_run.skeleton.names)
        )
        config["heading_definition"] = {
            "front": checked["front"],
            "back": checked["back"],
        }
        config["heading_cutoff"] = point_cutoff(track_run, detect)
        config["max_gap_s"] = float(config.get("max_gap_s", 0.5))
        config["mirror_hold_s"] = float(config.get("mirror_hold_s", 1.0))
        config["facing_camera_ratio"] = float(config.get("facing_camera_ratio", 0.5))
    else:
        config.update(
            heading=None,
            heading_definition=None,
            heading_cutoff=None,
            max_gap_s=_ALIGNMENT_DEFAULTS["max_gap_s"],
            mirror_hold_s=_ALIGNMENT_DEFAULTS["mirror_hold_s"],
            facing_camera_ratio=_ALIGNMENT_DEFAULTS["facing_camera_ratio"],
        )
    if align != "none":
        constructor = config.get("constructor")
        if config.get("renderer_spec") is None and constructor not in ALIGNABLE:
            raise ValueError(
                f"input image {constructor!r} does not crop to the subject, so it cannot "
                f"be aligned; use one of {', '.join(ALIGNABLE)}"
            )
    if pose != "none":
        detect_entry = {
            r["id"]: r for r in recording_data.get("keypoint_detect_runs", [])
        }.get(track_run.meta["detect_run_id"], {})
        kp_every = int(
            detect_entry.get("every_n_frames") or detect.meta.get("every_n_frames") or 1
        )
        every = int(config.get("every_n_frames") or 1)
        if every % kp_every:
            raise ValueError(
                f"keypoint track run {keypoint_run_id!r} has points on every {kp_every} "
                f"frame(s), and this run embeds every {every}; pose needs points on every "
                f"embedded frame, so use a multiple of {kp_every} for 'every N frames'"
            )
        floor = config.get("pose_min_score")
        config["pose_min_score"] = (
            float(floor) if floor is not None else point_cutoff(track_run, detect)
        )
        config["pose_weight"] = (
            float(config.get("pose_weight") or 1.0) if pose == "append" else None
        )
    else:
        config.update(pose_weight=None, pose_min_score=None)
    config["align"] = align
    config["pose"] = pose
    config["keypoint_track_identity"] = run_spec.spec_of(
        recording_data, "keypoint_track", entry, recording_dir
    )
    return config


def _resolve_defaults(config: dict) -> None:
    """Write the model's default ``model_id`` and ``constructor`` back into ``config``.

    Resolved here so the run record names the model and input image that actually ran
    (Tenet 9): ``model_validate`` fills the defaults on an instance that is discarded, and
    everything after it reads the raw dict. The constructor is resolved only when no
    ``renderer_spec`` is given: an explicit spec pins the input image itself, and recording a
    constructor name beside it would make ``validate_comparability`` treat a pinned spec as a
    recipe. Its key is still set (to ``None``) so the experiment's default-filling does not add
    the default name to the stored record afterwards.
    """
    fields = EmbeddingConfig.model_fields
    config["model_id"] = config.get("model_id") or fields["model_id"].default
    if config.get("renderer_spec") is not None:
        config.setdefault("constructor", None)
    else:
        config["constructor"] = (
            config.get("constructor") or fields["constructor"].default
        )


@register_processing_module
class EmbeddingModule(ProcessingModule):
    """Embedding stage (§14) as a processing module (§6.5)."""

    module_id = "embedding"
    # §14 alignment joins the identity: the mode, the resolved heading definition, the three
    # settings, and the keypoint track run's *identity* (never its id). The defaults make a
    # record written before alignment existed compare equal to an explicit ``none``, the same
    # way a missing every_n_frames means stride 1.
    identity_fields = (
        "model_id",
        "model_variant",
        "constructor",
        "every_n_frames",
        # Inert in this build, and kept on purpose: every embedding run recorded before it
        # states `joint: false` / `joint_subjects: null`, and so does every spec a saved config
        # carries (Start from, a stored selector). Specs compare by exact equality
        # (`run_spec.freeze_spec`), so dropping the two fields would stop those saved specs
        # matching the very runs they were made from.
        "joint",
        "joint_subjects",
        "align",
        "heading_definition",
        "max_gap_s",
        "mirror_hold_s",
        "facing_camera_ratio",
        "keypoint_track_identity",
        # §14 pose in the embedding: a drawn or appended pose is a different vector
        "pose",
        "pose_weight",
        "pose_min_score",
    )
    identity_defaults = {
        "every_n_frames": 1,
        "joint": False,
        "joint_subjects": None,
        "align": "none",
        "heading_definition": None,
        "max_gap_s": 0.5,
        "mirror_hold_s": 1.0,
        "facing_camera_ratio": 0.5,
        "keypoint_track_identity": None,
        "pose": "none",
        "pose_weight": None,
        "pose_min_score": None,
    }
    lineage_ref = ("tracking", "tracking_run_id")
    # Embedding's one second input (§6.5 *auxiliary inputs*): the keypoint track run an
    # aligned run reads its heading from. Resolved per recording like the upstream spec.
    aux_refs = (("keypoint_track", "keypoint_run_id"),)
    config_model = EmbeddingConfig
    id_prefix = "emb"
    runs_key = "embedding_runs"
    upstream = "tracking"
    job_class = EmbeddingJob
    progress_event_class = EmbeddingProgressEvent
    # frames_total is the unique frames embedded
    progress_counts_frames = True

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        run(job, run_dir, progress_cb, stop_event)

    def run_dir_for(self, experiment_root, job):
        return embedding_run_dir(experiment_root, job)

    def output_path(self, run_id: str) -> str:
        return f"embeddings/{run_id}/"

    def estimate_output_bytes(
        self, recording_dir, recording_data: dict, config: dict
    ) -> int:
        """Rows × (vector × 4 bytes + the per-row columns) (#743), from the schema.

        A row is one embedded frame of one series: each selected track. The vector is
        float32 and barely compresses (measured 3,949 B/frame against 4,096 for a 1024-d
        model), so the schema is the estimate.
        """
        frames = disk_space.frames_to_process(
            recording_data, config.get("every_n_frames")
        )
        series = max(len(config.get("track_ids") or []), 1)
        return frames * series * (4 * embedding_dim(config) + EMBEDDING_ROW_EXTRA_BYTES)

    def resolve_shared_config(
        self,
        experiment_dir,
        recording_ids: list[str],
        config: dict,
        on_recording=None,
    ) -> dict:
        """Resolve the parameters that must agree across the selection, once.

        The max-contrast background is always resolved here, from the subjects' colour
        pooled over every selected recording: one colour per run, so the background is never
        a cue that tells recordings apart. The canvas is resolved here only when its scope is
        ``shared``; a ``per_recording`` canvas is left for each recording's own ``validate``
        to derive. An explicit ``renderer_spec`` (the raw API path) already pins everything,
        so nothing is resolved for it.
        """
        _resolve_defaults(config)
        if config.get("renderer_spec") is not None:
            return config
        recipe = CONSTRUCTORS.get(config["constructor"])
        if recipe is None:
            return config
        wants_canvas = recipe.needs_canvas and config.get("canvas_scope") == "shared"
        wants_bg = recipe.needs_bg_color
        if not (wants_canvas or wants_bg):
            return config
        per_recording = []
        for sid in recording_ids:
            if on_recording is not None:
                on_recording(sid)
            per_recording.append(
                _shared_inputs(
                    experiment_dir, sid, self, dict(config), wants_canvas, wants_bg
                )
            )
        if wants_canvas:
            # The canvas must contain every mask in the selection, so the max is the only
            # value that does not crop the largest subject.
            config["canvas_px"] = max(px for px, _ in per_recording)
        if wants_bg:
            means = [mean for _, mean in per_recording if mean is not None]
            pooled = tuple(
                int(round(sum(m[i] for m in means) / len(means))) for i in range(3)
            )
            config["bg_color"] = _max_contrast_rgb(pooled)
        return config

    def validate(self, recording_dir, recording_data: dict, config: dict) -> dict:
        # The model covers a subset (batch_size + typed ids/flags); it enforces the batch
        # bound + shape. The nested compute, the resolved renderer_spec, and the
        # all_tracks-resolved track_ids stay outside it, so validate keeps its own
        # enrichment and returns the enriched config.
        EmbeddingConfig.model_validate(config)
        _resolve_defaults(config)
        recording_id = Path(recording_dir).name
        tracking_run_id = config["tracking_run_id"]
        model_id = config["model_id"]
        compute = config["compute"]
        run_entry = {r["id"]: r for r in recording_data.get("tracking_runs", [])}.get(
            tracking_run_id
        )
        if run_entry is None:
            raise ValueError(
                f"tracking run {tracking_run_id!r} not found in recording {recording_id!r}"
            )
        if run_entry.get("status") != "complete":
            raise ValueError(
                f"tracking run {tracking_run_id!r} is {run_entry.get('status')!r}, "
                "not complete"
            )
        teacher_window.refuse_teacher_run(
            run_entry,
            consumer="embedding",
            upstream=f"tracking run {tracking_run_id!r}",
        )
        # Embedding decodes the same source frames detection saw, so gate that video
        # (a referenced original could have gone missing/changed since detection ran).
        det_entry = _detection_run_entry(
            recording_data, run_entry.get("detection_run_id")
        )
        # Against **tracking's** cadence, not detection's: rows the tracker never visited
        # carry no object_id, so what embedding can read is already thinned to that stride.
        upstream_stride = int(
            run_entry.get("every_n_frames") or det_entry.get("every_n_frames") or 1
        )
        config["every_n_frames"] = resolve_downstream_every_n(
            config.get("every_n_frames"), upstream_stride
        )
        video_integrity.raise_on_video_issues(
            video_integrity.validate_source_videos(recording_dir, recording_data)
        )
        tracks_path = Path(recording_dir) / "tracking" / tracking_run_id / "tracks.zarr"
        existing = {int(t) for t in PartZarrReader(tracks_path).read()["track_id"]}
        # "All tracks" is resolved per recording, so each recording embeds its own
        # full set (track ids are assigned independently per recording).
        if config.get("all_tracks"):
            track_ids = sorted(existing)
            if not track_ids:
                raise ValueError(
                    f"tracking run {tracking_run_id!r} has no tracks to embed"
                )
            config["track_ids"] = track_ids
        else:
            # A selector may be an int (a track's raw position) or a string (an
            # animal id, resolved through *this recording's own* subject_labels —
            # §13) — resolved per recording so the same name means the same
            # animal everywhere, even where a bare int would not (track_merge's
            # merged runs assign object_id by source order, not by identity).
            selectors = list(config["track_ids"])
            subject_labels = (
                _subject_labels(recording_dir, tracking_run_id)
                if any(isinstance(t, str) for t in selectors)
                else None
            )
            track_ids = [resolve_track_id(t, subject_labels) for t in selectors]
            missing = [t for t in track_ids if int(t) not in existing]
            if missing:
                raise ValueError(
                    f"track_ids {missing} not in tracking run {tracking_run_id!r}; "
                    f"available: {sorted(existing)}"
                )
            config["track_ids"] = track_ids
        # §14 alignment: resolved before the renderer spec, since the aligned canvas is
        # sized from the turned masks and the alignment step is inserted into the spec.
        _resolve_alignment(recording_dir, recording_data, config, track_ids)
        align = config["align"]
        # The GUI sends a named constructor; the API may send a renderer_spec
        # directly (granular control). Resolve to a concrete spec here, per
        # recording, so the centered constructors size their canvas to this
        # recording's masks — unless a ``shared`` scope already resolved the value
        # across the whole selection (``resolve_shared_config``) and passed it in.
        # The resolved spec is what build_job records.
        renderer_spec = config.get("renderer_spec")
        if renderer_spec is None:
            constructor = config.get("constructor")
            if constructor is None:
                raise ValueError("config must provide a constructor or a renderer_spec")
            recipe = CONSTRUCTORS.get(constructor)
            canvas_px = config.get("canvas_px")
            if (
                align != "none"
                and canvas_px is None
                and recipe is not None
                and recipe.needs_canvas
            ):
                canvas_px = largest_turned_mask_px(
                    recording_dir,
                    tracking_run_id,
                    track_ids,
                    _alignment_settings(config),
                )
            renderer_spec = resolve_renderer_spec(
                recording_dir,
                tracking_run_id,
                track_ids,
                constructor,
                model_id=model_id,
                resize=config.get("resize"),
                input_size=config.get("input_size"),
                canvas_px=canvas_px,
                bg_color=config.get("bg_color"),
            )
            # The points are drawn before the turn, so a rotation carries them with the body.
            renderer_spec = with_alignment(
                with_pose(renderer_spec, _draw_pose_step(recording_dir, config)), align
            )
            config["renderer_spec"] = renderer_spec
        elif (alignment_step(renderer_spec) or "none") != align:
            raise ValueError(
                f"the renderer_spec carries alignment {alignment_step(renderer_spec)!r} "
                f"but align is {align!r}; an explicit spec must carry exactly the step "
                "align names"
            )
        else:
            check_pose_step(renderer_spec, config["pose"])
        model_cls = EMBEDDING_MODELS.get(model_id)
        if model_cls is None:
            raise ValueError(
                f"unknown model_id {model_id!r}; available: {sorted(EMBEDDING_MODELS)}"
            )
        if compute.backend == "cuda" and compute.device_indices:
            check_cuda_indices(compute.device_indices)
        # constructing validates the variant + read-out (raises ValueError if unknown) and
        # rejects an input_size / read-out on a model that does not support one.
        model = construct_embedding_model(
            model_cls,
            config.get("model_variant"),
            readout=config.get("readout"),
            input_size=config.get("input_size"),
        )
        # The *resolved* variant. Left unset it comes from `variant or DEFAULT_VARIANT or
        # next(iter(VARIANTS))` (protocols.py) — dict insertion order — and `model_variant` is
        # both a run identity field and part of the embedding recipe, so two runs recorded as
        # `None` would be judged interchangeable across a code change that reordered VARIANTS.
        config["model_variant"] = model.variant
        # Readiness is a fact about *this* machine's disk (and, if not installed, a
        # network check from here — Hard Rules §"Model weights"): whether the weights
        # are present where `load()` will look for them. A remote job never calls
        # `load()` here — the shepherd rewrites `compute` to the on-cluster backend and
        # the model actually loads on the cluster — so gating submission on *this*
        # machine's install state would demand a full local install for a job this
        # machine only submits. The cluster's own readiness is checked by
        # `executor/remote_shepherd.py` before it submits the sbatch job, so a missing
        # install still fails clearly — just not here, and not against this machine's
        # cache. Mirrors detection.py's identical guard.
        if compute.backend != "remote":
            status = model.readiness()
            if not status.ready:
                raise ModelNotReadyError(status)
        if effective_backend(compute) not in model.supported_backends:
            raise ValueError(
                f"model {model_id!r} does not support the {compute.backend!r} backend; "
                f"it requires one of {list(model.supported_backends)}"
            )
        remote_targets.require_registered(compute)
        if compute.backend != "remote":
            # this *install* on this backend (§5); a remote job runs on the cluster's
            # install, which the shepherd checks there before it submits
            model.check_backend(compute.backend)
        unknown = [s["type"] for s in renderer_spec if s["type"] not in TRANSFORMS]
        if unknown:
            raise ValueError(
                f"unknown transform types {unknown}; available: {sorted(TRANSFORMS)}"
            )
        # No active-run guard here: embedding runs write per-run-isolated outputs
        # (embeddings/<run_id>/...) and their recording.yaml record is reserved + updated under the
        # recording lock (storage/recording.py::recording_lock), so several embeddings can run
        # concurrently on one recording (e.g. pinned to different GPUs) without colliding.
        return config

    def build_job(self, recording_id: str, run_id: str, config: dict) -> EmbeddingJob:
        return EmbeddingJob(
            job_id=run_id,
            recording_id=recording_id,
            embedding_run_id=run_id,
            tracking_run_id=config["tracking_run_id"],
            track_ids=list(config["track_ids"]),
            model_id=config["model_id"],
            model_variant=config.get("model_variant"),
            constructor=config.get("constructor"),
            readout=config.get("readout"),
            input_size=config.get("input_size"),
            renderer_spec=config["renderer_spec"],
            batch_size=config.get("batch_size", 32),
            every_n_frames=int(config.get("every_n_frames") or 1),
            num_workers=config.get("num_workers", 0),
            compute=config["compute"],
            created_at=utc_timestamp(),
            canvas_scope=config.get("canvas_scope", "per_recording"),
            bg_color_scope=BG_COLOR_SCOPE,
            align=str(config.get("align") or "none"),
            keypoint_run_id=config.get("keypoint_run_id"),
            heading_definition=config.get("heading_definition"),
            heading_cutoff=config.get("heading_cutoff"),
            max_gap_s=float(config.get("max_gap_s", 0.5)),
            mirror_hold_s=float(config.get("mirror_hold_s", 1.0)),
            facing_camera_ratio=float(config.get("facing_camera_ratio", 0.5)),
            pose=str(config.get("pose") or "none"),
            pose_weight=config.get("pose_weight"),
            pose_min_score=config.get("pose_min_score"),
        )

    def extra_record_fields(self, config: dict) -> dict:
        compute = config.get("compute")
        return {
            "tracking_run_id": config["tracking_run_id"],
            # §14 alignment: the id for lineage, the identity for comparison.
            "align": str(config.get("align") or "none"),
            "keypoint_run_id": config.get("keypoint_run_id"),
            "heading": config.get("heading"),
            "heading_definition": config.get("heading_definition"),
            "heading_cutoff": config.get("heading_cutoff"),
            "max_gap_s": float(config.get("max_gap_s", 0.5)),
            "mirror_hold_s": float(config.get("mirror_hold_s", 1.0)),
            "facing_camera_ratio": float(config.get("facing_camera_ratio", 0.5)),
            "keypoint_track_identity": config.get("keypoint_track_identity"),
            "pose": str(config.get("pose") or "none"),
            "pose_weight": config.get("pose_weight"),
            "pose_min_score": config.get("pose_min_score"),
            "model_id": config["model_id"],
            "model_variant": config.get("model_variant"),
            "constructor": config.get("constructor"),
            "readout": config.get("readout"),
            "input_size": config.get("input_size"),
            # The size the renderer actually emits, when it is pinned independently of the
            # model input — a different image, so a different recipe.
            "resize": config.get("resize"),
            # Also recipe identity: two runs at different cadence sample the behavior
            # differently, so they are not interchangeable in a comparison.
            "every_n_frames": int(config.get("every_n_frames") or 1),
            # The scopes decide what the derived canvas/background actually resolved to, so
            # they are recipe identity too. ``_write_embedding_yaml`` records them on the run
            # itself, but a comparison reads this manifest entry
            # (``api/comparison._embedding_config``) — leaving them out here made a
            # ``shared``-vs-``per_recording`` mismatch invisible to the held-constant check.
            "canvas_scope": config.get("canvas_scope", "per_recording"),
            "bg_color_scope": BG_COLOR_SCOPE,
            # What order the model was handed the pixels in: a run before the fix says nothing
            # and reads as "bgr" (``run_preview._RECIPE_KEYS``).
            "channel_order": CHANNEL_ORDER,
            "track_ids": list(config["track_ids"]),
            # The two inert identity fields (see `identity_fields`): every run states them, as
            # every run recorded before the single-animal build did.
            "joint": False,
            "joint_subjects": None,
            # Known at reservation time — see detection.py's identical field.
            "compute": compute.to_dict() if compute is not None else None,
        }
