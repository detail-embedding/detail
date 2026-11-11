"""Give each pose instance to at most one tracked subject, per frame (§26 *Assignment*).

The first step of every keypoint track run, and of the distill teacher (§12): a keypoint detect
part holds pose instances with no subject; this decides whose pose each one is, against one
tracking run. It is the only place a pose meets a subject.

**The class of bug this step exists for: identity decided by box overlap, inside the detect
stage.** Before the 2026-10-21 revision the student gave a pose to the subject whose box it
overlapped (IoU ≥ 0.1) while detecting. When two animals touch, their boxes overlap heavily and
their masks do not, so a pose went to whichever box won; and because the choice was made in the
GPU stage, trying another tracking run or threshold meant re-detecting. Here the score is the
share of an instance's placed points inside the subject's *mask* (grown a little, since
segmentation clips noses and tail tips), the match is one-to-one and global per frame
(Hungarian, not greedy, so one confident pose cannot steal a subject and strand the pose that
fits it), and re-assigning is a CPU re-run of the track stage. The mask is never rasterised
for this: the grown-mask test is evaluated at the pose's points against the mask's run lengths
(``core.rle``), the same answer ``cv2.dilate`` over a decoded frame gave — and that decode and
dilate were most of a track run, to test three points per frame.

The tracking run's assignment zarr is joined to the detection rows **by row id, never by
position** (``subject_frames``, #657): a class-filtered run's assignment covers only the tracked
class's rows.

**The defaults — ``min_inside = 0.5``, ``mask_grow = 0.05`` of body scale — are policy, not
measured** (§26 *Open questions*). Both are part of a track run's identity (``match``).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml

from detail.core import atomic
from detail.core.rle import RunMask
from detail.models.patch_match.view import body_scale
from detail.pipeline.subject_frames import _mask_size
from detail.pipeline.subject_frames import _part_columns
from detail.pipeline.subject_frames import subject_frames
from detail.storage import recording as recording_store
from detail.storage.zarr_io import DetectionPartReader, PartZarrReader

DEFAULT_MIN_INSIDE = 0.5
DEFAULT_MASK_GROW = 0.05
# Ties in total score go to the higher-confidence instance: a per-instance bonus far below any
# real score difference (shares are k/n with n ≤ K points) decides between equal totals only.
_TIE_EPS = 1e-9


class SubjectOnFrame:
    """One subject on one frame: union ``bbox`` (normalized), ``runs`` (its detection rows'
    masks as :class:`RunMask` objects at the mask resolution), ``scale_px`` (``√(w·h)`` of the box
    in frame pixels — the track methods' length unit).

    ``runs`` is parsed from the stored strings on each read and kept nowhere; a caller that
    needs it twice keeps the list. A kept mask lives as long as the subject, and a caller that
    holds every frame (the keypoint dataset sampler holds a whole job's) would hold every mask
    it read — the memory class ``tests/pipeline/test_pose_match.py`` pins.
    """

    __slots__ = ("bbox", "scale_px", "_runs")

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        runs: Callable[[], list[RunMask]],
        scale_px: float,
    ) -> None:
        self.bbox = tuple(float(v) for v in bbox)
        self.scale_px = float(scale_px)
        self._runs = runs

    @property
    def runs(self) -> list[RunMask]:
        return list(self._runs())

    def __repr__(self) -> str:
        return f"SubjectOnFrame(bbox={self.bbox}, scale_px={self.scale_px:.1f})"


def grow_radius_px(scale_px: float, mask_grow: float) -> float:
    """How far a subject's mask is grown, in the pixels ``scale_px`` is in."""
    return float(mask_grow) * float(scale_px)


def to_mask_px(points_norm: np.ndarray, mask_hw: tuple[int, int]) -> np.ndarray:
    """Normalized full-frame points ``(..., 2)`` → mask pixel coordinates; pixel ``i`` covers
    ``[i, i + 1)``. A mask is stored at detection resolution, which need not be the frame's.
    """
    h, w = mask_hw
    return np.asarray(points_norm, dtype=np.float64) * np.array(
        [w, h], dtype=np.float64
    )


@functools.lru_cache(maxsize=None)
def _footprint_spans(r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The grow kernel by column: each column's x offset and its rows' first and last
    offsets from the centre. Read from the same ``cv2`` ellipse ``cv2.dilate`` would use, so
    the footprint is OpenCV's rasterised ellipse, not a re-derived disc; ``r <= 0`` is the
    pixel itself.
    """
    if r <= 0:
        zero = np.zeros(1, dtype=np.int64)
        return zero, zero, zero
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)) > 0
    cols = np.flatnonzero(kernel.any(axis=0))
    lo = kernel.argmax(axis=0)[cols]
    hi = kernel.shape[0] - 1 - kernel[::-1].argmax(axis=0)[cols]
    if not np.array_equal(kernel[:, cols].sum(axis=0), hi - lo + 1):
        raise RuntimeError("the grow kernel's columns are not contiguous")
    return cols - r, lo - r, hi - r


def inside_share(
    points_xy: np.ndarray, masks: Sequence[RunMask], radius_px: float
) -> np.ndarray:
    """Share of each instance's finite points inside the subject's masks grown by
    ``radius_px``, evaluated at the points: a point is inside the grown mask when any mask
    pixel lies within the grow kernel's footprint around it, which is ``cv2.dilate``'s
    definition read at one pixel rather than computed for every pixel of a frame.

    Args:
        points_xy: ``(N, K, 2)`` in mask pixels (:func:`to_mask_px`).
        masks: The subject's detection rows' masks, one :class:`RunMask` each (their OR is
            the subject; a hit in any is a hit).
        radius_px: Grow radius in mask pixels; ≤ 0.5 grows nothing.

    Returns:
        ``(N,)`` float32; 0.0 for an instance with no finite point. A point off the mask grid
        is outside.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    n = points_xy.shape[0]
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out
    finite = np.isfinite(points_xy).all(axis=-1)  # (N, K)
    n_placed = finite.sum(axis=1)
    masks = [m for m in masks if m.n_ones]
    if not finite.any() or not masks:
        return out
    h, w = masks[0].size
    r = int(round(max(float(radius_px), 0.0)))
    dx, dy_lo, dy_hi = _footprint_spans(r)
    # Clip before the cast: an absurd finite coordinate is off the grid either way, and must
    # not raise numpy's overflow warning on the way there.
    bound = float(max(h, w))
    xi = np.floor(np.clip(np.where(finite, points_xy[..., 0], -1.0), -1.0, bound))
    yi = np.floor(np.clip(np.where(finite, points_xy[..., 1], -1.0), -1.0, bound))
    xi, yi = xi.astype(np.int64), yi.astype(np.int64)
    on_grid = finite & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    cols = xi[..., None] + dx  # (N, K, C)
    y0 = np.maximum(yi[..., None] + dy_lo, 0)
    y1 = np.minimum(yi[..., None] + dy_hi + 1, h)
    valid = on_grid[..., None] & (cols >= 0) & (cols < w) & (y0 < y1)
    hit = np.zeros(on_grid.shape, dtype=bool)
    if valid.any():
        xs, ya, yb = cols[valid], y0[valid], y1[valid]
        found = np.zeros(len(xs), dtype=bool)
        for mask in masks:
            found |= mask.any_in_columns(xs, ya, yb)
        cell = np.zeros(valid.shape, dtype=bool)
        cell[valid] = found
        hit = cell.any(axis=-1)
    placed = n_placed > 0
    out[placed] = hit.sum(axis=1)[placed] / n_placed[placed]
    return out


def match_instances(
    points_xy: np.ndarray,
    confidence: np.ndarray,
    subject_masks: dict[int, Sequence[RunMask]],
    radius_px: dict[int, float],
    min_inside: float = DEFAULT_MIN_INSIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """One frame's instances to its subjects, one-to-one, maximizing the total inside share
    over pairs scoring at least ``min_inside``; ties go to the higher ``confidence`` (NaN
    lowest).

    Args:
        points_xy: ``(N, K, 2)`` in mask pixels.
        confidence: ``(N,)``; NaN for a method that reports none.
        subject_masks: ``track_id →`` the subject's masks (:func:`inside_share`).
        radius_px: ``track_id → `` grow radius in mask pixels.
        min_inside: The least inside share a pair may be matched at.

    Returns:
        ``subject_id`` ``(N,)`` int32 (−1 unassigned) and ``inside`` ``(N,)`` float32 (the
        matched pair's share, NaN when unassigned).
    """
    from scipy.optimize import linear_sum_assignment

    points_xy = np.asarray(points_xy, dtype=np.float64)
    n = points_xy.shape[0]
    subject_id = np.full(n, -1, dtype=np.int32)
    inside = np.full(n, np.nan, dtype=np.float32)
    track_ids = list(subject_masks)
    if n == 0 or not track_ids:
        return subject_id, inside
    score = np.stack(
        [
            inside_share(points_xy, subject_masks[t], radius_px[t]).astype(np.float64)
            for t in track_ids
        ],
        axis=1,
    )  # (N, M)
    allowed = score >= float(min_inside)
    if not allowed.any():
        return subject_id, inside
    conf = np.asarray(confidence, dtype=np.float64).reshape(n)
    rank = np.empty(n, dtype=np.float64)
    rank[np.argsort(np.where(np.isnan(conf), -np.inf, conf), kind="stable")] = (
        np.arange(n) + 1.0
    ) / n
    gain = np.where(allowed, score + _TIE_EPS * rank[:, None], 0.0)
    rows, cols = linear_sum_assignment(gain, maximize=True)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if allowed[i, j]:
            subject_id[i] = int(track_ids[j])
            inside[i] = np.float32(score[i, j])
    return subject_id, inside


def subject_mask_hw(recording_dir: Path | str, tracking_run_id: str) -> tuple[int, int]:
    """The ``(h, w)`` :func:`frame_subjects`' masks come at: the detection run's mask
    resolution, or the frame's when it kept no masks (each subject's box stands in)."""
    recording_dir = Path(recording_dir)
    detection_dir = _detection_dir(recording_dir, tracking_run_id)
    size = _mask_size(detection_dir)
    if size is not None:
        return size
    w, h = _frame_wh(recording_dir)
    return (h, w)


def frame_subjects(
    recording_dir: Path | str,
    tracking_run_id: str,
    part_id: str,
    every_n_frames: int,
) -> Iterator[tuple[int, int, dict[int, SubjectOnFrame]]]:
    """Every frame of the part with a subject, in frame order, at the cadence:
    ``(frame_idx, t_ns, {track_id: SubjectOnFrame})``.

    Each subject's masks are its detection rows' (``subject_frames``' grouping), read as
    runs only when a caller reads ``.runs``. A detection run that kept no masks gives each
    subject its box, filled, at frame resolution. A frame is let go once it is yielded, so a
    part's subjects and their decoders are not all held to its end.

    Raises:
        ValueError: No such tracking run in the recording.
    """
    recording_dir = Path(recording_dir)
    detection_dir = _detection_dir(recording_dir, tracking_run_id)
    det_path = detection_dir / f"{part_id}.zarr"
    assignment_path = (
        recording_dir / "tracking" / tracking_run_id / "assignments" / f"{part_id}.zarr"
    )
    if not det_path.is_dir() or not assignment_path.is_dir():
        return
    frame_wh = _frame_wh(recording_dir)
    mask_size = _mask_size(detection_dir)
    det = _part_columns(DetectionPartReader(det_path))
    object_id = np.asarray(PartZarrReader(assignment_path).column("object_id")[:])
    by_frame: dict[int, tuple[int, dict[int, SubjectOnFrame]]] = {}
    for track_id in sorted({int(t) for t in object_id.tolist() if int(t) >= 0}):
        for item in subject_frames(det, object_id, track_id, every_n_frames):
            if mask_size is None:
                runs = _box_decoder(item["bbox"], frame_wh)
            else:
                runs = _mask_decoder(det, item["rows"], mask_size)
            entry = by_frame.setdefault(item["frame_idx"], (item["t_ns"], {}))
            entry[1][track_id] = SubjectOnFrame(
                item["bbox"], runs, body_scale(item["bbox"], frame_wh)
            )
    for frame in sorted(by_frame):
        t_ns, subjects = by_frame.pop(frame)
        yield frame, t_ns, subjects


def assign_part(
    instances: dict,
    subjects_iter,
    mask_hw: tuple[int, int],
    *,
    min_inside: float = DEFAULT_MIN_INSIDE,
    mask_grow: float = DEFAULT_MASK_GROW,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[tuple[int, int, int, float]]]]:
    """Assignment for one part: each detect instance to at most one subject per frame.

    Args:
        instances: A keypoint detect part's rows — ``frame_idx`` ``(N,)``, ``points``
            ``(N, K, 2)`` normalized full-frame, ``confidence`` ``(N,)``.
        subjects_iter: :func:`frame_subjects`' frames (or the same shape).
        mask_hw: The subjects' mask resolution ``(h, w)`` (:func:`subject_mask_hw`).
        min_inside: The least inside share a pair may be matched at.
        mask_grow: How far a mask is grown, as a fraction of the subject's body scale.

    Returns:
        ``subject_id`` ``(N,)`` int32 (−1 unassigned), ``inside`` ``(N,)`` float32 (NaN
        unassigned) — parallel to the detect rows — and per subject, every frame it is on
        in order: ``(frame_idx, t_ns, instance_row or −1, scale_px)``.
    """
    frame_idx = np.asarray(instances["frame_idx"]).astype(np.int64)
    n = len(frame_idx)
    subject_id = np.full(n, -1, dtype=np.int32)
    inside = np.full(n, np.nan, dtype=np.float32)
    points = np.asarray(instances["points"])
    confidence = np.asarray(
        instances.get("confidence", np.full(n, np.nan)), dtype=np.float64
    )
    order = np.argsort(frame_idx, kind="stable")
    frames, starts, counts = np.unique(
        frame_idx[order], return_index=True, return_counts=True
    )
    rows_on = {
        int(f): order[s : s + c]
        for f, s, c in zip(frames.tolist(), starts.tolist(), counts.tolist())
    }
    mh, mw = mask_hw
    per_subject: dict[int, list[tuple[int, int, int, float]]] = {}
    for frame, t_ns, subjects in subjects_iter:
        rows = rows_on.get(int(frame))
        matched: dict[int, int] = {}
        if rows is not None and subjects:
            runs = {t: s.runs for t, s in subjects.items()}
            radius = {
                t: grow_radius_px(body_scale(s.bbox, (mw, mh)), mask_grow)
                for t, s in subjects.items()
            }
            sid, ins = match_instances(
                to_mask_px(points[rows], mask_hw),
                confidence[rows],
                runs,
                radius,
                min_inside,
            )
            subject_id[rows] = sid
            inside[rows] = ins
            matched = {int(t): int(r) for t, r in zip(sid.tolist(), rows) if t >= 0}
        for track_id in sorted(subjects):
            per_subject.setdefault(int(track_id), []).append(
                (
                    int(frame),
                    int(t_ns),
                    matched.get(int(track_id), -1),
                    subjects[track_id].scale_px,
                )
            )
    return subject_id, inside, per_subject


def _detection_dir(recording_dir: Path, tracking_run_id: str) -> Path:
    meta_path = recording_dir / "tracking" / tracking_run_id / "track.yaml"
    if not meta_path.is_file():
        raise ValueError(
            f"no tracking run {tracking_run_id!r} in recording {recording_dir.name!r}"
        )
    track_meta = yaml.safe_load(atomic.read_text(meta_path))
    return recording_dir / "detections" / track_meta["detection_run_id"]


def _frame_wh(recording_dir: Path) -> tuple[int, int]:
    resolution = recording_store.read_recording(recording_dir)["video"]["resolution"]
    return int(resolution[0]), int(resolution[1])


def _mask_decoder(det: dict, rows: list[int], mask_size) -> Callable[[], list[RunMask]]:
    column = det["mask_rle"]
    return lambda: [RunMask.from_counts_str(str(column[r]), mask_size) for r in rows]


def _box_decoder(bbox, frame_wh: tuple[int, int]) -> Callable[[], list[RunMask]]:
    w, h = frame_wh
    x0, y0 = int(np.floor(bbox[0] * w)), int(np.floor(bbox[1] * h))
    x1, y1 = int(np.ceil(bbox[2] * w)), int(np.ceil(bbox[3] * h))
    return lambda: [RunMask.from_box(x0, y0, x1, y1, (h, w))]
