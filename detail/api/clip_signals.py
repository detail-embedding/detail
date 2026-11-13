"""What a clip's animal is doing, as numbers over the clip popup's window: speed, mask area and
the distance to the nearest other animal, plus where the frame cannot be trusted.

**Read from the tracking run, not from an embedding's sidecar.** Every clustered clip has a
tracking run under it, so the strips work for every run on disk; the per-frame kinematics an
embedding run *may* have written would leave older runs blank. A frame's subject is its
detection rows, unioned as embedding unions them (``pipeline.subject_frames``), and the
assignment is laid onto the detection rows (``object_ids_on_detection_rows``), never paired by
position — a class-filtered tracking run's two arrays are not parallel.

**Units a person can read, never pixels.** A recording with a distance calibration gets mm and
mm/s; one without gets **body sizes** — the square root of the subject's median mask area in
this part — so a slow mouse and a slow rat read alike. Mask area is always relative to that
median (``1.0`` = typical), because what tells a rear from a walk in a top view is the *drop*
in area, not its value.

**Quality** marks the frames where the numbers would mislead: the subject is not tracked on
the frame, or its detection is below :data:`LOW_CONFIDENCE`.

**The body's shape, from the keypoints** (:func:`pose_signals`), when the embedding run read a
keypoint run and a heading definition says which points are the front and the back: body
length relative to typical (a stretch reads above 1, a hunch below), the bend at the body's
midline in degrees, and turning in degrees per second. All three are measured by
``analysis.body_axis``, the code an aligned embedding turns its pictures by, so the strips
describe the axis the clusters were built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from detail.core import yaml_load

#: Below this detection confidence a frame is flagged on the quality strip. Policy, not
#: measured: the detection run's own default threshold (0.5) is where it stopped being sure.
LOW_CONFIDENCE = 0.5
#: Frames either side averaged into a speed, so one jittery box does not draw a spike.
_SPEED_HALF = 2


@dataclass(frozen=True)
class ClipSignal:
    """One strip: a value per window frame (NaN where unknown), its unit, and what it is."""

    name: str
    unit: str
    values: np.ndarray


@dataclass(frozen=True)
class ClipSignals:
    """Everything the popup draws under the picture for one window of frames.

    ``frames`` are global frame numbers, ``signals`` the strips in display order,
    ``unreliable`` the quality strip's flags, ``centre`` the subject's normalized centre
    (NaN where untracked) for the motion trail, and ``t_ns`` each frame's time.
    """

    frames: np.ndarray
    t_ns: np.ndarray
    signals: list[ClipSignal]
    unreliable: np.ndarray
    centre: np.ndarray
    units_note: str = ""
    extras: dict = field(default_factory=dict)


@dataclass
class _Part:
    """One detection part with its assignment laid on, read once and kept."""

    frame_idx: np.ndarray
    bbox: np.ndarray
    confidence: np.ndarray
    mask_rle: object
    object_id: np.ndarray
    mask_hw: tuple[int, int] | None
    median_area: dict[int, float] = field(default_factory=dict)


class ClipSignalReader:
    """Reads :class:`ClipSignals` for one tracking run; keeps each part it has read."""

    def __init__(
        self,
        recording_dir: Path | str,
        tracking_run_id: str,
        part_ids: list[str],
        part_frame_counts: list[int],
        frame_wh: tuple[int, int],
        pixels_per_mm: float | None,
    ) -> None:
        self._dir = Path(recording_dir)
        self._tracking_run_id = tracking_run_id
        self._part_ids = list(part_ids)
        self._counts = [int(c) for c in part_frame_counts]
        self._starts = np.concatenate([[0], np.cumsum(self._counts)[:-1]]).astype(int)
        self._wh = (int(frame_wh[0]), int(frame_wh[1]))
        self._ppm = pixels_per_mm
        self._parts: dict[str, _Part | None] = {}

    def read(
        self, track_id: int, first: int, last: int, t_ns: np.ndarray
    ) -> ClipSignals:
        """The window ``[first, last]`` (inclusive global frames) for ``track_id``;
        ``t_ns`` is the recording's frame times, indexed by global frame."""
        frames = np.arange(int(first), int(last) + 1)
        n = len(frames)
        centre_px = np.full((n, 2), np.nan)
        area_rel = np.full(n, np.nan)
        nearest_px = np.full(n, np.nan)
        confidence = np.full(n, np.nan)
        body_px = np.nan
        w, h = self._wh
        for part_index, (start, count) in enumerate(zip(self._starts, self._counts)):
            lo, hi = max(first, start), min(last, start + count - 1)
            if lo > hi:
                continue
            part = self._part(self._part_ids[part_index])
            if part is None:
                continue
            median = self._median_area(part, int(track_id))
            if np.isfinite(median) and median > 0:
                body_px = np.sqrt(median)
            frame_col = part.frame_idx
            in_window = np.flatnonzero(
                (frame_col >= lo - start) & (frame_col <= hi - start)
            )
            by_frame: dict[int, dict[int, list[int]]] = {}
            for row in in_window.tolist():
                oid = int(part.object_id[row])
                if oid < 0:
                    continue
                by_frame.setdefault(int(frame_col[row]), {}).setdefault(oid, []).append(
                    row
                )
            for local, subjects in by_frame.items():
                i = start + local - first
                centres = {
                    oid: _box_centre_px(part.bbox[rows], w, h)
                    for oid, rows in subjects.items()
                }
                rows = subjects.get(int(track_id))
                if rows is None:
                    continue
                centre_px[i] = centres[int(track_id)]
                confidence[i] = float(np.min(part.confidence[rows]))
                if np.isfinite(median) and median > 0:
                    area_rel[i] = self._area_px(part, rows) / median
                others = [c for oid, c in centres.items() if oid != int(track_id)]
                if others:
                    nearest_px[i] = min(
                        float(np.hypot(*(np.asarray(c) - centre_px[i]))) for c in others
                    )
        times = np.asarray(t_ns, dtype=np.int64)[frames] if len(t_ns) else frames * 0
        speed_px = _speed(centre_px, times)
        if self._ppm:
            scale, dist_unit, speed_unit = 1.0 / self._ppm, "mm", "mm/s"
            note = "distances in mm (the recording's calibration)"
        elif np.isfinite(body_px):
            scale, dist_unit, speed_unit = 1.0 / body_px, "body sizes", "body sizes/s"
            note = (
                "distances in body sizes (no calibration): one is the square root of the "
                "animal's typical mask area"
            )
        else:
            scale, dist_unit, speed_unit = np.nan, "body sizes", "body sizes/s"
            note = "no masks to size the animal by, so no distances"
        signals = [
            ClipSignal("speed", speed_unit, speed_px * scale),
            ClipSignal("mask area", "x typical", area_rel),
            ClipSignal("to nearest animal", dist_unit, nearest_px * scale),
        ]
        untracked = ~np.isfinite(centre_px).all(axis=1)
        unsure = np.nan_to_num(confidence, nan=1.0) < LOW_CONFIDENCE
        return ClipSignals(
            frames=frames,
            t_ns=times,
            signals=signals,
            unreliable=untracked | unsure,
            centre=centre_px / np.asarray([w, h], dtype=np.float64),
            units_note=note,
        )

    def boxes(self, track_id: int | None, first: int, last: int) -> np.ndarray:
        """``(n, 4)`` normalized ``x0, y0, x1, y1`` of ``track_id`` over ``[first, last]``
        (inclusive global frames): the union of its boxes on each frame, NaN where it was not
        tracked. ``track_id`` ``None`` takes every tracked animal.

        What a clip montage crops each tile to (``api.clip_montage.crop_box``)."""
        frames = np.arange(int(first), int(last) + 1)
        out = np.full((len(frames), 4), np.nan)
        for part_index, (start, count) in enumerate(zip(self._starts, self._counts)):
            lo, hi = max(first, start), min(last, start + count - 1)
            if lo > hi:
                continue
            part = self._part(self._part_ids[part_index])
            if part is None:
                continue
            frame_col = part.frame_idx
            rows = np.flatnonzero((frame_col >= lo - start) & (frame_col <= hi - start))
            if track_id is None:
                rows = rows[part.object_id[rows] >= 0]
            else:
                rows = rows[part.object_id[rows] == int(track_id)]
            for row in rows.tolist():
                i = start + int(frame_col[row]) - first
                box = part.bbox[row]
                if np.isnan(out[i, 0]):
                    out[i] = box
                else:
                    out[i, :2] = np.minimum(out[i, :2], box[:2])
                    out[i, 2:] = np.maximum(out[i, 2:], box[2:])
        return out

    # -- reading --------------------------------------------------------------- #
    def _part(self, part_id: str) -> _Part | None:
        if part_id not in self._parts:
            self._parts[part_id] = self._read_part(part_id)
        return self._parts[part_id]

    def _read_part(self, part_id: str) -> _Part | None:
        from detail.core import atomic
        from detail.pipeline.subject_frames import _mask_size
        from detail.storage.zarr_io import (
            DetectionPartReader,
            PartZarrReader,
            object_ids_on_detection_rows,
        )

        tracking_dir = self._dir / "tracking" / self._tracking_run_id
        meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml")) or {}
        detection_dir = self._dir / "detections" / str(meta["detection_run_id"])
        det_path = detection_dir / f"{part_id}.zarr"
        assign_path = tracking_dir / "assignments" / f"{part_id}.zarr"
        if not det_path.is_dir() or not assign_path.is_dir():
            return None
        reader = DetectionPartReader(det_path)
        object_id = object_ids_on_detection_rows(
            reader, PartZarrReader(assign_path), meta.get("class_name")
        )
        size = _mask_size(detection_dir)
        return _Part(
            frame_idx=np.asarray(reader.column("frame_idx")[:]),
            bbox=np.asarray(reader.column("bbox")[:], dtype=np.float64),
            confidence=np.asarray(reader.column("confidence")[:], dtype=np.float64),
            mask_rle=reader.column("mask_rle"),
            object_id=np.asarray(object_id),
            mask_hw=None if size is None else (int(size[0]), int(size[1])),
        )

    def _median_area(self, part: _Part, track_id: int) -> float:
        """The subject's median area over the whole part, in frame pixels: "typical"."""
        if track_id not in part.median_area:
            rows = np.flatnonzero(part.object_id == track_id)
            by_frame: dict[int, list[int]] = {}
            for row in rows.tolist():
                by_frame.setdefault(int(part.frame_idx[row]), []).append(row)
            # Every frame would be exact and slow on an hour-long part; an even sample of a
            # few hundred frames gives the same median to well under a percent.
            frames = sorted(by_frame)
            if len(frames) > 400:
                frames = [
                    frames[i] for i in np.linspace(0, len(frames) - 1, 400).astype(int)
                ]
            areas = [self._area_px(part, by_frame[f]) for f in frames]
            finite = [a for a in areas if np.isfinite(a) and a > 0]
            part.median_area[track_id] = float(np.median(finite)) if finite else np.nan
        return part.median_area[track_id]

    def _area_px(self, part: _Part, rows: list[int]) -> float:
        """The subject's mask area on one frame, in frame pixels; NaN with no masks."""
        from pycocotools import mask as mask_utils

        w, h = self._wh
        mh, mw = part.mask_hw if part.mask_hw is not None else (h, w)
        total = 0.0
        found = False
        for row in rows:
            counts = str(part.mask_rle[row])
            if not counts:
                continue
            total += float(
                mask_utils.area({"size": [mh, mw], "counts": counts.encode()})
            )
            found = True
        if not found:
            return np.nan
        return total * (w * h) / (mw * mh)


def _box_centre_px(boxes: np.ndarray, w: int, h: int) -> tuple[float, float]:
    """The centre of the union of a frame's boxes for one subject, in frame pixels."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    return ((x0 + x1) / 2 * w, (y0 + y1) / 2 * h)


def pose_signals(
    points: np.ndarray,
    scores: np.ndarray,
    t_ns: np.ndarray,
    *,
    front: list[int],
    back: list[int],
    middle: list[int],
    frame_wh: tuple[int, int],
    cutoff: float | None,
    typical_length_px: float,
) -> list[ClipSignal]:
    """The body-shape strips over a window: ``points`` ``(n, K, 2)`` normalized and ``scores``
    ``(n, K)`` per window frame (NaN where the subject has no keypoints), ``t_ns`` each frame's
    time. ``typical_length_px`` is the track's median axis length, which ``body length`` is
    relative to, so 1.0 is the animal's usual length whatever the camera's distance.

    ``bend`` is left out when the skeleton has no midline point between the axis ends
    (``middle`` empty): a two-point axis cannot bend, and an empty strip would say it never
    did.
    """
    from detail.analysis import body_axis

    deg, length = body_axis.row_headings(points, scores, front, back, frame_wh, cutoff)
    has_axis = np.isfinite(length) & (length > 0)
    relative = np.full(len(length), np.nan)
    if typical_length_px > 0:
        relative[has_axis] = length[has_axis] / float(typical_length_px)
    deg = np.where(has_axis, deg, np.nan)
    out = [ClipSignal("body length", "x typical", relative)]
    if middle:
        bend = body_axis.row_bends(
            points, scores, front, back, middle, frame_wh, cutoff
        )
        out.append(ClipSignal("bend", "°", bend))
    out.append(
        ClipSignal("turning", "°/s", _turning(deg, np.asarray(t_ns, dtype=np.int64)))
    )
    return out


def _turning(deg: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    """Degrees per second at each frame, between the outermost frames with a heading within
    ``_SPEED_HALF`` either side, the short way round the circle (positive clockwise on
    screen); NaN without a heading on one side. The same reach as :func:`_speed`, for the same
    reason: one frame's keypoint jitter is not a turn."""
    n = len(deg)
    out = np.full(n, np.nan)
    known = np.isfinite(deg)
    k = _SPEED_HALF
    for i in range(n):
        before = [j for j in range(max(i - k, 0), i + 1) if known[j]]
        after = [j for j in range(i, min(i + k, n - 1) + 1) if known[j]]
        if not before or not after or before[0] == after[-1]:
            continue
        a, b = before[0], after[-1]
        dt = (int(t_ns[b]) - int(t_ns[a])) / 1e9
        if dt > 0:
            out[i] = ((float(deg[b]) - float(deg[a]) + 180.0) % 360.0 - 180.0) / dt
    return out


def _speed(centre_px: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    """Pixels per second at each frame, between the outermost tracked frames within
    ``_SPEED_HALF`` either side (the frame itself counts); NaN when there is no tracked frame on
    one side. A box centre moves in whole pixels, so a one-frame difference is mostly
    quantization; reaching a few frames out reads the motion instead."""
    n = len(centre_px)
    out = np.full(n, np.nan)
    tracked = np.isfinite(centre_px).all(axis=1)
    k = _SPEED_HALF
    for i in range(n):
        before = [j for j in range(max(i - k, 0), i + 1) if tracked[j]]
        after = [j for j in range(i, min(i + k, n - 1) + 1) if tracked[j]]
        if not before or not after or before[0] == after[-1]:
            continue
        a, b = before[0], after[-1]
        dt = (int(t_ns[b]) - int(t_ns[a])) / 1e9
        if dt > 0:
            out[i] = float(np.hypot(*(centre_px[b] - centre_px[a]))) / dt
    return out
