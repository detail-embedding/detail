"""Label-free quality of a single-subject tracking output, and the setting sweep's pick.

Pure computation, no Qt and no I/O: it reads columns a caller already loaded (a detection
part's ``frame_idx`` / ``confidence`` / ``bbox`` / ``mask_rle`` beside the tracker's
``object_id`` for each row) and returns numbers. ``pipeline/tune_tracking.py`` runs the
sweep that feeds it; any tracking run can be measured the same way.

**What is measured, per recording.** Every share is of the frames that were looked at:

- *tracked*: frames with a subject (``object_id 0``);
- *merged*: frames where the tracker attached at least one secondary detection;
- *grew*: frames where the union of the subject's pieces is more than ``GROW_FRACTION``
  larger than the anchor detection alone;
- *jumpy*, the headline: frames whose subject mask area is more than ``JUMP_FRACTION``
  away from its rolling median over ``ROLLING_S`` seconds. The window is a duration,
  turned into rows at the recording's own frame rate, so it means the same on any video;
- *median area*, the share of tracked frames whose anchor confidence is below
  ``LOW_CONFIDENCE``, and unassigned detections per frame.

Areas come from the stored COCO RLE (``pycocotools.mask.area`` / ``merge``), never a full
decode. **Nothing here knows what the subject is.** The thresholds define the metrics, not a
size: an area is only ever compared with the same recording's own rolling median.

**The pick** keeps the settings whose tracked share is within ``TRACKED_MARGIN`` points of
the best on every recording, takes the lowest mean jumpy share (each recording counts
equally), and breaks ties within ``TIE_MARGIN`` points towards fewer secondaries and then a
higher confidence. When the defaults are within that margin, the answer is "keep them".

**Known limits.** A clutter blob merged into every frame is stable and wrong, so a low jumpy
share is not proof the mask is right: merged, grew and median area sit beside it for that
reason. The tracker follows one subject, so a recording that looks multi-subject (two
confident detections far apart on most frames) is flagged, and merging is not recommended
there, because it could join two animals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_utils

JUMP_FRACTION = 0.35
GROW_FRACTION = 0.15
ROLLING_S = 1.0
LOW_CONFIDENCE = 0.5
#: Points of tracked share a setting may give up against the best one and still be picked.
TRACKED_MARGIN = 0.5
#: Points of mean jumpy share within which two settings are a tie.
TIE_MARGIN = 0.05
#: Share of frames with two confident, non-overlapping detections that reads multi-subject.
MULTI_SUBJECT_SHARE = 0.5

ANCHOR_CONFS = (0.1, 0.25, 0.5, 0.7)
SECONDARY_CONFS = (0.05, 0.1, 0.25, 0.5)
MAX_SECONDARIES = (0, 2, 4, 8)


@dataclass(frozen=True, order=True)
class Setting:
    """One point of the sweep, in the tracker's own terms.

    A secondary below the detection run's confidence was never stored, so the detection
    confidence a setting needs is its lowest threshold: ``secondary_conf``, or
    ``conf_threshold`` when nothing is merged.
    """

    conf_threshold: float
    secondary_conf: float
    max_secondary: int

    @property
    def detection_conf(self) -> float:
        return self.secondary_conf if self.max_secondary else self.conf_threshold

    def hyperparams(self) -> dict:
        return {
            "conf_threshold": self.conf_threshold,
            "secondary_conf": self.secondary_conf,
            "max_secondary": self.max_secondary,
        }

    def label(self) -> str:
        if not self.max_secondary:
            return f"{self.conf_threshold:g}, no merging"
        return (
            f"{self.conf_threshold:g} / {self.secondary_conf:g} / "
            f"max {self.max_secondary}"
        )

    def to_dict(self) -> dict:
        return {"detection_conf": self.detection_conf, **self.hyperparams()}

    @classmethod
    def from_dict(cls, d: dict) -> "Setting":
        return cls(
            float(d["conf_threshold"]),
            float(d["secondary_conf"]),
            int(d["max_secondary"]),
        )


def effective_setting(
    detection_conf: float,
    conf_threshold: float,
    secondary_conf: float,
    max_secondary: int,
) -> Setting:
    """What a detection run at ``detection_conf`` tracked with these hyperparameters really
    keeps: no threshold can reach below what detection stored."""
    anchor = max(conf_threshold, detection_conf)
    return Setting(
        anchor, min(anchor, max(secondary_conf, detection_conf)), max_secondary
    )


def grid() -> list[Setting]:
    """Every setting the sweep measures. ``secondary_conf`` never exceeds
    ``conf_threshold``; with no merging it is set equal, since it then does nothing."""
    out: list[Setting] = []
    for anchor in ANCHOR_CONFS:
        out.append(Setting(anchor, anchor, 0))
        for secondary in SECONDARY_CONFS:
            if secondary > anchor:
                continue
            for n in MAX_SECONDARIES:
                if n:
                    out.append(Setting(anchor, secondary, n))
    return out


def sweep_detection_conf() -> float:
    """The one detection confidence the sweep detects at: the lowest any setting needs."""
    return min(s.detection_conf for s in grid())


def rolling_rows(frame_rate: float) -> int:
    """``ROLLING_S`` as an odd count of rows at ``frame_rate`` (at least 3)."""
    rows = max(3, int(round(ROLLING_S * frame_rate)))
    return rows if rows % 2 else rows + 1


@dataclass
class FrameTable:
    """One window's per-frame subject numbers, ``n_frames`` long. ``area`` is NaN on a
    frame with no subject."""

    area: np.ndarray
    anchor_area: np.ndarray
    pieces: np.ndarray
    anchor_conf: np.ndarray
    unassigned: int

    @property
    def n_frames(self) -> int:
        return len(self.area)


def rle_areas(mask_rle, size: tuple[int, int]) -> np.ndarray:
    """The pixel area of each stored RLE (0 for an empty one), without decoding."""
    objs = [_rle(s, size) for s in mask_rle]
    out = np.zeros(len(objs), dtype=np.float64)
    keep = [i for i, o in enumerate(objs) if o is not None]
    if keep:
        out[keep] = mask_utils.area([objs[i] for i in keep])
    return out


def _rle(counts, size):
    text = str(counts)
    if not text:
        return None
    return {"counts": text.encode("ascii"), "size": [int(size[0]), int(size[1])]}


def frame_table(
    n_frames: int,
    frame_idx: np.ndarray,
    confidence: np.ndarray,
    mask_rle,
    object_id: np.ndarray,
    size: tuple[int, int],
    areas: np.ndarray | None = None,
) -> FrameTable:
    """Per-frame subject area, anchor area, piece count and anchor confidence for one
    window whose frames are ``0 .. n_frames-1``.

    The anchor is the subject's highest-confidence row, as the tracker chose it. ``areas``
    is ``rle_areas`` of the rows when the caller already has it (a sweep reuses it).
    """
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float64)
    object_id = np.asarray(object_id)
    if areas is None:
        areas = rle_areas(mask_rle, size)
    area = np.full(n_frames, np.nan)
    anchor_area = np.full(n_frames, np.nan)
    pieces = np.zeros(n_frames, dtype=np.int64)
    anchor_conf = np.full(n_frames, np.nan)
    subject = np.flatnonzero(object_id == 0)
    by_frame: dict[int, list[int]] = {}
    for row in subject:
        by_frame.setdefault(int(frame_idx[row]), []).append(int(row))
    for frame, rows in by_frame.items():
        if not 0 <= frame < n_frames:
            continue
        rows.sort(key=lambda r: -confidence[r])
        anchor = rows[0]
        pieces[frame] = len(rows)
        anchor_conf[frame] = confidence[anchor]
        anchor_area[frame] = areas[anchor]
        if len(rows) == 1:
            area[frame] = areas[anchor]
        else:
            merged = mask_utils.merge(
                [o for o in (_rle(mask_rle[r], size) for r in rows) if o is not None]
            )
            area[frame] = float(mask_utils.area(merged))
    return FrameTable(
        area=area,
        anchor_area=anchor_area,
        pieces=pieces,
        anchor_conf=anchor_conf,
        unassigned=int(np.count_nonzero(object_id < 0)),
    )


def jumpy_frames(area: np.ndarray, window_rows: int) -> np.ndarray:
    """Frames whose area is more than ``JUMP_FRACTION`` off its centred rolling median.

    The median ignores frames with no subject; it needs a tenth of the window present, as
    the manual measurement did. Frames with no subject are never jumpy."""
    n = len(area)
    if n == 0:
        return np.zeros(0, dtype=bool)
    half = window_rows // 2
    padded = np.concatenate([np.full(half, np.nan), area, np.full(half, np.nan)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_rows)
    present = np.count_nonzero(~np.isnan(windows), axis=1)
    enough = present >= max(1, window_rows // 10)
    median = np.full(n, np.nan)
    if np.any(enough):
        median[enough] = np.nanmedian(windows[enough], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.abs(area - median) / median
    return np.nan_to_num(off, nan=0.0) > JUMP_FRACTION


@dataclass(frozen=True)
class Quality:
    """One setting's numbers on one recording (shares in percent)."""

    n_frames: int
    tracked: float
    merged: float
    grew: float
    jumpy: float
    median_area: float | None
    low_conf: float
    unassigned_per_frame: float

    def to_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "tracked_pct": round(float(self.tracked), 3),
            "merged_pct": round(float(self.merged), 2),
            "grew_pct": round(float(self.grew), 2),
            "jumpy_pct": round(float(self.jumpy), 3),
            "median_area": None if self.median_area is None else int(self.median_area),
            "low_conf_pct": round(float(self.low_conf), 2),
            "unassigned_per_frame": round(float(self.unassigned_per_frame), 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Quality":
        return cls(
            n_frames=int(d["n_frames"]),
            tracked=float(d["tracked_pct"]),
            merged=float(d["merged_pct"]),
            grew=float(d["grew_pct"]),
            jumpy=float(d["jumpy_pct"]),
            median_area=d.get("median_area"),
            low_conf=float(d["low_conf_pct"]),
            unassigned_per_frame=float(d["unassigned_per_frame"]),
        )


def quality(tables: list[FrameTable], frame_rate: float) -> Quality:
    """Pool a recording's windows into one :class:`Quality`. The rolling median runs inside
    each window, never across the gap between two."""
    rows = rolling_rows(frame_rate)
    n = sum(t.n_frames for t in tables)
    if n == 0:
        return Quality(0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0)
    area = np.concatenate([t.area for t in tables])
    tracked = ~np.isnan(area)
    pieces = np.concatenate([t.pieces for t in tables])
    anchor_area = np.concatenate([t.anchor_area for t in tables])
    anchor_conf = np.concatenate([t.anchor_conf for t in tables])
    jumpy = np.concatenate([jumpy_frames(t.area, rows) for t in tables])
    with np.errstate(invalid="ignore"):
        grew = (pieces > 1) & (area > (1.0 + GROW_FRACTION) * anchor_area)
    n_tracked = int(np.count_nonzero(tracked))
    return Quality(
        n_frames=n,
        tracked=100.0 * n_tracked / n,
        merged=100.0 * np.count_nonzero(pieces > 1) / n,
        grew=100.0 * np.count_nonzero(grew) / n,
        jumpy=100.0 * np.count_nonzero(jumpy) / n,
        median_area=float(np.median(area[tracked])) if n_tracked else None,
        low_conf=(
            100.0 * np.count_nonzero(anchor_conf[tracked] < LOW_CONFIDENCE) / n_tracked
            if n_tracked
            else 0.0
        ),
        unassigned_per_frame=sum(t.unassigned for t in tables) / n,
    )


def multi_subject_share(
    frame_idx: np.ndarray, confidence: np.ndarray, bbox: np.ndarray
) -> float:
    """Share of frames (with any detection) holding two detections at or above
    ``LOW_CONFIDENCE`` whose boxes do not touch: what two animals look like, and what one
    animal found in two pieces does not."""
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float64)
    bbox = np.asarray(bbox, dtype=np.float64).reshape(-1, 4)
    frames = np.unique(frame_idx)
    if not len(frames):
        return 0.0
    confident = confidence >= LOW_CONFIDENCE
    hits = 0
    for frame in frames:
        rows = np.flatnonzero((frame_idx == frame) & confident)
        if len(rows) < 2:
            continue
        boxes = bbox[rows]
        if any(
            _apart(boxes[i], boxes[j])
            for i in range(len(rows))
            for j in range(i + 1, len(rows))
        ):
            hits += 1
    return hits / len(frames)


def _apart(a, b) -> bool:
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


@dataclass(frozen=True)
class Pick:
    """The sweep's answer: the setting, whether it is just the defaults, and why."""

    setting: Setting | None
    keep_defaults: bool
    reason: str
    warnings: list[str] = field(default_factory=list)


def mean_jumpy(per_recording: dict[str, Quality]) -> float:
    values = [q.jumpy for q in per_recording.values() if q.n_frames]
    return float(np.mean(values)) if values else math.inf


def pick(
    results: dict[Setting, dict[str, Quality]],
    default: Setting,
    *,
    allow_merging: bool = True,
) -> Pick:
    """Choose a setting from the sweep (rule in the module docstring).

    ``results[setting][recording_id]`` is that setting's :class:`Quality` on that recording.
    ``allow_merging=False`` (a recording looks multi-subject) leaves only settings that
    attach nothing.
    """
    warnings: list[str] = []
    candidates = {
        s: per for s, per in results.items() if allow_merging or not s.max_secondary
    }
    if not candidates:
        return Pick(None, False, "No setting was measured.", warnings)
    recordings = sorted({r for per in candidates.values() for r in per})
    best_tracked = {
        r: max(per[r].tracked for per in candidates.values() if r in per)
        for r in recordings
    }
    if all(v == 0 for v in best_tracked.values()):
        return Pick(
            None,
            False,
            "No setting tracked a subject in the sample: the model found nothing "
            "confident enough to follow there.",
            warnings,
        )

    def eligible(per: dict[str, Quality]) -> bool:
        return all(
            r in per and per[r].tracked >= best_tracked[r] - TRACKED_MARGIN
            for r in recordings
        )

    kept = {s: per for s, per in candidates.items() if eligible(per)}
    scores = {s: mean_jumpy(per) for s, per in kept.items()}
    best = min(scores.values())
    ties = [s for s, v in scores.items() if v <= best + TIE_MARGIN]
    chosen = min(
        ties, key=lambda s: (s.max_secondary, -s.conf_threshold, -s.secondary_conf)
    )
    if default in scores and scores[default] <= best + TIE_MARGIN:
        return Pick(
            default,
            True,
            f"The defaults are as stable as anything measured (jumpy {scores[default]:.2f}% "
            f"against a best of {best:.2f}%): keep them.",
            warnings,
        )
    reason = (
        f"Lowest mean jumpy share ({scores[chosen]:.2f}%) among the {len(kept)} settings "
        f"that track within {TRACKED_MARGIN:g} points of the best on every recording"
    )
    if default in results:
        reason += f"; the defaults measured {mean_jumpy(results[default]):.2f}%"
    return Pick(chosen, False, reason + ".", warnings)
