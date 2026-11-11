"""What a clip's animal is doing, as curves over the clip's frames — the strips under the clip
window's slider (§22).

**Built on the §14 kinematics sidecar through `frame_slices.from_kinematics`, never on a decode
of the detection store.** The sidecar already holds every embedded frame's mask centroid and area,
so the clip window reads what the embed pass wrote rather than decoding RLE masks a second time;
speed is `analysis.descriptors.step_curve` over the centroid — the one step rule, shared with the
research dynamics panels so the two cannot drift about what "speed" means — divided by the row's
own time step. A run from before the sidecar existed has none: the curves come back empty with a
note naming `backfill-kinematics`, and nothing here computes a substitute.

**At the embedding's cadence, which is the cadence the clustering saw.** A row is an embedded
frame; frames between rows are not samples, and frames with no row at all (before the track was
found, across a tracking gap, after it was lost) are reported as ``untracked`` spans so a strip
breaks its line there rather than drawing through time nobody observed.

**Units.** Pixels, or millimetres when the recording is calibrated. Not body lengths: the repo
already carried three definitions of "one body size", and a strip is scaled to its own range, so
a dip reads as a dip in any unit.

The body's shape — length, bend, turning — is `analysis.body_axis.body_shape_series` over the
keypoint rows at the same frames, when the caller has keypoints and a heading definition to
hand. Pure over arrays and paths: no Qt, no caches; the preview that caches is `run_preview`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from detail.analysis import body_axis
from detail.analysis.descriptors import FrameSlice, step_curve
from detail.api import frame_slices

#: One frame of footage, in nanoseconds, below which two rows cannot be told apart in time.
_MIN_STEP_NS = 1


@dataclass(frozen=True)
class Curve:
    """One strip: a value per row of :attr:`ClipCurves.frames` (NaN where unknown), what it
    is, and how its range is drawn — about a zero line when ``signed`` (a bend, a turn), from
    zero when ``floor_zero`` (a speed, an area), its own range otherwise."""

    name: str
    label: str
    unit: str
    values: np.ndarray
    signed: bool = False
    floor_zero: bool = False


@dataclass(frozen=True)
class ClipCurves:
    """The curves over one window of a clip.

    ``window`` is the inclusive global frames the bar spans; ``frames`` the embedded frames
    inside it, ascending, one row of every curve each; ``t_ns`` their times (int64, the
    sidecar's clock); ``untracked`` the spans of window frames with no row — before the first,
    after the last, and between two rows that are not adjacent at the run's stride; ``note``
    the units, or why there is nothing.
    """

    window: tuple[int, int]
    frames: np.ndarray
    t_ns: np.ndarray
    curves: list[Curve] = field(default_factory=list)
    untracked: list[tuple[int, int]] = field(default_factory=list)
    note: str = ""

    @property
    def names(self) -> list[str]:
        return [curve.name for curve in self.curves]


def read_clip_curves(
    recording_dir: Path | str,
    *,
    embed_run_id: str,
    track_id: int,
    first: int,
    last: int,
    pixels_per_mm: float | None,
    pose: tuple[np.ndarray, np.ndarray, body_axis.BodyAxis] | None = None,
) -> ClipCurves:
    """The curves of ``track_id`` over the inclusive global frames ``[first, last]``.

    ``pixels_per_mm`` (``Recording.pixels_per_mm``) switches speed and area to millimetres;
    ``None`` keeps pixels. ``pose`` is ``(points, scores, axis)`` with ``points`` ``(n, K, 2)``
    normalized and ``scores`` ``(n, K)`` for **every** window frame in order (what the clip
    window's keypoint reader hands over); the rows at the embedded frames are taken from it.

    Raises:
        ValueError: ``last`` before ``first``, or a ``pose`` whose rows do not cover the
            window.
    """
    first, last = int(first), int(last)
    if last < first:
        raise ValueError(f"a clip window runs forward: [{first}, {last}] does not")
    if pose is not None and len(pose[0]) != last - first + 1:
        raise ValueError(
            f"pose has {len(pose[0])} rows for a window of {last - first + 1} frames"
        )
    slice_ = frame_slices.from_kinematics(
        recording_dir, embed_run_id=embed_run_id, track_id=int(track_id)
    )
    if slice_ is None:
        return ClipCurves(
            window=(first, last),
            frames=np.empty(0, dtype=np.int64),
            t_ns=np.empty(0, dtype=np.int64),
            note=(
                f"no kinematics for {embed_run_id}: run `detail recording "
                "backfill-kinematics` (Recording.backfill_kinematics) to write them"
            ),
        )

    scale, unit = (1.0 / float(pixels_per_mm), "mm") if pixels_per_mm else (1.0, "px")
    speed = speed_curve(slice_, scale=scale, unit=unit)
    area = area_curve(slice_, scale=scale, unit=unit)

    frame_idx = np.asarray(slice_.frame_idx, dtype=np.int64)
    lo = int(np.searchsorted(frame_idx, first, side="left"))
    hi = int(np.searchsorted(frame_idx, last, side="right"))
    frames = frame_idx[lo:hi]
    curves = [
        Curve(
            speed.name, speed.label, speed.unit, speed.values[lo:hi], floor_zero=True
        ),
        Curve(area.name, area.label, area.unit, area.values[lo:hi], floor_zero=True),
    ]
    if pose is not None and len(frames):
        points, scores, axis = pose
        rows = frames - first
        shape = body_axis.body_shape_series(
            np.asarray(points)[rows], np.asarray(scores)[rows], slice_.t_ns[lo:hi], axis
        )
        curves.append(Curve("length", "body length", "x typical", shape["length"]))
        if "bend" in shape:
            curves.append(Curve("bend", "bend", "°", shape["bend"], signed=True))
        curves.append(Curve("turning", "turning", "°/s", shape["turning"], signed=True))
    return ClipCurves(
        window=(first, last),
        frames=frames,
        t_ns=np.asarray(slice_.t_ns, dtype=np.int64)[lo:hi],
        curves=curves,
        untracked=untracked_spans(frame_idx, slice_.stride, first, last),
        note=f"speed and area in {unit}",
    )


def speed_curve(slice_: FrameSlice, *, scale: float, unit: str) -> Curve:
    """How fast the subject's centre moved into each row, in ``unit`` per second: the
    `descriptors.step_curve` of its position over the row's own time step. NaN at the first
    row, across a tracking gap, and where two rows share a time."""
    position = slice_.position
    frame_idx = np.asarray(slice_.frame_idx, dtype=np.int64)
    t_ns = np.asarray(slice_.t_ns, dtype=np.int64)
    n = len(frame_idx)
    values = np.full(n, np.nan)
    if n >= 2 and position is not None:
        step = step_curve(position, frame_idx, slice_.stride)
        dt_ns = np.full(n, np.nan)
        dt_ns[1:] = np.diff(t_ns).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(dt_ns >= _MIN_STEP_NS, step / (dt_ns / 1e9), np.nan)
    return Curve("speed", "speed", f"{unit}/s", values * scale, floor_zero=True)


def area_curve(slice_: FrameSlice, *, scale: float, unit: str) -> Curve:
    """The subject's mask area at each row, in square ``unit``; NaN where no mask was found."""
    area = slice_.area
    values = (
        np.full(len(slice_.frame_idx), np.nan)
        if area is None
        else np.asarray(area, dtype=np.float64) * scale * scale
    )
    return Curve("area", "area", f"{unit}²", values, floor_zero=True)


def untracked_spans(
    frame_idx: np.ndarray, stride: int, first: int, last: int
) -> list[tuple[int, int]]:
    """The inclusive spans of ``[first, last]`` with no embedded row: before the first row in
    the window, after the last, and between two rows that are not adjacent at ``stride``
    (`descriptors.adjacent`'s rule). The whole window, when it holds no row at all."""
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    rows = frame_idx[(frame_idx >= first) & (frame_idx <= last)]
    if not len(rows):
        return [(int(first), int(last))]
    spans: list[tuple[int, int]] = []
    if rows[0] > first:
        spans.append((int(first), int(rows[0]) - 1))
    for a, b in zip(rows[:-1], rows[1:]):
        if int(b) - int(a) != int(stride):
            spans.append((int(a) + 1, int(b) - 1))
    if rows[-1] < last:
        spans.append((int(rows[-1]) + 1, int(last)))
    return spans
