"""Estimate a starting value for the labeling stage's *Min contour area* from the video alone.

``LabelingConfig.min_contour_area`` is applied by ``core.utils.mask_to_polygon`` to each
external contour of a teacher mask: a piece smaller than the limit is cut out of the label,
and a detection is dropped only when every piece is. Set too low, specks of mask survive into
the training labels; set too high, real pieces of the animal (a tail that detached from the
body) are cut. The right value depends on how big the animal is in the frame, which the form
cannot know. This module measures it without running the teacher.

**How.** Per recording, read the video sequentially (``grab``, no seeking) and keep one grey
frame every ``sample_every_s`` seconds of the timestamp map. The per-pixel median of those
frames is the empty background. On each kept frame, threshold ``|frame - background| >
DIFF_THRESHOLD``, open with a 5x5 ellipse, close with a 9x9 ellipse, and label the connected
components. The largest component is the animal; every other one is an "other blob".

**Known limits**, the reason the result is a starting point and not an answer:

- A dark animal on a dark background breaks into pieces under the subtraction, so the animal
  area is a **lower bound**, and the other blobs mix noise with fragments of the animal. On the
  Linear Track mirror crops the animal read about 1,900-2,700 px here, and the teacher's masks
  about 4,000-6,500 px by eye.
- One animal is assumed: with two in a frame, the second is an "other blob".
- An animal that stays in one place for most of the recording becomes part of the median
  background.
- Areas are pixel counts of components. The labeling stage filters on ``cv2.contourArea``,
  which is a little smaller for the same blob.

**The suggestion** (:func:`suggest`). The ceiling is half the smallest animal p1 over the
recordings: a piece below it cannot be a whole animal, even allowing for the subtraction
underestimating it. The floor is the largest other-blob **median**: the typical noise blob.
Not the p95, because the other blobs' upper tail is partly the animal's own fragments (the
first limit above), so on a dark animal the p95 reaches into the animal's range and would
refuse every value. The suggestion is the largest round number at or below the ceiling and
above the floor. When the floor reaches the ceiling there is no value that separates them,
and the result says so and suggests nothing rather than guessing.
"""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import math
from pathlib import Path

import numpy as np

#: A pixel is foreground when its grey level differs from the background by more than this.
DIFF_THRESHOLD = 20
#: Opening removes specks smaller than this ellipse; closing fills gaps narrower than this one.
OPEN_KERNEL = 5
CLOSE_KERNEL = 9
#: One sampled frame per this many seconds of video.
SAMPLE_EVERY_S = 4.0
#: At most this many sampled frames per recording; a longer recording samples more sparsely.
MAX_SAMPLES = 450
#: The kept frames of one recording are held in memory together for the median. This caps
#: their total pixel count (about 200 MB of grey frames), so a high-resolution video samples
#: fewer frames rather than exhausting memory.
PIXEL_BUDGET = 200_000_000
#: Below this many sampled frames the median is a poor background (a short clip), and the
#: result says so.
MIN_FRAMES = 10

ANIMAL_PERCENTILES = (1, 5, 50, 95, 99)
OTHER_PERCENTILES = (50, 95, 99)

Progress = Callable[[str, float, str], None]


@dataclass(frozen=True)
class RecordingVideo:
    """What the estimate reads from one recording: its video parts and their timestamp maps.

    ``parts`` is ``[(video_path, t_ns), ...]`` in part order, ``t_ns`` the part's active map
    (nanoseconds from the start of the first part, Tenet 8). Its length is the part's frame
    count. ``resolution`` is ``(width, height)`` when known; it sizes the frame sample.
    """

    recording_id: str
    name: str
    parts: list[tuple[Path, np.ndarray]]
    resolution: tuple[int, int] | None = None


@dataclass(frozen=True)
class RecordingAreas:
    """One recording's measured areas, in pixels.

    ``animal`` maps each of :data:`ANIMAL_PERCENTILES` to the largest component's area at that
    percentile, ``other`` each of :data:`OTHER_PERCENTILES` to the other blobs'. Either is
    empty when nothing of that kind was found.
    """

    recording_id: str
    name: str
    n_frames: int
    animal_min: float | None
    animal: dict[int, float]
    n_other: int
    other: dict[int, float]


@dataclass(frozen=True)
class MinAreaEstimate:
    """The per-recording table and the suggested limit.

    ``suggestion`` is None when no value separates the animal from the other blobs, or there
    was nothing to measure; ``reason`` says which, or how the value was reached. ``ceiling`` is
    half the smallest animal p1, ``floor`` the largest other-blob median.
    """

    per_recording: list[RecordingAreas]
    suggestion: int | None
    ceiling: float | None
    floor: float | None
    reason: str
    warnings: list[str] = field(default_factory=list)


def sample_indices(
    parts_t_ns: list[np.ndarray],
    *,
    sample_every_s: float = SAMPLE_EVERY_S,
    max_samples: int = MAX_SAMPLES,
) -> list[np.ndarray]:
    """Per part, the sorted local frame indices to keep: one per ``sample_every_s``.

    Spaced on the timestamp map, so a variable-rate video is sampled evenly in time. A
    recording longer than ``max_samples * sample_every_s`` widens the spacing to stay within
    ``max_samples``. Each part's frame count is the length of its map.
    """
    non_empty = [t for t in parts_t_ns if len(t)]
    if not non_empty or max_samples < 1:
        return [np.zeros(0, dtype=np.int64) for _ in parts_t_ns]
    start = int(non_empty[0][0])
    span = int(non_empty[-1][-1]) - start
    step = max(int(sample_every_s * 1e9), math.ceil(span / max_samples), 1)
    targets = start + step * np.arange(span // step + 1, dtype=np.int64)
    out = []
    for t in parts_t_ns:
        if not len(t):
            out.append(np.zeros(0, dtype=np.int64))
            continue
        inside = targets[(targets >= t[0]) & (targets <= t[-1])]
        idx = np.searchsorted(t, inside, side="left")
        out.append(np.unique(np.clip(idx, 0, len(t) - 1)).astype(np.int64))
    return out


def blob_areas(grey: np.ndarray, background: np.ndarray) -> np.ndarray:
    """The foreground components' pixel areas on one grey frame, largest first."""
    import cv2

    diff = np.abs(grey.astype(np.int16) - background.astype(np.int16))
    mask = (diff > DIFF_THRESHOLD).astype(np.uint8)
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
    closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing)
    _n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    return np.sort(stats[1:, cv2.CC_STAT_AREA].astype(np.int64))[::-1]


def measure(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``(animal_areas, other_areas)`` over the kept frames of one recording.

    The background is their per-pixel median. A frame with no component contributes no
    animal area.
    """
    if not frames:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    background = np.median(np.stack(frames), axis=0).astype(np.uint8)
    animal: list[int] = []
    other: list[int] = []
    for grey in frames:
        areas = blob_areas(grey, background)
        if len(areas):
            animal.append(int(areas[0]))
            other.extend(int(a) for a in areas[1:])
    return np.asarray(animal, dtype=np.int64), np.asarray(other, dtype=np.int64)


def summarize(
    recording_id: str,
    name: str,
    n_frames: int,
    animal: np.ndarray,
    other: np.ndarray,
) -> RecordingAreas:
    """One recording's percentile row from its raw areas."""

    def pct(values: np.ndarray, which: tuple[int, ...]) -> dict[int, float]:
        if not len(values):
            return {}
        return {p: float(v) for p, v in zip(which, np.percentile(values, which))}

    return RecordingAreas(
        recording_id=recording_id,
        name=name,
        n_frames=n_frames,
        animal_min=float(animal.min()) if len(animal) else None,
        animal=pct(animal, ANIMAL_PERCENTILES),
        n_other=int(len(other)),
        other=pct(other, OTHER_PERCENTILES),
    )


def round_down(value: float) -> int:
    """The largest "round" number at or below ``value``: a multiple of half its magnitude.

    476 -> 450, 1189 -> 1000, 97 -> 95, 8 -> 8.
    """
    if value < 10:
        return max(int(value), 0)
    step = 10 ** int(math.floor(math.log10(value))) // 2
    return int(value // step) * step


def suggest(
    recordings: list[RecordingAreas],
) -> tuple[int | None, float | None, float | None, str, list[str]]:
    """``(suggestion, ceiling, floor, reason, warnings)`` for these recordings.

    The rule is the module docstring's: the largest round number at or below half the smallest
    animal p1 and above the largest other-blob median. ``None`` when nothing separates them.
    """
    measured = [r for r in recordings if r.animal]
    if not measured:
        return (
            None,
            None,
            None,
            "No moving animal was found in the sampled frames, so there is nothing to "
            "size the limit by.",
            [],
        )
    smallest = min(measured, key=lambda r: r.animal[1])
    ceiling = smallest.animal[1] / 2
    noisy = [r for r in recordings if r.other]
    floor = max((r.other[50] for r in noisy), default=0.0)
    warnings = []
    few = [r.name for r in recordings if r.n_frames < MIN_FRAMES]
    if few:
        warnings.append(
            f"Fewer than {MIN_FRAMES} frames were sampled in "
            + ", ".join(few)
            + ", so "
            "the background there is a poor median and the areas are unreliable."
        )
    overlapping = [r.name for r in noisy if r.other[95] >= ceiling]
    if overlapping:
        warnings.append(
            "The larger other blobs in "
            + ", ".join(overlapping)
            + " reach past the ceiling: they are probably pieces of the animal that the "
            "subtraction split off. Check the value with Test prompts."
        )
    if floor >= ceiling:
        return (
            None,
            ceiling,
            floor,
            f"No value separates them: the typical other blob ({floor:.0f} px) is as large "
            f"as half the smallest animal ({ceiling:.0f} px, in {smallest.name}). The "
            "subtraction may be breaking the animal into pieces; set the value by eye with "
            "Test prompts.",
            warnings,
        )
    value = round_down(ceiling)
    if value <= floor:
        value = int(ceiling)
    return (
        value,
        ceiling,
        floor,
        f"{value} px: at or below half the smallest animal ({ceiling:.0f} px, the p1 of "
        f"{smallest.name}) and above the typical other blob ({floor:.0f} px).",
        warnings,
    )


def _read_samples(
    video: RecordingVideo,
    wanted: list[np.ndarray],
    on_frame: Callable[[int], None],
) -> list[np.ndarray]:
    """The wanted frames of every part, grey, read sequentially. ``on_frame(n)`` after each
    grabbed frame, ``n`` counting this recording's grabbed frames so far."""
    import cv2

    frames: list[np.ndarray] = []
    grabbed = 0
    for (path, _t_ns), indices in zip(video.parts, wanted):
        if not len(indices):
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(
                f"recording {video.name!r}: cannot open its video {Path(path).name!r}; "
                "locate the video and try again"
            )
        try:
            keep = set(int(i) for i in indices)
            last = int(indices[-1])
            for i in range(last + 1):
                if not cap.grab():
                    break
                if i in keep:
                    ok, frame = cap.retrieve()
                    if ok:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                grabbed += 1
                on_frame(grabbed)
        finally:
            cap.release()
    return frames


def _plan(
    video: RecordingVideo, sample_every_s: float, max_samples: int
) -> list[np.ndarray]:
    limit = max_samples
    if video.resolution is not None:
        width, height = video.resolution
        limit = max(1, min(limit, PIXEL_BUDGET // max(width * height, 1)))
    return sample_indices(
        [t for _path, t in video.parts],
        sample_every_s=sample_every_s,
        max_samples=limit,
    )


def estimate(
    videos: list[RecordingVideo],
    *,
    progress: Progress | None = None,
    sample_every_s: float = SAMPLE_EVERY_S,
    max_samples: int = MAX_SAMPLES,
) -> MinAreaEstimate:
    """Measure the animal and other-blob areas of each recording and suggest a limit.

    ``progress(stage, fraction, detail)`` is driven while each recording is read (weighed by
    the frames to read, which is nearly all the time) and when it is measured. It is called
    at most every ``core.progress.PROGRESS_INTERVAL_S`` while reading, and is where a caller's
    cancel lands.

    Raises:
        ValueError: If ``videos`` is empty.
        FileNotFoundError: If a recording's video cannot be opened.
    """
    from detail.core.progress import ThrottledProgress

    if not videos:
        raise ValueError("choose at least one recording with video to estimate from")
    plans = [_plan(v, sample_every_s, max_samples) for v in videos]
    weights = [sum(int(ix[-1]) + 1 for ix in plan if len(ix)) for plan in plans]
    total = max(sum(weights), 1)
    report = ThrottledProgress(
        (lambda event: progress(*event)) if progress is not None else None
    )
    rows = []
    done = 0
    for n, (video, plan, weight) in enumerate(zip(videos, plans, weights), start=1):
        of = f"recording {n} of {len(videos)}"
        stage = f"Reading {video.name}"
        report((stage, done / total, of), force=True)

        def on_frame(
            grabbed: int, stage=stage, of=of, base=done, weight=weight
        ) -> None:
            report(
                (
                    stage,
                    (base + grabbed) / total,
                    f"{of} · frame {grabbed:,}/{weight:,}",
                )
            )

        frames = _read_samples(video, plan, on_frame)
        done += weight
        report((f"Measuring {video.name}", done / total, of), force=True)
        animal, other = measure(frames)
        rows.append(
            summarize(video.recording_id, video.name, len(frames), animal, other)
        )
    value, ceiling, floor, reason, warnings = suggest(rows)
    report(("Done", 1.0, ""), force=True)
    return MinAreaEstimate(rows, value, ceiling, floor, reason, warnings)
