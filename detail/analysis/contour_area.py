"""Estimate a starting value for the labeling stage's *Min contour area* from the video alone.

``LabelingConfig.min_contour_area`` is applied by ``core.utils.mask_to_polygon`` to each
external contour of a teacher mask: a piece smaller than the limit is cut out of the label,
and a detection is dropped only when every piece is. Set too low, specks of mask survive into
the training labels; set too high, real pieces of the subject (a part of the body the mask
split off) are cut. The right value depends on how big the subject is in the frame, which the
form cannot know. This module measures it without running the teacher.

**Species-agnostic by construction.** Nothing here knows what is being filmed or how big it
is: there is no size in pixels anywhere in the method. Every quantity that would be one is
measured on the selected recordings:

- **Foreground** is ``|frame - background|`` (so a subject lighter or darker than the
  background is found the same way) above a threshold chosen per recording from that
  difference itself (:func:`foreground_threshold`): Otsu's split of the pooled differences,
  never below the background's own noise (its median plus ``NOISE_SIGMAS`` robust standard
  deviations). A low-contrast subject is not lost to a fixed grey level, and a frame with
  nothing moving does not threshold into noise.
- **Smoothing scales with the subject.** A first pass with no smoothing finds the typical
  largest moving blob; its width scale is the square root of its area. The opening that
  removes specks and the closing that rejoins a split body are ellipses ``OPEN_SHARE`` and
  ``CLOSE_SHARE`` of that width (:func:`kernel_sides`), and are skipped when that comes to
  less than 3 px. A fixed kernel would erase a subject a few pixels across, or barely touch
  one that fills the frame.
- **How many subjects** is counted, not assumed (:func:`count_animals`): a blob at least
  ``ANIMAL_SHARE`` of the typical largest one counts as an animal, and the recording's count is
  the median over its frames. Each animal's blob is measured as an animal; a second animal is
  never reported as a leftover blob. The result names any recording with more than one.

**How it reads.** Per recording, the video is read sequentially (``grab``, no seeking), keeping
one grey frame every ``sample_every_s`` seconds of the timestamp map; their per-pixel median is
the empty background.

**Known limits**, the reason the result is a starting point and not an answer:

- A subject close in brightness to the background breaks into pieces under the subtraction,
  so its area is a **lower bound**, and the leftover blobs mix noise with fragments of it.
- A subject that stays in one place for most of the recording becomes part of the median
  background.
- Two subjects that touch are one blob on that frame; the median count over frames absorbs
  it, the area on that frame does not.
- Areas are pixel counts of components. The labeling stage filters on ``cv2.contourArea``,
  which is a little smaller for the same blob.

**The suggestion** (:func:`suggest`) compares two distributions measured here, nothing else.
The ceiling is half the smallest animal p1 over the recordings: a piece below it cannot be a
whole animal, even allowing for the subtraction underestimating it. The floor is the largest
leftover-blob **median**: the typical noise blob. Not its p95, because the largest leftover
blobs are partly fragments of the subject (the first limit above), so on a low-contrast
subject they reach into the subject's own range and would refuse every value. The suggestion
is the largest round number at or below the ceiling and above the floor. When the floor
reaches the ceiling the two distributions overlap, no value separates them, and the result
says so and suggests nothing.
"""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import math
from pathlib import Path

import numpy as np

#: The foreground threshold is never below the background's noise: the median difference plus
#: this many robust standard deviations (1.4826 x the median absolute deviation).
NOISE_SIGMAS = 5.0
#: A blob at least this share of the recording's typical largest blob counts as an animal.
ANIMAL_SHARE = 1 / 3
#: Opening and closing kernel sides, as a share of the subject's width scale (sqrt of its
#: typical area). A side under 3 px skips the operation.
OPEN_SHARE = 0.1
CLOSE_SHARE = 0.2
#: Pixels sampled from each frame's difference image for the threshold (a strided subset).
THRESHOLD_SAMPLE = 20_000
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

    ``animal`` maps each of :data:`ANIMAL_PERCENTILES` to the animals' blob area at that
    percentile, ``other`` each of :data:`OTHER_PERCENTILES` to the leftover blobs'. Either is
    empty when nothing of that kind was found. ``n_animals`` is how many animals were counted
    at once (:func:`count_animals`); ``threshold`` the foreground threshold, in grey levels.
    """

    recording_id: str
    name: str
    n_frames: int
    animal_min: float | None
    animal: dict[int, float]
    n_other: int
    other: dict[int, float]
    n_animals: int = 1
    threshold: float | None = None


@dataclass(frozen=True)
class MinAreaEstimate:
    """The per-recording table and the suggested limit.

    ``suggestion`` is None when no value separates the animals from the leftover blobs, or
    there was nothing to measure; ``reason`` says which, or how the value was reached.
    ``ceiling`` is half the smallest animal p1, ``floor`` the largest leftover-blob median.
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


def _difference(grey: np.ndarray, background: np.ndarray) -> np.ndarray:
    """``|grey - background|`` as uint8: a subject lighter or darker than the background alike."""
    import cv2

    return cv2.absdiff(grey, background)


def foreground_threshold(diffs: list[np.ndarray]) -> float:
    """The grey-level difference above which a pixel is foreground, from these differences.

    Otsu's split of the pooled differences, raised to the background's noise floor (median +
    ``NOISE_SIGMAS`` robust standard deviations, at least one grey level each). Otsu alone
    separates a subject of any contrast from the background when the subject is a fair share
    of the pixels; when it is a few pixels in a large frame, Otsu can split the noise instead,
    and the floor stops that.
    """
    import cv2

    pooled = []
    for diff in diffs:
        flat = diff.ravel()
        step = max(1, len(flat) // THRESHOLD_SAMPLE)
        pooled.append(flat[::step])
    values = np.concatenate(pooled) if pooled else np.zeros(1, np.uint8)
    otsu, _ = cv2.threshold(
        values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    median = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values.astype(np.float64) - median)))
    return max(float(otsu), median + NOISE_SIGMAS * max(sigma, 1.0))


def kernel_sides(width: float) -> tuple[int, int]:
    """``(open, close)`` ellipse sides for a subject of this width scale; odd, 0 = skip."""

    def side(share: float) -> int:
        value = int(round(share * width))
        if value < 3:
            return 0
        return value if value % 2 else value + 1

    return side(OPEN_SHARE), side(CLOSE_SHARE)


def blob_areas(
    diff: np.ndarray, threshold: float, open_side: int = 0, close_side: int = 0
) -> np.ndarray:
    """The foreground components' pixel areas on one difference image, largest first."""
    import cv2

    mask = (diff > threshold).astype(np.uint8)
    if open_side:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_side, open_side))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_side:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_side, close_side))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    return np.sort(stats[1:, cv2.CC_STAT_AREA].astype(np.int64))[::-1]


def count_animals(per_frame: list[np.ndarray]) -> tuple[int, float]:
    """``(n_animals, typical_largest)`` from each frame's areas (largest first).

    A blob at least ``ANIMAL_SHARE`` of the typical (median) largest blob counts as an animal;
    the count is the median over the frames that have any blob.
    """
    largest = [int(areas[0]) for areas in per_frame if len(areas)]
    if not largest:
        return 0, 0.0
    typical = float(np.median(largest))
    counts = [
        int(np.sum(areas >= ANIMAL_SHARE * typical))
        for areas in per_frame
        if len(areas)
    ]
    return max(1, int(round(float(np.median(counts))))), typical


@dataclass(frozen=True)
class Measured:
    """What :func:`measure` found on one recording's frames."""

    animal: np.ndarray
    other: np.ndarray
    n_animals: int
    threshold: float | None
    kernels: tuple[int, int]


def measure(frames: list[np.ndarray]) -> Measured:
    """Animal and leftover-blob areas over the kept frames of one recording.

    The background is their per-pixel median, the threshold :func:`foreground_threshold`. A
    first pass without smoothing sizes the subject; the second smooths with kernels scaled to
    it (:func:`kernel_sides`), counts the animals (:func:`count_animals`) and splits each
    frame's blobs: up to that many blobs of animal size are animals, the rest are leftovers.
    """
    empty = np.zeros(0, dtype=np.int64)
    if not frames:
        return Measured(empty, empty, 0, None, (0, 0))
    background = np.median(np.stack(frames), axis=0).astype(np.uint8)
    diffs = [_difference(grey, background) for grey in frames]
    threshold = foreground_threshold(diffs)
    _n, typical = count_animals([blob_areas(d, threshold) for d in diffs])
    if not typical:
        return Measured(empty, empty, 0, threshold, (0, 0))
    kernels = kernel_sides(math.sqrt(typical))
    per_frame = [blob_areas(d, threshold, *kernels) for d in diffs]
    n_animals, typical = count_animals(per_frame)
    animal: list[int] = []
    other: list[int] = []
    for areas in per_frame:
        if not len(areas):
            continue
        big = int(np.sum(areas >= ANIMAL_SHARE * typical))
        taken = min(n_animals, max(big, 1))
        animal.extend(int(a) for a in areas[:taken])
        other.extend(int(a) for a in areas[taken:])
    return Measured(
        np.asarray(animal, dtype=np.int64),
        np.asarray(other, dtype=np.int64),
        n_animals,
        threshold,
        kernels,
    )


def summarize(
    recording_id: str,
    name: str,
    n_frames: int,
    animal: np.ndarray,
    other: np.ndarray,
    n_animals: int = 1,
    threshold: float | None = None,
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
        n_animals=n_animals,
        threshold=threshold,
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
    animal p1 and above the largest leftover-blob median. ``None`` when nothing separates them.
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
    several = [f"{r.name} ({r.n_animals})" for r in recordings if r.n_animals > 1]
    if several:
        warnings.append(
            "More than one animal at once in "
            + ", ".join(several)
            + ": each is measured as an animal, not as a leftover blob. Where they touch "
            "they read as one blob on that frame."
        )
    overlapping = [r.name for r in noisy if r.other[95] >= ceiling]
    if overlapping:
        warnings.append(
            "The larger leftover blobs in "
            + ", ".join(overlapping)
            + " reach past the ceiling: they are probably pieces of the animal that the "
            "subtraction split off. Check the value with Test prompts."
        )
    if floor >= ceiling:
        return (
            None,
            ceiling,
            floor,
            f"No value separates them: the typical leftover blob ({floor:.0f} px) is as large "
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
        f"{smallest.name}) and above the typical leftover blob ({floor:.0f} px).",
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
    """Measure the animal and leftover-blob areas of each recording and suggest a limit.

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
        found = measure(frames)
        rows.append(
            summarize(
                video.recording_id,
                video.name,
                len(frames),
                found.animal,
                found.other,
                found.n_animals,
                found.threshold,
            )
        )
    value, ceiling, floor, reason, warnings = suggest(rows)
    report(("Done", 1.0, ""), force=True)
    return MinAreaEstimate(rows, value, ceiling, floor, reason, warnings)
