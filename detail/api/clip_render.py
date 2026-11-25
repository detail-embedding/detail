"""One forward decode pass per video part, cutting every clip that part contains (§16).

Clip exports want subject-cropped mp4s of many bouts scattered through a few long videos. The
first two (``syllable_export`` and ``cluster_export``, both since removed) rendered them one at a
time, which is what made that unusable.

The cost model is the whole reason this module exists. Random access into an H.264 file is not
random: ``CAP_PROP_POS_FRAMES`` has to decode forward from the preceding keyframe, so a bout late in
a 30-minute recording pays for most of the file — ~7 s of decode for a 2 s clip, and a per-*frame*
seek (the shape ``source_crop`` has, being a single-frame boundary) pays it again for every frame,
~40 s per clip. Either way the bill scales with clips × file length.

So: plan every clip first, group by part, then read each video **strictly forward exactly once**.
``grab()`` decodes without the colour convert to step over frames nothing wants; a full ``read()``
happens only inside a bout; several bouts may be open at once where they overlap. Cost becomes one
decode of the part no matter how many clips come out of it, and frame accuracy is exact because
every frame is counted forward and none is ever sought past.

Qt-free and preview-agnostic: ``render_part`` takes a plain part dict (the ``_build_parts`` shape)
and a list of jobs, so it is unit-testable without a preview at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from detail.api.run_preview import _global_to_part, _square_crop

_FALLBACK_FPS = 30.0

#: What a clip can have drawn on it. Three flavours, on three pages: raw, raw plus mask, and
#: mask.
RAW = "raw"
OVERLAY = "overlay"
MASK = "mask"
RENDERS = (RAW, OVERLAY, MASK)


def resolve_renders(renders: Sequence[str]) -> tuple[str, ...]:
    """Validate and de-duplicate a render request, keeping ``RENDERS`` order.

    Ordered by the catalogue rather than by the caller so two callers asking for the same set
    write the same files in the same order — which is what makes the CLI and the GUI comparable
    byte for byte.
    """
    wanted = list(dict.fromkeys(renders))
    unknown = [r for r in wanted if r not in RENDERS]
    if unknown:
        raise ValueError(
            f"unknown clip render(s) {unknown}; expected any of {list(RENDERS)}"
        )
    if not wanted:
        raise ValueError("ask for at least one clip render: " + ", ".join(RENDERS))
    return tuple(r for r in RENDERS if r in wanted)


def flavour_path(base: Path, render: str, suffixed: bool) -> Path:
    """``clip.mp4`` → ``clip_mask.mp4`` when more than one flavour shares a bout.

    A single-flavour export keeps the bare name, so a caller that asks for one render writes
    exactly what it wrote before this existed.
    """
    base = Path(base)
    if not suffixed:
        return base
    return base.with_name(f"{base.stem}_{render}{base.suffix}")


def box_map(part: dict, track_id: int) -> dict[int, np.ndarray]:
    """``{local frame -> bbox}`` for one track in one part, read once from the zarrs.

    The per-frame decode path re-derives this inside every ``frame()`` call; a clip renderer
    needs the whole range, so it is cheaper to materialize the map once (thousands of rows)
    than to pay the lookup per frame.
    """
    return {f: box for f, (box, _rle) in track_rows(part, track_id).items()}


def track_rows(part: dict, track_id: int) -> dict[int, tuple[np.ndarray, str]]:
    """``{local frame -> (bbox, mask_rle)}`` for one track in one part, read once.

    Both columns come off the same filtered rows, so a renderer that draws the mask and one
    that crops to the box pay a single pass over the zarr rather than one each.
    """
    reader, assignment = part.get("reader"), part.get("assignment")
    if reader is None or assignment is None or not reader.n_rows:
        return {}
    frames = np.asarray(reader.column("frame_idx")[:])
    boxes = reader.column("bbox")[:]
    rle = reader.column("mask_rle")[:]
    object_id = np.asarray(assignment.column("object_id")[:], dtype=np.int64)
    keep = object_id == int(track_id)
    return {
        int(f): (box, str(counts))
        for f, box, counts in zip(
            frames[keep], np.asarray(boxes)[keep], np.asarray(rle)[keep]
        )
    }


def fit_within(
    width: int,
    height: int,
    *,
    max_width: int | None = None,
    max_longest: int | None = None,
) -> tuple[int, int]:
    """Scale ``(width, height)`` down — never up — so the width is at most ``max_width`` and
    the longest side at most ``max_longest``, keeping the aspect; at least 2 by 2.

    Even dimensions: several mp4 encoders reject an odd width or height outright. This is the
    one rule for the size an exported picture takes — the clip renders here (longest side) and
    ``api.animation``'s GIF and mp4 writers (width) — so a frame is never sized two ways.
    """
    scale = 1.0
    if max_longest is not None:
        scale = min(scale, float(max_longest) / max(width, height))
    if max_width is not None:
        scale = min(scale, float(max_width) / width)
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def plan_optional(
    part_frame_counts: list[int],
    start_frame: int,
    end_frame: int,
    duration_s: float,
    out_path: Path,
) -> tuple[int, dict] | None:
    """``plan_job``, returning ``None`` for a bout that is not in the video at all.

    A bout past the end of its recording — the shape a re-import shorter than the run that
    analysed it leaves behind — must not take a whole export down with it. Every clip export
    calls this rather than ``plan_job`` directly, so the rule ``finish`` already states for an
    undetected bout ("one unreadable clip must not kill the run") holds for an out-of-range one
    too, in every exporter rather than in whichever was being edited.

    ``plan_job`` still raises, because for any other caller an out-of-range frame is a bug
    rather than a data condition.
    """
    try:
        return plan_job(part_frame_counts, start_frame, end_frame, duration_s, out_path)
    except IndexError:
        return None


def plan_job(
    part_frame_counts: list[int],
    start_frame: int,
    end_frame: int,
    duration_s: float,
    out_path: Path,
) -> tuple[int, dict]:
    """Resolve a bout's **global** frame range to ``(part_index, job)`` in local part frames.

    The job carries its own fps — the bout's true rate (frames spanned / its duration), so a clip
    plays back at the speed the behavior happened, not at a nominal 30. That holds because
    ``render_part`` writes every frame of the span: a clip that skipped frames at this rate
    played fast (#32).
    """
    lo_part, lo = _global_to_part(part_frame_counts, int(start_frame))
    hi_part, hi = _global_to_part(part_frame_counts, int(end_frame))
    if lo_part != hi_part:
        # A bout never spans a part in practice (runs split at part boundaries); clip to the part
        # it starts in rather than silently producing a jump-cut across two videos.
        hi = part_frame_counts[lo_part] - 1
    fps = (hi - lo) / duration_s if hi > lo and duration_s > 0 else _FALLBACK_FPS
    return lo_part, {
        "lo": lo,
        "hi": hi,
        "fps": max(1.0, min(120.0, fps)),
        "path": Path(out_path),
        "writers": {},
        "paths": {},
        "n_written": 0,
    }


def render_part(
    part: dict,
    jobs: list[dict],
    track_id: int,
    size: int,
    *,
    renders: Sequence[str],
    crop: bool = True,
    progress=None,
) -> None:
    """Render every job of one video part in a single forward pass, writing each clip in place.

    ``jobs`` are ``plan_job`` dicts in **local** part frames; each gets its ``n_written`` filled in
    (0 means nothing was decodable — no video, or the track has no detection across the bout — and
    the caller drops the empty file). They are sorted here, so the caller need not.

    **Every frame of a bout is written**, detected or not, because the clip's rate is the
    bout's own (``plan_job``): skipping the undetected frames at that rate played a stride-5
    run's clips 4.4 times too fast (#32). On a frame the track has no detection on:

    - ``raw`` at full frame needs nothing from the track and is the frame as recorded;
    - ``overlay`` is the plain frame (no mask to draw), and ``mask`` an empty, black one;
    - a crop (``crop=True``) holds the box of the bout's last detection before it, and before
      the bout's first detection uses that first one — so a clip depends only on its own
      span, whichever bouts are rendered beside it.

    A bout the track is **never** detected across writes nothing: there is no box to cut
    around and nobody to show, and the caller keeps the row with an empty clip.

    **Two orthogonal axes**, which used to be one flag. ``renders`` says what is *drawn*;
    ``crop`` says how the frame is *framed*. Conflating them is why "export the mask on its own"
    had nowhere to go: ``crop=True`` meant a raw cut and ``crop=False`` meant a blended overlay,
    so four of the six combinations did not exist.

    ``renders`` — several in one pass, which is the whole point:

    - ``raw`` — the source frame, nothing drawn on it.
    - ``overlay`` — the frame with the subject's mask blended over it.
    - ``mask`` — the mask alone, black and white, for building something
      else later — it is what a heat map gets built from outside DETAIL.

    Asking for all three costs one decode, not three. Decode dominates (this module exists for
    that reason), so separate calls would triple the run time to write the same three files.

    ``crop`` — which framing the clip is *for*:

    - ``crop=True`` — a square cut around the subject, ``size`` px on a side. The pose fills the
      frame at a constant scale, which is what makes bouts comparable side by side (and matches the
      input images the embedding actually saw). The cost is that the crop's side tracks the bbox, so
      the *background* appears to zoom as the animal rears and settles.
    - ``crop=False`` — the whole frame, scaled so its longest side is ``size``. A steady camera,
      which is what you want when the question is "what is this animal doing", since the
      surroundings are half the answer.

    With one render the output keeps the job's own filename; with several, each file takes its
    render as a suffix. So a caller that asks for one gets byte-identical output to before.

    ``renders`` is **required**, deliberately. It decides whether the output is what the caller
    meant — a default spelling "as before" is where a missing fact hides, and this parameter has
    two existing callers who want *different* answers (a raw crop and a full-frame overlay). The
    repo's recorded class for exactly this: a capability added to shared code as an optional
    parameter reaches only the surface being worked on.

    ``progress`` is called with each job as it finishes, so a caller can drive a bar per clip.

    Raises:
        ValueError: if ``renders`` is empty or names something that is not a render.
    """
    import cv2

    from detail.storage import video as video_store

    wanted = resolve_renders(renders)
    if not jobs:
        return
    for job in jobs:
        job["n_written"] = 0
        job["writers"] = {}
        job["paths"] = {
            r: flavour_path(job["path"], r, len(wanted) > 1) for r in wanted
        }
    ordered = sorted(jobs, key=lambda j: (j["lo"], j["hi"]))
    rows = track_rows(part, track_id)
    detected = np.array(sorted(rows), dtype=np.int64)
    for job in ordered:
        # The bout's first detection — what a crop starts on, and None for a bout the track is
        # absent from, which writes nothing. ``held`` is the last detection passed so far.
        at = int(np.searchsorted(detected, job["lo"]))
        first = int(detected[at]) if at < len(detected) else None
        job["first_detected"] = (
            first if first is not None and first <= job["hi"] else None
        )
        job["held"] = None
    cap = cv2.VideoCapture(str(part["video"]))
    if not cap.isOpened():
        return
    last = max(j["hi"] for j in ordered)
    # Both are constant for a part but need a decoded frame's shape, so they are built on the
    # first frame that is actually written and reused for the rest of the pass.
    out_size = (size, size)
    sample: tuple[np.ndarray, np.ndarray] | None = None
    # The size the part's masks were encoded at; the footage can decode at another (§9).
    encoded = part.get("mask_hw")
    try:
        pending, active, frame_idx = list(ordered), [], 0
        while frame_idx <= last:
            while pending and pending[0]["lo"] <= frame_idx:
                active.append(pending.pop(0))
            if not active:
                # No bout wants this frame: grab it (cheap — no colour convert) and move on.
                if not cap.grab():
                    break
                frame_idx += 1
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            row = rows.get(frame_idx)
            if not crop and sample is None:
                out_size = fit_within(bgr.shape[1], bgr.shape[0], max_longest=size)
                sample = _sample_index(bgr.shape[:2], out_size)
            # One decode, every requested flavour, shared by every bout that frames this frame
            # the same way. This is the saving: re-decoding the source once per flavour is what
            # a separate call per flavour would cost, and decode is the bill this whole module
            # exists to pay once.
            drawn: dict[tuple, dict[str, np.ndarray]] = {}
            for job in active:
                if job["first_detected"] is None:
                    continue  # never detected across this bout: nothing to show
                if row is not None:
                    job["held"] = frame_idx
                box_frame = (
                    job["first_detected"] if job["held"] is None else job["held"]
                )
                # The box only frames a crop; a full frame is the same for every bout.
                key = (box_frame if crop else None, row is not None)
                payloads = drawn.get(key)
                if payloads is None:
                    drawing = (rows[box_frame][0], row[1] if row is not None else "")
                    payloads = drawn[key] = {
                        render: np.ascontiguousarray(
                            _payload(
                                render,
                                bgr,
                                drawing,
                                size,
                                out_size,
                                sample,
                                crop,
                                encoded,
                            )
                        )
                        for render in wanted
                    }
                for render, payload in payloads.items():
                    writer = job["writers"].get(render)
                    if writer is None:
                        writer, _ = video_store.open_video_writer(
                            job["paths"][render], job["fps"], *out_size
                        )
                        job["writers"][render] = writer
                    writer.write(payload)
                job["n_written"] += 1
            frame_idx += 1
            for job in [j for j in active if j["hi"] < frame_idx]:
                active.remove(job)
                _close(job)
                if progress is not None:
                    progress(job)
    finally:
        cap.release()
        for job in ordered:
            _close(job)


def _payload(
    render: str,
    bgr: np.ndarray,
    row: tuple[np.ndarray, str],
    size: int,
    out_size: tuple[int, int],
    sample: tuple[np.ndarray, np.ndarray] | None,
    crop: bool,
    encoded: tuple[int, int] | None = None,
) -> np.ndarray:
    """One decoded frame, drawn and framed for one flavour. ``encoded`` is the size the
    part's masks were encoded at (``None``: the frame's own)."""
    import cv2

    box, mask_rle = row
    if render == MASK:
        binary = _mask_image(bgr.shape[:2], mask_rle, encoded)
        # `_square_crop` is channel-agnostic, so the mask crops with the same call the frame
        # does — and cropping *before* the resize keeps the mask's own edges rather than
        # inventing intermediate greys.
        if crop:
            return _square_crop(binary, box, size)
        return binary[sample[0]][:, sample[1]]
    if crop:
        # `_square_crop` is channel-agnostic — it slices, zero-pads and resizes — so cropping
        # the BGR frame directly saves converting a 1080p frame to RGB and the crop straight
        # back (5.4 ms a frame).
        cropped = _square_crop(bgr, box, size)
        if render == RAW:
            return cropped
        return _blend_cropped(cropped, mask_rle, bgr.shape[:2], box, size, encoded)
    if render == RAW:
        return cv2.resize(bgr, out_size, interpolation=cv2.INTER_AREA)
    return _overlay(bgr, mask_rle, out_size, sample, encoded)


def _mask_image(
    src: tuple[int, int], mask_rle: str, encoded: tuple[int, int] | None = None
) -> np.ndarray:
    """The subject's mask as a black-and-white BGR frame at the source resolution.

    Three channels because the mp4 writers take them; the values stay strictly 0 or 255, which
    is what makes this loadable as a binary mask on the other side rather than as a grey image
    somebody has to threshold and guess at.
    """
    from detail.core.utils import decode_mask

    if not mask_rle:
        return np.zeros((*src, 3), np.uint8)
    mask = decode_mask(mask_rle, encoded or src, out=src)
    return np.repeat((mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)


def _blend_cropped(
    cropped: np.ndarray,
    mask_rle: str,
    src: tuple[int, int],
    box: np.ndarray,
    size: int,
    encoded: tuple[int, int] | None = None,
) -> np.ndarray:
    """A cropped frame with the subject's mask blended on, in the crop's own frame.

    The mask is cropped through a **read-only three-channel view** of the decoded one, not a
    materialised 0/255 image: building that image copied ~6 MB per 1080p frame only for all but
    the crop to be thrown away. The view is several times faster with identical pixels;
    the copy was the largest cost in building a node cache
    and, because it holds the GIL, what serialised the parts decoded in parallel.
    """
    from detail.core.utils import decode_mask
    from detail.core.visualize import MASK_TINT, bgr_to_rgb, draw_mask, rgb_to_bgr

    if not mask_rle:
        return cropped
    ones = decode_mask(mask_rle, encoded or src, out=src).view(np.uint8)
    window = _square_crop(np.broadcast_to(ones[:, :, None], (*src, 3)), box, size)
    return rgb_to_bgr(draw_mask(bgr_to_rgb(cropped), window[:, :, 0] > 0, MASK_TINT))


def _sample_index(
    src: tuple[int, int], out_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Row/column indices that subsample a ``src`` (h, w) mask down to ``out_size`` (w, h).

    Constant for a whole part, so it is built once and reused rather than paid per frame.
    """
    height, width = src
    out_w, out_h = out_size
    return (
        np.arange(out_h) * height // out_h,
        np.arange(out_w) * width // out_w,
    )


def _overlay(
    bgr: np.ndarray,
    mask_rle: str,
    out_size: tuple[int, int],
    sample: tuple[np.ndarray, np.ndarray],
    encoded: tuple[int, int] | None = None,
) -> np.ndarray:
    """The frame at ``out_size`` with the subject's mask blended on (BGR in, BGR out).

    **Downscale first, then blend.** Blending a 1080p frame and shrinking the result gives a
    marginally smoother mask edge, but it costs about ten times as much a frame, which
    turned an export of minutes into one of over an hour.
    At a third scale the difference is sub-pixel; the run time is not.

    The mask is subsampled by index rather than ``cv2.resize`` for the same reason: it is
    several times faster, and nearest-neighbour is what a bool mask supports anyway.
    """
    import cv2

    from detail.core.utils import decode_mask
    from detail.core.visualize import MASK_TINT, bgr_to_rgb, draw_mask, rgb_to_bgr

    out = cv2.resize(bgr, out_size, interpolation=cv2.INTER_AREA)
    if not mask_rle:
        return out
    src = bgr.shape[:2]
    mask = decode_mask(mask_rle, encoded or src, out=src)[sample[0]][:, sample[1]]
    return rgb_to_bgr(draw_mask(bgr_to_rgb(out), mask, MASK_TINT))


def _close(job: dict) -> None:
    for render, writer in list(job.get("writers", {}).items()):
        if writer is not None:
            writer.release()
            job["writers"][render] = None


def written_fps(job: dict | None) -> float | str:
    """The rate ``job``'s clip was written at, for its CSV row; blank when it wrote nothing.

    The rate the file plays at, so a reader can check a clip against its ``dur_s``: frames
    over this rate is the bout's duration, to within a frame (#32).
    """
    if not job or not job.get("n_written"):
        return ""
    return round(float(job["fps"]), 3)


def finish(job: dict) -> dict[str, str]:
    """Drop clips that got no frames; return ``{render: filename}`` for the ones that survived.

    A bout the track was never detected across produces no video; recording an empty result keeps
    the row (its times and rank are still true) instead of letting one unreadable clip kill the
    run. Returns a map rather than a name because one job can now write several flavours from the
    same decode.
    """
    paths = job.get("paths") or {}
    if job.get("n_written", 0) == 0:
        for path in paths.values():
            Path(path).unlink(missing_ok=True)
        return {}
    return {render: Path(path).name for render, path in paths.items()}
