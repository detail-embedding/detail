"""One mp4 of one frame range of one recording — the clip a player is showing (#354).

**Why this is its own door.** `behavior_clips.export_behavior_clips` writes a clip *per bout of
an artifact*, which is the right shape for a clustering result and the wrong one for the thing a
user is actually looking at: a span on a scrub bar, which is often not an artifact instance at
all — a query match, a calibrate preview, a window somebody dragged. The two share every line
below the planning (`clip_render.plan_job` → `render_part` → `finish`), so a clip exported from a
player and a clip exported from a run are the same bytes.

**The `preview` seam.** Passing an open `ClipQueryPreview` is how the caller avoids paying for a
second decode context, and how a test drives the render without a whole experiment on disk —
exactly the arrangement `export_behavior_clips` already uses (and the removed `syllable_export`
did).

**Never share a live player's preview.** A `ClipQueryPreview` holds one mutable `FrameCursor`,
closed and rebuilt on a part change, so handing a GUI player's own instance to an export worker
races its decode thread. The export run builds its own; only a caller that knows it is
single-threaded should pass one.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from detail.api import clip_render
from detail.core.utils import subject_key
from detail.core.video_codecs import DEFAULT_VIDEO_CODEC

#: What the `clip.csv` beside a clip carries — what it is, and where in the source it came from,
#: so the mp4 can be put back in the recording without opening it.
CLIP_COLUMNS = [
    "recording_id", "subject", "start_frame", "end_frame",
    "start_t_ns", "end_t_ns", "dur_s", "fps",
]  # fmt: skip


def clip_file_name(recording_id: str, start: int, end: int) -> str:
    """``<recording>_<start>-<end>.mp4`` — zero-padded, so a folder sorts in time order."""
    return f"{recording_id}_{int(start):06d}-{int(end):06d}.mp4"


def export_clip(
    experiment_dir: Path | str,
    dest: Path | str,
    *,
    recording_id: str,
    frames: tuple[int, int],
    embed_run_id: str,
    track_id: int = 0,
    renders: Sequence[str] = (clip_render.OVERLAY,),
    crop: bool = False,
    size: int = 640,
    start_t_ns: int | None = None,
    end_t_ns: int | None = None,
    progress=None,
    preview=None,
    codec: str = DEFAULT_VIDEO_CODEC,
) -> list[Path]:
    """Write one clip of ``frames`` into ``dest``, plus the ``clip.csv`` behind it.

    Args:
        experiment_dir: the experiment holding the recording.
        dest: directory to write into; must already exist.
        recording_id: the recording the frames belong to.
        frames: inclusive, recording-**global** ``(start, end)``.
        embed_run_id: the embedding run the frames are read through; it resolves the tracking
            and detection runs the masks come from.
        track_id: which subject to outline and, when ``crop``, to cut around.
        renders: any of `clip_render.RENDERS`. Several cost one decode, not several.
        crop: a square cut around the subject rather than the whole field of view.
        size: longest side of the output (the crop's side when ``crop``).
        start_t_ns / end_t_ns: the span's true times, if the caller already knows them. They set
            the clip's playback rate — `plan_job` derives fps from the duration, so a clip plays
            at the speed the behavior happened rather than at a nominal 30. Resolved from the
            recording when omitted.
        progress: a ``(stage, fraction, detail)`` callback.
        preview: an open `ClipQueryPreview` to reuse; one is opened if omitted.

    Returns:
        Every path written, ``clip.csv`` last. Empty when the range is not in the video.
    """
    from detail.api.run_preview import ClipQueryPreview

    dest = Path(dest)
    lo, hi = (int(frames[0]), int(frames[1]))
    lo, hi = min(lo, hi), max(lo, hi)
    wanted = clip_render.resolve_renders(renders)
    if preview is None:
        preview = ClipQueryPreview(experiment_dir)
    scrub, _t_ns = preview._scrub_ctx(recording_id, embed_run_id)

    if start_t_ns is None:
        start_t_ns = preview.frame_t_ns(recording_id, embed_run_id, lo)
    if end_t_ns is None:
        end_t_ns = preview.frame_t_ns(recording_id, embed_run_id, hi)
    duration_s = (int(end_t_ns) - int(start_t_ns)) / 1e9

    out_path = dest / clip_file_name(recording_id, lo, hi)
    planned = clip_render.plan_optional(
        scrub.part_frame_counts, lo, hi, duration_s, out_path
    )
    if planned is None:
        return (
            []
        )  # the range is not in the video; nothing to write and nothing to index
    part_index, job = planned

    if progress is not None:
        progress("rendering", 0.0, out_path.name)
    clip_render.render_part(
        scrub._parts[part_index],
        [job],
        subject_key(track_id),
        int(size),
        renders=wanted,
        crop=bool(crop),
        codec=codec,
    )
    names = clip_render.finish(job)

    written = [dest / name for name in names.values()]
    index = dest / "clip.csv"
    with open(index, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CLIP_COLUMNS + [f"video_{render}" for render in wanted])
        writer.writerow(
            [
                recording_id, subject_key(track_id), lo, hi,
                int(start_t_ns), int(end_t_ns),
                round(duration_s, 3), round(job["fps"], 3),
            ]  # fmt: skip
            + [names.get(render, "") for render in wanted]
        )
    written.append(index)
    if progress is not None:
        progress("rendering", 1.0, "done")
    return written
