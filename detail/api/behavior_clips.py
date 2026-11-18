"""Clips for any §19 behavior artifact — one exporter, every producer (#336).

**Why this exists when two clip exporters already did.** `cluster_export` served a persisted
clusterer and `syllable_export` a syllable-discovery run, and `detail/surfaces.py` hid *both* of
those subjects (all four are since removed). The surfaced clustering path — pose and temporal
clustering, read through `ClusterWorkspace` — had no clip export at all. That is the screen the
bug was reported from: the capability existed twice and was reachable from neither page the user
was on.

So this one is written against the **shared artifact** (`storage/behavior_output.py`) rather than
against a module. Every §19 producer — temporal clustering, Query, annotation —
gets clips from it, and a producer added later gets them for free. `BehaviorInstance` already
carries everything a clip needs: the recording, the subject, and the frame range.

Decoding goes through `clip_render`, so several flavours cost one forward pass per video part
rather than one per clip per flavour.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from detail.api import clip_render
from detail.api.run_preview import ClipQueryPreview
from detail.core.utils import safe_path_part, unique_stems

#: clips.csv — the row behind every clip, so the figure can be rebuilt outside DETAIL. This is
#: the "data required to recreate this video file" half, and it is why the clip target declares
#: `inputs` rather than `inputs_reason`.
CLIP_COLUMNS = [
    "instance_id", "source_label", "label", "label_path", "recording_id", "subject",
    "start_frame", "end_frame", "t_ns", "end_t_ns", "dur_s", "fps",
]  # fmt: skip


def _clip_stem(source_label: str, recording_id: str, start: int, end: int) -> str:
    stem = f"{source_label}_{recording_id}_{int(start):06d}-{int(end):06d}"
    return safe_path_part(stem, allow_dot=True)


def clip_names(rows: Sequence) -> list[str]:
    """``<cluster>_<recording>_<start>-<end>.mp4`` for every row, **one file per row**.

    The cluster comes first so a name-sorted file browser groups a cluster's clips together and
    plays them back to back — reviewing a whole result in one pass is the point, rather than
    opening a folder per cluster. The zero-padded frame range orders them in time within it, and
    still says where to look in the original video.

    Two rows can name one clip: two subjects matched on the same span of one recording, or two
    cluster names that clean to one stem (``a/b`` and ``a_b``). Those each carry their subject
    (``_subject1``), and a number if that still does not separate them, so neither is written
    over the other — and an empty clip, which is deleted by name, deletes only its own (#29).
    """
    stems = [
        _clip_stem(str(r.source_label), r.recording_id, r.start_frame, r.end_frame)
        for r in rows
    ]
    keys = [f"subject{int(r.subject or 0)}" for r in rows]
    return [f"{stem}.mp4" for stem in unique_stems(stems, keys, clean=False)]


def export_behavior_clips(
    experiment_dir: Path | str,
    dest: Path | str,
    *,
    instances: Sequence,
    embed_run_ids: Mapping[str, str],
    renders: Sequence[str] = (clip_render.OVERLAY,),
    crop: bool = False,
    size: int = 640,
    max_per_cluster: int | None = None,
    progress=None,
    preview: ClipQueryPreview | None = None,
) -> list[Path]:
    """Write a clip per instance into ``dest``, plus the ``clips.csv`` behind them.

    Args:
        experiment_dir: the experiment holding the recordings.
        dest: directory to write into; must already exist.
        instances: `BehaviorInstance` rows — already narrowed to whatever scope was asked for.
        embed_run_ids: each recording's embedding run the artifact was built from; it resolves
            that recording's tracking and detection runs, which is what the masks come from.
        renders: any of `clip_render.RENDERS`. Several cost one decode, not several.
        crop: a square cut around the subject rather than the whole frame. Full field of view is
            the default, so the surroundings stay visible.
        size: longest side of the output (the crop's side when ``crop``).
        max_per_cluster: cap the clips per cluster, for a quick look at a large result.
        progress: a ``(stage, fraction, detail)`` callback.
        preview: an open `ClipQueryPreview` to reuse; one is opened if omitted.

    Returns:
        Every path written, `clips.csv` last.
    """
    dest = Path(dest)
    rows = _capped(instances, max_per_cluster)
    if not rows:
        return []
    wanted = clip_render.resolve_renders(renders)
    if preview is None:
        preview = ClipQueryPreview(experiment_dir)

    # Plan every clip first, then render grouped by (recording, part, subject) — one forward
    # decode per part serves every bout it contains, instead of one seek (and its keyframe
    # rewind) per bout. The subject is in the key because `render_part` draws one track.
    by_part: dict[tuple[str, int, int], list[dict]] = {}
    jobs_by_instance: dict[str, dict] = {}
    for row, name in zip(rows, clip_names(rows)):
        subject = int(row.subject or 0)
        out_path = dest / name
        scrub, _t_ns = preview._scrub_ctx(
            row.recording_id, embed_run_ids[row.recording_id]
        )
        planned = clip_render.plan_optional(
            scrub.part_frame_counts,
            row.start_frame,
            row.end_frame,
            (int(row.end_t_ns) - int(row.t_ns)) / 1e9,
            out_path,
        )
        if planned is None:
            continue  # not in the video; the row is still indexed, with no clip
        part_index, job = planned
        by_part.setdefault((row.recording_id, part_index, subject), []).append(job)
        jobs_by_instance[row.instance_id] = job

    done, total = 0, len(jobs_by_instance) or 1

    def _tick(job: dict) -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress("rendering", done / total, Path(job["path"]).name)

    for (recording_id, part_index, subject), jobs in sorted(by_part.items()):
        scrub, _t_ns = preview._scrub_ctx(recording_id, embed_run_ids[recording_id])
        clip_render.render_part(
            scrub._parts[part_index],
            jobs,
            subject,
            size,
            renders=wanted,
            crop=crop,
            progress=_tick,
        )

    written: list[Path] = []
    table: list[list] = []
    for row in rows:
        job = jobs_by_instance.get(row.instance_id)
        names = clip_render.finish(job) if job is not None else {}
        written.extend(dest / name for name in names.values())
        table.append(
            [
                row.instance_id, row.source_label, row.label, row.label_path,
                row.recording_id,
                "" if row.subject is None else row.subject,
                row.start_frame, row.end_frame, row.t_ns, row.end_t_ns,
                round((int(row.end_t_ns) - int(row.t_ns)) / 1e9, 3),
                clip_render.written_fps(job),
            ]  # fmt: skip
            + [names.get(render, "") for render in wanted]
        )
    index = dest / "clips.csv"
    _write_csv(index, CLIP_COLUMNS + [f"video_{r}" for r in wanted], table)
    written.append(index)
    if progress is not None:
        progress("rendering", 1.0, "done")
    return written


def _capped(instances: Sequence, max_per_cluster: int | None) -> list:
    """At most ``max_per_cluster`` per source label, in the order the artifact gives them."""
    if not max_per_cluster or max_per_cluster <= 0:
        return list(instances)
    seen: dict[str, int] = {}
    out = []
    for row in instances:
        key = str(row.source_label)
        if seen.get(key, 0) >= max_per_cluster:
            continue
        seen[key] = seen.get(key, 0) + 1
        out.append(row)
    return out


def _write_csv(path: Path, columns: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
