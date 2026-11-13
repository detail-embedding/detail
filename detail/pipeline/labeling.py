"""Fresh labeling dataset source (§12) — build a YOLO-seg dataset by running a prompt-capable model
(SAM3) on a random sample of frames across the source recordings, with no existing detection
run. The *distill a strong model → a fast YOLO* path: label a small sample, train a YOLO, then run
the cheap YOLO everywhere.

Sampling is proportional to recording length (uniform-random, without replacement within each part).
Sampled frames are read by **seeking** straight to each index, and each is decoded **exactly**
(``_dataset_common.exact_sampled_frames``: ``read_frame_at`` with the part's timestamp map). A
labeling sample's frame index is a label, not a join key, so it may seek — the Tenet 11 exception
for this source — but a frame the decoder cannot land on exactly is redrawn, never labelled under
another frame's index. Frames are labeled in batches via ``predict_batch`` (``batch_size`` 1 for
models that do not batch, e.g. SAM3; a batched forward pass for those that do).

A frame where the teacher found none of the classes is a **background** example (an empty label),
kept up to ``background_share`` of the dataset so the student learns what an empty frame looks
like. A frame where it found something but every mask fell below ``min_contour_area`` is
skipped: an animal may be there, so it is not background. The train/val split is drawn after
labelling over the examples kept (``StagedExamples``), so the realised validation share is the
configured one. Output is the source-agnostic dataset layout shared with the run-export source;
``dataset.yaml`` records ``source.kind = "labeling"`` plus the ``source.model`` used.
"""

import contextlib
import json
import random
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2

from detail.core import atomic
from detail.core.dataclasses import (
    LabelingJob,
    LabelingProgressEvent,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.core.progress import ThrottledProgress, remaining_seconds
from detail.core.protocols import ModelNotReadyError
from detail.core.utils import mask_to_polygon, resolve_device
from detail.executor.runners import register_runner
from detail.models.resolve import resolve_detection_model
from detail.pipeline._dataset_common import StagedExamples, exact_sampled_frames
from detail.storage import datasets as datasets_store
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store

# Batch-capable models run one batched forward pass; SAM3 and
# any model with supports_batching=False degenerate to per-frame (batch_size 1).
_BATCH_SIZE = 4


def run_dir_for(experiment_root, job: LabelingJob) -> Path:
    """``datasets/<dataset_id>/`` — a experiment-level run dir (shared with the run-export source)."""
    return experiment_store.datasets_dir(experiment_root) / job.dataset_id


@register_runner("labeling", run_dir_for=run_dir_for)
def run(job: LabelingJob, run_dir, progress_cb, stop_event) -> None:
    run_dir = Path(run_dir)
    try:
        _label(job, run_dir, progress_cb, stop_event)
    except Exception as exc:
        # The dataset's status is authoritative for every reader, so a crash must not leave it
        # stuck at "running" — that would hide it from delete (gui_design §1.1).
        with contextlib.suppress(FileNotFoundError):
            datasets_store.set_dataset_status(
                run_dir, "failed", error=f"{type(exc).__name__}: {exc}"
            )
        raise


def _label(job: LabelingJob, run_dir: Path, progress_cb, stop_event) -> None:
    experiment_root = run_dir.parent.parent
    # With the frames already extracted there is no video to find, and on the machine the labeling
    # runs on there may be no video root to find it under.
    frames_dir = run_dir / job.source_frames if job.source_frames else None
    parts = _gather_parts(experiment_root, job, resolve_video=frames_dir is None)
    sampled = _sample_frames(parts, job)  # {part_index: sorted [frame_idx]}
    staged = StagedExamples(run_dir)

    start = time.monotonic()
    emit = ThrottledProgress(progress_cb)
    total = sum(len(v) for v in sampled.values())
    n_classes = len(job.classes)
    # Surface the (slow) model load so the bar isn't silent while SAM3 loads.
    emit(
        LabelingProgressEvent(
            0, total, 0, n_classes, time.monotonic() - start, None, "loading model"
        ),
        force=True,
    )
    model = _load_model(experiment_root, job)
    prompt = Prompt(
        targets=[
            PromptTarget(class_name=c.name, cue=TextCue(text=c.prompt))
            for c in job.classes
        ]
    )
    class_id = {c.name: c.class_id for c in job.classes}
    batch_size = _BATCH_SIZE if getattr(model, "supports_batching", False) else 1

    done = 0
    per_class: dict[int, int] = {c.class_id: 0 for c in job.classes}
    stats = {"undecodable": 0, "below_min_area": 0}
    interrupted = False
    batch: list = []  # (part_index, frame_idx, stem, frame)

    def _flush() -> None:
        nonlocal done
        if not batch:
            return
        frames = [item[3] for item in batch]
        results = model.predict_batch(frames, [prompt] * len(frames))
        for (part_index, frame_idx, stem, frame), result in zip(batch, results):
            lines, n_found = _labels_from_result(
                result, class_id, job.min_contour_area, per_class
            )
            if lines or n_found == 0:  # labelled, or background: nothing there at all
                staged.add(
                    parts[part_index]["recording_id"],
                    (part_index, frame_idx),
                    stem,
                    frame,
                    lines,
                )
            else:
                stats["below_min_area"] += 1
            done += 1  # count every processed frame so the bar reaches 100%
            elapsed = time.monotonic() - start
            classes_done = sum(1 for n in per_class.values() if n > 0)
            emit(
                LabelingProgressEvent(
                    done,
                    total,
                    classes_done,
                    n_classes,
                    elapsed,
                    remaining_seconds(done / total, elapsed) if total else None,
                )
            )
        batch.clear()

    stored = _read_manifest(frames_dir) if frames_dir is not None else None
    for part_index, part in enumerate(parts):
        if stop_event is not None and stop_event.is_set():
            interrupted = True
            break
        targets = sampled.get(part_index, [])
        frames = (
            exact_sampled_frames(
                part["video"],
                targets,
                part["timestamps"],
                random.Random(f"{job.dataset_id}:{part_index}:redraw"),
                stats,
            )
            if stored is None
            else _stored_frames(frames_dir, stored, part)
        )
        for frame_idx, frame in frames:
            stem = _example_stem(part["recording_id"], part["stem"], frame_idx)
            batch.append((part_index, frame_idx, stem, frame))
            if len(batch) >= batch_size:
                _flush()
        _flush()  # a part's frames never spill into the next part's batch

    # A clean finish forces 100% even if some sampled indices could not be decoded and ran out
    # of redraws (so `done` < `total`); an interrupted run shows real progress.
    elapsed = time.monotonic() - start
    classes_done = sum(1 for n in per_class.values() if n > 0)
    final_done = done if interrupted else total
    emit(
        LabelingProgressEvent(
            final_done,
            total,
            classes_done,
            n_classes,
            elapsed,
            remaining_seconds(final_done / total, elapsed) if total else None,
        ),
        force=True,
    )
    split = staged.commit(job.val_split, job.dataset_id, job.background_share)
    n_val = sum(1 for which in split.values() if which == "val")
    _finalize(
        run_dir,
        job,
        per_class,
        len(split) - n_val,
        n_val,
        staged.n_background,
        stats,
        interrupted,
    )


def _labels_from_result(result, class_id, min_area, per_class) -> tuple[list[str], int]:
    """YOLO-seg label lines for one frame's ``FrameResult`` (all class prompts), and how many
    detections of the job's classes the teacher returned — zero is what makes a background
    frame, not zero surviving lines."""
    lines: list[str] = []
    n_found = 0
    for det in result.detections():
        cid = class_id.get(det.class_name)
        if cid is None:
            continue
        n_found += 1
        poly = mask_to_polygon(det.mask, min_area)
        if poly is None:
            continue
        coords = " ".join(f"{v:.6f}" for v in poly)
        lines.append(f"{cid} {coords}")
        per_class[cid] += 1
    return lines, n_found


# --------------------------------------------------------------------------- #
# Recordings, sampling, model
# --------------------------------------------------------------------------- #
def _gather_parts(
    experiment_root, job: LabelingJob, *, resolve_video: bool = True
) -> list[dict]:
    """Per-part contexts (original video + frame count) for every source recording.

    ``resolve_video=False`` leaves ``video`` as ``None``: the frame count comes from the
    timestamp map, which lives in the experiment, so sampling needs no video at all.
    """
    parts: list[dict] = []
    for recording_id in job.source_recording_ids:
        recording_dir = experiment_store.recording_path(experiment_root, recording_id)
        data = recording_store.read_recording(recording_dir)
        files = recording_store.video_files(data)
        for part_index, file in enumerate(files):
            video = (
                recording_store.resolve_part_video_path(recording_dir, file)
                if resolve_video
                else None
            )
            timestamps = recording_store.load_active_timestamps(
                recording_dir, data, part_index
            ).values
            parts.append(
                {
                    "recording_id": recording_id,
                    "stem": recording_store.part_stem(file),
                    "video": video,
                    # Authoritative frame count = the PTS-derived timestamp map (Tenet 8), not
                    # cv2's fps-estimated CAP_PROP_FRAME_COUNT: an over-estimate would sample
                    # indices past the true end that never decode, silently shrinking the dataset
                    # below n_frames (CLAUDE.md frame-count rule).
                    "frame_count": len(timestamps),
                    # Handed to read_frame_at so each sampled frame is decoded exactly.
                    "timestamps": timestamps,
                }
            )
    return parts


def _draw_indices(count: int, k: int, strategy: str, rng: random.Random) -> list[int]:
    """``k`` sorted frame indices in ``[0, count)``. ``"stratified"`` draws one uniform frame
    from each of ``k`` equal-width temporal bins (even time coverage, still random within a bin);
    ``"uniform"`` draws ``k`` without replacement over the whole range. Requires ``1 <= k <= count``.
    """
    if strategy == "stratified":
        return [rng.randrange(count * i // k, count * (i + 1) // k) for i in range(k)]
    return sorted(rng.sample(range(count), k))


def _sample_frames(parts: list[dict], job: LabelingJob) -> dict[int, list[int]]:
    """``{part_index: sorted [frame_idx]}`` — ``n_frames`` allotted **proportionally to recording
    length** (then to parts within a recording), then drawn within each part per ``job.sampling``
    (``"stratified"`` = one per equal-width time bin, ``"uniform"`` = random). Deterministic given
    ``dataset_id``. If ``n_frames`` exceeds what's available it is clamped (``_allocate`` caps at
    the total) — the API validates the hard limit up front.
    """
    by_recording: dict[str, list[int]] = {}  # recording_id -> [part_index]
    for part_index, part in enumerate(parts):
        by_recording.setdefault(part["recording_id"], []).append(part_index)

    recording_ids = list(by_recording)
    recording_budgets = _allocate(
        job.n_frames,
        [
            sum(parts[pi]["frame_count"] for pi in by_recording[sid])
            for sid in recording_ids
        ],
    )
    rng = random.Random(job.dataset_id)
    sampled: dict[int, list[int]] = {}
    for recording_id, budget in zip(recording_ids, recording_budgets):
        part_indices = by_recording[recording_id]
        part_budgets = _allocate(
            budget, [parts[pi]["frame_count"] for pi in part_indices]
        )
        for part_index, part_budget in zip(part_indices, part_budgets):
            count = parts[part_index]["frame_count"]
            k = min(part_budget, count)
            if k > 0:
                sampled[part_index] = _draw_indices(count, k, job.sampling, rng)
    return sampled


def _allocate(total: int, weights: list[int]) -> list[int]:
    """Distribute ``total`` across buckets proportional to ``weights`` (largest-remainder), summing
    to exactly ``min(total, sum(weights))``."""
    total = min(total, sum(weights))
    s = sum(weights)
    if s == 0:
        return [0] * len(weights)
    raw = [total * w / s for w in weights]
    floors = [int(r) for r in raw]
    order = sorted(range(len(weights)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[: total - sum(floors)]:
        floors[i] += 1
    return floors


#: Where a remote labeling job's extracted frames live, relative to the dataset's run dir.
SOURCE_FRAMES_DIR = "source_frames"
_MANIFEST = "manifest.json"


def _example_stem(recording_id: str, part_stem: str, frame_idx: int) -> str:
    """The name a sampled frame is saved under: as a dataset example, and in ``source_frames``."""
    return f"{recording_id}_{part_stem}_{frame_idx:07d}"


def extract_source_frames(
    experiment_root,
    job: LabelingJob,
    out_dir,
    progress_cb: Callable[[int, int], None] | None = None,
    stop_event=None,
) -> int:
    """Write every frame this job will sample to ``out_dir`` as a lossless PNG; return the count.

    The sample is a pure function of the recordings' timestamp maps and the dataset id
    (``_sample_frames``), so it can be drawn on the machine that holds the video and only the
    images sent to wherever the labeling runs: ~1,000 frames instead of a whole video library
    (80 recordings was ~116 GB). Each frame is read exactly as a local labeling run reads it
    (``exact_sampled_frames``, redraws included) and saved **losslessly**: labeling re-encodes the frame as a JPEG for
    the dataset, and a second lossy step here would make a remote run's pictures differ from a
    local run's.

    **Idempotent**, so a retry resumes: a frame already on disk is kept. A ``manifest.json``
    naming every frame present is written last and only on a clean finish; the labeling reader
    treats it as the source of truth, so an incomplete transfer fails loudly rather than quietly
    labeling fewer frames than were asked for. An interrupted extraction writes none.

    ``progress_cb(done, total)`` is throttled to the shared live-bar cadence.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = _gather_parts(experiment_root, job)
    sampled = _sample_frames(parts, job)
    total = sum(len(frames) for frames in sampled.values())
    emit = ThrottledProgress(
        None if progress_cb is None else (lambda counts: progress_cb(*counts))
    )
    emit((0, total), force=True)
    present: list[str] = []
    done = 0
    for part_index, part in enumerate(parts):
        wanted = [
            (
                frame_idx,
                f"{_example_stem(part['recording_id'], part['stem'], frame_idx)}.png",
            )
            for frame_idx in sampled.get(part_index, [])
        ]
        missing = [f for f, name in wanted if not (out_dir / name).is_file()]
        present += [name for _f, name in wanted if (out_dir / name).is_file()]
        done += len(wanted) - len(missing)
        frames = exact_sampled_frames(
            part["video"],
            missing,
            part["timestamps"],
            random.Random(f"{job.dataset_id}:{part_index}:redraw"),
        )
        for frame_idx, frame in frames:
            if stop_event is not None and stop_event.is_set():
                return done
            name = f"{_example_stem(part['recording_id'], part['stem'], frame_idx)}.png"
            _write_png(out_dir / name, frame)
            present.append(name)
            done += 1
            emit((done, total))
        if stop_event is not None and stop_event.is_set():
            return done
    emit((done, total), force=True)
    manifest = out_dir / f"{_MANIFEST}.tmp"
    manifest.write_text(
        json.dumps({"schema_version": 1, "frames": sorted(present)}), encoding="utf-8"
    )
    atomic.replace(manifest, out_dir / _MANIFEST)
    return len(present)


def _write_png(path: Path, frame) -> None:
    """Write ``frame`` to ``path`` so no reader ever sees half a PNG: a sibling temp file, then a
    replace. A crash mid-write would otherwise leave a truncated file under its final name, which
    the idempotent skip above would then trust on the retry."""
    temp = path.with_name(f"{path.stem}.tmp.png")
    if not cv2.imwrite(str(temp), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise OSError(
            f"could not write the extracted frame {path.name} to {path.parent}"
        )
    atomic.replace(temp, path)


def _read_manifest(frames_dir: Path) -> set[str]:
    """The frame names a finished extraction recorded.

    Raises:
        FileNotFoundError: no manifest, meaning the extraction never finished or the frames did
            not all arrive.
    """
    path = frames_dir / _MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"no {_MANIFEST} in {frames_dir}: the sampled frames were not fully extracted, or "
            "not fully transferred, so labeling will not guess at which are missing"
        )
    return set(json.loads(atomic.read_text(path))["frames"])


def _stored_frames(
    frames_dir: Path, manifest: set[str], part: dict
) -> Iterator[tuple[int, object]]:
    """``(frame_idx, BGR frame)`` for every frame of ``part`` that extraction produced.

    Read from the manifest rather than ``targets``: extraction decodes exactly and redraws a
    target it cannot land on (``exact_sampled_frames``), so the frames it wrote are the ones a
    local run would have labelled, which need not be ``targets``. One the manifest names but the
    disk does not have, or cannot decode, is a transfer problem and raises.
    """
    prefix = f"{part['recording_id']}_{part['stem']}_"
    found = sorted(
        int(name[len(prefix) : -len(".png")])
        for name in manifest
        if name.startswith(prefix) and name[len(prefix) : -len(".png")].isdigit()
    )
    for frame_idx in found:
        name = f"{_example_stem(part['recording_id'], part['stem'], frame_idx)}.png"
        frame = cv2.imread(str(frames_dir / name))
        if frame is None:
            raise FileNotFoundError(
                f"the extracted frame {name} is missing or unreadable in {frames_dir}; "
                "the transfer from the machine that holds the video may be incomplete"
            )
        yield frame_idx, frame


def _load_model(experiment_root, job: LabelingJob):
    """Resolve, readiness-check, and load the job's model on its compute; set the confidence threshold."""
    model = resolve_detection_model(
        experiment_root, job.model_id, "pretrained", job.model_variant
    )
    status = model.readiness()
    if not status.ready:
        raise ModelNotReadyError(status)
    model.load(resolve_device(job.compute, model.supported_backends))
    model.set_conf_threshold(job.logit_threshold)
    return model


def _finalize(
    run_dir, job, per_class, n_train, n_val, n_background, stats, interrupted
) -> None:
    datasets_store.write_data_yaml(run_dir, {c.class_id: c.name for c in job.classes})
    datasets_store.write_dataset_yaml(
        run_dir,
        {
            "id": job.dataset_id,
            "name": job.name,
            "created_at": job.created_at,
            "source": {
                "kind": "labeling",
                "model": job.model_id,
                "model_variant": job.model_variant,
                "recording_ids": list(job.source_recording_ids),
                "classes": [c.to_dict() for c in job.classes],
                "n_frames": job.n_frames,
                "logit_threshold": job.logit_threshold,
                "min_contour_area": job.min_contour_area,
                "sampling": job.sampling,
            },
            "classes": [{"id": c.class_id, "name": c.name} for c in job.classes],
            "val_split": job.val_split,
            "background_share": job.background_share,
            "n_train": n_train,
            "n_val": n_val,
            # Empty-label frames among n_train + n_val, at most background_share of them.
            "n_background": n_background,
            # Sampled frames that made no example: the decoder could not land on them
            # exactly (each was redrawn), or the teacher found something too small to keep.
            "skipped": dict(stats),
            "per_class": {str(c.class_id): per_class[c.class_id] for c in job.classes},
            "status": "partial" if interrupted else "complete",
        },
    )
