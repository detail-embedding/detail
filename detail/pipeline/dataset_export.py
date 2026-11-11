"""Dataset export runner — build a YOLO training dataset from an existing
detection/tracking run (Section 12, primary source).

No new inference: it reuses the source run's stored masks. Frame reading is
**forward-sequential** (Tenet 11) — we read each part's source video start to
finish and pick off the selected frames as we pass them, so each extracted image
matches the mask used as its label (seeking H.264 can land on the wrong frame).
Output is the source-agnostic dataset layout; ``dataset.yaml`` records its source.
"""

import contextlib
import time
from pathlib import Path

import numpy as np
import yaml

from detail.core import atomic
from detail.core.dataclasses import DatasetExportJob, LabelingProgressEvent
from detail.core.progress import ThrottledProgress, remaining_seconds
from detail.core.utils import decode_mask, mask_to_polygon
from detail.executor import rundir
from detail.executor.runners import register_runner
from detail.pipeline._dataset_common import StagedExamples, sequential_frames
from detail.storage import datasets as datasets_store
from detail.storage import experiment as experiment_store
from detail.storage import runs as runs_store
from detail.storage import recording as recording_store
from detail.storage.zarr_io import DetectionPartReader, PartZarrReader

_SUPPORTED_SOURCES = ("detection", "tracking")


def run_dir_for(experiment_root, job: DatasetExportJob) -> Path:
    """``datasets/<dataset_id>/`` — a experiment-level run dir (not per-recording)."""
    return experiment_store.datasets_dir(experiment_root) / job.dataset_id


@register_runner("dataset_export", run_dir_for=run_dir_for)
def run(job: DatasetExportJob, run_dir, progress_cb, stop_event) -> None:
    run_dir = Path(run_dir)
    try:
        _export(job, run_dir, progress_cb, stop_event)
    except Exception as exc:
        # The dataset's status is authoritative for every reader (not only callers
        # that reconcile via DatasetRun.wait), so a crashed export must not be left
        # stuck at "running" — that would hide it from delete (gui_design §1.1).
        # The message is recorded too so the GUI can show why it failed.
        with contextlib.suppress(FileNotFoundError):
            datasets_store.set_dataset_status(
                run_dir, "failed", error=f"{type(exc).__name__}: {exc}"
            )
        raise


def _export(job: DatasetExportJob, run_dir: Path, progress_cb, stop_event) -> None:
    experiment_root = run_dir.parent.parent
    record = runs_store.read_run(experiment_root, job.source_run_id)
    module_id = record.get("module_id")
    if module_id not in _SUPPORTED_SOURCES:
        raise ValueError(
            f"dataset export source must be a detection or tracking run, "
            f"got module {module_id!r}"
        )

    parts = _gather_parts(experiment_root, record, module_id)
    class_map = _build_class_map(parts)  # {name: id}, stable (sorted)
    selected = _selected_frames(parts, job)  # [(part_index, frame_idx)]
    staged = StagedExamples(run_dir)

    total = len(selected)
    done = 0
    interrupted = False
    start = time.monotonic()
    emit = ThrottledProgress(progress_cb)
    for part_index, part in enumerate(parts):
        if stop_event is not None and stop_event.is_set():
            interrupted = True
            break
        rows_by_frame = _rows_by_frame(part)
        targets = sorted(fi for (pi, fi) in selected if pi == part_index)
        for frame_idx, frame in sequential_frames(part["video"], targets):
            lines = _label_lines(
                rows_by_frame.get(frame_idx, []),
                class_map,
                (part["frame_h"], part["frame_w"]),
                job.min_contour_area,
            )
            if lines:  # a contour survived → save the example (else skip a blank)
                stem = f"{part['recording_id']}_{part['stem']}_{frame_idx:07d}"
                staged.add(
                    part["recording_id"], (part_index, frame_idx), stem, frame, lines
                )
            done += 1  # count every processed frame so the bar reaches 100%
            elapsed = time.monotonic() - start
            emit(
                LabelingProgressEvent(
                    done,
                    total,
                    len(class_map),
                    len(class_map),
                    elapsed,
                    remaining_seconds(done / total, elapsed) if total else None,
                )
            )

    # A clean finish forces 100% even if a sampled frame couldn't be decoded (so
    # `done` < `total`); an interrupted run keeps its real progress.
    elapsed = time.monotonic() - start
    final_done = done if interrupted else total
    emit(
        LabelingProgressEvent(
            final_done,
            total,
            len(class_map),
            len(class_map),
            elapsed,
            remaining_seconds(final_done / total, elapsed) if total else None,
        ),
        force=True,
    )
    split = staged.commit(job.val_split, job.dataset_id)
    n_val = sum(1 for which in split.values() if which == "val")
    n_train = len(split) - n_val
    names = {cid: name for name, cid in class_map.items()}
    datasets_store.write_data_yaml(run_dir, names)
    datasets_store.write_dataset_yaml(
        run_dir,
        {
            "id": job.dataset_id,
            "name": job.name,
            "created_at": job.created_at,
            "source": {
                "kind": "run_export",
                "run_id": job.source_run_id,
                "module": module_id,
                "frame_selection": dict(job.frame_selection),
            },
            "classes": [{"id": cid, "name": name} for name, cid in class_map.items()],
            "val_split": job.val_split,
            "n_train": n_train,
            "n_val": n_val,
            "status": "partial" if interrupted else "complete",
        },
    )


# --------------------------------------------------------------------------- #
# Source resolution + selection
# --------------------------------------------------------------------------- #
def _gather_parts(experiment_root, record: dict, module_id: str) -> list[dict]:
    """Flatten the source run's complete per-recording runs into per-part contexts
    (the source video + detection part zarr + optional tracking assignments)."""
    parts: list[dict] = []
    for recording_id, info in record.get("per_recording", {}).items():
        if info.get("status") != "complete":
            continue
        run_id = info["run_id"]
        recording_dir = experiment_store.recording_path(experiment_root, recording_id)
        data = recording_store.read_recording(recording_dir)
        if module_id == "tracking":
            tracking_dir = recording_dir / "tracking" / run_id
            track_meta = yaml.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
            detection_run = track_meta["detection_run_id"]
            assignment_dir = tracking_dir / "assignments"
        else:
            detection_run, assignment_dir = run_id, None
        detection_dir = recording_dir / "detections" / detection_run
        frame_h, frame_w = rundir.read_envelope(detection_dir)["mask_resolution"]
        # Export reads the same frames detection saw. Its rows index the source video
        # whatever stride it ran at, so the candidate frames are simply the rows present.
        det_entry = {r["id"]: r for r in data.get("detection_runs", [])}.get(
            detection_run, {}
        )
        recording_store.raise_on_legacy_downsampling(det_entry)
        files = recording_store.video_files(data)
        for part_index, part_path in enumerate(
            sorted(detection_dir.glob("part_*.zarr"))
        ):
            parts.append(
                {
                    "recording_id": recording_id,
                    "stem": part_path.stem,
                    "video": recording_store.resolve_part_video_path(
                        recording_dir, files[part_index]
                    ),
                    "frame_h": int(frame_h),
                    "frame_w": int(frame_w),
                    "det_part": part_path,
                    "assignment": (
                        assignment_dir / part_path.name
                        if assignment_dir is not None
                        else None
                    ),
                }
            )
    return parts


def _kept_mask(part: dict) -> np.ndarray | None:
    """Boolean keep-mask over a part's detection rows: drop tracker-unassigned
    (``object_id < 0``) rows for a tracking source; keep all for detection."""
    if part["assignment"] is None:
        return None
    return PartZarrReader(part["assignment"]).column("object_id")[:] >= 0


def _candidate_frames(part: dict, job: DatasetExportJob) -> list[int]:
    """Frame indices with ≥1 kept detection that also pass the frame selection."""
    frame_idx = DetectionPartReader(part["det_part"]).column("frame_idx")[:]
    keep = _kept_mask(part)
    if keep is not None:
        frame_idx = frame_idx[keep]
    frames = np.unique(frame_idx)
    if job.frame_selection.get("mode") == "every_n":
        n = int(job.frame_selection.get("n", 1))
        frames = frames[frames % max(n, 1) == 0]
    return frames.tolist()


def _build_class_map(parts: list[dict]) -> dict[str, int]:
    names: set[str] = set()
    for part in parts:
        names.update(
            str(c)
            for c in DetectionPartReader(part["det_part"]).column("class_name")[:]
        )
    return {name: i for i, name in enumerate(sorted(names))}


def _selected_frames(parts: list[dict], job: DatasetExportJob) -> list[tuple[int, int]]:
    """``[(part_index, frame_idx)]`` the frame selection asks for. Their split is drawn after
    export over the frames that produced an example (``StagedExamples``)."""
    return [
        (part_index, frame_idx)
        for part_index, part in enumerate(parts)
        for frame_idx in _candidate_frames(part, job)
    ]


def _rows_by_frame(part: dict) -> dict[int, list[tuple[str, str]]]:
    reader = DetectionPartReader(part["det_part"]).read()
    keep = _kept_mask(part)
    out: dict[int, list[tuple[str, str]]] = {}
    for i in range(len(reader["frame_idx"])):
        if keep is not None and not keep[i]:
            continue
        out.setdefault(int(reader["frame_idx"][i]), []).append(
            (str(reader["class_name"][i]), str(reader["mask_rle"][i]))
        )
    return out


def _label_lines(rows, class_map, mask_size, min_area) -> list[str]:
    lines = []
    for class_name, mask_rle in rows:
        poly = mask_to_polygon(decode_mask(mask_rle, mask_size), min_area)
        if poly is None:
            continue
        coords = " ".join(f"{v:.6f}" for v in poly)
        lines.append(f"{class_map[class_name]} {coords}")
    return lines
