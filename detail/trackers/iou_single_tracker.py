"""IoUSingleTracker — single-subject tracker with IoU temporal continuity + merging
(Section 13).

``anchor_tracker`` enforces one subject but picks the per-frame argmax-confidence
detection with no memory, so a one-frame confidence spike on a distractor steals the
subject. This tracker keeps anchor's single-subject + fragment merge but chooses the
subject by **confidence plus IoU continuity** with its last position: per frame, among
arena/conf-gated candidates, while the subject is locked (seen within ``max_missed``
frames) it picks

    argmax( conf + iou_weight * IoU(candidate, last_subject_bbox) )

else it re-acquires by plain confidence. The IoU bonus is the whole point — a distant
high-confidence distractor earns no bonus and loses to the slightly-lower-confidence
subject sitting where it was last frame; when nothing overlaps, the bonus is ~0 and
selection falls back to confidence on its own. Fragment merging gates on **mask** IoU
(``merge_mask_iou``), which rejects a glass reflection whose bbox overlaps the subject
but whose pixels are a separate region.

Online single pass over the frame span (empty frames still age the subject, as the removed
``iou_tracker`` did); the subject state carries across parts. Resume is at part
granularity via ``part_handoff.json`` (Section 7 / Tenet 12). The shared single-subject
select/attach logic lives in ``trackers/_single_subject``.

Validated defaults are the config behind the earlier prototype's pose / clip-query runs, tuned on
``"rat"``-prompt SAM3 single-subject runs — a good default, not universal.
Known limitation (detector-side, not this tracker): a fully curled / balled-up subject
gets ~no detector confidence, so no candidate clears ``conf_threshold`` and that frame
has no subject.
"""

import time
from pathlib import Path

import numpy as np

from detail.core.dataclasses import TrackingJob, TrackingProgressEvent
from detail.core.progress import ThrottledProgress, remaining_seconds
from detail.core.protocols import Tracker
from detail.core.registries import register_tracker
from detail.executor import rundir
from detail.storage.zarr_io import (
    AssignmentPartWriter,
    DetectionPartReader,
    TracksSummaryWriter,
)
from detail.trackers import _common
from detail.trackers import _single_subject as ss

SCHEMA_VERSION = 1
SUBJECT_ID = _common.SUBJECT_ID


def _serialize_state(
    *,
    subject_bbox,
    missed,
    first_frame,
    last_frame,
    n_present_frames,
    conf_sum,
    class_name,
    n_assigned,
    n_unassigned,
    frames_done,
    global_offset,
) -> dict:
    """The cross-part snapshot committed at a part boundary (Section 7 resume).
    Plain JSON — the subject bbox becomes a list, no paths (Tenets 6/7)."""
    return {
        "subject_bbox": list(subject_bbox) if subject_bbox is not None else None,
        "missed": missed,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "n_present_frames": n_present_frames,
        "conf_sum": conf_sum,
        "class_name": class_name,
        "n_assigned": n_assigned,
        "n_unassigned": n_unassigned,
        "frames_done": frames_done,
        "global_offset": global_offset,
    }


@register_tracker("iou_single_tracker")
class IoUSingleTracker(Tracker):
    tracker_id = "iou_single_tracker"
    supported_backends = ["cpu"]

    HYPERPARAMS_SCHEMA = {
        "conf_threshold": (float, 0.05),  # min conf to be a subject candidate
        "secondary_conf": (float, 0.05),  # min conf to be an attachable fragment
        "merge_iou": (float, 0.1),  # bbox-overlap pre-filter for fragment merging
        "merge_mask_iou": (float, 0.5),  # mask-overlap floor: rejects glass reflections
        "max_merge": (int, 0),  # fragments merged per frame (0 = cleanest silhouette)
        "iou_weight": (float, 1.0),  # weight of the continuity bonus in the score
        "max_missed": (int, 30),  # frames the subject may be absent before re-acquire
        # Off by default: IoU continuity already identifies the subject, so the
        # contiguity gate only hurts — on busy detect runs it rejects the subject's
        # *split* mask and lets selection fall through to a contiguous distractor
        # (subject teleports). Default False also skips the per-frame mask decode.
        "require_contiguous": (bool, False),
    }

    def validate_hyperparams(self, hyperparams: dict) -> dict:
        return _common.validate_hyperparams(self.HYPERPARAMS_SCHEMA, hyperparams)

    def track(
        self,
        job: TrackingJob,
        detection_run_dir,
        output_dir,
        arena_polygons=None,
        part_frame_offsets=None,
        progress_cb=None,
        stop_event=None,
    ) -> None:
        p = self.validate_hyperparams(job.hyperparams)
        detection_run_dir = Path(detection_run_dir)
        output_dir = Path(output_dir)
        parts = sorted(detection_run_dir.glob("part_*.zarr"))
        (output_dir / "assignments").mkdir(parents=True, exist_ok=True)
        # ≤1 arena per run is enforced upstream (Recording._validate_tracking).
        arena_polygon = next(iter(arena_polygons.values())) if arena_polygons else None
        mask_h, mask_w = rundir.read_envelope(detection_run_dir)["mask_resolution"]
        mask_size = (int(mask_h), int(mask_w))
        every_n_frames = max(1, int(getattr(job, "every_n_frames", 1) or 1))

        frames_total = sum(
            int(
                np.count_nonzero(
                    np.unique(DetectionPartReader(p).column("frame_idx")[:])
                    % every_n_frames
                    == 0
                )
            )
            for p in parts
        )

        # Resume (Section 7): restore the subject state + accumulators from the last
        # committed part boundary; a fresh run starts with no subject at part 0.
        parts_completed, state = _common.read_handoff(output_dir)
        if state is not None:
            subject_bbox = (
                tuple(state["subject_bbox"])
                if state["subject_bbox"] is not None
                else None
            )
            missed = state["missed"]
            first_frame = state["first_frame"]
            last_frame = state["last_frame"]
            n_present_frames = state["n_present_frames"]
            conf_sum = state["conf_sum"]
            class_name = state["class_name"]
            n_assigned = state["n_assigned"]
            n_unassigned = state["n_unassigned"]
            frames_done = state["frames_done"]
            global_offset = state["global_offset"]
        else:
            subject_bbox = None
            missed = 0
            first_frame = None
            last_frame = None
            n_present_frames = 0
            conf_sum = 0.0
            class_name = ""
            n_assigned = 0
            n_unassigned = 0
            frames_done = 0
            # Fallback for direct callers without part_frame_offsets (tracking.run
            # always supplies them); it can't see trailing zero-detection frames (§9).
            global_offset = 0

        start = time.monotonic()
        interrupted = False
        emit = ThrottledProgress(progress_cb)

        def _event(
            parts_done: int, message: str | None = None
        ) -> TrackingProgressEvent:
            elapsed = time.monotonic() - start
            return TrackingProgressEvent(
                parts_done=parts_done,
                parts_total=len(parts),
                frames_done=frames_done,
                frames_total=frames_total,
                n_active_tracks=1 if first_frame is not None else 0,
                elapsed_s=elapsed,
                eta_s=(
                    remaining_seconds(frames_done / frames_total, elapsed)
                    if frames_total
                    else None
                ),
                message=message,
            )

        emit(_event(parts_completed, message="loading detections"), force=True)

        for part_index in range(parts_completed, len(parts)):
            if stop_event is not None and stop_event.is_set():
                interrupted = True
                break
            part_path = parts[part_index]
            _common.clear_zarr(output_dir / "assignments" / part_path.name)
            data = DetectionPartReader(part_path).read()
            conf = data["confidence"]
            bbox = data["bbox"]
            mask_rle = data["mask_rle"]
            n_rows = len(data["frame_idx"])
            object_ids = np.full(n_rows, -1, dtype=np.int32)
            groups = _common.group_by_frame(data["frame_idx"], every_n_frames)
            frame_offset = (
                part_frame_offsets[part_index]
                if part_frame_offsets is not None
                else global_offset
            )
            max_local = -1
            if groups:
                # Walk every processed frame so a zero-detection frame still ages the
                # subject (max_missed counts real elapsed processed frames).
                for frame in _common.frame_walk(groups, every_n_frames):
                    rows = groups.get(frame, [])
                    if arena_polygon is not None and rows:
                        rows = _common.filter_by_arena(
                            rows, bbox, arena_polygon, frame + frame_offset
                        )
                    subject = None
                    if rows:
                        if subject_bbox is not None and missed <= p["max_missed"]:
                            locked, weight = subject_bbox, p["iou_weight"]

                            def score(i, locked=locked, weight=weight):
                                cand = tuple(float(x) for x in bbox[i])
                                return float(conf[i]) + weight * _common.iou(
                                    cand, locked
                                )

                        else:
                            # cold / lost: re-acquire by confidence alone
                            def score(i):
                                return float(conf[i])

                        subject = ss.select_subject(
                            rows,
                            conf,
                            mask_rle,
                            floor=p["conf_threshold"],
                            require_contiguous=p["require_contiguous"],
                            resolution=mask_size,
                            score=score,
                        )
                    if subject is None:
                        missed += 1
                        if missed > p["max_missed"]:
                            subject_bbox = None  # lost — next sighting re-acquires
                    else:
                        object_ids[subject] = SUBJECT_ID
                        ss.attach_secondaries(
                            subject,
                            rows,
                            bbox,
                            conf,
                            mask_rle,
                            secondary_conf=p["secondary_conf"],
                            merge_iou=p["merge_iou"],
                            merge_mask_iou=p["merge_mask_iou"],
                            max_merge=p["max_merge"],
                            require_contiguous=p["require_contiguous"],
                            resolution=mask_size,
                            object_ids=object_ids,
                        )
                        subject_bbox = tuple(float(x) for x in bbox[subject])
                        missed = 0
                        global_frame = frame + frame_offset
                        if first_frame is None:
                            first_frame = global_frame
                            class_name = str(data["class_name"][subject])
                        last_frame = global_frame
                        n_present_frames += 1
                        conf_sum += float(conf[subject])
                        frames_done += 1
                        max_local = max(max_local, frame)
                    emit(_event(part_index))
            n_assigned += int((object_ids >= 0).sum())
            n_unassigned += int((object_ids < 0).sum())
            with AssignmentPartWriter(output_dir / "assignments" / part_path.name) as w:
                for object_id in object_ids:
                    w.add_object_id(int(object_id))
            if part_frame_offsets is None and max_local >= 0:
                global_offset += max_local + 1
            _common.write_handoff(
                output_dir,
                part_index + 1,
                _serialize_state(
                    subject_bbox=subject_bbox,
                    missed=missed,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    n_present_frames=n_present_frames,
                    conf_sum=conf_sum,
                    class_name=class_name,
                    n_assigned=n_assigned,
                    n_unassigned=n_unassigned,
                    frames_done=frames_done,
                    global_offset=global_offset,
                ),
            )
            emit(_event(part_index + 1), force=True)

        if first_frame is None and not interrupted:
            raise _common.zero_tracks_error(n_assigned + n_unassigned, p)

        _common.clear_zarr(output_dir / "tracks.zarr")
        self._write_tracks_zarr(
            output_dir / "tracks.zarr",
            first_frame=first_frame,
            last_frame=last_frame,
            n_present_frames=n_present_frames,
            conf_sum=conf_sum,
            class_name=class_name,
            every_n_frames=every_n_frames,
            part_frame_offsets=part_frame_offsets,
        )
        self._write_track_yaml(
            output_dir / "track.yaml",
            job,
            p,
            n_tracks=1 if first_frame is not None else 0,
            n_assigned=n_assigned,
            n_unassigned=n_unassigned,
            status="partial" if interrupted else "complete",
        )

    def _write_tracks_zarr(
        self,
        path,
        *,
        first_frame,
        last_frame,
        n_present_frames,
        conf_sum,
        class_name,
        every_n_frames,
        part_frame_offsets,
    ) -> None:
        with TracksSummaryWriter(path) as w:
            if first_frame is None:
                return
            # Frames the run looked at and missed, not source frames: at a stride, a frame
            # never processed is not a gap (§4, §13; #32).
            n_gaps = (
                _common.processed_frames_between(
                    first_frame, last_frame, every_n_frames, part_frame_offsets
                )
                - n_present_frames
            )
            mean_conf = conf_sum / n_present_frames if n_present_frames else 0.0
            w.add_track(
                track_id=SUBJECT_ID,
                class_name=class_name,
                first_frame=int(first_frame),
                last_frame=int(last_frame),
                n_detections=int(n_present_frames),
                n_gaps=int(max(0, n_gaps)),
                mean_conf=float(mean_conf),
            )

    def _write_track_yaml(
        self, path, job, params, *, n_tracks, n_assigned, n_unassigned, status
    ) -> None:
        data = {
            "schema_version": SCHEMA_VERSION,
            "id": job.track_id,
            "created_at": job.created_at,
            "detection_run_id": job.detection_run_id,
            "tracker_id": self.tracker_id,
            "hyperparams": params,
            "arena_ids": list(job.arena_ids),
            "subject_labels": None,
            "status": status,
            "n_tracks": n_tracks,
            "n_detections_assigned": n_assigned,
            "n_detections_unassigned": n_unassigned,
        }
        _common.write_track_yaml(path, data)
