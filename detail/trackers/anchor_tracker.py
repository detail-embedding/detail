"""AnchorTracker — confidence-anchored, contiguity-gated single-subject tracker.

For the known-subject-count case (one animal per recording), linking by frame-to-
frame IoU alone spawns a track per spurious detection when the
detector runs at a low confidence threshold. This tracker instead, per frame:

1. keeps only the single best (top-confidence) detection as the subject anchor,
   gated by a connected-component (contiguity) check and the arena,
2. attaches nearby overlapping lower-confidence detections (secondaries) with the
   **same** ``object_id`` — recovering fragments when the detector splits the
   animal across detections,
3. leaves everything else unassigned (``object_id = -1``).

Selection is stateless per frame (single subject, id always 0), so — unlike
``iou_single_tracker`` — a distant one-frame confidence spike can steal the subject;
``anchor`` is the memoryless baseline. The single-subject select/attach logic is
shared with ``iou_single_tracker`` via ``trackers/_single_subject`` (here the score
is plain confidence and the mask-IoU merge gate is off). Resume is at part
granularity via ``part_handoff.json`` (Section 7 / Tenet 12).

Single-subject only for now: ``num_subjects`` must be 1. Multi-subject needs a
cross-frame association layer (see DESIGN_PLAN §13).
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
    """The single-subject cross-part snapshot committed at a part boundary
    (Section 7 resume). Plain JSON, no paths (Tenets 6/7)."""
    return {
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


@register_tracker("anchor_tracker")
class AnchorTracker(Tracker):
    tracker_id = "anchor_tracker"
    supported_backends = ["cpu"]

    HYPERPARAMS_SCHEMA = {
        "num_subjects": (int, 1),
        "conf_threshold": (float, 0.1),
        "secondary_conf": (float, 0.05),
        "iou_threshold": (float, 0.1),
        "max_secondary": (int, 2),
        # Off by default. On, this is a **veto**: `select_subject` walks candidates in
        # score order and skips every one whose mask is more than one connected region,
        # so a frame where the detector found the animal clearly can end up with no
        # subject at all (`track`: `if anchor is None: continue`).
        # On real recordings of still animals it discarded confident, mid-frame
        # detections: single frames where a mask splits as the animal moves, and long
        # runs of a motionless animal holding a pose whose mask was stably split.
        #
        # `iou_single_tracker` already defaults this off and says why: the gate rejects the
        # subject's own split mask and lets selection fall through to whatever else is
        # contiguous. Anchor has no continuity to fall through to, so it simply loses the
        # frame. Keeping it as a *preference* — take the best candidate when none is
        # contiguous — is the better fix and is not this one; this restores the frames.
        "require_contiguous": (bool, False),
    }

    def validate_hyperparams(self, hyperparams: dict) -> dict:
        result = _common.validate_hyperparams(self.HYPERPARAMS_SCHEMA, hyperparams)
        if result["num_subjects"] != 1:
            raise ValueError(
                "anchor_tracker currently supports num_subjects=1 only; "
                "multi-subject association is not yet implemented (DESIGN_PLAN §13)"
            )
        return result

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
        params = self.validate_hyperparams(job.hyperparams)
        detection_run_dir = Path(detection_run_dir)
        output_dir = Path(output_dir)
        parts = _common.parts_in_play_order(detection_run_dir)
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
            for _play_index, p in parts
        )

        # Resume (Section 7): restore the single-subject accumulators from the last
        # committed part boundary; a fresh run starts empty at part 0.
        parts_completed, state = _common.read_handoff(output_dir)
        if state is not None:
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
            first_frame = None
            last_frame = None
            n_present_frames = 0
            conf_sum = 0.0
            class_name = ""
            n_assigned = 0
            n_unassigned = 0
            frames_done = 0
            # The pipeline (tracking.run) always passes part_frame_offsets; this
            # detection-derived accumulation is an approximate fallback for direct
            # callers only — it can't see trailing zero-detection frames (§9).
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

        # Initial event so a single-part run shows a bar immediately; the in-loop
        # emit below keeps it live (a per-part-only emit goes silent for the whole
        # part — the §6 live-bar contract).
        emit(_event(parts_completed, message="loading detections"), force=True)

        for part_index in range(parts_completed, len(parts)):
            if stop_event is not None and stop_event.is_set():
                interrupted = True
                break
            play_index, part_path = parts[part_index]
            _common.clear_zarr(output_dir / "assignments" / part_path.name)
            data = DetectionPartReader(part_path).read()
            n_rows = len(data["frame_idx"])
            object_ids = np.full(n_rows, -1, dtype=np.int32)
            groups = _common.group_by_frame(data["frame_idx"], every_n_frames)
            frame_offset = (
                part_frame_offsets[play_index]
                if part_frame_offsets is not None
                else global_offset
            )
            max_local = -1
            for frame, rows in groups.items():
                if arena_polygon is not None:
                    rows = _common.filter_by_arena(
                        rows, data["bbox"], arena_polygon, frame + frame_offset
                    )
                anchor = ss.select_subject(
                    rows,
                    data["confidence"],
                    data["mask_rle"],
                    floor=params["conf_threshold"],
                    require_contiguous=params["require_contiguous"],
                    resolution=mask_size,
                    score=lambda i: float(data["confidence"][i]),
                )
                if anchor is None:
                    continue
                object_ids[anchor] = SUBJECT_ID
                global_frame = frame + frame_offset
                if first_frame is None:
                    first_frame = global_frame
                    class_name = str(data["class_name"][anchor])
                last_frame = global_frame
                n_present_frames += 1
                conf_sum += float(data["confidence"][anchor])
                ss.attach_secondaries(
                    anchor,
                    rows,
                    data["bbox"],
                    data["confidence"],
                    data["mask_rle"],
                    secondary_conf=params["secondary_conf"],
                    merge_iou=params["iou_threshold"],
                    merge_mask_iou=0.0,  # anchor gates fragments on bbox IoU only
                    max_merge=params["max_secondary"],
                    require_contiguous=params["require_contiguous"],
                    resolution=mask_size,
                    object_ids=object_ids,
                )
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
            raise ValueError(
                "zero tracks produced; hyperparams may be too strict "
                f"(conf_threshold={params['conf_threshold']}, "
                f"require_contiguous={params['require_contiguous']})"
            )

        _common.clear_zarr(output_dir / "tracks.zarr")
        self._write_tracks_zarr(
            output_dir / "tracks.zarr",
            first_frame=first_frame,
            last_frame=last_frame,
            n_present_frames=n_present_frames,
            conf_sum=conf_sum,
            class_name=class_name,
        )
        self._write_track_yaml(
            output_dir / "track.yaml",
            job,
            params,
            n_tracks=1 if first_frame is not None else 0,
            n_assigned=n_assigned,
            n_unassigned=n_unassigned,
            status="partial" if interrupted else "complete",
        )

    def _write_tracks_zarr(
        self, path, *, first_frame, last_frame, n_present_frames, conf_sum, class_name
    ) -> None:
        with TracksSummaryWriter(path) as w:
            if first_frame is None:
                return
            n_gaps = (last_frame - first_frame + 1) - n_present_frames
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
