"""Track-merge stage — combine 2+ single-subject tracking runs into one (Section 13).

Two mice told apart at detection as two classes get one clean single-subject track each
(class-filtered tracking). This stage relabels those already-clean tracks into one
multi-subject tracking run — one ``object_id`` per source, at every detection row either
source kept — rather than asking a real multi-object tracker to disambiguate the animals
itself (the identity-fragmentation risk that ruled that approach out). The output is an
ordinary tracking run: same ``assignments/part_NNN.zarr`` / ``tracks.zarr`` / ``track.yaml``
shape every tracker writes, so every existing reader (preview, embedding, location export)
consumes it unchanged.

No resume: a merge reads only already-computed assignment data (no video decode, no model
inference), so it is fast enough that checkpointing would cost more than it saves — the same
call `embedding.py` makes for the same reason (its own docstring: "no resume support ...
partial would be misleading"). A stop request raises rather than writing a partial run.
"""

import time
from pathlib import Path

import numpy as np

from detail.api import run_spec
from detail.configs.track_merge import TrackMergeConfig
from detail.core import atomic
from detail.core.class_names import MERGED_SEPARATOR
from detail.core import yaml_load
from detail.core.dataclasses import TrackMergeJob, TrackingProgressEvent
from detail.core.progress import ThrottledProgress, remaining_seconds
from detail.core.utils import utc_timestamp
from detail.executor.runners import ProcessingModule, register_processing_module
from detail.pipeline import teacher_window
from detail.storage import recording as recording_store
from detail.storage.zarr_io import (
    AssignmentPartWriter,
    DetectionPartReader,
    PartZarrReader,
    TracksSummaryWriter,
    resolve_assignment_rows,
)
from detail.trackers import _common

SCHEMA_VERSION = 1


def track_merge_run_dir(experiment_root, job: TrackMergeJob) -> Path:
    """Run directory for a track-merge job: ``recordings/<id>/tracking/<track_id>/`` —
    the same namespace a tracking run lives in (§13); the merged run *is* one."""
    from detail.storage import experiment

    return (
        experiment.recording_path(experiment_root, job.recording_id)
        / "tracking"
        / job.track_id
    )


def _source_track_yaml(recording_dir: Path, run_id: str) -> dict:
    return yaml_load.safe_load(
        atomic.read_text(recording_dir / "tracking" / run_id / "track.yaml")
    )


def run(job: TrackMergeJob, run_dir, progress_cb, stop_event) -> None:
    run_dir = Path(run_dir)
    recording_dir = run_dir.parent.parent
    detection_run_dir = recording_dir / "detections" / job.detection_run_id
    parts = sorted(detection_run_dir.glob("part_*.zarr"))
    (run_dir / "assignments").mkdir(parents=True, exist_ok=True)
    every_n_frames = max(1, int(job.every_n_frames or 1))

    # Each source's own class_name (for resolve_assignment_rows) and its already-computed
    # single-subject stats (first/last frame, n_gaps, mean_conf) — copied rather than
    # recomputed, so the merge trusts the source tracker's own bookkeeping instead of
    # re-deriving numbers a different formula could get subtly wrong.
    sources = []
    for source_run_id in job.source_run_ids:
        meta = _source_track_yaml(recording_dir, source_run_id)
        tracks = PartZarrReader(
            recording_dir / "tracking" / source_run_id / "tracks.zarr"
        ).read()
        sources.append(
            {
                "run_id": source_run_id,
                "class_name": meta.get("class_name"),
                "assignments_dir": recording_dir
                / "tracking"
                / source_run_id
                / "assignments",
                "first_frame": (
                    int(tracks["first_frame"][0]) if len(tracks["first_frame"]) else 0
                ),
                "last_frame": (
                    int(tracks["last_frame"][0]) if len(tracks["last_frame"]) else 0
                ),
                "n_detections": (
                    int(tracks["n_detections"][0]) if len(tracks["n_detections"]) else 0
                ),
                "n_gaps": int(tracks["n_gaps"][0]) if len(tracks["n_gaps"]) else 0,
                "mean_conf": (
                    float(tracks["mean_conf"][0]) if len(tracks["mean_conf"]) else 0.0
                ),
            }
        )

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
    frames_done = 0
    n_assigned = 0
    n_unassigned = 0

    start = time.monotonic()
    emit = ThrottledProgress(progress_cb)

    def _event(parts_done: int, message: str | None = None) -> TrackingProgressEvent:
        elapsed = time.monotonic() - start
        return TrackingProgressEvent(
            parts_done=parts_done,
            parts_total=len(parts),
            frames_done=frames_done,
            frames_total=frames_total,
            n_active_tracks=len(sources),
            elapsed_s=elapsed,
            eta_s=(
                remaining_seconds(frames_done / frames_total, elapsed)
                if frames_total
                else None
            ),
            message=message,
        )

    emit(_event(0, message="merging tracks"), force=True)

    for part_index, part_path in enumerate(parts):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("track_merge stopped before completing")
        # A reprocessed part must start from a clean store — AssignmentPartWriter opens
        # mode="a" and would otherwise append onto a stale/partial part (no resume: a
        # relaunch always starts fresh, so any leftover part from an earlier attempt must
        # not be appended onto).
        _common.clear_zarr(run_dir / "assignments" / part_path.name)
        det_reader = DetectionPartReader(part_path)
        merged_object_id = np.full(det_reader.n_rows, -1, dtype=np.int32)
        for merge_id, source in enumerate(sources):
            assignment_path = source["assignments_dir"] / part_path.name
            if not assignment_path.exists():
                continue
            assign_reader = PartZarrReader(assignment_path)
            # object_id is parallel to the detection rows the source tracker kept, not the
            # detection reader's own rows (§13's class filter) — recover which raw row each
            # kept assignment row really is, exactly the mechanism that fixed the
            # class-filter preview/embedding alignment bug this session.
            raw_rows = resolve_assignment_rows(
                det_reader, assign_reader, source["class_name"]
            )
            kept = np.asarray(assign_reader.column("object_id")[: len(raw_rows)]) >= 0
            merged_object_id[raw_rows[kept]] = merge_id
        n_assigned += int((merged_object_id >= 0).sum())
        n_unassigned += int((merged_object_id < 0).sum())
        with AssignmentPartWriter(run_dir / "assignments" / part_path.name) as w:
            for object_id in merged_object_id:
                w.add_object_id(int(object_id))
        frame_idx = np.unique(det_reader.column("frame_idx")[: det_reader.n_rows])
        frames_done += int(np.count_nonzero(frame_idx % every_n_frames == 0))
        emit(_event(part_index + 1), force=True)

    _common.clear_zarr(run_dir / "tracks.zarr")
    with TracksSummaryWriter(run_dir / "tracks.zarr") as w:
        for merge_id, source in enumerate(sources):
            w.add_track(
                track_id=merge_id,
                class_name=source["class_name"],
                first_frame=source["first_frame"],
                last_frame=source["last_frame"],
                n_detections=source["n_detections"],
                n_gaps=source["n_gaps"],
                mean_conf=source["mean_conf"],
            )

    _common.write_track_yaml(
        run_dir / "track.yaml",
        {
            "schema_version": SCHEMA_VERSION,
            "id": job.track_id,
            "created_at": job.created_at,
            "detection_run_id": job.detection_run_id,
            "tracker_id": "track_merge",
            "hyperparams": {},
            "arena_ids": [],
            "class_name": None,  # spans every source's class; no single filter applies
            "every_n_frames": every_n_frames,
            "source_run_ids": list(job.source_run_ids),
            # The designed-but-never-populated slot check_track_is_unambiguous's own
            # error message names as the way out (core/utils.py) — each subject's own
            # class name, so a future caller can resolve_track_id("white mouse") instead
            # of a bare, order-dependent integer.
            "subject_labels": {i: s["class_name"] for i, s in enumerate(sources)},
            "status": "complete",
            "n_tracks": len(sources),
            "n_detections_assigned": n_assigned,
            "n_detections_unassigned": n_unassigned,
        },
    )


def _spans(windows: set) -> str:
    return ", ".join(
        sorted(
            "the whole recording" if w is None else teacher_window.window_label(w)
            for w in windows
        )
    )


@register_processing_module
class TrackMergeModule(ProcessingModule):
    """Track-merge stage (§13) as a processing module (§6.5)."""

    module_id = "track_merge"
    config_model = TrackMergeConfig
    id_prefix = (
        "track"  # shares tracking's track_NN namespace, like derive shares emb_NN
    )
    runs_key = "tracking_runs"
    upstream = "tracking"
    job_class = TrackMergeJob
    progress_event_class = (
        None  # reuses "tracking"'s TrackingProgressEvent, already registered
    )
    # frames_total is the detected frames the merge stepped through, as tracking's is
    progress_counts_frames = True
    identity_fields = ("source_run_ids",)
    lineage_ref = ("detection", "detection_run_id")

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        run(job, run_dir, progress_cb, stop_event)

    def run_dir_for(self, experiment_root, job):
        return track_merge_run_dir(experiment_root, job)

    def output_path(self, run_id: str) -> str:
        return f"tracking/{run_id}/"

    def estimate_output_bytes(
        self, recording_dir, recording_data: dict, config: dict
    ) -> int:
        """A merged run is a tracking run (#743): one object id per detection row a source
        kept, so it is estimated as one."""
        from detail.core import disk_space
        from detail.pipeline.tracking import TRACKING_BYTES_PER_FRAME

        return disk_space.frames_to_process(recording_data, None) * (
            TRACKING_BYTES_PER_FRAME
        )

    def validate(self, recording_dir, recording_data: dict, config: dict) -> dict:
        TrackMergeConfig.model_validate(config)
        recording_id = Path(recording_dir).name
        specs = config.get("source_specs") or []

        source_run_ids = []
        for spec in specs:
            run_id = run_spec.resolve_spec_in_recording(
                recording_data, "tracking", spec, recording_dir
            )
            if run_id is None:
                raise ValueError(
                    f"recording {recording_id!r} has no tracking run matching one of "
                    "the selected sources"
                )
            source_run_ids.append(run_id)
        if len(set(source_run_ids)) != len(source_run_ids):
            raise ValueError(
                f"recording {recording_id!r}: the same tracking run was selected more "
                f"than once ({source_run_ids})"
            )

        tracking_by_id = {r["id"]: r for r in recording_data.get("tracking_runs", [])}
        sources = [tracking_by_id[rid] for rid in source_run_ids]
        for run_id, entry in zip(source_run_ids, sources):
            if entry.get("status") != "complete":
                raise ValueError(
                    f"tracking run {run_id!r} is {entry.get('status')!r}, not complete"
                )

        detection_run_ids = {s.get("detection_run_id") for s in sources}
        if len(detection_run_ids) != 1:
            raise ValueError(
                f"recording {recording_id!r}: source tracking runs use different "
                f"detection runs ({sorted(detection_run_ids)}); track_merge needs every "
                "source built from the same detection run"
            )

        class_names = [s.get("class_name") for s in sources]
        if any(c is None for c in class_names):
            raise ValueError(
                f"recording {recording_id!r}: every source tracking run must have its "
                "own class filter (§13) so its rows can never overlap another source's — "
                f"got {class_names}"
            )
        if len(set(class_names)) != len(class_names):
            raise ValueError(
                f"recording {recording_id!r}: two source tracking runs share the same "
                f"class filter ({class_names}); track_merge needs each source to be a "
                "distinct class so rows never collide"
            )

        every_n_values = {int(s.get("every_n_frames") or 1) for s in sources}
        if len(every_n_values) != 1:
            raise ValueError(
                f"recording {recording_id!r}: source tracking runs use different "
                f"every_n_frames strides ({sorted(every_n_values)})"
            )

        # A merge of teacher tracks is a teacher run (§26): the keypoint setup's own chain
        # merges each animal's windowed track. It inherits the window its sources share and
        # records it, so every consumer that needs the whole recording still refuses it.
        windows = {teacher_window.window_of(s) for s in sources}
        if len(windows) != 1:
            raise ValueError(
                f"recording {recording_id!r}: source tracking runs cover different spans "
                f"of the recording ({_spans(windows)}); merged tracks need sources that "
                "cover the same one"
            )

        config["source_run_ids"] = source_run_ids
        config["source_classes"] = class_names
        config["detection_run_id"] = next(iter(detection_run_ids))
        config["every_n_frames"] = next(iter(every_n_values))
        config[teacher_window.WINDOW_KEY] = next(iter(windows))
        return config

    def build_job(self, recording_id: str, run_id: str, config: dict) -> TrackMergeJob:
        return TrackMergeJob(
            job_id=run_id,
            recording_id=recording_id,
            track_id=run_id,
            detection_run_id=config["detection_run_id"],
            source_run_ids=list(config["source_run_ids"]),
            created_at=utc_timestamp(),
            every_n_frames=int(config.get("every_n_frames") or 1),
        )

    def extra_record_fields(self, config: dict) -> dict:
        return {
            "detection_run_id": config["detection_run_id"],
            "source_run_ids": list(config["source_run_ids"]),
            # Satisfies TrackingModule's own identity_fields: run_spec.spec_of resolves
            # *any* tracking_runs record through _processing_module("tracking") regardless
            # of which module actually wrote it (the same reason DeriveModule writes
            # embedding's exact identity fields into its own embedding_runs record).
            "tracker_id": "track_merge",
            "arena_ids": [],
            # The ordered composition, not a bare None: two merges built from the same
            # sources in reversed order (A->0/B->1 vs B->0/A->1) would otherwise share
            # one identity — every field here was a constant, so spec_of/_own_label
            # could never tell them apart, and resolve_spec_in_recording would silently
            # pick whichever merge is newest per recording. A picker showing "track_merge"
            # either way is how a merged run's object_id 0 stopped reliably meaning the
            # same animal across recordings. class_name is already part of tracking's
            # identity_fields and already rendered by _own_label, so this reuses both for
            # free rather than adding a new field.
            "class_name": MERGED_SEPARATOR.join(config["source_classes"]),
            "hyperparams": {},
            "every_n_frames": int(config.get("every_n_frames") or 1),
            teacher_window.WINDOW_KEY: config.get(teacher_window.WINDOW_KEY),
        }
