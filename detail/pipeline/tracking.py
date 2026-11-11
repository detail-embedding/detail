"""Tracking stage runner — wires Trackers to the executor (Sections 6, 13).

Importing this module registers the tracking runner (and the trackers it can
dispatch to). The executor finds it via ``DETAIL_RUNNER_MODULES``.
"""

from pathlib import Path

from detail.configs.tracking import DEFAULT_TRACKER, TrackingConfig
from detail.core.dataclasses import (
    ACTIVE_STATUSES,
    TrackingJob,
    TrackingProgressEvent,
)
from detail.core.registries import TRACKERS
from detail.core.utils import resolve_downstream_every_n, utc_timestamp
from detail.executor.runners import ProcessingModule, register_processing_module
from detail.storage import experiment
from detail.storage import recording as recording_store
from detail.storage import spatial as spatial_store
from detail.trackers import (  # noqa: F401  (registers trackers)
    anchor_tracker as _anchor_tracker,
    iou_single_tracker as _iou_single_tracker,
)


def tracking_run_dir(experiment_root, job: TrackingJob) -> Path:
    """Run directory for a tracking job: ``recordings/<id>/tracking/<track_id>/``."""
    return (
        experiment.recording_path(experiment_root, job.recording_id)
        / "tracking"
        / job.track_id
    )


def _load_arena_polygons(recording_dir: Path, arena_ids: list[str]):
    if not arena_ids:
        return None
    arenas = {a["id"]: a for a in spatial_store.list_arenas(recording_dir)}
    polygons = {}
    for arena_id in arena_ids:
        entry = arenas[arena_id]  # validated upstream in Recording._validate_tracking
        polygons[arena_id] = spatial_store.read_arena_keyframes(
            recording_dir, arena_id, entry["n_vertices"]
        )
    return polygons


def run(job: TrackingJob, run_dir, progress_cb, stop_event) -> None:
    tracker_cls = TRACKERS.get(job.tracker_id)
    if tracker_cls is None:
        raise ValueError(
            f"unknown tracker_id {job.tracker_id!r}; available: {sorted(TRACKERS)}"
        )
    run_dir = Path(run_dir)
    recording_dir = run_dir.parent.parent
    detection_run_dir = run_dir.parent.parent / "detections" / job.detection_run_id
    arena_polygons = _load_arena_polygons(recording_dir, list(job.arena_ids))
    # Always supply offsets: they are needed not only for arena lookup but for
    # correct recording-global frame numbering across parts in the trackers' output.
    # (A tracker can't derive them from detections alone — zero-detection frames
    # have no rows, §9.) Every stage's rows index the source video, so there is one
    # set of offsets whatever stride the run used.
    part_frame_offsets = recording_store.part_frame_offsets(recording_dir)
    tracker_cls().track(
        job,
        detection_run_dir,
        run_dir,
        arena_polygons=arena_polygons,
        part_frame_offsets=part_frame_offsets,
        progress_cb=progress_cb,
        stop_event=stop_event,
    )


@register_processing_module
class TrackingModule(ProcessingModule):
    """Tracking stage (§13) as a processing module (§6.5)."""

    module_id = "tracking"
    config_model = TrackingConfig
    id_prefix = "track"
    runs_key = "tracking_runs"
    upstream = "detection"
    job_class = TrackingJob
    progress_event_class = TrackingProgressEvent
    # frames_total is the detected frames the tracker stepped through
    progress_counts_frames = True
    identity_fields = ("tracker_id", "arena_ids", "hyperparams", "every_n_frames")
    identity_defaults = {"every_n_frames": 1}
    lineage_ref = ("detection", "detection_run_id")

    def recompute_identity_from_job(self, job) -> dict:
        return {"hyperparams": dict(job.hyperparams)}

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        run(job, run_dir, progress_cb, stop_event)

    def run_dir_for(self, experiment_root, job):
        return tracking_run_dir(experiment_root, job)

    def output_path(self, run_id: str) -> str:
        return f"tracking/{run_id}/"

    def validate(self, recording_dir, recording_data: dict, config: dict) -> dict:
        # The model types the shape; hyperparams + arena_ids are enriched below (tracker-owned
        # hyperparam validation, arena normalization), so validate returns the enriched config.
        TrackingConfig.model_validate(config)
        recording_id = Path(recording_dir).name
        detection_run_id = config["detection_run_id"]
        # Resolved here and written back, so the run record names the tracker it actually used
        # (Tenet 9). `model_validate` above fills the model's default, but its instance is
        # discarded and every line below reads the raw dict — so without this a config that
        # omits the key would reach `build_job` as a KeyError rather than as the default.
        tracker_id = config.get("tracker_id") or DEFAULT_TRACKER
        config["tracker_id"] = tracker_id
        arena_ids = self._resolve_arena(config, recording_data, recording_id)
        det = {r["id"]: r for r in recording_data.get("detection_runs", [])}.get(
            detection_run_id
        )
        if det is None:
            raise ValueError(
                f"detection run {detection_run_id!r} not found in recording {recording_id!r}"
            )
        if det.get("status") != "complete":
            raise ValueError(
                f"detection run {detection_run_id!r} is {det.get('status')!r}, "
                "not complete"
            )
        recording_store.raise_on_legacy_downsampling(det)
        config["every_n_frames"] = resolve_downstream_every_n(
            config.get("every_n_frames"), int(det.get("every_n_frames") or 1)
        )
        tracker_cls = TRACKERS.get(tracker_id)
        if tracker_cls is None:
            raise ValueError(
                f"unknown tracker_id {tracker_id!r}; available: {sorted(TRACKERS)}"
            )
        # `or {}`, not `.get(k, {})`: the key is present and null when a caller passes
        # `hyperparams=None`, and a None reaches `set(hyperparams)` as a bare TypeError.
        config["hyperparams"] = tracker_cls().validate_hyperparams(
            config.get("hyperparams") or {}
        )

        for run_entry in recording_data.get("tracking_runs", []):
            if (
                run_entry.get("detection_run_id") == detection_run_id
                and run_entry.get("status") in ACTIVE_STATUSES
            ):
                raise ValueError(
                    f"tracking run {run_entry['id']!r} is already active on detection "
                    f"run {detection_run_id!r}"
                )
        config["arena_ids"] = arena_ids
        return config

    @staticmethod
    def _resolve_arena(
        config: dict, recording_data: dict, recording_id: str
    ) -> list[str]:
        """This recording's own arena id for the selected field of view.

        **A selection is a name; the stored value is an id.** Arena ids are minted per
        recording, so one id means a different polygon on the recording next to it — a
        cohort picked by id tracks different regions and every run looks fine. The name is
        what a person chose and what is the same thing across a rig; the id is what this
        recording happens to call it, and that is what the job, the run record and
        ``arena_dependents`` all need.

        ``arena_ids`` is still honoured when given directly and no name is set: the CLI's
        ``--arena``, saved baselines, and every config already on disk come that way.
        """
        arenas = recording_data.get("arenas", [])
        name = config.get("arena_name")
        if name:
            by_name = {a["name"]: a["id"] for a in arenas}
            if name in by_name:
                return [by_name[name]]
            # A bare id is accepted too, so `--arena arena_001` and every script written
            # before names keep working. It only ever means *this* recording's arena, which
            # is the whole reason a name is the better thing to pass.
            if any(a["id"] == name for a in arenas):
                return [name]
            raise ValueError(
                f"recording {recording_id!r} has no field of view named {name!r}; "
                f"it has: {sorted(by_name) or 'none'}"
            )
        arena_ids = list(config.get("arena_ids") or [])
        if len(arena_ids) > 1:
            raise ValueError(
                f"a tracking run accepts at most one arena_id; got {arena_ids}. "
                "Multi-arena combination semantics are not yet specified (Section 10)."
            )
        known = {a["id"] for a in arenas}
        missing = [a for a in arena_ids if a not in known]
        if missing:
            raise ValueError(
                f"unknown arena_ids {missing}; recording arenas: {sorted(known)}"
            )
        return arena_ids

    def build_job(self, recording_id: str, run_id: str, config: dict) -> TrackingJob:
        return TrackingJob(
            job_id=run_id,
            recording_id=recording_id,
            track_id=run_id,
            detection_run_id=config["detection_run_id"],
            tracker_id=config["tracker_id"],
            hyperparams=config["hyperparams"],
            arena_ids=list(config.get("arena_ids") or []),
            created_at=utc_timestamp(),
            every_n_frames=int(config.get("every_n_frames") or 1),
        )

    def extra_record_fields(self, config: dict) -> dict:
        return {
            "detection_run_id": config["detection_run_id"],
            "tracker_id": config["tracker_id"],
            "arena_ids": list(config.get("arena_ids") or []),
            # The name as well as the resolved id, so the record says which field of view
            # this was rather than only which of *this* recording's arenas it landed on —
            # and so `load_config` can restore the selection, which it could not while the
            # record held a number that means something else on the next recording.
            "arena_name": config.get("arena_name"),
            "hyperparams": dict(config.get("hyperparams") or {}),
            "every_n_frames": int(config.get("every_n_frames") or 1),
        }
