"""Location-export analysis module (§16) — the first AnalysisModule.

Wraps the pure per-recording compute (``analysis/locations.py``) as a experiment-scoped,
executor-backed analysis run: one run over a *selection* of recordings, writing each
recording's CSVs into ``analysis/<run_id>/<recording_id>/``. It is an *iterating*
analysis (per-recording, no cross-recording pooling), so it needs no grouping variables
or streams. The synchronous ``Recording.export_locations`` path remains the
lightweight scripting substrate; this is the recorded, batch, progress-bearing run.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from detail.configs.location_export import LocationExportConfig
from detail.core.progress import ThrottledProgress, remaining_seconds
from detail.core.utils import utc_timestamp
from detail.executor.runners import (
    AnalysisModule,
    job_params,
    register_analysis_module,
)
from detail.pipeline import teacher_window
from detail.storage import analysis_runs as analysis_store
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage import spatial as spatial_store

_CSV_KINDS = ("tracks", "roi_relations", "point_relations", "pair_relations")


@dataclass
class LocationAnalysisJob:
    """One location-export run over a recording selection. ``tracking_run_ids`` maps each
    recording to *its own* tracking run (§6.5 "Upstream run specs") — resolved from the
    selected spec at launch, so the recordings need not share a ``track_NN`` ordinal."""

    job_id: str
    source_recording_ids: list[str]
    tracking_run_ids: dict[str, str]
    created_at: str
    #: Each recording's own keypoint track run (§16 *Keypoints and pairs in the export*);
    #: empty when the export reads no keypoints.
    keypoint_run_ids: dict[str, str] = field(default_factory=dict)
    pair_edge_distance: bool = False

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "source_recording_ids": list(self.source_recording_ids),
            "tracking_run_ids": dict(self.tracking_run_ids),
            "created_at": self.created_at,
            "keypoint_run_ids": dict(self.keypoint_run_ids),
            "pair_edge_distance": bool(self.pair_edge_distance),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LocationAnalysisJob":
        recordings = list(d["source_recording_ids"])
        # Compat: a pre-spec job carried one shared ``tracking_run_id`` — read it as a
        # uniform per-recording map.
        run_ids = d.get("tracking_run_ids")
        if run_ids is None:
            run_ids = {sid: d["tracking_run_id"] for sid in recordings}
        return cls(
            job_id=d["job_id"],
            source_recording_ids=recordings,
            tracking_run_ids=dict(run_ids),
            created_at=d["created_at"],
            # A job written before keypoints and pairs read as "none" / "off".
            keypoint_run_ids=dict(d.get("keypoint_run_ids") or {}),
            pair_edge_distance=bool(d.get("pair_edge_distance", False)),
        )


@dataclass
class LocationProgressEvent:
    """Progress for an analysis run. ``recordings_done`` counts *completed* recordings;
    ``frames_done`` / ``frames_total`` are the detection-row progress *within the
    current recording*, so the bar advances during a single long recording (overall
    fraction = ``(recordings_done + frames_done/frames_total) / recordings_total``). The
    CLI bar reads ``frames_*`` per recording; ``message`` names the recording."""

    recordings_done: int
    recordings_total: int
    frames_done: int
    frames_total: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None


@register_analysis_module
class LocationExportModule(AnalysisModule):
    module_id = "location_export"
    config_model = LocationExportConfig
    id_prefix = "an"
    requires = ["tracking"]
    selects = ("tracking", "tracking_run_id")
    job_class = LocationAnalysisJob
    # One phase: this module has no distinct steps, only the same computation once per
    # recording — unlike a multi-phase module, "the plan" here is just the recording it names in
    # its own message.
    phases = ("computing",)
    progress_event_class = LocationProgressEvent
    # frames_total is rows of the location table, not video frames
    progress_counts_frames = False

    def validate(self, experiment_dir, config: dict) -> dict:
        LocationExportConfig.model_validate(config)  # bounds/shape only
        recording_ids = list(config.get("source_recording_ids") or [])
        if not recording_ids:
            raise ValueError("location export needs at least one source recording")
        # Resolve the selection to each recording's own complete tracking run (§6.5). The
        # model covers a subset, so return the *enriched* config unchanged (never
        # model_dump()).
        selection = self.resolve_selection(experiment_dir, recording_ids, config)
        chosen = selection.get("tracking_run_ids") or {}
        for sid in recording_ids:
            data = recording_store.read_recording(
                experiment_store.recording_path(experiment_dir, sid)
            )
            run_id = chosen.get(sid)
            teacher_window.refuse_teacher_run(
                {r["id"]: r for r in data.get("tracking_runs", [])}.get(run_id),
                consumer="location export",
                upstream=f"recording {sid!r}'s tracking run {run_id!r}",
            )
            calibration = data.get(spatial_store.CALIBRATION_KEY)
            resolution = (data.get("video") or {}).get("resolution")
            if calibration is not None and resolution:
                refusal = spatial_store.calibration_aspect_refusal(
                    calibration, *resolution
                )
                if refusal is not None:
                    raise ValueError(f"recording {sid!r}: {refusal}")
        config.update(selection)
        config["keypoint_run_ids"] = self._resolve_keypoints(
            experiment_dir, recording_ids, config, chosen
        )
        return config

    @staticmethod
    def _resolve_keypoints(experiment_dir, recording_ids, config, tracking) -> dict:
        """``{recording_id: kt_NN}`` — each recording's own keypoint track run: the run
        ``keypoint_track_overrides`` names for it (``--keypoints-for``, §6.5 per-recording
        overrides), else its run of a ``keypoint_track_spec`` (the GUI picker,
        ``--keypoints-ref``) or a literal ``keypoint_run_id``. A recording none of them
        reaches exports no keypoints.

        Raises:
            ValueError: A recording has no matching run, or its run is not complete, is a
                teacher-window run, or assigned its poses to another tracking run than the
                one this export reads there (its object_ids would be another run's); or an
                override names a recording outside the selection.
        """
        from detail.api import run_spec

        spec = config.get("keypoint_track_spec")
        literal = config.get("keypoint_run_id")
        entries = []
        for sid in recording_ids:
            recording_dir = experiment_store.recording_path(experiment_dir, sid)
            entries.append(
                (sid, recording_store.read_recording(recording_dir), recording_dir)
            )
        overrides = run_spec.resolve_overrides(
            entries, "keypoint_track", config.get("keypoint_track_overrides") or {}
        )
        if spec is None and not literal and not overrides:
            return {}
        resolved = {}
        for sid, data, recording_dir in entries:
            if sid in overrides:
                run_id = overrides[sid]
            elif spec is not None:
                run_id = run_spec.resolve_spec_in_recording(
                    data, "keypoint_track", spec, recording_dir
                )
            elif literal:
                run_id = literal
            else:
                continue
            entry = {r["id"]: r for r in data.get("keypoint_track_runs", [])}.get(
                run_id
            )
            if entry is None:
                raise ValueError(
                    f"recording {sid!r} has no keypoint track run "
                    + (f"{literal!r}" if spec is None else "with those settings")
                )
            if entry.get("status") != "complete":
                raise ValueError(
                    f"recording {sid!r}: keypoint track run {run_id!r} is "
                    f"{entry.get('status')!r}, not complete"
                )
            teacher_window.refuse_teacher_run(
                entry,
                consumer="location export",
                upstream=f"recording {sid!r}'s keypoint track run {run_id!r}",
            )
            from detail.storage.keypoint_runs import KeypointTrackRun

            tracked = str(
                KeypointTrackRun(recording_dir, run_id).meta["tracking_run_id"]
            )
            if tracked != str(tracking.get(sid)):
                raise ValueError(
                    f"recording {sid!r}: keypoint track run {run_id!r} assigned its poses "
                    f"to tracking run {tracked!r}, not this export's {tracking.get(sid)!r}; "
                    "its animals are not this export's, so its points would land on the "
                    "wrong animal"
                )
            resolved[sid] = run_id
        return resolved

    def build_job(self, run_id: str, config: dict) -> LocationAnalysisJob:
        recording_ids = list(config["source_recording_ids"])
        # validate() enriches with the resolved per-recording map; a direct caller may
        # pass a single ``tracking_run_id`` (uniform across the selection).
        mapping = config.get("tracking_run_ids")
        if mapping is None:
            run_id_single = config.get("tracking_run_id")
            mapping = {sid: run_id_single for sid in recording_ids}
        keypoints = config.get("keypoint_run_ids")
        if keypoints is None and config.get("keypoint_run_id"):
            keypoints = {sid: config["keypoint_run_id"] for sid in recording_ids}
        return LocationAnalysisJob(
            job_id=run_id,
            source_recording_ids=recording_ids,
            tracking_run_ids=dict(mapping),
            created_at=utc_timestamp(),
            keypoint_run_ids=dict(keypoints or {}),
            pair_edge_distance=bool(config.get("pair_edge_distance", False)),
        )

    def run_dir_for(self, experiment_root, job):
        return analysis_store.analysis_dir(experiment_root, job.job_id)

    def output_manifest(self, run_id: str, config: dict) -> list[dict]:
        """Relative per-recording CSV paths (no absolute paths in experiment files)."""
        keypoints = config.get("keypoint_run_ids") or {}
        return [
            {"path": f"{sid}/{kind}.csv", "kind": kind, "label": f"{sid} · {kind}"}
            for sid in config.get("source_recording_ids", [])
            for kind in _CSV_KINDS + (("keypoints",) if sid in keypoints else ())
        ]

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        from detail.analysis.locations import compute_location_tables
        from detail.api.recording import Recording
        from detail.storage.location_export import write_location_tables

        run_dir = Path(run_dir)
        experiment_root = run_dir.parent.parent  # <experiment>/analysis/<run_id>/
        recordings = job.source_recording_ids
        total = len(recordings)
        emit = ThrottledProgress(progress_cb)
        started = time.monotonic()

        per_recording: dict[str, dict] = {}
        succeeded: list[str] = []
        for sid in recordings:
            if stop_event.is_set():
                break

            def on_rows(done, rows_total, _i=len(succeeded), _sid=sid):
                elapsed = time.monotonic() - started
                within = done / rows_total if rows_total else 0.0
                overall = (_i + within) / total if total else 0.0
                emit(
                    LocationProgressEvent(
                        recordings_done=_i,
                        recordings_total=total,
                        frames_done=done,
                        frames_total=rows_total,
                        elapsed_s=elapsed,
                        eta_s=remaining_seconds(overall, elapsed),
                        message=f"computing · {_sid} ({_i + 1}/{total})",
                    )
                )

            on_rows(0, 1)  # a tick at recording start so the bar moves immediately
            try:
                recording = Recording(experiment_root, sid)
                tables = compute_location_tables(
                    recording,
                    job.tracking_run_ids[sid],
                    on_progress=on_rows,
                    pair_edge_distance=job.pair_edge_distance,
                    keypoint_run_id=job.keypoint_run_ids.get(sid),
                )
                write_location_tables(run_dir / sid, tables)
            except Exception as exc:  # noqa: BLE001 — per-recording isolation boundary
                # A recording (or its tracking run) deleted between validate and run
                # must not orphan the CSVs already written for earlier recordings: like
                # the run orchestrator, record the failure and keep going.
                per_recording[sid] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            per_recording[sid] = {"status": "complete"}
            succeeded.append(sid)

        emit(
            LocationProgressEvent(
                recordings_done=len(succeeded),
                recordings_total=total,
                frames_done=0,
                frames_total=0,
                elapsed_s=time.monotonic() - started,
                eta_s=0.0,
                message="done",
            ),
            force=True,
        )

        if len(succeeded) == total:
            status = "complete"
        elif succeeded:
            status = "partial"  # some recordings produced CSVs; others failed/cancelled
        else:
            status = "failed"

        # Finalize analysis.yaml with lineage + per-recording status + manifest (the run
        # id's record was pre-written with status=running by Experiment.run_analysis).
        record = analysis_store.read_analysis_yaml(run_dir)
        record["tracking_run_ids"] = dict(job.tracking_run_ids)
        record["per_recording"] = per_recording
        record["artifacts"] = self.output_manifest(
            job.job_id,
            {
                "source_recording_ids": succeeded,
                "keypoint_run_ids": job.keypoint_run_ids,
            },
        )
        record["status"] = status
        # Every number this run was launched with, derived from the job rather than
        # hand-listed (Tenet 9: a run is reproducible from its own metadata alone). Without it
        # the record names its recordings and its artifacts and nothing else — which is what
        # `an_17` on disk looks like, and why Refine re-opened such a run at defaults.
        record["params"] = job_params(job)
        analysis_store.write_analysis_yaml(run_dir, record)
