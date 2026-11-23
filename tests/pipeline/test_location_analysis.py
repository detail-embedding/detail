"""Location-export analysis module (§16) — the first concrete AnalysisModule."""

import json
import shutil
import threading

import numpy as np
import pytest
import yaml
from pycocotools import mask as mask_utils

import detail.pipeline.location_analysis  # noqa: F401 (registers the module)
from detail.api.experiment import Experiment
from detail.core import progress as progress_mod
from detail.executor.runners import ANALYSIS_MODULES
from detail.storage import analysis_runs as ar
from detail.storage import recording as recording_store
from detail.storage.zarr_io import AssignmentPartWriter, DetectionPartWriter
from tests.pipeline.phase_messages import assert_messages_name_their_phases

H = W = 40


def _rle(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode(
        "ascii"
    )


def _mask_square(r0, r1, c0, c1) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    m[r0:r1, c0:c1] = 1
    return m


def _add_tracked_recording(experiment, sid, *, complete=True):
    """A recording with a complete det_01 + track_01 (one detection assigned to 0)."""
    recording = experiment.create_recording(sid)
    sdir = recording.dir
    rel = "video/timestamps/generated/part_001.csv"
    (sdir / rel).parent.mkdir(parents=True, exist_ok=True)
    recording_store.write_timestamps_csv(
        sdir / rel, np.arange(5, dtype=np.int64) * 1000
    )
    data = recording_store.read_recording(sdir)
    data["video"] = {
        "fps_actual": 30.0,
        "resolution": [W, H],
        "files": [
            {
                "index": 0,
                "file": "video/part_001.mp4",
                "timestamps": {"generated": rel, "user": None, "active": "generated"},
            }
        ],
    }
    # Realistic run records carry their full identity (tracker + hyperparams + the
    # detection lineage), so a spec built from one recording resolves in another by real
    # equivalence — not by an accidental same-ordinal legacy sentinel (§6.5).
    data["detection_runs"] = [
        {
            "id": "det_01",
            "status": "complete",
            "model_id": "sam3",
            "model_variant": "large",
            "weights_source": "pretrained",
            "conf_threshold": 0.25,
            "prompt_fingerprint": "fp",
            "prompt_classes": ["mouse"],
        }
    ]
    data["tracking_runs"] = [
        {
            "id": "track_01",
            "status": "complete" if complete else "running",
            "detection_run_id": "det_01",
            "tracker_id": "anchor_tracker",
            "arena_ids": [],
            "hyperparams": {
                "conf_threshold": 0.5,
                "iou_threshold": 0.5,
                "max_missed_frames": 5,
            },
        }
    ]
    recording_store.write_recording(sdir, data)
    det_dir = sdir / "detections" / "det_01"
    det_dir.mkdir(parents=True)
    (det_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "mask_resolution": [H, W],
                "job": {"downsampling_run_id": None},
            }
        ),
        encoding="utf-8",
    )
    with DetectionPartWriter(det_dir / "part_001.zarr") as w:
        w.add_row(
            0,
            0,
            0,
            "mouse",
            0.9,
            (0.1, 0.1, 0.5, 0.5),
            _rle(_mask_square(4, 20, 4, 20)),
        )
    trk_dir = sdir / "tracking" / "track_01"
    (trk_dir / "assignments").mkdir(parents=True)
    (trk_dir / "track.yaml").write_text(
        yaml.dump({"detection_run_id": "det_01"}), encoding="utf-8"
    )
    with AssignmentPartWriter(trk_dir / "assignments" / "part_001.zarr") as w:
        w.add_object_id(0)
    return recording


def _config(*sids):
    return {"source_recording_ids": list(sids), "tracking_run_id": "track_01"}


def test_location_analysis_end_to_end(tmp_path):
    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    run = proj.run_analysis("location_export", _config("recording_001"))
    run.wait()
    assert run.status == "complete"

    base = proj.dir / "analysis" / "an_01" / "recording_001"
    for name in ("tracks.csv", "roi_relations.csv", "point_relations.csv"):
        assert (base / name).is_file()
    rec = proj.analysis_record("an_01")
    assert rec["status"] == "complete"
    # a single --tracking id becomes a uniform per-recording map (§6.5)
    assert rec["tracking_run_ids"] == {"recording_001": "track_01"}
    assert rec["artifacts"][0]["path"] == "recording_001/tracks.csv"  # relative path
    # options nobody set are still written down, resolved (Tenet 9): the record says every
    # frame rather than saying nothing
    assert rec["params"]["sample_fps"] is None
    assert "remove_thin_parts" not in rec["params"]


def test_location_analysis_accepts_tracking_spec(tmp_path):
    """A tracking *spec* (the GUI / CLI --tracking-ref path) resolves to each recording's
    own tracking run — recorded as a per-recording map (§6.5)."""
    from detail.api import run_spec

    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    _add_tracked_recording(proj, "recording_002")
    spec = run_spec.spec_from_run(
        proj.get_recording("recording_001"), "tracking", "track_01"
    )
    run = proj.run_analysis(
        "location_export",
        {
            "source_recording_ids": ["recording_001", "recording_002"],
            "tracking_spec": spec,
        },
    )
    run.wait()
    assert run.status == "complete"
    rec = proj.analysis_record(run.id)
    assert rec["tracking_run_ids"] == {
        "recording_001": "track_01",
        "recording_002": "track_01",
    }


def test_location_job_from_dict_legacy_single_id_shim():
    """A pre-spec job (single ``tracking_run_id``) reads as a uniform per-recording map, so
    a durable analysis launched before the spec change still replays from job.json (§6.5).
    """
    from detail.pipeline.location_analysis import LocationAnalysisJob

    legacy = {
        "job_id": "an_01",
        "source_recording_ids": ["s1", "s2"],
        "tracking_run_id": "track_01",
        "created_at": "t",
    }
    assert LocationAnalysisJob.from_dict(legacy).tracking_run_ids == {
        "s1": "track_01",
        "s2": "track_01",
    }
    # a job from before the measurement options existed measured every frame, which is
    # what the config model's defaults say
    assert LocationAnalysisJob.from_dict(legacy).sample_fps is None
    new = LocationAnalysisJob(
        "an_02", ["s1", "s2"], {"s1": "track_02", "s2": "track_01"}, "t", 5.0
    )
    assert LocationAnalysisJob.from_dict(new.to_dict()).sample_fps == 5.0
    # a job written while thin-part removal existed still reads; the keys are ignored
    older = {**new.to_dict(), "remove_thin_parts": False, "thin_part_scale": 0.12}
    assert LocationAnalysisJob.from_dict(older).sample_fps == 5.0
    assert LocationAnalysisJob.from_dict(new.to_dict()).tracking_run_ids == {
        "s1": "track_02",
        "s2": "track_01",
    }


def test_location_validate_resolves_tracking_spec_to_divergent_ordinals(tmp_path):
    """The analysis spec path resolves to each recording's OWN tracking run at divergent
    ordinals — a per-recording map, not a single shared id (§6.5)."""
    from detail.api import run_spec
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import ANALYSIS_MODULES, import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)

    def trk(rid, **over):
        return {
            "id": rid, "status": "complete", "output": f"tracking/{rid}/", "created_at": "t",
            "tracker_id": "anchor_tracker", "arena_ids": [], "hyperparams": {"conf_threshold": 0.5},
            **over,
        }  # fmt: skip

    proj = Experiment.create("p", tmp_path)
    for sid, trks in [
        ("s1", [trk("track_01", hyperparams={"conf_threshold": 0.9}), trk("track_02")]),
        ("s2", [trk("track_01")]),  # same "conf 0.5" kind, at a different ordinal
    ]:
        proj.create_recording(sid)
        sdir = proj.get_recording(sid).dir
        data = recording_store.read_recording(sdir)
        data["tracking_runs"] = trks
        recording_store.write_recording(sdir, data)

    spec = run_spec.spec_from_run(proj.get_recording("s2"), "tracking", "track_01")
    cfg = ANALYSIS_MODULES["location_export"].validate(
        proj.dir, {"source_recording_ids": ["s1", "s2"], "tracking_spec": spec}
    )
    assert cfg["tracking_run_ids"] == {"s1": "track_02", "s2": "track_01"}


def test_location_analysis_batches_over_recordings(tmp_path):
    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    _add_tracked_recording(proj, "recording_002")
    run = proj.run_analysis(
        "location_export", _config("recording_001", "recording_002")
    )
    run.wait()
    assert run.status == "complete"
    for sid in ("recording_001", "recording_002"):
        assert (proj.dir / "analysis" / "an_01" / sid / "tracks.csv").is_file()


def test_validate_rejects_incomplete_tracking(tmp_path):
    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001", complete=False)
    with pytest.raises(ValueError, match="not complete"):
        proj.run_analysis("location_export", _config("recording_001"))


def test_validate_rejects_unknown_tracking_run(tmp_path):
    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    with pytest.raises(ValueError, match="no tracking run"):
        proj.run_analysis(
            "location_export",
            {"source_recording_ids": ["recording_001"], "tracking_run_id": "track_99"},
        )


def _add_processing_tracking_run(proj, run_id, sid, stage_run_id="track_01"):
    from detail.storage import runs as runs_store

    runs_store.write_run(
        proj.dir,
        run_id,
        {
            "id": run_id,
            "module_id": "tracking",
            "recording_ids": [sid],
            "per_recording": {sid: {"run_id": stage_run_id, "status": "complete"}},
            "status": "complete",
        },
    )


def _write_analysis(proj, run_id, sid, status):
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, run_id),
        {
            "id": run_id,
            "module_id": "location_export",
            "recording_ids": [sid],
            "tracking_run_id": "track_01",
            "status": status,
        },
    )


def test_completed_analysis_does_not_block_tracking_delete(tmp_path):
    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    _add_processing_tracking_run(proj, "run_001", "recording_001")
    _write_analysis(proj, "an_01", "recording_001", "complete")
    proj.delete_run("run_001")  # a completed analysis run dangles — no block
    assert [r.id for r in proj.analysis_runs] == ["an_01"]  # untouched on disk


def test_active_analysis_blocks_tracking_delete(tmp_path):
    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    _add_processing_tracking_run(proj, "run_001", "recording_001")
    _write_analysis(proj, "an_01", "recording_001", "running")  # mid-read
    with pytest.raises(ValueError, match="still reading it"):
        proj.delete_run("run_001")


def test_progress_emits_intermediate_event(tmp_path, monkeypatch):
    # Call the runner directly (no executor poll race) with the throttle off, and
    # assert a per-recording intermediate event lands before the terminal one.
    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    _add_tracked_recording(proj, "recording_002")
    module = ANALYSIS_MODULES["location_export"]
    job = module.build_job("an_01", _config("recording_001", "recording_002"))
    run_dir = ar.analysis_dir(proj.dir, "an_01")
    ar.write_analysis_yaml(
        run_dir,
        {"id": "an_01", "module_id": "location_export", "status": "running"},
    )
    events = []
    module.runner(job, run_dir, events.append, threading.Event())
    dones = [e.recordings_done for e in events]
    assert dones[-1] == 2  # terminal (forced)
    assert any(0 < d < 2 for d in dones)  # an intermediate per-recording tick
    # §16, issue #393: the module's declared phase word actually reaches the message this
    # same run just emitted, not only the registry's static declaration.
    messages = [e.message for e in events if e.message]
    assert_messages_name_their_phases(messages, module.phases)


def _run_directly(proj, *sids):
    """Invoke the runner in-process (bypassing validate, like the worker subprocess
    would after a recording was deleted between submit and run). Returns the run dir."""
    module = ANALYSIS_MODULES["location_export"]
    job = module.build_job("an_01", _config(*sids))
    run_dir = ar.analysis_dir(proj.dir, "an_01")
    ar.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": "location_export", "status": "running"}
    )
    module.runner(job, run_dir, lambda *_: None, threading.Event())  # must not raise
    return run_dir


def test_runner_isolates_failed_recording(tmp_path):
    # A recording whose tracking run is deleted after validate must not fail the whole
    # batch: the earlier recording's CSVs survive and the run finishes "partial".
    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    bad = _add_tracked_recording(proj, "recording_002")
    shutil.rmtree(bad.dir / "tracking" / "track_01")  # gone between submit and run

    run_dir = _run_directly(proj, "recording_001", "recording_002")

    assert (
        run_dir / "recording_001" / "tracks.csv"
    ).is_file()  # good recording survives
    assert not (run_dir / "recording_002").exists()  # the failed one wrote nothing
    rec = ar.read_analysis_yaml(run_dir)
    assert rec["status"] == "partial"
    assert rec["per_recording"]["recording_001"]["status"] == "complete"
    assert rec["per_recording"]["recording_002"]["status"] == "failed"
    assert rec["per_recording"]["recording_002"]["error"]  # a reason is recorded
    assert {a["path"] for a in rec["artifacts"]} == {
        f"recording_001/{k}.csv" for k in ("tracks", "roi_relations", "point_relations")
    }  # only the recording whose CSVs are on disk


def test_runner_all_recordings_fail_is_failed(tmp_path):
    proj = Experiment.create("p", tmp_path)
    for sid in ("recording_001", "recording_002"):
        recording = _add_tracked_recording(proj, sid)
        shutil.rmtree(recording.dir / "tracking" / "track_01")

    run_dir = _run_directly(proj, "recording_001", "recording_002")

    rec = ar.read_analysis_yaml(run_dir)
    assert rec["status"] == "failed"
    assert rec["artifacts"] == []
    assert all(v["status"] == "failed" for v in rec["per_recording"].values())


def test_validate_refuses_a_calibration_of_another_shape(tmp_path):
    """The _mm columns would be wrong in some direction, so the run is refused up front
    rather than failing per recording — or, as before, writing a plausible wrong number.
    """
    proj = Experiment.create("p", tmp_path)
    rec = _add_tracked_recording(proj, "recording_001")
    data = recording_store.read_recording(rec.dir)
    data["calibration"] = {
        "schema_version": 1,
        "points": None,
        "distance_mm": None,
        "pixels_per_mm": 5.0,
        "ref_resolution": [W * 2, H],
    }
    recording_store.write_recording(rec.dir, data)
    with pytest.raises(ValueError, match="recording_001.*different video shape"):
        proj.run_analysis("location_export", _config("recording_001"))


def test_the_runner_measures_with_the_options_the_job_carries(tmp_path):
    """The options reach the compute through the job, not only through ``validate``: a worker
    rebuilds the job from ``job.json`` and never sees the config (detail#75)."""
    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    module = ANALYSIS_MODULES["location_export"]
    config = {**_config("recording_001"), "sample_fps": 5.0}
    job = type(module.build_job("an_01", config)).from_dict(
        module.build_job("an_01", config).to_dict()
    )
    run_dir = ar.analysis_dir(proj.dir, "an_01")
    ar.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": "location_export", "status": "running"}
    )
    module.runner(job, run_dir, lambda *_: None, threading.Event())
    manifest = yaml.safe_load(
        (run_dir / "recording_001" / "locations.yaml").read_text(encoding="utf-8-sig")
    )
    assert manifest["sample_fps"] == 5.0
    record = ar.read_analysis_yaml(run_dir)
    assert record["status"] == "complete"
    assert record["params"]["sample_fps"] == 5.0


def test_validate_refuses_an_option_outside_its_bounds(tmp_path):
    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    module = ANALYSIS_MODULES["location_export"]
    with pytest.raises(ValueError, match="sample_fps"):
        module.validate(proj.dir, {**_config("recording_001"), "sample_fps": -1.0})
    # and a good config comes back with every option resolved, set or not
    resolved = module.validate(proj.dir, {**_config("recording_001"), "sample_fps": 5})
    assert resolved["sample_fps"] == 5.0
    assert "remove_thin_parts" not in resolved


def test_validate_refuses_a_config_asking_for_thin_parts_removed(tmp_path):
    # the model ignores a key it does not have; this one would silently measure the tail
    proj = Experiment.create("p", tmp_path)
    _add_tracked_recording(proj, "recording_001")
    module = ANALYSIS_MODULES["location_export"]
    with pytest.raises(ValueError, match="no longer removes thin parts"):
        module.validate(
            proj.dir, {**_config("recording_001"), "remove_thin_parts": True}
        )
    # an older record's explicit "off" is what it measures anyway, so it still runs
    module.validate(proj.dir, {**_config("recording_001"), "remove_thin_parts": False})
