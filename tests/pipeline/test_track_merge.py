"""Track-merge (§13): combine 2+ single-subject tracking runs into one multi-subject run."""

import json

import numpy as np
import pytest

from detail.api import run_spec
from detail.core.dataclasses import TrackingJob
from detail.executor.runners import PROCESSING_MODULES, import_runner_modules
from detail.pipeline import teacher_window, track_merge, tracking
from detail.storage import recording as recording_store
from detail.storage.zarr_io import (
    DetectionPartWriter,
    PartZarrReader,
)


@pytest.fixture(autouse=True)
def _runner_modules():
    from detail.api._runner_modules import PIPELINE_RUNNERS

    import_runner_modules(PIPELINE_RUNNERS)


def _write_part(path, frames):
    with DetectionPartWriter(path) as w:
        for frame_idx, dets in frames:
            for d_idx, (bbox, conf, cname) in enumerate(dets):
                w.add_row(
                    frame_idx, frame_idx * 1000, d_idx, cname, float(conf), bbox, ""
                )


def _build_two_class_detection_run(experiment_root, n_frames=4):
    """Both a black mouse and a white mouse detected in every frame (§13's class filter)."""
    det = experiment_root / "recordings" / "recording_001" / "detections" / "det_01"
    det.mkdir(parents=True)
    (det / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [24, 32]}), encoding="utf-8"
    )
    _write_part(
        det / "part_001.zarr",
        [
            (
                f,
                [
                    ((0.1, 0.1, 0.2, 0.2), 0.9, "black mouse"),
                    ((0.6, 0.6, 0.7, 0.7), 0.8, "white mouse"),
                ],
            )
            for f in range(n_frames)
        ],
    )
    return det


def _run_class_filtered_tracking(experiment_root, track_id, class_name):
    job = TrackingJob(
        job_id=track_id,
        recording_id="recording_001",
        track_id=track_id,
        detection_run_id="det_01",
        tracker_id="anchor_tracker",
        hyperparams={"conf_threshold": 0.05},
        arena_ids=[],
        created_at="2026-01-01T00:00:00Z",
        class_name=class_name,
    )
    run_dir = experiment_root / "recordings" / "recording_001" / "tracking" / track_id
    run_dir.mkdir(parents=True)
    tracking.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    return job


def _write_recording_record(experiment_root, tracking_ids_and_classes):
    sdir = experiment_root / "recordings" / "recording_001"
    data = {
        "id": "recording_001",
        "detection_runs": [{"id": "det_01", "status": "complete"}],
        "tracking_runs": [
            {
                "id": track_id,
                "status": "complete",
                "detection_run_id": "det_01",
                "tracker_id": "anchor_tracker",
                "arena_ids": [],
                "class_name": class_name,
                "hyperparams": {"conf_threshold": 0.05},
                "every_n_frames": 1,
                "created_at": "2026-01-01T00:00:00Z",
            }
            for track_id, class_name in tracking_ids_and_classes
        ],
    }
    recording_store.write_recording(sdir, data)
    return data


def _two_track_recording(tmp_path, n_frames=4):
    """Two class-filtered tracking runs (track_black, track_white) over one 2-class
    detection run, plus the recording record they need to be discoverable by spec."""
    _build_two_class_detection_run(tmp_path, n_frames=n_frames)
    _run_class_filtered_tracking(tmp_path, "track_black", "black mouse")
    _run_class_filtered_tracking(tmp_path, "track_white", "white mouse")
    data = _write_recording_record(
        tmp_path, [("track_black", "black mouse"), ("track_white", "white mouse")]
    )
    return data


def _spec_for(data, recording_dir, run_id):
    record = next(r for r in data["tracking_runs"] if r["id"] == run_id)
    return run_spec.spec_of(data, "tracking", record, recording_dir)


def test_validate_resolves_two_matching_sources(tmp_path):
    data = _two_track_recording(tmp_path)
    recording_dir = tmp_path / "recordings" / "recording_001"
    module = PROCESSING_MODULES["track_merge"]
    config = module.validate(
        recording_dir,
        data,
        {
            "source_specs": [
                _spec_for(data, recording_dir, "track_black"),
                _spec_for(data, recording_dir, "track_white"),
            ]
        },
    )
    assert set(config["source_run_ids"]) == {"track_black", "track_white"}
    assert config["detection_run_id"] == "det_01"
    assert config["every_n_frames"] == 1


def test_reversed_source_order_gets_a_different_identity(tmp_path):
    """Two merges built from the same two sources in opposite order must not share a
    spec (§13): object_id 0 is source_run_ids[0], so it names a different animal in
    each, and a picker or resolve_spec_in_recording's newest-wins tie-break has to be
    able to tell them apart rather than silently treating "track_merge" as one thing
    regardless of which source became which object_id."""
    data = _two_track_recording(tmp_path)
    recording_dir = tmp_path / "recordings" / "recording_001"
    module = PROCESSING_MODULES["track_merge"]

    forward = module.validate(
        recording_dir,
        data,
        {
            "source_specs": [
                _spec_for(data, recording_dir, "track_black"),
                _spec_for(data, recording_dir, "track_white"),
            ]
        },
    )
    reverse = module.validate(
        recording_dir,
        data,
        {
            "source_specs": [
                _spec_for(data, recording_dir, "track_white"),
                _spec_for(data, recording_dir, "track_black"),
            ]
        },
    )
    assert forward["source_classes"] == ["black mouse", "white mouse"]
    assert reverse["source_classes"] == ["white mouse", "black mouse"]

    forward_record = {
        "id": "track_forward",
        "status": "complete",
        **module.extra_record_fields(forward),
    }
    reverse_record = {
        "id": "track_reverse",
        "status": "complete",
        **module.extra_record_fields(reverse),
    }
    forward_spec = run_spec.spec_of(data, "tracking", forward_record, recording_dir)
    reverse_spec = run_spec.spec_of(data, "tracking", reverse_record, recording_dir)
    assert run_spec.freeze_spec(forward_spec) != run_spec.freeze_spec(reverse_spec)
    assert run_spec.spec_label(forward_spec) != run_spec.spec_label(reverse_spec)


def test_validate_rejects_same_class_twice(tmp_path):
    """Two genuinely distinct tracking runs (different hyperparams, so different specs
    and different resolved run ids) that happen to share a class filter — distinct from
    ``test_validate_rejects_the_same_run_selected_twice`` below, which is the same run
    picked twice."""
    _build_two_class_detection_run(tmp_path)
    job_a = TrackingJob(
        job_id="track_a",
        recording_id="recording_001",
        track_id="track_a",
        detection_run_id="det_01",
        tracker_id="anchor_tracker",
        hyperparams={"conf_threshold": 0.05},
        arena_ids=[],
        created_at="2026-01-01T00:00:00Z",
        class_name="black mouse",
    )
    run_dir_a = tmp_path / "recordings" / "recording_001" / "tracking" / "track_a"
    run_dir_a.mkdir(parents=True)
    tracking.run(job_a, run_dir_a, progress_cb=lambda e: None, stop_event=None)
    job_b = TrackingJob(
        job_id="track_b",
        recording_id="recording_001",
        track_id="track_b",
        detection_run_id="det_01",
        tracker_id="anchor_tracker",
        hyperparams={"conf_threshold": 0.1},  # differs, so its spec differs too
        arena_ids=[],
        created_at="2026-01-01T00:00:00Z",
        class_name="black mouse",
    )
    run_dir_b = tmp_path / "recordings" / "recording_001" / "tracking" / "track_b"
    run_dir_b.mkdir(parents=True)
    tracking.run(job_b, run_dir_b, progress_cb=lambda e: None, stop_event=None)

    sdir = tmp_path / "recordings" / "recording_001"
    data = {
        "id": "recording_001",
        "detection_runs": [{"id": "det_01", "status": "complete"}],
        "tracking_runs": [
            {
                "id": "track_a",
                "status": "complete",
                "detection_run_id": "det_01",
                "tracker_id": "anchor_tracker",
                "arena_ids": [],
                "class_name": "black mouse",
                "hyperparams": {"conf_threshold": 0.05},
                "every_n_frames": 1,
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "id": "track_b",
                "status": "complete",
                "detection_run_id": "det_01",
                "tracker_id": "anchor_tracker",
                "arena_ids": [],
                "class_name": "black mouse",
                "hyperparams": {"conf_threshold": 0.1},
                "every_n_frames": 1,
                "created_at": "2026-01-01T00:00:00Z",
            },
        ],
    }
    recording_store.write_recording(sdir, data)
    recording_dir = sdir
    module = PROCESSING_MODULES["track_merge"]
    with pytest.raises(ValueError, match="share the same class filter"):
        module.validate(
            recording_dir,
            data,
            {
                "source_specs": [
                    _spec_for(data, recording_dir, "track_a"),
                    _spec_for(data, recording_dir, "track_b"),
                ]
            },
        )


def test_validate_rejects_an_unfiltered_source(tmp_path):
    _build_two_class_detection_run(tmp_path)
    _run_class_filtered_tracking(tmp_path, "track_a", "black mouse")
    _run_class_filtered_tracking(tmp_path, "track_b", None)
    data = _write_recording_record(
        tmp_path, [("track_a", "black mouse"), ("track_b", None)]
    )
    recording_dir = tmp_path / "recordings" / "recording_001"
    module = PROCESSING_MODULES["track_merge"]
    with pytest.raises(ValueError, match="own class filter"):
        module.validate(
            recording_dir,
            data,
            {
                "source_specs": [
                    _spec_for(data, recording_dir, "track_a"),
                    _spec_for(data, recording_dir, "track_b"),
                ]
            },
        )


def test_validate_rejects_mismatched_detection_runs(tmp_path):
    data = _two_track_recording(tmp_path)
    recording_dir = tmp_path / "recordings" / "recording_001"
    # Forge a third source pointing at a different detection run.
    data["tracking_runs"].append(
        {
            "id": "track_other",
            "status": "complete",
            "detection_run_id": "det_02",
            "tracker_id": "anchor_tracker",
            "arena_ids": [],
            "class_name": "gray mouse",
            "hyperparams": {},
            "every_n_frames": 1,
            "created_at": "2026-01-01T00:00:00Z",
        }
    )
    module = PROCESSING_MODULES["track_merge"]
    with pytest.raises(ValueError, match="different detection runs"):
        module.validate(
            recording_dir,
            data,
            {
                "source_specs": [
                    _spec_for(data, recording_dir, "track_black"),
                    _spec_for(data, recording_dir, "track_other"),
                ]
            },
        )


def test_merge_runner_produces_one_object_id_per_source(tmp_path):
    _two_track_recording(tmp_path, n_frames=4)
    recording_dir = tmp_path / "recordings" / "recording_001"

    job = track_merge.TrackMergeJob(
        job_id="track_merged",
        recording_id="recording_001",
        track_id="track_merged",
        detection_run_id="det_01",
        source_run_ids=["track_black", "track_white"],
        created_at="2026-01-01T00:00:00Z",
        every_n_frames=1,
    )
    run_dir = recording_dir / "tracking" / "track_merged"
    run_dir.mkdir(parents=True)
    track_merge.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)

    assigned = PartZarrReader(run_dir / "assignments" / "part_001.zarr").read()
    det_data = PartZarrReader(
        recording_dir / "detections" / "det_01" / "part_001.zarr"
    ).read()
    object_id = assigned["object_id"]
    class_name = np.asarray([str(c) for c in det_data["class_name"]])
    assert (object_id[class_name == "black mouse"] == 0).all()
    assert (object_id[class_name == "white mouse"] == 1).all()

    tracks = PartZarrReader(run_dir / "tracks.zarr").read()
    assert tracks["track_id"].tolist() == [0, 1]
    assert [str(c) for c in tracks["class_name"]] == ["black mouse", "white mouse"]

    import yaml

    from detail.core import atomic

    track_yaml = yaml.safe_load(atomic.read_text(run_dir / "track.yaml"))
    assert track_yaml["subject_labels"] == {0: "black mouse", 1: "white mouse"}
    assert track_yaml["tracker_id"] == "track_merge"
    assert track_yaml["status"] == "complete"
    assert track_yaml["n_tracks"] == 2


def _merge_config(data, recording_dir):
    return {
        "source_specs": [
            _spec_for(data, recording_dir, "track_black"),
            _spec_for(data, recording_dir, "track_white"),
        ]
    }


def test_merging_teacher_tracks_makes_a_teacher_run(tmp_path):
    """The two-animal keypoint setup (§26) merges each animal's teacher track, so the real
    validate takes sources that share a teacher window — and records it, so the merged run is
    a teacher run that embedding, location export and every other whole-recording consumer
    still refuses (``teacher_window.refuse_teacher_run`` reads that field)."""
    data = _two_track_recording(tmp_path)
    recording_dir = tmp_path / "recordings" / "recording_001"
    for run in data["tracking_runs"]:
        run["teacher_window_s"] = 300.0
    recording_store.write_recording(recording_dir, data)
    module = PROCESSING_MODULES["track_merge"]
    config = module.validate(recording_dir, data, _merge_config(data, recording_dir))
    record = module.extra_record_fields(config)
    assert record["teacher_window_s"] == 300.0
    with pytest.raises(ValueError, match="keypoint teacher run .first 5 min"):
        teacher_window.refuse_teacher_run(
            record, consumer="embedding", upstream="the merged run"
        )


def test_a_whole_recording_merge_is_not_a_teacher_run(tmp_path):
    data = _two_track_recording(tmp_path)
    recording_dir = tmp_path / "recordings" / "recording_001"
    module = PROCESSING_MODULES["track_merge"]
    config = module.validate(recording_dir, data, _merge_config(data, recording_dir))
    assert module.extra_record_fields(config)["teacher_window_s"] is None


def test_validate_refuses_sources_over_different_spans(tmp_path):
    """One teacher track and one whole-recording track: the merged ids would end at five
    minutes for one animal and run on for the other, read as one run."""
    data = _two_track_recording(tmp_path)
    recording_dir = tmp_path / "recordings" / "recording_001"
    data["tracking_runs"][0]["teacher_window_s"] = 300.0
    recording_store.write_recording(recording_dir, data)
    with pytest.raises(ValueError, match="first 5 min, the whole recording"):
        PROCESSING_MODULES["track_merge"].validate(
            recording_dir, data, _merge_config(data, recording_dir)
        )
