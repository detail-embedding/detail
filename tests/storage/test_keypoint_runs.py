"""The keypoint run records (§26 ``detect.yaml`` / ``track.yaml``) and their readers,
including the refusal of runs in the per-track layout from before 2026-10-21."""

import numpy as np
import pytest
import yaml

from detail.core.dataclasses import ComputeTarget, FrameResult
from detail.core.keypoints import KeypointDetectJob, KeypointTrackJob
from detail.storage import keypoint_runs
from detail.storage.keypoint_runs import LegacyKeypointLayoutError
from detail.storage.keypoint_zarr import (
    KeypointAssignmentPartWriter,
    KeypointDetectPartWriter,
    KeypointTrackPartWriter,
)

SKELETON = {"names": ["nose", "tail"], "edges": [["nose", "tail"]], "pairs": []}


def _job(**overrides) -> KeypointDetectJob:
    fields = dict(
        job_id="kd_01",
        recording_id="recording_001",
        detect_run_id="kd_01",
        detection_run_id="det_01",
        min_pose_confidence=None,
        model_id="dinov3_match",
        model_variant="vit_base",
        backbone={"id": "dinov3_vit", "variant": "vit_base"},
        example_set={"id": "ex_01", "digest": "abc", "frames": [], "marks": []},
        skeleton=SKELETON,
        n_candidates=5,
        every_n_frames=1,
        crop_px=448,
        decode="skeleton",
        side_rule="off",
        matching={"topk": 5},
        batch_size=16,
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-10-07T00:00:00Z",
    )
    fields.update(overrides)
    return KeypointDetectJob(**fields)


def _track_job() -> KeypointTrackJob:
    return KeypointTrackJob(
        job_id="kt_01",
        recording_id="recording_001",
        track_run_id="kt_01",
        detect_run_id="kd_01",
        tracking_run_id="track_01",
        match={"min_inside": 0.5, "mask_grow": 0.05},
        method="temporal_decode",
        config={"smooth_window_s": 0.133, "despike": True},
        skeleton=SKELETON,
        created_at="2026-10-08T00:00:00Z",
    )


def test_jobs_round_trip_and_an_old_job_is_refused():
    job = _job(min_pose_confidence=0.25)
    assert KeypointDetectJob.from_dict(job.to_dict()) == job
    old = job.to_dict()
    del old["detection_run_id"]
    old["tracking_run_id"], old["track_ids"] = "track_01", [0]
    with pytest.raises(ValueError, match="re-run keypoint detection"):
        KeypointDetectJob.from_dict(old)
    track = _track_job()
    assert KeypointTrackJob.from_dict(track.to_dict()) == track
    old = track.to_dict()
    del old["tracking_run_id"], old["match"]
    with pytest.raises(ValueError, match="re-run keypoint detection"):
        KeypointTrackJob.from_dict(old)


def test_detect_yaml_is_written_at_start_and_completed_at_the_end(tmp_path):
    run_dir = keypoint_runs.detect_run_dir(tmp_path / "recording_001", "kd_01")
    record = keypoint_runs.detect_record(_job(), "4b7e")
    assert list(record)[:5] == [
        "schema_version",  # Tenet 7
        "id",
        "created_at",
        "detection_run_id",
        "min_pose_confidence",
    ]
    keypoint_runs.write_detect_yaml(run_dir, record)
    text = (run_dir / "detect.yaml").read_text(encoding="utf-8-sig")
    assert text.startswith("schema_version: 1\n")
    data = yaml.safe_load(text)
    assert data["detection_run_id"] == "det_01" and data["min_pose_confidence"] is None
    assert "tracking_run_id" not in data and "track_ids" not in data
    assert data["backbone"] == {"id": "dinov3_vit", "variant": "vit_base"}
    assert data["skeleton_digest"] == "4b7e" and data["resolved"] == {}
    assert "status" not in data  # the executor's status is the only status
    keypoint_runs.update_detect_yaml(run_dir, resolved={"side_rules": {}})
    coverage = {"part_001": {"frames": 5, "frames_with_instance": 4, "instances": 6}}
    keypoint_runs.update_detect_yaml(
        run_dir,
        coverage=coverage,
        timing={"frames": 5, "seconds": 0.5, "frames_per_s": 10.0},
    )
    final = keypoint_runs.read_detect_yaml(run_dir)
    assert final["resolved"] == {"side_rules": {}} and final["coverage"] == coverage
    assert not list(run_dir.glob("*.tmp"))  # atomic: no leftover temp file
    assert keypoint_runs.detect_record(_job(min_pose_confidence=0.25), "x")[
        "min_pose_confidence"
    ] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="schema_version must be the first key"):
        keypoint_runs.write_yaml(run_dir / "x.yaml", {"id": 1, "schema_version": 1})


def _instances(frame: int, n: int) -> FrameResult:
    return FrameResult(
        frame_idx=frame,
        t_ns=frame * 10,
        n=n,
        detection_row=np.arange(n, dtype=np.int32),
        points=np.full((n, 2, 2), frame / 10, np.float32),
        point_scores=np.full((n, 2), 0.5, np.float32),
    )


def test_the_run_reader_finds_a_frames_instances_and_checks_the_skeleton(tmp_path):
    recording_dir = tmp_path / "recording_001"
    run_dir = keypoint_runs.detect_run_dir(recording_dir, "kd_01")
    keypoint_runs.write_detect_yaml(run_dir, keypoint_runs.detect_record(_job(), "x"))
    path = keypoint_runs.part_path(run_dir, "part_001")
    assert path == run_dir / "part_001.zarr"
    with KeypointDetectPartWriter(path, ("nose", "tail"), 1) as writer:
        for frame, n in ((2, 1), (5, 2), (9, 1)):
            result = _instances(frame, n)
            for row in range(n):
                writer.add_instance(result, row)
    run = keypoint_runs.KeypointDetectRun(recording_dir, "kd_01")
    assert run.skeleton.names == ("nose", "tail") and run.part_ids() == ["part_001"]
    assert run.detection_run_id == "det_01"
    assert run.n_candidates == 5 and run.coverage() == {}
    five = run.at("part_001", 5)
    assert five["row"].tolist() == [1, 2] and five["points"].shape == (2, 2, 2)
    assert float(five["points"][0, 0, 0]) == pytest.approx(0.5)
    assert five["point_scores"].tolist() == [[0.5, 0.5], [0.5, 0.5]]
    assert "candidates" not in five  # the store's candidate count, not the yaml's
    empty = run.at("part_001", 6)  # no instance on that frame
    assert len(empty["row"]) == 0 and empty["points"].shape == (0, 2, 2)
    assert len(run.at("part_002", 5)["frame_idx"]) == 0  # no store for that part
    assert run.rows("part_001")["frame_idx"].tolist() == [2, 5, 5, 9]
    run.meta["skeleton"]["names"] = ["a", "b"]  # yaml and attrs disagreeing
    with pytest.raises(ValueError, match="cannot be read"):
        run.rows("part_001")


@pytest.mark.parametrize("shape", ["track_dir", "yaml"])
def test_a_detect_run_in_the_per_track_layout_is_refused(tmp_path, shape):
    """Both marks of the old layout — a ``track_<id>/`` folder, a yaml naming a tracking run
    and no detection run — refuse the run with the one thing to do about it."""
    recording_dir = tmp_path / "recording_001"
    run_dir = keypoint_runs.detect_run_dir(recording_dir, "kd_01")
    record = keypoint_runs.detect_record(_job(), "x")
    if shape == "yaml":
        del record["detection_run_id"]
        record["tracking_run_id"], record["track_ids"] = "track_01", [0]
    keypoint_runs.write_detect_yaml(run_dir, record)
    if shape == "track_dir":
        (run_dir / "track_0" / "part_001.zarr").mkdir(parents=True)
    with pytest.raises(LegacyKeypointLayoutError, match="re-run keypoint detection"):
        keypoint_runs.KeypointDetectRun(recording_dir, "kd_01")


def test_track_yaml_and_reader(tmp_path):
    recording_dir = tmp_path / "recording_001"
    run_dir = keypoint_runs.track_run_dir(recording_dir, "kt_01")
    record = keypoint_runs.track_record(_track_job(), "4b7e")
    assert list(record)[0] == "schema_version"
    assert record["tracking_run_id"] == "track_01"
    assert record["match"] == {"min_inside": 0.5, "mask_grow": 0.05}
    keypoint_runs.write_track_yaml(run_dir, record)
    text = (run_dir / "track.yaml").read_text(encoding="utf-8-sig")
    assert text.startswith("schema_version: 1\n") and "status" not in text
    keypoint_runs.update_track_yaml(
        run_dir, coverage={"track_3": {"part_001": {"placed": 1, "total": 2}}}
    )
    path = keypoint_runs.track_part_path(run_dir, 3, "part_001")
    assert path == run_dir / "track_3" / "part_001.zarr"
    path.parent.mkdir(parents=True)
    with KeypointTrackPartWriter(path, ("nose", "tail")) as writer:
        writer.add_rows(
            np.array([2, 5]),
            np.array([20, 50]),
            np.array([0, -1]),
            np.full((2, 2, 2), 0.25, np.float32),
            np.full((2, 2), 0.9, np.float32),
            np.array([[0, 1], [2, 0]], np.int8),
        )
    apath = keypoint_runs.assignment_path(run_dir, "part_001")
    assert apath == run_dir / "assignments" / "part_001.zarr"
    with KeypointAssignmentPartWriter(apath) as writer:
        writer.add(3, 0.8)
        writer.add(-1, float("nan"))
    run = keypoint_runs.KeypointTrackRun(recording_dir, "kt_01")
    assert run.meta["method"] == "temporal_decode"
    assert run.skeleton.names == ("nose", "tail") and run.track_ids == [3]
    assert run.part_ids(3) == ["part_001"] and run.part_ids(0) == []
    assert run.coverage() == {"track_3": {"part_001": {"placed": 1, "total": 2}}}
    points, scores = run.at(3, "part_001", 5)
    assert points.shape == (2, 2) and scores.tolist() == pytest.approx([0.9, 0.9])
    assert run.at(3, "part_001", 4) is None
    rows = run.rows(3, "part_001")
    assert rows["instance_row"].tolist() == [0, -1]
    assert rows["changed"].tolist() == [[0, 1], [2, 0]]
    assignments = run.assignments("part_001")
    assert assignments["subject_id"].tolist() == [3, -1]
    run.meta["skeleton"]["names"] = ["a", "b"]
    with pytest.raises(ValueError, match="cannot be read"):
        run.rows(3, "part_001")


def test_a_track_run_from_a_per_track_detect_run_is_refused(tmp_path):
    recording_dir = tmp_path / "recording_001"
    run_dir = keypoint_runs.track_run_dir(recording_dir, "kt_01")
    record = keypoint_runs.track_record(_track_job(), "4b7e")
    del record["tracking_run_id"], record["match"]
    keypoint_runs.write_track_yaml(run_dir, record)
    with pytest.raises(LegacyKeypointLayoutError, match="re-run keypoint detection"):
        keypoint_runs.KeypointTrackRun(recording_dir, "kt_01")
