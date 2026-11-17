import json

import numpy as np
import pytest
import yaml
from pycocotools import mask as mask_utils

from detail.api.experiment import Experiment
from detail.storage import recording as recording_store
from detail.storage.zarr_io import AssignmentPartWriter, DetectionPartWriter

H = W = 40  # mask resolution + video resolution (px == norm * 40 for simplicity)


def _rle(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode(
        "ascii"
    )


def _mask_square(r0, r1, c0, c1) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    m[r0:r1, c0:c1] = 1
    return m


def _setup(tmp_path, detections, object_ids):
    """Experiment+recording with a 1-part video (40x40) + a complete det_01/track_01.

    ``detections`` = list of (frame_idx, detection_idx, bbox, mask); ``object_ids``
    is parallel (the assignment for each detection row)."""
    recording = Experiment.create("p", tmp_path).create_recording("recording_001")
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
    data["tracking_runs"] = [
        {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
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
        for fr, didx, bbox, mask in detections:
            w.add_row(fr, fr * 1000, didx, "mouse", 0.9, bbox, _rle(mask))

    trk_dir = sdir / "tracking" / "track_01"
    (trk_dir / "assignments").mkdir(parents=True)
    (trk_dir / "track.yaml").write_text(
        yaml.dump({"detection_run_id": "det_01"}), encoding="utf-8"
    )
    with AssignmentPartWriter(trk_dir / "assignments" / "part_001.zarr") as w:
        for oid in object_ids:
            w.add_object_id(oid)
    return recording


def _dicts(table):
    columns, rows = table
    return [dict(zip(columns, row)) for row in rows]


def _one_track(tmp_path):
    # one detection: bbox (0.1,0.1,0.5,0.5), mask filling that box (cols/rows 4..20)
    return _setup(
        tmp_path,
        detections=[(0, 0, (0.1, 0.1, 0.5, 0.5), _mask_square(4, 20, 4, 20))],
        object_ids=[0],
    )


def test_on_progress_climbs_per_row(tmp_path):
    # The compute reports row progress so a long single recording shows a live bar.
    from detail.analysis.locations import compute_location_tables

    recording = _setup(
        tmp_path,
        detections=[
            (f, 0, (0.1, 0.1, 0.3, 0.3), _mask_square(4, 12, 4, 12)) for f in range(3)
        ],
        object_ids=[0, 0, 0],
    )
    calls = []
    compute_location_tables(
        recording,
        "track_01",
        on_progress=lambda done, total: calls.append((done, total)),
    )
    assert calls  # called as rows are consumed
    assert [d for d, _ in calls] == [1, 2, 3]  # monotonic, one per track-frame
    assert all(total == 3 for _, total in calls)  # denominator = assigned-row count


def test_on_progress_total_excludes_unassigned(tmp_path):
    # The denominator is the assigned-row count, so the bar reaches 100% even when an
    # arena filter leaves some detections unlinked (object_id == -1). With an
    # all-detections denominator the bar would stall at 2/3 and jump at recording end.
    from detail.analysis.locations import compute_location_tables

    recording = _setup(
        tmp_path,
        detections=[
            (f, 0, (0.1, 0.1, 0.3, 0.3), _mask_square(4, 12, 4, 12)) for f in range(3)
        ],
        object_ids=[0, -1, 0],  # the middle detection is unassigned
    )
    calls = []
    compute_location_tables(
        recording,
        "track_01",
        on_progress=lambda done, total: calls.append((done, total)),
    )
    assert calls
    assert calls[-1] == (2, 2)  # reaches 100% (2 assigned rows), not stuck at 2/3
    assert all(total == 2 for _, total in calls)  # denominator excludes the unassigned


def test_tracks_table_geometry(tmp_path):
    recording = _one_track(tmp_path)
    tables = recording.location_tables("track_01")
    rows = _dicts(tables.tracks)
    assert len(rows) == 1
    r = rows[0]
    assert r["frame_idx"] == 0 and r["object_id"] == 0 and r["t_ns"] == 0
    assert r["bbox_cx_norm"] == pytest.approx(0.3) and r[
        "bbox_cy_norm"
    ] == pytest.approx(0.3)
    assert r["bbox_area_norm"] == pytest.approx(0.16)
    assert r["bbox_cx_px"] == pytest.approx(12.0)  # 0.3 * 40


def test_roi_relations_inside_outside(tmp_path):
    recording = _one_track(tmp_path)
    recording.define_roi(
        "left", [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])]
    )
    recording.define_roi(
        "right", [(0, [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)])]
    )
    rows = {
        r["roi_name"]: r
        for r in _dicts(recording.location_tables("track_01").roi_relations)
    }
    assert set(rows) == {"left", "right"}
    assert rows["left"]["centroid_in_roi"] == 1
    assert rows["left"]["bbox_fully_in_roi"] == 1
    assert rows["left"]["bbox_frac_in_roi"] == pytest.approx(1.0)
    assert rows["left"]["signed_dist_norm"] > 0  # inside → positive
    assert rows["right"]["centroid_in_roi"] == 0
    assert rows["right"]["bbox_frac_in_roi"] == pytest.approx(0.0)
    assert rows["right"]["signed_dist_norm"] < 0  # outside → negative


def test_point_relations(tmp_path):
    recording = _one_track(tmp_path)
    recording.define_point("inside", [(0, [(0.3, 0.3)])])
    recording.define_point("outside", [(0, [(0.9, 0.9)])])
    rows = {
        r["point_name"]: r
        for r in _dicts(recording.location_tables("track_01").point_relations)
    }
    assert rows["inside"]["point_in_bbox"] == 1 and rows["inside"]["point_in_mask"] == 1
    assert rows["inside"]["centroid_dist_norm"] == pytest.approx(0.0, abs=0.01)
    assert rows["outside"]["point_in_bbox"] == 0


# --- real-world (_mm) columns via distance calibration (§10) ---------------- #
def test_mm_columns_blank_without_calibration(tmp_path):
    recording = _one_track(tmp_path)
    recording.define_roi("l", [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])])
    recording.define_point("p", [(0, [(0.9, 0.9)])])
    tables = recording.location_tables("track_01")
    track = _dicts(tables.tracks)[0]
    assert track["mask_cx_mm"] == "" and track["bbox_area_mm2"] == ""
    assert _dicts(tables.roi_relations)[0]["signed_dist_mm"] == ""
    assert _dicts(tables.point_relations)[0]["centroid_dist_mm"] == ""
    assert tables.source["pixels_per_mm"] is None


def test_mm_columns_with_calibration(tmp_path):
    recording = _one_track(tmp_path)
    recording.define_roi("l", [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])])
    recording.define_point("p", [(0, [(0.9, 0.9)])])
    recording.set_pixels_per_mm(2.0)  # at 40×40 → 2 px per mm (1 px = 0.5 mm)

    tables = recording.location_tables("track_01")
    assert tables.source["pixels_per_mm"] == 2.0
    track = _dicts(tables.tracks)[0]
    assert track["mask_cx_mm"] == pytest.approx(track["mask_cx_px"] / 2.0, abs=0.01)
    assert track["bbox_area_mm2"] == pytest.approx(track["bbox_area_px"] / 4.0, abs=0.5)
    roi = _dicts(tables.roi_relations)[0]
    assert roi["signed_dist_mm"] == pytest.approx(roi["signed_dist_px"] / 2.0, abs=0.01)
    point = _dicts(tables.point_relations)[0]
    assert point["centroid_dist_mm"] == pytest.approx(
        point["centroid_dist_px"] / 2.0, abs=0.01
    )


def test_no_rois_or_points_just_tracks(tmp_path):
    recording = _one_track(tmp_path)
    tables = recording.location_tables("track_01")
    assert len(_dicts(tables.tracks)) == 1
    assert tables.roi_relations[1] == [] and tables.point_relations[1] == []


def test_inactive_roi_yields_no_row(tmp_path):
    recording = _one_track(tmp_path)
    # active window 10..20 excludes frame 0 → no relation row at frame 0
    recording.define_roi(
        "later",
        [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])],
        active=(10, 20),
    )
    assert recording.location_tables("track_01").roi_relations[1] == []


def test_unassigned_rows_skipped_and_or_merge(tmp_path):
    # two detections at frame 0: one assigned to track 0, one unassigned (-1).
    # The track-0 group is a single detection here; the -1 row must be dropped.
    recording = _setup(
        tmp_path,
        detections=[
            (0, 0, (0.1, 0.1, 0.3, 0.3), _mask_square(4, 12, 4, 12)),
            (0, 1, (0.6, 0.6, 0.9, 0.9), _mask_square(24, 36, 24, 36)),
        ],
        object_ids=[0, -1],
    )
    rows = _dicts(recording.location_tables("track_01").tracks)
    assert len(rows) == 1 and rows[0]["object_id"] == 0
    assert rows[0]["bbox_x2_norm"] == pytest.approx(0.3)  # only the track-0 box


def test_two_subjects_same_frame_get_own_roi_rows(tmp_path):
    # Two assigned tracks in one frame: the ROI geometry is resolved once for the
    # frame and reused, but each track must still get its own relation row.
    recording = _setup(
        tmp_path,
        detections=[
            (0, 0, (0.1, 0.1, 0.3, 0.3), _mask_square(4, 12, 4, 12)),  # left half
            (0, 1, (0.6, 0.6, 0.9, 0.9), _mask_square(24, 36, 24, 36)),  # right half
        ],
        object_ids=[0, 1],
    )
    recording.define_roi(
        "left", [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])]
    )
    rows = {
        r["object_id"]: r
        for r in _dicts(recording.location_tables("track_01").roi_relations)
    }
    assert set(rows) == {0, 1}
    assert rows[0]["centroid_in_roi"] == 1  # left subject inside the left ROI
    assert rows[1]["centroid_in_roi"] == 0  # right subject outside it


def test_export_writes_csvs_and_manifest(tmp_path):
    recording = _one_track(tmp_path)
    recording.define_roi(
        "left", [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])]
    )
    recording.define_point("p", [(0, [(0.3, 0.3)])])
    out = recording.export_locations("track_01", tmp_path / "loc_out")
    for name in (
        "tracks.csv",
        "roi_relations.csv",
        "point_relations.csv",
        "locations.yaml",
    ):
        assert (out / name).is_file()
    manifest_text = (out / "locations.yaml").read_text(encoding="utf-8-sig")
    assert manifest_text.splitlines()[0].startswith("schema_version:")
    manifest = yaml.safe_load(manifest_text)
    assert manifest["tracking_run_id"] == "track_01"
    assert manifest["detection_run_id"] == "det_01"


def test_a_second_export_into_one_folder_keeps_the_first(tmp_path):
    """The four names are fixed, so writing them straight into ``--out`` replaced one
    recording's files with the next one's (#29). Each export makes its own folder."""
    recording = _one_track(tmp_path)
    first = recording.export_locations("track_01", tmp_path / "loc")
    (first / "tracks.csv").write_text("the first export", encoding="utf-8")
    second = recording.export_locations("track_01", tmp_path / "loc")
    assert first != second and first.parent == second.parent == tmp_path / "loc"
    assert (first / "tracks.csv").read_text(encoding="utf-8-sig") == "the first export"
    assert (second / "tracks.csv").is_file()
    assert "recording_001" in first.name and "track_01" in first.name


def test_export_rejects_incomplete_tracking(tmp_path):
    recording = _one_track(tmp_path)
    data = recording_store.read_recording(recording.dir)
    data["tracking_runs"][0]["status"] = "running"
    recording_store.write_recording(recording.dir, data)
    with pytest.raises(ValueError, match="not complete"):
        recording.export_locations("track_01", tmp_path / "x")
