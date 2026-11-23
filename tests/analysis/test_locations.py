"""The location export (``analysis/locations.py``): positions, regions and points per frame.

**A position computed against the wrong grid (#32).** A mask cell ``i`` covers ``[i, i+1)``, and
the export averaged cell indices, so every mask centre sat half a cell up and left of the
kinematics centre for the same frame; and box corners rounded to the mask grid made a box under
one cell wide cover no cell, 0% inside a region it sat in. The rules: one centre,
``geometry.mask_centre_norm``, for every caller; a box covers every cell it touches.

**A body measure no other tool can reproduce.** The export once offered to remove thin parts (a
tail) from the body with a morphological opening sized by a parameter of its own. A number made
that way is specific to this tool, and a reader cannot recompute it from the masks, so it was
taken out: what counts as the animal is decided where the mask is made. The rules: the body
*is* the mask, to the last digit, tail and all; its length and width are the standard
second-moment ellipse axes; an option asking for thin parts to be removed is refused, not
dropped.

**Rows that were never tracked.** Sampling keeps the tracked frame nearest each tick of a clock
and never interpolates, so a gap in tracking stays a gap in every table.
"""

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


# --- positions against the right grid (#32) ------------------------------------------------ #
def test_the_mask_centre_is_the_kinematics_centre(tmp_path):
    """A mask cell ``i`` covers ``[i, i+1)``, so its centre is ``i + 0.5``. The location export
    averaged the cell *indices* and so put every mask centre half a cell up and left of the
    animal (#32, finding 1): a symmetric mask over the box 0.1-0.5 read 0.2875, not 0.3, and
    disagreed with the kinematics centre for the same frame. One helper now gives both.
    """
    from detail.analysis import geometry

    mask = _mask_square(4, 20, 4, 20).astype(bool)
    recording = _one_track(tmp_path)
    (r,) = _dicts(recording.location_tables("track_01").tracks)
    kinematics = geometry.frame_geometry(mask, [(0.1, 0.1, 0.5, 0.5)], (H, W))
    assert r["mask_cx_norm"] == pytest.approx(0.3)
    assert r["mask_cy_norm"] == pytest.approx(0.3)
    assert r["mask_cx_norm"] == pytest.approx(r["bbox_cx_norm"])
    assert r["mask_cx_norm"] == pytest.approx(kinematics.cx_norm)
    assert r["mask_cx_px"] == pytest.approx(12.0)  # 0.3 * 40, not 11.5


def test_a_symmetric_animal_on_a_region_edge_reads_the_same_by_box_and_by_mask(
    tmp_path,
):
    """The user-visible half of finding 1: a region whose left edge is exactly at the animal's
    centre said the box centre was in it and the mask centre was not."""
    recording = _one_track(tmp_path)
    recording.define_roi(
        "right", [(0, [(0.3, 0.0), (1.0, 0.0), (1.0, 1.0), (0.3, 1.0)])]
    )
    (r,) = _dicts(recording.location_tables("track_01").roi_relations)
    assert r["centroid_in_roi"] == 1
    assert r["mask_centroid_in_roi"] == r["centroid_in_roi"]


def test_a_box_thinner_than_a_mask_cell_is_not_zero_percent_inside(tmp_path):
    """Box corners were rounded to the mask grid, so a box under one cell wide rounded to no
    cells at all and read ``bbox_frac_in_roi 0.0`` inside a region it sat in (#32, finding 5).
    The low corner is floored and the high one ceiled, so a box covers every cell it touches.
    """
    thin = (0.3025, 0.3025, 0.31, 0.31)  # cells 12.1-12.4: both corners round to 12
    recording = _setup(
        tmp_path,
        detections=[(0, 0, thin, _mask_square(12, 13, 12, 13))],
        object_ids=[0],
    )
    recording.define_roi(
        "left", [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])]
    )
    (r,) = _dicts(recording.location_tables("track_01").roi_relations)
    assert r["centroid_in_roi"] == 1
    assert r["bbox_frac_in_roi"] == pytest.approx(1.0)
    assert r["bbox_fully_in_roi"] == 1


# --------------------------------------------------------------------------------------------- #
# the body: shape, sampling (detail#75)
# --------------------------------------------------------------------------------------------- #
#: The columns an export had before the body was added. A reader that takes columns by position
#: must still find these where they were.
_COLUMNS_BEFORE_THE_BODY = [
    "frame_idx", "t_ns", "object_id",
    "bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm",
    "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
    "bbox_cx_norm", "bbox_cy_norm", "bbox_cx_px", "bbox_cy_px",
    "mask_cx_norm", "mask_cy_norm", "mask_cx_px", "mask_cy_px",
    "bbox_area_norm", "bbox_area_px", "mask_area_norm", "mask_area_px",
    "bbox_cx_mm", "bbox_cy_mm", "mask_cx_mm", "mask_cy_mm",
    "bbox_area_mm2", "mask_area_mm2",
]  # fmt: skip
_BODY_COLUMNS = [
    "body_cx_norm", "body_cy_norm", "body_cx_px", "body_cy_px",
    "body_area_norm", "body_area_px",
    "body_length_px", "body_width_px", "body_angle_deg",
    "body_cx_mm", "body_cy_mm", "body_area_mm2",
    "body_length_mm", "body_width_mm",
]  # fmt: skip


def _rat(tail: bool = True) -> np.ndarray:
    """A 20 x 10 cell body (rows 10-19, columns 4-23) with a 1-cell-thick tail to its right."""
    m = _mask_square(10, 20, 4, 24)
    if tail:
        m[15, 24:38] = 1
    return m


def _frames(masks, object_ids=None):
    """``_setup`` arguments for one mask per frame, all of track 0 unless told otherwise."""
    detections = [(f, 0, (0.0, 0.0, 1.0, 1.0), m) for f, m in enumerate(masks)]
    return detections, (object_ids or [0] * len(masks))


def test_the_body_columns_come_after_every_column_an_older_export_has(tmp_path):
    columns, _rows = _one_track(tmp_path).location_tables("track_01").tracks
    assert columns == _COLUMNS_BEFORE_THE_BODY + _BODY_COLUMNS


def test_the_body_is_the_whole_mask_tail_and_all(tmp_path):
    recording = _setup(tmp_path, *_frames([_rat()]))
    recording.set_pixels_per_mm(2.0)
    tables = recording.location_tables("track_01")
    row = _dicts(tables.tracks)[0]
    for what in ("cx_norm", "cy_norm", "cx_px", "cy_px", "area_norm", "area_px"):
        assert row[f"body_{what}"] == row[f"mask_{what}"], what
    for what in ("cx_mm", "cy_mm", "area_mm2"):
        assert row[f"body_{what}"] == row[f"mask_{what}"], what
    # nothing is taken out: the tail counts toward the length, as it would in any tool
    tailless = _dicts(
        _setup(tmp_path / "b", *_frames([_rat(tail=False)]))
        .location_tables("track_01")
        .tracks
    )[0]
    assert row["body_length_px"] > 1.2 * tailless["body_length_px"]
    assert "remove_thin_parts" not in tables.source
    assert tables.source["body_axis"] == "shape"


def test_length_and_width_are_the_axes_of_the_body(tmp_path):
    # a 20 x 10 px rectangle: four standard deviations of a uniform length L is 4 L / sqrt(12)
    recording = _setup(tmp_path, *_frames([_rat(tail=False)]))
    recording.set_pixels_per_mm(2.0)
    row = _dicts(recording.location_tables("track_01").tracks)[0]
    assert row["body_length_px"] == pytest.approx(4 * 20 / 12**0.5, abs=0.01)
    assert row["body_width_px"] == pytest.approx(4 * 10 / 12**0.5, abs=0.01)
    assert row["body_angle_deg"] == pytest.approx(0.0, abs=0.01)  # long axis along x
    assert row["body_length_mm"] == pytest.approx(row["body_length_px"] / 2.0, abs=0.01)
    assert row["body_width_mm"] == pytest.approx(row["body_width_px"] / 2.0, abs=0.01)


def test_an_upright_body_points_up_and_a_blank_calibration_blanks_only_mm(tmp_path):
    upright = _mask_square(4, 34, 16, 24)  # 30 tall, 8 wide
    row = _dicts(
        _setup(tmp_path, *_frames([upright])).location_tables("track_01").tracks
    )[0]
    assert row["body_angle_deg"] == pytest.approx(90.0, abs=0.01)
    assert row["body_length_px"] == pytest.approx(4 * 30 / 12**0.5, abs=0.01)
    assert row["body_length_mm"] == "" and row["body_width_mm"] == ""
    assert row["body_cx_mm"] == "" and row["body_area_mm2"] == ""


def test_the_same_animal_measures_the_same_on_a_coarser_mask_grid():
    from detail.analysis.locations import body_axes

    fine = np.zeros((80, 80), bool)
    fine[20:40, 8:48] = True  # the 20 x 10 body, drawn at twice the resolution
    coarse = np.zeros((40, 40), bool)
    coarse[10:20, 4:24] = True
    a, b = body_axes(fine, 0.5, 0.5), body_axes(coarse, 1.0, 1.0)
    assert a["length_px"] == pytest.approx(b["length_px"], rel=1e-9)
    assert a["width_px"] == pytest.approx(b["width_px"], rel=1e-9)
    # a grid that is finer one way than the other is still measured in video pixels
    wide_cells = np.zeros((40, 20), bool)
    wide_cells[10:20, 2:12] = True
    assert body_axes(wide_cells, 2.0, 1.0)["length_px"] == pytest.approx(b["length_px"])
    assert body_axes(np.zeros((4, 4), bool), 1.0, 1.0) == {
        "length_px": None, "width_px": None, "angle_deg": None
    }  # fmt: skip


def test_sampling_keeps_the_frame_nearest_each_tick():
    from detail.analysis.locations import sampled_indices

    ns = 1_000_000_000
    t = (np.arange(100) * ns / 62.5).astype(np.int64)  # 1.6 s of a 62.5 fps video
    kept = sampled_indices(t, 5.0)
    # the answer worked out the slow way: for each tick, the frame nearest it
    period = ns / 5
    nearest_tick = np.rint(t / period)
    expected = [
        int(
            np.flatnonzero(nearest_tick == k)[
                np.argmin(np.abs(t - k * period)[nearest_tick == k])
            ]
        )
        for k in np.unique(nearest_tick)
    ]
    assert kept.tolist() == expected
    assert (
        len(kept) == 9
    )  # ticks at 0, 0.2, ... 1.6 s; the last frame is 16 ms from 1.6
    assert kept[0] == 0 and np.all(np.diff(kept) > 0)
    # a rate above the video's keeps every frame, and never invents one
    assert sampled_indices(t, 1000.0).tolist() == list(range(100))
    assert sampled_indices(np.array([], dtype=np.int64), 5.0).tolist() == []


def test_sampling_leaves_a_gap_in_tracking_a_gap():
    from detail.analysis.locations import sampled_indices

    ns = 1_000_000_000
    # tracked for 1 s, lost for 3 s, tracked for 1 s, at 50 fps
    t = np.r_[np.arange(50), 200 + np.arange(50)] * (ns // 50)
    kept = sampled_indices(t, 5.0)
    seconds = t[kept] / ns
    assert not np.any(
        (seconds > 1.1) & (seconds < 3.9)
    )  # nothing made up across the gap
    assert len(kept) == 12  # ticks 0 to 1.0 s and 4.0 to 5.0 s, six each
    # two frames equally near one tick: the earlier one, every time
    assert sampled_indices(
        np.array([90, 110], dtype=np.int64) * (ns // 1000), 10.0
    ).tolist() == [0]


def test_a_sampled_export_has_the_same_frames_in_every_table(tmp_path):
    masks = [_rat()] * 12  # frames 0..11, 1000 ns apart
    recording = _setup(tmp_path, *_frames(masks))
    recording.define_roi(
        "left", [(0, [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])]
    )
    recording.define_point("p", [(0, [(0.9, 0.9)])])
    seen = []
    tables = recording.location_tables(
        "track_01", sample_fps=250_000.0
    )  # every 4000 ns
    for table in (tables.tracks, tables.roi_relations, tables.point_relations):
        seen.append([row["frame_idx"] for row in _dicts(table)])
    # ticks at 0, 4000, 8000 and 12000 ns: frame 11, at 11000, is the nearest to the last
    assert seen == [[0, 4, 8, 11]] * 3
    assert tables.source["sample_fps"] == 250_000.0
    every = recording.location_tables("track_01")
    assert [row["frame_idx"] for row in _dicts(every.tracks)] == list(range(12))
    assert every.source["sample_fps"] is None


def test_progress_counts_only_the_rows_a_sampled_export_keeps(tmp_path):
    from detail.analysis.locations import compute_location_tables
    from detail.configs.location_export import LocationExportConfig

    recording = _setup(tmp_path, *_frames([_rat()] * 12))
    calls = []
    compute_location_tables(
        recording,
        "track_01",
        on_progress=lambda done, total: calls.append((done, total)),
        options=LocationExportConfig(sample_fps=250_000.0),
    )
    assert calls == [
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
    ]  # a bar over 12 would stop at a third


def test_each_track_is_sampled_on_its_own_frames(tmp_path):
    # track 0 on even frames, track 1 on odd ones: a clock every 4000 ns keeps the nearest
    # frame of each, not the frames of whichever track is on the tick
    detections, ids = [], []
    for frame in range(12):
        detections.append((frame, 0, (0.0, 0.0, 1.0, 1.0), _rat()))
        ids.append(frame % 2)
    rows = _dicts(
        _setup(tmp_path, detections, ids)
        .location_tables("track_01", sample_fps=250_000.0)
        .tracks
    )
    by_track = {
        oid: [r["frame_idx"] for r in rows if r["object_id"] == oid] for oid in (0, 1)
    }
    assert by_track[0] == [0, 4, 8]
    # frames 3 and 5 are equally near the tick at 4000 ns, 7 and 9 the one at 8000: the earlier
    assert by_track[1] == [1, 3, 7, 11]


def test_an_option_the_export_does_not_have_is_refused_not_dropped(tmp_path):
    recording = _one_track(tmp_path)
    with pytest.raises(ValueError, match="no option sample_rate; it has sample_fps"):
        recording.location_tables("track_01", sample_rate=5)
    with pytest.raises(ValueError, match="sample_fps"):
        recording.location_tables("track_01", sample_fps=0)
    # the removed thin-part options are refused too, never silently measured with the tail
    with pytest.raises(ValueError, match="no option remove_thin_parts"):
        recording.export_locations("track_01", tmp_path / "out", remove_thin_parts=True)


def test_the_manifest_says_how_the_body_was_measured(tmp_path):
    recording = _setup(tmp_path, *_frames([_rat()] * 3))
    out = recording.export_locations("track_01", tmp_path / "out", sample_fps=5.0)
    manifest = yaml.safe_load((out / "locations.yaml").read_text(encoding="utf-8-sig"))
    assert list(manifest)[0] == "schema_version"
    assert manifest["sample_fps"] == 5.0
    assert manifest["body_axis"] == "shape"
    assert not any(key.startswith(("remove_thin", "thin_part")) for key in manifest)
    assert manifest["columns"]["tracks"][-len(_BODY_COLUMNS) :] == _BODY_COLUMNS


def test_a_window_of_the_grid_gives_the_centre_the_whole_grid_gives():
    from detail.analysis.geometry import (
        mask_centre_norm,
        mask_extent,
        window_centre_norm,
    )

    rng = np.random.default_rng(0)
    for _ in range(50):
        mask = np.zeros((108, 192), bool)
        y, x = rng.integers(0, 80), rng.integers(0, 150)
        blob = rng.random((rng.integers(3, 28), rng.integers(3, 42))) > 0.4
        mask[y : y + blob.shape[0], x : x + blob.shape[1]] = blob
        if not mask.any():
            continue
        y0, y1, x0, x1 = mask_extent(mask)
        assert mask[y0:y1, x0:x1].sum() == mask.sum()
        # equal, not approximately: an export made on the window must not differ in the last
        # digit from one made on the grid, or a re-run would not reproduce an older file
        assert window_centre_norm(
            mask[y0:y1, x0:x1], x0, y0, 192, 108
        ) == mask_centre_norm(mask)
    assert mask_extent(np.zeros((4, 4), bool)) is None
