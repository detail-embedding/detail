"""The keypoint export targets (§26 "Analysis and export") over runs the real stages wrote.

Every format is written from one `PointSource`, so the tests check each file's own layout
against the same rows: the fake detect method places point 0 on a subject's box corner
``(x0, y0)`` and point 1 on ``(x1, y1)``, scored 0.5, over five 40x40 frames; track 0 is on
every frame, track 1 on frames 1 and 3.

**A missing value averaged in as zero (#32).** SLEAP's ``instance_scores`` set a missing point's
NaN score to 0 before the mean, so a half-placed animal scored half. The rule: a mean over
points is a mean over the points placed (``sleap_instance_scores``).
"""

import csv
import importlib.util
import types

import cv2
import numpy as np
import pytest

from detail.api import export_targets, exporters
from detail.api.exporters import ExportSubject
from detail.api.keypoint_export import (
    INSTANCE_LONG_COLUMNS,
    LONG_COLUMNS,
    PartRows,
    PointSource,
    write_nwb,
)
from detail.core.registries import KEYPOINT_MODELS
from detail.core.skeleton import Skeleton
from detail.storage import recording as recording_store
from detail.storage import runs as runs_store
from detail.storage.zarr_io import TracksSummaryWriter
from tests.pipeline.test_keypoint_detect import (
    BOX_A,
    BOX_B,
    _FakeKeypointModel,
    _config,
    _run,
)
from tests.pipeline.test_keypoint_track import _track_run

DETECT_RUN, TRACK_RUN = "run_010", "run_011"
NAME = "Mouse 7, day 2"
STEM = export_targets.safe_name(NAME)
#: The SLEAP and NWB writers need the ``keypoints_export`` extra; without it those two targets
#: are offered as unavailable (``export_targets._needs_extra``), so a test that asks for all five
#: cannot pass. CI installs the extra; a checkout without it skips rather than fails.
needs_export_extra = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("h5py", "pynwb", "ndx_pose")),
    reason="the keypoints_export extra is not installed",
)

FIVE = (
    "keypoints",
    "keypoints_dlc",
    "keypoints_sleap",
    "keypoints_nwb",
    "keypoints_video",
)


@pytest.fixture
def keypoint_runs(processed_recording, monkeypatch):
    """``recording_002`` with ``kd_01`` and ``kt_01`` written by the real stages, and the two
    experiment run records (``run_010`` detect, ``run_011`` track) an export reads."""
    it = processed_recording
    sq = it.square
    detections = [
        (0, 0, BOX_A, sq(4, 20, 4, 20)),
        (1, 0, BOX_A, sq(4, 20, 4, 20)),
        (1, 1, BOX_B, sq(20, 36, 20, 36)),
        (2, 0, BOX_A, sq(4, 20, 4, 20)),
        (3, 0, BOX_A, sq(4, 20, 4, 20)),
        (3, 1, BOX_B, sq(20, 36, 20, 36)),
        (4, 0, BOX_A, sq(4, 20, 4, 20)),
    ]
    recording = it.add_recording("recording_002", detections, [0, 0, 1, 0, 0, 1, 0])
    video = recording.dir / "video" / "part_001.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (40, 40)
    )
    for _ in range(5):
        writer.write(np.zeros((40, 40, 3), np.uint8))  # black: anything drawn shows
    writer.release()
    with TracksSummaryWriter(
        recording.dir / "tracking" / "track_01" / "tracks.zarr"
    ) as w:
        w.add_track(0, "mouse", 0, 4, 5, 0, 0.9)
        w.add_track(1, "mouse", 1, 3, 2, 1, 0.9)
    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp", _FakeKeypointModel)
    _run(it, recording, _config())
    data = recording_store.read_recording(recording.dir)
    data["name"] = NAME
    data["keypoint_detect_runs"] = [
        {"id": "kd_01", "status": "complete", "model_id": "fake_kp", "output": "x"}
    ]
    recording_store.write_recording(recording.dir, data)
    _track_run(it, recording, {"detect_run_id": "kd_01", "method": None, "config": {}})
    for run_id, module_id, stage in (
        (DETECT_RUN, "keypoint_detect", "kd_01"),
        (TRACK_RUN, "keypoint_track", "kt_01"),
    ):
        runs_store.write_run(
            it.experiment.dir,
            run_id,
            {
                "id": run_id,
                "module_id": module_id,
                "recording_ids": ["recording_002"],
                "per_recording": {
                    "recording_002": {"run_id": stage, "status": "complete"}
                },
                "status": "complete",
                "params": {"method": "despike_smooth"},
            },
        )
    return it, recording


def _subject(it, run_id):
    return ExportSubject("run", it.experiment.dir, run_id)


def _export(it, run_id, target_ids, dest, **options):
    return exporters.write_export(
        _subject(it, run_id),
        [exporters.target("run", t) for t in target_ids],
        dest,
        **options,
    )


def _rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


@needs_export_extra
def test_a_keypoint_run_offers_the_five_targets_and_names_what_a_tracking_run_lacks(
    keypoint_runs,
):
    it, _recording = keypoint_runs
    for run_id in (TRACK_RUN, DETECT_RUN):
        runnable = {t.id for t in exporters.available_targets(_subject(it, run_id))}
        assert set(FIVE) <= runnable, run_id
    tracking = _subject(it, it.tracking_run_id)
    reason = exporters.unavailable_reason(
        exporters.target("run", "keypoints"), tracking
    )
    assert reason == (
        "this is a tracking run; these come from a keypoint track run, or a keypoint "
        "detect run for its raw, unassigned poses"
    )
    assert export_targets._needs_extra("no_such_module_xyz").startswith(
        "needs no_such_module_xyz: install the keypoints_export extra"
    )
    assert export_targets._needs_extra("csv") is None


def test_the_long_table_has_a_row_per_frame_track_and_point(keypoint_runs, tmp_path):
    it, _recording = keypoint_runs
    result = _export(it, TRACK_RUN, ["keypoints"], tmp_path)
    rows = _rows(result.dest / f"{STEM}_keypoints.csv")
    assert list(rows[0]) == LONG_COLUMNS
    assert len(rows) == 7 * 2  # 5 + 2 subject frames, two points each
    first = rows[0]
    assert (first["track_id"], first["frame_idx"], first["keypoint"]) == (
        "0",
        "0",
        "nose",
    )
    assert float(first["x_norm"]) == pytest.approx(0.1)
    assert float(first["x_px"]) == pytest.approx(4.0)  # 0.1 × 40
    assert float(first["score"]) == pytest.approx(0.5)
    assert first["visible"] == "true" and first["changed"] in {"0", "2"}
    assert {r["track_id"] for r in rows} == {"0", "1"}
    assert {r["frame_idx"] for r in rows if r["track_id"] == "1"} == {"1", "3"}
    coverage = _rows(result.dest / "coverage.csv")
    totals = {r["track_id"]: r for r in coverage if r["keypoint"] == ""}
    assert (totals["0"]["placed"], totals["0"]["total"]) == ("5", "5")
    assert (totals["1"]["placed"], totals["1"]["total"]) == ("2", "2")
    per_point = [
        r for r in coverage if r["keypoint"] == "nose" and r["track_id"] == "0"
    ]
    assert per_point[0]["visible"] == "5"
    assert "params.csv" in result.files and "coverage.csv" in result.files


def test_the_cutoff_marks_a_point_not_visible_and_the_blank_option_removes_it(
    keypoint_runs, tmp_path
):
    it, _recording = keypoint_runs
    marked = _export(it, TRACK_RUN, ["keypoints"], tmp_path / "a", cutoff=0.6)
    rows = _rows(marked.dest / f"{STEM}_keypoints.csv")
    assert all(r["visible"] == "false" for r in rows)
    assert all(r["x_px"] != "" for r in rows)  # a view, never a filter
    blanked = _export(
        it,
        TRACK_RUN,
        ["keypoints"],
        tmp_path / "b",
        cutoff=0.6,
        blank_below_cutoff=True,
    )
    rows = _rows(blanked.dest / f"{STEM}_keypoints.csv")
    assert all(r["x_px"] == "" and r["x_norm"] == "" for r in rows)
    assert all(float(r["score"]) == pytest.approx(0.5) for r in rows)  # the score stays
    coverage = _rows(blanked.dest / "coverage.csv")
    assert all(r["visible"] == "0" for r in coverage if r["keypoint"])


def test_source_detect_reads_the_detect_runs_own_placements(keypoint_runs, tmp_path):
    it, recording = keypoint_runs
    from_track = _export(it, TRACK_RUN, ["keypoints"], tmp_path / "t", source="detect")
    rows = _rows(from_track.dest / f"{STEM}_keypoints.csv")
    assert all(r["changed"] == "0" for r in rows)
    from_detect = _export(it, DETECT_RUN, ["keypoints"], tmp_path / "d")
    same = _rows(from_detect.dest / f"{STEM}_keypoints.csv")
    assert [r["x_px"] for r in rows] == [r["x_px"] for r in same]
    source = PointSource(recording.dir, "kt_01", source="detect")
    assert source.kind == "detect" and source.run.run_id == "kd_01"
    with pytest.raises(ValueError, match="source must be one of"):
        PointSource(recording.dir, "kt_01", source="nope")
    refused = _export(it, TRACK_RUN, ["keypoints"], tmp_path / "x", source="nope")
    assert refused.outcomes[0].status == "failed"
    assert "source must be one of" in refused.outcomes[0].error


def test_a_detect_run_exports_its_instances_unassigned(keypoint_runs, tmp_path):
    """§26 (revised 2026-10-21): a detect run's poses belong to no subject until a track run
    assigns them, so its export never invents one. A frame with two instances is two rows per
    point with an empty ``track_id`` and ``instance`` 0 and 1 — DeepLabCut's untracked
    individuals — whatever tracking says about that frame."""
    it, recording = keypoint_runs
    result = _export(
        it, DETECT_RUN, ["keypoints", "keypoints_dlc", "keypoints_video"], tmp_path
    )
    assert all(o.status == "complete" for o in result.outcomes), result.outcomes
    with open(
        result.dest / f"{STEM}_keypoints.csv", newline="", encoding="utf-8-sig"
    ) as handle:
        assert next(csv.reader(handle)) == INSTANCE_LONG_COLUMNS
    rows = _rows(result.dest / f"{STEM}_keypoints.csv")
    assert rows and all(r["track_id"] == "" for r in rows)
    noses = [r for r in rows if r["keypoint"] == "nose"]
    on = {f: [r for r in noses if r["frame_idx"] == f] for f in "01234"}
    assert [len(on[f]) for f in "01234"] == [1, 2, 1, 2, 1], "a row per instance"
    # each instance sits on its own detection's box corner: the frame's first is BOX_A's
    assert sorted((r["instance"], round(float(r["x_norm"]), 4)) for r in on["1"]) == [
        ("0", BOX_A[0]),
        ("1", BOX_B[0]),
    ]
    csv_path = result.dest / f"{STEM}_part_001_DLC.csv"
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        table = list(csv.reader(handle))
    assert table[1][1:7] == ["instance_0"] * 6 and table[1][7:] == ["instance_1"] * 6
    body = table[4:]
    assert body[0][7:] == [""] * 6 and body[1][7:10] != ["", "", ""]
    assert (result.dest / f"{STEM}_part_001_keypoints.mp4").is_file()
    coverage = _rows(result.dest / "coverage.csv")
    part = next(r for r in coverage if not r["keypoint"])
    assert part["track_id"] == "" and (part["placed"], part["total"]) == ("5", "5")
    assert {r["keypoint"]: r["visible"] for r in coverage if r["keypoint"]} == {
        "nose": "7",
        "tail": "7",
    }, "every instance of the part counted, not one per track"


def test_dlc_table_is_one_row_per_video_frame_under_the_four_header_rows(
    keypoint_runs, tmp_path
):
    it, recording = keypoint_runs
    result = _export(it, TRACK_RUN, ["keypoints_dlc"], tmp_path)
    csv_path = result.dest / f"{STEM}_part_001_DLC.csv"
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        table = list(csv.reader(handle))
    assert [row[0] for row in table[:4]] == [
        "scorer",
        "individuals",
        "bodyparts",
        "coords",
    ]
    assert table[0][1] == "DETAIL_despike_smooth_kt_01"
    assert table[1][1:7] == ["track_0"] * 6 and table[1][7:] == ["track_1"] * 6
    assert table[2][1:4] == ["nose"] * 3 and table[3][1:4] == ["x", "y", "likelihood"]
    body = table[4:]
    assert [row[0] for row in body] == ["0", "1", "2", "3", "4"]  # every video frame
    assert len(body[0]) == 1 + 2 * 2 * 3
    assert [float(v) for v in body[0][1:4]] == pytest.approx([4.0, 4.0, 0.5])
    assert body[0][7:] == [""] * 6  # track 1 is not on frame 0
    assert body[1][7:10] != ["", "", ""]
    stamps = _rows(result.dest / "timestamps.csv")
    assert [r["frame_idx"] for r in stamps] == ["0", "1", "2", "3", "4"]
    assert (
        stamps[0]["part_id"] == "part_001" and stamps[0]["recording_id"] == recording.id
    )
    h5_path = result.dest / f"{STEM}_part_001_DLC.h5"
    if importlib.util.find_spec("tables") is None:
        assert not h5_path.exists()
        return
    pd = pytest.importorskip("pandas")
    frame = pd.read_hdf(h5_path, key="df_with_missing")
    assert frame.shape == (5, 12)
    assert list(frame.columns.names) == ["scorer", "individuals", "bodyparts", "coords"]
    scorer = frame.columns[0][0]
    assert frame.loc[0, (scorer, "track_0", "nose", "x")] == pytest.approx(4.0)
    assert np.isnan(frame.loc[0, (scorer, "track_1", "nose", "x")])


def test_sleap_analysis_file_follows_sleaps_own_layout(keypoint_runs, tmp_path):
    h5py = pytest.importorskip("h5py")
    it, _recording = keypoint_runs
    result = _export(it, TRACK_RUN, ["keypoints_sleap"], tmp_path)
    with h5py.File(result.dest / f"{STEM}_part_001.analysis.h5", "r") as f:
        assert [n.decode() for n in f["node_names"][:]] == ["nose", "tail"]
        assert [n.decode() for n in f["track_names"][:]] == ["track_0", "track_1"]
        tracks = f["tracks"][:]
        assert tracks.shape == (2, 2, 2, 5)  # (n_tracks, xy, n_nodes, n_frames)
        assert tracks[0, 0, 0, 0] == pytest.approx(4.0)  # track 0, x, nose, frame 0
        assert tracks[0, 1, 1, 0] == pytest.approx(20.0)  # track 0, y, tail (0.5 × 40)
        assert (
            np.isnan(tracks[1, :, :, 0]).all() and np.isfinite(tracks[1, :, :, 1]).all()
        )
        occupancy = f["track_occupancy"][:]
        assert occupancy.shape == (5, 2)
        assert occupancy[:, 0].tolist() == [1, 1, 1, 1, 1]
        assert occupancy[:, 1].tolist() == [0, 1, 0, 1, 0]
        assert f["point_scores"].shape == (2, 2, 5)
        assert f["point_scores"][0, 0, 0] == pytest.approx(0.5)
        assert f["instance_scores"].shape == (2, 5)
        assert f["edge_inds"][:].tolist() == [[0, 1]]
        assert f["video_path"][()].decode() == "part_001.mp4"
    assert (result.dest / "timestamps.csv").is_file()


def test_nwb_file_round_trips_through_pynwb(keypoint_runs, tmp_path):
    pytest.importorskip("pynwb")
    pytest.importorskip("ndx_pose")
    from pynwb import NWBHDF5IO

    it, recording = keypoint_runs
    result = _export(it, TRACK_RUN, ["keypoints_nwb"], tmp_path)
    assert result.outcomes[0].status == "complete", result.outcomes[0].error
    path = result.dest / f"{STEM}_part_001.nwb"
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        behavior = nwb.processing["behavior"]
        pose = behavior["track_0"]
        assert set(pose.pose_estimation_series) == {"nose", "tail"}
        nose = pose.pose_estimation_series["nose"]
        assert nose.data.shape == (5, 2) and nose.unit == "pixels"
        assert nose.data[0].tolist() == pytest.approx([4.0, 4.0])
        assert nose.confidence[0] == pytest.approx(0.5)
        t_ns = PointSource(recording.dir, "kt_01").rows(0, "part_001").t_ns
        assert nose.timestamps[1] == pytest.approx(t_ns[1] / 1e9)
        assert "not a calibrated probability" in nose.confidence_definition
        assert list(pose.skeleton.nodes[:]) == ["nose", "tail"]
        assert list(pose.source_video.dimension[:]) == [40, 40]
        assert pose.source_video.external_file[0] == "part_001.mp4"
        assert pose.source_video.rate == pytest.approx(30.0)
        assert behavior["track_1"].pose_estimation_series["nose"].data.shape == (2, 2)
        assert "Skeletons" in behavior.data_interfaces


class _LaterPart:
    """The second part of a two-part recording, as `write_nwb` reads a `PointSource`: its map
    starts where part 1's ends (``add_video_part`` writes recording-wide time), and the poses'
    ``t_ns`` are on that clock."""

    kind, method, assigned, scorer = "track", "fake", True, "DETAIL_fake_run_011"
    score_meaning = "the method's point score"
    recording_id = recording_name = "recording_002"
    frame_wh = (40, 40)
    skeleton = Skeleton(names=("nose", "tail"), edges=(("nose", "tail"),))
    run = types.SimpleNamespace(
        run_id="run_011", meta={"created_at": "2026-01-01T00:00:00Z"}
    )
    part_1_frames = 5
    stamps = (np.arange(4, dtype=np.int64) + part_1_frames) * 33_333_333

    def fps(self, part_id):
        return 30.0

    def frame_count(self, part_id):
        return len(self.stamps)

    def timestamps(self, part_id):
        return self.stamps

    def series_name(self, series_id):
        return f"track_{series_id}"

    def to_px(self, points):
        return points * 40.0

    def rows_in(self, part_id):
        n = len(self.stamps)
        return {
            0: PartRows(
                frame_idx=np.arange(n),
                t_ns=self.stamps,
                points=np.full((n, 2, 2), 0.5),
                scores=np.full((n, 2), 0.9),
                visible=np.ones((n, 2), bool),
                changed=np.zeros((n, 2), np.int8),
            )
        }


def test_a_later_parts_nwb_video_starts_where_its_poses_do(tmp_path):
    """Issue #23: the video series started at 0 s while the poses beside it use the
    recording's clock, so part 2's video ran one part-length ahead of its poses."""
    pytest.importorskip("pynwb")
    pytest.importorskip("ndx_pose")
    from pynwb import NWBHDF5IO

    source = _LaterPart()
    path = write_nwb(
        source,
        "part_002",
        tmp_path / "p2.nwb",
        video_name="part_002.mp4",
        experiment_name="p",
    )
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        video = nwb.acquisition["video_part_002"]
        nose = nwb.processing["behavior"]["track_0"].pose_estimation_series["nose"]
        assert video.starting_time == pytest.approx(source.stamps[0] / 1e9)
        assert video.starting_time == pytest.approx(nose.timestamps[0])


def _frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out


def test_the_video_export_draws_the_points_on_every_frame(keypoint_runs, tmp_path):
    it, _recording = keypoint_runs
    result = _export(it, TRACK_RUN, ["keypoints_video"], tmp_path / "p")
    out = result.dest / f"{STEM}_part_001_keypoints.mp4"
    frames = _frames(out)
    assert len(frames) == 5  # every frame of the part, drawn or not
    lit = [(f.max(axis=2) > 60).sum() for f in frames]
    assert all(n > 0 for n in lit), "the points were drawn on the black footage"
    assert lit[1] > lit[0], "frame 1 carries two subjects' points, frame 0 one"
    assert len(_rows(result.dest / "keypoints.csv")) == 14
    assert "keypoints.csv" in result.files and "coverage.csv" in result.files
    overlaid = _export(it, TRACK_RUN, ["keypoints_video"], tmp_path / "o", overlay=True)
    with_masks = _frames(overlaid.dest / f"{STEM}_part_001_keypoints.mp4")
    assert (with_masks[0].max(axis=2) > 60).sum() > lit[0], "the mask lit more pixels"
    hidden = _export(
        it, TRACK_RUN, ["keypoints_video"], tmp_path / "h", cutoff=0.9
    )  # every point under the cut-off: the footage alone
    bare = _frames(hidden.dest / f"{STEM}_part_001_keypoints.mp4")
    assert all((f.max(axis=2) > 60).sum() == 0 for f in bare)


def test_every_keypoint_format_in_one_export_reads_the_same_rows(
    keypoint_runs, tmp_path
):
    """One `_keypoint_sources` per target with the same options: the CSV, the DLC table and
    the video's own table carry the same point."""
    it, _recording = keypoint_runs
    result = _export(
        it,
        TRACK_RUN,
        ["keypoints", "keypoints_dlc", "keypoints_video"],
        tmp_path,
        cutoff=0.6,
    )
    assert all(o.status == "complete" for o in result.outcomes), result.outcomes
    long_rows = _rows(result.dest / f"{STEM}_keypoints.csv")
    video_rows = _rows(result.dest / "keypoints.csv")
    assert [r["x_px"] for r in long_rows] == [r["x_px"] for r in video_rows]
    assert all(r["visible"] == "false" for r in video_rows)


def test_loader_round_trip_with_movement(keypoint_runs, tmp_path):
    """The spec's guard: each exported file opened by a tool that reads the format."""
    pytest.importorskip("movement")
    from movement.io import load_poses

    it, _recording = keypoint_runs
    result = _export(it, TRACK_RUN, ["keypoints_dlc", "keypoints_sleap"], tmp_path)
    dlc = load_poses.from_dlc_file(result.dest / f"{STEM}_part_001_DLC.csv")
    assert dlc.sizes["time"] == 5 and list(dlc.keypoints.values) == ["nose", "tail"]
    sleap = load_poses.from_sleap_file(result.dest / f"{STEM}_part_001.analysis.h5")
    assert sleap.sizes["time"] == 5 and list(sleap.keypoints.values) == ["nose", "tail"]


def test_a_sleap_instance_score_averages_only_the_points_placed():
    """A missing point is NaN, and it was set to 0 before the mean, so an animal with two of
    four points placed at 0.9 scored 0.45 in ``instance_scores`` (#32, finding 7). SLEAP's own
    mean instance score skips a missing node; so does this one. An instance with no point
    placed stays NaN, not 0."""
    from detail.api.keypoint_export import sleap_instance_scores

    nan = np.nan
    likelihood = np.array(
        [
            [[0.9, 0.9, nan, nan], [0.2, 0.4, 0.6, 0.8]],
            [[nan, nan, nan, nan], [0.5, nan, nan, nan]],
        ]
    )  # (frames, tracks, keypoints)
    got = sleap_instance_scores(likelihood)
    assert got.shape == (2, 2)
    assert got[0, 0] == pytest.approx(0.9)
    assert got[0, 1] == pytest.approx(0.5)
    assert np.isnan(got[1, 0])
    assert got[1, 1] == pytest.approx(0.5)
    assert sleap_instance_scores(np.zeros((0, 2, 4))).shape == (0, 2)
