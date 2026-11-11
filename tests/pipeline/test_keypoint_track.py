"""``keypoint_track`` (§26 *Assignment*, "Track methods"): the two methods on rows a test can
see, and the stage end to end — a detect run in the instances layout the real detect stage
wrote, a tracking run on the same detection run, the assignment, and each subject's series.

The fixture is ``tracked`` from the detect tests: five 40x40 frames, nine detections, track 0
on every frame (two of its detections overlap on frame 2), track 1 on frames 1 and 3, and one
detection on frame 4 no track holds. Two detect runs are written over it:

- ``kd_01`` — per-detection, one instance per detection, two candidates per point;
- ``kd_02`` — whole-frame, the poses ``POSE_CONFIDENCES`` lists at or above 0.25, every point
  at (0.25, 0.25), which is inside track 0's mask and outside track 1's.
"""

import numpy as np
import pytest

from detail.configs.keypoint_track import (
    DespikeSmoothConfig,
    KeypointTrackConfig,
    TemporalDecodeConfig,
)
from detail.core import progress as progress_mod
from detail.core.registries import KEYPOINT_TRACK_METHODS
from detail.core.skeleton import Skeleton
from detail.pipeline import keypoint_track
from detail.pipeline import pose_match
from detail.storage import recording as recording_store
from detail.storage.keypoint_runs import (
    KeypointDetectRun,
    KeypointTrackRun,
    assignment_path,
    detect_run_dir,
    read_detect_yaml,
    read_track_yaml,
    track_part_path,
    update_detect_yaml,
)
from detail.storage.keypoint_zarr import (
    KeypointAssignmentPartReader,
    KeypointTrackPartReader,
    LegacyKeypointLayoutError,
)
from tests.pipeline.test_keypoint_detect import SKELETON
from tests.pipeline.test_keypoint_detect import _config as _detect_config
from tests.pipeline.test_keypoint_detect import _run as _detect_run
from tests.pipeline.test_keypoint_detect import tracked  # noqa: F401 — the fixture

PAIRED = Skeleton(
    names=("nose", "left_ear", "right_ear"),
    edges=(("nose", "left_ear"), ("nose", "right_ear")),
    pairs=(("left_ear", "right_ear"),),
)
WH = (400, 400)
FPS = 30.0


# --------------------------------------------------------------------------- #
# The methods, on rows a test can see
# --------------------------------------------------------------------------- #
def _detect_rows(n: int, mirrored: bool = True) -> dict:
    """One subject's series of ``n`` rows at 30 fps: a steady nose, and an ear pair whose
    candidate 0 is the detect stage's choice and candidate 1 a decoy far away."""
    t_ns = (np.arange(n) * 1e9 / FPS).astype(np.int64)
    base = np.array(
        [[0.5, 0.5], [0.4, 0.4], [0.6, 0.4]], np.float32
    )  # nose, L ear, R ear
    points = np.tile(base, (n, 1, 1))
    cands = np.zeros((n, 3, 2, 2), np.float32)
    cands[:, :, 0] = points
    cands[:, :, 1] = points + np.array([0.3, 0.3], np.float32)  # the decoy
    scores = np.zeros((n, 3, 2), np.float32)
    scores[:, :, 0] = 0.8
    scores[:, :, 1] = 0.6
    rows = {
        "frame_idx": np.arange(n, dtype=np.int32),
        "t_ns": t_ns,
        "instance_row": np.arange(n, dtype=np.int32),
        "points": points.copy(),
        "point_scores": scores[:, :, 0].copy(),
        "candidates": cands,
        "candidate_scores": scores,
    }
    if mirrored:
        rows["extras/mirrored"] = np.full((n, 3), 0.2, np.float32)
    return rows


def _unassign(rows: dict, i: int) -> None:
    """Row ``i`` as the stage builds a subject-frame no instance was assigned to."""
    rows["instance_row"][i] = -1
    for name in ("points", "point_scores", "candidates", "candidate_scores"):
        rows[name][i] = np.nan
    if "extras/mirrored" in rows:
        rows["extras/mirrored"][i] = np.nan


def _track(method_id, rows, config):
    return KEYPOINT_TRACK_METHODS[method_id]().track(
        rows,
        PAIRED,
        config,
        rows_per_second=FPS,
        frame_wh=WH,
        body_scale_px=np.full(len(rows["frame_idx"]), 100.0),
    )


def test_despike_smooth_leaves_a_steady_track_and_removes_a_spike():
    rows = _detect_rows(40)
    rows["points"][20, 0] += (0.25, 0.25)  # one-frame jump of the nose
    out = _track("despike_smooth", rows, DespikeSmoothConfig())
    assert out.points.shape == (40, 3, 2) and out.changed.dtype == np.int8
    assert out.changed[20, 0] == 2  # moved by a filter
    np.testing.assert_allclose(out.points[20, 0], (0.5, 0.5), atol=0.01)
    assert (out.changed[:, 1:] == 0).all()  # a steady point is left as detected
    np.testing.assert_allclose(out.points[:, 1], rows["points"][:, 1], atol=1e-5)
    off = _track(
        "despike_smooth", rows, DespikeSmoothConfig(despike=False, smooth=False)
    )
    np.testing.assert_array_equal(off.points, rows["points"])
    assert not off.changed.any()


def test_temporal_decode_keeps_the_consistent_candidate_and_marks_another_choice():
    rows = _detect_rows(30)
    # the detect stage's own point on row 15 was the decoy; the path corrects it
    rows["points"][15] = rows["candidates"][15, :, 1]
    rows["candidate_scores"][15, :, 1] = 0.95
    out = _track(
        "temporal_decode", rows, TemporalDecodeConfig(despike=False, smooth=False)
    )
    np.testing.assert_allclose(out.points[15], rows["candidates"][15, :, 0], atol=1e-6)
    assert (out.changed[15] == 1).all()  # another candidate than the stored point
    assert not out.changed[:15].any() and not out.changed[16:].any()
    assert np.isfinite(out.point_scores).all()


def test_temporal_decode_goes_missing_below_the_cut_off_and_never_fills_it():
    rows = _detect_rows(10)
    rows["candidate_scores"][4] = 0.1  # every candidate weak on one row
    out = _track(
        "temporal_decode",
        rows,
        TemporalDecodeConfig(decode_cutoff=0.5, despike=False, smooth=False),
    )
    assert np.isnan(out.points[4]).all() and np.isnan(out.point_scores[4]).all()
    assert np.isfinite(out.points[[3, 5]]).all()


@pytest.mark.parametrize("method_id", ["temporal_decode", "despike_smooth"])
def test_a_row_with_no_instance_stays_a_gap_in_both_methods(method_id):
    """§26: a subject-frame no instance was assigned to is NaN; each method takes it as
    missing — never filled from its neighbours — and the rows around it are unharmed."""
    rows = _detect_rows(12)
    for i in (5, 6):
        _unassign(rows, i)
    config = KEYPOINT_TRACK_METHODS[method_id].config_model()  # filters on
    out = _track(method_id, rows, config)
    assert np.isnan(out.points[[5, 6]]).all()
    assert np.isnan(out.point_scores[[5, 6]]).all()
    assert (out.changed[[5, 6]] == 0).all()
    assert np.isfinite(out.points[[4, 7]]).all()
    np.testing.assert_allclose(out.points[[4, 7]], rows["points"][[4, 7]], atol=1e-5)


def test_a_persistent_mirror_signal_swaps_the_pair_and_a_flicker_does_not():
    rows = _detect_rows(40)
    rows["extras/mirrored"][
        20:, 1:
    ] = 1.5  # from row 20 the mirrored examples match better
    out = _track(
        "temporal_decode", rows, TemporalDecodeConfig(despike=False, smooth=False)
    )
    np.testing.assert_allclose(out.points[:20, 1], rows["points"][:20, 1], atol=1e-6)
    np.testing.assert_allclose(out.points[20:, 1], rows["points"][20:, 2], atol=1e-6)
    np.testing.assert_allclose(out.points[20:, 2], rows["points"][20:, 1], atol=1e-6)
    assert (out.changed[20:, 1:] == 1).all() and (out.changed[:20] == 0).all()
    flicker = _detect_rows(40)
    flicker["extras/mirrored"][20, 1:] = 1.3  # one row: not worth two switches
    out = _track(
        "temporal_decode", flicker, TemporalDecodeConfig(despike=False, smooth=False)
    )
    np.testing.assert_allclose(out.points, flicker["points"], atol=1e-6)
    without = _detect_rows(10, mirrored=False)  # a method with no mirrored extras
    out = _track(
        "temporal_decode", without, TemporalDecodeConfig(despike=False, smooth=False)
    )
    assert out.points.shape == (10, 3, 2)


def test_subject_series_gathers_the_assigned_instance_and_fills_a_gap():
    """The series builder: columns gathered by ``instance_row``, NaN (−1 for an integer
    extra) where none was assigned, and the body scale per row from the subject."""
    instances = {
        "frame_idx": np.array([0, 2, 2], np.int32),
        "points": np.arange(12, dtype=np.float32).reshape(3, 2, 2),
        "point_scores": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], np.float32),
        "extras/ref_index": np.array([[[1], [2]], [[3], [4]], [[5], [6]]], np.int16),
    }
    entries = [(0, 0, 0, 10.0), (1, 33, -1, 11.0), (2, 66, 2, 12.0)]
    rows, scale = keypoint_track.subject_series(instances, entries)
    assert rows["instance_row"].tolist() == [0, -1, 2]
    assert rows["frame_idx"].dtype == np.int32 and rows["t_ns"].dtype == np.int64
    np.testing.assert_array_equal(rows["points"][2], instances["points"][2])
    assert np.isnan(rows["points"][1]).all() and np.isnan(rows["point_scores"][1]).all()
    assert rows["extras/ref_index"].dtype == np.int16
    assert (rows["extras/ref_index"][1] == -1).all()
    assert rows["extras/ref_index"][2].ravel().tolist() == [5, 6]
    assert scale.tolist() == [10.0, 11.0, 12.0]


# --------------------------------------------------------------------------- #
# The stage over a real detect run and a real tracking run
# --------------------------------------------------------------------------- #
@pytest.fixture
def detect_runs(tracked):  # noqa: F811 — the imported fixture
    """``kd_01`` (per-detection, two candidates) and ``kd_02`` (whole-frame, one), both
    complete in recording.yaml, over ``tracked``'s detection run and its one tracking run.
    """
    it, recording = tracked
    _detect_run(it, recording, _detect_config())
    _detect_run(it, recording, _detect_config(model_id="fake_pose"), run_id="kd_02")
    data = recording_store.read_recording(recording.dir)
    data["keypoint_detect_runs"] = [
        {"id": "kd_01", "status": "complete", "model_id": "fake_kp", "output": "x"},
        {"id": "kd_02", "status": "complete", "model_id": "fake_pose", "output": "y"},
    ]
    recording_store.write_recording(recording.dir, data)
    return it, recording


def _track_run(it, recording, config, *, run_id="kt_01", stop_event=None):
    module = keypoint_track.KeypointTrackModule()
    data = recording_store.read_recording(recording.dir)
    config = module.validate(recording.dir, data, config)
    job = module.build_job(recording.id, run_id, config)
    run_dir = module.run_dir_for(it.experiment.dir, job)
    run_dir.mkdir(parents=True, exist_ok=True)
    events: list = []
    keypoint_track.run(job, run_dir, events.append, stop_event)
    return module, config, run_dir, events


def _config(detect_run_id="kd_01", **overrides) -> dict:
    config = {"detect_run_id": detect_run_id, "method": None, "config": {}}
    config.update(overrides)
    return config


def _add_tracking_run(recording, run_id, detection_run_id, status="complete"):
    data = recording_store.read_recording(recording.dir)
    data["tracking_runs"].append(
        {"id": run_id, "status": status, "detection_run_id": detection_run_id}
    )
    recording_store.write_recording(recording.dir, data)


def _drop_tracking_runs(recording):
    data = recording_store.read_recording(recording.dir)
    data["tracking_runs"] = []
    recording_store.write_recording(recording.dir, data)


def _validate(recording, config):
    data = recording_store.read_recording(recording.dir)
    return keypoint_track.KeypointTrackModule().validate(recording.dir, data, config)


def test_the_assignment_is_written_parallel_to_the_detect_rows(detect_runs):
    """Every detect instance gets a row in ``assignments/``; on frame 2 track 0's two
    overlapping detections are two instances and only one is given to it; the detection no
    track holds (frame 4) is unassigned; each track row names a detect row on its own frame
    that the assignment gave to that track."""
    it, recording = detect_runs
    _m, config, run_dir, _e = _track_run(it, recording, _config())
    detect = KeypointDetectRun(recording.dir, "kd_01").rows("part_001")
    reader = KeypointAssignmentPartReader(assignment_path(run_dir, "part_001"))
    assert reader._root.attrs["schema_version"] == 1
    for name, dtype in (("subject_id", "int32"), ("inside", "float32")):
        assert str(reader.column(name).dtype) == dtype
        assert tuple(reader.column(name).chunks) == (10000,)
    assigned = KeypointTrackRun(recording.dir, "kt_01").assignments("part_001")
    subject, inside = assigned["subject_id"], assigned["inside"]
    assert len(subject) == len(detect["frame_idx"]) == 9
    on_frame_2 = subject[detect["frame_idx"] == 2]
    assert sorted(on_frame_2.tolist()) == [-1, 0]  # two instances, one subject
    assert subject[8] == -1  # frame 4's untracked detection
    assert np.isnan(inside[subject < 0]).all()
    assert (inside[subject >= 0] >= 0.5).all()

    run = KeypointTrackRun(recording.dir, "kt_01")
    assert run.track_ids == [0, 1]
    for track_id, frames in ((0, [0, 1, 2, 3, 4]), (1, [1, 3])):
        rows = run.rows(track_id, "part_001")
        assert rows["frame_idx"].tolist() == frames
        inst = rows["instance_row"]
        assert (inst >= 0).all()  # every tracked frame here has its own detection
        assert detect["frame_idx"][inst].tolist() == frames
        assert (subject[inst] == track_id).all()
    meta = read_track_yaml(run_dir)
    n_assigned = int((subject >= 0).sum())
    assert meta["assignment"] == {
        "part_001": {
            "instances": 9,
            "assigned": n_assigned,
            "unassigned": 9 - n_assigned,
        }
    }
    assert meta["coverage"] == {
        "track_0": {"part_001": {"placed": 5, "total": 5}},
        "track_1": {"part_001": {"placed": 2, "total": 2}},
    }
    assert meta["tracking_run_id"] == "track_01" and "status" not in meta
    assert meta["match"] == {"min_inside": 0.5, "mask_grow": 0.05}
    assert meta["timing"]["frames"] == 7  # subject-frames


def test_the_track_part_has_the_instance_row_schema(detect_runs):
    it, recording = detect_runs
    _m, _c, run_dir, _e = _track_run(it, recording, _config())
    reader = KeypointTrackPartReader(track_part_path(run_dir, 0, "part_001"))
    assert reader._root.attrs["schema_version"] == 1
    assert tuple(reader._root.attrs["skeleton_names"]) == SKELETON.names
    expected = {
        "frame_idx": ("int32", (10000,)),
        "t_ns": ("int64", (10000,)),
        "instance_row": ("int32", (10000,)),
        "points": ("float32", (10000, 2, 2)),
        "point_scores": ("float32", (10000, 2)),
        "changed": ("int8", (10000, 2)),
    }
    assert set(reader.column_names) == set(expected)
    for name, (dtype, chunks) in expected.items():
        assert str(reader.column(name).dtype) == dtype, name
        assert tuple(reader.column(name).chunks) == chunks, name


def test_a_subject_frame_with_no_instance_is_a_nan_row(detect_runs):
    """``kd_02``: no pose on frames 0 and 3 (the one on 3 is under ``min_pose_confidence``),
    so track 0 has gaps there and track 1 — whose mask no pose falls in — is all gaps. Two
    poses on one frame both inside track 0 go to the more confident one."""
    it, recording = detect_runs
    _m, config, run_dir, _e = _track_run(it, recording, _config("kd_02"))
    assert config["method"] == "despike_smooth"  # the fake whole-frame method's default
    detect = KeypointDetectRun(recording.dir, "kd_02").rows("part_001")
    run = KeypointTrackRun(recording.dir, "kt_01")
    rows0 = run.rows(0, "part_001")
    assert rows0["frame_idx"].tolist() == [0, 1, 2, 3, 4]
    assert rows0["instance_row"].tolist() == [-1, 0, 1, -1, 3]
    np.testing.assert_allclose(detect["confidence"][[0, 1, 3]], [0.9, 0.8, 0.6])
    gap = rows0["instance_row"] < 0
    assert np.isnan(rows0["points"][gap]).all()
    assert np.isnan(rows0["point_scores"][gap]).all()
    assert (rows0["changed"][gap] == 0).all()
    assert np.isfinite(rows0["points"][~gap]).all()
    rows1 = run.rows(1, "part_001")
    assert rows1["frame_idx"].tolist() == [1, 3]
    assert rows1["instance_row"].tolist() == [-1, -1]
    assert np.isnan(rows1["points"]).all()
    meta = read_track_yaml(run_dir)
    assert meta["assignment"]["part_001"] == {
        "instances": 5,
        "assigned": 3,
        "unassigned": 2,
    }
    assert meta["coverage"]["track_0"]["part_001"] == {"placed": 3, "total": 5}
    assert meta["coverage"]["track_1"]["part_001"] == {"placed": 0, "total": 2}


def test_the_tracking_run_is_resolved_and_written_back(detect_runs):
    """Tenet 9: with none named, the one complete tracking run on the detect run's detection
    run is resolved and recorded — in the config, the job, the run record and the yaml.
    """
    it, recording = detect_runs
    module, config, run_dir, _e = _track_run(it, recording, _config())
    assert config["tracking_run_id"] == "track_01"
    assert config["match"] == {"min_inside": 0.5, "mask_grow": 0.05}
    record = module.run_record("kt_01", config, "complete")
    assert set(module.identity_fields) <= set(record)
    assert record["tracking_run_id"] == "track_01"
    assert record["match"] == {"min_inside": 0.5, "mask_grow": 0.05}
    # the tracking run reaches identity as its settings, never as its per-recording id
    assert record["tracking_identity"]["module"] == "tracking"
    assert record["config"] == config["config"]
    assert read_track_yaml(run_dir)["tracking_run_id"] == "track_01"


def test_a_second_tracking_run_on_the_detection_run_is_refused_naming_both(
    detect_runs,
):
    _it, recording = detect_runs
    _add_tracking_run(recording, "track_02", "det_01")
    _add_tracking_run(recording, "track_03", "det_01", status="running")  # not a choice
    with pytest.raises(ValueError, match="track_01, track_02; choose one"):
        _validate(recording, _config())
    out = _validate(recording, _config(tracking_run_id="track_02"))
    assert out["tracking_run_id"] == "track_02"  # naming one settles it


@pytest.mark.parametrize(
    "setup, config, fragment",
    [
        (
            lambda rec: _add_tracking_run(rec, "track_09", "det_02"),
            _config(tracking_run_id="track_09"),
            "is on detection run 'det_02', but keypoint detect run 'kd_01' is on 'det_01'",
        ),
        (
            None,
            _config(tracking_run_id="track_99"),
            "tracking run 'track_99' not found",
        ),
        (
            lambda rec: _add_tracking_run(rec, "track_05", "det_01", status="failed"),
            _config(tracking_run_id="track_05"),
            "is 'failed', not complete",
        ),
        (
            _drop_tracking_runs,
            _config(),
            "no complete tracking run on detection run 'det_01'",
        ),
    ],
)
def test_validate_refuses_a_tracking_run_that_does_not_fit(
    detect_runs, setup, config, fragment
):
    _it, recording = detect_runs
    if setup is not None:
        setup(recording)
    with pytest.raises(ValueError, match=fragment):
        _validate(recording, config)


def test_a_detect_run_in_the_per_track_layout_is_refused(detect_runs):
    """A detect run from before 2026-10-21 is refused with the re-run message, never read
    through a second path."""
    _it, recording = detect_runs
    (detect_run_dir(recording.dir, "kd_01") / "track_0").mkdir()
    with pytest.raises(LegacyKeypointLayoutError, match="re-run keypoint detection"):
        _validate(recording, _config())


@pytest.mark.parametrize(
    "model_id, expected",
    [("dinov3_match", "temporal_decode"), ("model_001", "despike_smooth")],
)
def test_each_detect_method_defaults_to_its_own_track_method(
    detect_runs, model_id, expected
):
    """§26: ``dinov3_match`` → ``temporal_decode``, a distilled student → ``despike_smooth``,
    through the one model lookup (``keypoint_method_class``) — the student is not in
    ``KEYPOINT_MODELS``, so a registry-only lookup would find no default for it."""
    import detail.models.patch_match  # noqa: F401 — registers dinov3_match
    from detail.storage import models as models_store

    it, recording = detect_runs
    mdir = models_store.model_dir(it.experiment.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir, {"id": "model_001", "name": "s", "status": "complete", "task": "pose"}
    )
    update_detect_yaml(detect_run_dir(recording.dir, "kd_01"), model_id=model_id)
    assert _validate(recording, _config())["method"] == expected


def test_progress_assigns_first_then_filters_per_subject(detect_runs, monkeypatch):
    """§26 *Job dataclasses and progress*: ``assigning`` per part — forced at the boundary,
    with no subject — then ``filtering`` per (subject, part); the bar moves inside the
    assignment of a single part, with an ETA."""
    it, recording = detect_runs
    # a throttle that never lets a non-forced event through: what is left is what is forced
    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 3600.0)
    _m, _c, _d, forced = _track_run(it, recording, _config())
    messages = [e.message for e in forced]
    assert messages[:2] == ["assigning", "assigning"]  # before the count, at the part
    assert forced[0].track_id is None and forced[1].track_id is None
    first_filter = messages.index("filtering")
    assert "assigning" not in messages[first_filter:]
    assert {e.track_id for e in forced if e.message == "filtering"} == {0, 1}
    assert messages[-1] == "complete"
    assert forced[-1].frames_done == forced[-1].frames_total == 7  # subject-frames
    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    _m, _c, _d, events = _track_run(it, recording, _config(), run_id="kt_02")
    mid = [
        e
        for e in events
        if e.message == "assigning" and 0 < e.frames_done < e.frames_total
    ]
    assert mid and mid[-1].parts_total == 1 and mid[-1].eta_s is not None
    assert all(e.track_id is None for e in events if e.message == "assigning")


def test_the_progress_total_is_the_frames_the_run_records(detect_runs):
    """The bar's ``frames_total`` is read as a frame count — the performance report divides it
    by seconds — so it must be the subject-frames the run processed, the number ``track.yaml``
    records, not the two-pass step count that weights the bar. Reporting the steps doubled the
    benchmark's keypoint-tracking fps (192,816 "frames" for 96,408)."""
    it, recording = detect_runs
    _m, _c, run_dir, events = _track_run(it, recording, _config())
    recorded = read_track_yaml(run_dir)["timing"]["frames"]
    assert recorded > 0
    assert {e.frames_total for e in events if e.frames_total} == {recorded}
    assert events[-1].frames_done == recorded
    done = [e.frames_done for e in events]
    assert done == sorted(done), "the bar went backwards"


def test_temporal_decode_over_the_stage(detect_runs):
    it, recording = detect_runs
    _m, config, _d, events = _track_run(
        it,
        recording,
        _config(method="temporal_decode", config={"smooth_window_s": 0.2}),
        run_id="kt_02",
    )
    assert config["config"]["smooth_window_s"] == 0.2
    assert config["config"]["transition_sigma_body"] == 0.15  # the defaults filled in
    assert "decoding" in [e.message for e in events]
    rows = KeypointTrackRun(recording.dir, "kt_02").rows(0, "part_001")
    assert rows["points"].shape == (5, 2, 2)
    assert set(np.unique(rows["changed"])) <= {0, 1, 2}


def test_re_running_the_same_id_is_a_whole_re_run(detect_runs):
    it, recording = detect_runs
    _track_run(it, recording, _config())
    _track_run(it, recording, _config())
    run = KeypointTrackRun(recording.dir, "kt_01")
    assert run.rows(0, "part_001")["frame_idx"].tolist() == [0, 1, 2, 3, 4]
    assert len(run.assignments("part_001")["subject_id"]) == 9


@pytest.mark.parametrize(
    "config, fragment",
    [
        (_config("kd_99"), "not found"),
        (_config(method="nope"), "unknown track method"),
        (_config("kd_02", method="temporal_decode"), "stored one per point"),
        (
            _config(method="despike_smooth", config={"despike_k": -1}),
            "despike_k",
        ),
        (_config(match_min_inside=0.0), "match_min_inside"),
        (_config(match_min_inside=1.5), "match_min_inside"),
        (_config(match_mask_grow=-0.1), "match_mask_grow"),
    ],
)
def test_validate_names_what_to_fix(detect_runs, config, fragment):
    _it, recording = detect_runs
    with pytest.raises(ValueError, match=fragment):
        _validate(recording, config)


def test_validate_refuses_an_incomplete_detect_run_and_a_recording_without_fps(
    detect_runs,
):
    _it, recording = detect_runs
    data = recording_store.read_recording(recording.dir)
    data["keypoint_detect_runs"][0]["status"] = "running"
    recording_store.write_recording(recording.dir, data)
    with pytest.raises(ValueError, match="is 'running', not complete"):
        _validate(recording, _config())
    data["keypoint_detect_runs"][0]["status"] = "complete"
    data["video"]["fps_actual"] = None
    recording_store.write_recording(recording.dir, data)
    with pytest.raises(ValueError, match="no fps_actual"):
        _validate(recording, _config())


def test_the_match_defaults_are_pose_matchs():
    """The config's defaults and ``pose_match``'s (which the distill teacher calls with) are
    one policy; a changed one must change both."""
    fields = KeypointTrackConfig.model_fields
    assert fields["match_min_inside"].default == pose_match.DEFAULT_MIN_INSIDE
    assert fields["match_mask_grow"].default == pose_match.DEFAULT_MASK_GROW


def test_the_detect_runs_skeleton_ref_is_carried_into_track_yaml(detect_runs):
    """§26 *The Keypoints manager*: the library's dependents search reads `skeleton_ref.id`
    from `track.yaml` as well as `detect.yaml`, so the track stage carries it on."""
    it, recording = detect_runs
    ref = {"id": "sk_03", "name": "two points", "digest": SKELETON.digest()}
    update_detect_yaml(detect_run_dir(recording.dir, "kd_01"), skeleton_ref=ref)
    _m, config, run_dir, _e = _track_run(it, recording, _config())
    assert config["skeleton_ref"] == ref
    assert read_track_yaml(run_dir)["skeleton_ref"] == ref
    assert (
        read_detect_yaml(detect_run_dir(recording.dir, "kd_01"))["skeleton_ref"] == ref
    )
    _m, plain, plain_dir, _e = _track_run(
        it, recording, _config("kd_02"), run_id="kt_02"
    )
    assert plain["skeleton_ref"] is None  # the detect run carried none
    assert read_track_yaml(plain_dir)["skeleton_ref"] is None


def test_a_keypoint_track_run_is_a_dependent_of_its_tracking_run():
    """§6.5 ``aux_refs``: deleting a tracking run a keypoint track run assigned against must
    be refused, so the stage declares the tracking run as its auxiliary input."""
    from detail.api import recording as recording_api

    assert keypoint_track.KeypointTrackModule.aux_refs == (
        ("tracking", "tracking_run_id"),
    )
    assert (
        "keypoint_track",
        "keypoint_track_runs",
        "tracking_run_id",
    ) in recording_api._downstream_edges("tracking")
