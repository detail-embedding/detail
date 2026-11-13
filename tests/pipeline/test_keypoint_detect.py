"""``keypoint_detect`` (§26): the stage over fake methods, on the shared tracked recording.

The method is the one legitimate mock (it is an external inference call); everything else —
the recording, the detection and tracking zarrs, the video, the writers — is real and on disk.

The stage runs on a **detection run** and writes one row per pose instance, no subject. Two
fakes cover the two kinds of method: ``_FakeKeypointModel`` is per-detection (one instance per
detection it is handed, reads masks), ``_FakePoseModel`` is whole-frame (any number of poses per
frame, each with a confidence ``min_pose_confidence`` applies to).
"""

import threading

import cv2
import numpy as np
import pytest

from detail.core.dataclasses import ComputeTarget, FrameResult
from detail.core.keypoints import ScoreSpec
from detail.core.protocols import KeypointModel
from detail.core.registries import KEYPOINT_MODELS
from detail.core.skeleton import Skeleton
from detail.pipeline import keypoint_detect
from detail.storage import recording as recording_store
from detail.storage.keypoint_runs import (
    KeypointDetectRun,
    part_path,
    read_detect_yaml,
)
from detail.storage.keypoint_zarr import KeypointDetectPartReader
from detail.storage.zarr_io import TracksSummaryWriter

SKELETON = Skeleton(names=("nose", "tail"), edges=(("nose", "tail"),))
BOX_A = (0.1, 0.1, 0.5, 0.5)
BOX_A2 = (0.3, 0.3, 0.7, 0.7)  # a second detection of the same subject on frame 2
BOX_B = (0.5, 0.5, 0.9, 0.9)
#: Detection rows of the fixture's part, in order: (frame_idx, bbox).
DET_ROWS = [
    (0, BOX_A),
    (1, BOX_A),
    (1, BOX_B),
    (2, BOX_A),
    (2, BOX_A2),
    (3, BOX_A),
    (3, BOX_B),
    (4, BOX_A),
    (4, BOX_B),
]


class _FakeKeypointModel(KeypointModel):
    """Per-detection: places each detection's two points on its box corners and reports the
    mask area it was handed, so a test can read back what the stage gave it."""

    model_id = "fake_kp"
    supported_backends = ["cpu"]
    needs_examples = False
    per_detection = True
    needs_mask = True
    n_candidates = 2
    score_spec = ScoreSpec("heatmap peak", 0.0, 1.0, None)
    default_track_method = "despike_smooth"
    instances: list = []

    def __init__(self, variant=None, *, crop_px=None):
        super().__init__(variant)
        self.crop_px = crop_px
        self.loaded_on = None
        self.batches: list[int] = []
        self.frames_seen: list[int] = []
        self.on_batch = None
        type(self).instances.append(self)

    def skeleton(self) -> Skeleton:
        return SKELETON

    def load(self, device):
        self.loaded_on = device

    def predict_batch(self, frames):
        self.batches.append(len(frames))
        out = []
        for sf in frames:
            self.frames_seen.append(sf.frame_idx)
            rows = sorted(sf.boxes)
            n = len(rows)
            points = np.zeros((n, 2, 2), np.float32)
            area = np.zeros((n, 2), np.float32)
            for i, row in enumerate(rows):
                x0, y0, x1, y1 = sf.boxes[row]
                points[i, 0] = (x0, y0)
                points[i, 1] = (x1, y1)
                area[i] = float(sf.masks[row].sum()) if sf.masks else -1.0
            out.append(
                FrameResult(
                    frame_idx=sf.frame_idx,
                    t_ns=sf.t_ns,
                    n=n,
                    bbox=np.asarray([sf.boxes[r] for r in rows], np.float32).reshape(
                        n, 4
                    ),
                    confidence=np.full(n, np.nan, np.float32),
                    detection_row=np.asarray(rows, np.int32),
                    points=points,
                    point_scores=np.full((n, 2), 0.5, np.float32),
                    candidates=np.stack([points, points], axis=2),
                    candidate_scores=np.full((n, 2, 2), 0.5, np.float32),
                    extras={"mask_area": area},
                )
            )
        if self.on_batch is not None:
            self.on_batch(self)
        return out


#: The whole-frame fake's poses per frame: their box confidences.
POSE_CONFIDENCES = {0: [], 1: [0.9], 2: [0.8, 0.3, 0.1], 3: [0.24], 4: [0.6, 0.5]}


class _FakePoseModel(KeypointModel):
    """Whole-frame, like the YOLO-pose student: sees every frame, returns any number of poses
    per frame (``POSE_CONFIDENCES``), keeps those at or above the threshold it was given.
    """

    model_id = "fake_pose"
    supported_backends = ["cpu"]
    per_detection = False
    reports_confidence = True
    n_candidates = 1
    score_spec = ScoreSpec("detector confidence", 0.0, 1.0, 0.5)
    default_track_method = "despike_smooth"
    instances: list = []

    def __init__(self, variant=None):
        super().__init__(variant)
        self.threshold = None
        self.frames_seen: list[int] = []
        self.boxes_seen: list[dict] = []
        type(self).instances.append(self)

    def skeleton(self) -> Skeleton:
        return SKELETON

    def load(self, device):
        pass

    def set_min_pose_confidence(self, threshold):
        self.threshold = threshold

    def predict_batch(self, frames):
        assert self.threshold is not None, "the stage never set min_pose_confidence"
        out = []
        for sf in frames:
            self.frames_seen.append(sf.frame_idx)
            self.boxes_seen.append(sf.boxes)
            conf = np.asarray(
                [c for c in POSE_CONFIDENCES[sf.frame_idx] if c >= self.threshold],
                np.float32,
            )
            n = len(conf)
            out.append(
                FrameResult(
                    frame_idx=sf.frame_idx,
                    t_ns=sf.t_ns,
                    n=n,
                    bbox=np.tile(np.asarray(BOX_A, np.float32), (n, 1)),
                    confidence=conf,
                    points=np.full((n, 2, 2), 0.25, np.float32),
                    point_scores=np.tile(conf[:, None], (1, 2)),
                )
            )
        return out


@pytest.fixture
def tracked(processed_recording, monkeypatch):
    """A complete detection run ``det_01`` over five 40x40 frames — nine detections, two on
    frame 2 that overlap — plus the video, and a tracking run ``track_01`` on it (two tracks,
    one row unassigned) for the track stage's tests.
    """
    it = processed_recording
    sq = it.square
    masks = [
        sq(4, 20, 4, 20),
        sq(4, 20, 4, 20),
        sq(20, 36, 20, 36),
        sq(4, 20, 4, 20),
        sq(12, 28, 12, 28),  # an anchor secondary of track 0
        sq(4, 20, 4, 20),
        sq(20, 36, 20, 36),
        sq(4, 20, 4, 20),
        sq(20, 36, 20, 36),
    ]
    index_in_frame = [0, 0, 1, 0, 1, 0, 1, 0, 1]
    detections = [
        (frame, index, box, mask)
        for (frame, box), index, mask in zip(DET_ROWS, index_in_frame, masks)
    ]
    object_ids = [0, 0, 1, 0, 0, 0, 1, 0, -1]
    recording = it.add_recording("recording_002", detections, object_ids)
    _complete_detection_run(recording)
    _write_video(recording)
    with TracksSummaryWriter(
        recording.dir / "tracking" / "track_01" / "tracks.zarr"
    ) as w:
        w.add_track(0, "mouse", 0, 4, 6, 0, 0.9)
        w.add_track(1, "mouse", 1, 3, 2, 1, 0.9)
    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp", _FakeKeypointModel)
    monkeypatch.setitem(KEYPOINT_MODELS, "fake_pose", _FakePoseModel)
    _FakeKeypointModel.instances.clear()
    _FakePoseModel.instances.clear()
    return it, recording


def _complete_detection_run(recording) -> None:
    data = recording_store.read_recording(recording.dir)
    data["detection_runs"] = [
        {"id": "det_01", "status": "complete", "every_n_frames": 1}
    ]
    recording_store.write_recording(recording.dir, data)


def _write_video(recording) -> None:
    video = recording.dir / "video" / "part_001.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (40, 40)
    )
    assert writer.isOpened()
    for i in range(5):
        writer.write(np.full((40, 40, 3), 30 * i, np.uint8))
    writer.release()


def _config(**overrides) -> dict:
    config = {
        "detection_run_id": "det_01",
        "model_id": "fake_kp",
        "model_variant": None,
        "example_set_id": None,
        "min_pose_confidence": None,
        "decode": None,
        "side_rule": None,
        "crop_px": None,
        "every_n_frames": None,
        "batch_size": 2,
        "compute": ComputeTarget("cpu", None, None, None),
    }
    config.update(overrides)
    return config


def _run(it, recording, config, *, stop_event=None, run_id="kd_01"):
    module = keypoint_detect.KeypointDetectModule()
    data = recording_store.read_recording(recording.dir)
    config = module.validate(recording.dir, data, config)
    job = module.build_job(recording.id, run_id, config)
    run_dir = module.run_dir_for(it.experiment.dir, job)
    run_dir.mkdir(parents=True, exist_ok=True)
    events: list = []
    keypoint_detect.run(job, run_dir, events.append, stop_event)
    return module, config, run_dir, events


def test_a_per_detection_method_writes_one_row_per_detection_with_no_subject(tracked):
    it, recording = tracked
    module, config, run_dir, _events = _run(it, recording, _config())
    run = KeypointDetectRun(recording.dir, "kd_01")
    assert run.skeleton == SKELETON and run.part_ids() == ["part_001"]
    assert run.detection_run_id == "det_01"
    rows = run.rows("part_001")
    assert "track_id" not in rows  # a pose meets a subject only in keypoint_track
    assert rows["frame_idx"].tolist() == [f for f, _box in DET_ROWS]
    assert rows["t_ns"].tolist() == [f * 1000 for f, _box in DET_ROWS]
    assert rows["detection_row"].tolist() == list(range(len(DET_ROWS)))
    np.testing.assert_allclose(rows["bbox"], [box for _f, box in DET_ROWS])
    assert np.isnan(rows["confidence"]).all()  # the method reports none
    # frame 2: two overlapping detections are two instances, each cropped and masked alone
    np.testing.assert_allclose(rows["points"][4, 1], BOX_A2[2:], atol=1e-6)
    assert rows["extras/mask_area"][3, 0] == it.square(4, 20, 4, 20).sum()
    assert rows["extras/mask_area"][4, 0] == it.square(12, 28, 12, 28).sum()
    assert rows["candidates"].shape == (9, 2, 2, 2)
    model = _FakeKeypointModel.instances[
        -1
    ]  # validate builds one too; the run's is last
    assert model.batches == [2, 2, 1]  # bounded batches of frames, each frame once
    assert model.frames_seen == [0, 1, 2, 3, 4] and model.loaded_on == "cpu"
    meta = read_detect_yaml(run_dir)
    assert meta["detection_run_id"] == "det_01" and meta["min_pose_confidence"] is None
    assert "tracking_run_id" not in meta and "track_ids" not in meta
    assert meta["n_candidates"] == 2 and meta["skeleton_digest"] == SKELETON.digest()
    assert meta["coverage"] == {
        "part_001": {"frames": 5, "frames_with_instance": 5, "instances": 9}
    }
    assert meta["timing"]["frames"] == 5 and "status" not in meta
    record = module.run_record("kd_01", config, "complete")
    assert set(module.identity_fields) <= set(record)
    assert record["detection_run_id"] == "det_01" and "tracking_run_id" not in record
    assert record["output"] == "keypoint_detect/kd_01/"


def test_the_stages_store_has_the_instance_schema(tracked):
    """Names, dtypes, chunks and the ``layout`` attr of what the stage itself wrote (§26)."""
    it, recording = tracked
    _m, _c, run_dir, _e = _run(it, recording, _config())
    reader = KeypointDetectPartReader(part_path(run_dir, "part_001"))
    assert reader._root.attrs["layout"] == "instances"
    assert reader._root.attrs["schema_version"] == 1
    expected = {
        "frame_idx": ("int32", (10000,)),
        "t_ns": ("int64", (10000,)),
        "bbox": ("float32", (10000, 4)),
        "confidence": ("float32", (10000,)),
        "detection_row": ("int32", (10000,)),
        "points": ("float32", (10000, 2, 2)),
        "point_scores": ("float32", (10000, 2)),
        "candidates": ("float32", (10000, 2, 2, 2)),
        "candidate_scores": ("float32", (10000, 2, 2)),
        "extras/mask_area": ("float32", (10000, 2)),
    }
    names = set(reader.column_names) | {"extras/" + n for n in reader.extras_names}
    assert names == set(expected)
    for name, (dtype, chunks) in expected.items():
        assert str(reader.column(name).dtype) == dtype, name
        assert tuple(reader.column(name).chunks) == chunks, name


def test_a_whole_frame_method_visits_every_frame_and_keeps_poses_above_the_threshold(
    tracked,
):
    """Zero, one and many instances per frame; nothing written for a frame with none; the
    default ``min_pose_confidence`` resolved, written back and handed to the model."""
    it, recording = tracked
    _m, config, run_dir, _e = _run(it, recording, _config(model_id="fake_pose"))
    assert config["min_pose_confidence"] == 0.25
    model = _FakePoseModel.instances[-1]
    assert model.threshold == 0.25
    assert model.frames_seen == [0, 1, 2, 3, 4]  # frame 0 has no pose and is still seen
    assert all(boxes == {} for boxes in model.boxes_seen)  # never handed detections
    rows = KeypointDetectRun(recording.dir, "kd_01").rows("part_001")
    assert rows["frame_idx"].tolist() == [1, 2, 2, 4, 4]
    np.testing.assert_allclose(rows["confidence"], [0.9, 0.8, 0.3, 0.6, 0.5])
    assert rows["detection_row"].tolist() == [-1] * 5
    assert "candidates" not in rows
    meta = read_detect_yaml(run_dir)
    assert meta["min_pose_confidence"] == 0.25
    assert meta["coverage"]["part_001"] == {
        "frames": 5,
        "frames_with_instance": 3,
        "instances": 5,
    }


def test_an_explicit_min_pose_confidence_is_identity_and_reaches_the_model(tracked):
    it, recording = tracked
    module, config, _d, _e = _run(
        it, recording, _config(model_id="fake_pose", min_pose_confidence=0.55)
    )
    rows = KeypointDetectRun(recording.dir, "kd_01").rows("part_001")
    assert rows["frame_idx"].tolist() == [1, 2, 4]
    assert "min_pose_confidence" in module.identity_fields
    assert module.run_record("kd_01", config, "complete")["min_pose_confidence"] == 0.55


def _set_detection_confidence(recording, values) -> None:
    """Give the fixture's ``det_01`` rows these confidences (the shared fixture writes 0.9
    for every row, which no cut-off can tell apart)."""
    from detail.storage import zarr_store

    root = zarr_store.open_group(
        recording.dir / "detections" / "det_01" / "part_001.zarr", mode="a"
    )
    root["confidence"][:] = np.asarray(values, np.float32)


def test_part_frames_keeps_only_the_detections_at_or_above_the_cut_off():
    det = {
        "frame_idx": np.array([0, 0, 1, 1, 2]),
        "t_ns": np.array([0, 0, 10, 10, 20]),
        "confidence": np.array([0.97, 0.1, 0.5, 0.95, 0.05], np.float32),
    }
    kept = keypoint_detect.part_frames(
        det, np.zeros(0), 1, per_detection=True, min_detection_confidence=0.5
    )
    # row 2 sits on the cut-off and is kept; frame 2 has nothing left and is not visited
    assert kept == [(0, 0, [0]), (1, 10, [2, 3])]
    # unset keeps every row, and needs no confidence column (the run reads none then)
    no_conf = {k: v for k, v in det.items() if k != "confidence"}
    every = keypoint_detect.part_frames(
        no_conf, np.zeros(0), 1, per_detection=True, min_detection_confidence=None
    )
    assert every == [(0, 0, [0, 1]), (1, 10, [2, 3]), (2, 20, [4])]


def test_a_detection_cut_off_crops_only_the_detections_above_it_and_is_identity(
    tracked,
):
    """A detection run kept at a low confidence holds faint boxes beside each animal, and a
    per-detection method crops every one (§26 *Visibility*). With the cut-off it crops only the
    rows at or above it; a frame whose rows all fall below is not visited; the record, the
    detect.yaml and identity carry the value."""
    it, recording = tracked
    # DET_ROWS: frame 1's, 2's and 3's second boxes are faint, and both of frame 4's
    _set_detection_confidence(recording, [0.9, 0.9, 0.1, 0.9, 0.2, 0.9, 0.1, 0.3, 0.05])
    module, config, run_dir, _e = _run(
        it, recording, _config(min_detection_confidence=0.5)
    )
    rows = KeypointDetectRun(recording.dir, "kd_01").rows("part_001")
    assert rows["detection_row"].tolist() == [0, 1, 3, 5]
    assert rows["frame_idx"].tolist() == [0, 1, 2, 3]
    assert _FakeKeypointModel.instances[-1].frames_seen == [0, 1, 2, 3]
    meta = read_detect_yaml(run_dir)
    assert meta["min_detection_confidence"] == 0.5
    assert meta["coverage"]["part_001"] == {
        "frames": 4,
        "frames_with_instance": 4,
        "instances": 4,
    }
    assert "min_detection_confidence" in module.identity_fields
    record = module.run_record("kd_01", config, "complete")
    assert record["min_detection_confidence"] == 0.5


def test_a_record_from_before_the_detection_cut_off_matches_a_run_that_leaves_it_unset(
    tracked,
):
    """Every detect run written before the cut-off existed cropped every detection, so a
    record without ``min_detection_confidence`` reads as None (``identity_defaults``) and
    shares a spec with a new run that leaves it unset; a run that sets it is another run.
    """
    from detail.api import run_spec

    it, recording = tracked
    module, config, _d, _e = _run(it, recording, _config())
    record = module.run_record("kd_01", config, "complete")
    assert record["min_detection_confidence"] is None
    data = recording_store.read_recording(recording.dir)

    def spec(rec):
        return run_spec.freeze_spec(run_spec.spec_of(data, "keypoint_detect", rec))

    before = {k: v for k, v in record.items() if k != "min_detection_confidence"}
    assert spec(before) == spec(record)
    assert spec({**record, "min_detection_confidence": 0.5}) != spec(record)


def test_every_n_frames_keeps_the_source_frame_rule(tracked):
    it, recording = tracked
    _run(it, recording, _config(every_n_frames=2))
    run = KeypointDetectRun(recording.dir, "kd_01")
    rows = run.rows("part_001")
    assert rows["frame_idx"].tolist() == [0, 2, 2, 4, 4]
    assert rows["detection_row"].tolist() == [0, 3, 4, 7, 8]
    assert run.coverage()["part_001"]["frames"] == 3
    assert read_detect_yaml(run.dir)["every_n_frames"] == 2


def test_the_viewer_reads_every_instance_on_a_frame(tracked):
    it, recording = tracked
    _run(it, recording, _config())
    run = KeypointDetectRun(recording.dir, "kd_01")
    two = run.at("part_001", 2)
    assert two["row"].tolist() == [3, 4] and two["points"].shape == (2, 2, 2)
    assert two["detection_row"].tolist() == [3, 4]
    one = run.at("part_001", 0)
    assert one["row"].tolist() == [0]
    none = run.at("part_001", 7)
    assert len(none["row"]) == 0 and none["points"].shape == (0, 2, 2)
    assert len(run.at("part_009", 0)["row"]) == 0  # a part with no store


def test_progress_names_each_phase_and_counts_frames_across_the_run(tracked):
    it, recording = tracked
    _m, _c, _d, events = _run(it, recording, _config())
    messages = [e.message for e in events]
    assert messages[0] == "counting frames"
    assert messages[1] == "loading model"
    assert "placing keypoints" in messages and messages[-1] == "complete"
    assert all(e.track_id is None for e in events)  # the detect stage has no subjects
    assert events[1].frames_total == 5 and events[1].frames_done == 0
    done = [e.frames_done for e in events[2:]]
    assert done == sorted(done) and done[-1] == 5
    assert all(e.parts_total == 1 for e in events)
    assert events[-1].parts_done == 1 and events[-1].eta_s == 0.0
    assert any(e.eta_s is not None for e in events[2:-1])


def test_a_stopped_run_resumes_after_the_last_whole_frame(tracked, monkeypatch):
    """Stop after the first batch (frames 0, 1), resume: the second run starts at frame 2,
    places nothing twice, and the store equals an uninterrupted run's."""
    it, recording = tracked
    stop = threading.Event()
    original = _FakeKeypointModel.__init__

    def init(self, *a, **k):
        original(self, *a, **k)
        self.on_batch = lambda model: stop.set()

    monkeypatch.setattr(_FakeKeypointModel, "__init__", init)
    _m, _c, run_dir, events = _run(it, recording, _config(), stop_event=stop)
    assert events[-1].message == "stopped"
    first = KeypointDetectRun(recording.dir, "kd_01").rows("part_001")
    assert first["frame_idx"].tolist() == [0, 1, 1]
    assert read_detect_yaml(run_dir)["coverage"]["part_001"] == {
        "frames": 2,
        "frames_with_instance": 2,
        "instances": 3,
    }
    monkeypatch.setattr(_FakeKeypointModel, "__init__", original)
    _FakeKeypointModel.instances.clear()
    _m, _c, run_dir, events = _run(it, recording, _config())
    model = _FakeKeypointModel.instances[-1]
    assert model.frames_seen == [2, 3, 4]
    assert events[1].frames_done == 2 and events[1].frames_total == 5
    rows = KeypointDetectRun(recording.dir, "kd_01").rows("part_001")
    assert rows["frame_idx"].tolist() == [f for f, _box in DET_ROWS]
    assert rows["detection_row"].tolist() == list(range(9))
    assert read_detect_yaml(run_dir)["coverage"]["part_001"]["instances"] == 9


def test_a_finished_run_resumed_places_nothing(tracked):
    it, recording = tracked
    _run(it, recording, _config())
    _FakeKeypointModel.instances.clear()
    _m, _c, run_dir, events = _run(it, recording, _config())
    assert _FakeKeypointModel.instances[-1].batches == []
    assert events[1].frames_done == 5 == events[1].frames_total
    assert len(KeypointDetectRun(recording.dir, "kd_01").rows("part_001")["t_ns"]) == 9


def test_a_stop_before_any_frame_keeps_the_record_and_writes_no_store(tracked):
    it, recording = tracked
    stop = threading.Event()
    stop.set()
    _m, _c, run_dir, events = _run(it, recording, _config(), stop_event=stop)
    assert read_detect_yaml(run_dir)["coverage"]["part_001"] == {
        "frames": 0,
        "frames_with_instance": 0,
        "instances": 0,
    }
    assert events[-1].message == "stopped"
    assert KeypointDetectRun(recording.dir, "kd_01").part_ids() == []


def test_a_part_with_no_detection_places_nothing_per_detection_but_every_frame_whole(
    processed_recording, monkeypatch
):
    """The empty-part edge: a detection part with zero rows. A per-detection method has no
    frame to visit; a whole-frame method still walks every frame."""
    it = processed_recording
    recording = it.add_recording("recording_003", [], [])
    _complete_detection_run(recording)
    _write_video(recording)
    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp", _FakeKeypointModel)
    monkeypatch.setitem(KEYPOINT_MODELS, "fake_pose", _FakePoseModel)
    _m, _c, run_dir, _e = _run(it, recording, _config())
    assert read_detect_yaml(run_dir)["coverage"]["part_001"] == {
        "frames": 0,
        "frames_with_instance": 0,
        "instances": 0,
    }
    assert KeypointDetectRun(recording.dir, "kd_01").part_ids() == []
    _m, _c, run_dir, _e = _run(
        it, recording, _config(model_id="fake_pose"), run_id="kd_02"
    )
    assert read_detect_yaml(run_dir)["coverage"]["part_001"]["frames"] == 5


# --------------------------------------------------------------------------- #
# validate(): §26 "Validation checks", each named
# --------------------------------------------------------------------------- #
def _validate(recording, config):
    data = recording_store.read_recording(recording.dir)
    return keypoint_detect.KeypointDetectModule().validate(recording.dir, data, config)


def test_validate_resolves_and_writes_back_every_defaulted_input(tracked):
    _it, recording = tracked
    out = _validate(recording, _config())
    assert out["every_n_frames"] == 1 and out["batch_size"] == 2
    assert out["skeleton"] == SKELETON.to_dict()
    assert out["skeleton_digest"] == SKELETON.digest()
    assert out["n_candidates"] == 2 and out["matching"] is None
    assert out["backbone"] is None and out["example_set"] is None
    assert out["min_pose_confidence"] is None  # a per-detection method reports none
    assert _validate(recording, _config(model_id="fake_pose"))[
        "min_pose_confidence"
    ] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"detection_run_id": "det_99"}, "detection run 'det_99' not found"),
        ({"model_id": "nope"}, "unknown keypoint method 'nope'"),
        ({"example_set_id": "ex_01"}, "takes no example set"),
        ({"model_id": "dinov3_match"}, "learns from an example set"),
        (
            {"compute": ComputeTarget("cuda", [0], None, None)},
            "does not support backend",
        ),
        ({"min_pose_confidence": 0.3}, "'fake_kp' reports no pose confidence"),
        ({"min_pose_confidence": 1.5}, "less than or equal to 1"),
        (
            {"model_id": "fake_pose", "min_detection_confidence": 0.5},
            "'fake_pose' looks at the whole frame",
        ),
        ({"min_detection_confidence": -0.1}, "greater than or equal to 0"),
        ({"every_n_frames": 0}, "greater than or equal to 1"),
    ],
)
def test_validate_names_what_to_fix(tracked, overrides, fragment):
    _it, recording = tracked
    with pytest.raises(ValueError, match=fragment):
        _validate(recording, _config(**overrides))


def test_validate_refuses_an_incomplete_detection_run(tracked):
    _it, recording = tracked
    data = recording_store.read_recording(recording.dir)
    data["detection_runs"][0]["status"] = "running"
    recording_store.write_recording(recording.dir, data)
    with pytest.raises(ValueError, match="is 'running', not complete"):
        _validate(recording, _config())


def test_validate_refuses_a_cadence_finer_than_the_detection_runs(tracked):
    _it, recording = tracked
    data = recording_store.read_recording(recording.dir)
    data["detection_runs"][0]["every_n_frames"] = 2
    recording_store.write_recording(recording.dir, data)
    with pytest.raises(ValueError):
        _validate(recording, _config(every_n_frames=3))
    assert _validate(recording, _config())["every_n_frames"] == 2


def test_validate_passes_the_methods_settings_through_and_refuses_what_it_lacks(
    tracked,
):
    _it, recording = tracked
    out = _validate(recording, _config(crop_px=448))
    assert out["crop_px"] == 448  # the fake takes crop_px; the resolver passed it
    with pytest.raises(ValueError, match="takes no \\['decode'\\]"):
        _validate(recording, _config(decode="skeleton"))


def test_the_example_sets_skeleton_ref_reaches_the_record_and_detect_yaml(
    tracked, monkeypatch
):
    """§26 *The Keypoints manager*: everything that copies a skeleton records which library
    skeleton it came from, and the library's dependents search reads `skeleton_ref.id` from
    `detect.yaml`; a skeleton a run copied must not be editable or deletable under it.
    """
    from detail.core.examples import ExampleFrame, ExampleMark, ExampleSet
    from detail.pipeline import keypoint_detect as stage

    it, recording = tracked
    ref = {"id": "sk_03", "name": "two points", "digest": SKELETON.digest()}
    frame = ExampleFrame(
        image="frames/abc.png",
        recording_id=recording.id,
        tracking_run_id="track_01",
        track_id=0,
        part_id="part_001",
        frame_idx=0,
        t_ns=0,
        bbox=BOX_A,
        marks=[ExampleMark("nose", x=0.2, y=0.2), ExampleMark("tail", x=0.4, y=0.4)],
    )
    examples = ExampleSet(
        id="ex_01",
        name="dots",
        purpose="example",
        created_at="2026-10-08T00:00:00Z",
        skeleton=SKELETON,
        frames=[frame],
        skeleton_ref=ref,
    )
    monkeypatch.setattr(stage.example_sets, "read_set", lambda d, set_id: examples)
    monkeypatch.setattr(stage.example_sets, "load_images", lambda d, es: {})
    monkeypatch.setattr(stage.example_sets, "image_hashes", lambda d, es: {})
    monkeypatch.setattr(
        stage.example_sets, "load_image", lambda *a, **k: np.zeros((8, 8, 3), np.uint8)
    )

    class _Learner(_FakeKeypointModel):
        model_id = "fake_learner"
        needs_examples = True

        def __init__(
            self, variant=None, *, examples=None, example_images=None, crop_px=None
        ):
            super().__init__(variant, crop_px=crop_px)

    monkeypatch.setitem(KEYPOINT_MODELS, "fake_learner", _Learner)
    module, config, run_dir, _events = _run(
        it, recording, _config(model_id="fake_learner", example_set_id="ex_01")
    )
    assert config["skeleton_ref"] == ref
    assert read_detect_yaml(run_dir)["skeleton_ref"] == ref
    assert module.build_job(recording.id, "kd_09", config).skeleton_ref == ref
    _m, plain, plain_dir, _e = _run(it, recording, _config(), run_id="kd_02")
    assert plain["skeleton_ref"] is None  # a method that takes no example set
    assert read_detect_yaml(plain_dir)["skeleton_ref"] is None


# --------------------------------------------------------------------------- #
# A keypoint model from the machine catalog (§12 *Keypoint models in the catalog*)
# --------------------------------------------------------------------------- #
def _catalog_student(model_id: str):
    """A pose record in the (per-test isolated) machine catalog, with a weights file."""
    from detail.models import imported as imported_store
    from detail.storage import models as models_store

    mdir = imported_store.model_dir(model_id)
    (mdir / "weights").mkdir(parents=True)
    (mdir / "weights" / "best.pt").write_bytes(b"weights")
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "name": "catalog student",
            "status": "complete",
            "role": "keypoints",
            "task": "pose",
            "model_family": "yolo_pose",
            "weights": {"best": "weights/best.pt"},
            "skeleton": SKELETON.to_dict(),
        },
    )
    return mdir


@pytest.fixture
def fake_ultralytics(monkeypatch):
    """The one external call: Ultralytics' YOLO. Records which weights each model loaded
    and returns no pose, so the stage runs end to end on the real ``TrainedYoloPose``.
    """
    import sys
    import types

    from detail.models import yolo_pose

    loaded: list = []

    class _Yolo:
        def __init__(self, path):
            loaded.append(path)

        def predict(self, frames, **_kwargs):
            return [types.SimpleNamespace(boxes=None, keypoints=None) for _ in frames]

    class _Split:  # SplitPredict's halves over the fake: the whole predict in run
        def __init__(self, model, **kwargs):
            self._model, self._kwargs = model, kwargs

        def prepare(self, frames):
            return frames

        def run(self, prepared, frames):
            return self._model.predict(frames, **self._kwargs)

    fake = types.ModuleType("ultralytics")
    fake.YOLO = _Yolo
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setattr(yolo_pose, "import_ultralytics", lambda: None)
    monkeypatch.setattr(yolo_pose, "is_importable", lambda _m: True)
    monkeypatch.setattr(yolo_pose, "SplitPredict", _Split)
    return loaded


def test_a_catalog_student_records_weights_source_imported_and_builds_from_the_catalog(
    tracked, fake_ultralytics
):
    """Lookup is built-in → experiment → catalog; the run writes back where the model was
    found (Tenet 9) — on the record, the job and ``detect.yaml`` — and is identity."""
    it, recording = tracked
    mdir = _catalog_student("rat_pose")

    module, config, run_dir, _events = _run(
        it, recording, _config(model_id="rat_pose", batch_size=4)
    )

    assert config["weights_source"] == "imported"
    assert module.extra_record_fields(config)["weights_source"] == "imported"
    assert "weights_source" in module.identity_fields
    assert read_detect_yaml(run_dir)["weights_source"] == "imported"
    assert fake_ultralytics == [str(mdir / "weights" / "best.pt")]
    plain = _validate(recording, _config())
    assert plain["weights_source"] is None  # a registered method has no source


def test_the_worker_builds_the_recorded_source_not_the_first_match(
    tracked, fake_ultralytics
):
    """An experiment model and a catalog model can share an id (§12): the job carries the
    source validate resolved, and the worker builds that one rather than searching again.
    """
    from detail.storage import models as models_store

    it, recording = tracked
    catalog = _catalog_student("model_001")
    mine = models_store.model_dir(it.experiment.dir, "model_001")
    (mine / "weights").mkdir(parents=True)
    (mine / "weights" / "best.pt").write_bytes(b"weights")
    models_store.write_model_yaml(
        mine, {**models_store.read_model_yaml(catalog), "name": "mine"}
    )

    config = _validate(recording, _config(model_id="model_001"))
    assert config["weights_source"] == "trained"  # the experiment's comes first

    module = keypoint_detect.KeypointDetectModule()
    config["weights_source"] = "imported"
    job = module.build_job(recording.id, "kd_05", config)
    assert type(job).from_dict(job.to_dict()).weights_source == "imported"
    model = keypoint_detect._build_model(job, it.experiment.dir)
    assert model._weights_path == catalog / "weights" / "best.pt"


# --------------------------------------------------------------------------- #
# The four stages (§26 *Execution*)
# --------------------------------------------------------------------------- #
class _SplitKeypointModel(_FakeKeypointModel):
    """``_FakeKeypointModel`` opted into the three-part split, recording the thread each half
    ran on and each forward's batch size. Its rows come from each frame's own boxes and masks,
    so a batch handed to the wrong stage call writes the wrong rows."""

    model_id = "fake_kp_split"
    instances: list = []

    def __init__(self, variant=None, *, crop_px=None):
        super().__init__(variant, crop_px=crop_px)
        self.threads: dict[str, set] = {"prepare": set(), "run": set(), "finish": set()}
        self.forwards: list[int] = []
        self.on_forward = None

    def predict_batch(self, frames):
        return self.finish_batch(self.run_prepared(self.prepare_batch(frames)))

    def prepare_batch(self, frames):
        self.threads["prepare"].add(threading.current_thread().name)
        return list(frames)

    def run_prepared(self, prepared):
        self.threads["run"].add(threading.current_thread().name)
        self.forwards.append(len(prepared))
        if self.on_forward is not None:
            self.on_forward(self)
        return prepared

    def finish_batch(self, raw):
        self.threads["finish"].add(threading.current_thread().name)
        return _FakeKeypointModel.predict_batch(self, raw)


def _stage_threads():
    return sorted(t.name for t in threading.enumerate() if "lookahead" in t.name)


@pytest.fixture
def split_kp(tracked, monkeypatch):
    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp_split", _SplitKeypointModel)
    _SplitKeypointModel.instances.clear()
    return tracked


def _assert_det_rows(rows, frames=None) -> None:
    """The rows the fixture's detections make, built from ``DET_ROWS`` rather than from any
    run, so a fault in the stage cannot also be in the reference."""
    want = [
        (i, f, box)
        for i, (f, box) in enumerate(DET_ROWS)
        if frames is None or f in frames
    ]
    assert rows["frame_idx"].tolist() == [f for _i, f, _b in want]
    assert rows["t_ns"].tolist() == [f * 1000 for _i, f, _b in want]
    assert rows["detection_row"].tolist() == [i for i, _f, _b in want]
    np.testing.assert_array_equal(
        rows["bbox"], np.asarray([b for *_x, b in want], np.float32)
    )
    np.testing.assert_array_equal(
        rows["points"][:, 1], np.asarray([b[2:] for *_x, b in want], np.float32)
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4, 100])
def test_a_split_method_writes_the_rows_and_each_half_has_its_thread(
    split_kp, batch_size
):
    it, recording = split_kp
    _run(it, recording, _config(model_id="fake_kp_split", batch_size=batch_size))
    _run(it, recording, _config(batch_size=batch_size), run_id="kd_02")
    split = _SplitKeypointModel.instances[-1]
    got = KeypointDetectRun(recording.dir, "kd_01").rows("part_001")
    whole = KeypointDetectRun(recording.dir, "kd_02").rows("part_001")
    _assert_det_rows(got)
    _assert_det_rows(whole)  # a method that does not split
    for name in got:
        np.testing.assert_array_equal(got[name], whole[name], err_msg=name)
    assert sum(split.forwards) == 5 and max(split.forwards) <= batch_size
    assert split.threads["run"] == {threading.current_thread().name}
    assert split.threads["prepare"] == {"lookahead-producer"}
    assert len(split.threads["finish"]) == 1
    assert next(iter(split.threads["finish"])).startswith("lookahead-worker")
    assert _stage_threads() == []


def test_a_stop_writes_the_computed_batches_and_nothing_past_them(
    split_kp, monkeypatch
):
    it, recording = split_kp
    stop = threading.Event()
    original = _SplitKeypointModel.__init__

    def init(self, *a, **k):
        original(self, *a, **k)
        self.on_forward = lambda model: stop.set()

    monkeypatch.setattr(_SplitKeypointModel, "__init__", init)
    _m, _c, run_dir, events = _run(
        it, recording, _config(model_id="fake_kp_split"), stop_event=stop
    )
    assert events[-1].message == "stopped"
    assert _SplitKeypointModel.instances[-1].forwards == [2]
    rows = KeypointDetectRun(recording.dir, "kd_01").rows("part_001")
    _assert_det_rows(rows, {0, 1})
    assert read_detect_yaml(run_dir)["coverage"]["part_001"]["frames"] == 2
    assert _stage_threads() == []


def test_a_write_error_fails_the_run_after_the_batches_before_it(split_kp, monkeypatch):
    from detail.storage import keypoint_zarr

    it, recording = split_kp
    add = keypoint_zarr.KeypointDetectPartWriter.add_instance
    calls = []

    def add_then_fail(self, result, row):
        calls.append(1)
        if len(calls) == 4:  # frame 2's first instance: the second batch
            raise OSError("disk full")
        return add(self, result, row)

    monkeypatch.setattr(
        keypoint_zarr.KeypointDetectPartWriter, "add_instance", add_then_fail
    )
    with pytest.raises(OSError, match="disk full"):
        _run(it, recording, _config(model_id="fake_kp_split"))
    rows = KeypointDetectRun(recording.dir, "kd_01").rows("part_001")
    _assert_det_rows(rows, {0, 1})
    assert _stage_threads() == []


class _FailingSplit(_SplitKeypointModel):
    model_id = "fake_kp_failing"
    half = "prepare"

    def prepare_batch(self, frames):
        if self.half == "prepare" and self.forwards:
            raise ValueError("prepare failed")
        return super().prepare_batch(frames)

    def run_prepared(self, prepared):
        if self.half == "run" and len(self.forwards) == 1:
            raise RuntimeError("forward failed")
        return super().run_prepared(prepared)


@pytest.mark.parametrize(
    "half,match", [("prepare", "prepare failed"), ("run", "forward failed")]
)
def test_no_stage_thread_outlives_a_failed_run(split_kp, monkeypatch, half, match):
    it, recording = split_kp
    monkeypatch.setattr(_FailingSplit, "half", half)
    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp_failing", _FailingSplit)
    with pytest.raises((ValueError, RuntimeError), match=match):
        _run(it, recording, _config(model_id="fake_kp_failing", batch_size=1))
    assert _stage_threads() == []


def test_a_decode_error_reaches_the_caller_after_the_batches_before_it(split_kp):
    # The video is shorter than the detection run: the decode thread raises at frame 3, and
    # the whole batch before it is on disk.
    it, recording = split_kp
    video = recording.dir / "video" / "part_001.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (40, 40)
    )
    for i in range(3):
        writer.write(np.full((40, 40, 3), 30 * i, np.uint8))
    writer.release()
    with pytest.raises(RuntimeError, match="frame 3"):
        _run(it, recording, _config(model_id="fake_kp_split"))
    rows = KeypointDetectRun(recording.dir, "kd_01").rows("part_001")
    _assert_det_rows(rows, {0, 1})
    assert _stage_threads() == []
