"""The label stage of a keypoint distill (§12 *Distill keypoints*), over a real detect run the
detect stage wrote on the shared tracked recording, its instances assigned to the tracking
run's subjects by §26 *Assignment*'s function (``pose_match.assign_part``).

The teacher is the fake keypoint method the detect tests use (an external inference call is
the one legitimate mock); the recording, the zarrs, the video and the writers are real.
"""

import numpy as np
import pytest
import yaml

from detail.core.keypoints import KeypointDatasetJob
from detail.core.skeleton import Skeleton
from detail.pipeline import keypoint_dataset
from detail.storage import recording as recording_store
from tests.pipeline.test_keypoint_detect import (  # noqa: F401 — `tracked` is a fixture
    SKELETON,
    _config,
    _run,
    tracked,
)

DETECT_TEACHER = {
    "kind": "keypoint_detect",
    "detect_runs": {"recording_002": "kd_01"},
    "tracking_runs": {"recording_002": "track_01"},
    "match": {"min_inside": 0.5, "mask_grow": 0.05},
}


@pytest.fixture
def taught(tracked):
    """A detect run over the tracked recording's detection run, listed in recording.yaml.
    No track run: the teacher assigns the drawn frames' instances itself."""
    it, recording = tracked
    _run(it, recording, _config())  # kd_01
    data = recording_store.read_recording(recording.dir)
    data["keypoint_detect_runs"] = [
        {
            "id": "kd_01",
            "status": "complete",
            "model_id": "fake_kp",
            "detection_run_id": "det_01",
            "output": "x",
        }
    ]
    recording_store.write_recording(recording.dir, data)
    return it, recording


def _job(it, **over) -> KeypointDatasetJob:
    fields = dict(
        job_id="dataset_001",
        dataset_id="dataset_001",
        name="rats",
        teacher=dict(DETECT_TEACHER),
        n_frames=4,
        val_split=0.0,
        score_cutoff=0.4,
        exclude_audit_s=0.0,
        every_n_frames=1,
        created_at="2026-10-08T00:00:00Z",
    )
    fields.update(over)
    return KeypointDatasetJob(**fields)


def _label(it, job, stop_event=None):
    run_dir = keypoint_dataset.run_dir_for(it.experiment.dir, job)
    run_dir.mkdir(parents=True, exist_ok=True)
    events: list = []
    keypoint_dataset.run(job, run_dir, events.append, stop_event)
    record = yaml.safe_load((run_dir / "dataset.yaml").read_text(encoding="utf-8-sig"))
    data = yaml.safe_load((run_dir / "data.yaml").read_text(encoding="utf-8-sig"))
    return run_dir, record, data, events


def _labels(run_dir) -> dict[int, list[list[str]]]:
    out = {}
    for path in sorted((run_dir / "labels" / "train").glob("*.txt")) + sorted(
        (run_dir / "labels" / "val").glob("*.txt")
    ):
        frame = int(path.stem.rsplit("_", 1)[1])
        out[frame] = [
            line.split() for line in path.read_text(encoding="utf-8-sig").splitlines()
        ]
    return out


# --- the unit pieces ------------------------------------------------------------- #
def test_flip_index_swaps_pairs_and_keeps_the_midline():
    sk = Skeleton(
        names=("nose", "left_ear", "right_ear", "tail"),
        pairs=(("left_ear", "right_ear"),),
    )
    assert keypoint_dataset.flip_index(sk) == [0, 2, 1, 3]
    assert keypoint_dataset.flip_index(SKELETON) == [0, 1]


def test_label_line_teaches_the_box_and_only_confident_points():
    line = keypoint_dataset.label_line(
        (0.1, 0.2, 0.5, 0.6),
        np.array([[0.2, 0.3], [np.nan, np.nan], [0.4, 0.5]]),
        np.array([0.9, np.nan, 0.3]),
        cutoff=0.5,
    )
    tokens = line.split()
    assert tokens[:5] == ["0", "0.300000", "0.400000", "0.400000", "0.400000"]
    assert tokens[5:8] == ["0.200000", "0.300000", "2"]
    assert tokens[8:11] == [
        "0",
        "0",
        "0",
    ], "an absent point is unlabelled, not placed at 0"
    assert tokens[11:14] == [
        "0",
        "0",
        "0",
    ], "a point below the cut-off is taught as not visible"


# --- the stage -------------------------------------------------------------------- #
def test_the_stage_writes_pose_labels_for_every_complete_frame(taught):
    it, recording = taught

    run_dir, record, data, events = _label(it, _job(it))

    labels = _labels(run_dir)
    # frame 4 carries an unassigned detection, so it is never a training image
    assert sorted(labels) == [0, 1, 2, 3]
    assert (
        len(labels[1]) == 2 and len(labels[3]) == 2
    ), "both tracked animals are labeled"
    assert len(labels[0]) == 1 and len(labels[2]) == 1
    for lines in labels.values():
        for tokens in lines:
            assert len(tokens) == 5 + 2 * 3 and tokens[0] == "0"
            assert tokens[7] == "2" and tokens[10] == "2"
    # frame 2: the teacher's box is the union of the subject's two detections
    box = [float(v) for v in labels[2][0][1:5]]
    assert box == pytest.approx([0.4, 0.4, 0.6, 0.6], abs=1e-5)
    assert (
        run_dir / "images" / "train" / "recording_002_part_001_0000002.jpg"
    ).is_file()

    assert data["kpt_shape"] == [2, 3] and data["flip_idx"] == [0, 1]
    assert data["names"] == {0: "subject"}
    assert record["task"] == "pose" and record["status"] == "complete"
    assert record["skeleton_digest"] == SKELETON.digest()
    assert record["source"]["kind"] == "keypoint_detect"
    assert record["source"]["detect_runs"] == {"recording_002": "kd_01"}
    assert record["source"]["tracking_runs"] == {"recording_002": "track_01"}
    assert record["source"]["match"] == {"min_inside": 0.5, "mask_grow": 0.05}
    assert [f["frame_idx"] for f in record["frames"]] == [0, 1, 2, 3]
    assert record["frames"][1]["track_ids"] == [0, 1]
    assert record["frames"][0]["source"] == "teacher"
    assert record["excluded"] == {
        "audit": 0,
        "unassigned": 1,
        "no_instance": 0,
        "undecodable": 0,
    }
    assert record["n_train"] == 4 and record["n_val"] == 0
    assert record["keypoint_dataset"]["score_cutoff"] == 0.4
    messages = [e.message for e in events]
    assert messages[0] == "reading teacher rows" and messages[-1] == "complete"


def test_the_teacher_assigns_instances_by_mask_not_by_detection(tracked, monkeypatch):
    """§12 / §26 *Assignment*: a teacher's instances are given to the tracking run's subjects
    by the share of their points inside each subject's mask (``pose_match``), never by the
    detection they were placed on. Frame 0's only instance lies outside every mask, so the
    subject there is taught its box and no visible point; on frame 1 each detection's pose
    lies on the *other* animal, so each subject is taught the pose on its own body."""
    from detail.core.registries import KEYPOINT_MODELS
    from tests.pipeline.test_keypoint_detect import _FakeKeypointModel

    class Misplaced(_FakeKeypointModel):
        def predict_batch(self, frames):
            out = super().predict_batch(frames)
            for result in out:
                if result.frame_idx == 0:
                    result.points[:] = (0.95, 0.05)  # off every mask
                if result.frame_idx == 1:  # row 1 is subject 0's, row 2 subject 1's
                    result.points[0] = (0.7, 0.7)  # inside subject 1's mask
                    result.points[1] = (0.3, 0.3)  # inside subject 0's mask
            return out

    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp", Misplaced)
    it, recording = tracked
    _run(it, recording, _config())
    run_dir, _record, _data, _events = _label(it, _job(it))
    labels = _labels(run_dir)

    assert len(labels[0]) == 1, "the subject on frame 0 is still taught"
    assert (
        labels[0][0][5:] == ["0"] * 6
    ), "an instance no subject took is taught to none"
    assert float(labels[0][0][3]) > 0, "its box is taught"
    track_0, track_1 = labels[1]
    assert track_0[5:8] == ["0.300000", "0.300000", "2"]
    assert track_1[5:8] == ["0.700000", "0.700000", "2"]
    assert [float(v) for v in track_0[1:3]] == pytest.approx([0.3, 0.3]), "box order"
    raw = np.load(run_dir / keypoint_dataset.TEACHER_OUTPUT)
    first = np.flatnonzero(raw["frame"] == 0)
    assert np.isnan(raw["points"][first]).all(), "teacher.npz keeps the gap as NaN"


def test_a_score_below_the_cutoff_is_taught_as_not_visible(taught):
    it, _recording = taught
    run_dir, _record, _data, _events = _label(it, _job(it, score_cutoff=0.6))
    for lines in _labels(run_dir).values():
        for tokens in lines:
            assert tokens[5:] == ["0", "0", "0", "0", "0", "0"]
            assert float(tokens[3]) > 0, "the box is still taught"


def test_frames_are_shared_by_row_count_and_spread_in_time(taught):
    it, _recording = taught
    run_dir, record, _data, _events = _label(it, _job(it, n_frames=2))
    frames = sorted(_labels(run_dir))
    assert len(frames) == 2
    assert frames[0] in (0, 1) and frames[1] in (2, 3), "one per equal time slice"
    assert len(record["frames"]) == 2


def test_audit_frames_and_their_neighbours_are_never_used(taught):
    it, recording = taught
    from detail.core.examples import ExampleFrame
    from detail.storage import example_sets

    proj = it.experiment
    audit = example_sets.create_set(proj.dir, "au", SKELETON, purpose="audit")
    audit.frames.append(
        ExampleFrame(
            image="frames/x.png",
            recording_id="recording_002",
            tracking_run_id="track_01",
            track_id=0,
            part_id="part_001",
            frame_idx=2,
            t_ns=2000,
            bbox=(0.1, 0.1, 0.5, 0.5),
            round=1,
        )
    )
    example_sets.write_set(proj.dir, audit)

    run_dir, record, _d, _e = _label(it, _job(it, exclude_audit_s=0.0))
    assert sorted(_labels(run_dir)) == [0, 1, 3]
    assert record["excluded"]["audit"] == 1

    # the fixture's rows are 1 µs apart, so a 1.5 µs window reaches one neighbour each way
    run_dir, record, _d, _e = _label(
        it, _job(it, dataset_id="dataset_002", exclude_audit_s=1.5e-6)
    )
    assert sorted(_labels(run_dir)) == [0]
    assert record["excluded"]["audit"] == 3


def test_a_candidate_stride_skips_neighbouring_frames(taught):
    it, _recording = taught
    run_dir, _record, _data, _events = _label(it, _job(it, every_n_frames=2))
    assert sorted(_labels(run_dir)) == [0, 2]


def test_nothing_to_label_is_an_error_and_a_failed_dataset(taught):
    it, _recording = taught
    from detail.core.examples import ExampleFrame
    from detail.storage import example_sets

    audit = example_sets.create_set(it.experiment.dir, "au", SKELETON, purpose="audit")
    audit.frames.append(
        ExampleFrame(
            image="frames/x.png",
            recording_id="recording_002",
            tracking_run_id="track_01",
            track_id=0,
            part_id="part_001",
            frame_idx=2,
            t_ns=2000,
            bbox=(0.1, 0.1, 0.5, 0.5),
        )
    )
    example_sets.write_set(it.experiment.dir, audit)
    job = _job(it, exclude_audit_s=1.0)  # 1 s reaches every row of this 5 µs recording
    run_dir = keypoint_dataset.run_dir_for(it.experiment.dir, job)
    run_dir.mkdir(parents=True, exist_ok=True)
    from detail.storage import datasets as datasets_store

    datasets_store.write_dataset_yaml(
        run_dir, {"id": job.dataset_id, "status": "running"}
    )
    with pytest.raises(ValueError, match="no frame can be labeled"):
        keypoint_dataset.run(job, run_dir, lambda e: None, None)
    record = yaml.safe_load((run_dir / "dataset.yaml").read_text(encoding="utf-8-sig"))
    assert record["status"] == "failed"


def test_an_example_frame_joins_with_the_users_clicks():
    """The hand marks the teacher was built from are labels too, but only when the tracking
    says the marked subject was alone on the frame."""
    skeleton = SKELETON
    marks = [
        {"target": "nose", "kind": "point", "x": 0.2, "y": 0.3, "visible": True},
        {"target": "tail", "kind": "point", "visible": False},
    ]
    frame = {
        "image": "frames/a.png",
        "recording_id": "recording_002",
        "tracking_run_id": "track_01",
        "track_id": 0,
        "part_id": "part_001",
        "frame_idx": 7,
        "t_ns": 7000,
        "bbox": [0.1, 0.1, 0.5, 0.5],
        "marks": marks,
    }
    gathered = {
        "sources": {
            "recording_002": {
                "example_set": {
                    "id": "ex_01",
                    "marks": [frame, {**frame, "frame_idx": 8}],
                },
                "boxes": {
                    ("part_001", 7): {0: (0.1, 0.1, 0.5, 0.5)},
                    ("part_001", 8): {0: (0.1, 0.1, 0.5, 0.5), 1: (0.6, 0.6, 0.9, 0.9)},
                },
            }
        }
    }
    out = keypoint_dataset._example_frames(gathered, skeleton, None)
    assert [e["frame_idx"] for e in out] == [7], "frame 8 has an unmarked second animal"
    bbox, points, scores, *_candidates = out[0]["tracks"][0]
    assert tuple(bbox) == (0.1, 0.1, 0.5, 0.5)
    np.testing.assert_allclose(points[0], (0.2, 0.3))
    assert np.isnan(points[1]).all() and scores[0] == 1.0
    assert keypoint_dataset.label_line(bbox, points, scores, 0.5).split()[5:] == [
        "0.200000",
        "0.300000",
        "2",
        "0",
        "0",
        "0",
    ]


# --- the patch-method teacher (kind: model) ------------------------------------------ #
@pytest.fixture
def patch_teacher(tracked, monkeypatch):
    """The tracked recording, a real example set marked on frame 0 (track 0 alone), and a
    learning fake registered as the patch method: ``dinov3_match`` is an external inference
    call, the one legitimate stand-in. It reads masks and places each subject's points on its
    box corners at score 0.5."""
    from detail.core.examples import ExampleMark
    from detail.core.registries import KEYPOINT_MODELS
    from tests.pipeline.test_keypoint_detect import _FakeKeypointModel

    seen = {"masks": [], "devices": []}

    class Learner(_FakeKeypointModel):
        model_id = "fake_learner"
        needs_examples = True

        def __init__(
            self,
            variant=None,
            *,
            examples=None,
            example_images=None,
            crop_px=None,
            decode=None,
            side_rule=None,
        ):
            super().__init__(variant, crop_px=crop_px)
            self.examples = examples

        def load(self, device):
            seen["devices"].append(device)
            super().load(device)

        def predict_batch(self, frames):
            seen["masks"].extend(sf.masks is not None for sf in frames)
            return super().predict_batch(frames)

    monkeypatch.setitem(KEYPOINT_MODELS, "fake_learner", Learner)
    it, recording = tracked
    proj = it.experiment
    made = proj.create_example_set("dots", SKELETON)
    proj.add_example_frame(made.id, recording.id, "track_01", "part_001", 0, 0)
    proj.set_example_marks(
        made.id,
        0,
        [
            ExampleMark(target="nose", x=0.2, y=0.2),
            ExampleMark(target="tail", x=0.4, y=0.4),
        ],
    )
    return it, recording, made.id, seen


def _patch_job(it, set_id, **over):
    from detail.api import keypoint_distill

    cfg, teacher, cutoff, compute = keypoint_distill.plan_keypoint_dataset(
        it.experiment,
        "rats",
        n_frames=over.pop("n_frames", 4),
        val_split=0.0,
        score_cutoff=0.4,
        exclude_audit_s=0.0,
        example_set=set_id,
        tracking=["recording_002"],
        model_id="fake_learner",
    )
    return _job(
        it,
        teacher=teacher,
        compute=compute,
        n_frames=cfg.n_frames,
        score_cutoff=cutoff,
        **over,
    )


def test_the_patch_method_labels_the_sampled_frames_itself(patch_teacher):
    it, _recording, set_id, seen = patch_teacher
    run_dir, record, data, events = _label(it, _patch_job(it, set_id))

    assert record["source"]["kind"] == "model"
    assert record["source"]["model_id"] == "fake_learner"
    assert record["source"]["example_set"]["id"] == set_id
    assert record["source"]["tracking_runs"] == {"recording_002": "track_01"}
    assert record["excluded"] == {"audit": 0, "unassigned": 1, "undecodable": 0}
    assert (record["n_requested"], record["n_from_teacher"]) == (4, 4)
    assert data["kpt_shape"] == [2, 3]
    labels = _labels(run_dir)
    assert sorted(labels) == [0, 1, 2, 3], "every complete frame; frame 4 is unassigned"
    assert len(labels[1]) == 2 and len(labels[0]) == 1, "one line per tracked animal"
    nose = labels[0][0][5:8]
    assert nose == ["0.100000", "0.100000", "2"], "the method's own placement, visible"
    assert seen["devices"] == ["cpu"] and seen["masks"] and all(seen["masks"])
    phases = [e.message for e in events]
    for phase in ("reading tracked frames", "loading fake_learner", "labeling frames"):
        assert phase in phases, phase
    assert phases.index("loading fake_learner") < phases.index("labeling frames")
    assert phases[-1] == "complete"


def test_a_frame_that_cannot_be_decoded_is_counted_not_lost(patch_teacher, monkeypatch):
    it, _recording, set_id, _seen = patch_teacher
    real = keypoint_dataset._decode

    def decode(source, part_id, frame_idx):
        return None if frame_idx == 2 else real(source, part_id, frame_idx)

    monkeypatch.setattr(keypoint_dataset, "_decode", decode)
    run_dir, record, _data, _events = _label(it, _patch_job(it, set_id))
    assert record["excluded"]["undecodable"] == 1
    assert record["n_from_teacher"] == 3 and 2 not in _labels(run_dir)


def test_the_split_is_drawn_over_the_frames_written(patch_teacher, monkeypatch):
    """A frame that could not be decoded never holds a validation slot: the split is drawn
    once the frames are written (`_dataset_common.StagedExamples`), so the realised share is
    the configured one, and each frame's recorded split is where its files are."""
    it, _recording, set_id, _seen = patch_teacher
    real = keypoint_dataset._decode
    skipped = {1, 3}

    def decode(source, part_id, frame_idx):
        return None if frame_idx in skipped else real(source, part_id, frame_idx)

    monkeypatch.setattr(keypoint_dataset, "_decode", decode)
    run_dir, record, _data, _events = _label(it, _patch_job(it, set_id, val_split=0.5))
    frames = record["frames"]
    assert len(frames) == 2
    assert record["n_val"] == round(len(frames) * 0.5) == 1
    for frame in frames:
        stem = f"{frame['recording_id']}_{frame['part_id']}_{frame['frame_idx']:07d}"
        assert (run_dir / "labels" / frame["split"] / f"{stem}.txt").is_file()
    assert not (run_dir / "staging").exists()


def test_the_patch_teacher_refuses_what_it_cannot_resolve(patch_teacher):
    from detail.api import keypoint_distill

    it, _recording, set_id, _seen = patch_teacher
    proj = it.experiment
    plan = keypoint_distill.plan_keypoint_dataset
    patch = dict(n_frames=4, example_set=set_id, tracking=["recording_002"])
    with pytest.raises(ValueError, match="give one teacher"):
        plan(proj, "r", ["recording_002"], **patch)
    with pytest.raises(ValueError, match="give one teacher"):
        plan(proj, "r", n_frames=4)
    with pytest.raises(ValueError, match="'fake_kp' is not one"):
        plan(proj, "r", model_id="fake_kp", **patch)
    with pytest.raises(ValueError, match="needs the tracking runs"):
        plan(proj, "r", n_frames=4, example_set=set_id, model_id="fake_learner")
    with pytest.raises(ValueError, match="no complete tracking run 'track_09'"):
        plan(
            proj,
            "r",
            n_frames=4,
            example_set=set_id,
            tracking={"recording_002": "track_09"},
            model_id="fake_learner",
        )
    with pytest.raises(ValueError, match="fake_learner sets no default score cut-off"):
        plan(proj, "r", model_id="fake_learner", **patch)
    _cfg, teacher, cutoff, compute = plan(
        proj, "r", model_id="fake_learner", score_cutoff=0.3, **patch
    )
    assert teacher["example_set"]["id"] == set_id and teacher["example_set"]["marks"]
    assert teacher["skeleton"] == SKELETON.to_dict() and cutoff == 0.3
    assert compute.backend == "cpu"


def test_the_teachers_raw_output_is_kept_beside_the_labels(taught, patch_teacher):
    """``teacher.npz`` holds what each teacher gave for every labelled (frame, track), before
    the cut-off, so a cut-off can be re-applied without running the teacher again."""
    it, _recording, set_id, _seen = patch_teacher
    for job in (
        _job(it),
        _patch_job(it, set_id, job_id="dataset_002", dataset_id="dataset_002"),
    ):
        run_dir, record, _data, _events = _label(it, job)
        raw = np.load(run_dir / keypoint_dataset.TEACHER_OUTPUT)
        frames = record["frames"]
        pairs = {(int(f), int(t)) for f, t in zip(raw["frame"], raw["track_id"])}
        expected = {
            (i, int(t)) for i, frame in enumerate(frames) for t in frame["track_ids"]
        }
        assert pairs == expected, "one row per labelled (frame, track)"
        assert raw["points"].shape == (len(raw["frame"]), 2, 2)
        assert raw["point_scores"].shape == (len(raw["frame"]), 2)
        assert raw["candidates"].shape[:2] == (len(raw["frame"]), 2)
        assert raw["candidates"].shape[2] >= 1, "both teachers keep their candidates"
        teacher_rows = [
            i for i, f in enumerate(raw["frame"]) if frames[f]["source"] == "teacher"
        ]
        assert np.isfinite(raw["point_scores"][teacher_rows]).any()
