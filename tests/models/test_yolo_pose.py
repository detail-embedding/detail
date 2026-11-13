"""TrainedYoloPose: the distilled keypoint student (§12 *Distill keypoints*, §26).

The backend is stubbed at the one place it is used: a fake Ultralytics result (boxes +
keypoints) handed back through a stand-in for ``SplitPredict`` (``fake_split``), and a fake
``ultralytics`` module for ``introspect``. That ``SplitPredict`` itself is Ultralytics' own
``predict`` is ``tests/models/test_ultralytics.py``'s, on a real (locally built) pose
checkpoint.
What is asserted is the contract — every pose at or above ``min_pose_confidence`` and none
below, no subject assigned (that is ``keypoint_track``'s, tested with the pose matching), a
confidence of 0 is NaN, the skeleton's K is the student's K — not what a net predicts.
"""

import sys
import types

import numpy as np
import pytest

from detail.core.keypoints import InstanceFrame
from detail.core.skeleton import Skeleton
from detail.models import yolo_pose
from detail.models.yolo_pose import TrainedYoloPose

SK = Skeleton(
    names=("nose", "left_ear", "right_ear"), pairs=(("left_ear", "right_ear"),)
)


class _Tensor:
    """Enough of a torch tensor: ``.cpu()`` then array conversion."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self._values


class _Boxes:
    def __init__(self, xyxyn, conf):
        self.xyxyn = _Tensor(xyxyn)
        self.conf = _Tensor(conf)

    def __len__(self):
        return len(self.conf._values)


class _Keypoints:
    def __init__(self, xyn, conf):
        self.xyn = _Tensor(xyn)
        self.conf = None if conf is None else _Tensor(conf)


class _Result:
    def __init__(self, boxes, keypoints):
        self.boxes = boxes
        self.keypoints = keypoints


class _FakeYolo:
    def __init__(self, results):
        self._results = results
        self.calls = []
        self.frames = []

    def predict(self, frames, **kwargs):
        self.calls.append((len(frames), kwargs))
        self.frames.extend(frames)
        return self._results[: len(frames)]


class _FakeSplit:
    """``SplitPredict``'s two halves over a fake model: the whole ``predict`` in ``run``."""

    def __init__(self, model, **kwargs):
        self._model = model
        self._kwargs = kwargs

    def prepare(self, frames):
        return frames

    def run(self, prepared, frames):
        return self._model.predict(frames, **self._kwargs)


@pytest.fixture
def fake_split(monkeypatch):
    monkeypatch.setattr(yolo_pose, "SplitPredict", _FakeSplit)


def _frame(boxes: dict, idx=0) -> InstanceFrame:
    return InstanceFrame(
        frame_idx=idx,
        t_ns=idx * 1000,
        frame=np.zeros((40, 60, 3), np.uint8),
        boxes=boxes,
    )


def test_class_level_contract():
    assert TrainedYoloPose.needs_examples is False
    assert TrainedYoloPose.needs_mask is False
    assert TrainedYoloPose.per_detection is False  # the whole frame, every animal
    assert TrainedYoloPose.reports_confidence is True
    assert TrainedYoloPose.n_candidates == 1
    assert TrainedYoloPose.default_track_method == "despike_smooth"
    assert TrainedYoloPose.score_spec.meaning == "detector confidence"
    assert TrainedYoloPose.score_spec.default_cutoff == 0.5
    assert TrainedYoloPose.supported_backends == ["cpu", "mps", "cuda"]


def test_init_keeps_the_skeleton_and_validates_it(tmp_path):
    model = TrainedYoloPose(tmp_path / "best.pt", SK)
    assert model.skeleton() == SK
    assert model.variant is None
    with pytest.raises(ValueError, match="appears twice"):
        TrainedYoloPose(tmp_path / "best.pt", Skeleton(names=("a", "a")))


def test_predict_before_load_raises(tmp_path):
    model = TrainedYoloPose(tmp_path / "best.pt", SK)
    with pytest.raises(RuntimeError, match="load.. must be called"):
        model.predict_batch([_frame({})])
    assert model.predict_batch([]) == []  # an empty batch needs no model


def test_readiness_names_missing_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(yolo_pose, "is_importable", lambda m: True)
    status = TrainedYoloPose(tmp_path / "best.pt", SK).readiness()
    assert not status.ready and status.reason == "missing_weights"
    (tmp_path / "best.pt").write_bytes(b"")
    assert TrainedYoloPose(tmp_path / "best.pt", SK).readiness().ready
    monkeypatch.setattr(yolo_pose, "is_importable", lambda m: False)
    assert (
        TrainedYoloPose(tmp_path / "best.pt", SK).readiness().reason
        == "missing_dependency"
    )


def test_every_pose_at_or_above_the_threshold_is_returned_with_no_subject(
    tmp_path, fake_split
):
    """§26: the student returns every pose whose box confidence is at least
    ``min_pose_confidence`` (0.25 by default), each with its box and confidence and
    ``detection_row = −1``; none below; no assignment to a track."""
    model = TrainedYoloPose(tmp_path / "best.pt", SK)
    det_boxes = [
        [0.6, 0.1, 0.9, 0.5],
        [0.1, 0.1, 0.4, 0.5],
        [0.2, 0.6, 0.3, 0.9],
        [0.4, 0.4, 0.5, 0.5],
    ]
    xyn = [
        [[0.7, 0.2], [0.65, 0.3], [0.85, 0.3]],
        [[0.2, 0.2], [0.15, 0.3], [0.0, 0.0]],  # right_ear unseen: (0, 0) at conf 0
        [[0.25, 0.7], [0.22, 0.8], [0.28, 0.8]],
        [[0.45, 0.45], [0.42, 0.48], [0.48, 0.48]],
    ]
    kp_conf = [[0.9, 0.8, 0.7], [0.95, 0.6, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    conf = [
        0.9,
        0.8,
        0.25,
        0.2499,
    ]  # exactly at the threshold is kept; just below is not
    result = _Result(_Boxes(det_boxes, conf), _Keypoints(xyn, kp_conf))
    model._model = _FakeYolo([result])

    (fr,) = model.predict_batch([_frame({}, idx=7)])

    assert fr.frame_idx == 7 and fr.t_ns == 7000 and fr.n == 3
    assert fr.track_id is None  # identity is keypoint_track's
    assert fr.detection_row.tolist() == [-1, -1, -1]
    assert fr.confidence.tolist() == pytest.approx([0.9, 0.8, 0.25])
    np.testing.assert_allclose(fr.bbox, det_boxes[:3])
    assert fr.points.shape == (3, 3, 2) and fr.points.dtype == np.float32
    np.testing.assert_allclose(fr.points[0, 0], [0.7, 0.2])
    np.testing.assert_allclose(fr.points[1, 0], [0.2, 0.2])
    assert np.isnan(fr.points[1, 2]).all(), "a zero-confidence point is NaN, not (0, 0)"
    assert np.isnan(fr.point_scores[1, 2])
    assert fr.point_scores[0].tolist() == pytest.approx([0.9, 0.8, 0.7])
    assert model._model.calls[0][1]["conf"] == pytest.approx(0.25)


def test_the_runs_min_pose_confidence_reaches_the_predict_call(tmp_path, fake_split):
    model = TrainedYoloPose(tmp_path / "best.pt", SK)
    result = _Result(
        _Boxes([[0.1, 0.1, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9]], [0.9, 0.6]),
        _Keypoints(np.full((2, 3, 2), 0.3), np.full((2, 3), 0.9)),
    )
    model._model = _FakeYolo([result])
    model.set_min_pose_confidence(0.7)
    (fr,) = model.predict_batch([_frame({})])
    assert fr.confidence.tolist() == pytest.approx([0.9])
    assert model._model.calls[0][1]["conf"] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="not in"):
        model.set_min_pose_confidence(1.5)


def test_the_student_is_handed_bgr_the_order_it_was_trained_on(tmp_path, fake_split):
    """An ``InstanceFrame`` is RGB and Ultralytics reads a numpy array as BGR; the student was
    trained on images ``cv2.imwrite`` wrote, which are BGR. Handing it the RGB array swaps red
    and blue against training: nothing fails, the points are just placed on the wrong picture.
    """
    model = TrainedYoloPose(tmp_path / "best.pt", SK)
    model._model = _FakeYolo(
        [_Result(_Boxes(np.zeros((0, 4)), []), _Keypoints(np.zeros((0, 3, 2)), None))]
    )
    red = np.zeros((40, 60, 3), np.uint8)
    red[..., 0] = 255  # pure red, in the InstanceFrame's RGB order
    frame = InstanceFrame(frame_idx=0, t_ns=0, frame=red, boxes={})

    model.predict_batch([frame])

    (seen,) = model._model.frames
    assert seen[..., 2].min() == 255, "red belongs in channel 2 of a BGR array"
    assert seen[..., 0].max() == 0, "the student was handed RGB as if it were BGR"
    assert seen.flags["C_CONTIGUOUS"]


def test_an_empty_result_and_a_k_mismatch(tmp_path, fake_split):
    model = TrainedYoloPose(tmp_path / "best.pt", SK)
    model._model = _FakeYolo(
        [_Result(_Boxes(np.zeros((0, 4)), []), _Keypoints(np.zeros((0, 3, 2)), None))]
    )
    fr = model.predict_batch([_frame({})])[0]
    assert fr.n == 0 and fr.points.shape == (0, 3, 2) and fr.bbox.shape == (0, 4)

    model = TrainedYoloPose(tmp_path / "best.pt", SK)  # a fresh split for the new fake
    model._model = _FakeYolo(
        [
            _Result(
                _Boxes([[0.1, 0.1, 0.5, 0.5]], [0.9]),
                _Keypoints([[[0.2, 0.2], [0.3, 0.3]]], None),
            )
        ]
    )
    with pytest.raises(ValueError, match="places 2 points but its skeleton names 3"):
        model.predict_batch([_frame({})])


def test_introspect_reads_k_and_flip_idx_and_refuses_a_non_pose_checkpoint(
    tmp_path, monkeypatch
):
    class _Inner:
        yaml = {"kpt_shape": [3, 3], "flip_idx": [0, 2, 1]}
        kpt_shape = [3, 3]

    class _Yolo:
        def __init__(self, path):
            self.task = "pose" if "pose" in str(path) else "segment"
            self.model = _Inner()

    fake = types.ModuleType("ultralytics")
    fake.YOLO = _Yolo
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setattr(yolo_pose, "import_ultralytics", lambda: None)

    assert TrainedYoloPose.introspect(tmp_path / "pose.pt") == (3, [0, 2, 1])
    with pytest.raises(ValueError, match="not a pose model"):
        TrainedYoloPose.introspect(tmp_path / "seg.pt")


# --- student_input: crop (§12) ----------------------------------------------------- #
def test_a_crop_student_takes_the_animal_at_its_crops_centre_back_to_the_frame(
    tmp_path,
    fake_split,
):
    """Each detection is run on its own crop (BGR, like the whole frame); the detection covering
    the crop's centre is its, and its points go back through the crop's view."""
    from detail.models.patch_match.view import make_view

    model = TrainedYoloPose(tmp_path / "best.pt", SK, student_input="crop", crop_px=64)
    # in the crop: a neighbour at the edge, then the subject, centred
    det_boxes = [[0.0, 0.0, 0.2, 0.2], [0.3, 0.3, 0.7, 0.7]]
    xyn = [
        [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],
        [[0.5, 0.4], [0.4, 0.6], [0.6, 0.6]],
    ]
    result = _Result(
        _Boxes(det_boxes, [0.7, 0.9]), _Keypoints(xyn, [[0.9] * 3, [0.8] * 3])
    )
    model._model = _FakeYolo([result])
    box = (0.2, 0.25, 0.4, 0.75)
    red = np.zeros((40, 60, 3), np.uint8)
    red[..., 0] = 255

    (fr,) = model.predict_batch(
        [InstanceFrame(frame_idx=3, t_ns=3000, frame=red, boxes={5: box})]
    )

    assert model._model.calls[0][0] == 1  # one crop for the one detection
    (seen,) = model._model.frames
    assert seen.shape == (64, 64, 3)
    assert seen[32, 32, 2] == 255 and seen[32, 32, 0] == 0, "the crop is handed BGR"
    view = make_view(box, (60, 40), 64, 1)
    x, y = view.to_source(0.5 * 64, 0.4 * 64)
    np.testing.assert_allclose(fr.points[0, 0], [x / 60, y / 40], atol=1e-6)
    # the detection it was cut around; no subject until keypoint track assigns one (§26)
    assert fr.detection_row.tolist() == [5] and fr.confidence[0] == pytest.approx(0.9)


def test_a_crop_with_nothing_on_its_centre_places_no_points(tmp_path, fake_split):
    """A detection that does not cover the crop's centre is a neighbour reaching in, never
    this track."""
    model = TrainedYoloPose(tmp_path / "best.pt", SK, student_input="crop", crop_px=64)
    result = _Result(
        _Boxes([[0.0, 0.0, 0.2, 0.2]], [0.9]),
        _Keypoints([[[0.1, 0.1]] * 3], [[0.9] * 3]),
    )
    model._model = _FakeYolo([result])
    (fr,) = model.predict_batch([_frame({0: (0.2, 0.25, 0.4, 0.75)})])
    assert np.isnan(fr.points).all() and np.isnan(fr.confidence).all()


def test_a_student_is_run_the_way_its_record_says_it_was_trained():
    from detail.models.resolve import student_input_of

    assert student_input_of({}) == {"student_input": "whole_frame", "crop_px": 448}
    assert student_input_of({"student_input": "crop", "student_crop_px": 320}) == {
        "student_input": "crop",
        "crop_px": 320,
    }
    # a student the stage before §26 trained: a crop_px and no student_input
    assert student_input_of({"crop_px": 320}) == {
        "student_input": "crop",
        "crop_px": 320,
    }
    with pytest.raises(ValueError, match="whole_frame or crop"):
        TrainedYoloPose("x.pt", SK, student_input="sideways")


def test_a_catalog_student_from_before_the_rebuild_runs_on_its_crop(tmp_path):
    """The stage before §26 promoted each distilled student into the catalog, trained on
    crops, with its old skeleton shape; it resolves to a crop-mode student with that skeleton
    in §26's shape."""
    from detail.models import imported as imported_store
    from detail.models.resolve import keypoint_method_class
    from detail.models.resolve import keypoint_students
    from detail.models.resolve import resolve_keypoint_model
    from detail.storage import models as models_store

    model_dir = imported_store.model_dir("old_rat_pose")
    (model_dir / "weights").mkdir(parents=True)
    (model_dir / "weights" / "best.pt").write_bytes(b"weights")
    models_store.write_model_yaml(
        model_dir,
        {
            "id": "old_rat_pose",
            "role": "keypoints",
            "task": "pose",
            "model_family": "yolo_pose",
            "crop_px": 320,
            "weights": {"best": "weights/best.pt"},
            "skeleton": {
                "keypoints": [
                    {"name": "nose", "side": "midline", "pair": None},
                    {"name": "left_ear", "side": "left", "pair": "right_ear"},
                    {"name": "right_ear", "side": "right", "pair": "left_ear"},
                ],
                "edges": [],
            },
            "status": "complete",
        },
    )
    assert {"id": "old_rat_pose", "source": "imported"}.items() <= next(
        s for s in keypoint_students(None) if s["id"] == "old_rat_pose"
    ).items()
    assert keypoint_method_class(None, "old_rat_pose") is TrainedYoloPose
    student = resolve_keypoint_model("old_rat_pose")
    assert isinstance(student, TrainedYoloPose)
    assert student.resolved() == {"student_input": "crop", "student_crop_px": 320}
    assert student.skeleton().pairs == (("left_ear", "right_ear"),)
