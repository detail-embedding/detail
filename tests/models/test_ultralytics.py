"""``models/_ultralytics.py::SplitPredict``: Ultralytics' own ``predict``, cut at its stages.

A real, locally built checkpoint (no download) run through Ultralytics, so this file is in the
``slow`` tier. The segmentation model's check is in ``tests/models/test_trained_yolo.py``; this
is the pose student's, through ``TrainedYoloPose`` (§26 *Execution*).
"""

import numpy as np
import pytest

from detail.core.keypoints import InstanceFrame
from detail.core.skeleton import Skeleton
from detail.models import yolo_pose
from detail.models._ultralytics import import_ultralytics
from detail.models.yolo_pose import TrainedYoloPose


def _detecting_pose_weights(path):
    """A stand-in pose checkpoint (17 points, Ultralytics' default) whose random weights *do*
    detect: the class logits are lifted from Ultralytics' prior of about -10, so nearly every
    anchor scores and the comparison below has hundreds of poses to disagree on. A net built
    from the YAML lacks the ``kpt_shape`` attribute a trained checkpoint carries, so it is set.
    """
    import torch

    import_ultralytics()
    from ultralytics import YOLO

    torch.manual_seed(0)
    built = YOLO("yolo11n-pose.yaml")
    for branch in built.model.model[-1].cv3:
        branch[-1].bias.data += 12.0
    built.model.kpt_shape = built.model.yaml["kpt_shape"]
    built.save(str(path))
    return path


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predict_batch_is_ultralytics_predict_split_at_its_own_stages(tmp_path):
    # Class guarded: **the split predict (§26 *Execution*: letterbox on the prep thread, the
    # forward on the inference thread) returns something other than Ultralytics' own
    # ``predict``** — a skipped /255, a lost BGR flip, a letterbox padded differently — or an
    # Ultralytics release moves a stage the split calls by hand.
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    from detail.core.visualize import rgb_to_bgr

    weights = _detecting_pose_weights(tmp_path / "detecting-pose.pt")
    skeleton = Skeleton(names=tuple(f"p{i}" for i in range(17)))
    model = TrainedYoloPose(weights, skeleton)
    model.load("cpu")
    model.set_min_pose_confidence(0.05)
    reference = YOLO(str(weights))
    rng = np.random.default_rng(0)

    def frame(h, w, idx):
        image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        return InstanceFrame(frame_idx=idx, t_ns=idx * 10, frame=image, boxes={})

    batches = [
        [frame(96, 128, i) for i in range(3)],
        [frame(96, 128, 3)],
        [
            frame(96, 128, 4),
            frame(64, 160, 5),
        ],  # mixed shapes: the full-square letterbox
    ]
    found = 0
    for frames in batches:
        got = model.predict_batch(frames)
        want = reference.predict(
            [rgb_to_bgr(f.frame) for f in frames],
            conf=0.05,
            device="cpu",
            verbose=False,
        )
        assert len(got) == len(want) == len(frames)
        for f, result, ultra in zip(frames, got, want):
            expected = model._poses(f, yolo_pose._host_pose_arrays(ultra))
            for field in (
                "bbox",
                "confidence",
                "detection_row",
                "points",
                "point_scores",
            ):
                np.testing.assert_array_equal(
                    getattr(result, field), getattr(expected, field), err_msg=field
                )
            assert (result.frame_idx, result.t_ns, result.n) == (
                expected.frame_idx,
                expected.t_ns,
                expected.n,
            )
            found += result.n
    assert found > 100  # the comparison saw poses, not two empty lists

    # Still one batched forward pass per call.
    bgr = [rgb_to_bgr(f.frame) for f in batches[0]]
    predictor = model._split.predictor(bgr)
    forwards = []
    inference = predictor.inference

    def counted(im, *args, **kwargs):
        forwards.append(im.shape[0])
        return inference(im, *args, **kwargs)

    predictor.inference = counted
    model.predict_batch(batches[0])
    assert forwards == [3]
