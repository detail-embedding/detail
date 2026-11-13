"""SuperAnimal, run natively (§26): the contract, the geometry, and agreement with deeplabcut.

The re-implementation is only worth having if it *is* the model, so the geometry is held to
deeplabcut 3.0.2's own answers (``make_superanimal_fixture.py``), which need no weights and run
every time: its crop offsets, scales and pixels for boxes inside and off every edge, and its
heatmap decoder on a seeded heatmap. A wrong crop convention or stride fails those.

The whole-model comparison — crop -> HRNet -> argmax -> back-projection against deeplabcut's
predictions on the same frame and box — needs the weights, and tests never read the machine's
model cache (``conftest._isolate_model_cache``) or download (``test_no_implicit_network``). It is
opt-in: point ``DETAIL_TEST_SUPERANIMAL_WEIGHTS`` at a folder holding the two ``.pt`` files.
"""

import hashlib
import importlib.util
import json
import os
import pathlib

import cv2
import numpy as np
import pytest

from detail.core.installed import is_importable
from detail.core.keypoints import InstanceFrame
from detail.core.registries import KEYPOINT_MODELS
from detail.models.superanimal import (
    INPUT_PX,
    QUADRUPED_SKELETON,
    STRIDE,
    TOPVIEWMOUSE_SKELETON,
    SuperAnimalQuadruped,
    SuperAnimalTopViewMouse,
    crop_with_context,
    decode_heatmaps,
)

DATA = pathlib.Path(__file__).resolve().parent / "data"
ANSWER = json.loads((DATA / "superanimal_dlc302.json").read_text(encoding="utf-8-sig"))


def test_registered_with_their_repos_and_manifests():
    for model_id, cls, repo, weights in (
        (
            "superanimal_quadruped",
            SuperAnimalQuadruped,
            "mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped",
            "superanimal_quadruped_hrnet_w32.pt",
        ),
        (
            "superanimal_topviewmouse",
            SuperAnimalTopViewMouse,
            "mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse",
            "superanimal_topviewmouse_hrnet_w32.pt",
        ),
    ):
        assert KEYPOINT_MODELS[model_id] is cls
        model = cls()
        assert model.hf_repo == repo
        assert model.gated is False
        assert model.variant == "hrnet_w32"
        assert model.hf_files == [
            weights
        ]  # what is_installed verifies and install fetches
        assert model.needs_mask is False and model.needs_examples is False
        assert (
            model.n_candidates == 1 and model.default_track_method == "despike_smooth"
        )
        assert "non-commercial" in model.license.name
        assert model.license.url.endswith(repo)


def test_only_supported_sizes_are_offered():
    """The repos list more sizes than the native code runs; a variant is offered only once
    the code supports it and a known-answer fixture pins it (§26)."""
    for cls in (SuperAnimalQuadruped, SuperAnimalTopViewMouse):
        assert cls.variants() == ["hrnet_w32"]
        with pytest.raises(ValueError, match="unknown variant"):
            cls("rtmpose_x")


@pytest.mark.parametrize(
    "model_id,skeleton",
    [
        ("superanimal_quadruped", QUADRUPED_SKELETON),
        ("superanimal_topviewmouse", TOPVIEWMOUSE_SKELETON),
    ],
)
def test_keypoint_names_are_deeplabcuts_own_in_its_order(model_id, skeleton):
    """The checkpoints carry no names; the output columns mean what deeplabcut's project
    config says, in its order — taken from deeplabcut itself when the fixture was made.
    """
    assert list(skeleton.names) == ANSWER["models"][model_id]["bodyparts"]


@pytest.mark.parametrize("skeleton", [QUADRUPED_SKELETON, TOPVIEWMOUSE_SKELETON])
def test_pairs_are_declared_and_consistent(skeleton):
    skeleton.validate()
    assert skeleton.pairs
    for left, right in skeleton.pairs:
        assert "left" in left and "right" in right
        assert left.replace("left", "right") == right


def _subject(frame, bbox, detection_row=0):
    return InstanceFrame(frame_idx=0, t_ns=0, frame=frame, boxes={detection_row: bbox})


def test_crop_with_context_widens_to_a_square_and_maps_back():
    frame = np.zeros((200, 300, 3), np.uint8)
    crop, (ox, oy), (sx, sy) = crop_with_context(frame, (100.0, 50.0, 80.0, 40.0))
    assert crop.shape == (INPUT_PX, INPUT_PX, 3)
    # An 80x40 box becomes an 80x80 square around the same centre (140, 70).
    assert (ox, oy) == (100, 30)
    assert sx == pytest.approx(80 / INPUT_PX) and sy == pytest.approx(80 / INPUT_PX)
    assert ox + (INPUT_PX / 2) * sx == pytest.approx(140)


def test_crop_with_context_zero_fills_outside_the_frame():
    frame = np.full((100, 100, 3), 200, np.uint8)
    crop, (ox, oy), (sx, sy) = crop_with_context(frame, (-20.0, -20.0, 60.0, 60.0))
    assert (ox, oy) == (-20, -20)  # the offset still names the square's true corner
    assert sx == pytest.approx(60 / INPUT_PX)
    assert crop[0, 0].tolist() == [0, 0, 0]  # outside the frame is black
    assert crop[-1, -1].tolist() == [200, 200, 200]


def _fixture_frame() -> np.ndarray:
    return cv2.cvtColor(
        cv2.imread(str(DATA / "superanimal_frame.png")), cv2.COLOR_BGR2RGB
    )


@pytest.mark.parametrize("case", ANSWER["crops"], ids=lambda c: str(c["box_xywh"]))
def test_crop_is_deeplabcuts_crop(case):
    """Offset, scale and every pixel as deeplabcut's ``top_down_crop`` produced them."""
    crop, offset, scale = crop_with_context(_fixture_frame(), tuple(case["box_xywh"]))
    assert list(offset) == case["offset"]
    assert list(scale) == pytest.approx(case["scale"], abs=1e-12)
    assert hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest() == (
        case["crop_sha256"]
    )


def test_decode_is_deeplabcuts_heatmap_predictor():
    spec = ANSWER["decode"]
    heat = (
        np.random.default_rng(spec["seed"]).random(
            tuple(spec["shape"]), dtype=np.float32
        )
        * spec["scale"]
    )
    xy, score = decode_heatmaps(heat)
    expected = np.asarray(spec["poses"])  # (n, K, 3): x, y, score in crop pixels
    np.testing.assert_allclose(xy, expected[..., :2])
    np.testing.assert_allclose(score, expected[..., 2], atol=1e-7)


def test_decode_heatmaps_takes_each_peak_at_its_cell_centre():
    heat = np.zeros((1, 2, 64, 64), np.float32)
    heat[0, 0, 10, 20] = 0.8
    heat[0, 1, 0, 63] = 3.0  # above 1: the score is clipped, as deeplabcut clips it
    xy, score = decode_heatmaps(heat)
    assert xy[0, 0].tolist() == [20 * STRIDE + STRIDE / 2, 10 * STRIDE + STRIDE / 2]
    assert xy[0, 1].tolist() == [63 * STRIDE + STRIDE / 2, STRIDE / 2]
    assert score[0].tolist() == pytest.approx([0.8, 1.0])


def test_predict_before_load_is_refused():
    with pytest.raises(RuntimeError, match="load"):
        SuperAnimalQuadruped().predict_batch(
            [_subject(np.zeros((8, 8, 3), np.uint8), (0, 0, 1, 1))]
        )


def test_readiness_names_a_missing_backend(monkeypatch):
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        return None if name == "timm" else real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    status = SuperAnimalTopViewMouse().readiness()
    assert status.ready is False
    assert status.reason == "missing_dependency"
    assert "timm" in status.message


WEIGHTS_ENV = "DETAIL_TEST_SUPERANIMAL_WEIGHTS"


@pytest.mark.parametrize("cls", [SuperAnimalQuadruped, SuperAnimalTopViewMouse])
def test_matches_deeplabcut_on_the_cpu(cls, monkeypatch):
    """Every keypoint where deeplabcut 3.0.2 put it, on the CPU, from the same frame and box."""
    weights_dir = os.environ.get(WEIGHTS_ENV)
    if not weights_dir or not (is_importable("torch") and is_importable("timm")):
        pytest.skip(
            f"opt-in: set {WEIGHTS_ENV} to a folder with the SuperAnimal .pt files"
        )
    model = cls()
    weights = pathlib.Path(weights_dir) / model.weight
    assert weights.is_file(), f"{WEIGHTS_ENV} has no {model.weight}"
    # The one seam the test replaces: where load() finds the file. Everything after is real.
    monkeypatch.setattr(model, "local_file", lambda name: weights)
    expected = ANSWER["models"][cls.model_id]
    frame = _fixture_frame()
    model.load("cpu")
    (result,) = model.predict_batch(
        [_subject(frame, tuple(ANSWER["box_norm"]), detection_row=7)]
    )
    # the detection it was placed in; no subject until keypoint track assigns one (§26)
    assert result.detection_row.tolist() == [7]
    xy, score = result.points[0], result.point_scores[0]
    assert xy.dtype == np.float32 and xy.shape == (len(model.skeleton()), 2)
    np.testing.assert_allclose(xy, np.asarray(expected["xy_norm"]), atol=1e-6)
    np.testing.assert_allclose(score, np.asarray(expected["score"]), atol=1e-4)
