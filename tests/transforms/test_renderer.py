import numpy as np

from detail.core.dataclasses import RenderContext
from detail.transforms import basic  # noqa: F401 — registers the transforms
from detail.transforms.renderer import MaskRenderer

H, W = 20, 30


def _ctx():
    frame = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), bool)
    mask[5:15, 10:20] = True
    return RenderContext(
        frame=frame,
        mask=mask,
        bbox=(0.1, 0.2, 0.5, 0.6),
        image=None,
        frame_idx=0,
        t_ns=0,
        class_name="mouse",
        confidence=0.9,
        object_id=0,
        aux={},
    )


def test_render_binary_pipeline():
    spec = [
        {"type": "init_binary_mask"},
        {"type": "pad_to_square"},
        {"type": "resize", "size": 224},
    ]
    image = MaskRenderer.from_spec(spec).render(_ctx())
    assert image.shape == (224, 224, 3)


def test_render_masked_rgb_pipeline_shape():
    spec = [
        {"type": "init_binary_mask"},
        {"type": "crop_to_bbox", "padding": 0.1},
        {"type": "pad_to_square"},
        {"type": "resize", "size": 64},
    ]
    image = MaskRenderer.from_spec(spec).render(_ctx())
    assert image.shape == (64, 64, 3)


def test_spec_round_trip():
    spec = [
        {"type": "init_binary_mask"},
        {"type": "crop_to_bbox", "padding": 0.1},
        {"type": "pad_to_square"},
        {"type": "resize", "size": 224},
    ]
    assert MaskRenderer.from_spec(spec).to_spec() == spec


def test_render_direct_construction():
    renderer = MaskRenderer([basic.InitBinaryMask(), basic.Resize(size=64)])
    assert renderer.render(_ctx()).shape == (64, 64, 3)


def test_needs_frame_false_for_mask_only_specs():
    for spec in (
        [{"type": "init_binary_mask"}, {"type": "pad_to_square"}],
        [
            {"type": "crop_to_bbox"},
            {"type": "init_binary_mask"},
            {"type": "center_in_square", "size": 40},
        ],
    ):
        assert MaskRenderer.from_spec(spec).needs_frame is False


def test_needs_frame_true_when_apply_mask_present():
    spec = [
        {"type": "crop_to_bbox"},
        {"type": "apply_mask", "background": 0},
        {"type": "resize", "size": 64},
    ]
    assert MaskRenderer.from_spec(spec).needs_frame is True


def test_crop_first_equals_crop_after_for_silhouette():
    # The efficient crop-first order must be output-identical to the old crop-after
    # order (crop commutes with mask→binary-image), just cheaper (§14).
    crop_first = MaskRenderer.from_spec(
        [
            {"type": "crop_to_bbox"},
            {"type": "init_binary_mask"},
            {"type": "center_in_square", "size": 40},
        ]
    ).render(_ctx())
    crop_after = MaskRenderer.from_spec(
        [
            {"type": "init_binary_mask"},
            {"type": "crop_to_bbox"},
            {"type": "center_in_square", "size": 40},
        ]
    ).render(_ctx())
    assert np.array_equal(crop_first, crop_after)


def test_crop_first_equals_crop_after_for_masked_color_at_equal_res():
    # The masked-color reorder is pixel-identical too *at the production invariant* —
    # frame and mask share a resolution (_ctx uses one size for both), since detection
    # emits frame-resolution masks (§14). Guards the equal-resolution assumption the
    # crop-first order relies on.
    crop_first = MaskRenderer.from_spec(
        [
            {"type": "crop_to_bbox"},
            {"type": "apply_mask", "background": 0},
            {"type": "center_in_square", "size": 40},
        ]
    ).render(_ctx())
    crop_after = MaskRenderer.from_spec(
        [
            {"type": "apply_mask", "background": 0},
            {"type": "crop_to_bbox"},
            {"type": "center_in_square", "size": 40},
        ]
    ).render(_ctx())
    assert np.array_equal(crop_first, crop_after)
