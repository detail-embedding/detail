import numpy as np
import pytest

from detail.core.dataclasses import RenderContext
from detail.transforms import basic  # noqa: F401 — registers transforms
from detail.transforms import constructors
from detail.transforms.renderer import MaskRenderer

H, W = 40, 60


def _ctx() -> RenderContext:
    frame = np.full((H, W, 3), 200, np.uint8)
    mask = np.zeros((H, W), bool)
    mask[10:30, 20:40] = True
    return RenderContext(
        frame=frame,
        mask=mask,
        bbox=(0.2, 0.1, 0.8, 0.9),
        image=None,
        frame_idx=0,
        t_ns=0,
        class_name="mouse",
        confidence=0.9,
        object_id=0,
        aux={},
    )


def test_registry_lists_all_constructors():
    assert set(constructors.CONSTRUCTORS) == {
        "silhouette_centered",
        "masked_color_centered",
        "masked_color_centered_contrast",
        "full_frame",
    }
    assert constructors.DEFAULT_CONSTRUCTOR in constructors.CONSTRUCTORS


def test_new_constructors_flags():
    recipe = constructors.CONSTRUCTORS["masked_color_centered_contrast"]
    assert recipe.needs_canvas is True
    assert recipe.needs_bg_color is True


def test_contrast_constructor_requires_bg_color_and_fills_it():
    with pytest.raises(ValueError, match="canvas_px"):
        constructors.masked_color_centered_contrast()
    with pytest.raises(ValueError, match="bg_color"):
        constructors.masked_color_centered_contrast(canvas_px=50)
    spec = constructors.masked_color_centered_contrast(
        canvas_px=50, bg_color=[0, 255, 0]
    )
    assert spec[1] == {"type": "apply_mask", "background": [0, 255, 0]}
    assert spec[2] == {"type": "center_in_square", "size": 50, "fill": [0, 255, 0]}
    out = MaskRenderer.from_spec(spec).render(_ctx())
    assert out[0, 0].tolist() == [0, 255, 0]  # padding is the computed contrast color


def test_needs_canvas_flags():
    assert constructors.CONSTRUCTORS["silhouette_centered"].needs_canvas is True
    assert constructors.CONSTRUCTORS["masked_color_centered"].needs_canvas is True
    assert constructors.CONSTRUCTORS["full_frame"].needs_canvas is False


def test_centered_constructors_require_canvas_px():
    with pytest.raises(ValueError, match="canvas_px"):
        constructors.silhouette_centered()
    with pytest.raises(ValueError, match="canvas_px"):
        constructors.masked_color_centered()


def test_silhouette_centered_spec_uses_canvas():
    # crop-first: the mask is cropped before the binary image is built (§14).
    spec = constructors.silhouette_centered(canvas_px=50)
    assert spec == [
        {"type": "crop_to_bbox"},
        {"type": "init_binary_mask"},
        {"type": "center_in_square", "size": 50},
    ]


def test_masked_color_centered_spec_leads_with_crop():
    # crop-first: crop the frame+mask before apply_mask (§14).
    spec = constructors.masked_color_centered(canvas_px=50)
    assert spec[0] == {"type": "crop_to_bbox"}
    assert spec[1] == {"type": "apply_mask", "background": 0}
    assert spec[-1] == {"type": "center_in_square", "size": 50}


def test_full_frame_spec_ignores_mask_and_bbox():
    spec = constructors.full_frame(resize=224)
    assert spec == [
        {"type": "init_frame"},
        {"type": "pad_to_square"},
        {"type": "resize", "size": 224},
    ]


def test_no_mask_constructors_need_the_frame():
    assert MaskRenderer.from_spec(constructors.full_frame()).needs_frame is True


def _ctx_with_frame(frame: np.ndarray) -> RenderContext:
    ctx = _ctx()
    ctx.frame = frame
    return ctx


def test_all_specs_render_through_maskrenderer():
    specs = [
        constructors.silhouette_centered(canvas_px=50),
        constructors.masked_color_centered(canvas_px=50),
        constructors.full_frame(),
    ]
    for spec in specs:
        image = MaskRenderer.from_spec(spec).render(_ctx())
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.dtype == np.uint8


def test_masked_color_keeps_frame_pixels():
    image = MaskRenderer.from_spec(
        constructors.masked_color_centered(canvas_px=50)
    ).render(_ctx())
    assert (image == 200).any()  # real frame pixels survive (silhouette would be 255)
    assert not (image == 255).any()


def test_with_alignment_inserts_after_the_crop_and_before_the_canvas():
    from detail.transforms.constructors import (
        ALIGNABLE,
        CONSTRUCTORS,
        alignment_step,
        with_alignment,
    )

    spec = CONSTRUCTORS["masked_color_centered_contrast"].build(
        canvas_px=100, resize=224, bg_color=[0, 0, 255]
    )
    rotated = with_alignment(spec, "rotate")
    assert [s["type"] for s in rotated] == [
        "crop_to_bbox",
        "apply_mask",
        "align_rotate",
        "center_in_square",
        "resize",
    ]
    assert rotated[2]["fill"] == [
        0,
        0,
        255,
    ]  # the canvas fill reaches the turn's corners
    mirrored = with_alignment(spec, "mirror")
    assert mirrored[2] == {"type": "align_mirror"}
    assert alignment_step(rotated) == "rotate" and alignment_step(mirrored) == "mirror"
    assert alignment_step(spec) is None
    assert (
        with_alignment(spec, "none") == spec
        and with_alignment(spec, "none") is not spec
    )
    for key in ALIGNABLE:
        recipe = CONSTRUCTORS[key]
        built = recipe.build(
            canvas_px=100,
            resize=64,
            **({"bg_color": [1, 2, 3]} if recipe.needs_bg_color else {}),
        )
        assert alignment_step(with_alignment(built, "rotate")) == "rotate", key
    with pytest.raises(ValueError, match="crops to the subject"):
        with_alignment(CONSTRUCTORS["full_frame"].build(resize=224), "rotate")
    with pytest.raises(ValueError, match="align must be one of"):
        with_alignment(spec, "sideways")
