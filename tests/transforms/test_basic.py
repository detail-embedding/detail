import cv2
import numpy as np
import pytest

from detail.core.dataclasses import RenderContext
from detail.core.registries import TRANSFORMS
from detail.transforms import basic

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


def test_transforms_registered():
    for name in ("init_binary_mask", "crop_to_bbox", "pad_to_square", "resize"):
        assert name in TRANSFORMS


def test_init_binary_mask():
    ctx = basic.InitBinaryMask().apply(_ctx())
    assert ctx.image.shape == (H, W, 3)
    assert ctx.image[10, 15].tolist() == [255, 255, 255]  # inside mask
    assert ctx.image[0, 0].tolist() == [0, 0, 0]  # outside


def test_crop_to_bbox():
    ctx = basic.CropToBbox(padding=0.0).apply(basic.InitBinaryMask().apply(_ctx()))
    # bbox (0.1,0.2,0.5,0.6) on (20,30) -> x 3..15, y 4..12 -> (8, 12, 3)
    assert ctx.image.shape == (8, 12, 3)


def test_crop_to_bbox_none_bbox_is_noop():
    ctx = _ctx()
    ctx.bbox = None
    ctx = basic.InitBinaryMask().apply(ctx)
    before = ctx.image.shape
    ctx = basic.CropToBbox().apply(ctx)
    assert ctx.image.shape == before


def test_crop_to_bbox_crops_source_when_no_image():
    # crop-first: with no image yet, crop_to_bbox crops the mask + frame (§14).
    ctx = _ctx()  # image None, mask/frame (20, 30), bbox (0.1, 0.2, 0.5, 0.6)
    ctx = basic.CropToBbox(padding=0.0).apply(ctx)
    assert ctx.image is None  # nothing built yet
    assert ctx.mask.shape == (8, 12)  # x 3..15, y 4..12 on (20, 30)
    assert ctx.frame.shape == (8, 12, 3)  # frame cropped to the same region


def test_reads_frame_flags():
    # Only apply_mask consumes the decoded pixels; the others build from the mask,
    # so a silhouette renderer can skip video decode.
    assert basic.ApplyMask().reads_frame is True
    for t in (
        basic.InitBinaryMask(),
        basic.CropToBbox(),
        basic.PadToSquare(),
        basic.Resize(size=64),
        basic.CenterInSquare(size=64),
    ):
        assert t.reads_frame is False


def test_reads_frame_not_a_dataclass_field():
    # ClassVar, not a constructor/serialized field — a stray field would leak into
    # to_spec and break run reproducibility.
    assert "reads_frame" not in basic.ApplyMask(background=0).to_spec()


def test_pad_to_square():
    ctx = basic.PadToSquare().apply(
        basic.InitBinaryMask().apply(_ctx())
    )  # (20,30)->(30,30)
    assert ctx.image.shape == (30, 30, 3)
    assert ctx.image[0, 0].tolist() == [0, 0, 0]  # black padding


def test_resize():
    ctx = basic.Resize(size=224).apply(basic.InitBinaryMask().apply(_ctx()))
    assert ctx.image.shape == (224, 224, 3)


def test_resize_uses_inter_area_on_downscale():
    # A large noisy image downscaled to the model input must use INTER_AREA
    # (antialiased), not INTER_LINEAR — the two differ measurably here (§14 fidelity).
    ctx = _ctx()
    big = np.random.default_rng(0).integers(0, 256, (80, 80, 3), dtype=np.uint8)
    ctx.image = big
    out = basic.Resize(size=20).apply(ctx).image
    assert np.array_equal(out, cv2.resize(big, (20, 20), interpolation=cv2.INTER_AREA))
    assert not np.array_equal(
        out, cv2.resize(big, (20, 20), interpolation=cv2.INTER_LINEAR)
    )


def test_resize_uses_inter_linear_on_upscale():
    ctx = _ctx()
    small = np.zeros((10, 10, 3), np.uint8)
    small[::2] = 255
    ctx.image = small
    out = basic.Resize(size=40).apply(ctx).image
    assert np.array_equal(
        out, cv2.resize(small, (40, 40), interpolation=cv2.INTER_LINEAR)
    )


def test_new_transforms_registered():
    for name in ("apply_mask", "center_in_square"):
        assert name in TRANSFORMS


def test_fill_transforms_registered():
    for name in ("fill_background_noise", "fill_background_bands"):
        assert name in TRANSFORMS


def _centered_masked_ctx(size: int = 48):
    """A ctx whose image + mask are centered in a ``size`` square (the fill-transform input)."""
    ctx = _ctx()
    ctx.frame = np.full((H, W, 3), 200, np.uint8)
    ctx = basic.ApplyMask(background=0).apply(ctx)
    ctx = basic.CenterInSquare(size=size).apply(ctx)
    return ctx


def test_center_in_square_carries_mask_into_canvas():
    ctx = _centered_masked_ctx(size=48)
    assert ctx.mask.shape == (48, 48)  # mask now lives in canvas coordinates
    assert int(ctx.mask.sum()) == 10 * 10  # the 10x10 subject block, unscaled
    assert (ctx.image[ctx.mask] == 200).all()  # subject aligns with the carried mask
    assert (ctx.image[~ctx.mask] == 0).all()


def test_fill_background_noise_keeps_subject_and_is_deterministic():
    ctx = _centered_masked_ctx()
    subject = ctx.mask.copy()
    ctx = basic.FillBackgroundNoise(seed=7).apply(ctx)
    assert (ctx.image[subject] == 200).all()  # subject pixels untouched
    assert ctx.image[~subject].std() > 0  # background is noise, not flat
    # same seed + shape -> identical field (a FIXED background across frames)
    again = basic.FillBackgroundNoise(seed=7).apply(_centered_masked_ctx()).image
    assert np.array_equal(ctx.image, again)
    assert basic.FillBackgroundNoise(seed=7).to_spec() == {
        "type": "fill_background_noise",
        "seed": 7,
    }


def test_fill_background_bands_alternates_ring_and_alt_from_the_outline():
    ctx = _centered_masked_ctx()
    subject = ctx.mask.copy()
    ring, alt = [255, 0, 0], [0, 0, 0]
    ctx = basic.FillBackgroundBands(ring=ring, alt=alt, thickness=3).apply(ctx)
    assert (ctx.image[subject] == 200).all()  # subject kept
    bg_colors = {
        tuple(c) for c in np.unique(ctx.image[~subject].reshape(-1, 3), axis=0)
    }
    assert bg_colors <= {tuple(ring), tuple(alt)}  # background is only ring/alt
    assert tuple(ring) in bg_colors and tuple(alt) in bg_colors
    # the band nearest the outline (distance band 0) is the ring color
    boundary = (
        cv2.dilate(
            subject.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
        ).astype(bool)
        & ~subject
    )
    assert (ctx.image[boundary] == ring).all()
    assert basic.FillBackgroundBands(ring=ring, alt=alt, thickness=4).to_spec() == {
        "type": "fill_background_bands",
        "ring": ring,
        "alt": alt,
        "thickness": 4,
    }


def test_fill_background_bands_is_noop_without_a_mask():
    ctx = _ctx()
    ctx.mask = None
    ctx.image = np.full((10, 10, 3), 50, np.uint8)
    out = (
        basic.FillBackgroundBands(ring=[255, 0, 0], alt=[0, 0, 0], thickness=2)
        .apply(ctx)
        .image
    )
    assert (out == 50).all()  # nothing to radiate from -> unchanged


def test_apply_mask_keeps_frame_pixels_under_mask():
    ctx = _ctx()
    ctx.frame = np.full((H, W, 3), 200, np.uint8)  # real (flat) frame pixels
    ctx = basic.ApplyMask(background=0).apply(ctx)
    assert ctx.image.shape == (H, W, 3)
    assert ctx.image[10, 15].tolist() == [200, 200, 200]  # inside mask -> frame kept
    assert ctx.image[0, 0].tolist() == [0, 0, 0]  # outside mask -> background


def test_apply_mask_supports_rgb_background():
    ctx = _ctx()
    ctx.frame = np.full((H, W, 3), 200, np.uint8)
    ctx = basic.ApplyMask(background=[255, 0, 0]).apply(ctx)
    assert ctx.image[10, 15].tolist() == [200, 200, 200]  # inside mask -> frame kept
    assert ctx.image[0, 0].tolist() == [
        255,
        0,
        0,
    ]  # outside mask -> the [B,G,R] background
    assert basic.ApplyMask(background=[255, 0, 0]).to_spec()["background"] == [
        255,
        0,
        0,
    ]


def test_apply_mask_resizes_mask_to_frame():
    ctx = _ctx()
    ctx.frame = np.full((H * 2, W * 2, 3), 100, np.uint8)  # frame larger than mask
    ctx = basic.ApplyMask().apply(ctx)
    assert ctx.image.shape == (H * 2, W * 2, 3)  # output matches the frame, not mask


def test_center_in_square_preserves_scale_and_centers():
    ctx = basic.CropToBbox(padding=0.0).apply(basic.InitBinaryMask().apply(_ctx()))
    cropped = ctx.image.copy()
    white_before = int((cropped == 255).all(axis=2).sum())
    ctx = basic.CenterInSquare(size=40).apply(ctx)
    assert ctx.image.shape == (40, 40, 3)
    # unscaled: the same number of white pixels survives, just re-positioned
    assert int((ctx.image == 255).all(axis=2).sum()) == white_before
    assert ctx.image[0, 0].tolist() == [0, 0, 0]  # black canvas border


def test_center_in_square_fill_color_pads_with_color():
    ctx = basic.CropToBbox(padding=0.0).apply(basic.InitBinaryMask().apply(_ctx()))
    ctx = basic.CenterInSquare(size=64, fill=[255, 0, 0]).apply(ctx)
    assert ctx.image.shape == (64, 64, 3)
    assert ctx.image[0, 0].tolist() == [255, 0, 0]  # colored canvas border, not black
    assert basic.CenterInSquare(size=64, fill=[255, 0, 0]).to_spec()["fill"] == [
        255,
        0,
        0,
    ]
    assert "fill" not in basic.CenterInSquare(size=64).to_spec()  # omitted when default


def test_center_in_square_center_crops_when_larger():
    ctx = basic.InitBinaryMask().apply(_ctx())  # (20, 30, 3)
    ctx = basic.CenterInSquare(size=10).apply(ctx)
    assert ctx.image.shape == (10, 10, 3)


def test_to_spec():
    assert basic.CropToBbox(padding=0.1).to_spec() == {
        "type": "crop_to_bbox",
        "padding": 0.1,
    }
    assert basic.Resize(size=64).to_spec() == {"type": "resize", "size": 64}
    assert basic.ApplyMask(background=0).to_spec() == {
        "type": "apply_mask",
        "background": 0,
    }
    assert basic.CenterInSquare(size=128).to_spec() == {
        "type": "center_in_square",
        "size": 128,
    }


# --------------------------------------------------------------------------- #
# §14 egocentric alignment
# --------------------------------------------------------------------------- #
def _diagonal_ctx():
    """A cropped subject: a thin diagonal bar from top-left to bottom-right on a 40x40 crop,
    heading 45° (down-right in image coordinates)."""
    from detail.core.dataclasses import RenderContext

    mask = np.zeros((40, 40), dtype=bool)
    for i in range(6, 34):
        mask[i - 1 : i + 2, i - 1 : i + 2] = True
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[mask] = (200, 100, 50)
    return RenderContext(
        frame=None,
        mask=mask,
        bbox=(0.0, 0.0, 1.0, 1.0),
        image=image,
        frame_idx=0,
        t_ns=0,
        class_name="",
        confidence=0.0,
        object_id=0,
        aux={"heading_deg": 45.0, "align_mirrored": False},
    )


def test_align_rotate_turns_the_axis_flat_and_recrops_to_the_turned_mask():
    from detail.transforms.basic import AlignRotate, turned_mask_extent

    ctx = _diagonal_ctx()
    before_w, before_h = turned_mask_extent(ctx.mask, 0.0)
    out = AlignRotate(fill=[255, 0, 0]).apply(ctx)
    h, w = out.image.shape[:2]
    assert w > 2 * h, (w, h)  # a diagonal bar turned flat: wide and short
    assert out.mask.shape == (h, w) and out.mask.any()
    assert (before_w, before_h) == (30, 30)
    expected_w, expected_h = turned_mask_extent(_diagonal_ctx().mask, 45.0)
    assert abs(w - expected_w) <= 2 and abs(h - expected_h) <= 2
    # the bar kept its colour (bilinear edges blend with the fill), the exposed corners
    # took the fill: green marks the bar, pure blue the fill
    assert (out.image[out.mask][:, 1] > 50).mean() > 0.6
    corners = out.image[~out.mask]
    assert (
        corners[:, 0].mean() > corners[:, 1].mean()
    )  # the fill's blue dominates outside
    assert AlignRotate(fill=[255, 0, 0]).to_spec() == {
        "type": "align_rotate",
        "fill": [255, 0, 0],
    }


def test_align_rotate_refuses_a_frame_without_a_heading():
    from detail.transforms.basic import AlignRotate

    ctx = _diagonal_ctx()
    ctx.aux = {"heading_deg": float("nan")}
    with pytest.raises(ValueError, match="no heading"):
        AlignRotate().apply(ctx)
    ctx.aux = {}
    with pytest.raises(ValueError, match="no heading"):
        AlignRotate().apply(ctx)


def test_align_mirror_flips_only_when_flagged():
    from detail.transforms.basic import AlignMirror

    ctx = _diagonal_ctx()
    same = AlignMirror().apply(_diagonal_ctx())
    np.testing.assert_array_equal(same.image, ctx.image)
    ctx.aux["align_mirrored"] = True
    flipped = AlignMirror().apply(ctx)
    np.testing.assert_array_equal(flipped.image, _diagonal_ctx().image[:, ::-1])
    np.testing.assert_array_equal(flipped.mask, _diagonal_ctx().mask[:, ::-1])
    assert AlignMirror().to_spec() == {"type": "align_mirror"}


def test_turned_mask_extent_is_the_rotated_box():
    from detail.transforms.basic import turned_mask_extent

    mask = np.zeros((20, 50), dtype=bool)
    mask[8:12, 5:45] = True  # a 40x4 bar
    assert turned_mask_extent(mask, 0.0) == (40, 4)
    w, h = turned_mask_extent(mask, 90.0)
    assert (w, h) == (4, 40)
    w, h = turned_mask_extent(mask, 45.0)
    assert 30 <= w <= 33 and 30 <= h <= 33
    assert turned_mask_extent(np.zeros((5, 5), bool), 10.0) == (0, 0)


def _every_pixel_turned(mask: np.ndarray, deg: float) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    theta = np.radians(-deg)
    rx = xs * np.cos(theta) - ys * np.sin(theta)
    ry = xs * np.sin(theta) + ys * np.cos(theta)
    return (
        int(np.ceil(round(float(rx.max() - rx.min()), 6))) + 1,
        int(np.ceil(round(float(ry.max() - ry.min()), 6))) + 1,
    )


@pytest.mark.parametrize("seed", range(5))
def test_turned_mask_extent_turns_only_each_columns_ends_and_gets_the_same_box(seed):
    """It turns each column's top and bottom pixel, not every pixel: along any direction a
    column's extreme is one of its ends. Masks with holes, gaps inside a column and several
    pieces are where a shortcut like that would go wrong, so those are what is drawn."""
    from detail.transforms.basic import turned_mask_extent

    rng = np.random.default_rng(seed)
    for _ in range(40):
        mask = rng.random((int(rng.integers(3, 40)), int(rng.integers(3, 40)))) < 0.3
        deg = float(rng.uniform(-360, 360))
        assert turned_mask_extent(mask, deg) == _every_pixel_turned(mask, deg)
