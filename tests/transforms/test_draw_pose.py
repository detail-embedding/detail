"""``draw_pose``: the keypoints painted onto the image the model sees (§14 "Pose in the embedding").

**The failure to guard is a dot in the wrong place that still looks like a dot.** The points are
stored as full-frame normalized coordinates, and the image they are drawn on has already been
cropped to the animal (and may then be turned). A step that forgot the crop would draw every dot in
the corner region of the crop instead of on the body, in a run whose images all look fine. So each
test puts a point at a known place and asks *which pixel of the finished image* is coloured —
through the crop, and through a turn.

The step carries the keypoint track run's skeleton (§26); a spec recorded before §26 carries
``sides`` / ``edges`` by column and must still draw the same colours.
"""

import numpy as np
import pytest

from detail.core.dataclasses import RenderContext
from detail.core.visualize import KEYPOINT_COLORS as SIDE_COLORS
from detail.transforms import basic  # noqa: F401 — registers the transforms
from detail.transforms.constructors import check_pose_step, with_alignment, with_pose
from detail.transforms.renderer import MaskRenderer

FRAME = 100  # a 100x100 source grid
# point 0 on the midline, point 1 a left one (paired with an unseen right point 2)
SKELETON = {
    "names": ["nose", "left_ear", "right_ear"],
    "edges": [["nose", "left_ear"]],
    "pairs": [["left_ear", "right_ear"]],
}
DOT = {"type": "draw_pose", "skeleton": SKELETON, "size": 0.03}
# a pre-§26 spec: sides by column (a left always came with its right partner) and edges
OLD_DOT = {
    "type": "draw_pose",
    "sides": ["midline", "left", "right"],
    "edges": [[0, 1]],
    "size": 0.03,
}


def _ctx(mask, bbox, xy, score=(0.9, 0.9, 0.9)):
    return RenderContext(
        frame=None,
        mask=mask,
        bbox=bbox,
        image=None,
        frame_idx=0,
        t_ns=0,
        class_name="",
        confidence=0.0,
        object_id=0,
        aux={"pose": {"xy": np.asarray(xy, float), "score": np.asarray(score, float)}},
    )


def _square(r0, r1, c0, c1):
    mask = np.zeros((FRAME, FRAME), bool)
    mask[r0:r1, c0:c1] = True
    return mask


def _centroid(image, rgb):
    hit = (image == np.asarray(rgb)).all(axis=-1)
    ys, xs = np.nonzero(hit)
    return float(xs.mean()), float(ys.mean())


def test_a_point_lands_on_its_pixel_of_the_cropped_image():
    # The subject fills x 40..80, y 20..60; the crop is that box, so a point at frame
    # (60, 40) is the middle of the crop, not pixel (60, 40) of anything.
    mask = _square(20, 60, 40, 80)
    bbox = (0.40, 0.20, 0.80, 0.60)
    spec = [{"type": "crop_to_bbox"}, {"type": "init_binary_mask"}, DOT]
    image = MaskRenderer.from_spec(spec).render(
        _ctx(mask, bbox, [[0.60, 0.40], [np.nan, np.nan], [np.nan, np.nan]])
    )
    assert image.shape[:2] == (40, 40)
    cx, cy = _centroid(image, SIDE_COLORS["midline"])
    assert (round(cx), round(cy)) == (20, 20)


def test_a_point_off_the_corner_of_the_crop_is_not_where_the_uncropped_frame_would_put_it():
    mask = _square(20, 60, 40, 80)
    bbox = (0.40, 0.20, 0.80, 0.60)
    spec = [{"type": "crop_to_bbox"}, {"type": "init_binary_mask"}, DOT]
    # the subject's top-left corner, in frame coordinates
    image = MaskRenderer.from_spec(spec).render(
        _ctx(mask, bbox, [[0.40, 0.20], [np.nan, np.nan], [np.nan, np.nan]])
    )
    cx, cy = _centroid(image, SIDE_COLORS["midline"])
    # The dot is cut by the image's own corner, so its visible centre sits a pixel in.
    assert cx <= 2 and cy <= 2


def test_points_below_the_floor_or_not_placed_are_not_drawn():
    mask = _square(20, 60, 40, 80)
    bbox = (0.40, 0.20, 0.80, 0.60)
    step = {**DOT, "min_score": 0.5}
    spec = [{"type": "crop_to_bbox"}, {"type": "init_binary_mask"}, step]
    image = MaskRenderer.from_spec(spec).render(
        _ctx(
            mask,
            bbox,
            [[0.6, 0.4], [0.5, 0.3], [np.nan, np.nan]],
            score=(0.9, 0.2, 0.0),
        )
    )
    assert (image == np.asarray(SIDE_COLORS["midline"])).all(axis=-1).any()
    assert not (image == np.asarray(SIDE_COLORS["left"])).all(axis=-1).any()


def test_a_frame_the_run_has_no_points_for_is_left_as_the_mask():
    mask = _square(20, 60, 40, 80)
    bbox = (0.40, 0.20, 0.80, 0.60)
    spec = [{"type": "crop_to_bbox"}, {"type": "init_binary_mask"}, DOT]
    ctx = _ctx(mask, bbox, [[0.6, 0.4], [0.5, 0.3], [np.nan, np.nan]])
    ctx.aux.pop("pose")
    plain = MaskRenderer.from_spec(spec[:2]).render(
        _ctx(mask, bbox, [[0.6, 0.4], [0.5, 0.3], [np.nan, np.nan]])
    )
    assert (MaskRenderer.from_spec(spec).render(ctx) == plain).all()


def test_a_turn_carries_the_points_with_the_body():
    """Drawn before ``align_rotate``, so a point on the subject's tail end stays on it."""
    mask = _square(40, 60, 20, 80)  # a wide bar, 60 across and 20 down
    bbox = (0.20, 0.40, 0.80, 0.60)
    spec = with_alignment(
        with_pose([{"type": "crop_to_bbox"}, {"type": "init_binary_mask"}], DOT),
        "rotate",
    )
    assert [s["type"] for s in spec] == [
        "crop_to_bbox",
        "init_binary_mask",
        "draw_pose",
        "align_rotate",
    ]
    ctx = _ctx(
        mask, bbox, [[0.78, 0.50], [np.nan, np.nan], [np.nan, np.nan]]
    )  # near the bar's right end
    ctx.aux["heading_deg"] = (
        90.0  # turn it a quarter: the right end goes to the top or bottom
    )
    image = MaskRenderer.from_spec(spec).render(ctx)
    ys, xs = np.nonzero((image > 0).any(axis=-1))
    assert (xs.max() - xs.min()) < (ys.max() - ys.min())  # the bar now stands upright
    cx, cy = _centroid(image, SIDE_COLORS["midline"])
    # an end of the upright bar, not its middle
    assert abs(cy - (ys.min() + ys.max()) / 2) > 0.3 * (ys.max() - ys.min())


def test_the_step_goes_before_the_alignment_and_the_canvas_and_only_one():
    base = [
        {"type": "crop_to_bbox"},
        {"type": "init_binary_mask"},
        {"type": "center_in_square", "size": 64},
        {"type": "resize", "size": 224},
    ]
    drawn = with_pose(base, DOT)
    assert [s["type"] for s in drawn] == [
        "crop_to_bbox",
        "init_binary_mask",
        "draw_pose",
        "center_in_square",
        "resize",
    ]
    assert with_pose(base, None) == base
    check_pose_step(drawn, "draw")
    check_pose_step(base, "none")
    with pytest.raises(ValueError, match="pose steps"):
        check_pose_step(base, "draw")
    with pytest.raises(ValueError, match="pose steps"):
        check_pose_step(drawn, "append")


def test_the_spec_round_trips():
    renderer = MaskRenderer.from_spec([{**DOT, "min_score": 0.4}])
    assert renderer.to_spec() == [{**DOT, "min_score": 0.4}]
    default_size = MaskRenderer.from_spec(
        [{"type": "draw_pose", "skeleton": SKELETON}]
    ).to_spec()
    assert default_size == [{"type": "draw_pose", "skeleton": SKELETON}]
    # a pre-§26 spec keeps its own shape
    assert MaskRenderer.from_spec([OLD_DOT]).to_spec() == [OLD_DOT]


def test_a_spec_recorded_before_the_rebuild_draws_the_same_colours():
    mask = _square(20, 60, 40, 80)
    bbox = (0.40, 0.20, 0.80, 0.60)
    xy = [[0.6, 0.4], [0.5, 0.3], [np.nan, np.nan]]
    old = MaskRenderer.from_spec(
        [{"type": "crop_to_bbox"}, {"type": "init_binary_mask"}, OLD_DOT]
    ).render(_ctx(mask, bbox, xy))
    assert (old == np.asarray(SIDE_COLORS["midline"])).all(axis=-1).any()
    assert (old == np.asarray(SIDE_COLORS["left"])).all(axis=-1).any()
