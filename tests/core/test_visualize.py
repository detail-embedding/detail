import numpy as np
import pytest

from detail.core import visualize
from detail.core.dataclasses import Detection
from detail.core.skeleton import Skeleton
from detail.core.visualize import (
    PALETTE_OKABE_ITO,
    assign_colors,
    draw_bbox,
    draw_detections,
    draw_detections_by_class,
    draw_mask,
)


def _frame(h=24, w=32, fill=128) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


# --- assign_colors --------------------------------------------------------- #
def test_assign_colors_one_color_per_key():
    colors = assign_colors(["a", "b", "c"], seed=0)
    assert set(colors) == {"a", "b", "c"}
    for v in colors.values():
        assert v in PALETTE_OKABE_ITO


def test_assign_colors_deterministic_with_seed():
    a = assign_colors(["a", "b", "c", "d"], seed=42)
    b = assign_colors(["a", "b", "c", "d"], seed=42)
    assert a == b


def test_assign_colors_random_without_seed_eventually_differs():
    # Two fresh-RNG draws over 6 keys × 6 palette colors will (with overwhelming
    # probability) differ. If they don't, the function is using a fixed seed.
    runs = [tuple(assign_colors(["a"] * 6, seed=None).values()) for _ in range(10)]
    assert len(set(runs)) > 1


def test_assign_colors_cycles_when_keys_exceed_palette():
    n = len(PALETTE_OKABE_ITO) + 3
    colors = assign_colors([f"k{i}" for i in range(n)], seed=0)
    assert len(colors) == n
    assert set(colors.values()) <= set(PALETTE_OKABE_ITO)


def test_assign_colors_empty_palette_raises():
    with pytest.raises(ValueError, match="at least one color"):
        assign_colors(["a"], palette=())


# --- draw_mask ------------------------------------------------------------- #
def test_draw_mask_blends_in_place_on_copy():
    frame = _frame()
    before = frame.copy()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[5:10, 5:10] = True
    out = draw_mask(frame, mask, (200, 100, 50), alpha=0.5)
    np.testing.assert_array_equal(frame, before)  # input unmutated
    assert out.shape == frame.shape and out.dtype == frame.dtype
    # the masked region should differ from the background, the rest should not
    assert not np.array_equal(out[5:10, 5:10], before[5:10, 5:10])
    np.testing.assert_array_equal(out[0:5, :], before[0:5, :])


def test_draw_mask_none_returns_copy():
    frame = _frame()
    out = draw_mask(frame, None, (255, 0, 0))
    np.testing.assert_array_equal(out, frame)
    assert out is not frame  # still a copy


def test_draw_mask_shape_mismatch_returns_copy():
    frame = _frame()
    bad_mask = np.zeros((10, 10), dtype=bool)  # wrong shape
    out = draw_mask(frame, bad_mask, (255, 0, 0))
    np.testing.assert_array_equal(out, frame)


def test_draw_mask_empty_mask_is_noop():
    frame = _frame()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    out = draw_mask(frame, mask, (255, 0, 0))
    np.testing.assert_array_equal(out, frame)


# --- draw_bbox ------------------------------------------------------------- #
def test_draw_bbox_returns_new_array_with_changes():
    frame = _frame()
    before = frame.copy()
    out = draw_bbox(frame, (0.1, 0.1, 0.9, 0.9), (255, 0, 0))
    np.testing.assert_array_equal(frame, before)
    assert out.shape == frame.shape and out.dtype == frame.dtype
    assert not np.array_equal(out, frame)


def test_draw_bbox_none_returns_copy():
    frame = _frame()
    out = draw_bbox(frame, None, (255, 0, 0))
    np.testing.assert_array_equal(out, frame)
    assert out is not frame


def test_draw_bbox_label_overlays_text():
    # Just sanity-check that adding a label changes the output vs no-label.
    frame = _frame()
    without = draw_bbox(frame, (0.1, 0.1, 0.5, 0.5), (255, 0, 0))
    with_lbl = draw_bbox(frame, (0.1, 0.1, 0.5, 0.5), (255, 0, 0), label="hello")
    assert not np.array_equal(without, with_lbl)


# --- frame labels: readable on any frame, in any palette colour -------------------------- #
def _contrast(a, b) -> float:
    la = visualize._relative_luminance(a)
    lb = visualize._relative_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


@pytest.mark.parametrize(
    "color",
    [*PALETTE_OKABE_ITO, visualize.DETECTION_BOX_COLOR, (255, 255, 255), (0, 0, 0)],
)
def test_every_label_colour_gets_ink_that_reads_on_it(color):
    # 4.5:1 is WCAG AA for body text: the pale blue/pink tags that were unreadable meet it.
    assert _contrast(visualize.readable_text_color(color), color) >= 4.5


def test_readable_text_color_picks_dark_on_light_and_light_on_dark():
    assert visualize.readable_text_color((255, 255, 255)) == visualize.LABEL_INK_DARK
    assert visualize.readable_text_color((0, 0, 0)) == visualize.LABEL_INK_LIGHT


def test_a_box_label_sits_on_a_solid_tag_not_on_the_footage():
    # The old label was text in the box colour straight onto the frame, so the pixels
    # between its letters were footage. A tag fills them with the box colour.
    frame = _frame(h=200, w=300, fill=128)
    color = (86, 180, 233)
    out = draw_bbox(frame, (0.3, 0.5, 0.8, 0.9), color, label="white mouse 0.97")
    x0, y_top = int(0.3 * 300), int(0.5 * 200)
    tag = out[y_top - 12 : y_top - 1, x0 : x0 + 60].reshape(-1, 3)
    filled = (tag == color).all(axis=1).mean()
    # Anti-aliased strokes at a small scale rarely hit the ink exactly; nearer the ink
    # than the tag is what "drawn in the contrasting ink" looks like in pixels.
    ink = np.asarray(visualize.readable_text_color(color))
    to_ink = np.abs(tag.astype(int) - ink).sum(axis=1)
    inked = (to_ink < np.abs(np.asarray(color) - ink).sum() / 2).any()
    assert filled > 0.5, "the space around the letters is the tag, not the frame"
    assert inked, "the letters are drawn in the contrasting ink"
    assert not (tag == 128).all(axis=1).any(), "no footage shows through the tag"


def test_a_label_with_no_room_above_hangs_inside_the_frame():
    frame = _frame(h=100, w=100, fill=0)
    out = draw_bbox(frame, (0.9, 0.0, 1.0, 0.5), (230, 159, 0), label="a long label")
    assert (out[2, 99] == (230, 159, 0)).all(), "tag clamped to the right and top edges"


def test_label_scale_grows_with_the_frame():
    assert visualize.label_scale(1080) > visualize.label_scale(480)
    assert visualize.label_scale(10) == pytest.approx(0.4)


# --- draw_detections ------------------------------------------------------- #
def test_draw_detections_empty_list_returns_unchanged_copy():
    frame = _frame()
    out = draw_detections(frame, [], (255, 0, 0))
    np.testing.assert_array_equal(out, frame)


def test_draw_detections_draws_each_detection():
    frame = _frame()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[5:10, 5:10] = True
    dets = [
        Detection(
            mask=mask,
            bbox=(0.1, 0.1, 0.5, 0.5),
            class_name="mouse",
            confidence=0.9,
            object_id=None,
        ),
        Detection(
            mask=None,
            bbox=(0.6, 0.6, 0.9, 0.9),
            class_name="pellet",
            confidence=0.7,
            object_id=None,
        ),
    ]
    out = draw_detections(frame, dets, (0, 158, 115))
    assert out.shape == frame.shape and out.dtype == frame.dtype
    assert not np.array_equal(out, frame)


# --- display threshold + copy cost ----------------------------------------- #
def _det(confidence, row=5, class_name="mouse", shape=(24, 32)):
    """A detection whose mask and bbox both sit at ``row``, so two of them never overlap."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[row : row + 4, row : row + 4] = True
    return Detection(
        mask=mask,
        bbox=(row / w, row / h, (row + 4) / w, (row + 4) / h),
        class_name=class_name,
        confidence=confidence,
        object_id=None,
    )


def test_draw_detections_min_conf_skips_the_ones_below_it():
    # Room enough that the 0.9 detection's label tag cannot reach the 0.3 one's pixels.
    shape = (200, 300)
    frame = _frame(*shape)
    dets = [_det(0.9, row=2, shape=shape), _det(0.3, row=150, shape=shape)]
    everything = draw_detections(frame, dets, (255, 0, 0))
    filtered = draw_detections(frame, dets, (255, 0, 0), min_conf=0.5)

    # The 0.9 detection is drawn either way.
    assert not np.array_equal(filtered[2:6, 2:6], frame[2:6, 2:6])
    # The 0.3 one is drawn without a floor and gone with one.
    assert not np.array_equal(everything[150:154, 150:154], frame[150:154, 150:154])
    np.testing.assert_array_equal(filtered[150:154, 150:154], frame[150:154, 150:154])


def test_draw_detections_min_conf_is_inclusive_at_the_threshold():
    frame = _frame()
    drawn = draw_detections(frame, [_det(0.5)], (255, 0, 0), min_conf=0.5)
    assert not np.array_equal(drawn, frame)  # >=, so 0.5 survives a 0.5 floor


def test_draw_detections_does_not_chain_the_copying_primitives(monkeypatch):
    """A wide-open preview puts every proposal on one frame. The old form chained
    ``draw_mask`` / ``draw_bbox``, each of which copies the whole frame, so N detections
    cost 2N full-frame copies — invisible at a threshold of 0.5 and ruinous at 0.0.
    Restore the chained form and this fails."""
    import detail.core.visualize as viz

    calls = []
    monkeypatch.setattr(viz, "draw_mask", lambda *a, **k: calls.append("mask") or a[0])
    monkeypatch.setattr(viz, "draw_bbox", lambda *a, **k: calls.append("bbox") or a[0])
    dets = [_det(0.9, row=r) for r in range(0, 16, 4)]
    out = viz.draw_detections(_frame(), dets, (255, 0, 0))

    assert calls == [], "draw_detections must not go through the copying wrappers"
    assert not np.array_equal(out, _frame())  # it still drew


def test_draw_detections_returns_a_copy_not_the_input():
    frame = _frame()
    out = draw_detections(frame, [_det(0.9)], (255, 0, 0))
    assert out is not frame
    np.testing.assert_array_equal(frame, _frame())  # input untouched


def test_draw_detections_by_class_colors_each_detection():
    frame = _frame()
    colors = {"mouse": (255, 0, 0), "cage": (0, 0, 255)}
    dets = [_det(0.9, row=2, class_name="mouse"), _det(0.9, row=14, class_name="cage")]
    out = draw_detections_by_class(frame, dets, colors)

    # Each class picked up its own color, so the two regions differ from each other.
    assert not np.array_equal(out[3:5, 3:5], out[15:17, 15:17])
    # An unknown class falls back to the default rather than raising.
    stranger = draw_detections_by_class(frame, [_det(0.9, class_name="?")], colors)
    assert not np.array_equal(stranger, frame)


def test_draw_detections_by_class_honours_min_conf():
    frame = _frame()
    colors = {"mouse": (255, 0, 0)}
    out = draw_detections_by_class(frame, [_det(0.2)], colors, min_conf=0.5)
    np.testing.assert_array_equal(out, frame)


def test_bgr_to_rgb_matches_the_numpy_reverse_it_replaces():
    """The swap must be bit-identical to the numpy reverse it replaced.

    The swap is an optimization, so it has to
    be bit-identical to the ``frame[:, :, ::-1].copy()`` it replaced at every call site, or it
    is silently changing every picture the preview shows."""
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    out = visualize.bgr_to_rgb(bgr)
    assert np.array_equal(out, np.ascontiguousarray(bgr[:, :, ::-1]))
    assert out.flags["C_CONTIGUOUS"]


def test_rgb_to_bgr_round_trips():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (12, 9, 3), dtype=np.uint8)
    assert np.array_equal(visualize.rgb_to_bgr(visualize.bgr_to_rgb(rgb)), rgb)


# --- blending inside the mask's box ---------------------------------------------------------- #
def _reference_blend(frame, mask, color, alpha):
    """What the blend did before it was restricted to the mask's bounding box.

    Kept as the oracle rather than deleted: the change is an optimisation, so the only thing that
    makes it correct is that it produces the *same bytes*, and that claim needs the old code to
    compare against. A test asserting the new code against itself would pass on any behaviour.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    if mask is None or mask.shape != (h, w):
        return out
    pixels = mask > 0
    if not pixels.any():
        return out
    tint = np.asarray(color, dtype=np.float32)
    out[pixels] = ((1.0 - alpha) * out[pixels] + alpha * tint).astype(np.uint8)
    return out


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param("blob", id="a blob in the middle"),
        pytest.param("corner", id="touching the top-left corner"),
        pytest.param("edges", id="touching all four edges"),
        pytest.param("scattered", id="two disconnected pieces"),
        pytest.param("single", id="one pixel"),
        pytest.param("full", id="the whole frame"),
        pytest.param("empty", id="nothing masked"),
    ],
)
def test_blending_in_the_mask_box_is_byte_identical(shape):
    """The optimisation is only correct if the picture does not change.

    Restricting the blend to the mask's bounding box made it several times faster on a
    1080p frame, because a subject fills a small part of an arena and the old form built,
    scanned, gathered and scattered a frame-sized boolean whatever the size of the animal.

    The edge cases are the ones a bounding box gets wrong: a mask touching an edge (an
    off-by-one in the slice), disconnected pieces (a box spanning both), one pixel (an empty
    slice if the end index is exclusive by mistake), and nothing at all.
    """
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)
    mask = np.zeros((40, 60), dtype=np.uint8)
    if shape == "blob":
        mask[10:20, 15:30] = 1
    elif shape == "corner":
        mask[0:5, 0:5] = 1
    elif shape == "edges":
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 1
    elif shape == "scattered":
        mask[2:5, 2:5] = 1
        mask[35:39, 50:58] = 1
    elif shape == "single":
        mask[7, 41] = 1
    elif shape == "full":
        mask[:] = 1

    got = visualize.draw_mask(frame, mask, (230, 159, 0), alpha=0.35)
    expected = _reference_blend(frame, mask, (230, 159, 0), 0.35)

    assert np.array_equal(got, expected), (
        f"a {shape} mask blends differently than it used to — the optimisation changed the "
        "picture, which is the one thing it may not do"
    )
    assert (
        not np.array_equal(got, frame) or shape == "empty"
    ), "nothing was drawn at all, so the comparison above proves nothing"


def test_the_blend_leaves_everything_outside_the_mask_alone():
    """The failure a bounding box invites: tinting the box rather than the mask inside it."""
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5, 5] = 1
    mask[14, 14] = 1  # a box spanning rows 5-14 and columns 5-14, two pixels set

    out = visualize.draw_mask(frame, mask, (255, 255, 255), alpha=1.0)

    assert out[5, 5].tolist() == [255, 255, 255] and out[14, 14].tolist() == [
        255,
        255,
        255,
    ]
    assert out[9, 9].tolist() == [
        0,
        0,
        0,
    ], "a pixel inside the box but outside the mask was tinted"
    assert int(out.sum()) == 2 * 3 * 255, "more than the two masked pixels were touched"


# --- preview style ----------------------------------------------------------- #
def _preview_det(*, mask=None, bbox=(0.1, 0.1, 0.5, 0.5), object_id=1):
    return Detection(
        mask=mask, bbox=bbox, class_name="mouse", confidence=0.75, object_id=object_id
    )


def _solid_color(_det):
    return (255, 0, 0)


def test_resolve_preview_style_raises_on_unknown():
    with pytest.raises(ValueError, match="unknown preview style"):
        visualize.resolve_preview_style("sepia")


def test_preview_needs_mask_matches_the_styles_that_draw_a_mask():
    needs = {s: visualize.preview_needs_mask(s) for s in visualize.PREVIEW_STYLES}
    assert needs == {
        visualize.MASK_BOX_CONF: True,
        visualize.MASK_BOX: True,
        visualize.MASK_ONLY: True,
        visualize.BOX_CONF: False,
        visualize.BOX_ONLY: False,
        visualize.RAW_VIDEO: False,
        visualize.RAW_MASK: True,  # the green-screen cutout needs the mask to cut with
    }


def test_raw_video_draws_nothing():
    frame = _frame()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[5:10, 5:10] = True
    out = visualize.draw_preview_overlay(
        frame, [_preview_det(mask=mask)], visualize.RAW_VIDEO, color_fn=_solid_color
    )
    np.testing.assert_array_equal(out, frame)
    assert out is not frame


def test_mask_only_draws_no_box_and_never_asks_for_a_label():
    frame = _frame()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[5:10, 5:10] = True

    def _label_fn(_det, _conf):
        raise AssertionError("mask_only has no box, so a label must never be asked for")

    out = visualize.draw_preview_overlay(
        frame,
        [_preview_det(mask=mask)],
        visualize.MASK_ONLY,
        color_fn=_solid_color,
        label_fn=_label_fn,
    )
    assert not np.array_equal(out[5:10, 5:10], frame[5:10, 5:10])  # the mask blended in
    np.testing.assert_array_equal(
        out[0:5, :], frame[0:5, :]
    )  # nothing outside it moved


def test_box_only_draws_a_box_and_asks_for_a_label():
    frame = _frame()
    seen = []

    def _label_fn(det, show_confidence):
        seen.append(show_confidence)
        return None

    out = visualize.draw_preview_overlay(
        frame,
        [_preview_det()],
        visualize.BOX_ONLY,
        color_fn=_solid_color,
        label_fn=_label_fn,
    )
    assert not np.array_equal(out, frame)  # the box outline drew something
    assert seen == [False]  # box_only's own show_confidence flag


def test_mask_box_conf_asks_for_a_label_with_confidence_on():
    frame = _frame()
    seen = []
    out = visualize.draw_preview_overlay(
        frame,
        [_preview_det()],
        visualize.MASK_BOX_CONF,
        color_fn=_solid_color,
        label_fn=lambda det, conf: seen.append(conf) or f"{det.class_name}",
    )
    assert seen == [True]
    assert not np.array_equal(out, frame)


def test_raw_mask_is_green_screen_with_the_subjects_real_pixels_cut_out():
    frame = np.full((20, 20, 3), 100, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True

    out = visualize.draw_preview_overlay(
        frame, [_preview_det(mask=mask)], visualize.RAW_MASK, color_fn=_solid_color
    )
    assert out[0, 0].tolist() == [0, 255, 0]  # green screen outside the mask
    assert out[7, 7].tolist() == [
        100,
        100,
        100,
    ]  # the subject's real pixel, unrecolored
    assert out.shape == frame.shape and out.dtype == frame.dtype


def test_raw_mask_with_no_detections_is_solid_green():
    frame = np.full((10, 10, 3), 100, dtype=np.uint8)
    out = visualize.draw_preview_overlay(
        frame, [], visualize.RAW_MASK, color_fn=_solid_color
    )
    assert (out == [0, 255, 0]).all()


def test_track_color_is_stable_and_untracked_falls_back_to_the_detection_color():
    assert visualize.track_color(3) == visualize.track_color(3)
    assert visualize.track_color(None) == visualize.DETECTION_BOX_COLOR
    assert visualize.track_color(None) not in set(PALETTE_OKABE_ITO)


# --- draw_keypoints (§26) -------------------------------------------------- #
_SKELETON = Skeleton(
    names=("nose", "left_ear", "right_ear", "tail_base"),
    edges=(("nose", "tail_base"),),
    pairs=(("left_ear", "right_ear"),),
)


def _points(*xy) -> np.ndarray:
    return np.array(xy, dtype=np.float64)


def test_keypoint_roles_come_from_the_pairs():
    assert visualize.keypoint_roles(_SKELETON) == [
        "midline",
        "left",
        "right",
        "midline",
    ]


def test_draw_keypoints_colours_each_point_by_its_role_and_returns_a_new_frame():
    frame = _frame(h=160, w=160, fill=0)
    before = frame.copy()
    pts = _points((0.25, 0.25), (0.5, 0.25), (0.75, 0.25), (0.25, 0.75))

    out = visualize.draw_keypoints(frame, pts, _SKELETON)

    np.testing.assert_array_equal(frame, before)
    assert out[40, 40].tolist() == list(visualize.KEYPOINT_COLORS["midline"])
    assert out[40, 80].tolist() == list(visualize.KEYPOINT_COLORS["left"])
    assert out[40, 120].tolist() == list(visualize.KEYPOINT_COLORS["right"])
    assert out[120, 40].tolist() == list(visualize.KEYPOINT_COLORS["midline"])


def test_draw_keypoints_skips_nan_and_hidden_points():
    frame = _frame(h=160, w=160, fill=0)
    pts = _points((0.25, 0.25), (np.nan, np.nan), (0.75, 0.25), (0.25, 0.75))

    out = visualize.draw_keypoints(
        frame, pts, _SKELETON, visible=np.array([True, True, False, True])
    )

    assert out[40, 80].tolist() == [0, 0, 0], "a NaN point was drawn"
    assert out[40, 120].tolist() == [0, 0, 0], "a hidden point was drawn"
    assert out[40, 40].tolist() == list(visualize.KEYPOINT_COLORS["midline"])


def test_draw_keypoints_draws_an_edge_only_between_two_drawn_points():
    frame = _frame(h=160, w=160, fill=0)
    both = _points((0.25, 0.25), (0.5, 0.5), (0.75, 0.5), (0.25, 0.75))
    one = both.copy()
    one[3] = np.nan

    with_edge = visualize.draw_keypoints(frame, both, _SKELETON)
    without = visualize.draw_keypoints(frame, one, _SKELETON)

    # the nose -> tail_base edge passes through (40, 80), halfway down the left column
    assert with_edge[80, 40].sum() > 0
    assert without[80, 40].tolist() == [0, 0, 0]


def test_draw_keypoints_draws_a_hollow_point_as_a_ring():
    """*Test cut-off…* (§12): a point below the cut-off is placed but not taught, so it is
    drawn as a ring in its role's colour, never a filled dot, and still drawn."""
    frame = _frame(h=160, w=160, fill=0)
    pts = _points((0.25, 0.25), (0.5, 0.25), (0.75, 0.25), (0.25, 0.75))

    out = visualize.draw_keypoints(
        frame, pts, _SKELETON, radius=5, hollow=np.array([False, True, False, False])
    )

    assert out[40, 40].tolist() == list(visualize.KEYPOINT_COLORS["midline"])
    assert out[40, 80].tolist() == [0, 0, 0], "a hollow point's centre was filled"
    ring = out[34:47, 74:87].reshape(-1, 3).tolist()
    assert list(visualize.KEYPOINT_COLORS["left"]) in ring, "the ring was not drawn"


def test_draw_keypoints_dims_a_low_score_rather_than_hiding_it():
    frame = _frame(h=160, w=160, fill=0)
    pts = _points((0.25, 0.25), (0.5, 0.25), (0.75, 0.25), (0.25, 0.75))

    out = visualize.draw_keypoints(
        frame, pts, _SKELETON, scores=np.array([1.0, 0.0, 0.5, np.nan])
    )

    assert out[40, 40].tolist() == list(visualize.KEYPOINT_COLORS["midline"])
    assert out[40, 80].tolist() == [150, 150, 150], "score 0 is grey, still drawn"
    full = np.array(visualize.KEYPOINT_COLORS["right"])
    half = out[40, 120].astype(int)
    assert np.all(np.abs(half - (full + 150) / 2) <= 1)
    assert out[120, 40].tolist() == [150, 150, 150], "a NaN score counts as 0"


def test_draw_keypoints_refuses_the_wrong_number_of_points():
    with pytest.raises(ValueError, match="3 points for a skeleton of 4"):
        visualize.draw_keypoints(
            _frame(), _points((0.1, 0.1), (0.2, 0.2), (0.3, 0.3)), _SKELETON
        )
