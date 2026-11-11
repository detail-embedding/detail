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
    frame = _frame()
    dets = [_det(0.9, row=2), _det(0.3, row=14)]
    everything = draw_detections(frame, dets, (255, 0, 0))
    filtered = draw_detections(frame, dets, (255, 0, 0), min_conf=0.5)

    # The 0.9 detection is drawn either way.
    assert not np.array_equal(filtered[2:6, 2:6], frame[2:6, 2:6])
    # The 0.3 one is drawn without a floor and gone with one.
    assert not np.array_equal(everything[14:18, 14:18], frame[14:18, 14:18])
    np.testing.assert_array_equal(filtered[14:18, 14:18], frame[14:18, 14:18])


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


# --- the trail primitives (the clip window's; the montage shares `fade_weights`) ---------- #
def _whole_frame_onion_skin(canvas, layers, colour):
    """The onion skin as it was first written: one float32 blend over the whole frame. Kept
    as the reference the bounded blend must equal byte for byte — the slow form a test may
    hold, so the two cannot drift apart about what the fade means."""
    weight = np.zeros(canvas.shape[:2], dtype=np.float32)
    n = len(layers)
    for age, layer in enumerate(reversed(layers)):
        if layer is None:
            continue
        y0, x0, crop = layer
        alpha = np.float32(visualize.ONION_ALPHA * (1.0 - age / n))
        region = weight[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]]
        np.maximum(region, np.where(crop, alpha, np.float32(0.0)), out=region)
    if not weight.any():
        return canvas
    tint = np.asarray(colour, dtype=np.float32)
    share = weight[..., None]
    blended = canvas.astype(np.float32) * (1.0 - share) + tint * share
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def _trail_layers(rng, h, w, n):
    """``n`` random past masks as ``(y0, x0, crop)``, some frames without one."""
    layers = []
    for _ in range(n):
        if rng.random() < 0.2:
            layers.append(None)
            continue
        ch, cw = rng.integers(1, h // 3), rng.integers(1, w // 3)
        y0, x0 = rng.integers(0, h - ch), rng.integers(0, w - cw)
        layers.append((int(y0), int(x0), rng.random((ch, cw)) < 0.6))
    return layers


@pytest.mark.parametrize("seed", range(6))
def test_the_onion_skin_is_the_whole_frame_blend_byte_for_byte(seed):
    rng = np.random.default_rng(seed)
    canvas = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    layers = _trail_layers(rng, 90, 160, int(rng.integers(1, 12)))
    before = canvas.copy()
    np.testing.assert_array_equal(
        visualize.draw_onion_skin(canvas, layers, visualize.MASK_TINT),
        _whole_frame_onion_skin(canvas, layers, visualize.MASK_TINT),
    )
    np.testing.assert_array_equal(
        canvas, before
    )  # the canvas handed in is not drawn on


def test_fade_weights_runs_from_alpha_at_the_newest_to_a_share_at_the_oldest():
    """Newest layer (last in the list) gets the full alpha; age ``k`` gets ``alpha(1-k/n)``;
    where layers overlap the newer one shows."""
    crop = np.ones((2, 2), bool)
    layers = [(0, 0, crop), (1, 1, crop), (0, 0, crop)]  # oldest ... newest
    weight, top, left = visualize.fade_weights(layers)
    assert (top, left) == (0, 0)
    a = visualize.ONION_ALPHA
    assert weight[0, 0] == pytest.approx(a)  # newest covers (0,0)
    assert weight[2, 2] == pytest.approx(
        a * (1 - 1 / 3)
    )  # only the middle layer reaches (2,2)
    assert weight[1, 1] == pytest.approx(a)  # overlap: the newest wins


def test_fade_weights_is_none_when_no_layer_is_present():
    assert visualize.fade_weights([None, None]) is None
    assert visualize.fade_weights([]) is None


class _Watched(np.ndarray):
    """A canvas that records the shape of every float copy taken of it or its slices."""

    shapes: list = []

    def astype(self, dtype, *args, **kwargs):
        if np.dtype(dtype).kind == "f":
            _Watched.shapes.append(self.shape)
        return super().astype(dtype, *args, **kwargs)


def test_the_onion_skin_blends_only_where_the_trail_is():
    """A trail is a few small masks on a large frame: the blend covers their bounding box, not
    the frame. The float copy it makes is that box, 30x40 here, on a 1080p canvas — where the
    whole-frame blend copied all 1920x1080 pixels for every picture drawn (measured on a real
    Raw-off montage: 56 ms of a 119 ms tile)."""
    canvas = np.full((1080, 1920, 3), 255, np.uint8).view(_Watched)
    crop = np.ones((20, 30), bool)
    _Watched.shapes = []
    out = visualize.draw_onion_skin(
        canvas, [(500, 900, crop), None, (510, 910, crop)], visualize.MASK_TINT
    )
    assert _Watched.shapes, "the blend took no float copy: re-derive this guard"
    assert max(s[0] * s[1] for s in _Watched.shapes) == 30 * 40
    assert (np.asarray(out)[:500] == 255).all() and (np.asarray(out)[530:] == 255).all()


def test_fading_paths_break_at_a_nan_and_leave_the_input_alone():
    """A NaN point breaks its path; a point that did not move leaves no trail; the frame handed
    in is never drawn on."""
    frame = np.zeros((60, 80, 3), np.uint8)
    before = frame.copy()
    moving = np.array([[[0.1, 0.5]], [[0.3, 0.5]], [[0.5, 0.5]]])  # (n=3, K=1, 2)
    out = visualize.draw_fading_paths(frame, moving, [(255, 0, 0)])
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert out.any(), "a moving point leaves a trail"
    np.testing.assert_array_equal(frame, before)

    broken = np.array([[[0.1, 0.5]], [[np.nan, np.nan]], [[0.5, 0.5]]])
    assert not visualize.draw_fading_paths(frame, broken, [(255, 0, 0)]).any()

    still = np.array([[[0.4, 0.5]], [[0.4, 0.5]], [[0.4, 0.5]]])
    assert not visualize.draw_fading_paths(frame, still, [(255, 0, 0)]).any()


# --- mask_outline: a ring round the animal, never over it --------------------------------- #
def test_the_outline_sits_outside_the_mask_at_the_asked_thickness():
    from detail.core.visualize import mask_outline

    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    ring = mask_outline(mask, 2)
    assert ring.dtype == bool and ring.shape == mask.shape
    assert not (ring & mask).any(), "the line never covers the animal"
    assert ring[8, 20] and ring[9, 20] and not ring[7, 20], "two pixels wide, outside"
    assert ring[20, 31] and not ring[20, 33]
    assert mask_outline(mask, 0).sum() > 0, "a thickness below one still draws a line"


def test_an_empty_mask_has_no_outline():
    from detail.core.visualize import mask_outline

    assert not mask_outline(np.zeros((8, 8), dtype=bool), 3).any()
