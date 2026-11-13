"""Patch matching (§26): the research method's known-answer checks, with no DINO and no labels.

Ported from the research method's own check script (run before the skeleton decode
met any label) via #626: synthetic score maps over a 24x24 grid, so each check states what the
decode must do with evidence it can see. The backbone is the only thing faked, and only in the
end-to-end checks at the bottom, where a colour-histogram "feature" stands in for DINOv3's (the
external inference call is the one legitimate mock).
"""

import numpy as np
import pytest

from detail.core.examples import ExampleFrame, ExampleMark, ExampleSet
from detail.core.keypoints import InstanceFrame
from detail.core.skeleton import PRESETS, Skeleton
from detail.models.patch_match import decode as dc
from detail.models.patch_match import view as vw
from detail.models.patch_match.dinov3_match import DINOv3Match

SKELETON = PRESETS["quadruped_overhead_12"][1]
PLAN = dc.DecodePlan.from_skeleton(SKELETON)
PATCH = 16
G = 24  # grid cells; the view is the identity, so a cell is 16 px
VIEW = vw.View(0.0, 0.0, 1.0, G * PATCH, PATCH)
SCALE = 200.0

# The animal faces -x (west); cells as (gy, gx). Its left side is image +y (south).
TRUE = {
    "nose": (12, 3),
    "neck_base": (12, 6),
    "mid_back": (12, 10),
    "tail_base": (12, 14),
    "tail_mid": (12, 17),
    "tail_tip": (12, 20),
    "left_ear": (14, 5),
    "right_ear": (10, 5),
    "left_shoulder": (14, 8),
    "right_shoulder": (10, 8),
    "left_hip": (14, 13),
    "right_hip": (10, 13),
}


def px(cell):
    return ((cell[1] + 0.5) * PATCH, (cell[0] + 0.5) * PATCH)


def cell_of(xy):
    return (int(xy[1] // PATCH), int(xy[0] // PATCH))


def bump(center, height, sigma=0.7):
    yy, xx = np.mgrid[0:G, 0:G]
    return height * np.exp(
        -((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * sigma**2)
    )


def maps_with(overrides=None):
    maps = {k: 0.2 + bump(TRUE[k], 0.6) for k in SKELETON.names}
    for k, peaks in (overrides or {}).items():
        s = np.full((G, G), 0.2)
        for c, h in peaks:
            s = np.maximum(s, 0.2 + bump(c, h))
        maps[k] = s
    return {k: v.astype(np.float32) for k, v in maps.items()}


def stats(learned=True, side_rule="soft", truth=None, plan=PLAN):
    if not learned:
        return dc.SkeletonStats()
    clicks = {k: px(c) for k, c in (truth or TRUE).items()}
    return dc.learn_skeleton([(clicks, SCALE)], plan, side_rule)


def decode(maps, side_rule="soft", learned=True, truth=None, plan=PLAN):
    cands = dc.candidates(maps, VIEW)
    return dc.decode_skeleton(
        cands, plan, stats(learned, side_rule, truth, plan), SCALE, VIEW.patch_px
    )


def cells(out, *names):
    return tuple(cell_of(out[n][0]) for n in names)


# --------------------------------------------------------------------------- #
# Side rules
# --------------------------------------------------------------------------- #
def test_sides_are_learned_from_the_clicks():
    learned = stats()
    assert set(learned.sides) == set(PLAN.pairs)
    assert len(set(learned.sides.values())) == 1  # every pair, the same handedness


def test_examples_that_disagree_on_a_side_set_no_side_rule():
    flipped = {k: px(c) for k, c in TRUE.items()}
    flipped["left_ear"], flipped["right_ear"] = (
        flipped["right_ear"],
        flipped["left_ear"],
    )
    learned = dc.learn_skeleton(
        [({k: px(c) for k, c in TRUE.items()}, SCALE), (flipped, SCALE)], PLAN
    )
    assert ("left_ear", "right_ear") not in learned.sides
    assert ("left_hip", "right_hip") in learned.sides


def test_a_collapsed_ear_pair_is_split():
    out = decode(
        maps_with(
            {
                "left_ear": [(TRUE["right_ear"], 0.62), (TRUE["left_ear"], 0.55)],
                "right_ear": [(TRUE["right_ear"], 0.62), (TRUE["left_ear"], 0.50)],
            }
        )
    )
    assert cells(out, "left_ear", "right_ear") == (TRUE["left_ear"], TRUE["right_ear"])


@pytest.mark.parametrize("side_rule", ["soft", "hard"])
def test_a_mild_swap_is_corrected(side_rule):
    out = decode(
        maps_with(
            {
                "left_hip": [(TRUE["right_hip"], 0.60), (TRUE["left_hip"], 0.55)],
                "right_hip": [(TRUE["left_hip"], 0.60), (TRUE["right_hip"], 0.55)],
            }
        ),
        side_rule,
    )
    assert cells(out, "left_hip", "right_hip") == (TRUE["left_hip"], TRUE["right_hip"])


STRONG_SWAP = {
    "left_shoulder": [(TRUE["right_shoulder"], 0.70), (TRUE["left_shoulder"], 0.40)],
    "right_shoulder": [(TRUE["left_shoulder"], 0.70), (TRUE["right_shoulder"], 0.40)],
}


def test_the_soft_rule_leaves_a_swap_strong_evidence_backs():
    out = decode(maps_with(STRONG_SWAP), "soft")
    assert cells(out, "left_shoulder", "right_shoulder") == (
        TRUE["right_shoulder"],
        TRUE["left_shoulder"],
    )


def test_the_hard_rule_never_puts_left_on_the_right():
    out = decode(maps_with(STRONG_SWAP), "hard")
    assert cells(out, "left_shoulder", "right_shoulder") == (
        TRUE["left_shoulder"],
        TRUE["right_shoulder"],
    )


def test_side_rule_off_ignores_sides():
    out = decode(maps_with(STRONG_SWAP), "off")
    assert cells(out, "left_shoulder", "right_shoulder") == (
        TRUE["right_shoulder"],
        TRUE["left_shoulder"],
    )


def test_a_skeleton_listed_tail_first_decodes_the_same():
    """The orientation the decode derives from the edges is a convention: the learned sign
    cancels it, so the hard rule corrects the same swap whichever end the columns start at.
    """
    reordered = Skeleton(
        names=tuple(reversed(SKELETON.names)),
        edges=SKELETON.edges,
        pairs=SKELETON.pairs,
    )
    plan = dc.DecodePlan.from_skeleton(reordered)
    assert plan.midline_path == tuple(reversed(PLAN.midline_path))
    assert plan.parents != PLAN.parents  # a different axis convention...
    out = decode(maps_with(STRONG_SWAP), "hard", plan=plan)
    assert cells(out, "left_shoulder", "right_shoulder") == (  # ...the same answer
        TRUE["left_shoulder"],
        TRUE["right_shoulder"],
    )


# The right hip's best peak is *south* of the midline (gy 13 > 12), beside the left hip: the
# relative order still holds (left is further south), so `hard` accepts both on one side.
SHARED_SIDE = {
    "right_hip": [((13, 13), 0.65), (TRUE["right_hip"], 0.55)],
}


def test_hard_lets_a_pair_share_a_side_of_the_midline():
    out = decode(maps_with(SHARED_SIDE), "hard")
    assert cells(out, "left_hip", "right_hip") == (TRUE["left_hip"], (13, 13))


@pytest.mark.parametrize("side_rule", ["midline", "auto"])
def test_midline_puts_each_member_on_its_own_side(side_rule):
    out = decode(maps_with(SHARED_SIDE), side_rule)
    assert cells(out, "left_hip", "right_hip") == (TRUE["left_hip"], TRUE["right_hip"])


def test_auto_picks_midline_from_top_view_clicks_and_hard_from_side_view_clicks():
    assert set(stats(side_rule="auto").rules.values()) == {"midline"}
    # From the side, a near and a far ear both sit above the body: no straddle, same order.
    side_view = dict(TRUE, left_ear=(15, 5), right_ear=(14, 5))
    rules = stats(side_rule="auto", truth=side_view).rules
    assert rules[("left_ear", "right_ear")] == "hard"
    assert rules[("left_hip", "right_hip")] == "midline"


def only_near(score, cell):
    """``score`` forbidden outside the 3x3 around ``cell`` — what the subject's mask does to a
    real map. Without it the synthetic background offers a candidate everywhere, so some
    combination always keeps the rule and no search ever has to widen or relax."""
    out = np.full_like(score, -np.inf)
    ys = slice(cell[0] - 1, cell[0] + 2)
    xs = slice(cell[1] - 1, cell[1] + 2)
    out[ys, xs] = score[ys, xs]
    return out


def test_midline_relaxes_to_hard_when_no_candidate_keeps_it():
    # the right hip has one place, south of the midline: midline cannot hold, hard can
    maps = maps_with({"right_hip": [((13, 13), 0.65)]})
    maps["right_hip"] = only_near(maps["right_hip"], (13, 13))
    out = decode(maps, "midline")
    assert cells(out, "left_hip", "right_hip") == (TRUE["left_hip"], (13, 13))


def test_a_rule_searches_past_five_candidates_before_giving_up():
    """Five strong peaks on the wrong side and the right place sixth: the research's five
    candidates found no combination keeping the rule and fell back to the penalty, which five
    units of evidence beat; the wider search finds the sixth."""
    wrong = [
        ((9, 5), 0.7),
        ((9, 8), 0.7),
        ((9, 11), 0.7),
        ((7, 6), 0.7),
        ((7, 10), 0.7),
    ]
    maps = maps_with({"left_shoulder": wrong + [(TRUE["left_shoulder"], 0.45)]})
    maps["right_shoulder"] = only_near(maps["right_shoulder"], TRUE["right_shoulder"])
    assert len(dc.candidates(maps, VIEW)["nms"]["left_shoulder"]) == dc.SKEL_TOPK_WIDE
    out = decode(maps, "hard")
    assert cell_of(out["left_shoulder"][0]) == TRUE["left_shoulder"]
    # ...where the research's five-candidate search gave the rule up for the penalty
    original = dc._PASSES["hard"]
    dc._PASSES["hard"] = (("hard", dc.SKEL_TOPK), ("penalty", dc.SKEL_TOPK))
    try:
        narrow = decode(maps, "hard")
    finally:
        dc._PASSES["hard"] = original
    assert cell_of(narrow["left_shoulder"][0]) != TRUE["left_shoulder"]


# --------------------------------------------------------------------------- #
# The skeleton as the decode reads it: no front, no back
# --------------------------------------------------------------------------- #
def test_the_midline_is_the_longest_chain_through_unpaired_points():
    assert PLAN.midline_path == (
        "nose",
        "neck_base",
        "mid_back",
        "tail_base",
        "tail_mid",
        "tail_tip",
    )
    side = dc.DecodePlan.from_skeleton(PRESETS["quadruped_side_5"][1])
    assert side.midline_path == ("nose", "mid_back", "tail_base")
    two = Skeleton(names=("a", "b"), edges=(("a", "b"),))
    assert dc.DecodePlan.from_skeleton(two).midline_path == ("a", "b")
    all_paired = Skeleton(
        names=("l", "r", "a"), edges=(("a", "l"), ("a", "r")), pairs=(("l", "r"),)
    )
    assert dc.DecodePlan.from_skeleton(all_paired).midline_path == ()


def test_a_pair_hangs_off_the_unpaired_point_both_members_join():
    assert PLAN.parents[("left_ear", "right_ear")] == ("neck_base", "nose", +1)
    assert PLAN.parents[("left_hip", "right_hip")] == ("tail_base", "mid_back", +1)
    side = dc.DecodePlan.from_skeleton(PRESETS["quadruped_side_5"][1])
    # the ears hang off the nose, the chain's start: the axis runs the other way, reversed
    assert side.parents == {("left_ear", "right_ear"): ("nose", "mid_back", -1)}


def test_midline_side_is_judged_at_the_nearest_segment():
    # a bent midline: start at the top, then a turn east; segments point toward the start
    path = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    assert dc.midline_side((-1.0, 5.0), path) == -dc.midline_side((1.0, 5.0), path)
    assert dc.midline_side((5.0, 11.0), path) == -dc.midline_side((5.0, 9.0), path)
    assert dc.midline_side((1.0, 1.0), path[:1]) == 0


def test_resolved_settings_default_to_auto_and_refuse_what_the_skeleton_cannot_honour():
    resolve = DINOv3Match.resolve_settings
    assert resolve(SKELETON, None, None, None) == {
        "decode": "skeleton",
        "side_rule": "auto",
        "crop_px": vw.DEFAULT_CROP_PX,
    }
    no_edges = Skeleton(names=("a", "b"))
    assert resolve(no_edges, None, None, 672)["decode"] == "per_point"
    with pytest.raises(ValueError, match="needs edges"):
        resolve(no_edges, "skeleton", None, None)
    paired_no_midline = Skeleton(
        names=("a", "l", "r"), edges=(("a", "l"), ("a", "r")), pairs=(("l", "r"),)
    )
    assert resolve(paired_no_midline, None, None, None)["side_rule"] == "auto"
    with pytest.raises(ValueError, match="needs a midline"):
        resolve(paired_no_midline, None, "midline", None)
    with pytest.raises(ValueError, match="needs a left/right pair"):
        resolve(Skeleton(names=("a", "b"), edges=(("a", "b"),)), None, "hard", None)
    with pytest.raises(ValueError, match="works through the skeleton decode"):
        resolve(SKELETON, "per_point", "hard", None)
    with pytest.raises(ValueError, match="multiple of 16"):
        resolve(SKELETON, None, None, 450)


# --------------------------------------------------------------------------- #
# Decode geometry
# --------------------------------------------------------------------------- #
def test_flat_skeleton_terms_reduce_to_each_points_best_peak():
    maps = maps_with()
    out = decode(maps, learned=False)
    per_point = dc.decode_per_point(dc.candidates(maps, VIEW))
    for k in SKELETON.names:
        assert cell_of(out[k][0]) == cell_of(per_point[k][0]) == TRUE[k]


def test_flat_terms_with_one_collapsed_pair_keep_the_rest_and_split_the_pair():
    maps = maps_with(
        {"left_ear": [(TRUE["right_ear"], 0.62), (TRUE["left_ear"], 0.55)]}
    )
    out = decode(maps, learned=False)
    for k in SKELETON.names:
        if k not in ("left_ear", "right_ear"):
            assert cell_of(out[k][0]) == TRUE[k]
    assert cell_of(out["left_ear"][0]) != cell_of(out["right_ear"][0])


def test_a_midline_decoy_is_pulled_back_by_the_edge_lengths():
    out = decode(
        maps_with({"tail_base": [((12, 21), 0.62), (TRUE["tail_base"], 0.58)]})
    )
    assert cell_of(out["tail_base"][0]) == TRUE["tail_base"]


def test_a_broad_peak_cannot_slide_toward_a_wrong_neighbour():
    """Midline candidates are true local maxima, so a wrong mid back cannot drag the neck base
    along its own wide peak (the greedy candidates let it slide in the research)."""
    maps = maps_with({"mid_back": [((12, 17), 0.65)]})
    yy, xx = np.mgrid[0:G, 0:G]
    maps["neck_base"] = (
        0.2
        + 0.6
        * np.exp(
            -((yy - TRUE["neck_base"][0]) ** 2 + (xx - TRUE["neck_base"][1]) ** 2)
            / (2 * 1.8**2)
        )
    ).astype(np.float32)
    out = decode(maps)
    assert cell_of(out["neck_base"][0]) == TRUE["neck_base"]


def test_a_point_with_nothing_inside_the_mask_is_not_placed():
    maps = maps_with()
    maps["tail_tip"] = np.full((G, G), -np.inf, dtype=np.float32)
    assert "tail_tip" not in decode(maps)
    assert "tail_tip" not in dc.decode_per_point(dc.candidates(maps, VIEW))


def test_a_candidate_remembers_the_cell_it_came_from():
    cands = dc.candidates(maps_with(), VIEW)
    xy, score, cell = cands["peaks"]["nose"][0]
    assert cell == TRUE["nose"] and cell_of(xy) == TRUE["nose"]
    assert score == pytest.approx(0.8, abs=0.01)
    pool = dc.candidate_pool(cands, PLAN)
    assert pool["nose"] is cands["peaks"]["nose"]  # a midline point: true peaks
    assert pool["left_ear"] is cands["nms"]["left_ear"]  # a pair member: suppression


def test_refine_peak_is_sub_patch_and_symmetric():
    s = np.zeros((5, 5), dtype=np.float32)
    s[2, 2], s[2, 3] = 1.0, 1.0
    gy, gx = dc.refine_peak(s, 2, 2)
    assert gy == pytest.approx(2.0)
    assert gx == pytest.approx(2.5, abs=1e-3)


# --------------------------------------------------------------------------- #
# The view and the mask
# --------------------------------------------------------------------------- #
def test_the_view_maps_back_to_the_source():
    view = vw.make_view((0.25, 0.4, 0.5, 0.6), (1920, 1080), 448, 16)
    assert view.size == 448 and view.grid == 28
    for x, y in ((480.0, 432.0), (960.0, 648.0)):
        u, v = view.to_view(x, y)
        assert view.to_source(u, v) == pytest.approx((x, y))
    # the longer side of the box, padded 15% each way, fills the crop
    assert 448 / view.s == pytest.approx(0.25 * 1920 * 1.3)
    assert view.patch_px == pytest.approx(16 / view.s)
    with pytest.raises(ValueError, match="multiple of the patch size 16"):
        vw.make_view((0.25, 0.4, 0.5, 0.6), (1920, 1080), 450, 16)


def test_allowed_cells_cover_the_mask_dilated_by_one_patch():
    frame_wh = (320, 320)
    mask = np.zeros((320, 320), dtype=bool)
    mask[150:170, 150:170] = True
    view = vw.View(0.0, 0.0, 1.0, 320, 16)
    allowed = vw.allowed_cells(mask, view, frame_wh)
    assert allowed.shape == (20, 20)
    assert allowed[9:11, 9:11].all()
    assert not allowed[0, 0] and not allowed[19, 19]
    assert vw.allowed_cells(np.zeros_like(mask), view, frame_wh) is None


def test_a_full_frame_click_lands_on_the_same_pixel_in_the_matching_crop():
    """The labeling screen stores marks in full-frame normalized coordinates; the engine crops
    a square around the tracked box and resizes it before matching. The example descriptor
    must come from the pixel the user clicked — a dot's centre must land on the dot in the
    rendered crop, whatever the crop's offset and scale."""
    for frame_wh, bbox, dot in (
        ((800, 600), (0.18, 0.29, 0.46, 0.75), (300, 330)),  # a real run's shape
        ((1920, 1080), (0.40, 0.20, 0.52, 0.70), (900, 500)),  # a tall box on 1080p
        ((640, 480), (0.05, 0.10, 0.30, 0.25), (60, 70)),  # crop runs off the frame
    ):
        w, h = frame_wh
        frame = np.zeros((h, w, 3), np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        frame[(xx - dot[0]) ** 2 + (yy - dot[1]) ** 2 <= 6**2] = (255, 0, 0)
        mark = ExampleMark(
            "nose", x=dot[0] / w, y=dot[1] / h
        )  # what the labeler stores
        view = vw.make_view(bbox, frame_wh, 448, 16)
        crop = vw.render(frame, view)
        u, v = view.to_view(mark.x * w, mark.y * h)  # what load() samples at
        assert 0 <= u < 448 and 0 <= v < 448
        assert tuple(crop[int(round(v)), int(round(u))]) == (255, 0, 0)
        assert (u, v) != pytest.approx(
            dot, abs=2
        )  # not an identity passing by accident


# --------------------------------------------------------------------------- #
# The registered method
# --------------------------------------------------------------------------- #
def test_dinov3_match_is_a_keypoint_model_on_the_vit_weights():
    from detail.api import models as api_models
    from detail.core.registries import KEYPOINT_MODELS
    from detail.models.dinov3 import DINOv3ViT

    assert KEYPOINT_MODELS["dinov3_match"] is DINOv3Match
    assert api_models.all_models()["dinov3_match"] == ("keypoint", DINOv3Match)
    assert DINOv3Match.shares_weights_with == "dinov3_vit"
    for variant in DINOv3Match.variants():
        assert DINOv3Match(variant).hf_repo == DINOv3ViT(variant).hf_repo
    assert DINOv3Match.default_variant() == "vit_large"
    assert (DINOv3Match.needs_examples, DINOv3Match.needs_mask) == (True, True)
    assert DINOv3Match.per_detection and not DINOv3Match.reports_confidence
    assert DINOv3Match.n_candidates == 5
    assert DINOv3Match.score_spec.meaning == "cosine similarity"
    assert DINOv3Match.default_track_method == "temporal_decode"
    assert DINOv3Match().patch_size == 16


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"decode": "joint"}, "unknown decode"),
        ({"side_rule": "strict"}, "unknown side_rule"),
        ({"crop_px": 450}, "multiple of 16"),
        ({"example_images": []}, "together"),
    ],
)
def test_constructor_refuses_unknown_settings(kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        DINOv3Match(**kwargs)


def test_without_an_example_set_there_is_no_skeleton_and_no_load():
    model = DINOv3Match()
    with pytest.raises(ValueError, match="example set"):
        model.skeleton()
    with pytest.raises(ValueError, match="needs an example set"):
        model.load("cpu")
    assert model.resolved() == {}


# --------------------------------------------------------------------------- #
# End to end through load() and predict_batch(), with a stand-in backbone
# --------------------------------------------------------------------------- #
PALETTE = np.array(
    [[128, 128, 128], [255, 0, 0], [0, 200, 0], [0, 0, 255], [0, 0, 0]],
    dtype=np.float32,
)


class _ColourBackbone:
    """Per patch, the histogram of nearest-palette colours: a feature that identifies a dot."""

    patch_size = 16
    hf_repo = "stand-in"

    def load(self, device):
        self.device = device

    def patch_grids(self, images):
        torch = pytest.importorskip("torch")
        out = []
        for im in images:
            d = ((im[..., None, :].astype(np.float32) - PALETTE) ** 2).sum(-1)
            onehot = np.eye(len(PALETTE), dtype=np.float32)[d.argmin(-1)]
            g = im.shape[0] // 16
            hist = onehot.reshape(g, 16, g, 16, -1).mean(axis=(1, 3))
            grid = torch.from_numpy(hist).permute(2, 0, 1)
            out.append(torch.nn.functional.normalize(grid, dim=0))
        return out


def _frame(points_px, size=(640, 480)):
    w, h = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    body = ((xx - 320) / 150.0) ** 2 + ((yy - 240) / 60.0) ** 2 <= 1
    frame[body] = (128, 128, 128)
    for colour, (x, y) in zip(PALETTE[1:4].astype(np.uint8), points_px):
        frame[(xx - x) ** 2 + (yy - y) ** 2 <= 10**2] = colour
    return frame, body


DOTS = Skeleton(
    names=("nose", "mid_back", "tail_base"),
    edges=(("nose", "mid_back"), ("mid_back", "tail_base")),
)
BBOX = (170 / 640, 180 / 480, 470 / 640, 300 / 480)


def _examples(ref_pts):
    return ExampleSet(
        id="ex_01",
        name="dots",
        purpose="example",
        created_at="",
        skeleton=DOTS,
        frames=[
            ExampleFrame(
                image="frames/000.png",
                recording_id="recording_001",
                tracking_run_id="track_01",
                track_id=0,
                part_id="part_001",
                frame_idx=0,
                t_ns=0,
                bbox=BBOX,
                marks=[
                    ExampleMark(n, x=x / 640, y=y / 480)
                    for n, (x, y) in zip(DOTS.names, ref_pts)
                ],
            )
        ],
    )


def _loaded(monkeypatch, **kwargs):
    ref_pts = [(200, 240), (320, 240), (440, 240)]
    ref_img, _ = _frame(ref_pts)
    model = DINOv3Match(examples=_examples(ref_pts), example_images=[ref_img], **kwargs)
    monkeypatch.setattr(model, "_make_backbone", lambda: _ColourBackbone())
    model.load("cpu")
    return model


def test_end_to_end_matches_marked_dots_on_a_new_frame(monkeypatch):
    pytest.importorskip("torch")
    model = _loaded(monkeypatch, crop_px=448)
    new_pts = [(230, 250), (330, 230), (420, 236)]  # the animal moved; same dots
    frame, body = _frame(new_pts)
    empty = InstanceFrame(frame_idx=4, t_ns=40, frame=frame, boxes={}, masks={})
    nowhere = np.zeros_like(body)
    [result, none] = model.predict_batch(
        [
            InstanceFrame(
                frame_idx=3,
                t_ns=30,
                frame=frame,
                boxes={12: BBOX, 7: BBOX},
                masks={7: body, 12: nowhere},
            ),
            empty,
        ]
    )
    # one instance per detection, keyed back by its detection row; no subject
    assert (result.frame_idx, result.t_ns, result.n) == (3, 30, 2)
    assert result.detection_row.tolist() == [7, 12] and result.track_id is None
    np.testing.assert_allclose(result.bbox, [BBOX, BBOX])
    assert np.isnan(result.confidence).all()  # reports_confidence = False
    assert result.points.shape == (2, 3, 2)
    got = result.points[0] * np.array([640, 480])
    assert np.abs(got - np.array(new_pts)).max() < 16  # within one patch of each dot
    assert np.isfinite(result.point_scores).all()
    assert result.candidates.shape == (2, 3, 5, 2)
    assert result.candidate_scores.shape == (2, 3, 5)
    assert result.extras["ref_index"].shape == (2, 3, 5)
    assert result.extras["ref_index"].dtype == np.int16
    # the winning example row of the best candidate is this target's own example
    for j, name in enumerate(DOTS.names):
        row = int(result.extras["ref_index"][0, j, 0])
        assert model._learned[None].bank.sources[row] == (0, name)
    assert result.extras["mirrored"].shape == (2, 3)
    assert np.isfinite(result.extras["mirrored"]).all()
    # a frame with no detection: zero rows, every column still shaped
    assert none.n == 0 and none.points.shape == (0, 3, 2)
    assert none.candidates.shape == (0, 3, 5, 2) and none.detection_row.shape == (0,)
    assert none.bbox.shape == (0, 4) and none.confidence.shape == (0,)


def test_resolved_records_the_side_rules_and_the_matching_constants(monkeypatch):
    pytest.importorskip("torch")
    model = _loaded(monkeypatch)
    assert model.settings == {"decode": "skeleton", "side_rule": "off", "crop_px": 448}
    assert model.resolved() == {"side_rules": {}, "matching": dc.MATCHING}
    assert set(dc.MATCHING) == {
        "refine_tau",
        "skel_tau",
        "topk",
        "topk_wide",
        "len_tol",
        "len_floor",
        "side_penalty",
    }


def test_load_refuses_an_example_set_a_run_cannot_learn_from(monkeypatch):
    pytest.importorskip("torch")
    ref_pts = [(200, 240), (320, 240), (440, 240)]
    ref_img, _ = _frame(ref_pts)
    examples = _examples(ref_pts)
    examples.frames[0].marks[0] = ExampleMark("nose", visible=False)
    model = DINOv3Match(examples=examples, example_images=[ref_img])
    monkeypatch.setattr(model, "_make_backbone", lambda: _ColourBackbone())
    with pytest.raises(ValueError, match=r"no frame marks \['nose'\] visible"):
        model.load("cpu")


def test_each_example_is_scored_by_the_others(monkeypatch):
    """The held-out check behind *Test cut-off…*: every example frame placed by a bank built
    from the other frames only, measured against the user's own clicks."""
    pytest.importorskip("torch")
    positions = [
        [(200, 240), (320, 240), (440, 240)],
        [(220, 250), (330, 230), (430, 236)],
        [(210, 236), (310, 246), (445, 244)],
    ]
    images = [_frame(pts)[0] for pts in positions]
    examples = _examples(positions[0])
    examples.frames = [
        ExampleFrame(
            image=f"frames/{i:03d}.png",
            recording_id="recording_001",
            tracking_run_id="track_01",
            track_id=0,
            part_id="part_001",
            frame_idx=i,
            t_ns=i,
            bbox=BBOX,
            marks=[
                ExampleMark(n, x=x / 640, y=y / 480)
                for n, (x, y) in zip(DOTS.names, pts)
            ],
        )
        for i, pts in enumerate(positions)
    ]
    model = DINOv3Match(examples=examples, example_images=images, crop_px=448)
    monkeypatch.setattr(model, "_make_backbone", lambda: _ColourBackbone())
    model.load("cpu")
    bank_before = model._learned[None].bank

    rows = model.score_held_out()

    assert len(rows) == 9, "every (example frame, visible point)"
    assert {(r["frame"], r["point"]) for r in rows} == {
        (i, n) for i in range(3) for n in DOTS.names
    }
    for r in rows:
        assert r["error_px"] < 16, r  # within one patch of the user's click
        assert np.isfinite(r["score"]) and r["body_px"] > 0
    assert model._learned[None].bank is bank_before, "the run's own bank is untouched"


def test_one_example_frame_has_nothing_to_hold_out(monkeypatch):
    pytest.importorskip("torch")
    model = _loaded(monkeypatch)
    assert model.score_held_out() == []


# -- the split keypoint_detect's stages run (§26 *Execution*) ------------------------------ #
# ``predict_batch`` is ``finish_batch(run_prepared(prepare_batch(frames)))``; the stage runs the
# three halves on three threads. What can go wrong is pairing: a crop's similarities read
# against another detection's view or mask, or the main bank's against the mirror's.


def _split_frames():
    """Frames whose detections differ in place, box and mask, so a mispairing moves a number:
    two detections with different masks, one with no mask, one with a mask that is all zero
    (no constraint), a frame with no detection, and the animal moved."""
    a_pts = [(230, 250), (330, 230), (420, 236)]
    b_pts = [(210, 236), (318, 244), (430, 240)]
    frame_a, body_a = _frame(a_pts)
    frame_b, body_b = _frame(b_pts)
    half = body_a.copy()
    half[:, :400] = (
        False  # the tail end only: its own view allows the tail dot, the other not
    )
    shifted = (BBOX[0] + 0.1, BBOX[1], BBOX[2] + 0.1, BBOX[3])  # four patches over
    return [
        InstanceFrame(
            frame_idx=0,
            t_ns=0,
            frame=frame_a,
            boxes={7: BBOX, 12: shifted},
            masks={7: body_a, 12: half},
        ),
        InstanceFrame(frame_idx=1, t_ns=10, frame=frame_b, boxes={3: BBOX}, masks={}),
        InstanceFrame(frame_idx=2, t_ns=20, frame=frame_b, boxes={}, masks={}),
        InstanceFrame(
            frame_idx=3,
            t_ns=30,
            frame=frame_b,
            boxes={0: shifted},
            masks={0: np.zeros_like(body_b)},
        ),
    ]


_FIELDS = ("bbox", "detection_row", "points", "point_scores", "candidates")


def _same(a, b) -> None:
    assert (a.frame_idx, a.t_ns, a.n) == (b.frame_idx, b.t_ns, b.n)
    for name in (*_FIELDS, "candidate_scores", "confidence"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert a.extras.keys() == b.extras.keys()
    for name in a.extras:
        np.testing.assert_array_equal(a.extras[name], b.extras[name], err_msg=name)


def test_a_batch_scores_each_frame_as_it_would_alone(monkeypatch):
    pytest.importorskip("torch")
    model = _loaded(monkeypatch, crop_px=448)
    frames = _split_frames()
    alone = [r for f in frames for r in model.predict_batch([f])]
    together = model.predict_batch(frames)
    regrouped = model.predict_batch(frames[:3]) + model.predict_batch(frames[3:])
    assert [r.n for r in together] == [2, 1, 0, 1]
    for a, b, c in zip(alone, together, regrouped):
        _same(a, b)
        _same(a, c)


def test_each_detection_is_scored_on_its_own_crop_and_mask(monkeypatch):
    """An independent route per detection: its own view, crop, grid and mask through the
    bank's public ``score_maps`` / ``best_scores`` and the decode, against what the halves
    produced for it inside a batch."""
    pytest.importorskip("torch")
    model = _loaded(monkeypatch, crop_px=448)
    frames = _split_frames()
    results = model.predict_batch(frames)
    backbone = _ColourBackbone()
    checked = 0
    for sf, result in zip(frames, results):
        wh = (sf.frame.shape[1], sf.frame.shape[0])
        for i, det_row in enumerate(sorted(sf.boxes)):
            view = vw.make_view(sf.boxes[det_row], wh, 448, PATCH)
            (grid,) = backbone.patch_grids([vw.render(sf.frame, view)])
            allowed = vw.allowed_cells(sf.masks.get(det_row), view, wh)
            maps, _ = model._learned[None].bank.score_maps(grid, allowed)
            pool = dc.candidate_pool(dc.candidates(maps, view), model._plan)
            mirrored = model._learned[None].mirror_bank.best_scores(grid, allowed)
            for j, name in enumerate(model._plan.names):
                want = [score for _xy, score, _cell in pool.get(name, [])[:5]]
                got = result.candidate_scores[i, j, : len(want)]
                np.testing.assert_array_equal(got, np.float32(want), err_msg=name)
                assert result.extras["mirrored"][i, j] == np.float32(mirrored[name])
            checked += 1
    assert checked == 4


def test_the_model_thread_hands_the_writer_host_arrays_only(monkeypatch):
    """Every copy off the device happens in ``run_prepared``: a ``.cpu()`` on the writer
    thread would queue behind the next batch's forward on the default CUDA stream, which
    no CPU test can time, so the guard is that no tensor crosses."""
    torch = pytest.importorskip("torch")
    model = _loaded(monkeypatch, crop_px=448)
    raw = model.run_prepared(model.prepare_batch(_split_frames()))

    def tensors(x):
        if isinstance(x, torch.Tensor):
            yield x
        elif isinstance(x, (list, tuple)):
            for y in x:
                yield from tensors(y)
        elif isinstance(x, dict):
            for y in x.values():
                yield from tensors(y)

    assert list(tensors(raw)) == []
    assert isinstance(raw[2], np.ndarray) and isinstance(raw[3], np.ndarray)
