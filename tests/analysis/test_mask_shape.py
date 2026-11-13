"""A mask's outline on one frame (``analysis/mask_shape.py``), and the measures built on it.

The classes guarded:

- **A missing mask is NaN, never zero.** Zero is a real shape (a point at the corner), so a zero
  filler would read as an animal.
- **A side-only measure is out of scope from any other view**, including a view nobody set —
  a height measured from above is perspective, not posture.
- **The rhythm measure refuses to read the sampling.** A bout sampled too slowly to see the
  band, or too short to hold a cycle, is NaN, not a small number.
"""

import numpy as np
import pytest

from detail.analysis import mask_shape
from detail.analysis.descriptors import DESCRIPTORS, FrameSlice, in_scope

H, W = 200, 300


def _lying():
    mask = np.zeros((H, W), bool)
    mask[100:140, 50:200] = True
    return mask


def _rearing():
    mask = np.zeros((H, W), bool)
    mask[40:180, 120:160] = True  # tall and narrow
    return mask


def test_a_lying_animal_is_long_low_and_wide_on_the_floor():
    s = mask_shape.frame_shape(_lying(), (H, W))
    assert s.mask_top_norm == pytest.approx(0.5)
    assert s.mask_bottom_norm == pytest.approx(0.7)
    assert s.floor_width_norm == pytest.approx(150 / W)
    assert s.major_axis_norm > 3 * s.minor_axis_norm
    assert s.orientation_deg == pytest.approx(0.0, abs=1.0)
    assert s.solidity == pytest.approx(1.0)
    assert s.protrusions == 0


def test_a_rearing_animal_is_tall_and_narrow_on_the_floor():
    lying = mask_shape.frame_shape(_lying(), (H, W))
    rearing = mask_shape.frame_shape(_rearing(), (H, W))
    height = (rearing.mask_bottom_norm - rearing.mask_top_norm) * H
    assert height == pytest.approx(140)
    assert rearing.floor_width_norm < lying.floor_width_norm
    assert rearing.orientation_deg == pytest.approx(90.0, abs=1.0)


def test_a_limb_sticking_out_is_a_protrusion_and_lowers_solidity():
    mask = _lying()
    mask[60:100, 60:80] = True
    s = mask_shape.frame_shape(mask, (H, W))
    assert s.protrusions >= 1
    assert s.solidity < 0.9


@pytest.mark.parametrize("mask", [None, np.zeros((H, W), bool)])
def test_no_mask_is_nan_never_zero(mask):
    s = mask_shape.frame_shape(mask, (H, W))
    assert all(np.isnan(v) for v in s)


def test_the_sidecar_schema_is_the_shape_fields():
    from detail.storage.zarr_io import KINEMATICS_SHAPE_COLUMNS

    assert KINEMATICS_SHAPE_COLUMNS == mask_shape.FIELDS


# --- the measures --------------------------------------------------------------------- #
def _slice(n=20, *, view="side", heights=None, area=None, rate_hz=30.0):
    t_ns = (np.arange(n) * (1e9 / rate_hz)).astype(np.int64)
    heights = np.full(n, 40.0) if heights is None else np.asarray(heights, float)
    shape = {
        "top": 100.0 - heights,
        "bottom": np.full(n, 100.0),
        "floor_width": np.full(n, 20.0),
        "major": np.full(n, 50.0),
        "minor": np.full(n, 25.0),
        "orientation": np.linspace(0, 170, n) % 180,
        "solidity": np.full(n, 0.9),
        "circularity": np.full(n, 0.6),
        "protrusions": np.full(n, 1.0),
        "protrusion_depth": np.full(n, 5.0),
    }
    return FrameSlice(
        frame_idx=np.arange(n),
        t_ns=t_ns,
        centroid=np.zeros((n, 2)),
        area=np.full(n, 500.0) if area is None else np.asarray(area, float),
        shape=shape,
        body_length=50.0,
        camera_view=view,
    )


def test_heights_are_in_body_lengths():
    s = _slice(heights=np.full(20, 75.0))
    assert DESCRIPTORS["mask_height"].fn(s) == pytest.approx(1.5)
    assert DESCRIPTORS["floor_contact"].fn(s) == pytest.approx(0.4)
    assert DESCRIPTORS["elongation"].fn(s) == pytest.approx(2.0)


@pytest.mark.parametrize("view", ["top_down", "angled_down", "other", None])
def test_side_only_measures_are_out_of_scope_from_other_views(view):
    for name in ("mask_height", "mask_height_peak", "floor_contact"):
        assert not in_scope(DESCRIPTORS[name], _slice(view=view)), (name, view)
        assert in_scope(DESCRIPTORS[name], _slice(view="side"))
    assert in_scope(DESCRIPTORS["solidity"], _slice(view=None)), "not view-gated"


def test_a_still_animal_is_still():
    assert DESCRIPTORS["stillness"].fn(_slice()) == pytest.approx(1.0)


def test_axis_turning_ignores_a_head_tail_flip():
    s = _slice(n=3)
    s.shape["orientation"] = np.array(
        [179.0, 1.0, 179.0]
    )  # 2 degrees each step, not 178
    assert DESCRIPTORS["axis_turning"].fn(s) == pytest.approx(2.0 * 30.0)


def test_rhythm_finds_a_regular_back_and_forth():
    n, rate = 90, 30.0
    t = np.arange(n) / rate
    licking = 500 + 50 * np.sin(2 * np.pi * 6.0 * t)
    noisy = 500 + np.random.default_rng(0).normal(0, 50, n)
    assert DESCRIPTORS["rhythm"].fn(_slice(n=n, area=licking)) > 0.8
    assert DESCRIPTORS["rhythm"].fn(_slice(n=n, area=noisy)) < 0.3


@pytest.mark.parametrize("n, rate", [(90, 4.0), (20, 30.0)])
def test_rhythm_is_nan_when_the_samples_cannot_show_one(n, rate):
    area = 500 + 50 * np.sin(np.arange(n))
    assert np.isnan(DESCRIPTORS["rhythm"].fn(_slice(n=n, area=area, rate_hz=rate)))
