"""One position per behavior instance, from its frames (§19 ``instance_features.csv``).

What the spatial map plots. The failures this guards are all "a plausible number in the wrong
place" rather than crashes: a span matched to the wrong rows, an all-NaN bout filled with a
coordinate, or an unsorted lookup returning a silently wrong slice.
"""

import numpy as np
import pytest

from detail.analysis import positions
from detail.storage import behavior_output as bo


def _instance(instance_id: str, t_ns: int, end_t_ns: int, **kw) -> bo.BehaviorInstance:
    data = dict(
        recording_id="recording_001",
        subject=0,
        t_ns=t_ns,
        end_t_ns=end_t_ns,
        start_frame=0,
        end_frame=0,
        source_label="3",
        instance_id=instance_id,
    )
    data.update(kw)
    return bo.BehaviorInstance(**data)


def _kin(t_ns, cx, cy) -> dict:
    return {
        "t_ns": np.asarray(t_ns, dtype=np.int64),
        "cx_norm": np.asarray(cx, dtype=np.float64),
        "cy_norm": np.asarray(cy, dtype=np.float64),
    }


UNIT = ("recording_001", 0)


def test_a_bouts_position_is_the_mean_over_its_own_frames():
    kin = {UNIT: _kin([0, 100, 200, 300, 400], [0.0, 0.2, 0.4, 0.6, 0.8], [0.5] * 5)}
    got = positions.instance_positions([_instance("a", 100, 300)], kin)
    # Rows at 100, 200, 300 — inclusive at both ends, which is what a §19 span means.
    assert got["a"] == pytest.approx((0.4, 0.5))


def test_the_span_is_inclusive_at_both_ends():
    """Off by one row at either end is a real position, just not this bout's."""
    kin = {UNIT: _kin([0, 10, 20], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0])}
    only_middle = positions.instance_positions([_instance("a", 10, 10)], kin)
    assert only_middle["a"] == pytest.approx((1.0, 1.0))
    all_three = positions.instance_positions([_instance("b", 0, 20)], kin)
    assert all_three["b"] == pytest.approx((1 / 3, 1 / 3))


def test_two_bouts_in_one_unit_get_their_own_positions():
    kin = {UNIT: _kin([0, 100, 200, 300], [0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8])}
    got = positions.instance_positions(
        [_instance("early", 0, 100), _instance("late", 200, 300)], kin
    )
    assert got["early"] == pytest.approx((0.1, 0.2))
    assert got["late"] == pytest.approx((0.9, 0.8))


def test_a_bout_whose_frames_are_all_undefined_reports_nothing():
    """NaN, not a coordinate. A filled-in zero is the top-left corner and plots as a real
    reading — the same rule the geometry primitives follow."""
    kin = {UNIT: _kin([0, 10], [np.nan, np.nan], [np.nan, np.nan])}
    got = positions.instance_positions([_instance("a", 0, 10)], kin)
    assert np.isnan(got["a"][0]) and np.isnan(got["a"][1])


def test_undefined_frames_are_skipped_rather_than_poisoning_the_mean():
    kin = {UNIT: _kin([0, 10, 20], [0.2, np.nan, 0.4], [0.6, np.nan, 0.8])}
    got = positions.instance_positions([_instance("a", 0, 20)], kin)
    assert got["a"] == pytest.approx((0.3, 0.7))


def test_an_instance_from_a_unit_with_no_kinematics_reports_nothing():
    got = positions.instance_positions([_instance("a", 0, 10)], {})
    assert np.isnan(got["a"][0])


def test_each_unit_reads_its_own_kinematics():
    """Two recordings' rows share a time axis; matching by time alone across units would put
    one recording's position on the other's bout."""
    kin = {
        ("recording_001", 0): _kin([0, 10], [0.1, 0.1], [0.1, 0.1]),
        ("recording_002", 0): _kin([0, 10], [0.9, 0.9], [0.9, 0.9]),
    }
    got = positions.instance_positions(
        [
            _instance("a", 0, 10),
            _instance("b", 0, 10, recording_id="recording_002"),
        ],
        kin,
    )
    assert got["a"] == pytest.approx((0.1, 0.1))
    assert got["b"] == pytest.approx((0.9, 0.9))


def test_unsorted_kinematics_are_sorted_before_the_binary_search():
    """`searchsorted` on unsorted input returns a wrong span rather than raising, and every
    position built on it would be a plausible number."""
    kin = {UNIT: _kin([200, 0, 100], [0.9, 0.1, 0.5], [0.9, 0.1, 0.5])}
    got = positions.instance_positions([_instance("a", 0, 100)], kin)
    assert got["a"] == pytest.approx(
        (0.3, 0.3)
    )  # the rows at 0 and 100, not the first two
