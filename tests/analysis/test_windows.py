"""Windows tile without overlapping, which is what makes a measure additive (§23).

The class: **a boundary convention that double-counts.** With both bounds inclusive a frame on a
shared edge falls in two neighbouring windows and is credited a full period to each, so two halves
of a recording sum to more than the whole. Nothing raises — every individual window looks
reasonable — so the only signal is a sum that does not match.
"""

import pytest

from detail.analysis import windows as w


def test_whole_is_one_window_covering_everything():
    (only,) = w.whole(0, 1000)
    assert (only.window_id, only.start_t_ns, only.end_t_ns) == ("whole", 0, 1000)


def test_absolute_tiles_exactly_once():
    tiles = w.absolute(0, 250, every_ns=100)
    assert sum(t.length_ns for t in tiles) == 250
    for prev, nxt in zip(tiles, tiles[1:]):
        assert prev.end_t_ns == nxt.start_t_ns  # touching, never overlapping


def test_absolute_keeps_a_short_last_window_rather_than_dropping_it():
    """Dropping it loses real time from the tail; extending it past the recording gives the last
    window a denominator no observation can fill."""
    tiles = w.absolute(0, 250, every_ns=100)
    assert (tiles[-1].start_t_ns, tiles[-1].end_t_ns) == (200, 250)


def test_normalized_pins_the_last_edge_to_the_end():
    """Integer division must not leave a sliver of the recording in no window at all."""
    parts = w.normalized(0, 100, parts=3)
    assert parts[-1].end_t_ns == 100
    assert sum(p.length_ns for p in parts) == 100


def test_sliding_windows_may_overlap_which_is_the_point():
    slid = w.sliding(0, 100, length_ns=40, step_ns=20)
    assert slid[0].end_t_ns > slid[1].start_t_ns  # overlapping, deliberately


def test_a_boundary_instant_belongs_to_exactly_one_window():
    """The half-open rule, stated as the property it exists for."""
    a, b = w.absolute(0, 200, every_ns=100)
    assert a.contains_instant(99) and not a.contains_instant(100)
    assert b.contains_instant(100)


def test_overlap_is_half_open():
    assert w.Window("x", 0, 100).overlap_ns(90, 110) == 10
    assert w.Window("x", 0, 100).overlap_ns(100, 110) == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: w.absolute(0, 10, every_ns=0),
        lambda: w.normalized(0, 10, parts=0),
        lambda: w.sliding(0, 10, length_ns=0, step_ns=1),
        lambda: w.sliding(0, 10, length_ns=1, step_ns=0),
        lambda: w.Window("x", 10, 0),
    ],
)
def test_a_window_that_cannot_be_built_raises(call):
    with pytest.raises(w.WindowError):
        call()
