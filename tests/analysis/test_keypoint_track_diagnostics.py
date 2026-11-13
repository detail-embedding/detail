"""``keypoint_track_diagnostics``: the label-free figures under *Preview tracking*'s clips.

Each figure on a series built to have exactly one of the things it counts, so a figure that
counted the wrong thing (a gap read as a move, a crossing read as two swaps) is off by a known
amount.
"""

import numpy as np
import pytest

from detail.analysis import keypoint_track_diagnostics as diag
from detail.core.skeleton import Skeleton

SKELETON = Skeleton(
    names=("nose", "left_ear", "right_ear"),
    edges=(("nose", "left_ear"), ("nose", "right_ear"), ("left_ear", "right_ear")),
    pairs=(("left_ear", "right_ear"),),
)
FPS = 30.0
STEP_NS = int(1e9 / FPS)
SCALE = 100.0
STILL = np.array([[50.0, 20.0], [30.0, 60.0], [70.0, 60.0]])
ALL_TRACKED = np.ones(3, dtype=bool)


def _series(n: int = 61):
    points = np.repeat(STILL[None], n, axis=0)
    t_ns = np.arange(n, dtype=np.int64) * STEP_NS
    return points, t_ns, np.full(n, SCALE)


def _tally(points, t_ns, scale, tracked=ALL_TRACKED, detected=None, scores=None):
    """Detection, unless given, is the still animal, every point scored alike."""
    if detected is None:
        detected = _series(len(t_ns))[0]
    if scores is None:
        scores = np.ones(detected.shape[:2])
    return diag.tally(
        points,
        t_ns,
        scale,
        SKELETON,
        tracked,
        FPS,
        detected_px=detected,
        detected_scores=scores,
    )


def _figures(points, t_ns, scale, tracked=ALL_TRACKED, **detection):
    return _tally(points, t_ns, scale, tracked, **detection).figures()


def test_a_still_animal_moves_nothing_jumps_never_and_keeps_its_shape():
    figures = _figures(*_series())
    assert figures == pytest.approx(
        {
            "placed": 1.0,
            "median_move": 0.0,
            "jumps_per_min": 0.0,
            "swaps_per_min": 0.0,
            "stretch": 0.0,
            "agreement": 1.0,
        },
        abs=1e-12,
    )


def test_a_one_row_spike_is_two_jumps_out_and_back():
    points, t_ns, scale = _series()
    points[30, 0] += (0.0, 0.8 * SCALE)  # the nose leaves the head for one row
    figures = _figures(points, t_ns, scale)
    minutes = 60 / FPS / 60  # 60 consecutive steps
    assert figures["jumps_per_min"] == pytest.approx(2 / minutes)
    assert figures["median_move"] == 0.0
    assert figures["stretch"] > 0


def test_a_left_right_exchange_is_one_swap():
    points, t_ns, scale = _series()
    points[30:, [1, 2]] = points[30:, [2, 1]]
    figures = _figures(points, t_ns, scale)
    assert figures["swaps_per_min"] == pytest.approx(1 / (60 / FPS / 60))
    # the edges keep their lengths through a swap
    assert figures["stretch"] == pytest.approx(0.0, abs=1e-12)


def test_a_gap_in_time_is_not_a_move():
    points, t_ns, scale = _series()
    t_ns[30:] += 10 * STEP_NS  # the subject was not detected for ten rows
    points[30:, :, 0] += 0.8 * SCALE  # and came back somewhere else
    figures = _figures(points, t_ns, scale)
    assert figures["jumps_per_min"] == 0.0
    assert figures["median_move"] == 0.0


def test_an_unplaced_point_counts_against_placed_and_nothing_else():
    points, t_ns, scale = _series()
    points[:10, 0] = np.nan
    figures = _figures(points, t_ns, scale)
    assert figures["placed"] == pytest.approx(1 - 10 / (61 * 3))
    assert figures["median_move"] == 0.0


def test_a_point_left_as_detected_moves_no_figure():
    points, t_ns, scale = _series()
    points[30, 0] += (0.0, 0.8 * SCALE)
    figures = _figures(points, t_ns, scale, np.array([False, True, True]))
    assert figures["jumps_per_min"] == 0.0
    assert figures["placed"] == 1.0


def test_pooled_figures_are_over_the_pooled_rows():
    a = _tally(*_series())
    points, t_ns, scale = _series()
    points[30, 0] += (0.0, 0.8 * SCALE)
    b = _tally(points, t_ns, scale)
    pooled = diag.Tally()
    pooled.add(a)
    pooled.add(b)
    assert pooled.figures()["jumps_per_min"] == pytest.approx(2 / (120 / FPS / 60))


def test_nothing_to_measure_is_none_not_zero():
    points, t_ns, scale = _series(1)
    figures = _figures(points, t_ns, scale)
    assert figures["placed"] == 1.0
    assert figures["median_move"] is None and figures["jumps_per_min"] is None


def test_agreement_counts_only_the_points_detection_was_sure_of():
    points, t_ns, scale = _series()
    scores = np.ones((61, 3))
    scores[:30] = 0.5  # the first 30 rows sit below each point's median score
    # the nose wanders 0.2 body off detection on 20 rows detection was unsure of, and on
    # 10 it was sure of; the ears stay put
    points[:20, 0, 0] += 0.2 * SCALE
    points[40:50, 0, 0] += 0.2 * SCALE
    figures = _figures(points, t_ns, scale, scores=scores)
    confident = 31 * 3  # rows 30..60, every point
    assert figures["agreement"] == pytest.approx((confident - 10) / confident)


def test_agreement_holds_within_the_radius_and_misses_an_unplaced_point():
    points, t_ns, scale = _series()
    points[:10, 1, 0] += 0.05 * SCALE  # inside AGREE_BODY: still agrees
    points[10:20, 2] = np.nan  # not placed where detection was: a miss
    figures = _figures(points, t_ns, scale)
    assert figures["agreement"] == pytest.approx(1 - 10 / (61 * 3))


def test_agreement_is_none_where_detection_placed_nothing():
    points, t_ns, scale = _series()
    detected = np.full_like(points, np.nan)
    assert _figures(points, t_ns, scale, detected=detected)["agreement"] is None


def test_the_front_is_what_nothing_beats_on_every_figure():
    figures = {
        "steady": {
            "agreement": 0.9,
            "placed": 0.9,
            "jumps_per_min": 1,
            "swaps_per_min": 0,
        },
        "smoothed": {
            "agreement": 0.6,
            "placed": 0.9,
            "jumps_per_min": 0,
            "swaps_per_min": 0,
        },
        "beaten": {
            "agreement": 0.8,
            "placed": 0.9,
            "jumps_per_min": 2,
            "swaps_per_min": 0,
        },
        "tied": {
            "agreement": 0.9,
            "placed": 0.9,
            "jumps_per_min": 1,
            "swaps_per_min": 0,
        },
    }
    # smoothing's fewer jumps keep it on the front against its lower agreement: a trade
    assert diag.pareto_front(figures) == ["steady", "smoothed", "tied"]


def test_the_front_skips_a_figure_one_setting_lacks():
    figures = {
        "a": {
            "agreement": 0.9,
            "placed": 0.9,
            "jumps_per_min": 1,
            "swaps_per_min": None,
        },
        "b": {"agreement": 0.8, "placed": 0.9, "jumps_per_min": 1, "swaps_per_min": 0},
    }
    assert diag.pareto_front(figures) == ["a"]
