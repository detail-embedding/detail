"""The score a transition detector cuts on, exposed for diagnostics.

``TRANSITION_DETECTORS`` returns only cut indices, so a diagnostic could not show *why* a cut
landed where it did. ``TRANSITION_CURVES`` is a parallel, **optional** registry carrying that
score. The contract that matters: peak-picking the curve at its threshold must reproduce the
detector's own cuts exactly — otherwise the plot explains a segmentation that did not happen.
"""

import numpy as np
import pytest

from detail.analysis import segmentation as sg


def _blocky(n_blocks=6, block=20, dim=8, seed=0):
    """A trajectory that sits in one pose direction per block — cuts belong at the seams."""
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            np.eye(dim)[k % 3][None, :] + 0.02 * rng.standard_normal((block, dim))
            for k in range(n_blocks)
        ]
    )


_PARAMS = {"percentile": 90.0, "min_size": 3, "half": 8, "penalty": 2.0}


@pytest.mark.parametrize("detector", ["speed", "novelty"])
def test_curve_peaks_reproduce_the_detector_cuts(detector):
    """The whole point: the curve explains the cuts the fit actually made."""
    from scipy.signal import find_peaks

    X = _blocky()
    xruns = [(0, len(X))]
    cuts = sg.detect_transitions(X, xruns, detector, _PARAMS)
    curve = sg.transition_curve(X, xruns, detector, _PARAMS)
    peaks, _ = find_peaks(
        np.nan_to_num(curve.values),
        height=curve.threshold,
        distance=_PARAMS["min_size"],
    )
    assert [int(p) + curve.offset for p in peaks] == cuts
    assert cuts  # a blocky trajectory must produce cuts, or the test proves nothing


def test_novelty_cuts_land_on_the_block_seams():
    """Guards the refactor's arithmetic, not just its self-consistency."""
    X = _blocky(n_blocks=6, block=20)
    cuts = sg.detect_transitions(X, [(0, len(X))], "novelty", _PARAMS)
    assert cuts == [20, 40, 60, 80, 100]


def test_a_detector_without_a_curve_returns_none():
    """binseg recursively splits rather than scoring every index — it has no curve, and a
    caller must handle that rather than assume every detector explains itself."""
    X = _blocky()
    assert "binseg" in sg.TRANSITION_DETECTORS
    assert "binseg" not in sg.TRANSITION_CURVES
    assert sg.transition_curve(X, [(0, len(X))], "binseg", _PARAMS) is None


def test_speed_offset_is_one_because_it_scores_steps():
    """`speed` scores the gap *between* two features, so its curve is one shorter and each
    peak names the earlier feature — the offset is what keeps it aligned with the cuts.
    """
    X = _blocky()
    curve = sg.transition_curve(X, [(0, len(X))], "speed", _PARAMS)
    assert curve.offset == 1
    assert sg.transition_curve(X, [(0, len(X))], "novelty", _PARAMS).offset == 0


def test_curve_spans_feature_index_space_with_gaps_between_runs():
    """Per-run scoring laid back into one array: a plot must break at a tracking gap, not
    draw a line through it."""
    X = _blocky(n_blocks=6, block=20)
    xruns = [(0, 50), (60, 120)]  # rows 50..59 are a gap
    curve = sg.transition_curve(X, xruns, "novelty", _PARAMS)
    assert len(curve.values) == len(X)
    assert np.isnan(curve.values[50:60]).all()
    assert not np.isnan(curve.values[0:50]).any()


def test_cuts_never_span_a_run_boundary():
    X = _blocky(n_blocks=6, block=20)
    xruns = [(0, 50), (60, 120)]
    cuts = sg.detect_transitions(X, xruns, "novelty", _PARAMS)
    assert all(not (50 <= c < 60) for c in cuts)


@pytest.mark.parametrize("detector", ["speed", "novelty"])
def test_too_short_a_trajectory_yields_an_empty_curve(detector):
    X = np.eye(8)[:2]
    curve = sg.transition_curve(X, [(0, 2)], detector, _PARAMS)
    assert np.isnan(curve.values).all()
    assert sg.detect_transitions(X, [(0, 2)], detector, _PARAMS) == []
