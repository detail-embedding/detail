"""Dynamics metrics — the curves a segmentation decision gets read against.

The properties that matter are not "does it return numbers". They are:

- **Gaps break.** Every metric must be ``NaN`` where a value is undefined — across a tracking
  gap, at a run's first row, where a mask is empty. A curve that interpolates a gap draws
  behavior that never happened, and the video is then lying at exactly the moment a person is
  looking hardest.
- **The pose family cannot drift from the segmenter.** ``novelty`` reuses
  ``segmentation._novelty_curve``, and ``window_features`` must equal
  ``segmentation.build_features(pool="mean")`` — otherwise the panel shows one thing and the
  fit cuts on another.
- **Nothing pools across a run boundary**, matching ``detect_transitions``.
"""

import numpy as np
import pytest

from detail.analysis import dynamics as dyn
from detail.analysis import segmentation


def _inputs(n=40, dim=6, *, gap_at=None, stride=1, seed=0):
    """N rows of a smooth trajectory, optionally with a tracking gap before ``gap_at``."""
    rng = np.random.default_rng(seed)
    emb = np.cumsum(rng.normal(size=(n, dim)), axis=0) + 10.0
    frame_idx = np.arange(n, dtype=np.int64) * stride
    if gap_at is not None:
        frame_idx[gap_at:] += (
            50 * stride
        )  # a jump: rows gap_at-1 and gap_at aren't adjacent
    centroid = np.cumsum(rng.normal(size=(n, 2)), axis=0) + 100.0
    area = np.abs(rng.normal(size=n)) * 100 + 500.0
    bbox = np.stack(
        [
            centroid[:, 0] - 20,
            centroid[:, 1] - 10,
            centroid[:, 0] + 20,
            centroid[:, 1] + 30,
        ],
        axis=1,
    )
    return dyn.DynamicsInputs(
        embeddings=emb,
        frame_idx=frame_idx,
        stride=stride,
        centroid=centroid,
        area=area,
        bbox=bbox,
    )


# --- the registries --------------------------------------------------------------------- #
def test_every_metric_returns_one_value_per_row():
    d = _inputs()
    for name in dyn.METRICS:
        curve = dyn.METRICS[name].fn(d)
        assert curve.shape == (len(d.embeddings),), name
        assert curve.dtype.kind == "f", name  # float, so NaN is representable


def test_compute_metrics_preserves_order_and_fails_loud_on_a_typo():
    d = _inputs()
    out = dyn.compute_metrics(d, ["mask_area", "novelty"], {})
    assert [label for label, _ in out] == [
        dyn.METRICS["mask_area"].label,
        dyn.METRICS["novelty"].label,
    ]
    # Fail before rendering, which is the expensive part.
    with pytest.raises(ValueError, match="unknown metric"):
        dyn.compute_metrics(d, ["mask_area", "nope"], {})
    with pytest.raises(ValueError, match="no metrics requested"):
        dyn.compute_metrics(d, [], {})


# --- gaps ------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name",
    ["pose_step", "window_step", "mask_speed", "bbox_speed", "mask_area_step"],
)
def test_a_step_metric_is_nan_across_a_tracking_gap(name):
    """A step between non-adjacent frames is meaningless — it must be absent, not large."""
    gap = 20
    d = _inputs(gap_at=gap)
    curve = dyn.METRICS[name].fn(d)
    assert np.isnan(curve[gap]), f"{name} drew a step across the gap"
    assert np.isnan(curve[0]), f"{name} invented a step at row 0"


@pytest.mark.parametrize("name", ["novelty", "texttile_depth", "pose_curvature"])
def test_a_pooling_metric_is_nan_at_each_run_start(name):
    """These pool a window either side, so a run's first row has no left window at all."""
    gap = 20
    d = _inputs(gap_at=gap)
    curve = dyn.METRICS[name].fn(d)
    assert np.isnan(curve[0])
    assert np.isnan(curve[gap])


def test_novelty_does_not_pool_poses_from_across_a_gap():
    """The segmenter runs its detectors per contiguous run. If the metric pooled across the
    gap it would plot a peak the segmenter never sees — so the two halves must be identical
    to running each alone."""
    d = _inputs(n=40, gap_at=20)
    joined = dyn.METRICS["novelty"].fn(d)
    first = dyn.METRICS["novelty"].fn(_inputs(n=20, seed=0))
    # Same underlying rows (same seed, same slice) → the first run's values must match.
    np.testing.assert_allclose(joined[1:20], first[1:20], atol=1e-12)


def test_run_edges_splits_exactly_at_the_gap():
    d = _inputs(n=40, gap_at=20)
    assert dyn.run_edges(d.frame_idx, d.stride) == [(0, 20), (20, 40)]


def test_a_stride_greater_than_one_still_reads_as_adjacent():
    """An embed run at every Nth frame has frame_idx steps of N — that is not a gap."""
    d = _inputs(n=30, stride=4)
    assert dyn.run_edges(d.frame_idx, d.stride) == [(0, 30)]
    assert not np.isnan(dyn.METRICS["pose_step"].fn(d)[1:]).any()


# --- the tie to the segmenter ----------------------------------------------------------- #
def test_window_features_equals_the_segmenters_own_pooling():
    """The panel must show what the fit consumes. build_features stacks (N, window, D); this
    is a cumsum — they must agree or the video is plotting a different signal."""
    d = _inputs(n=40)
    window = 8
    mine = dyn.window_features(d.embeddings, d.frame_idx, d.stride, window)
    theirs, row_of_x, _xruns = segmentation.build_features(
        d.embeddings, [(0, len(d.embeddings))], window=window, pool="mean"
    )
    placed = mine[row_of_x]
    assert np.isfinite(placed).all()
    np.testing.assert_allclose(placed, theirs, atol=1e-12)


def test_window_features_never_straddles_a_gap():
    d = _inputs(n=40, gap_at=20)
    out = dyn.window_features(d.embeddings, d.frame_idx, d.stride, 8)
    # Run 1 is [0, 20): the last window that fits is [12, 20), placed at 12 + 8//2 = 16. So
    # row 16 is legitimately defined and 17-19 are not — a window reaching them would have to
    # pool poses from the far side of the gap.
    assert np.isfinite(out[16]).all()
    assert np.isnan(out[17:20]).all()


def test_window_features_rejects_a_zero_window():
    d = _inputs()
    with pytest.raises(ValueError, match="window must be >= 1"):
        dyn.window_features(d.embeddings, d.frame_idx, d.stride, 0)


# --- individual metrics ----------------------------------------------------------------- #
def test_mask_centroid_is_the_mass_center_and_undefined_when_empty():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 4:8] = True
    x, y = dyn.mask_centroid(mask)
    assert (x, y) == (5.5, 3.5)
    assert all(np.isnan(v) for v in dyn.mask_centroid(np.zeros((10, 10), bool)))


def test_pose_curvature_reads_minus_one_on_a_reversal():
    """The apex of an out-and-back gesture — where speed is high and says nothing.

    The path has to turn *on the unit sphere*: rows are L2-normalized first, so a trajectory
    along a ray through the origin collapses to a single point and has no direction at all.
    """
    t = np.linspace(0, 1, 10)[:, None]
    base = np.array([[1.0, 0.0, 0.0, 0.0]])
    step = np.array([[0.0, 1.0, 0.0, 0.0]])
    out = base + t * step  # an arc: direction genuinely changes
    emb = np.vstack([out, out[::-1][1:]])
    d = dyn.DynamicsInputs(
        embeddings=emb,
        frame_idx=np.arange(len(emb), dtype=np.int64),
        stride=1,
        centroid=np.zeros((len(emb), 2)),
        area=np.ones(len(emb)),
        bbox=np.tile([0.0, 0.0, 1.0, 1.0], (len(emb), 1)),
    )
    curve = dyn.METRICS["pose_curvature"].fn(d)
    assert curve[10] == pytest.approx(-1.0, abs=1e-6)  # the turn-around row
    # Mid-run reads ~0.996, not exactly 1: this path is straight in *raw* space, and
    # normalizing makes it an arc on the sphere. The docstring's "~0.999, not 1".
    assert 0.99 < curve[5] < 1.0


def test_pose_curvature_rejects_a_zero_lag():
    with pytest.raises(ValueError, match="lag must be >= 1"):
        dyn.METRICS["pose_curvature"].fn(_inputs(), lag=0)


def test_bbox_aspect_is_height_over_width():
    d = _inputs(n=5)
    curve = dyn.METRICS["bbox_aspect"].fn(d)
    np.testing.assert_allclose(curve, 40.0 / 40.0)  # the fixture's box is 40x40


def test_smoothing_zero_passes_recovers_the_raw_curve():
    """The tie-back that proves the smoother is not silently always-on."""
    y = np.array([0.0, 1.0, 0.2, 0.9, 0.1])
    np.testing.assert_array_equal(dyn._smooth(y, 0, 3), y)
    np.testing.assert_array_equal(dyn._smooth(y, 2, 0), y)


def test_depth_score_rewards_prominence_not_height():
    """The one thing texttile adds over novelty: a high point on a rising baseline is not a
    boundary, an isolated spike is."""
    ramp = np.linspace(0.0, 1.0, 9)  # climbs to 1.0
    spike = np.array([0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.1])  # only reaches 0.6
    assert spike.max() < ramp.max()  # the shorter curve...
    assert (
        dyn._depth_score(spike).max() > dyn._depth_score(ramp).max()
    )  # ...scores higher
    assert dyn._depth_score(np.zeros(9)).max() == pytest.approx(0.0, abs=1e-12)


# --- curve detectors -------------------------------------------------------------------- #
def test_every_curve_detector_returns_in_range_indices():
    d = _inputs(n=120, seed=3)
    curve = dyn.METRICS["novelty"].fn(d)
    for name in dyn.CURVE_DETECTORS:
        cuts = dyn.detect_curve_transitions(curve, name, {})
        assert all(0 <= c < len(curve) for c in cuts), name
        assert cuts == sorted(set(cuts)), name


def test_no_cut_lands_inside_a_gap():
    d = _inputs(n=80, gap_at=40, seed=5)
    curve = dyn.METRICS["mask_speed"].fn(d)
    nan_rows = set(np.flatnonzero(~np.isfinite(curve)).tolist())
    for name in dyn.CURVE_DETECTORS:
        cuts = dyn.detect_curve_transitions(curve, name, {})
        assert not (set(cuts) & nan_rows), name


def test_texttile_reads_the_curves_native_scale():
    """It must NOT standardize — a depth score's magnitude is the signal, and z-scoring it
    throws away exactly what distinguishes a prominent peak from a mediocre one."""
    y = np.array([0.0, 0.5, 0.0, 0.05, 0.0, 0.5, 0.0, 0.05, 0.0] * 3)
    strict = dyn._curve_texttile(y, threshold_mode="frac_max", frac=0.6, min_size=2)
    loose = dyn._curve_texttile(y, threshold_mode="frac_max", frac=0.01, min_size=2)
    assert len(strict) < len(loose)  # the cutoff bites, so it is reading real values


def test_texttile_rejects_an_unknown_threshold_mode():
    # Needs a curve with peaks: a monotone one returns early, before the mode is read.
    peaky = np.array([0.0, 0.5, 0.0, 0.4, 0.0] * 4)
    with pytest.raises(ValueError, match="threshold_mode must be"):
        dyn._curve_texttile(peaky, threshold_mode="nope", min_size=2)


def test_an_unknown_curve_detector_fails_loud():
    with pytest.raises(ValueError, match="unknown curve detector"):
        dyn.detect_curve_transitions(np.zeros(20), "nope", {})


def test_an_all_nan_curve_yields_no_cuts():
    assert dyn.detect_curve_transitions(np.full(20, np.nan), "novelty", {}) == []


# --- abd: the metric panel and the scalar detector ------------------------------------- #
def _drifting_steps(seed=0, drift=2.0):
    """A step signal on a rising baseline: where a global percentile cuts by baseline height."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.full(40, v) for v in (0.0, 1.0, 0.0, 2.0)])
    return y + rng.normal(0, 0.08, 160) + np.linspace(0, drift, 160), [40, 80, 120]


def test_abd_metric_is_the_segmenters_own_curve():
    """The panel must plot what the detector cuts on, not a second implementation of it."""
    from detail.analysis.segmentation import _abd_curve

    d = _inputs(n=60)
    panel = dyn.METRICS["abd"].fn(d)
    expected = _abd_curve(d.embeddings, smooth_half=2).values
    assert np.isnan(panel[0]), "the first row scores no step and must break the line"
    np.testing.assert_allclose(panel[1:], expected, rtol=1e-9)


def test_abd_metric_breaks_at_a_tracking_gap():
    d = _inputs(n=60, gap_at=30)
    panel = dyn.METRICS["abd"].fn(d)
    assert np.isnan(
        panel[30]
    ), "the second run's first row must not score across the gap"


def test_abd_curve_detector_finds_the_boundaries_a_percentile_misses():
    """The reason for the rule: on a drifting baseline a global cutoff fires where the baseline
    is high, while a locally prominent peak still lands on the real changes."""
    y, truth = _drifting_steps()
    cuts = dyn.detect_curve_transitions(y, "abd", {})
    for boundary in truth:
        assert any(abs(c - boundary) <= 3 for c in cuts), (boundary, cuts)


def test_abd_curve_detector_smoothing_has_the_same_cliff_as_the_multivariate_one():
    """Pins why the default is small: widening it loses true boundaries."""
    found = {}
    for smooth_half in (2, 6):
        hits = 0
        for seed in range(6):
            y, truth = _drifting_steps(seed=seed)
            cuts = dyn.detect_curve_transitions(y, "abd", {"smooth_half": smooth_half})
            hits += sum(any(abs(c - b) <= 3 for c in cuts) for b in truth)
        found[smooth_half] = hits
    assert found[2] > found[6], found


def test_abd_curve_detector_respects_gaps():
    y, _truth = _drifting_steps()
    y[70:80] = np.nan
    cuts = dyn.detect_curve_transitions(y, "abd", {})
    assert all(not (70 <= c < 80) for c in cuts), cuts
