"""Per-row dynamics metrics — the signals a segmentation decision can be read against.

Clustering cuts the pose trajectory at transitions and groups the bouts, but judging *why* a
cut landed where it did needs the underlying signals plotted **against the video**. These are
those signals: one ``(N,)`` curve per metric over a recording's embedded rows, consumed by
``scripts/research_dynamics_video.py``.

Ported from an earlier prototype's dynamics module, where the equivalent video was
what actually drove segmentation decisions. Two families, deliberately mixed:

- **pose** (``pose_step``, ``window_step``, ``novelty``, ``pose_curvature``,
  ``texttile_depth``) — what the segmenter sees. The novelty curve is imported from
  ``segmentation``, never reimplemented, so the video cannot drift from what the detector
  actually cuts on.
- **kinematic** (``mask_speed``, ``bbox_speed``, ``mask_area``, and the aspect/divergence
  extras) — from the tracker's masks and boxes. This family is not redundant: the
  ``*_centered`` constructors remove position *as a coordinate*, so locomotion is not legible
  in the embedding at all and these are the only place it survives.

``METRICS`` is a registry mirroring ``segmentation.TRANSITION_DETECTORS``: every metric is
``fn(d: DynamicsInputs, **params) -> (N,)``, so a panel list is a config change.

``CURVE_DETECTORS`` is the scalar counterpart — ``speed`` / ``novelty`` / ``binseg`` asking
"where does *this* curve change?", plus ``texttile`` (TextTiling's prominent-peak rule) — so a
metric can be marked with its own transitions. They are separate functions rather than a
reshape of the multivariate ones because those normalize each row to unit length first:
meaningful for poses (compare directions), fatal for a scalar, where it collapses the whole
curve to ``sign(x)``. The first three z-score the curve so one config spans panels of wildly
different scale; ``texttile`` deliberately does not (its cutoff is native to the curve).

Curves carry ``NaN`` where a value is undefined — across a tracking gap (consecutive rows are
not adjacent in time, so a step between them is meaningless) or where a mask has no pixels.
**Callers must break the line, not fill it:** a gap is missing data, and drawing through it
invents behavior.

Pure over arrays: numpy only, no zarr, no paths, no Qt. Nothing in the pipeline imports this
module — it is a research surface, and keeping it that way is what makes it safe to change.
"""

from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# The shape of a "step" — what counts as adjacent, and the change across a pair — lives in
# `descriptors`, which the pipeline may import and this module may not be imported *by* (see the
# note at the end of this docstring). Aliased rather than reimplemented: the research video and
# the interpretation panel must not be able to disagree about what mask speed means, and two
# copies of a formula eventually do.
from detail.analysis.descriptors import bbox_center as _bbox_center
from detail.analysis.descriptors import adjacent as _adjacent
from detail.analysis.descriptors import cosine_step as _cosine_step
from detail.analysis.descriptors import step_curve as _step_curve

# The one novelty implementation (a private name, deliberately: importing the *same* function
# the segmenter cuts on is the point — a public re-export would invite a second copy).
# Re-exported for this module's existing callers. The definition moved to `analysis/geometry.py`
# when the embedding pass needed it: importing *this* module from the pipeline would break the
# invariant the docstring above states, and the whole point of that invariant is that a research
# surface stays free to change.
from detail.analysis.geometry import mask_centroid  # noqa: F401
from detail.analysis.segmentation import _abd_curve, _novelty_curve

# fn: (DynamicsInputs, **params) -> (N,) curve; label: the video panel's caption.
Metric = namedtuple("Metric", ["fn", "label"])

METRICS: dict[str, Metric] = {}

# fn: ((n,) finite curve, **params) -> list[int] cut indices within that curve.
CURVE_DETECTORS: dict[str, Callable] = {}


def register_metric(name: str, label: str) -> Callable[[Callable], Callable]:
    """Register a metric ``fn(d, **params) -> (N,) curve`` under ``name``."""

    def deco(fn: Callable) -> Callable:
        METRICS[name] = Metric(fn, label)
        return fn

    return deco


def register_curve_detector(name: str) -> Callable[[Callable], Callable]:
    """Register a 1-D detector ``fn(y, **params) -> list[int]`` (cut indices in ``y``)."""

    def deco(fn: Callable) -> Callable:
        CURVE_DETECTORS[name] = fn
        return fn

    return deco


@dataclass(frozen=True)
class DynamicsInputs:
    """One recording window's per-row inputs, all aligned to the embedded rows.

    Index ``i`` is the same embedded frame across every field: the caller resolves each row's
    source frame to the subject's detection rows, so the kinematics describe the same
    silhouette the embedding was taken from.

    Pixel units throughout. The detection store's ``bbox`` is normalized xyxy; a speed in
    normalized units would be anisotropic (x and y scale differently) and incomparable to a
    mask-pixel speed, so the caller rescales by ``(width, height)`` before constructing this.
    """

    embeddings: np.ndarray  # (N, D) pose embeddings, one row per embedded frame
    frame_idx: np.ndarray  # (N,) source-frame index per row
    stride: int  # rows are adjacent iff their frame_idx differs by this
    centroid: np.ndarray  # (N, 2) mask centroid (x, y) px; NaN where the mask is empty
    area: np.ndarray  # (N,) mask pixel count
    bbox: np.ndarray  # (N, 4) bbox, xyxy px


def run_edges(frame_idx: np.ndarray, stride: int) -> list[tuple[int, int]]:
    """Contiguous row runs ``[start, end)``, as ``segmentation`` computes them.

    Anything that pools across rows must stay inside one run: the segmenter applies its
    detectors per run (``detect_transitions``), so pooling across a gap would mix poses from
    different times and show a signal the segmenter never sees.
    """
    n = len(frame_idx)
    breaks = np.nonzero(~_adjacent(frame_idx, stride))[0] + 1
    edges = np.concatenate([[0], breaks, [n]])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


def window_features(
    embeddings: np.ndarray, frame_idx: np.ndarray, stride: int, window: int
) -> np.ndarray:
    """(N, D) mean-pooled sliding-window features, one per row; NaN rows where undefined.

    Matches ``segmentation.build_features(pool="mean")``: rows are L2-normalized *first*, the
    window ``[s, s+window)`` is averaged (and **not** renormalized), and the feature is placed
    at row ``s + window // 2`` — so this plots what the fit's window pooling actually consumes.
    Windows never straddle a tracking gap.

    Built by cumsum rather than an ``(N, window, D)`` stack of windows, which is ``window`` times
    the trajectory (``build_features`` stacks in bounded chunks for the same reason, #581);
    ``tests/analysis/test_dynamics.py`` asserts the two agree.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    n, dim = x.shape
    out = np.full((n, dim), np.nan)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    csum = np.concatenate([np.zeros((1, dim)), np.cumsum(xn, axis=0)])
    for a, b in run_edges(frame_idx, stride):
        starts = np.arange(a, b - window + 1)
        if len(starts) == 0:
            continue
        out[starts + window // 2] = (csum[starts + window] - csum[starts]) / window
    return out


@register_metric("pose_step", "pose_step  1-cos(x_t-1, x_t)")
def _pose_step(d: DynamicsInputs, **_) -> np.ndarray:
    """How fast the per-frame pose is changing — the ``speed`` detector's signal."""
    return _cosine_step(d.embeddings, d.frame_idx, d.stride)


@register_metric("window_step", "window_step  1-cos(w_t-1, w_t)")
def _window_step(d: DynamicsInputs, *, window: int = 8, **_) -> np.ndarray:
    """Pose-change rate after window pooling — what smoothing leaves for the detector."""
    feats = window_features(d.embeddings, d.frame_idx, d.stride, window)
    return _cosine_step(feats, d.frame_idx, d.stride)


@register_metric("pose_curvature", "pose_curvature  cos(v_t-lag, v_t)")
def _pose_curvature(d: DynamicsInputs, *, lag: int = 1, **_) -> np.ndarray:
    """Is the pose trajectory running straight, turning, or doubling back?

    With ``v_t = x_t - x_{t-lag}`` (the step the pose took), this is ``cos(v_{t-lag}, v_t)``:
    ``+1`` continues in the same direction (the animal keeps doing the same thing), ``0`` is a
    right-angle turn (the motion changes character), ``-1`` is a reversal (the pose retraces
    its path — the apex of an out-and-back gesture).

    **Orthogonal to** ``pose_step``, which is a *speed*: a rear moves fast on the way up and
    fast on the way down, so speed is high throughout and flat at the apex, while curvature
    dips to -1 exactly there. Speed says how much, curvature says which way.

    **The null is -0.5, not 0.** Read the panel against that. For a pose held still with iid
    jitter, ``x_t = mu + eps_t``, so ``v_t = eps_t - eps_{t-1}`` and
    ``E[v_t . v_{t+1}] = -sigma^2`` while ``E[||v||^2] = 2 sigma^2`` — hence ``E[cos] = -0.5``
    exactly. A bounded random walk *must* double back, so **pure noise always looks like
    reversal**. A real recording sits somewhere above that floor and far below a
    straight run. To measure the *path's* shape rather
    than the jitter's, difference non-overlapping window means instead of raw frames.

    It is the **cosine of the turning angle**, not differential-geometric curvature
    (``k = dtheta/ds``), which divides by step length and explodes when the animal is still
    and the direction is pure noise; the turning cosine degrades gracefully instead.

    ``lag`` is the velocity's own step. Raising it lengthens the baseline but does **not**
    average noise (both endpoints stay single noisy rows), so it helps only if the pose is
    actually travelling. Computed per run; each run's first ``2*lag`` rows are NaN.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    x = np.asarray(d.embeddings, dtype=np.float64)
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    out = np.full(len(xn), np.nan)
    for a, b in run_edges(d.frame_idx, d.stride):
        if b - a < 2 * lag + 1:
            continue
        block = xn[a:b]
        v = block[lag:] - block[:-lag]  # (m - lag, D) velocity vectors
        vn = np.linalg.norm(v, axis=1, keepdims=True)
        # A pose held exactly still has no direction to turn — leave it undefined.
        unit = v / np.where(vn > 1e-12, vn, np.nan)
        cos = np.sum(unit[lag:] * unit[:-lag], axis=1)
        out[a + 2 * lag : b] = np.clip(cos, -1.0, 1.0)
    return out


@register_metric("novelty", "novelty  left-vs-right window")
def _novelty_metric(d: DynamicsInputs, *, half: int = 8, **_) -> np.ndarray:
    """The novelty detector's own cut signal (``segmentation._novelty_curve``).

    Computed **per run**, as ``segmentation.detect_transitions`` applies its detectors — the
    curve pools ``half`` rows either side, so running it across a gap would average poses from
    different times and plot a peak the segmenter never sees. Each run's first row has no left
    window (the curve forces it to 0); it is NaN'd here so the plot breaks at gaps rather than
    dropping to a misleading zero.

    This is novelty over the *per-frame* poses. A fit that pools into windows first sees a
    different curve — pair this panel with ``window_step`` to compare.
    """
    out = np.full(len(d.embeddings), np.nan)
    for a, b in run_edges(d.frame_idx, d.stride):
        if b - a < 4:
            continue
        run = _novelty_curve(d.embeddings[a:b], half=half).values
        out[a + 1 : b] = run[1:]  # row a has no left window
    return out


@register_metric("abd", "abd  smoothed adjacent step")
def _abd_metric(d: DynamicsInputs, *, smooth_half: int = 2, **_) -> np.ndarray:
    """The abd detector's own cut signal (``segmentation._abd_curve``).

    Imported rather than reimplemented, for the same reason ``novelty`` is: the panel has to be
    the curve the segmenter actually cuts on, and a second copy would be free to drift from it.

    The curve scores the *step between* rows, so value ``i`` is placed at row ``i + 1`` — the row
    the cut would land on, matching the curve's own ``offset``. Computed per run, since the
    smoothing pools rows either side and would otherwise average across a tracking gap.

    This is abd over the *per-frame* poses; a fit that pools into windows first sees a different
    curve, so pair it with ``window_step`` when comparing.
    """
    out = np.full(len(d.embeddings), np.nan)
    for a, b in run_edges(d.frame_idx, d.stride):
        if b - a < 3:
            continue
        out[a + 1 : b] = _abd_curve(d.embeddings[a:b], smooth_half=smooth_half).values
    return out


def _smooth(y: np.ndarray, passes: int, window: int) -> np.ndarray:
    """``passes`` sweeps of a symmetric +/-``window`` moving average (clipped at the ends).

    TextTiling smooths its coherence curve before scoring so one noisy row cannot look like a
    boundary. ``passes < 1`` or ``window < 1`` is a no-op returning a copy — which is how
    ``smoothing_passes=0`` recovers the raw novelty curve (a tested tie-back).
    """
    y = np.asarray(y, dtype=np.float64)
    if passes < 1 or window < 1:
        return y.copy()
    n = len(y)
    t = np.arange(n)
    lo = np.maximum(0, t - window)
    hi = np.minimum(n, t + window + 1)
    for _ in range(passes):
        c = np.concatenate([[0.0], np.cumsum(y)])
        y = (c[hi] - c[lo]) / (hi - lo)
    return y


def _depth_score(y: np.ndarray) -> np.ndarray:
    """Hearst's TextTiling depth score, mirrored onto a peaks-are-boundaries curve.

    For each interior row descend to the nearest local minimum on each side and sum the two
    drops: ``(y[i] - left_min) + (y[i] - right_min)``. A prominent peak scores high
    *regardless of the curve's baseline*, which is the point of the transform: a lone spike
    reaching only 0.6 scores 1.00, while a ramp climbing all the way to 1.0 scores 0.75.
    Prominence, not height. Endpoints are 0 (no two sides to compare).

    A long monotone slope is **not** scored ~0, contrary to what one might assume: with no
    interior local minimum to stop at, the walk descends to the run's edge, so the left term
    is the entire rise. That is canonical Hearst on an unbounded slope — the smoothing pass
    and the finite render window are what bound it in practice.

    Hearst and the reference implementation compute this on the coherence curve, where
    boundaries are valleys and the walk climbs to maxima; the novelty curve is the complement
    (boundaries are peaks), so the walk descends to minima instead. Same magnitude, opposite
    convention.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    out = np.zeros(n)
    for i in range(1, n - 1):
        left, right = i - 1, i + 1
        while left > 0 and y[left - 1] < y[left]:
            left -= 1
        while right < n - 1 and y[right + 1] < y[right]:
            right += 1
        out[i] = (y[i] - y[left]) + (y[i] - y[right])
    return out


@register_metric("texttile_depth", "texttile_depth  Hearst valley prominence")
def _texttile_depth(
    d: DynamicsInputs,
    *,
    tt_window: int = 5,
    smoothing_passes: int = 2,
    smoothing_window: int = 1,
    **_,
) -> np.ndarray:
    """TextTiling's boundary-strength signal over the pose trajectory (Solbiati et al. 2021).

    The block-comparison coherence is our own novelty curve — mean-pooled left/right window
    cosine distance, so boundaries are peaks; this smooths it and applies Hearst's depth score,
    so a peak counts only when it is *locally* prominent, not merely high. That prominence
    transform is the one thing this adds over the ``novelty`` metric: robustness to a drifting
    baseline rather than to an absolute level. Pair it with the ``texttile`` curve detector to
    reproduce TextTiling end to end.

    **Unfinished, deliberately.** An earlier note found the depth peaks line up
    with visible behavior changes, but the default ``hearst`` cut over-segments (~60 cuts per
    1000 frames against novelty's ~14). The *curve* is the promising part; the boundary
    threshold still needs a pass, which is what the video is for.

    Computed **per run** (like the novelty metric): the window pools ``tt_window`` rows either
    side, so it must not straddle a tracking gap. Each run's first row (no left window) is NaN;
    the depth curve's own endpoints are 0 (no prominence measurable at the very edge).
    """
    out = np.full(len(d.embeddings), np.nan)
    for a, b in run_edges(d.frame_idx, d.stride):
        if b - a < 4:
            continue
        # Drop row a: no left window, so its novelty is a forced 0, not a measurement.
        nov = _novelty_curve(d.embeddings[a:b], half=tt_window).values[1:]
        out[a + 1 : b] = _depth_score(_smooth(nov, smoothing_passes, smoothing_window))
    return out


@register_metric("mask_speed", "mask_speed  px/frame")
def _mask_speed(d: DynamicsInputs, **_) -> np.ndarray:
    """How fast the silhouette's center of mass is moving."""
    return _step_curve(d.centroid, d.frame_idx, d.stride)


@register_metric("bbox_speed", "bbox_speed  px/frame")
def _bbox_speed(d: DynamicsInputs, **_) -> np.ndarray:
    """How fast the detection box's center is moving."""
    return _step_curve(_bbox_center(d.bbox), d.frame_idx, d.stride)


@register_metric("mask_area", "mask_area  px")
def _mask_area(d: DynamicsInputs, **_) -> np.ndarray:
    """Silhouette footprint — shrinks as the animal rears up out of the floor plane."""
    return np.asarray(d.area, dtype=np.float64)


@register_metric("mask_area_step", "mask_area_step  px/frame")
def _mask_area_step(d: DynamicsInputs, **_) -> np.ndarray:
    """Rate of footprint change."""
    return _step_curve(np.asarray(d.area, dtype=np.float64), d.frame_idx, d.stride)


@register_metric("bbox_aspect", "bbox_aspect  h/w")
def _bbox_aspect(d: DynamicsInputs, **_) -> np.ndarray:
    """Box elongation — the posture axis centroid speed cannot see."""
    bbox = np.asarray(d.bbox, dtype=np.float64)
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    return np.where(w > 0, h / np.where(w > 0, w, 1.0), np.nan)


@register_metric("centroid_divergence", "centroid_divergence  px")
def _centroid_divergence(d: DynamicsInputs, **_) -> np.ndarray:
    """Distance between the mask's mass center and the box's geometric center — a posture
    signal that needs no motion (mass shifts while the box does not)."""
    return np.linalg.norm(
        np.asarray(d.centroid, dtype=np.float64) - _bbox_center(d.bbox), axis=1
    )


def compute_metrics(
    d: DynamicsInputs, names: list[str], params: dict
) -> list[tuple[str, np.ndarray]]:
    """Evaluate ``names`` against ``d``, returning ``[(label, curve), ...]`` in order.

    Raises:
        ValueError: An unknown metric name, or an empty request — fail loud before any
            rendering, which is the expensive part.
    """
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; have {sorted(METRICS)}")
    if not names:
        raise ValueError("no metrics requested")
    return [(METRICS[n].label, METRICS[n].fn(d, **params)) for n in names]


# --- 1-D transition detection on a metric curve ---------------------------------------- #


def _standardize(y: np.ndarray) -> np.ndarray:
    """Z-score a finite curve; all-zeros if it is constant.

    Metric curves differ by orders of magnitude (``pose_step`` ~0.1, ``mask_area`` ~1e4), so an
    absolute knob like binseg's ``penalty`` is meaningless until they share a scale. Percentile
    knobs are already scale-free, so this leaves ``speed``/``novelty`` unchanged — it exists so
    one config works across every panel.
    """
    y = np.asarray(y, dtype=np.float64)
    sd = float(y.std())
    if sd <= 0.0:
        return np.zeros_like(y)
    return (y - float(y.mean())) / sd


@register_curve_detector("speed")
def _curve_speed(
    y: np.ndarray, *, percentile: float = 96.0, min_size: int = 6, **_
) -> list[int]:
    """Cut where the curve jumps — peaks of ``|dy|``. The scalar analog of the ``speed``
    detector's per-step cosine change."""
    from scipy.signal import find_peaks

    z = _standardize(y)
    if len(z) < 3:
        return []
    step = np.abs(np.diff(z))
    if not step.any():
        return []
    peaks, _ = find_peaks(
        step, height=np.percentile(step, percentile), distance=min_size
    )
    return [int(p) + 1 for p in peaks]


@register_curve_detector("novelty")
def _curve_novelty(
    y: np.ndarray, *, half: int = 8, percentile: float = 96.0, min_size: int = 6, **_
) -> list[int]:
    """Cut where the curve's mean over ``[t-half, t)`` differs most from ``[t, t+half)``.

    The scalar analog of the ``novelty`` detector: same windows, same asymmetry, but the
    discrepancy is ``|mean_right - mean_left|`` rather than a cosine — a scalar has no
    direction to compare.
    """
    from scipy.signal import find_peaks

    z = _standardize(y)
    n = len(z)
    if n < 4:
        return []
    c = np.concatenate([[0.0], np.cumsum(z)])
    t = np.arange(n)
    l0 = np.maximum(0, t - half)
    r1 = np.minimum(n, t + half)
    ml = (c[t] - c[l0]) / np.maximum(1, t - l0)
    mr = (c[r1] - c[t]) / np.maximum(1, r1 - t)
    nov = np.abs(mr - ml)
    nov[t - l0 < 1] = 0.0  # row 0 has no left window
    pos = nov[nov > 0]
    height = np.percentile(pos, percentile) if len(pos) else 0.0
    peaks, _ = find_peaks(nov, height=height, distance=min_size)
    return [int(p) for p in peaks]


@register_curve_detector("binseg")
def _curve_binseg(
    y: np.ndarray,
    *,
    penalty: float | None = 10.0,
    n_bkps: int | None = None,
    min_size: int = 6,
    **_,
) -> list[int]:
    """Binary segmentation with a mean-shift cost — cuts where the curve's *level* shifts.

    The scalar analog of the ``binseg`` detector: the same greedy split on within-segment
    scatter, here ``sum(y^2) - sum(y)^2/len`` over the standardized curve. ``penalty`` is in
    standardized units (hence ``_standardize``); lower cuts more.
    """
    import heapq

    z = _standardize(y)
    n = len(z)
    if n < 2 * min_size:
        return []
    if penalty is None and n_bkps is None:
        raise ValueError("binseg needs penalty or n_bkps")
    c1 = np.concatenate([[0.0], np.cumsum(z)])
    c2 = np.concatenate([[0.0], np.cumsum(z * z)])

    def cost(a: int, b: int) -> float:
        s = c1[b] - c1[a]
        return float((c2[b] - c2[a]) - s * s / (b - a))

    def best(a: int, b: int) -> tuple[float, int]:
        ks = np.arange(a + min_size, b - min_size + 1)
        if len(ks) == 0:
            return (-np.inf, -1)
        sl, sr = c1[ks] - c1[a], c1[b] - c1[ks]
        ql, qr = c2[ks] - c2[a], c2[b] - c2[ks]
        nl, nr = ks - a, b - ks
        gain = cost(a, b) - ((ql - sl * sl / nl) + (qr - sr * sr / nr))
        i = int(np.argmax(gain))
        return (float(gain[i]), int(ks[i]))

    cuts: list[int] = []
    g, k = best(0, n)
    heap = [(-g, 0, n, k)] if k >= 0 else []
    while heap:
        neg_g, a, b, k = heapq.heappop(heap)
        if k < 0:
            continue
        if penalty is not None and -neg_g < penalty:
            break
        if n_bkps is not None and len(cuts) >= n_bkps:
            break
        cuts.append(k)
        for a2, b2 in ((a, k), (k, b)):
            g2, k2 = best(a2, b2)
            if k2 >= 0:
                heapq.heappush(heap, (-g2, a2, b2, k2))
    return sorted(cuts)


@register_curve_detector("texttile")
def _curve_texttile(
    y: np.ndarray,
    *,
    threshold_mode: str = "hearst",
    hearst_c: float = 0.5,
    frac: float = 0.6,
    min_size: int = 6,
    **_,
) -> list[int]:
    """Mark the prominent peaks of a depth-like curve — TextTiling's boundary rule.

    Unlike ``speed`` / ``novelty`` / ``binseg`` this does **not** ``_standardize``: both
    cutoffs read the curve's native scale, which is the whole point of a depth score.
    ``"hearst"`` keeps peaks above ``mean - hearst_c*std`` over the *peak* values (canonical
    TextTiling / nltk "HC", ``c=0.5``) — so a peak must stand out among the others, and a lone
    or all-equal set yields nothing; ``"frac_max"`` keeps peaks above ``frac*max`` (the
    reference implementation's absolute rule). Intended for ``texttile_depth`` but works on any
    curve.

    Raises:
        ValueError: An unknown ``threshold_mode``.
    """
    from scipy.signal import find_peaks

    y = np.asarray(y, dtype=np.float64)
    if len(y) < 3:
        return []
    peaks, _ = find_peaks(y, distance=min_size)
    if len(peaks) == 0:
        return []
    vals = y[peaks]
    if threshold_mode == "hearst":
        cutoff = float(vals.mean()) - hearst_c * float(vals.std())
    elif threshold_mode == "frac_max":
        cutoff = frac * float(np.max(y))
    else:
        raise ValueError(
            f"threshold_mode must be 'hearst' or 'frac_max', got {threshold_mode!r}"
        )
    return [int(p) for p in peaks[vals > cutoff]]


@register_curve_detector("abd")
def _curve_abd(
    y: np.ndarray,
    *,
    smooth_half: int = 2,
    nms_window: int = 5,
    prominence_k: float = 0.75,
    **_,
) -> list[int]:
    """Cut at locally prominent jumps: the scalar analog of the ``abd`` detector.

    Differs from ``speed`` in the rule, not the signal: ``speed`` keeps steps above a global
    percentile, so on a drifting baseline it cuts wherever the baseline is high, while this keeps
    a step only if it dominates its own neighbourhood. Like the other z-scoring detectors here it
    standardizes first, so one ``prominence_k`` spans panels of wildly different scale.

    Smoothing is deliberately light (see ``segmentation._abd``): widening it past the curve's own
    correlation length blurs away the transitions being looked for — measured here as well as on
    the multivariate side, recall of the true boundaries falls from 1.0 at +/-2 to 0.39 at +/-6.
    ``prominence_k`` is the real lever and needs to be higher than the multivariate detector's,
    because a scalar has no averaging across dimensions to suppress per-row noise: at 0.25 this
    cuts ~18 times where the truth is 3, at 0.75 it is 4 with full recall.
    """
    from scipy.signal import find_peaks

    z = _standardize(y)
    if len(z) < 3:
        return []
    smoothed = _smooth(z, passes=1, window=max(1, smooth_half))
    step = np.abs(np.diff(smoothed))
    if not step.any():
        return []
    peaks, _ = find_peaks(
        step,
        distance=max(1, nms_window),
        prominence=float(np.std(step)) * prominence_k,
    )
    return [int(p) + 1 for p in peaks]


def detect_curve_transitions(
    curve: np.ndarray, detector: str, params: dict
) -> list[int]:
    """Cut indices in ``curve``, detected within each finite stretch.

    The curve's ``NaN`` runs are its tracking gaps, so detecting per finite stretch is the same
    rule ``segmentation.detect_transitions`` applies per contiguous run — no cut spans a gap,
    and no window pools across one.

    Raises:
        ValueError: An unknown detector name.
    """
    if detector not in CURVE_DETECTORS:
        raise ValueError(
            f"unknown curve detector {detector!r}; have {sorted(CURVE_DETECTORS)}"
        )
    fn = CURVE_DETECTORS[detector]
    y = np.asarray(curve, dtype=np.float64)
    finite = np.isfinite(y)
    if not finite.any():
        return []
    # Contiguous finite stretches: detect inside each, then offset back to curve indices.
    edges = np.flatnonzero(np.diff(finite.astype(np.int8)))
    bounds = np.concatenate([[0], edges + 1, [len(y)]])
    cuts: list[int] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if not finite[a] or b - a < 3:
            continue
        for c in fn(y[a:b], **params):
            if 0 < c < b - a:
                cuts.append(int(a) + int(c))
    return sorted(set(cuts))
