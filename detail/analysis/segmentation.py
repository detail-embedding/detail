"""Change-point segmentation → bout clustering (§22 temporal clustering; written for the removed
§16 syllable discovery).

Recurring behavior "syllables" are found by cutting a subject's pose-embedding trajectory at
transitions, then clustering the resulting variable-length bouts by DTW into syllable types.
Unlike a matrix-profile self-join this segments *everything* (rare bouts included), so it is not
frequency-biased; unlike an AR-HMM it needs no learnable dynamics — the trajectory *changes
region* at behavior boundaries, which is what the detectors pick up. (Ported from the earlier prototype's
research bench, where the AR-HMM and matrix-profile routes were tried and dropped.)

Pure compute over arrays, in **rows** (like ``retrieval.py``): the caller passes a subject's
``(M, D)`` trajectory plus its continuous-run row ranges (from ``clip_query.segment_runs`` — so
no bout spans a tracking/time gap), and every length parameter here is in rows. The caller
(``bout_segmentation.py``, for temporal clustering) owns the ``t_ns`` ⇄ rows conversion (Tenet 8),
subject iteration, and I/O.

The trajectory is first turned into a **feature matrix** by ``build_features`` — pooled sliding
windows that smooth the per-frame jitter — living in its own *feature-index* space; ``row_of_X``
maps each feature back to a trajectory row so bouts come out as trajectory-row spans.

Pieces (numpy + a lazily-imported ``scipy.signal.find_peaks`` / ``sklearn.cluster.HDBSCAN`` — so
importing this module needs neither, only *running* a discovery does):

- ``TRANSITION_DETECTORS`` — a pluggable registry: ``"binseg"`` (mean-shift binary
  segmentation), ``"novelty"`` (windowed left/right cosine discrepancy), ``"speed"`` (cos-step
  peaks), ``"abd"`` (locally prominent peaks of a smoothed step curve, which over-cuts on
  purpose). Swap or add algorithms freely.
- ``CLUSTER_RULES`` — the second pluggable registry, for the *other* half of the pipeline: given
  the bout distance matrix, which grouping algorithm turns it into labels: ``"hdbscan"`` (density,
  the default, the only one that leaves noise), ``"finch"`` (parameter-free nearest-neighbour
  hierarchy), ``"agglomerative"`` (merge to a target count — abd's refinement step). A rule is
  ``fn(dist, **params) -> labels``, so anything that consumes a precomputed distance matrix drops
  in without touching the fit runner.
- ``build_features`` → ``(X, row_of_X, xruns)``; ``detect_transitions`` → cut indices;
  ``segment_bouts`` → variable-length bouts; ``cluster_labels`` → labels via the chosen rule
  (``cluster_from_dist`` is HDBSCAN directly); ``medoid_order`` → a syllable's members ranked by
  closeness to its center.
"""

import heapq
from collections import namedtuple
from collections.abc import Callable

import numpy as np

TRANSITION_DETECTORS: dict[str, Callable] = {}

# The score a peak-picking detector cuts on: ``values`` per feature index, the ``threshold``
# a peak must clear, and the ``offset`` to add to a peak index to get the cut index (``speed``
# scores steps *between* features, so its curve is one shorter and its cuts sit one later).
# Diagnostics plot this to show *why* a cut landed where it did; nothing in the fit reads it.
TransitionCurve = namedtuple("TransitionCurve", ["values", "threshold", "offset"])

# Parallel to TRANSITION_DETECTORS, keyed the same way, but **optional**: a detector that does
# not score every index (``binseg`` recursively splits) simply has no entry, and a diagnostic
# then shows its cuts with no curve. Keeping this separate leaves the detector contract
# (``fn(X, **params) -> list[int]``) untouched, so a third-party detector needs no change.
TRANSITION_CURVES: dict[str, Callable] = {}

# The grouping half of the pipeline, mirroring TRANSITION_DETECTORS: ``fn(dist, **params) ->
# labels`` over a **precomputed** bout distance matrix (parallel to the bouts; -1 = noise).
# Keeping the contract at "distance matrix in, labels out" is what lets a rule be swapped without
# the fit runner knowing anything about it — everything downstream of the labels (medoids,
# assignment rows, noise fraction) is already rule-agnostic. A method that never builds a
# distance matrix does not belong here; it needs the frame path instead.
CLUSTER_RULES: dict[str, Callable] = {}


def register_transition_detector(name: str) -> Callable[[Callable], Callable]:
    """Register a detector ``fn(X, **params) -> list[int]`` (cut indices within ``X``)."""

    def deco(fn: Callable) -> Callable:
        TRANSITION_DETECTORS[name] = fn
        return fn

    return deco


def register_transition_curve(name: str) -> Callable[[Callable], Callable]:
    """Register the score a detector cuts on: ``fn(X, **params) -> TransitionCurve``."""

    def deco(fn: Callable) -> Callable:
        TRANSITION_CURVES[name] = fn
        return fn

    return deco


def register_cluster_rule(name: str) -> Callable[[Callable], Callable]:
    """Register a clustering rule ``fn(dist, **params) -> labels`` (-1 = noise)."""

    def deco(fn: Callable) -> Callable:
        CLUSTER_RULES[name] = fn
        return fn

    return deco


# The *other* shape a method can take: label every trajectory row directly and let segments emerge
# as runs of the same label, rather than cutting first and clustering the pieces. This is the
# majority shape in the action-segmentation literature and it cannot be expressed as a CLUSTER_RULE
# because there is no bout distance matrix to consume — the method never builds one.
# ``fn(trajectory, t_ns, runs, **params) -> labels`` (one per row).
#
# A frame rule works **within one recording**: it sees a single trajectory, so its cluster ids mean
# nothing across recordings.
FRAME_RULES: dict[str, Callable] = {}


def register_frame_rule(name: str) -> Callable[[Callable], Callable]:
    """Register a per-row rule ``fn(trajectory, t_ns, runs, **params) -> labels``."""

    def deco(fn: Callable) -> Callable:
        FRAME_RULES[name] = fn
        return fn

    return deco


@register_frame_rule("tw_finch")
def _tw_finch_frame_rule(
    trajectory: np.ndarray,
    t_ns: np.ndarray,
    runs: list[tuple[int, int]],
    *,
    n_clusters: int = 8,
    partition: int | None = None,
    max_rows: int = 6000,
    **_,
) -> np.ndarray:
    """Temporally weighted nearest-neighbour clustering of the trajectory (Sarfraz et al.).

    Each continuous run is clustered on its own and the labels are then made disjoint across runs:
    the temporal weighting is defined within one sequence, so carrying a label across a tracking
    gap would assert a continuity the data does not have.
    """
    from detail.analysis.tw_finch import tw_finch_labels

    trajectory = np.asarray(trajectory, dtype=np.float64)
    labels = np.full(len(trajectory), -1, dtype=np.int64)
    offset = 0
    for a, b in runs or [(0, len(trajectory))]:
        if b <= a:
            continue
        block = tw_finch_labels(
            trajectory[a:b],
            np.asarray(t_ns[a:b], dtype=np.float64),
            n_clusters=n_clusters,
            partition=partition,
            max_rows=max_rows,
        )
        labels[a:b] = block + offset
        offset += int(block.max()) + 1 if len(block) else 0
    return labels


def labels_to_bout_rows(frame_labels, frame_idx, t_ns, runs=None) -> list[dict]:
    """Run-length-encode per-row labels into bout assignment rows.

    This gives a frame-labeling method the same rows as the segment-then-cluster path, so a
    scorer reads both without knowing which produced them: a "bout" is simply a maximal run of
    one label. ``runs`` are half-open row ranges an encoded span may not
    cross, so a label that happens to continue across a tracking gap still yields two rows rather
    than one that claims the animal was doing the same thing throughout the gap.
    """
    labels = np.asarray(frame_labels)
    n = len(labels)
    if n == 0:
        return []
    bounds = runs or [(0, n)]
    rows: list[dict] = []
    for a, b in bounds:
        if b <= a:
            continue
        block = labels[a:b]
        # Boundaries wherever the label changes, plus the block's own edges.
        edges = [0, *(np.nonzero(np.diff(block) != 0)[0] + 1).tolist(), len(block)]
        for lo, hi in zip(edges, edges[1:]):
            r0, r1 = a + lo, a + hi - 1
            rows.append(
                {
                    "start_frame": int(frame_idx[r0]),
                    "end_frame": int(frame_idx[r1]),
                    "start_t_ns": int(t_ns[r0]),
                    "end_t_ns": int(t_ns[r1]),
                    "syllable_id": int(block[lo]),
                    "medoid_dist": 0.0,
                }
            )
    return rows


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0.0, n, 1.0)


# --- detectors (numpy + a lazily-imported scipy.signal.find_peaks) ---------------


@register_transition_curve("speed")
def _speed_curve(X: np.ndarray, *, percentile: float = 90.0, **_) -> TransitionCurve:
    """Per-step cosine change, and the height a peak must clear."""
    xn = _normalize(np.asarray(X, dtype=np.float64))
    if len(xn) < 3:
        return TransitionCurve(np.zeros(0), 0.0, 1)
    step = 1.0 - np.sum(
        xn[:-1] * xn[1:], axis=1
    )  # (n-1,) — scores the gaps, not the rows
    return TransitionCurve(step, float(np.percentile(step, percentile)), 1)


@register_transition_detector("speed")
def _speed(
    X: np.ndarray, *, percentile: float = 90.0, min_size: int = 3, **_
) -> list[int]:
    """Cut where the per-step cosine change spikes (a fast pose transition)."""
    from scipy.signal import find_peaks

    curve = _speed_curve(X, percentile=percentile)
    if not len(curve.values):
        return []
    peaks, _ = find_peaks(curve.values, height=curve.threshold, distance=min_size)
    return [int(p) + curve.offset for p in peaks]


@register_transition_curve("novelty")
def _novelty_curve(
    X: np.ndarray, *, half: int = 8, percentile: float = 90.0, **_
) -> TransitionCurve:
    """Left-vs-right windowed cosine discrepancy, and the height a peak must clear."""
    xn = _normalize(np.asarray(X, dtype=np.float64))
    n = len(xn)
    if n < 4:
        return TransitionCurve(np.zeros(0), 0.0, 0)
    c = np.concatenate([np.zeros((1, xn.shape[1])), np.cumsum(xn, axis=0)])
    t = np.arange(n)
    l0 = np.maximum(0, t - half)
    r1 = np.minimum(n, t + half)
    ml = (c[t] - c[l0]) / np.maximum(1, t - l0)[:, None]
    mr = (c[r1] - c[t]) / np.maximum(1, r1 - t)[:, None]
    ml = ml / (np.linalg.norm(ml, axis=1, keepdims=True) + 1e-12)
    mr = mr / (np.linalg.norm(mr, axis=1, keepdims=True) + 1e-12)
    nov = 1.0 - np.sum(ml * mr, axis=1)
    nov[(t - l0 < 1) | (r1 - t < 1)] = 0.0
    pos = nov[nov > 0]
    return TransitionCurve(
        nov, float(np.percentile(pos, percentile)) if len(pos) else 0.0, 0
    )


@register_transition_detector("novelty")
def _novelty(
    X: np.ndarray, *, half: int = 8, percentile: float = 90.0, min_size: int = 3, **_
) -> list[int]:
    """Cut where the mean pose of the left window differs most from the right window (cosine)."""
    from scipy.signal import find_peaks

    curve = _novelty_curve(X, half=half, percentile=percentile)
    if not len(curve.values):
        return []
    peaks, _ = find_peaks(curve.values, height=curve.threshold, distance=min_size)
    return [int(p) + curve.offset for p in peaks]


def _prefix(xn: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((1, xn.shape[1])), np.cumsum(xn, axis=0)])


def _seg_cost(csum: np.ndarray, a: int, b: int) -> float:
    """Within-segment scatter of unit vectors: (b-a) - ||sum||^2/(b-a)."""
    s = csum[b] - csum[a]
    return (b - a) - float(s @ s) / (b - a)


def _best_split(csum: np.ndarray, a: int, b: int, min_size: int) -> tuple[float, int]:
    ks = np.arange(a + min_size, b - min_size + 1)
    if len(ks) == 0:
        return (-np.inf, -1)
    left, right = csum[ks] - csum[a], csum[b] - csum[ks]
    ll, lr = ks - a, b - ks
    cl = ll - np.sum(left * left, axis=1) / ll
    cr = lr - np.sum(right * right, axis=1) / lr
    gain = _seg_cost(csum, a, b) - (cl + cr)
    i = int(np.argmax(gain))
    return (float(gain[i]), int(ks[i]))


@register_transition_detector("binseg")
def _binseg(
    X: np.ndarray,
    *,
    penalty: float | None = None,
    n_bkps: int | None = None,
    min_size: int = 3,
    **_,
) -> list[int]:
    """Binary segmentation with a mean-shift cost (cumulative sums) on L2-normalized features —
    cuts where the feature distribution's mean shifts. Stops at ``n_bkps`` splits or when the best
    remaining gain falls below ``penalty``."""
    xn = _normalize(np.asarray(X, dtype=np.float64))
    n = len(xn)
    if n < 2 * min_size:
        return []
    if penalty is None and n_bkps is None:
        raise ValueError("binseg needs penalty or n_bkps")
    csum = _prefix(xn)
    cuts: list[int] = []
    g, k = _best_split(csum, 0, n, min_size)
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
            g2, k2 = _best_split(csum, a2, b2, min_size)
            if k2 >= 0:
                heapq.heappush(heap, (-g2, a2, b2, k2))
    return sorted(cuts)


@register_transition_curve("abd")
def _abd_curve(X: np.ndarray, *, smooth_half: int = 2, **_) -> TransitionCurve:
    """Adjacent-step dissimilarity of **temporally smoothed** features (Du et al., CVPR 2022).

    ABD's observation is that the raw adjacent-frame similarity is too noisy to threshold, but
    after smoothing over ±``smooth_half`` it takes a "⊓" shape per action instance, so boundaries
    are the dips. Keep ``smooth_half`` small: smoothing wider than the features' own correlation
    length blurs away the transitions it is meant to sharpen — measured, recall of the true
    boundaries falls from 1.0 to below 0.6 going from ±2 to ±8. Returned as a dissimilarity (``1 - cos``) so boundaries are *peaks*, matching
    every other curve here and the diagnostics that plot them.

    The threshold is reported for display only: the detector picks locally prominent peaks by
    non-maximum suppression rather than by clearing a global cutoff, which is the whole point —
    a global cutoff is what over-segments a drifting curve.
    """
    xs = _smooth_rows(np.asarray(X, dtype=np.float64), smooth_half)
    if len(xs) < 3:
        return TransitionCurve(np.zeros(0), 0.0, 1)
    xn = _normalize(xs)
    step = 1.0 - np.sum(xn[:-1] * xn[1:], axis=1)  # (n-1,) — scores the gaps
    return TransitionCurve(step, float(np.median(step)), 1)


def _smooth_rows(x: np.ndarray, half: int) -> np.ndarray:
    """Box-filter each column over a ±``half`` row window (ABD's smoothing step)."""
    if half < 1 or len(x) == 0:
        return x
    csum = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half)
    hi = np.minimum(len(x), idx + half + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)[:, None]


@register_transition_detector("abd")
def _abd(
    X: np.ndarray,
    *,
    smooth_half: int = 2,
    nms_window: int = 5,
    min_size: int = 3,
    **_,
) -> list[int]:
    """Cut at locally prominent dips in adjacent-step similarity (ABD).

    Differs from ``speed`` in what decides a cut: ``speed`` keeps peaks above a global percentile,
    which over-segments wherever the curve's baseline drifts, while this keeps a peak only if it is
    the largest within ``nms_window`` either side. Deliberately over-segments — ABD pairs it with a
    merge back down to a target count, which here is the job of a clustering rule.
    """
    from scipy.signal import find_peaks

    curve = _abd_curve(X, smooth_half=smooth_half)
    if not len(curve.values):
        return []
    # A peak that dominates its neighbourhood: `distance` enforces the NMS spacing, and the
    # prominence floor drops peaks that are merely the top of a flat stretch.
    peaks, _ = find_peaks(
        curve.values,
        distance=max(1, nms_window),
        prominence=float(np.std(curve.values)) * 0.25,
    )
    return [int(p) + curve.offset for p in peaks if p + curve.offset >= min_size]


# --- features -------------------------------------------------------------------


#: The most bytes of stacked windows :func:`build_features` holds at once. Bounds the pooling's
#: working memory whatever the window or the recording's length; the output is unchanged by it.
_POOL_CHUNK_BYTES = 64 * 1024 * 1024


def _window_starts(runs, window: int, feat_stride: int):
    """Each run's window starts — the one place the feature grid is laid out."""
    for a, b in runs:
        starts = np.arange(a, b - window + 1, feat_stride)
        if len(starts):
            yield starts


def feature_rows(runs, *, window: int, feat_stride: int = 1) -> np.ndarray:
    """``row_of_X`` from :func:`build_features`, without the trajectory.

    The rows a feature matrix maps to are a function of the runs and the window alone, never of the
    values being pooled — so a caller that only needs *which* rows were analyzed (a reprojection
    building coverage) can ask for them without reading hundreds of megabytes of embedding per
    recording. Shares the grid with :func:`build_features` rather than repeating it, so the two
    cannot drift.
    """
    parts = [
        starts + window // 2 for starts in _window_starts(runs, window, feat_stride)
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def build_features(
    trajectory: np.ndarray,
    runs: list[tuple[int, int]],
    *,
    window: int,
    pool: str = "mean",
    feat_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """The feature matrix the detector consumes, the trajectory row each feature maps to, and the
    contiguous-run ranges in feature-index space.

    Within each continuous run, slide a ``window``-row window at ``feat_stride`` and pool it
    (``"mean"`` / ``"max"`` → ``(N, D)``, ``"flatten"`` → ``(N, window*D)``) — pooling smooths
    per-frame jitter. ``row_of_X[k]`` is the window's **center** trajectory row. ``window <= 1``
    degenerates to per-frame features. A run shorter than ``window`` contributes nothing; if no
    features build at all, returns empties (the caller records 0 syllables rather than raising).

    ``"mean"`` averages the window's poses; ``"max"`` takes the per-dimension maximum, so a
    distinctive pose occupying a few frames survives instead of being averaged away — the
    difference matters most when the window is long relative to the behavior. Both emit width
    ``D``; only ``"flatten"`` widens.

    Returns (X, row_of_X, xruns), with ``xruns`` half-open ``[start, end)`` feature-index ranges.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if pool not in ("mean", "max", "flatten"):
        raise ValueError(f"pool must be 'mean', 'max' or 'flatten', got {pool!r}")
    if feat_stride < 1:
        raise ValueError(f"feat_stride must be >= 1, got {feat_stride}")
    tn = _normalize(np.asarray(trajectory, dtype=np.float64))
    dim = tn.shape[1] if tn.ndim == 2 else 0
    # Windows are stacked a chunk at a time into one preallocated output, never a whole run at
    # once: a run's full `(windows, window, D)` stack is `window` times the trajectory, which at a
    # 0.53 s window was 22 GB for one 30-minute recording (#581). Each window reduces on its own,
    # so chunking changes no value — the output is bit-identical to stacking the run whole.
    grid = list(_window_starts(runs, window, feat_stride))
    width = window * dim if pool == "flatten" else dim
    if not grid:
        # Only ``flatten`` widens — the width above is the one place that is decided, so an empty
        # run returns the same shape a populated one would.
        return np.empty((0, width), dtype=np.float64), np.empty(0, dtype=np.int64), []
    out = np.empty((sum(len(starts) for starts in grid), width), dtype=tn.dtype)
    per_chunk = max(1, _POOL_CHUNK_BYTES // max(1, window * dim * tn.itemsize))
    xruns, xstart = [], 0
    for starts in grid:
        for lo in range(0, len(starts), per_chunk):
            chunk = starts[lo : lo + per_chunk]
            wins = np.stack([tn[s : s + window] for s in chunk])  # (chunk, window, D)
            at = slice(xstart + lo, xstart + lo + len(chunk))
            if pool == "mean":
                out[at] = wins.mean(axis=1)
            elif pool == "max":
                out[at] = wins.max(axis=1)
            else:
                out[at] = wins.reshape(len(chunk), -1)
        xruns.append((xstart, xstart + len(starts)))
        xstart += len(starts)
    # The window's center trajectory row.
    return out, np.concatenate([starts + window // 2 for starts in grid]), xruns


# --- segmentation ----------------------------------------------------------------


def detect_transitions(
    X: np.ndarray,
    xruns: list[tuple[int, int]],
    detector: str,
    params: dict,
) -> list[int]:
    """Run ``detector`` within each feature-run (so no cut spans a gap); return the sorted global
    cut indices in **feature-index** space. ``xruns`` are half-open ``[start, end)`` ranges.
    """
    if detector not in TRANSITION_DETECTORS:
        raise ValueError(
            f"unknown detector {detector!r}; have {sorted(TRANSITION_DETECTORS)}"
        )
    fn = TRANSITION_DETECTORS[detector]
    feats = np.asarray(X, dtype=np.float64)
    cuts: list[int] = []
    for xs, xe in xruns:
        for c in fn(feats[xs:xe], **params):
            if 0 < c < xe - xs:
                cuts.append(xs + c)
    return sorted(set(cuts))


def transition_curve(
    X: np.ndarray,
    xruns: list[tuple[int, int]],
    detector: str,
    params: dict,
) -> TransitionCurve | None:
    """The detector's score over **feature-index** space, aligned with ``detect_transitions``.

    Diagnostics only: shows *why* a cut landed where it did. ``None`` when the detector scores
    no such curve (``binseg`` recursively splits rather than peak-picking), so a caller must
    handle its absence rather than assume every detector explains itself.

    Each feature-run is scored independently, exactly as the cuts are, and the per-run scores
    are laid back into one full-length array; gaps between runs stay ``nan`` so a plot breaks
    the line there instead of drawing through a tracking gap. The threshold is the max over
    runs — each run percentile-thresholds itself, so a single number is only indicative, and
    the per-run values are what actually decided a cut.
    """
    fn = TRANSITION_CURVES.get(detector)
    if fn is None:
        return None
    feats = np.asarray(X, dtype=np.float64)
    values = np.full(len(feats), np.nan, dtype=np.float64)
    threshold, offset = 0.0, 0
    for xs, xe in xruns:
        curve = fn(feats[xs:xe], **params)
        if not len(curve.values):
            continue
        values[xs : xs + len(curve.values)] = curve.values
        threshold = max(threshold, curve.threshold)
        offset = curve.offset
    return TransitionCurve(values, threshold, offset)


def segment_bouts(
    cuts: list[int],
    row_of_X: np.ndarray,
    xruns: list[tuple[int, int]],
    min_len: int,
) -> list[tuple[int, int]]:
    """Feature-index cut edges → inclusive ``(start_row, end_row)`` trajectory-row bouts (mapped
    through ``row_of_X``), dropping bouts shorter than ``min_len`` rows. ``xruns`` are the
    feature-index ranges that tile the feature matrix."""
    cutset = sorted(cuts)
    row_of_X = np.asarray(row_of_X)
    bouts: list[tuple[int, int]] = []
    for xs, xe in xruns:
        edges = [xs] + [c for c in cutset if xs < c < xe] + [xe]
        for i in range(len(edges) - 1):
            r0 = int(row_of_X[edges[i]])
            r1 = int(row_of_X[edges[i + 1] - 1])
            if r1 - r0 + 1 >= min_len:
                bouts.append((r0, r1))
    return bouts


# --- clustering ------------------------------------------------------------------


def subsample_indices(n: int, max_segments: int | None) -> np.ndarray | None:
    """Indices of the evenly-spaced ``max_segments`` subsample of ``n`` bouts, or ``None`` for all.

    Shared so a caller that keeps its own per-bout table (a provenance/origin row per bout) can
    subset it by exactly the bouts ``pairwise_dtw`` will keep — a caller re-deriving the rule
    would silently misalign its table against the distance matrix.
    """
    if max_segments is None or n <= max_segments:
        return None
    return np.linspace(0, n - 1, max_segments).round().astype(int)


def pairwise_dtw(
    trajectory: np.ndarray,
    bouts: list[tuple[int, int]],
    band: int,
    max_segments: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """The K×K banded fixed-endpoint DTW distance matrix over the bouts (band auto-widened past
    the length gap so unequal-length bouts stay comparable). Bouts are evenly subsampled to
    ``max_segments`` if set — this is the expensive step and the only bound on its O(K²) cost.

    The matrix is filled by ``_dtw_numba.pairwise_block`` — a compiled, parallel kernel that fuses
    the cosine surface and the DP over every pair (numba imported lazily, so importing this module
    still needs no backend). It runs one row-block at a time so ``progress_cb(pairs_done,
    pairs_total)`` can report between blocks — a ``prange`` kernel can't call back into Python
    mid-run — so a single-subject run still shows live movement (§6)."""
    from detail.analysis import _dtw_numba

    idx = subsample_indices(len(bouts), max_segments)
    if idx is not None:
        bouts = [bouts[i] for i in idx]
    # Normalize each bout's rows once (cosine_cost normalized per pair before) and pack them
    # contiguously, so the kernel reads bout i as flat[offs[i] : offs[i] + lens[i]].
    segs = [
        _normalize(np.asarray(trajectory[a : b + 1], dtype=np.float64))
        for a, b in bouts
    ]
    k = len(segs)
    lens = np.array([len(s) for s in segs], dtype=np.int64)
    offs = np.zeros(k, dtype=np.int64)
    if k:
        offs[1:] = np.cumsum(lens)[:-1]
    flat = (
        np.vstack(segs)
        if segs
        else np.zeros((0, trajectory.shape[1]), dtype=np.float64)
    )
    dist = np.zeros((k, k))
    pairs_total = k * (k - 1) // 2
    # Row-blocks: small enough that progress reports often (>= ~64 emits), large enough to amortize
    # the kernel launch; the kernel parallelizes across the block's rows.
    block = max(1, k // 64)
    for i0 in range(0, k, block):
        i1 = min(i0 + block, k)
        _dtw_numba.pairwise_block(flat, offs, lens, band, i0, i1, dist)
        if progress_cb is not None:
            # pairs in rows [0, i1): row r has (k-1-r) upper-triangle pairs; the sum telescopes.
            done = i1 * (k - 1) - i1 * (i1 - 1) // 2
            progress_cb(done, pairs_total)
    return bouts, dist


def cluster_from_dist(
    dist: np.ndarray,
    min_cluster_size: int,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
) -> np.ndarray:
    """HDBSCAN labels from a precomputed distance matrix (parallel to the bouts; -1 = noise).
    Cheap given the matrix, so a caller may sweep granularities off one ``pairwise_dtw``.
    ``cluster_selection_method`` ``"eom"`` = most-stable clusters, ``"leaf"`` = the fine end.
    """
    from sklearn.cluster import HDBSCAN

    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="precomputed",
        copy=True,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_method=cluster_selection_method,
    ).fit_predict(dist)


@register_cluster_rule("hdbscan")
def _hdbscan_rule(
    dist: np.ndarray,
    *,
    min_cluster_size: int,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    **_,
) -> np.ndarray:
    """The registry face of ``cluster_from_dist`` — density clustering, noise allowed.

    A thin wrapper rather than decorating ``cluster_from_dist`` itself: that function takes its
    parameters positionally and the registry contract is keyword-only.
    """
    return cluster_from_dist(
        dist, min_cluster_size, cluster_selection_epsilon, cluster_selection_method
    )


def _union_find_labels(parent: list[int]) -> np.ndarray:
    """Compact connected-component ids from a union-find parent array."""

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    roots = [find(i) for i in range(len(parent))]
    order = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([order[r] for r in roots], dtype=np.int64)


def _components_of(nearest: np.ndarray) -> np.ndarray:
    """Connected components of a 1-nearest-neighbour graph, as compact labels.

    Sarfraz et al.'s observation is that the 1-nearest-neighbour graph already partitions data
    into meaningful groups — ``i`` joins ``j`` when ``j`` is ``i``'s neighbour, when ``i`` is
    ``j``'s, or when they share one. Unioning each ``i`` with its own nearest neighbour produces
    exactly those components, so no adjacency matrix is materialized.

    Split out from ``_finch_partition`` because the vector path finds its neighbours a different
    way (blocked cosine rather than an argmin over a precomputed matrix) and the component step is
    the half that must not differ between them.
    """
    n = len(nearest)
    if n < 2:
        return np.zeros(n, dtype=np.int64)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        a, b = find(i), find(int(nearest[i]))
        if a != b:
            parent[a] = b
    return _union_find_labels(parent)


def _finch_partition(dist: np.ndarray) -> np.ndarray:
    """One FINCH partition over a precomputed distance matrix."""
    n = len(dist)
    if n < 2:
        return np.zeros(n, dtype=np.int64)
    d = np.array(dist, dtype=np.float64, copy=True)
    np.fill_diagonal(d, np.inf)
    return _components_of(np.argmin(d, axis=1))


def _nearest_by_dot(rows: np.ndarray, block: int = 512) -> np.ndarray:
    """Each row's nearest neighbour under ``1 - a·b``, computed in row blocks.

    That one quantity is both things the recursion needs, which is what lets the whole hierarchy be
    built without ever forming an N×N matrix:

    - On **L2-normalized rows** it is the cosine distance.
    - On **unnormalized cluster means** it is the *average linkage* between those clusters. For unit
      vectors, ``mean_{a∈A, b∈B} (1 - x_a·x_b) = 1 - m_A·m_B`` where ``m`` is the plain mean, since
      the double sum factorizes. So average linkage never needs the pairwise distances it is defined
      over — see ``finch_hierarchy_from_vectors``.

    Only the arg-min per row is needed, so materializing the full matrix would cost O(N²) memory for
    a quantity that is O(N) — the same reason ``tw_finch._nearest_neighbors`` is blocked. Time is
    still O(N²); the caller bounds it by subsampling.
    """
    n = len(rows)
    out = np.empty(n, dtype=np.int64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        chunk = 1.0 - rows[start:stop] @ rows.T
        # Only this block's own diagonal is a self-distance; the rest of the rows are elsewhere.
        np.fill_diagonal(chunk[:, start:stop], np.inf)
        out[start:stop] = np.argmin(chunk, axis=1)
    return out


def _average_linkage(dist: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cluster-level distances: the mean pairwise distance between two clusters' members.

    FINCH recurses on cluster *means*, which a precomputed distance matrix does not have; average
    linkage is the standard stand-in and keeps the recursion in distance space throughout.
    """
    k = int(labels.max()) + 1 if len(labels) else 0
    onehot = np.zeros((len(labels), k))
    onehot[np.arange(len(labels)), labels] = 1.0
    counts = onehot.sum(axis=0)
    totals = onehot.T @ np.asarray(dist, dtype=np.float64) @ onehot
    out = totals / np.outer(counts, counts)
    np.fill_diagonal(out, 0.0)
    return out


def finch_hierarchy(dist: np.ndarray) -> list[np.ndarray]:
    """Every FINCH partition from finest to coarsest, each labeling all the original items.

    Parameter-free by construction: the recursion stops when a step stops merging or one cluster
    remains, which is usually a handful of levels.
    """
    n = len(dist)
    if n == 0:
        return []
    levels: list[np.ndarray] = []
    labels = _finch_partition(dist)
    levels.append(labels)
    current = _average_linkage(dist, labels)
    while len(current) > 1:
        step = _finch_partition(current)
        if len(set(step.tolist())) == len(current):
            break  # nothing merged; recursing again would not either
        labels = step[labels]
        levels.append(labels)
        current = _average_linkage(dist, labels)
        if len(set(labels.tolist())) <= 1:
            break
    return levels


def _cluster_means(feats: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Plain (unnormalized) mean feature per cluster.

    Deliberately not renormalized. ``1 - m_A·m_B`` on these means *is* the average linkage between
    the clusters; renormalizing would make it a cosine between centroids instead, which is a
    different aggregation and gives a different hierarchy.
    """
    k = int(labels.max()) + 1 if len(labels) else 0
    onehot = np.zeros((len(labels), k))
    onehot[np.arange(len(labels)), labels] = 1.0
    return (onehot.T @ feats) / onehot.sum(axis=0)[:, None]


def finch_hierarchy_from_vectors(feats: np.ndarray) -> list[np.ndarray]:
    """Every FINCH partition over **vectors**, finest first, each labeling all the original rows.

    The same hierarchy ``finch_hierarchy`` produces, computed without ever forming the N×N distance
    matrix — so it is bounded by time rather than by memory, and can fit on row counts the dense
    path cannot hold (1.15 GB at 12,000 rows, impossible beyond).

    The equality is exact, not approximate, and rests on one identity: for unit vectors the average
    linkage between two clusters is ``1 - m_A·m_B`` on their **plain means**, because the double sum
    over pairs factorizes. So the dense path's ``_average_linkage`` — which exists only because a
    precomputed matrix has no means to recurse on — is recoverable from the means themselves.
    ``tests/analysis/test_finch_vectors.py`` pins the two paths as identical at every level rather
    than merely similar, which is the only version of that guarantee worth having.

    ``feats`` need not be normalized; rows are L2-normalized on entry so the first level is a cosine
    neighbour graph.
    """
    x = np.asarray(feats, dtype=np.float64)
    if len(x) == 0:
        return []
    if len(x) < 2:
        return [np.zeros(len(x), dtype=np.int64)]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(norms > 0.0, norms, 1.0)

    labels = _components_of(_nearest_by_dot(x))
    levels = [labels]
    current = _cluster_means(x, labels)
    while len(current) > 1:
        step = _components_of(_nearest_by_dot(current))
        if len(set(step.tolist())) == len(current):
            break  # nothing merged; recursing again would not either
        labels = step[labels]
        levels.append(labels)
        current = _cluster_means(x, labels)
        if len(set(labels.tolist())) <= 1:
            break
    return levels


def parent_map(fine: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    """Each fine cluster's cluster in ``coarse``, read off any member.

    Only meaningful when ``coarse`` is a coarsening of ``fine`` — which is what a hierarchy's
    adjacent levels are, by construction. Every fine cluster then sits inside exactly one coarse
    one, so "any member" is well defined and the assignment is a single scatter.
    """
    fine = np.asarray(fine)
    coarse = np.asarray(coarse)
    if not len(fine):
        return np.zeros(0, dtype=np.int64)
    out = np.zeros(int(fine.max()) + 1, dtype=np.int64)
    out[fine] = coarse
    return out


def _greedy_merges(
    totals: np.ndarray, counts: np.ndarray, ids: list[int], next_node: int
) -> tuple[list[tuple[int, int, float]], int, int]:
    """Merge every item into one, closest average linkage first, recording the order.

    ``totals[i, j]`` is the **sum** of pairwise distances between two items and ``counts`` their
    sizes, so average linkage is ``totals / outer(counts, counts)`` and a merge is a plain
    addition — the exact Lance-Williams update ``_refine_to`` uses, for the same reason: rebuilding
    the linkage after every merge is O(n²k) and does not finish at the sizes a frame-level fit
    reaches.

    Returns the merges as ``(node_a, node_b, height)`` in real node ids, the id of the single node
    left at the end, and the next free id. Kept separate from ``_refine_to`` because that one
    answers "give me k clusters" and throws the path away; this one *is* the path.
    """
    totals = np.array(totals, dtype=np.float64, copy=True)
    counts = np.array(counts, dtype=np.float64, copy=True)
    ids = list(ids)
    merges: list[tuple[int, int, float]] = []
    while len(ids) > 1:
        linkage = totals / np.outer(counts, counts)
        np.fill_diagonal(linkage, np.inf)
        i, j = np.unravel_index(np.argmin(linkage), linkage.shape)
        lo, hi = (int(i), int(j)) if i < j else (int(j), int(i))
        merges.append((ids[lo], ids[hi], float(linkage[lo, hi])))
        ids[lo] = next_node
        next_node += 1
        del ids[hi]
        totals[lo, :] += totals[hi, :]
        totals[:, lo] += totals[:, hi]
        counts[lo] += counts[hi]
        keep = np.arange(len(counts)) != hi
        totals = totals[keep][:, keep]
        counts = counts[keep]
    return merges, ids[0], next_node


def _totals_from_means(means: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Pairwise distance **sums** between clusters, from their plain means.

    The same identity ``finch_hierarchy_from_vectors`` rests on: for unit rows the average linkage
    between two clusters is ``1 - m_A·m_B`` on their unnormalized means, so the sum is that times
    ``|A|·|B|``. This is what keeps the tree off the N×N matrix — the block is one group of
    clusters wide, never one row per fitted frame.
    """
    weights = np.asarray(counts, dtype=np.float64)
    linkage = (
        1.0
        - np.asarray(means, dtype=np.float64) @ np.asarray(means, dtype=np.float64).T
    )
    return linkage * np.outer(weights, weights)


def _tree_from_blocks(levels: list[np.ndarray], totals_for) -> "ClusterTree":
    """Walk a hierarchy's levels into one merge sequence, group by group.

    ``totals_for(level_index, members)`` supplies the pairwise-distance sums between the level's
    clusters named in ``members``, plus their sizes. Each of the coarser level's groups is merged
    down to one node **on its own**, so the level lands exactly on a merge boundary and every cut
    of the finished tree refines every coarser cut. A hierarchy that stops above a single cluster
    gets an unconstrained tail, which is also the shape a method with no nested levels of its own
    would take.
    """
    from detail.core.dataclasses import ClusterTree

    if not levels or not len(levels[0]):
        empty_int = np.zeros(0, dtype=np.int32)
        return ClusterTree(
            children=np.zeros((0, 2), dtype=np.int32),
            height=np.zeros(0, dtype=np.float64),
            size=empty_int,
            n_rows=empty_int,
            leaf_of_row=np.zeros(0, dtype=np.int32),
            rung_merges=(),
        )

    leaf_of_row = np.asarray(levels[0], dtype=np.int64)
    n_leaves = int(leaf_of_row.max()) + 1
    merges: list[tuple[int, int, float]] = []
    rung_merges = [0]
    node_of_cluster = list(range(n_leaves))
    next_node = n_leaves

    for index in range(len(levels) - 1):
        step = parent_map(levels[index], levels[index + 1])
        merged: list[int] = []
        for group in range(int(step.max()) + 1 if len(step) else 0):
            members = np.flatnonzero(step == group)
            if len(members) == 1:
                merged.append(node_of_cluster[int(members[0])])
                continue
            totals, counts = totals_for(index, members)
            block, final, next_node = _greedy_merges(
                totals, counts, [node_of_cluster[int(m)] for m in members], next_node
            )
            merges.extend(block)
            merged.append(final)
        node_of_cluster = merged
        rung_merges.append(len(merges))

    if len(node_of_cluster) > 1:
        last = len(levels) - 1
        members = np.arange(len(node_of_cluster))
        totals, counts = totals_for(last, members)
        block, _, next_node = _greedy_merges(
            totals, counts, list(node_of_cluster), next_node
        )
        merges.extend(block)

    children = np.array([[a, b] for a, b, _ in merges], dtype=np.int32).reshape(-1, 2)
    height = np.array([h for _, _, h in merges], dtype=np.float64)
    n_nodes = 2 * n_leaves - 1
    size = np.zeros(n_nodes, dtype=np.int32)
    rows = np.zeros(n_nodes, dtype=np.int32)
    size[:n_leaves] = 1
    rows[:n_leaves] = np.bincount(leaf_of_row, minlength=n_leaves)
    for merge in range(len(merges)):
        a, b = int(children[merge, 0]), int(children[merge, 1])
        size[n_leaves + merge] = size[a] + size[b]
        rows[n_leaves + merge] = rows[a] + rows[b]
    return ClusterTree(
        children=children,
        height=height,
        size=size,
        n_rows=rows,
        leaf_of_row=leaf_of_row.astype(np.int32),
        rung_merges=tuple(rung_merges),
    )


def cluster_tree_from_vectors(
    feats: np.ndarray, levels: list[np.ndarray]
) -> "ClusterTree":
    """The merge tree behind a vector hierarchy, built from cluster means.

    Never forms the row-to-row matrix: the widest array allocated is one group of clusters
    squared, against the ``(rows, rows)`` float64 the old between-rung refinement built — 288 MB
    at the default fit size and 20 GB at the largest the form offers.
    """
    x = np.asarray(feats, dtype=np.float64)
    if len(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.where(norms > 0.0, norms, 1.0)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def totals_for(index: int, members: np.ndarray):
        if index not in cache:
            labels = np.asarray(levels[index])
            cache[index] = (
                _cluster_means(x, labels),
                np.bincount(labels, minlength=int(labels.max()) + 1),
            )
        means, counts = cache[index]
        picked = counts[members]
        return _totals_from_means(means[members], picked), picked

    return _tree_from_blocks(levels, totals_for)


def cluster_tree_from_distances(
    dist: np.ndarray, levels: list[np.ndarray]
) -> "ClusterTree":
    """The merge tree behind a hierarchy over a precomputed distance matrix.

    The dense counterpart of ``cluster_tree_from_vectors``, for a method whose items have no
    vectors to average — the matrix is already in hand, so the cluster-level linkage comes from
    ``_average_linkage`` rather than from means.
    """
    d = np.asarray(dist, dtype=np.float64)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def totals_for(index: int, members: np.ndarray):
        if index not in cache:
            labels = np.asarray(levels[index])
            cache[index] = (
                _average_linkage(d, labels),
                np.bincount(labels, minlength=int(labels.max()) + 1),
            )
        linkage, counts = cache[index]
        picked = counts[members].astype(np.float64)
        block = linkage[np.ix_(members, members)]
        return block * np.outer(picked, picked), picked

    return _tree_from_blocks(levels, totals_for)


def tree_arrays(tree: "ClusterTree") -> dict[str, np.ndarray]:
    """A tree as plain arrays, for ``np.savez``. One writer, because three methods store one.

    ``rung_merges`` rides as an array rather than a tuple so the shard needs no pickled objects.
    """
    return {
        "tree_children": tree.children,
        "tree_height": tree.height,
        "tree_size": tree.size,
        "tree_n_rows": tree.n_rows,
        "tree_leaf_of_row": tree.leaf_of_row,
        "tree_rung_merges": np.asarray(tree.rung_merges, dtype=np.int64),
    }


def tree_from_arrays(data) -> "ClusterTree | None":
    """Read back what ``tree_arrays`` wrote, or ``None`` for a shard written before it existed.

    ``None`` rather than a raise: fits made before the tree was stored are on disk in real
    experiments, and the caller rebuilds. Keyed on one name so a half-written shard cannot be
    mistaken for a whole one.
    """
    from detail.core.dataclasses import ClusterTree

    if "tree_children" not in data:
        return None
    return ClusterTree(
        children=np.asarray(data["tree_children"], dtype=np.int32).reshape(-1, 2),
        height=np.asarray(data["tree_height"], dtype=np.float64),
        size=np.asarray(data["tree_size"], dtype=np.int32),
        n_rows=np.asarray(data["tree_n_rows"], dtype=np.int32),
        leaf_of_row=np.asarray(data["tree_leaf_of_row"], dtype=np.int32),
        rung_merges=tuple(int(v) for v in data["tree_rung_merges"]),
    )


@register_cluster_rule("finch")
def _finch_rule(
    dist: np.ndarray,
    *,
    n_clusters: int | None = None,
    partition: int | None = None,
    **_,
) -> np.ndarray:
    """Parameter-free nearest-neighbour clustering, selecting one level of its hierarchy.

    Unlike HDBSCAN this labels **every** bout — there is no noise class — so a fit using it scores
    at full coverage by construction, and coverage stops being a lever for comparing it against a
    density rule. ``n_clusters`` refines the coarsest partition with at least that many clusters
    down to exactly that count; ``partition`` picks a level directly (0 = finest). With neither,
    the finest partition is used.
    """
    levels = finch_hierarchy(np.asarray(dist, dtype=np.float64))
    if not levels:
        return np.zeros(len(dist), dtype=np.int64)
    if n_clusters is not None:
        return _refine_to(dist, levels, int(n_clusters))
    index = 0 if partition is None else max(0, min(int(partition), len(levels) - 1))
    return levels[index]


def _refine_to(
    dist: np.ndarray, levels: list[np.ndarray], n_clusters: int
) -> np.ndarray:
    """Merge one pair at a time from the tightest partition that still has enough clusters.

    The pairwise-distance **sums** are kept and updated rather than rebuilt. Average linkage is a
    sum over the product of the two cluster sizes, so a merge simply adds the two summands — the
    Lance-Williams update, exact rather than approximate. Each merge is then O(k) plus the O(k²)
    argmin, against the O(n²k) of rebuilding ``_average_linkage`` from scratch every time.

    That rebuild is why this had to change. It is free on the few hundred bouts the segment path
    produces, and it does not finish on the thousands of rows a frame-level fit needs: one 6,000-row
    cell ran 18 minutes without completing, and 21 seconds after this change, for the same
    partition. ``tests/analysis/test_refine_to.py`` pins the equality against a naive reference.

    The diagonal is deliberately left however the two updates leave it: every read fills it with
    ``inf`` first, so only the off-diagonals are load-bearing.
    """
    from scipy import sparse

    usable = [lab for lab in levels if int(lab.max()) + 1 >= n_clusters]
    labels = np.array(usable[-1] if usable else levels[0], copy=True)
    k = int(labels.max()) + 1
    n = len(labels)

    # One nonzero per row, so `onehot.T @ dist` costs O(n²) rather than O(n²k).
    onehot = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), labels)), shape=(n, k), dtype=np.float64
    )
    counts = np.asarray(onehot.sum(axis=0)).ravel()
    totals = np.asarray((onehot.T @ np.asarray(dist, dtype=np.float64)) @ onehot)

    while k > n_clusters:
        linkage = totals / np.outer(counts, counts)
        np.fill_diagonal(linkage, np.inf)
        i, j = np.unravel_index(np.argmin(linkage), linkage.shape)
        lo, hi = (int(i), int(j)) if i < j else (int(j), int(i))
        labels[labels == hi] = lo
        labels[labels > hi] -= 1
        totals[lo, :] += totals[hi, :]
        totals[:, lo] += totals[:, hi]
        counts[lo] += counts[hi]
        keep = np.arange(k) != hi
        totals = totals[keep][:, keep]
        counts = counts[keep]
        k -= 1
    return labels


@register_cluster_rule("agglomerative")
def _agglomerative_rule(dist: np.ndarray, *, n_clusters: int = 8, **_) -> np.ndarray:
    """Repeatedly merge the two closest groups until ``n_clusters`` remain (average linkage).

    This is ABD's refinement step (Du et al., CVPR 2022): having deliberately over-segmented, it
    merges the most similar segments until the target count is reached, which is what heals a
    behavior that got cut into fragments. Like ``finch`` it assigns every bout — no noise class.

    Not the same experiment as the distance-*threshold* agglomerative already ruled
    out: the count is fixed here rather than a cut height, so it cannot fail by
    picking a threshold.
    """
    from sklearn.cluster import AgglomerativeClustering

    n = len(dist)
    target = max(1, min(int(n_clusters), n))
    if n <= 1:
        return np.zeros(n, dtype=np.int64)
    return AgglomerativeClustering(
        n_clusters=target, metric="precomputed", linkage="average"
    ).fit_predict(np.asarray(dist, dtype=np.float64))


def cluster_labels(dist: np.ndarray, rule: str, params: dict) -> np.ndarray:
    """Labels for a precomputed bout distance matrix via the named rule (-1 = noise).

    Mirrors ``detect_transitions``: the caller names a rule and hands it a params dict, so
    swapping the grouping algorithm is a config change rather than a code change.

    Fewer than two bouts is answered here rather than by each rule: a lone bout cannot form a
    cluster under any grouping algorithm, so it is noise by definition. Handling it centrally
    keeps the contract total — a rule may assume it gets at least two bouts — and replaces
    HDBSCAN's ``n_samples=1`` traceback, which a short recording or an aggressive detector can
    still reach today, with the honest answer of zero syllables.
    """
    if rule not in CLUSTER_RULES:
        raise ValueError(f"unknown cluster rule {rule!r}; have {sorted(CLUSTER_RULES)}")
    dist = np.asarray(dist)
    if len(dist) < 2:
        return np.full(len(dist), -1, dtype=np.int64)
    return CLUSTER_RULES[rule](dist, **params)


def medoid_order(dist: np.ndarray, members: list[int]) -> list[tuple[int, float]]:
    """A syllable's members ranked by closeness to its **medoid** — ascending mean DTW distance to
    the other members (most central first). Returns ``[(bout_index, mean_distance), ...]``; a lone
    member scores 0. Reuses the bout DTW matrix, so ranking every member is essentially free.
    """
    members = list(members)
    if not members:
        return []
    if len(members) == 1:
        return [(int(members[0]), 0.0)]
    idx = np.asarray(members, dtype=int)
    sub = np.asarray(dist, dtype=np.float64)[np.ix_(idx, idx)]  # (m, m), diagonal 0
    mean_to_others = sub.sum(axis=1) / (len(idx) - 1)
    order = np.argsort(mean_to_others, kind="stable")
    return [(int(idx[o]), float(mean_to_others[o])) for o in order]
