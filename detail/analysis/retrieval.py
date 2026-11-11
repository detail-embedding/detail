"""Time-tolerant subsequence retrieval over pose embeddings (§16 clip query).

"Query for clip": given a query subsequence of pose embeddings, find the stretches
of a recording's embedding trajectory most similar to it by cosine, tolerating mild
speed/duration differences. The matcher is **banded subsequence dynamic time warping
(DTW)** over a cosine-distance surface: the full query is matched, the start and end
positions in the database float, and a Sakoe-Chiba ``band`` caps how far the alignment
may drift from 1:1 — so the same action over ``N`` vs ``N+M`` frames still matches
(``M`` up to ``band``). ``band=0`` reduces to a rigid fixed-length cosine match.

Pure compute over arrays (Tenet 2/3 — no zarr, no plotting, no paths); the storage +
orchestration live in ``clip_query.py`` / the query and query-eval modules. Ported from the earlier
prototype's matcher with three changes:

- **Correct + vectorized DP.** The earlier prototype kept the lowest-cost path *per cell* and relaxed
  the left step with a per-column Python loop. But the band is measured against the
  path's start column, so two paths reaching a cell from different starts are not
  interchangeable — collapsing over start is a greedy approximation (it can report a
  cost well above the true optimum, and the result depends on the relaxation order).
  Here the DP is indexed by **drift** ``d = (j - start) - i`` (``d ∈ [-band, band]``):
  each of down/diag/left changes ``d`` by exactly ``-1``/``0``/``+1``, so a left move
  always *raises* ``d`` and never depends on the same drift level — the whole DP
  vectorizes over the database axis with an outer loop of only ``2*band+1`` drift
  levels. This is the exact banded-subsequence-DTW optimum in ``O(L·band·M)``.
- **Normalize the database once** across all query examples (the earlier prototype renormalized it
  per example).
- **Distance threshold**, not a fixed top-k, for match extraction.
"""

import bisect
import logging
from collections import namedtuple
from collections.abc import Callable, Sequence

import numpy as np

_log = logging.getLogger(__name__)

# A match into the database: inclusive row span + its (lower-is-better) score + the index of
# the example it best matches (0 for a single-example query; the argmin exemplar otherwise).
Match = namedtuple(
    "Match", ["start_row", "end_row", "score", "example_idx"], defaults=(0,)
)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm > 0.0, norm, 1.0)


def _cosine_cost_norm(query_norm: np.ndarray, database_norm: np.ndarray) -> np.ndarray:
    """(L, M) cosine-distance surface from **already-normalized** inputs (one matmul)."""
    return 1.0 - np.clip(query_norm @ database_norm.T, -1.0, 1.0)


def cosine_cost(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """(L, M) cosine-distance surface ``1 - cosine_similarity`` between every query row
    and every database row.

    Args
    query: (L, D) the query subsequence embeddings
    database: (M, D) the recording's pose trajectory embeddings

    Returns
    cost: (L, M) float32 in [0, 2]; 0 = identical direction
    """
    q = _l2_normalize(np.asarray(query, dtype=np.float32))
    d = _l2_normalize(np.asarray(database, dtype=np.float32))
    return _cosine_cost_norm(q, d)


def dtw_distance(cost: np.ndarray, band: int) -> float:
    """Length-normalized **fixed-endpoint** banded DTW distance over a cost surface.

    Unlike :func:`subsequence_dtw` (free start/end), both ends are pinned — it aligns the
    whole of one sequence to the whole of another, for comparing two extracted bouts (the
    bout comparison inside temporal clustering's ``dtw_finch``). ``band`` bounds ``|i - j|`` (0 =
    rigid diagonal). Returns the accumulated cost divided by the **realized path length**, so a
    distance is an average per-frame cosine distance — the same normalization :func:`match_curve`
    uses, which makes clustering distances comparable to retrieval scores.

    Args
    cost: (L1, L2) per-frame cost surface (e.g. ``cosine_cost`` between two bouts)
    band: Sakoe-Chiba warp tolerance in rows

    Returns
    distance: length-normalized accumulated DTW cost (``inf`` if the endpoint is
        unreachable, i.e. ``band < |L1 - L2|``)
    """
    # The min-plus DP is a sequential per-cell loop run K²/2 times in clustering — the whole
    # cost of a bout-clustering fit. It is compiled in ``_dtw_numba`` (imported lazily so the
    # package still imports without the analysis extra; see that module's docstring).
    from detail.analysis import _dtw_numba

    cost = np.asarray(cost, dtype=np.float64)
    return float(_dtw_numba.dtw_dp(cost, band))


def subsequence_dtw(
    cost: np.ndarray,
    band: int,
    drift_bounds: tuple[int, int] | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Banded subsequence DTW: match the full query (rows of ``cost``) against the
    database (columns) with a free start and end column.

    The Sakoe-Chiba band bounds ``|(j - start) - i| <= band`` along a path (database
    progress minus query progress), which is both the warp limit and the matched-length
    tolerance. The DP is indexed by that quantity, the **drift** ``d = (j - start) - i``
    (``d ∈ [-band, band]``, ``W = 2*band+1`` levels): down decreases ``d`` by 1, diagonal
    keeps it, left increases it by 1 — so left never reads its own drift level and the
    whole recurrence vectorizes over the database axis with a short outer loop over drift.
    Per database column ``j`` it returns the best full-query path *ending* at ``j``.

    ``drift_bounds`` restricts which drifts may *win*, which is how a caller asks for the best
    match of a bounded **duration**: a path's matched span is ``L + drift`` rows, so a span
    range maps to a drift range exactly. Constraining here rather than filtering the result
    afterwards is the difference between "the best in-range match" and "the best match, or
    nothing": with a post-filter, a column whose unconstrained winner is out of range is
    discarded even when an in-range path through the same column scored perfectly well.

    ``progress_cb`` is called with the fraction of **query rows** folded so far. This loop is
    the finest thing a long search has to report against: one exemplar against one continuous
    stretch is a single call from every caller above, so without a hook here the ordinary
    one-exemplar run shows no movement at all until it finishes (measured: a single-exemplar
    search emitted ``[1.0, 1.0]`` and nothing else). Each iteration does the same vectorized
    work over ``width x n_db``, so the fraction is honest rather than nominal — per-iteration
    cost measured near-uniform.

    Returns
    end_cost: (M,) accumulated cosine cost of that path (``inf`` where unreachable)
    start_col: (M,) database column where the path starts
    path_len: (M,) number of steps in the path (for length normalization)
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_query, n_db = cost.shape
    width = 2 * band + 1
    j_idx = np.arange(n_db)

    # Row 0: a full-query path is a single cell (0, j), which is its own start, so the
    # drift is 0 (the row `band` of the drift axis) with cost cost[0, j], length 1.
    cost_dp = np.full((width, n_db), np.inf)
    length_dp = np.zeros((width, n_db), dtype=np.int64)
    cost_dp[band] = cost[0]
    length_dp[band] = 1
    # ...and query row 0 may also be *held* across several database columns — the path
    # (0,s), (0,s+1), ... (0,j), whose drift is j - s. Seeding drift 0 alone left that whole
    # family unrepresentable, because the left step below lives inside the `i >= 1` loop, so
    # the search was not the optimum this function documents itself as computing.
    #
    # Holding is not "the same path plus extra cells": it starts query row 1 further along, so
    # the rest of the query aligns somewhere else and can be genuinely cheaper. That is why
    # the omission changed answers rather than merely costing paths — and under `drift_bounds`
    # it also forced the left moves onto rows 1..L-1, returning a suboptimal in-range path or,
    # for a single-row query, none at all.
    for k in range(band + 1, width):
        held_cost = np.full(n_db, np.inf)
        held_len = np.zeros(n_db, dtype=np.int64)
        held_cost[1:] = cost_dp[k - 1][:-1]
        held_len[1:] = length_dp[k - 1][:-1]
        start = j_idx - (k - band)
        ok = np.isfinite(held_cost) & (start >= 0)
        cost_dp[k] = np.where(ok, held_cost + cost[0], np.inf)
        length_dp[k] = np.where(ok, held_len + 1, 0)

    # ~20 updates per exemplar: enough that a minutes-long pass never looks stopped, few
    # enough that the callback chain up to the throttle costs nothing measurable.
    step = max(1, (n_query - 1) // 20)
    for i in range(1, n_query):
        if progress_cb is not None and i % step == 0:
            progress_cb(i / (n_query - 1))
        local = cost[i]
        prev_cost, prev_len = cost_dp, length_dp
        cost_dp = np.full((width, n_db), np.inf)
        length_dp = np.zeros((width, n_db), dtype=np.int64)
        # Drift levels ascend so the left step reads the already-filled level below it.
        for k in range(width):
            drift = k - band
            best = np.full(n_db, np.inf)
            best_len = np.zeros(n_db, dtype=np.int64)
            if k + 1 < width:  # down: predecessor (i-1, j), drift d+1
                best, best_len = _take(
                    best, best_len, prev_cost[k + 1], prev_len[k + 1]
                )
            diag_cost = np.full(n_db, np.inf)  # diagonal: (i-1, j-1), same drift
            diag_len = np.zeros(n_db, dtype=np.int64)
            diag_cost[1:] = prev_cost[k][:-1]
            diag_len[1:] = prev_len[k][:-1]
            best, best_len = _take(best, best_len, diag_cost, diag_len)
            if k - 1 >= 0:  # left: current row (i, j-1), drift d-1
                left_cost = np.full(n_db, np.inf)
                left_len = np.zeros(n_db, dtype=np.int64)
                left_cost[1:] = cost_dp[k - 1][:-1]
                left_len[1:] = length_dp[k - 1][:-1]
                best, best_len = _take(best, best_len, left_cost, left_len)
            start = j_idx - i - drift  # invariant along the path
            ok = np.isfinite(best) & (start >= 0) & (start <= j_idx)
            cost_dp[k] = np.where(ok, best + local, np.inf)
            length_dp[k] = np.where(ok, best_len + 1, 0)

    end_cost = np.full(n_db, np.inf)
    start_col = np.full(n_db, -1, dtype=np.int64)
    path_len = np.ones(n_db, dtype=np.int64)
    lo_drift, hi_drift = (-band, band) if drift_bounds is None else drift_bounds
    for k in range(width):
        drift = k - band
        if not lo_drift <= drift <= hi_drift:
            continue
        better = cost_dp[k] < end_cost
        end_cost = np.where(better, cost_dp[k], end_cost)
        start_col = np.where(better, j_idx - (n_query - 1) - drift, start_col)
        path_len = np.where(better, length_dp[k], path_len)
    return end_cost, start_col, path_len


def _take(
    best: np.ndarray, best_len: np.ndarray, cand: np.ndarray, cand_len: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise ``min`` of a predecessor candidate into the running best, carrying
    its path length wherever it wins."""
    better = cand < best
    return np.where(better, cand, best), np.where(better, cand_len, best_len)


def _match_curve_norm(
    database_norm: np.ndarray,
    query: np.ndarray,
    band: int,
    length_bounds: tuple[int, int] | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``match_curve`` on an already-normalized database (normalize the query here).

    ``length_bounds`` is an inclusive span-length range; a path's span is ``L + drift``, so it
    converts to a drift range and constrains the DP's choice rather than filtering after it.
    """
    query_norm = _l2_normalize(np.asarray(query, dtype=np.float32))
    cost = _cosine_cost_norm(query_norm, database_norm)
    drift_bounds = None
    if length_bounds is not None:
        n_query = len(query_norm)
        drift_bounds = (length_bounds[0] - n_query, length_bounds[1] - n_query)
    end_cost, start_col, path_len = subsequence_dtw(
        cost, band, drift_bounds, progress_cb
    )
    return end_cost / path_len, start_col  # path_len >= 1; inf/finite stays inf


# --- Selectable similarity methods (matcher seam; §16 exploration) --------------------
# Each curve_fn maps (database_norm (M,D), query_norm (L,D), band, params) to the SAME
# contract the DTW curve produces: (norm_cost (M,), start_col (M,)) — the cost of the best
# window *ending* at each row (``inf`` where none fits) and that window's inclusive start.
# So extract_matches / fusion / example_of / max_results stay matcher-agnostic. All costs
# are on the [0, 2] cosine scale, so the permissive-threshold eval needs no recalibration.
# ``band`` is the shared duration tolerance: pooled/keyframe pick the best window length
# ``L ± band`` ending at each row, giving variable-length spans (so temporal IoU is
# meaningful) at O(M·D) instead of the DTW DP's cost.
#
# Each also takes ``progress_cb``, called with the fraction of this one example's pass. Only
# DTW has anything to report against — its DP walks the query row by row — so the O(M·D)
# matchers accept the argument and ignore it rather than the seam differing per matcher.


def _dtw_curve(database_norm, query_norm, band, params, progress_cb=None):
    # `length_bounds` is the only param DTW reads: it constrains which spans the DP may
    # choose, so a bounded duration returns the best *in-range* match rather than being
    # filtered down to nothing wherever the unconstrained winner falls outside.
    return _match_curve_norm(
        database_norm,
        query_norm,
        band,
        (params or {}).get("length_bounds"),
        progress_cb,
    )


# Cap the swept window lengths so a large band stays affordable: each length is an O(M·D)
# pass, so sweeping every integer over L±band (band can be hundreds of rows) is memory-churn
# bound. A strided set of this many lengths keeps the duration tolerance at bounded cost.
_MAX_WINDOW_STEPS = 16


def _window_lengths(query_len: int, band: int, n_db: int) -> list[int]:
    if band <= 0:
        return [query_len] if query_len <= n_db else []
    lo, hi = max(1, query_len - band), min(n_db, query_len + band)
    if hi < lo:
        return []
    if hi - lo + 1 <= _MAX_WINDOW_STEPS:
        return list(range(lo, hi + 1))
    return sorted({int(round(x)) for x in np.linspace(lo, hi, _MAX_WINDOW_STEPS)})


def _pool_descriptor(rows: np.ndarray, pool: str) -> np.ndarray:
    desc = rows.mean(axis=0) if pool == "mean" else rows.max(axis=0)
    return _l2_normalize(desc[None, :].astype(np.float32))[0]


def _sliding_reduce(x: np.ndarray, length: int, reduce: str) -> np.ndarray:
    """(M-L+1, ...) window max/min over a length-``length`` window along axis 0."""
    view = np.lib.stride_tricks.sliding_window_view(x, length, axis=0)
    return view.max(axis=-1) if reduce == "max" else view.min(axis=-1)


def _update_best(best, start, cost, length, n_db) -> None:
    ends = np.arange(length - 1, n_db)
    improved = cost < best[ends]
    hit = ends[improved]
    best[hit] = cost[improved]
    start[hit] = hit - length + 1


def _pooled_curve(database_norm, query_norm, band, params, progress_cb=None):
    """Pool the query to one descriptor and each sliding window likewise; cosine. ``mean``
    pooling sweeps window length via a cumsum (cheap); ``max`` uses the exemplar length.
    """
    pool = params.get("pool", "mean")
    n_db, dim = database_norm.shape
    q_desc = _pool_descriptor(query_norm, pool)
    best = np.full(n_db, np.inf)
    start = np.full(n_db, -1, dtype=np.int64)
    if pool == "mean":
        csum = np.cumsum(np.vstack([np.zeros((1, dim)), database_norm]), axis=0)
        for length in _window_lengths(len(query_norm), band, n_db):
            win = _l2_normalize(
                ((csum[length:] - csum[:-length]) / length).astype(np.float32)
            )
            cost = 1.0 - np.clip(win @ q_desc, -1.0, 1.0)
            _update_best(best, start, cost, length, n_db)
    else:  # max: a per-dim sliding max over D is O(M·D·L); use the exemplar length only
        length = min(len(query_norm), n_db)
        win = _l2_normalize(
            _sliding_reduce(database_norm, length, "max").astype(np.float32)
        )
        cost = 1.0 - np.clip(win @ q_desc, -1.0, 1.0)
        _update_best(best, start, cost, length, n_db)
    return best, start


def _select_keyframes(query_norm: np.ndarray, database_norm: np.ndarray, params: dict):
    """The query's ``n_keyframes`` most query-defining rows by ``keyframe_select``."""
    n = max(1, int(params.get("n_keyframes", 1)))
    how = params.get("keyframe_select", "medoid")
    if how == "medoid":  # closest to the query centroid — the typical frame
        center = _l2_normalize(query_norm.mean(axis=0)[None, :])[0]
        rank = np.argsort(-(query_norm @ center))
    elif how == "distinctive":  # farthest from the background (db) mean — stands out
        db_mean = _l2_normalize(database_norm.mean(axis=0)[None, :])[0]
        rank = np.argsort(query_norm @ db_mean)
    elif how == "rare":  # least similar to the database on average — rarest pose
        rank = np.argsort((query_norm @ database_norm.T).mean(axis=1))
    else:
        raise ValueError(f"unknown keyframe_select {how!r}")
    return query_norm[rank[:n]]


def _keyframe_curve(database_norm, query_norm, band, params, progress_cb=None):
    """Reduce the query to its most-defining frame(s); a good length-``L`` window is one that
    *contains* a frame close to each keyframe (per-keyframe sliding-min, averaged)."""
    n_db = len(database_norm)
    kf = _select_keyframes(query_norm, database_norm, params)  # (K, D)
    dist = 1.0 - np.clip(database_norm @ kf.T, -1.0, 1.0)  # (M, K)
    best = np.full(n_db, np.inf)
    start = np.full(n_db, -1, dtype=np.int64)
    for length in _window_lengths(len(query_norm), band, n_db):
        cost = _sliding_reduce(dist, length, "min").mean(axis=1)  # (M-L+1,)
        _update_best(best, start, cost, length, n_db)
    return best, start


# method id → (curve fn, the params that *define* this method as opposed to tune it).
# The two pooled entries are one implementation behind two ids, and which reduction each uses
# is data about the method rather than a branch on its name: a third pooled variant is a row
# here, not another `elif` in the resolver.
_CURVE_FNS: dict[str, tuple] = {
    "dtw": (_dtw_curve, {}),
    "clip_pooled_mean": (_pooled_curve, {"pool": "mean"}),
    "clip_pooled_max": (_pooled_curve, {"pool": "max"}),
    "keyframe": (_keyframe_curve, {}),
}


def _resolve_curve_fn(method: str, method_params: dict | None) -> tuple:
    """The curve fn for ``method`` and the params to call it with.

    A method's own defining params win over the caller's: the pipeline passes one flat param
    dict for every method (``keyframe_select``/``n_keyframes`` go to ``dtw`` too, harmlessly),
    so a caller must not be able to turn ``clip_pooled_max`` into a mean by passing ``pool``.
    """
    entry = _CURVE_FNS.get(method)
    if entry is None:
        raise ValueError(f"unknown method {method!r}; one of {sorted(_CURVE_FNS)}")
    curve_fn, defining = entry
    return curve_fn, {**(method_params or {}), **defining}


def match_curve(
    database: np.ndarray, query: np.ndarray, band: int
) -> tuple[np.ndarray, np.ndarray]:
    """Length-normalized cost of the best full-query match *ending* at each database
    position, plus that match's start column.

    Args
    database: (M, D) the recording's pose trajectory
    query: (L, D) the query subsequence
    band: warp / duration tolerance in rows (0 = rigid fixed-length match)

    Returns
    norm_cost: (M,) average per-frame cosine distance of the best path ending at each
        column; ``inf`` where no full-query match can end there
    start: (M,) the database column where that best path starts
    """
    _validate_shapes(database, query, band)
    return _match_curve_norm(
        _l2_normalize(np.asarray(database, dtype=np.float32)), query, band
    )


def _validate_shapes(database: np.ndarray, query: np.ndarray, band: int) -> None:
    n_query, n_db = len(query), len(database)
    if n_query == 0:
        raise ValueError("query is empty")
    if n_query > n_db:
        raise ValueError(f"query length {n_query} exceeds database length {n_db}")
    if band < 0:
        raise ValueError(f"band must be >= 0, got {band}")


def _too_close(
    start: int, end: int, other: tuple[int, int] | None, separation: int
) -> bool:
    """True if ``[start, end]`` is within ``separation`` of the inclusive span ``other``.

    Private again: it was made public for a chunk-boundary caller that never arrived. §20's
    methods merge *curves* and extract once over the whole trajectory, so no seam applies this
    rule on its own — ``extract_matches`` is the only caller, and it is the one that is
    genuinely cross-module.
    """
    if other is None:
        return False
    other_start, other_end = other
    return not (end + separation < other_start or other_end + separation < start)


def _overlap_rows(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Inclusive-row overlap between ``[a_start, a_end]`` and ``[b_start, b_end]``
    (``<= 0`` when disjoint or merely adjacent)."""
    return min(a_end, b_end) - max(a_start, b_start) + 1


class _SpanIndex:
    """Spans kept sorted by start, so a conflict query examines a handful instead of all of them.

    ``extract_matches`` used to ask ``any(predicate(span, t) for t in taken)`` per candidate.
    That is linear in the matches taken so far, and at §20's **permissive** limit every finite
    row is a candidate — so the extraction went quadratic in trajectory rows. Measured on the
    real shape (``min_separation`` = the longest exemplar, so the match count scales with length
    too): 0.16 s at 30k rows, 9.03 s at 240k, a clean 4x per doubling. Extrapolated to a
    2.5-hour recording at 60 fps that is ~46 s **per unit**, re-paid on every cutoff, which is
    what made a calibration slider impossible rather than merely slow.

    The removed ``clip_query`` module never hit it: it extracted at the user's own threshold, so
    only a small subset of rows was ever a candidate. The permissive extraction is what §20's cutoff-free
    search introduced, and this is the structure that pays for it.

    **The bound is exact, not a heuristic.** A stored span can only touch ``[start, end]`` if it
    begins at or before ``end + reach``, and — since it is at most ``max_len`` long — at or
    after ``start - reach - max_len + 1``. Both bounds are binary-searched and every span
    between them is handed to the real predicate, so the answer is identical to the full scan;
    only the spans that could not possibly match are skipped. Nothing here assumes the stored
    spans are disjoint or that their ends are ordered.
    """

    __slots__ = ("_starts", "_spans", "_max_len")

    def __init__(self, spans: Sequence[tuple[int, int]] = ()) -> None:
        ordered = sorted(spans)
        self._starts = [span[0] for span in ordered]
        self._spans = ordered
        self._max_len = max((b - a + 1 for a, b in ordered), default=0)

    def add(self, span: tuple[int, int]) -> None:
        i = bisect.bisect_left(self._starts, span[0])
        self._starts.insert(i, span[0])
        self._spans.insert(i, span)
        self._max_len = max(self._max_len, span[1] - span[0] + 1)

    def nearby(self, start: int, end: int, reach: int) -> list[tuple[int, int]]:
        """Every stored span that could touch ``[start, end]`` widened by ``reach``."""
        if not self._spans:
            return []
        lo = bisect.bisect_left(self._starts, start - reach - self._max_len + 1)
        hi = bisect.bisect_right(self._starts, end + reach)
        return self._spans[lo:hi]


def _same_match(a: tuple[int, int], b: tuple[int, int], overlap: int) -> bool:
    """Whether two inclusive-row spans are the *same* match under an ``overlap``-row
    tolerance: they overlap by more than ``overlap`` rows, **or** the shorter span is fully
    covered by the other. The containment clause makes an identical or contained span always
    a duplicate — even when ``overlap`` is larger than the span's own length (else a labeled
    match would not exclude itself)."""
    rows = _overlap_rows(a[0], a[1], b[0], b[1])
    shorter = min(a[1] - a[0] + 1, b[1] - b[0] + 1)
    return rows > overlap or rows >= shorter


def _mask_out_of_range(
    norms: np.ndarray,
    starts: np.ndarray,
    bounds: Sequence[tuple[int, int]],
) -> np.ndarray:
    """``inf`` out every position whose matched span falls outside its example's range.

    Masked **per example, before fusion**, so an example only contributes positions its own
    duration range allows — rather than after ``argmin`` has already picked a winner, which
    would drop a position the winning example cannot reach even when another example spans it
    acceptably. Under ``fusion="mean"`` the ``inf`` propagates, which is that mode's existing
    "every example must reach this position" semantics applied to length as well as cost.
    """
    positions = np.arange(norms.shape[1])
    masked = norms.copy()
    for e, (lo, hi) in enumerate(bounds):
        lengths = positions - starts[e] + 1
        masked[e] = np.where((lengths >= lo) & (lengths <= hi), norms[e], np.inf)
    return masked


def extract_matches(
    norm: np.ndarray,
    start: np.ndarray,
    *,
    threshold: float,
    min_separation: int,
    excludes: Sequence[tuple[int, int]] = (),
    overlap: int | None = None,
    example_of: np.ndarray | None = None,
    max_results: int | None = None,
) -> list[Match]:
    """Every position with ``norm <= threshold``, best (lowest) first, kept apart from an
    already-taken match or an ``excludes`` span. ``max_results`` is a safety cap that
    **logs** when it truncates (a low threshold can otherwise flood) — never a silent limit.

    Public because §20's query methods call it: turning a stored curve into instances is the
    same extraction the in-module search does, and a method that reimplemented it would be
    free to disagree with the pipeline about what a match is.

    **Returned best-first**, which is what makes one extraction serve every cutoff: greedy
    suppression takes candidates in ascending cost, so a worse candidate can never suppress a
    better one and the result at a stricter threshold is exactly a prefix of this list.

    Spacing rule: ``overlap is None`` keeps matches **non-overlapping** with at least
    ``min_separation`` rows between them and from every ``excludes`` span (the default,
    clip-query behavior). ``overlap`` (rows, ``>= 0``) switches to **overlap-tolerant** —
    a candidate is dropped when it overlaps a taken match or an ``excludes`` span by **more
    than** ``overlap`` rows, or is fully contained in one (so an identical/contained span is
    always a duplicate); ``min_separation`` is not applied in this branch.
    """
    if min_separation < 0:
        raise ValueError(f"min_separation must be >= 0, got {min_separation}")
    if overlap is not None and overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    reachable = np.nonzero(np.isfinite(norm) & (norm <= threshold))[0]
    order = reachable[np.argsort(norm[reachable], kind="stable")]
    matches: list[Match] = []
    # Both sets are queried per candidate, so both are indexed (see `_SpanIndex` for why the
    # plain scan could not survive the permissive limit). `reach` is how far outside its own
    # span the predicate can look: `min_separation` for the spacing rule, nothing for the
    # overlap rule, which needs genuine overlap.
    excluded = _SpanIndex(excludes)
    taken = _SpanIndex()
    reach = min_separation if overlap is None else 0
    for end in order:
        end = int(end)
        span = (int(start[end]), end)
        if overlap is None:
            if any(
                _too_close(*span, ex, min_separation)
                for ex in excluded.nearby(*span, reach)
            ):
                continue
            if any(
                _too_close(*span, t, min_separation) for t in taken.nearby(*span, reach)
            ):
                continue
        else:
            if any(
                _same_match(span, ex, overlap) for ex in excluded.nearby(*span, reach)
            ):
                continue
            if any(_same_match(span, t, overlap) for t in taken.nearby(*span, reach)):
                continue
        example_idx = int(example_of[end]) if example_of is not None else 0
        matches.append(Match(span[0], end, float(norm[end]), example_idx))
        taken.add(span)
        if max_results is not None and len(matches) >= max_results:
            n_remaining = len(order) - list(order).index(end) - 1
            if n_remaining > 0:
                _log.warning(
                    "clip query hit max_results=%d at threshold=%g; %d further "
                    "candidate position(s) not returned — raise the threshold or "
                    "max_results to see them",
                    max_results,
                    threshold,
                    n_remaining,
                )
            break
    return matches


# --- per-frame scoring: score every row against the examples (§16) -------------------
# Structurally *not* a curve_fn (no windowing). ``frame_scores`` is the raw per-frame cosine
# similarity to the example keyframes (the reference curve the similarity figure overlays);
# ``soft_knn_frame_scores`` is the per-frame classifier the eval ranks into a per-frame
# Average Precision — a softmax-weighted vote calibrated against sampled background frames.

DEFAULT_SPAN_KEYFRAMES = 5


def select_keyframes(rows: np.ndarray, n: int) -> list[int]:
    """Indices of ``n`` spread-out representative rows, farthest-point in cosine space.

    Seeded from the medoid (the row most similar to the rest — the example's typical
    pose), then repeatedly taking whichever row is furthest from everything picked so
    far, so the set covers the example's range rather than clustering on its mode.
    Returns every row when the example has ``<= n`` of them.
    """
    if len(rows) <= n:
        return list(range(len(rows)))
    sims = rows @ rows.T
    picked = [int(sims.sum(axis=1).argmax())]
    while len(picked) < n:
        min_dist = (1.0 - sims[:, picked]).min(axis=1)
        min_dist[picked] = -1.0  # never re-pick a row already in the set
        picked.append(int(min_dist.argmax()))
    return picked


def frame_scores(
    database: np.ndarray,
    queries: Sequence[np.ndarray],
    *,
    fusion: str = "min",
    n_keyframes: int = DEFAULT_SPAN_KEYFRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row similarity to the example clips, plus which example each row resembles.

    Each example contributes ``n_keyframes`` keyframes; a row's score against one example
    is its best cosine to any of that example's keyframes. ``fusion`` then combines the
    per-example scores exactly as it does for the windowed matchers: ``"min"`` keeps the
    best single example (a row need only resemble *one* — recall), ``"mean"`` averages
    them (it must resemble *all* — consensus).

    Args
    database: (M, D) the recording's pose trajectory
    queries: list of (L_e, D) example subsequences (may differ in length)
    fusion: ``"min"`` (any example) or ``"mean"`` (every example)
    n_keyframes: keyframes taken per example

    Returns
    score: (M,) cosine similarity, **higher is better** (the inverse of the windowed
        matchers' cost scale). The raw similarity curve the eval's similarity figure overlays.
    example_of: (M,) index of the example each row best resembles
    """
    if not queries:
        raise ValueError("queries is empty")
    if fusion not in ("mean", "min"):
        raise ValueError(f"fusion must be 'mean' or 'min', got {fusion!r}")
    if n_keyframes < 1:
        raise ValueError(f"n_keyframes must be >= 1, got {n_keyframes}")
    database_norm = _l2_normalize(np.asarray(database, dtype=np.float32))
    per_example = np.empty((len(queries), len(database_norm)), dtype=np.float32)
    for e, query in enumerate(queries):
        query_norm = _l2_normalize(np.asarray(query, dtype=np.float32))
        if len(query_norm) == 0:
            raise ValueError(f"example {e} is empty")
        keyframes = query_norm[select_keyframes(query_norm, n_keyframes)]
        per_example[e] = np.clip(database_norm @ keyframes.T, -1.0, 1.0).max(axis=1)
    score = per_example.max(axis=0) if fusion == "min" else per_example.mean(axis=0)
    return score, per_example.argmax(axis=0)


#: Database rows scored per pass in :func:`soft_knn_frame_scores`. Sized so the ``(block, L)``
#: similarity block stays in cache at the labeled-set sizes this is used at: at L = 5130 it is
#: 168 MB against 2.2 GB for the whole matrix, and it runs *faster* for it, because the
#: reduction reads rows that are still warm. The gain grows with L, which is where the
#: memory problem was anyway.
_SOFT_KNN_BLOCK = 8192


def soft_knn_frame_scores(
    database: np.ndarray,
    pos_frames: np.ndarray,
    neg_frames: np.ndarray,
    *,
    k: int,
    tau: float,
    block: int = _SOFT_KNN_BLOCK,
    progress_cb: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Per-frame P(positive) by a soft (weighted) k-NN vote over labeled frames.

    Each database row's score is a ``softmax(cos / tau)``-weighted vote over its ``k``
    nearest labeled neighbors, where the labeled set is the positive exemplar frames
    (``pos_frames``) plus sampled background negatives (``neg_frames``). This is the
    DINO/DINOv2 frozen-feature evaluation protocol: the cosine similarity *is* the
    confidence, and the negatives calibrate it against the background rather than a
    proposal cutoff. Unlike :func:`frame_scores` (positive-only), a frame close to a
    negative scores lower even when it is also close to a positive.

    **Scored in row blocks, so nothing (M, L) is ever formed.** Only a per-row reduction is
    needed — the top ``k`` similarities — so materializing the whole similarity matrix costs
    O(M·L) memory for a quantity that is O(M·k): 2.2 GB at 107,700 rows against 5,130 labeled
    frames, which is where the eval path was. The same move as
    ``analysis/segmentation.py::_nearest_by_dot`` and ``tw_finch._nearest_neighbors``, and it
    is not only a memory fix — blocked is *faster* (see ``_SOFT_KNN_BLOCK``). The database is
    normalized per block too, so the (M, D) copy is never held either.

    ``block`` is a parameter rather than a constant only so a test can drive a block boundary
    through the middle of the data; the result does not depend on it.

    Args
    database: (M, D) the recording's pose trajectory
    pos_frames: (P, D) labeled positive frames (the held-out exemplar frames)
    neg_frames: (N, D) labeled background/negative frames (may be empty)
    k: neighbors voting per row (>= 1)
    tau: softmax temperature (> 0; lower = sharper weighting of the closest neighbors)
    block: database rows scored per pass
    progress_cb: called with the fraction of rows scored so far, once per block. This loop
        is the finest thing a long search has to report against — one subject is a single
        call from every caller above, so without a hook here an ordinary run shows no
        movement until it finishes. Every block does the same vectorized work, so the
        fraction is measured rather than nominal.

    Returns
    score: (M,) P(positive) in [0, 1], **higher is better**
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    pos = np.asarray(pos_frames, dtype=np.float32)
    if len(pos) == 0:
        raise ValueError("pos_frames is empty")
    neg = np.asarray(neg_frames, dtype=np.float32)
    labeled = np.vstack([pos, neg]) if len(neg) else pos
    labeled_norm = _l2_normalize(labeled)
    n_pos = len(pos)
    db = np.asarray(database, dtype=np.float32)
    kk = min(int(k), labeled_norm.shape[0])
    out = np.empty(len(db), dtype=np.float32)
    for start in range(0, len(db), block):
        stop = min(start + block, len(db))
        sims = np.clip(
            _l2_normalize(db[start:stop]) @ labeled_norm.T, -1.0, 1.0
        )  # (block, L) cosine
        # top-kk neighbors per row (unordered — softmax is permutation-invariant).
        idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]  # (block, kk)
        topk = np.take_along_axis(sims, idx, axis=1)  # (block, kk)
        z = topk / tau
        z = z - z.max(axis=1, keepdims=True)  # stabilize the exponent
        w = np.exp(z)
        pos_w = np.where(idx < n_pos, w, 0.0).sum(
            axis=1
        )  # weight on the positive class
        out[start:stop] = (pos_w / w.sum(axis=1)).astype(np.float32)
        if progress_cb is not None:
            progress_cb(stop / len(db))
    return out


def query_multi(
    database: np.ndarray,
    queries: Sequence[np.ndarray],
    *,
    fusion: str = "mean",
    band: int,
    threshold: float,
    min_separation: int,
    excludes: Sequence[tuple[int, int]] = (),
    overlap: int | None = None,
    max_results: int | None = None,
    length_bounds: Sequence[tuple[int, int]] | None = None,
    method: str = "dtw",
    method_params: dict | None = None,
) -> list[Match]:
    """Database stretches within ``threshold`` of several fused example queries
    (multi-exemplar retrieval), kept non-overlapping.

    Each example is matched independently against the once-normalized database, then the
    per-position normalized costs are fused: ``"mean"`` requires a position to be
    reachable by *all* examples (consensus — it must resemble every example;
    precision-oriented), ``"min"`` by *any* (recall — it need only resemble one). The
    reported span at each position is from the example that fits it best.

    Args
    database: (M, D) the recording's pose trajectory
    queries: list of (L_e, D) example subsequences (may differ in length)
    fusion: ``"mean"`` (consensus) or ``"min"`` (any)
    band: warp / duration tolerance in rows (0 = rigid fixed-length match)
    threshold: keep matches whose fused normalized cosine distance is <= this
    min_separation: minimum row gap between returned matches and from excluded spans
        (ignored when ``overlap`` is set)
    excludes: inclusive spans to drop (e.g. each example's own location in this database,
        or already-labeled spans in the annotation loop)
    overlap: if set (rows), keep matches that overlap a taken match or an ``excludes`` span
        by at most this many rows instead of enforcing ``min_separation`` non-overlap
    max_results: optional safety cap on the number of matches (logs if it truncates)
    length_bounds: optional per-example inclusive ``(min_rows, max_rows)`` span lengths. A
        position whose matched span falls outside its example's range is dropped **before**
        the spacing rule runs, which is the whole point of passing it here rather than
        filtering the returned matches: suppression is greedy best-first, so an out-of-range
        window that scores well would otherwise suppress the in-range window overlapping it
        and then be filtered away, taking a real match with it. That loss is not uniform — a
        short window can hug a low-cost segment and outscore the full-extent one — so it
        biases against exactly the long matches a widened ``band`` exists to find.

    Returns
    matches: Match(start_row, end_row, score), best (lowest) first
    """
    fused, fused_start, best = fused_curve(
        database,
        queries,
        fusion=fusion,
        band=band,
        length_bounds=length_bounds,
        method=method,
        method_params=method_params,
    )
    return extract_matches(
        fused,
        fused_start,
        threshold=threshold,
        min_separation=min_separation,
        excludes=excludes,
        overlap=overlap,
        example_of=best,
        max_results=max_results,
    )


def fused_curve(
    database: np.ndarray,
    queries: Sequence[np.ndarray],
    *,
    fusion: str = "mean",
    band: int,
    length_bounds: Sequence[tuple[int, int]] | None = None,
    method: str = "dtw",
    method_params: dict | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fused per-row cost curve, **before** any extraction.

    ``query_multi`` is this plus one ``extract_matches``. They are separate because a chunked
    search must concatenate *curves* and extract **once** over the whole thing: extraction is
    greedy in ascending cost over the entire recording, so a per-chunk extraction makes that
    order per-chunk and no seam repair restores it. A candidate suppressed inside its own chunk
    is simply gone, and the match it would have freed never reappears (§20, Parallelism).

    Returns
    ``progress_cb`` is called with the fraction of examples scored so far. Each example is its
    own full pass over the database, so this is where a long search's only movement comes from
    — the caller has nothing finer to report against.

    Returns
    fused: (M,) the fused normalized cost per end row (``inf`` where unreachable)
    fused_start: (M,) the winning window's inclusive start
    example_of: (M,) which example fits each row best
    """
    if not queries:
        raise ValueError("queries is empty")
    if fusion not in ("mean", "min"):
        raise ValueError(f"fusion must be 'mean' or 'min', got {fusion!r}")
    for q in queries:
        _validate_shapes(database, q, band)
    curve_fn, mparams = _resolve_curve_fn(method, method_params)
    database_norm = _l2_normalize(np.asarray(database, dtype=np.float32))
    if length_bounds is not None and len(length_bounds) != len(queries):
        raise ValueError(
            f"length_bounds has {len(length_bounds)} entries for {len(queries)} "
            "queries; it is one (min_rows, max_rows) per example"
        )
    # The bounds ride in each example's own params: a matcher that can honour them narrows
    # its search (DTW constrains the DP), and one that cannot ignores the key and is held to
    # the range by the mask below instead. Either way nothing out of range is ever returned.
    per_example = length_bounds if length_bounds is not None else [None] * len(queries)
    # Weighted by rows, not counted. An example's pass costs one DP row per query row, so a
    # long clip and a short one are not half the work each, though a count reports
    # them as 50/50. `total_rows` is never 0: `queries` is non-empty
    # and an empty exemplar cannot reach here (`_validate_shapes` rejects it).
    rows = [max(len(q), 1) for q in queries]
    total_rows = sum(rows)
    curves = []
    for done, (q, bounds) in enumerate(zip(queries, per_example)):
        before = sum(rows[:done])
        curves.append(
            curve_fn(
                database_norm,
                _l2_normalize(np.asarray(q, dtype=np.float32)),
                band,
                mparams if bounds is None else {**mparams, "length_bounds": bounds},
                (
                    None
                    if progress_cb is None
                    else (
                        lambda inner, _b=before, _r=rows[done]: progress_cb(
                            (_b + _r * inner) / total_rows
                        )
                    )
                ),
            )
        )
        if progress_cb is not None:
            progress_cb((before + rows[done]) / total_rows)
    norms = np.stack([c[0] for c in curves])  # (E, M)
    starts = np.stack([c[1] for c in curves])  # (E, M)
    if length_bounds is not None:
        norms = _mask_out_of_range(norms, starts, length_bounds)
    # mean propagates inf (all examples must reach); min keeps a position any reaches.
    fused = norms.mean(axis=0) if fusion == "mean" else norms.min(axis=0)
    best = np.argmin(norms, axis=0)  # the example that fits each position best
    fused_start = starts[best, np.arange(norms.shape[1])]
    return fused, fused_start, best


def query_subsequence(
    database: np.ndarray,
    query: np.ndarray,
    *,
    band: int,
    threshold: float,
    min_separation: int,
    exclude: tuple[int, int] | None = None,
    max_results: int | None = None,
    method: str = "dtw",
    method_params: dict | None = None,
) -> list[Match]:
    """Database stretches within ``threshold`` of a single ``query`` by the selected
    matcher, ranked by length-normalized cosine cost and kept non-overlapping (the
    single-example case of :func:`query_multi`)."""
    return query_multi(
        database,
        [query],
        fusion="mean",
        band=band,
        threshold=threshold,
        min_separation=min_separation,
        excludes=[exclude] if exclude is not None else [],
        max_results=max_results,
        method=method,
        method_params=method_params,
    )
