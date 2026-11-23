"""Cosine retrieval primitives over pose embeddings (§16, §20): per-frame soft k-NN scoring
and banded subsequence dynamic time warping (DTW).

Pure compute over arrays (Tenet 2/3 — no zarr, no plotting, no paths). Two things live here:

- **Per-frame scoring.** ``soft_knn_frame_scores`` is the softmax-weighted k-NN vote that
  §20's ``soft_knn`` query method (``query_methods/soft_knn.py``) scores every frame with.
- **Banded subsequence DTW** over a cosine-distance surface (``cosine_cost``): given a query
  subsequence, ``match_curve`` gives the cost of the best match ending at each row of a
  recording's trajectory, tolerating mild speed/duration differences. The full query is
  matched, the start and end positions in the database float, and a Sakoe-Chiba ``band`` caps
  how far the alignment may drift from 1:1 — so the same action over ``N`` vs ``N+M`` frames
  still matches (``M`` up to ``band``). ``band=0`` reduces to a rigid fixed-length cosine match.
  ``dtw_distance`` is the readable whole-sequence DTW that ``_dtw_numba`` reproduces.

**The DP is exact and vectorized.** The prototype this was ported from kept the lowest-cost
path *per cell* and relaxed the left step with a per-column Python loop. But the band is
measured against the path's start column, so two paths reaching a cell from different starts
are not interchangeable — collapsing over start is a greedy approximation (it can report a
cost well above the true optimum, and the result depends on the relaxation order). Here the DP
is indexed by **drift** ``d = (j - start) - i`` (``d ∈ [-band, band]``): each of
down/diag/left changes ``d`` by exactly ``-1``/``0``/``+1``, so a left move always *raises*
``d`` and never depends on the same drift level — the whole DP vectorizes over the database
axis with an outer loop of only ``2*band+1`` drift levels. This is the exact
banded-subsequence-DTW optimum in ``O(L·band·M)``.

The windowed matchers that were built on this DTW (``query_multi``, ``extract_matches`` and
the pooled and keyframe curves) were removed with query eval, their last caller. No stage
calls ``match_curve`` now; ``tests/test_no_orphan_functions.py`` records why it stays.
"""

from collections.abc import Callable

import numpy as np


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


# --- per-frame scoring: score every row against the labeled frames (§20) --------------
# No windowing: a softmax-weighted vote calibrated against sampled background frames.

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
    proposal cutoff: a frame close to a negative scores lower even when it is also close
    to a positive.

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
