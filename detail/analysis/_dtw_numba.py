"""Numba-compiled DTW kernels for comparing bouts (temporal clustering's ``dtw_finch``, §22).

The banded fixed-endpoint DTW distance (``retrieval.dtw_distance``) is a sequential min-plus
dynamic program — not a matmul — run K²/2 times over a recording's bouts, which made it the whole
run's cost as a pure-Python loop. Compiling the loop removes the per-cell interpreter overhead
(``dtw_dp``, ~7×); fusing the cosine surface + DP over every pair into one ``prange`` kernel
(``pairwise_block``) removes the per-pair numpy allocation and parallelizes across cores (~30×).

``numba`` is imported at this module's top level, but the module is imported **lazily** by its
callers (``retrieval`` / ``segmentation`` import it inside their functions), so ``import
detail`` still pulls no numba — the same guarantee the lazy scipy/sklearn imports give. numba
ships in the ``analysis`` extra, which bout clustering already requires (HDBSCAN).

``fastmath`` is left **off**: the DP uses ``np.inf`` to mark unreachable cells, and fastmath tells
the compiler to assume no infinities — unsafe here, and measured slower besides.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def dtw_dp(cost: np.ndarray, band: int) -> float:
    """Path-length-normalized fixed-endpoint banded DTW cost over a ``(n, m)`` cost surface — the
    compiled form of the loop in ``retrieval.dtw_distance`` (bit-identical). ``band`` bounds
    ``|i - j|``; returns ``inf`` if the endpoint is unreachable (``band < |n - m|``).

    The accumulated cost is divided by the **realized path length** (the cells the best path
    visits), not the constant bound ``n + m``, so the result is a true average per-frame cosine
    distance — the same normalization ``retrieval.match_curve`` uses. A path visits between
    ``max(n, m)`` and ``n + m - 1`` cells, so dividing by ``n + m`` over-divided near-diagonal
    pairs by up to 2× relative to maximally-warped ones, which silently rewarded well-aligned
    bouts and put clustering distances on a different scale from retrieval scores.
    """
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    plen = np.zeros((n + 1, m + 1), dtype=np.int64)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            p = acc[i - 1, j]
            q = acc[i, j - 1]
            r = acc[i - 1, j - 1]
            # On equal costs the later candidate wins. Any minimal path is a valid alignment,
            # but tracking plen makes the *choice* observable, so it is pinned here rather than
            # left to argmin ordering.
            if p < q:
                best = p
                bl = plen[i - 1, j]
            else:
                best = q
                bl = plen[i, j - 1]
            if not (best < r):
                best = r
                bl = plen[i - 1, j - 1]
            acc[i, j] = cost[i - 1, j - 1] + best
            plen[i, j] = bl + 1
    end = acc[n, m]
    if not np.isfinite(end):
        return np.inf
    return end / plen[n, m]


@njit(cache=True, parallel=True)
def pairwise_block(
    flat: np.ndarray,
    offs: np.ndarray,
    lens: np.ndarray,
    band: int,
    i0: int,
    i1: int,
    out: np.ndarray,
) -> None:
    """Fill rows ``[i0, i1)`` of the K×K DTW distance matrix ``out`` in parallel.

    ``flat`` is every bout's **already-L2-normalized** rows concatenated; bout ``i`` occupies
    ``flat[offs[i] : offs[i] + lens[i]]``. For each pair ``(i, j>i)`` the cosine cost surface is
    ``1 - Xi @ Xjᵀ`` (unit rows, so this equals ``retrieval.cosine_cost`` up to float noise), the
    effective band is ``max(band, |li - lj|)`` (so unequal-length bouts stay reachable), and the
    fixed-endpoint DP fills ``out[i, j] = out[j, i]``. Blocked by rows so the caller can report
    progress between blocks — a ``prange`` kernel can't call back into Python mid-run.
    """
    k = len(lens)
    for i in prange(i0, i1):
        oi = offs[i]
        li = lens[i]
        xi = flat[oi : oi + li]
        for j in range(i + 1, k):
            oj = offs[j]
            lj = lens[j]
            xj = flat[oj : oj + lj]
            cost = 1.0 - xi @ xj.T
            dl = li - lj if li > lj else lj - li
            eff = band if band > dl else dl
            d = dtw_dp(cost, eff)
            out[i, j] = d
            out[j, i] = d
