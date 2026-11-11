"""Numba-compiled Viterbi for the §22 frame-codebook method.

The decode is a sequential max-plus dynamic program over every frame — not a matmul — and it is
``O(rows x k^2)``: linear in the study, quadratic in the state count. It runs **once per fit**, at
the codebook's finest rung; a granularity change maps that labeling upward and never re-decodes
(``codebook_viterbi``'s docstring). This said the decode ran on every granularity change and so
"has to be quick", benchmarked only at small state counts — while a pooled multi-recording
fit at a large one spent hours in it (#483).

**Measured, because the obvious rationale is wrong.** The pure-Python version is not slow for the
usual reason: its inner step is a vectorized numpy op over a ``(k, k)`` array, so this is not
interpreter overhead being removed. What compiling buys is the per-frame allocation and the
temporaries, and the win is real but modest.

**At the state counts a real fit produces** the square dominates: throughput falls
about fourfold per doubling of k (#480).
``codewords`` sets k, which is why its help text says lowering it is the first lever. Nothing here
tiles the ``(k, k)`` access or parallelizes across blocks; both are untried.

Same arrangement as ``_dtw_numba``: numba is imported at this module's top level, and the module is
imported **lazily** by ``decode.viterbi``, so ``import detail`` still pulls no numba.
``decode.viterbi_reference`` is the definition this must match bit for bit, pinned by a test — a
fast decode that quietly decodes differently is worse than a slow one.

``fastmath`` is left **off**, as it is there. The recursion adds log-probabilities that can be very
negative, and telling the compiler to assume no infinities is exactly the sort of latitude that
turns an unreachable state into a reachable one.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def viterbi_blocks(
    emission: np.ndarray, bounds: np.ndarray, log_transition: np.ndarray
) -> np.ndarray:
    """Best state path per continuity block — the compiled form of ``decode.viterbi_reference``.

    ``emission`` is ``(rows, k)`` log-likelihoods, ``bounds`` is ``(n_blocks, 2)`` of half-open
    ``[lo, hi)`` row ranges, and ``log_transition`` is ``(k, k)``. Rows outside every block keep
    label 0; the caller's blocks cover everything it cares about.

    Ties go to the **lowest** state index, which is what ``np.argmax`` does — the tie-break is part
    of the contract, not an accident, because the reference is tested for equality rather than for
    agreement up to ties.
    """
    n_rows = emission.shape[0]
    n_states = emission.shape[1]
    out = np.zeros(n_rows, dtype=np.int64)
    delta = np.empty(n_states, dtype=np.float64)
    nxt = np.empty(n_states, dtype=np.float64)

    for b in range(bounds.shape[0]):
        lo = bounds[b, 0]
        hi = bounds[b, 1]
        if hi <= lo:
            continue
        length = hi - lo
        back = np.zeros((length, n_states), dtype=np.int32)
        for s in range(n_states):
            delta[s] = emission[lo, s]

        for t in range(1, length):
            for s in range(n_states):
                best_prev = 0
                best_score = delta[0] + log_transition[0, s]
                for p in range(1, n_states):
                    score = delta[p] + log_transition[p, s]
                    if score > best_score:
                        best_score = score
                        best_prev = p
                back[t, s] = best_prev
                nxt[s] = best_score + emission[lo + t, s]
            for s in range(n_states):
                delta[s] = nxt[s]

        best_last = 0
        for s in range(1, n_states):
            if delta[s] > delta[best_last]:
                best_last = s
        out[hi - 1] = best_last
        for t in range(length - 1, 0, -1):
            best_last = back[t, best_last]
            out[lo + t - 1] = best_last
    return out
