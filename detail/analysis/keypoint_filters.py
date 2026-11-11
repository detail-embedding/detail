"""Temporal filters and the candidate path over a keypoint track (§26 "Track methods") — pure
numpy, no I/O.

**Despike** — a Hampel filter per keypoint across rows: a point farther from its window's median
position than ``k`` robust deviations is a spike and takes that median. Steady motion moves the
median with it and is left alone; a jump shorter than half the window is removed, and a
left/right swap of a few frames is two such jumps. Absence is never filled — a NaN stays NaN.
Measured on the research recordings, it halves frame-to-frame jumps.

**Smooth** — each point replaced by a local quadratic fitted to its neighbours in time,
Gaussian-weighted in ``t_ns``. It removes the small frame-to-frame jitter the despike leaves (a
still animal's points wobble by about a third of a DINOv3 patch) and keeps turning points a plain
average would flatten. In an internal evaluation it cut still-frame jitter about
twelvefold at 0.133 s and *lowered* per-frame error against two hand-labeled sets.

**The path** (``viterbi_path``, ``two_state_path``) — the dynamic programs ``temporal_decode``
runs: one per point over its stored candidates plus a *missing* state (the Anipose filter's
shape), and one per left/right pair over *normal* / *swapped*, so a swap must hold over frames to
win. The transition scale and the penalties are **policy** (§26 *Open questions* 6).

Ported from the evaluation scripts via #626; the
tests hold each filter equal to the per-row reference it was measured as.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# The deviation never drops below this fraction of body scale, so a still animal's sub-pixel
# jitter is not read as a spike.
DESPIKE_FLOOR = 0.03
_MAD_TO_SD = 1.4826

# The smoothing's weights reach this many window widths (sigmas) either side, and a point needs
# this many weighted neighbours (itself included) for a quadratic; below it, the weighted mean.
SMOOTH_TRUNCATE = 3.0
SMOOTH_MIN_POINTS = 5
SMOOTH_KERNEL = "local_quadratic"
_NS_PER_S = 1_000_000_000


def rows_for_seconds(window_s: float, rows_per_second: float) -> int:
    """A window in seconds as a row count at a cadence, never below one row — the one
    place seconds become rows (``tests/configs/test_unit_conventions.py``)."""
    if rows_per_second <= 0:
        raise ValueError(f"rows_per_second must be positive, got {rows_per_second}")
    return max(1, int(round(float(window_s) * float(rows_per_second))))


def hampel_despike(
    xy: np.ndarray,
    scale: np.ndarray,
    half_window: int = 3,
    k: float = 3.0,
    floor: float = DESPIKE_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace each keypoint's spikes by its local median: ``(despiked xy, replaced)``.

    Args:
        xy: ``(N, K, 2)`` positions over consecutive rows, in pixels (distances must be
            isotropic), NaN where a point is absent.
        scale: ``(N,)`` body scale per row in the same pixels (``sqrt`` of the subject box's
            area) — what ``floor`` is a fraction of.
        half_window: rows either side; a point needs ``half_window + 1`` present neighbours
            (itself included) before it can be called a spike.
        k: robust deviations (1.4826 x the median distance to the window median) a point may
            sit from that median.
        floor: the deviation's lower bound, as a fraction of ``scale``.

    Returns:
        The filtered positions (a copy) and an ``(N, K)`` bool mask of the replaced points.
    """
    xy = np.asarray(xy, dtype=np.float64)
    n, kp, _ = xy.shape
    out = xy.copy()
    replaced = np.zeros((n, kp), dtype=bool)
    if n == 0:
        return out, replaced
    width = 2 * half_window + 1
    pad = np.full((half_window, 2), np.nan)
    limit = k * np.asarray(scale, dtype=np.float64) * floor  # (N,) the floored minimum
    for j in range(kp):
        track = xy[:, j]
        present = np.isfinite(track).all(axis=1)
        windows = sliding_window_view(np.concatenate([pad, track, pad]), width, axis=0)
        windows = np.moveaxis(windows, -1, 1)  # (N, width, 2)
        in_window = np.isfinite(windows).all(axis=2)  # (N, width)
        count = in_window.sum(axis=1)
        eligible = present & (count >= half_window + 1)
        if not eligible.any():
            continue
        # Median only where a decision is possible, so no all-NaN window reaches nanmedian.
        w = windows[eligible]
        med = np.nanmedian(w, axis=1)  # (M, 2)
        spread = np.nanmedian(np.linalg.norm(w - med[:, None, :], axis=2), axis=1)
        threshold = np.maximum(k * _MAD_TO_SD * spread, limit[eligible])
        distance = np.linalg.norm(track[eligible] - med, axis=1)
        spike = distance > threshold
        rows = np.flatnonzero(eligible)[spike]
        out[rows, j] = med[spike]
        replaced[rows, j] = True
    return out, replaced


# Rows smoothed per block: bounds the moment arrays to ~50 MB for a 27-point skeleton however
# long the part is.
_SMOOTH_BLOCK = 20_000


def smooth_local_quadratic(
    xy: np.ndarray,
    t_ns: np.ndarray,
    window_s: float,
    truncate: float = SMOOTH_TRUNCATE,
    min_points: int = SMOOTH_MIN_POINTS,
) -> np.ndarray:
    """Replace each point by a Gaussian-weighted local quadratic fit to its neighbours in time.

    Args:
        xy: ``(N, K, 2)`` positions over consecutive rows in time order, in pixels (distances
            must be isotropic), NaN where a point is absent.
        t_ns: ``(N,)`` each row's time. Weights are in time, not rows, so rows on either side of
            a gap in the track, or a run made every n frames, carry the weight their distance in
            time earns.
        window_s: the weights' standard deviation in seconds.
        truncate: how many ``window_s`` either side the weights reach.
        min_points: weighted neighbours (the point itself included) a quadratic needs; with
            fewer, the point takes the weighted mean.

    Returns:
        The smoothed positions (a copy). A point absent at a row stays absent, and absent
        neighbours carry no weight.
    """
    xy = np.asarray(xy, dtype=np.float64)
    out = np.full_like(xy, np.nan)
    n, kp, _ = xy.shape
    if n == 0:
        return out
    t = np.asarray(t_ns, dtype=np.int64)
    present = np.isfinite(xy).all(axis=2)  # (N, K)
    filled = np.where(present[..., None], xy, 0.0)
    reach_s = truncate * float(window_s)
    rows = np.arange(n)
    lo = np.searchsorted(t, t - reach_s * _NS_PER_S, side="left")
    hi = np.searchsorted(t, t + reach_s * _NS_PER_S, side="right")
    back, ahead = int((rows - lo).max()), int((hi - 1 - rows).max())
    pair = np.add.outer(np.arange(3), np.arange(3))  # dt^i * dt^j -> moment i + j
    for start in range(0, n, _SMOOTH_BLOCK):
        target = rows[start : start + _SMOOTH_BLOCK]
        m = len(target)
        moments = np.zeros((m, kp, 5))  # sum of w * dt^p, p = 0..4
        rhs = np.zeros((m, kp, 3, 2))  # sum of w * dt^p * xy, p = 0..2
        count = np.zeros((m, kp), dtype=np.int64)
        for offset in range(-back, ahead + 1):
            j = target + offset
            valid = (j >= 0) & (j < n)
            i, j = np.flatnonzero(valid), j[valid]
            dt = (t[j] - t[target[i]]) / _NS_PER_S
            w = np.exp(-0.5 * (dt / window_s) ** 2) * (np.abs(dt) <= reach_s)
            wk = w[:, None] * present[j]  # (m', K)
            powers = dt[:, None] ** np.arange(5)  # (m', 5)
            moments[i] += wk[..., None] * powers[:, None, :]
            rhs[i] += (wk[..., None] * powers[:, None, :3])[..., None] * filled[j][
                :, :, None, :
            ]
            count[i] += wk > 1e-6
        here = present[target]
        weight = moments[..., 0:1]
        value = np.divide(
            rhs[..., 0, :], weight, out=np.full((m, kp, 2), np.nan), where=weight > 0
        )
        normal = moments[..., pair]  # (m, K, 3, 3)
        fit = here & (count >= min_points) & (np.abs(np.linalg.det(normal)) > 1e-18)
        if fit.any():
            value[fit] = np.linalg.solve(normal[fit], rhs[fit])[:, 0, :]
        out[target] = np.where(here[..., None], value, np.nan)
    return out


# --------------------------------------------------------------------------- #
# The candidate path (temporal_decode)
# --------------------------------------------------------------------------- #
def viterbi_path(
    emission: np.ndarray,
    transition_cost,
    missing_penalty: float,
) -> np.ndarray:
    """The best state per row over ``C`` candidates plus a *missing* state.

    Args:
        emission: ``(N, C)`` the score of taking candidate ``c`` on row ``n``; NaN where the
            candidate does not exist. Missing scores ``-missing_penalty`` on every row.
        transition_cost: ``transition_cost(n) -> (C, C)`` the cost (non-negative) of moving
            from each candidate on row ``n - 1`` to each on row ``n``; a move into or out of
            *missing* costs nothing beyond the penalty.
        missing_penalty: what a row pays to be missing.

    Returns:
        ``(N,)`` int: the candidate chosen per row, ``-1`` where the path is missing.
    """
    emission = np.asarray(emission, dtype=np.float64)
    n, c = emission.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    # state C is "missing"
    present = np.isfinite(emission)
    score = np.where(present, emission, -np.inf)
    total = np.full((n, c + 1), -np.inf)
    back = np.zeros((n, c + 1), dtype=np.int64)
    total[0, :c] = score[0]
    total[0, c] = -missing_penalty
    for i in range(1, n):
        cost = np.asarray(transition_cost(i), dtype=np.float64)  # (C, C) prev -> cur
        prev_cand = total[i - 1, :c]
        # from a candidate: pay the move; from missing: pay nothing more
        via_cand = prev_cand[:, None] - cost  # (C_prev, C_cur)
        via_cand = np.where(np.isfinite(via_cand), via_cand, -np.inf)
        best_prev = np.argmax(via_cand, axis=0)
        best_val = via_cand[best_prev, np.arange(c)]
        from_missing = total[i - 1, c]
        take_missing = from_missing > best_val
        total[i, :c] = np.where(take_missing, from_missing, best_val) + score[i]
        back[i, :c] = np.where(take_missing, c, best_prev)
        # into missing: from the best previous state of any kind
        prev_best = int(np.argmax(total[i - 1]))
        total[i, c] = total[i - 1, prev_best] - missing_penalty
        back[i, c] = prev_best
    path = np.zeros(n, dtype=np.int64)
    state = int(np.argmax(total[-1]))
    for i in range(n - 1, -1, -1):
        path[i] = -1 if state == c else state
        state = int(back[i, state])
    return path


def two_state_path(evidence: np.ndarray, switch_cost: float) -> np.ndarray:
    """``(N,)`` bool — the *swapped* reading per row, from per-row evidence for it (positive
    favours swapped, negative normal, NaN says nothing) and a cost to switch between rows, so
    the answer holds over frames rather than flickering."""
    evidence = np.asarray(evidence, dtype=np.float64)
    n = len(evidence)
    if n == 0:
        return np.zeros(0, dtype=bool)
    e = np.where(np.isfinite(evidence), evidence, 0.0)
    # state 0 = normal (gains -e/2), state 1 = swapped (gains +e/2)
    gain = np.stack([-e / 2, e / 2], axis=1)
    total = gain[0].copy()
    back = np.zeros((n, 2), dtype=np.int64)
    for i in range(1, n):
        stay = total
        switch = total[::-1] - switch_cost
        choose_switch = switch > stay
        back[i] = np.where(choose_switch, [1, 0], [0, 1])
        total = np.where(choose_switch, switch, stay) + gain[i]
    swapped = np.zeros(n, dtype=bool)
    state = int(np.argmax(total))
    for i in range(n - 1, -1, -1):
        swapped[i] = state == 1
        state = int(back[i, state])
    return swapped
