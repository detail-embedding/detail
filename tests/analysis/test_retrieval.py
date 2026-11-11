"""Tests for the banded subsequence-DTW matcher (detail/analysis/retrieval.py).

The drift-indexed ``subsequence_dtw`` is pinned against ``_brute_optimum`` — an
independent per-start dynamic program that exhaustively enumerates the free start
column. It is the true banded-subsequence-DTW optimum, so ``end_cost`` must match it at
every database column (the earlier prototype's per-cell DP is a greedy approximation and does *not*).
"""

import logging

import numpy as np
import pytest

from detail.analysis import retrieval


def _brute_optimum(cost, band, j_end, drift_bounds=None):
    """True min-cost full-query alignment ending at ``j_end`` — min over every start
    column of a per-start banded DP (down/diag/left).

    ``drift_bounds`` restricts the matched span the way the DP's own argument does: a path
    from ``s`` to ``j_end`` spans ``j_end - s + 1`` rows, so a drift range is a range of
    start columns and the enumeration simply narrows.

    Row 0 is relaxed rightwards before the rest, because a path may **hold** query row 0
    across several database columns. This reference omitted that and so shared the exact
    blind spot of the DP it exists to check — the two agreed with each other and both were
    wrong. Unconstrained the omission is invisible for small shapes, which is why it survived.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_query, n_db = cost.shape
    starts = range(0, j_end + 1)
    if drift_bounds is not None:
        lo, hi = drift_bounds
        first = max(0, j_end - (n_query - 1) - hi)
        last = min(j_end, j_end - (n_query - 1) - lo)
        starts = range(first, last + 1)
    best = np.inf
    for s in starts:
        dp = np.full((n_query, n_db), np.inf)
        dp[0][s] = cost[0][s]
        for col in range(s + 1, n_db):  # query row 0 held: (0,s) ... (0,col)
            if col - s > band:
                break
            if dp[0][col - 1] < np.inf:
                dp[0][col] = dp[0][col - 1] + cost[0][col]
        for i in range(1, n_query):
            for col in range(n_db):
                if abs((col - s) - i) > band:
                    continue
                c = np.inf
                if dp[i - 1][col] < c:
                    c = dp[i - 1][col]
                if col - 1 >= 0 and dp[i - 1][col - 1] < c:
                    c = dp[i - 1][col - 1]
                if col - 1 >= 0 and dp[i][col - 1] < c:
                    c = dp[i][col - 1]
                if c < np.inf:
                    dp[i][col] = c + cost[i][col]
        best = min(best, dp[n_query - 1][j_end])
    return best


@pytest.mark.parametrize("band", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("shape", [(1, 8), (3, 12), (5, 20), (8, 25), (12, 30)])
def test_dtw_matches_brute_force_optimum(band, shape):
    rng = np.random.default_rng(band * 1000 + shape[0])
    cost = rng.random(shape)
    end_cost, start_col, path_len = retrieval.subsequence_dtw(cost, band)
    for j in range(shape[1]):
        brute = _brute_optimum(cost, band, j)
        if np.isinf(brute):
            assert not np.isfinite(end_cost[j])
        else:
            np.testing.assert_allclose(end_cost[j], brute, rtol=1e-9, atol=1e-9)
    # start/length must be self-consistent where a match exists.
    reachable = np.isfinite(end_cost)
    assert np.all(start_col[reachable] >= 0)
    assert np.all(start_col[reachable] <= np.arange(shape[1])[reachable])
    assert np.all(path_len[reachable] >= shape[0])  # at least one cell per query row


def test_band_zero_is_rigid_fixed_length_window():
    rng = np.random.default_rng(1)
    n_query, n_db = 4, 12
    cost = rng.random((n_query, n_db))
    end_cost, start_col, path_len = retrieval.subsequence_dtw(cost, band=0)
    # band=0 permits only the pure diagonal: a length-L window ending at j starts at
    # j-L+1, so end_cost[j] = sum_i cost[i, j-L+1+i] and path_len == L.
    for j in range(n_db):
        if j < n_query - 1:
            assert not np.isfinite(end_cost[j])
            continue
        s = j - n_query + 1
        expected = sum(cost[i, s + i] for i in range(n_query))
        assert path_len[j] == n_query
        assert start_col[j] == s
        np.testing.assert_allclose(end_cost[j], expected)


def test_match_curve_empty_query_raises():
    with pytest.raises(ValueError, match="query is empty"):
        retrieval.match_curve(np.zeros((5, 3)), np.zeros((0, 3)), band=1)


def test_match_curve_query_longer_than_db_raises():
    with pytest.raises(ValueError, match="exceeds database length"):
        retrieval.match_curve(np.zeros((3, 4)), np.zeros((5, 4)), band=1)


def test_match_curve_negative_band_raises():
    with pytest.raises(ValueError, match="band must be >= 0"):
        retrieval.match_curve(np.zeros((5, 3)), np.zeros((2, 3)), band=-1)


def test_single_row_query_is_per_frame_distance():
    db = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    query = np.array([[1.0, 0.0]])
    norm, start = retrieval.match_curve(db, query, band=0)
    # cosine distance to [1,0]: rows 0 and 2 are identical (0), row 1 orthogonal (1).
    np.testing.assert_allclose(norm, [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_array_equal(start, [0, 1, 2])


def test_cosine_cost_is_float32():
    cost = retrieval.cosine_cost(np.ones((2, 4)), np.ones((3, 4)))
    assert cost.dtype == np.float32
    np.testing.assert_allclose(cost, 0.0, atol=1e-6)  # identical direction → 0


def _planted_database(rng, pattern, at, total):
    """A random database of unit rows with ``pattern`` copied in starting at ``at``."""
    d = pattern.shape[1]
    db = rng.standard_normal((total, d)).astype(np.float32)
    db[at : at + len(pattern)] = pattern
    return db


def test_query_finds_planted_repeat():
    rng = np.random.default_rng(3)
    pattern = rng.standard_normal((6, 8)).astype(np.float32)
    db = _planted_database(rng, pattern, at=40, total=100)
    matches = retrieval.query_subsequence(
        db, pattern, band=2, threshold=0.05, min_separation=6
    )
    assert matches, "planted pattern should be found"
    best = min(matches, key=lambda m: m.score)
    assert best.start_row == 40
    assert best.score < 1e-4


def test_exclude_span_drops_self_match():
    rng = np.random.default_rng(4)
    pattern = rng.standard_normal((5, 8)).astype(np.float32)
    db = _planted_database(rng, pattern, at=10, total=60)
    matches = retrieval.query_subsequence(
        db, pattern, band=1, threshold=0.05, min_separation=5, exclude=(10, 14)
    )
    assert all(not (m.start_row <= 14 and m.end_row >= 10) for m in matches)


def test_threshold_filters_and_orders():
    rng = np.random.default_rng(5)
    pattern = rng.standard_normal((5, 8)).astype(np.float32)
    db = _planted_database(rng, pattern, at=20, total=80)
    tight = retrieval.query_subsequence(
        db, pattern, band=1, threshold=0.01, min_separation=5
    )
    loose = retrieval.query_subsequence(
        db, pattern, band=1, threshold=1.5, min_separation=5
    )
    assert len(loose) >= len(tight)
    scores = [m.score for m in loose]
    assert scores == sorted(scores)  # best (lowest) first
    assert all(m.score <= 1.5 for m in loose)


def test_min_separation_suppresses_adjacent_hits():
    rng = np.random.default_rng(6)
    pattern = rng.standard_normal((4, 8)).astype(np.float32)
    db = _planted_database(rng, pattern, at=30, total=80)
    matches = retrieval.query_subsequence(
        db, pattern, band=1, threshold=2.0, min_separation=10
    )
    starts = sorted(m.start_row for m in matches)
    assert all(b - a > 10 for a, b in zip(starts, starts[1:]))


def test_fusion_mean_requires_all_examples_min_requires_any():
    # Two examples that each match a different, far-apart region. mean (consensus)
    # finds neither well; min (any) finds both.
    rng = np.random.default_rng(7)
    pat_a = rng.standard_normal((5, 8)).astype(np.float32)
    pat_b = rng.standard_normal((5, 8)).astype(np.float32)
    db = rng.standard_normal((100, 8)).astype(np.float32)
    db[20:25] = pat_a
    db[70:75] = pat_b
    min_matches = retrieval.query_multi(
        db, [pat_a, pat_b], fusion="min", band=1, threshold=0.05, min_separation=5
    )
    starts = {m.start_row for m in min_matches}
    assert 20 in starts and 70 in starts
    mean_matches = retrieval.query_multi(
        db, [pat_a, pat_b], fusion="mean", band=1, threshold=0.05, min_separation=5
    )
    # consensus: neither region resembles BOTH examples, so nothing passes a tight bar.
    assert mean_matches == []


def test_bad_fusion_raises():
    with pytest.raises(ValueError, match="fusion must be"):
        retrieval.query_multi(
            np.zeros((5, 3)),
            [np.zeros((2, 3))],
            fusion="median",
            band=1,
            threshold=0.5,
            min_separation=1,
        )


def test_empty_queries_raises():
    with pytest.raises(ValueError, match="queries is empty"):
        retrieval.query_multi(
            np.zeros((5, 3)), [], band=1, threshold=0.5, min_separation=1
        )


def test_max_results_caps_and_logs(caplog):
    rng = np.random.default_rng(8)
    pattern = rng.standard_normal((3, 8)).astype(np.float32)
    db = _planted_database(rng, pattern, at=10, total=120)
    with caplog.at_level(logging.WARNING, logger="detail.analysis.retrieval"):
        matches = retrieval.query_subsequence(
            db, pattern, band=1, threshold=2.0, min_separation=1, max_results=3
        )
    assert len(matches) == 3
    assert any("max_results" in r.message for r in caplog.records)


# --- dtw_distance (fixed-endpoint whole-to-whole DTW, the clustering distance) -----


def test_dtw_distance_all_ones_cost_takes_the_diagonal():
    pytest.importorskip("numba")  # dtw_distance compiles its DP via _dtw_numba
    # 3x3 all-ones cost: the cheapest full alignment is the diagonal (3 cells of cost 1),
    # normalized by the realized path length (3) -> 1.0, the mean per-cell cost. A
    # deterministic pin on the recurrence. (Was 0.5 under the old /(n + m) normalization.)
    d = retrieval.dtw_distance(np.ones((3, 3)), band=2)
    assert d == pytest.approx(1.0)


def test_dtw_distance_is_the_mean_per_frame_cost_regardless_of_warping():
    """The normalization property the realized-path-length divisor exists to give.

    A path visits between max(n, m) and n + m - 1 cells. Dividing by the constant bound
    (n + m) therefore over-divided a near-diagonal alignment by up to 2x relative to a
    maximally-warped one — systematically flattering well-aligned bouts and putting
    clustering distances on a different scale from retrieval's match_curve. Under the
    realized-path divisor a uniform cost surface reads as that cost either way.
    """
    pytest.importorskip("numba")
    # Equal lengths, band 0: the path is forced down the diagonal (3 cells).
    rigid = retrieval.dtw_distance(np.full((3, 3), 0.25), band=0)
    # Unequal lengths force warping: the path must take 6 cells to span 2 -> 5.
    warped = retrieval.dtw_distance(np.full((2, 5), 0.25), band=4)
    assert rigid == pytest.approx(0.25)
    assert warped == pytest.approx(0.25)


def test_dtw_distance_matches_retrieval_match_curve_scale():
    """Clustering (fixed-endpoint) and retrieval (subsequence) distances are the same unit.

    Both divide by the realized path length, so aligning a sequence to itself scores the
    same near-zero in either engine. Before the fix the two differed by the path-length
    ratio, so a clustering threshold could not be reasoned about from a retrieval score.
    """
    pytest.importorskip("numba")
    rng = np.random.default_rng(11)
    seq = rng.standard_normal((5, 8)).astype(np.float32)
    fixed = retrieval.dtw_distance(retrieval.cosine_cost(seq, seq), band=2)
    curve, _start = retrieval.match_curve(seq, seq, band=2)
    assert fixed == pytest.approx(float(curve.min()), abs=1e-6)


def test_dtw_distance_identical_sequences_is_zero():
    pytest.importorskip("numba")
    rng = np.random.default_rng(3)
    a = rng.standard_normal((4, 8)).astype(np.float32)
    # cosine_cost(a, a) is 0 down the diagonal (float32 rounding aside), so the
    # whole-to-whole distance is 0.
    assert retrieval.dtw_distance(retrieval.cosine_cost(a, a), band=2) == pytest.approx(
        0.0, abs=1e-6
    )


def test_dtw_distance_is_symmetric_under_transpose():
    pytest.importorskip("numba")
    rng = np.random.default_rng(4)
    cost = rng.random((4, 6))
    assert retrieval.dtw_distance(cost, band=3) == pytest.approx(
        retrieval.dtw_distance(cost.T, band=3)
    )


def test_dtw_distance_unreachable_endpoint_is_inf():
    pytest.importorskip("numba")
    # band 1 cannot bridge a length gap of |2 - 5| = 3, so the endpoint is unreachable.
    # (Confirms the inf sentinel survives compilation with fastmath off.)
    assert retrieval.dtw_distance(np.zeros((2, 5)), band=1) == np.inf


def test_soft_knn_scores_calibrate_against_negatives():
    """Each row's score is a softmax(cos/tau) vote over its k nearest labeled frames. A
    row on the positive prototype scores ~1; a row on a negative scores ~0; a row equally
    near both lands between. Higher = more example-like."""
    dim = 6
    pos = np.tile(np.eye(dim)[0], (3, 1)).astype("float32")  # positive prototype
    neg = np.tile(np.eye(dim)[1], (3, 1)).astype("float32")  # background prototype
    mixed = (np.eye(dim)[0] + np.eye(dim)[1]).astype("float32")  # halfway
    db = np.stack([np.eye(dim)[0], np.eye(dim)[1], mixed]).astype("float32")
    score = retrieval.soft_knn_frame_scores(db, pos, neg, k=6, tau=0.07)
    assert score.shape == (3,)
    assert score[0] > 0.9  # on the positive prototype
    assert score[1] < 0.1  # on the negative prototype
    assert 0.2 < score[2] < 0.8  # equally near both
    assert score.dtype == np.float32


def test_soft_knn_with_no_negatives_is_all_positive():
    dim = 4
    pos = np.eye(dim)[:2].astype("float32")
    empty = np.empty((0, dim), dtype="float32")
    score = retrieval.soft_knn_frame_scores(
        np.eye(dim).astype("float32"), pos, empty, k=2, tau=0.1
    )
    assert np.allclose(score, 1.0)  # every neighbor is positive


def test_soft_knn_rejects_bad_params():
    dim = 4
    pos = np.eye(dim)[:1].astype("float32")
    empty = np.empty((0, dim), dtype="float32")
    with pytest.raises(ValueError, match="k must be"):
        retrieval.soft_knn_frame_scores(np.eye(dim), pos, empty, k=0, tau=0.1)
    with pytest.raises(ValueError, match="tau must be"):
        retrieval.soft_knn_frame_scores(np.eye(dim), pos, empty, k=1, tau=0.0)
    with pytest.raises(ValueError, match="pos_frames is empty"):
        retrieval.soft_knn_frame_scores(np.eye(dim), empty, empty, k=1, tau=0.1)
    with pytest.raises(ValueError, match="block must be"):
        retrieval.soft_knn_frame_scores(np.eye(dim), pos, empty, k=1, tau=0.1, block=0)


# --- soft-kNN is scored in row blocks -------------------------------------------------
#
# The class: **a per-row reduction computed through a whole materialized matrix.** Only the
# top `k` similarities per row are needed — O(M·k) — but forming `(M, L)` to get them costs
# O(M·L), and `argpartition` allocates a second one at int64. Measured on the eval path: 2.2 GB
# at 107,700 rows against 5,130 labeled frames, on a quantity that needs kilobytes.
#
# Nothing is *wrong* at the line where it is written, which is why the fix needs a guard: the
# scores are correct either way, so no behavioural test can see it. `segmentation._nearest_by_dot`
# and `tw_finch._nearest_neighbors` make the same move for the same reason.


def _knn_inputs(n_rows, dim=16, n_labeled=500, seed=0):
    rng = np.random.default_rng(seed)
    db = rng.normal(size=(n_rows, dim)).astype("float32")
    pos = rng.normal(size=(10, dim)).astype("float32")
    neg = rng.normal(size=(n_labeled - 10, dim)).astype("float32")
    return db, pos, neg


@pytest.mark.parametrize("block", [1, 2, 7, 999, 1000, 1001, 5000])
def test_soft_knn_scores_do_not_depend_on_the_block_size(block):
    """A block boundary falling mid-trajectory must not change any row's score.

    Includes blocks that divide the row count exactly, that leave a remainder of one, and
    that exceed it — the three ways a tiling gets written wrong.

    **To float32 precision, not bit-for-bit**, and that is a property of BLAS rather than of
    this loop: a ``(1, D) @ (D, L)`` product and a ``(1000, D) @ (D, L)`` one take different
    kernels with different accumulation orders, so the cosines differ in the last digit.
    Measured here at 3.3e-7 absolute / 2.2e-6 relative, against a float32 epsilon of 1.2e-7 —
    and exactly zero at most block sizes. The tolerance is far below anything that could move
    a candidate and far above the wobble, so a genuine tiling bug (a skipped or doubled row
    reads uninitialized memory) still fails it. Callers that need reproducibility get it from
    the block size being fixed, not from the arithmetic being associative.
    """
    db, pos, neg = _knn_inputs(1000)
    reference = retrieval.soft_knn_frame_scores(
        db, pos, neg, k=20, tau=0.07, block=1000
    )
    scored = retrieval.soft_knn_frame_scores(db, pos, neg, k=20, tau=0.07, block=block)
    np.testing.assert_allclose(scored, reference, rtol=1e-5, atol=1e-6)


def _peak_bytes(n_rows, block):
    """Peak numpy allocation while scoring `n_rows` rows at this block size."""
    import tracemalloc

    db, pos, neg = _knn_inputs(n_rows)
    tracemalloc.start()
    try:
        retrieval.soft_knn_frame_scores(db, pos, neg, k=20, tau=0.07, block=block)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_soft_knn_peak_memory_does_not_scale_with_trajectory_length():
    """Four times the rows at a fixed block must not cost four times the memory.

    Asserted as a *ratio* rather than against a byte threshold: the invariant is that peak
    memory is O(block x L) and independent of M, and a threshold would encode this machine's
    numbers instead of that property.
    """
    small = _peak_bytes(4_000, block=1024)
    large = _peak_bytes(16_000, block=1024)
    assert large < 1.5 * small, (
        f"peak grew {large / small:.1f}x for 4x the rows ({small} -> {large} bytes), so "
        "something (M, L) is being formed rather than reduced per block"
    )


def test_the_memory_guard_would_catch_an_unblocked_implementation():
    """Guard the guard: with the loop in place the ratio above is ~1 forever.

    `block >= len(database)` is one pass over everything, which *is* the unblocked form this
    replaced — so the negative case runs the real code path rather than a stand-in.
    """
    small = _peak_bytes(4_000, block=4_000)
    large = _peak_bytes(16_000, block=16_000)
    assert large > 3.0 * small, (
        "scoring in one pass should cost 4x the memory for 4x the rows; it did not "
        f"({small} -> {large} bytes), so the guard above proves nothing"
    )


# --- bounded-duration search ----------------------------------------------------------
#
# `drift_bounds` constrains which spans the DP may *choose*. The alternative — letting it
# choose freely and filtering the result — is not equivalent: the DP keeps one winner per end
# column, so a column whose unconstrained winner is out of range is discarded even when an
# in-range path through the same column scored well. That is a silent loss of recall in
# exactly the range a caller widened to gain some.


def _cost_surface(n_query=4, n_db=40, seed=3):
    rng = np.random.default_rng(seed)
    return rng.random((n_query, n_db))


def test_drift_bounds_pin_the_span_length():
    cost = _cost_surface()
    n_query = cost.shape[0]
    end_cost, start_col, _ = retrieval.subsequence_dtw(
        cost, band=6, drift_bounds=(0, 0)
    )
    reachable = np.isfinite(end_cost)
    spans = np.arange(cost.shape[1])[reachable] - start_col[reachable] + 1
    assert spans.size and np.all(
        spans == n_query
    )  # drift 0 == exactly the query length


def test_drift_bounds_keep_columns_the_unconstrained_search_would_lose():
    """The recall claim, and the reason this is not a post-filter.

    Constrain to a narrow band of drifts and compare against filtering the unconstrained
    result to the same span range: the constrained search answers at columns the filter
    throws away.
    """
    cost = _cost_surface()
    n_query, n_db = cost.shape
    allowed = (-1, 1)  # spans of n_query-1 .. n_query+1

    free_cost, free_start, _ = retrieval.subsequence_dtw(cost, band=6)
    bound_cost, bound_start, _ = retrieval.subsequence_dtw(
        cost, band=6, drift_bounds=allowed
    )

    columns = np.arange(n_db)
    free_span = columns - free_start + 1
    free_in_range = np.isfinite(free_cost) & (np.abs(free_span - n_query) <= allowed[1])
    bound_reachable = np.isfinite(bound_cost)
    bound_span = columns - bound_start + 1

    # Everything the constrained search returns is in range. Without this the test passes
    # against a search that ignored the bounds entirely, since then it "answers" everywhere
    # the free search does — with out-of-range spans.
    assert np.all(np.abs(bound_span[bound_reachable] - n_query) <= allowed[1])
    # every column the filter would have kept is still answered...
    assert np.all(bound_reachable[free_in_range])
    # ...and some column the filter would have dropped is answered too
    recovered = bound_reachable & ~free_in_range
    assert recovered.any(), "fixture no longer exercises the recovered-column case"


def test_constraining_never_beats_the_free_optimum():
    # A constrained search explores a subset, so its cost can only be equal or worse. If it
    # ever came out lower, the free search was not finding the optimum.
    cost = _cost_surface()
    free_cost, _, _ = retrieval.subsequence_dtw(cost, band=6)
    bound_cost, _, _ = retrieval.subsequence_dtw(cost, band=6, drift_bounds=(-1, 1))
    both = np.isfinite(free_cost) & np.isfinite(bound_cost)
    assert np.all(bound_cost[both] >= free_cost[both] - 1e-12)


def test_no_drift_bounds_is_the_unconstrained_search():
    cost = _cost_surface()
    a = retrieval.subsequence_dtw(cost, band=5)
    b = retrieval.subsequence_dtw(cost, band=5, drift_bounds=(-5, 5))
    for left, right in zip(a, b):
        assert np.array_equal(left, right)


@pytest.mark.parametrize(
    "shape,band,bounds",
    [((3, 14), 3, (1, 3)), ((4, 18), 4, (2, 4)), ((5, 16), 3, (-2, 0))],
)
def test_a_constrained_search_is_the_optimum_of_the_spans_it_allows(
    shape, band, bounds
):
    """The duration-constrained result must be the *best in-range path*, not merely an
    in-range one.

    The DP could not hold query row 0, so under a positive drift bound it was forced to spend
    its left moves on rows 1..L-1 and returned a worse in-range alignment than exists. The
    unconstrained test could not see it: every path the omission lost is reachable another way
    once the drift budget is free, so both sides agreed. Only pinning the *constrained* search
    against an independent enumeration exposes it.
    """
    rng = np.random.default_rng(shape[0] * 7 + band)
    cost = rng.random(shape)
    end_cost, _, _ = retrieval.subsequence_dtw(cost, band, bounds)
    for j in range(shape[1]):
        brute = _brute_optimum(cost, band, j, drift_bounds=bounds)
        if np.isinf(brute):
            assert not np.isfinite(
                end_cost[j]
            ), f"column {j} reachable but should not be"
        else:
            np.testing.assert_allclose(end_cost[j], brute, rtol=1e-9, atol=1e-9)


def test_a_single_row_query_can_still_stretch_to_a_bounded_span():
    """The starkest form: with one exemplar row every path *is* a held row 0.

    Seeding drift 0 alone made the whole curve `inf` for any non-zero drift bound — a query
    the user marked as one frame could never match a span longer than one frame, whatever
    duration range they asked for, and it returned emptiness rather than an error.
    """
    rng = np.random.default_rng(11)
    cost = rng.random((1, 10))
    end_cost, start_col, path_len = retrieval.subsequence_dtw(cost, 5, (1, 5))
    reachable = np.isfinite(end_cost)
    assert reachable.any(), "a one-row query cannot reach any span longer than itself"
    spans = np.arange(10)[reachable] - start_col[reachable] + 1
    assert np.all(spans >= 2) and np.all(spans <= 6)  # L + drift, drift in [1, 5]
    assert np.array_equal(path_len[reachable], spans)
