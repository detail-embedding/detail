"""The soft-kNN query method (detail/query_methods/soft_knn.py).

Every guard here covers a failure that produces *plausible output* rather than an error, which
is the only kind this method can have — a per-frame score always yields spans of some sort, and
they always look like matches.

- **Chunk equivalence.** A split search must agree with an unsplit one. This method's ownership
  rule is the subtle part: a peak needs `context_rows` on both sides to grow to its true
  extent, so `chunk_overlap_rows` declares **twice** that, and a chunk owns only the interior
  it resolved fully. Declaring the context once instead leaves a band owned by no chunk and
  every match peaking inside it silently disappears — from the middle of long recordings only,
  which is exactly where nobody checks.
- **Monotonicity.** The calibration page is a frequency curve a user drags along. §20 permits a
  method to re-form extents and warns the count may not fall; this one fixes extents before any
  cutoff exists, so it does fall, and a change that broke that would make the curve unreadable
  rather than raise.
- **Progress cadence.** With one unit, one exemplar and one continuous run there is no
  structural loop left to move a bar — whatever still moves it is inside the scorer, which is
  where the hook has to be. §20's cadence rule, and CLAUDE.md's "cut the fixture to one of
  everything the stage loops over".
"""

import numpy as np
import pytest

from detail.configs.query_soft_knn import SoftKnnQueryConfig
from detail.query_methods.soft_knn import (
    SoftKnnQueryMethod,
    SoftKnnState,
    _candidates,
    _smooth,
)

DIM = 6


def _method():
    return SoftKnnQueryMethod()


def _motif(length=20, seed=7):
    return np.random.default_rng(seed).normal(size=(length, DIM)).astype(np.float32)


def _trajectory(n_rows, motif, at, seed=0):
    """Background rows with ``motif`` planted at ``at`` — the one thing to be found."""
    rng = np.random.default_rng(seed)
    out = rng.normal(size=(n_rows, DIM)).astype(np.float32)
    out[at : at + len(motif)] = motif
    return out


def _background(n=64, seed=3):
    return np.random.default_rng(seed).normal(size=(n, DIM)).astype(np.float32)


def _prepared(motif, params=None):
    return _method().prepare([motif], params or {}, sample=_background())


# --------------------------------------------------------------------------- #
# it finds the thing, and says so with a confidence
# --------------------------------------------------------------------------- #
def test_the_planted_motif_is_the_strongest_candidate():
    motif = _motif()
    emb = _trajectory(400, motif, at=150)
    method = _method()
    result = method.search(_prepared(motif), emb, [(0, len(emb))], {})

    assert len(result.peak), "nothing was found at all"
    best = method.instances_at(result, 0.0, {})[0]
    assert best.start_row <= 150 + len(motif) // 2 <= best.end_row
    assert best.confidence > 0.5


def test_no_match_spans_a_run_boundary():
    """A tracking gap is a jump in **time**, not in row index.

    `clip_query.segment_runs` splits on the `t_ns` step, so its runs are row-*contiguous*:
    a real gap looks like `[(0, 150), (150, 400)]`, with nothing between row 149 and row 150
    to notice. That is why candidates are formed per run rather than over the whole array —
    over the array there is no boundary to stop growth, and a match happily spans minutes of
    footage nobody recorded while looking like any other match.

    The fixture is built the way `segment_runs` builds one, which the first version of this
    test was not: it left a row gap the scorer had never filled, so the score there was zero,
    growth stopped on its own, and the test passed against the broken version too.
    """
    motif = _motif()
    emb = _trajectory(400, motif, at=140)  # straddling the boundary below
    result = _method().search(_prepared(motif), emb, [(0, 150), (150, 400)], {})

    crossing = [
        (int(lo), int(hi)) for lo, hi in zip(result.start, result.end) if lo < 150 <= hi
    ]
    assert not crossing, f"these matches span the run boundary at row 150: {crossing}"


def test_rows_in_no_run_are_never_scored_or_returned():
    """The other half: a subject searched over part of its trajectory returns nothing from
    the rest, rather than a zero-confidence match nobody looked for."""
    motif = _motif()
    emb = _trajectory(400, motif, at=40)
    result = _method().search(_prepared(motif), emb, [(0, 150)], {})

    assert np.all(
        result.score[150:] == 0.0
    ), "rows outside every run were given a score"
    assert np.all(result.end < 150), "a match was returned outside the searched runs"


# --------------------------------------------------------------------------- #
# monotonicity: the property the calibration curve rests on
# --------------------------------------------------------------------------- #
def test_raising_the_cutoff_only_ever_removes_instances():
    motif = _motif()
    emb = _trajectory(600, motif, at=200)
    method = _method()
    result = method.search(_prepared(motif), emb, [(0, len(emb))], {})

    previous = None
    for cutoff in np.linspace(0.0, 0.95, 20):
        kept = {
            (c.start_row, c.end_row) for c in method.instances_at(result, cutoff, {})
        }
        if previous is not None:
            assert kept <= previous, (
                f"cutoff {cutoff:.2f} returned an instance that a looser cutoff did not: "
                f"{sorted(kept - previous)}. Extents are fixed before any cutoff exists, so "
                "a stricter cutoff can only ever be a subset."
            )
        previous = kept


# --------------------------------------------------------------------------- #
# chunk equivalence
# --------------------------------------------------------------------------- #
def _chunks(n_rows, overlap, target):
    from detail.pipeline.query import _plan_chunks

    return _plan_chunks(n_rows, overlap, target)


@pytest.mark.parametrize("target", [200, 260, 333])
def test_a_chunked_search_agrees_with_an_unchunked_one(target):
    """Driven through the module's own `_plan_chunks`, so the tiling under test is the one a
    long recording actually gets rather than one the test invented."""
    motif = _motif()
    emb = _trajectory(900, motif, at=400)
    method = _method()
    state = _prepared(motif)

    whole = method.search(state, emb, [(0, len(emb))], {})
    overlap = method.chunk_overlap_rows(state, {})
    plan = _chunks(len(emb), overlap, target)
    assert len(plan) > 1, f"target {target} did not actually split the trajectory"
    parts = [
        method.search(state, emb[lo:hi], [(0, hi - lo)], {}, row_offset=lo)
        for lo, hi in plan
    ]
    merged = method.merge(parts, state, {})

    assert merged.n_rows == whole.n_rows
    # The spans are exact — that is the structural claim, and the one a broken ownership rule
    # would break.
    assert [(int(a), int(b)) for a, b in zip(merged.start, merged.end)] == [
        (int(a), int(b)) for a, b in zip(whole.start, whole.end)
    ]
    # The confidences agree to float32, not bit-for-bit, and that is BLAS rather than this
    # merge: a chunk is a different number of rows, so the scorer tiles it into different
    # blocks and the cosines accumulate in a different order (measured at 2.7e-5 relative
    # here; see `test_soft_knn_scores_do_not_depend_on_the_block_size`). Far below anything
    # that could move a span, and far above the wobble.
    np.testing.assert_allclose(merged.peak, whole.peak, rtol=1e-3, atol=1e-5)


def test_the_declared_overlap_is_twice_the_context_a_peak_needs():
    """The arithmetic the ownership rule depends on, asserted rather than assumed.

    `_plan_chunks` advances by `target - overlap`. A chunk owns `[start + C, end - C)`, so
    those owned ranges tile exactly when `overlap == 2 * C` and leave a `C`-wide hole when it
    is `C`. Reverting the doubling passes every smoke test and loses matches mid-recording.
    """
    state = _prepared(_motif())
    method = _method()
    context = state.context_rows
    overlap = method.chunk_overlap_rows(state, {})
    assert overlap == 2 * context

    target = 400
    plan = _chunks(2000, overlap, target)
    owned = []
    for index, (lo, hi) in enumerate(plan):
        owned.append(
            (lo if index == 0 else lo + context,
             hi if index == len(plan) - 1 else hi - context)
        )  # fmt: skip
    for (_a, previous_end), (next_start, _b) in zip(owned, owned[1:]):
        assert previous_end == next_start, f"gap between owned ranges: {owned}"
    assert owned[0][0] == 0 and owned[-1][1] == 2000


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
def test_progress_moves_within_one_run_of_one_exemplar(monkeypatch):
    """One unit, one exemplar, one continuous run — every structural loop cut to one.

    What is left is the scorer's own block loop, so this fails unless the hook is inside it.
    The block size is patched down for the same reason the fixture is cut: at the shipped
    8192 a test-sized trajectory is a single block, and a single block reports only `1.0`.
    """
    monkeypatch.setattr("detail.query_methods.soft_knn.SCORE_BLOCK", 64)
    motif = _motif()
    emb = _trajectory(500, motif, at=100)
    seen: list[float] = []
    _method().search(
        _prepared(motif), emb, [(0, len(emb))], {}, progress_cb=seen.append
    )

    assert seen, "the search reported no progress at all"
    assert any(0.0 < f < 1.0 for f in seen), (
        f"only terminal progress was reported ({seen}); a single-exemplar search over one "
        "continuous run is the ordinary case, and it must still move"
    )
    assert seen == sorted(seen), f"progress went backwards: {seen}"


# --------------------------------------------------------------------------- #
# declarations and round-tripping
# --------------------------------------------------------------------------- #
def test_a_missing_background_is_refused_rather_than_scored_without_one():
    """The method declares a `SamplingRequest`, so arriving without one means the run was
    assembled wrong. Scoring against the exemplars alone would return a different number that
    still looks like a confidence."""
    with pytest.raises(ValueError, match="background"):
        _method().prepare([_motif()], {})


def test_non_examples_join_the_labelled_index():
    """`negatives` is `optional` on this method — the first one where it is — so a marked
    non-example has to actually reach the index rather than being accepted and dropped.
    """
    motif = _motif()
    plain = _method().prepare([motif], {}, sample=_background())
    with_negatives = _method().prepare(
        [motif], {}, negatives=[_motif(length=12, seed=11)], sample=_background()
    )
    assert len(with_negatives.labeled) == len(plain.labeled) + 12
    assert with_negatives.n_positive == plain.n_positive


def test_a_shard_round_trips(tmp_path):
    motif = _motif()
    emb = _trajectory(300, motif, at=100)
    method = _method()
    result = method.search(_prepared(motif), emb, [(0, len(emb))], {})
    path = tmp_path / f"shard{method.shard_suffix}"
    method.serialize(result, path)
    back = method.load(path)

    np.testing.assert_array_equal(back.start, result.start)
    np.testing.assert_array_equal(back.end, result.end)
    np.testing.assert_allclose(back.peak, result.peak)
    assert back.row_offset == result.row_offset


def test_a_foreign_shard_says_so_rather_than_raising_a_key_error(tmp_path):
    path = tmp_path / "not_ours.npz"
    np.savez_compressed(path, something_else=np.zeros(3))
    with pytest.raises(ValueError, match="missing arrays"):
        _method().load(path)


def test_the_row_scales_follow_the_exemplar_rather_than_a_constant():
    """A window written in rows means a different duration on every recording. These are
    derived from the example clip, so a 30 Hz recording and a 60 Hz one get the same
    *duration* from the same multiplier."""
    short = _method().prepare([_motif(length=20)], {}, sample=_background())
    long = _method().prepare([_motif(length=80)], {}, sample=_background())
    assert long.smooth_rows > short.smooth_rows
    assert long.separation_rows > short.separation_rows
    assert long.max_extent_rows > short.max_extent_rows


def test_the_extent_fraction_changes_how_long_matches_are():
    """The one genuinely new user-facing knob. A parameter that reads well and does nothing
    is the defect the whole method-params PR existed to close."""
    motif = _motif()
    emb = _trajectory(600, motif, at=200)
    method = _method()

    def median_length(fraction):
        params = SoftKnnQueryConfig(extent_fraction=fraction)
        state = _method().prepare([motif], params, sample=_background())
        result = method.search(state, emb, [(0, len(emb))], params)
        return float(np.median(result.end - result.start + 1))

    assert median_length(0.3) > median_length(
        0.8
    ), "growing to a lower fraction of the peak must yield longer instances"


def test_the_labelled_index_cannot_tell_a_gap_crossing_clip_from_a_contiguous_one():
    """Why this method declares ``requires_contiguous=False``.

    ``prepare`` vstacks every exemplar into one index and reads nothing back but the row
    count, so a clip whose rows straddle a tracking gap contributes exactly what the same
    rows contribute without one. The module used to refuse such a clip on ``dtw``'s behalf —
    a path premise applied to a method with no path, and the commonest way a user's marked
    clip was rejected.

    Asserted on the index itself rather than on a search result, because the state is what
    the declaration is a claim about: a scored trajectory could agree by luck.
    """
    method = _method()
    background = _background()
    rows = _motif(length=24, seed=5)

    # The same 24 rows, once as one clip and once as the two stretches a gap splits them
    # into. A pooling method holds one index either way.
    whole = method.prepare([rows], {}, sample=background)
    split = method.prepare([rows[:10], rows[10:]], {}, sample=background)

    assert np.array_equal(whole.labeled, split.labeled)
    assert whole.n_positive == split.n_positive == 24
    # The property above is only half the guard. `test_query_module` branches on the
    # declaration, so a method that proves it could take a gap-crossing clip and still declares
    # it cannot would be refused by the module and pass both files.
    assert method.exemplars.requires_contiguous is False, (
        "soft_knn's index is identical with and without a gap, yet it declares it needs a "
        "contiguous clip — the module will refuse gap-crossing clips for the default method"
    )


def test_the_row_scales_read_the_clip_count_a_gap_changes():
    """The one thing a gap *does* change here, pinned so it is a decision rather than a
    surprise. The scales come from the **median clip length**, so two stretches of 10 and 14
    rows scale differently from one clip of 24 — and rows, not wall-clock, is the right unit:
    the trajectory these are applied to is also only its observed rows, so the gap is absent
    from both sides and they stay commensurate.
    """
    method = _method()
    background = _background()
    rows = _motif(length=24, seed=5)

    whole = method.prepare([rows], {}, sample=background)
    split = method.prepare([rows[:10], rows[10:]], {}, sample=background)

    assert whole.smooth_rows != split.smooth_rows
    assert whole.max_extent_rows > split.max_extent_rows


# --------------------------------------------------------------------------- #
# the run's minimum match length (QueryConfig.min_match_s)
# --------------------------------------------------------------------------- #
def _bare_state(separation_rows, max_extent_rows=200):
    """Only the scales ``_candidates`` reads; the index plays no part in forming spans."""
    return SoftKnnState(
        labeled=np.zeros((1, DIM), dtype=np.float32),
        n_positive=1,
        smooth_rows=2,
        separation_rows=separation_rows,
        max_extent_rows=max_extent_rows,
    )


def _spike_beside_a_bump():
    """A broad, moderate bump (rows 20-79) with a brief tall spike on its shoulder (62-64).

    The spike seeds first and is cut off at half its own height, so it comes out 3 rows long.
    With a separation of 25 its suppression zone covers the bump's peak at row 40, so the bump
    is left only the fragment ahead of that zone.
    """
    score = np.zeros(100, dtype=np.float32)
    score[20:80] = 0.3
    score[40] = 0.35
    score[62:65] = 0.9
    return score


def test_a_short_spike_cannot_suppress_the_longer_match_beside_it():
    """Why the minimum is applied while matches form, not filtered afterwards.

    Formed first and filtered later, the spike suppresses the bump's peak, the bump survives
    only as a 17-row fragment, and a 20-row minimum then drops both — the stretch returns
    nothing. Rejected during formation, the spike suppresses nothing, and the bump grows
    across it into one match whose confidence is the highest score inside it.
    """
    score = _spike_beside_a_bump()
    state = _bare_state(separation_rows=25)

    unfiltered = _candidates(score, state, 0.5)
    filtered_after = [s for s in unfiltered if s[1] - s[0] + 1 >= 20]
    assert filtered_after == [], "the fixture no longer shows the loss it exists for"

    formed = _candidates(score, state, 0.5, min_rows=20)
    assert formed == [(20, 79, pytest.approx(0.9))]


def test_no_match_is_shorter_than_the_minimum():
    rng = np.random.default_rng(11)
    score = np.clip(rng.normal(0.4, 0.2, size=2000), 0.0, 1.0).astype(np.float32)
    state = _bare_state(separation_rows=5, max_extent_rows=60)
    spans = _candidates(score, state, 0.5, min_rows=8)
    assert spans, "nothing formed — the check below would pass on an empty list"
    assert min(hi - lo + 1 for lo, hi, _ in spans) >= 8
    for lo, hi, peak in spans:
        assert peak == pytest.approx(float(score[lo : hi + 1].max()))


def test_a_zero_minimum_forms_exactly_what_no_minimum_did():
    """The default must not move a single existing result: with nothing rejected, every seed
    is the highest row of its own span, so the reported confidence is unchanged."""
    rng = np.random.default_rng(12)
    score = np.clip(rng.normal(0.4, 0.2, size=2000), 0.0, 1.0).astype(np.float32)
    state = _bare_state(separation_rows=5, max_extent_rows=60)
    spans = _candidates(score, state, 0.5, min_rows=0)
    # The seed's score is what the method reported before the minimum existed.
    for lo, hi, peak in spans:
        assert peak == float(score[lo : hi + 1].max())


def test_search_honours_the_minimum():
    motif = _motif()
    emb = _trajectory(600, motif, at=200)
    method = _method()
    state = _prepared(motif)
    result = method.search(state, emb, [(0, len(emb))], {}, min_rows=15)
    assert len(result.start)
    assert int((result.end - result.start + 1).min()) >= 15


@pytest.mark.parametrize("window", [2, 3, 4, 5, 9])
def test_smoothing_does_not_shift_a_bout_in_time(window):
    """A centred average of a flat-topped bout is centred on the bout, not one row after it."""
    score = np.zeros(200, dtype=np.float32)
    score[100:120] = 1.0
    smoothed = _smooth(score, window)
    assert smoothed.shape == score.shape
    expected = np.array(
        [
            score[max(0, j - window // 2) : j + (window - 1) // 2 + 1].mean()
            for j in range(len(score))
        ]
    )
    np.testing.assert_allclose(smoothed[10:-10], expected[10:-10], atol=1e-6)
    if window % 2:
        on = np.flatnonzero(smoothed > 0.5)
        assert (on[0], on[-1]) == (100, 119)
