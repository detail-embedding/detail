"""Tests for the clip helpers (detail/analysis/clip_query.py):
continuity segmentation, t_ns→row resolution, band-in-time, and span checks.
"""

import numpy as np
import pytest

from detail.analysis import clip_query

_PERIOD = 33_000_000  # ~30 fps in ns


def _make_db(rng, total, *, dim=8, gaps=()):
    """A synthetic subject database (emb, frame_idx, t_ns). ``gaps`` = rows before which
    a large time/frame jump is inserted (a tracking discontinuity)."""
    emb = rng.standard_normal((total, dim)).astype(np.float32)
    t_ns = (np.arange(total) * _PERIOD).astype(np.int64)
    for g in gaps:
        t_ns[g:] += _PERIOD * 50
    frame_idx = (t_ns // _PERIOD).astype(np.int32)
    return emb, frame_idx, t_ns


def test_nominal_period_and_default_gap():
    _, _, t_ns = _make_db(np.random.default_rng(0), 50)
    assert clip_query.nominal_period_ns(t_ns) == _PERIOD
    assert clip_query.default_max_gap_ns(t_ns) == int(
        clip_query.DEFAULT_GAP_FACTOR * _PERIOD
    )


def test_segment_runs_no_gap_is_single_run():
    _, _, t_ns = _make_db(np.random.default_rng(1), 40)
    runs = clip_query.segment_runs(t_ns, clip_query.default_max_gap_ns(t_ns))
    assert runs == [(0, 40)]


def test_segment_runs_splits_on_gap():
    _, _, t_ns = _make_db(np.random.default_rng(2), 40, gaps=(25,))
    runs = clip_query.segment_runs(t_ns, clip_query.default_max_gap_ns(t_ns))
    assert runs == [(0, 25), (25, 40)]


def test_segment_runs_nonpositive_gap_is_single_run():
    _, _, t_ns = _make_db(np.random.default_rng(3), 10, gaps=(5,))
    assert clip_query.segment_runs(t_ns, 0) == [(0, 10)]


def test_segment_runs_empty():
    assert clip_query.segment_runs(np.array([], dtype=np.int64), 1) == []


def test_span_to_rows_basic_and_errors():
    _, _, t_ns = _make_db(np.random.default_rng(4), 30)
    assert clip_query.span_to_rows(t_ns, int(t_ns[10]), int(t_ns[15])) == (10, 15)
    with pytest.raises(ValueError, match="before start_t_ns"):
        clip_query.span_to_rows(t_ns, int(t_ns[15]), int(t_ns[10]))


def test_span_to_rows_no_rows_raises():
    # a time range that falls inside a gap (no embedded rows) raises.
    _, _, t_ns = _make_db(np.random.default_rng(5), 30, gaps=(10,))
    with pytest.raises(ValueError, match="no embedded rows"):
        clip_query.span_to_rows(t_ns, int(t_ns[9]) + 1, int(t_ns[10]) - 1)


def test_band_rows_scales_with_cadence():
    fast = (np.arange(20) * _PERIOD).astype(np.int64)  # 30 fps
    slow = (np.arange(20) * _PERIOD * 3).astype(np.int64)  # 10 fps
    # a 0.5 s band is 3x as many rows at 3x the cadence.
    assert clip_query.band_rows(fast, _PERIOD * 2) == 2
    assert clip_query.band_rows(slow, _PERIOD * 6) == 2
    assert clip_query.band_rows(fast, _PERIOD * 6) == 6
    assert clip_query.band_rows(fast, 0) == 0
    assert clip_query.band_rows(np.array([5], dtype=np.int64), _PERIOD) == 0


def test_frame_span_to_t_ns():
    _, frame_idx, t_ns = _make_db(np.random.default_rng(11), 30)
    s, e = clip_query.frame_span_to_t_ns(frame_idx, t_ns, 10, 15)
    assert (s, e) == (int(t_ns[10]), int(t_ns[15]))
    with pytest.raises(ValueError, match="no embedded rows"):
        _, _, t2 = _make_db(np.random.default_rng(12), 30, gaps=(10,))
        f2 = (t2 // _PERIOD).astype(np.int32)
        clip_query.frame_span_to_t_ns(f2, t2, int(f2[9]) + 1, int(f2[10]) - 1)


def test_check_span_accepts_a_continuous_span():
    _, frame_idx, t_ns = _make_db(np.random.default_rng(20), 40)
    check = clip_query.check_span(frame_idx, t_ns, int(t_ns[10]), int(t_ns[20]))
    assert check.ok and check.reason is None and check.gaps == ()
    assert (check.start_row, check.end_row, check.n_rows) == (10, 20, 11)
    assert (check.start_frame, check.end_frame) == (10, 20)


def test_check_span_reports_the_gap_it_crosses():
    _, frame_idx, t_ns = _make_db(np.random.default_rng(21), 60, gaps=(30,))
    check = clip_query.check_span(frame_idx, t_ns, int(t_ns[28]), int(t_ns[32]))
    assert not check.ok
    assert "crosses a break in tracking of 1.68 s" in check.reason
    assert "50 frames with no detection" in check.reason
    (gap,) = check.gaps
    assert gap.duration_ns == _PERIOD * 51 and gap.n_frames == 50
    assert (gap.start_frame, gap.end_frame) == (29, 80)
    # The span still resolves — the rows are reported so a caller can show what it found.
    assert (check.start_row, check.end_row) == (28, 32)


def test_check_span_counts_every_gap_and_names_the_longest():
    # Two breaks, the second twice the first (its row is listed twice), so the reason has
    # to report the longest rather than the one it met first.
    _, frame_idx, t_ns = _make_db(np.random.default_rng(22), 60, gaps=(20, 40, 40))
    check = clip_query.check_span(frame_idx, t_ns, int(t_ns[10]), int(t_ns[50]))
    assert not check.ok and len(check.gaps) == 2
    assert "crosses 2 breaks in tracking" in check.reason
    assert (
        f"{(_PERIOD * 101) / 1e9:.2f} s" in check.reason
    )  # the longest, not the first


def test_check_span_edges_empty_single_row_and_boundaries():
    empty = np.array([], dtype=np.int64)
    assert not clip_query.check_span(empty, empty, 0, 10).ok
    assert (
        "no embedded frames in this run"
        in clip_query.check_span(empty, empty, 0, 10).reason
    )

    _, one_f, one_t = _make_db(np.random.default_rng(23), 1)
    assert clip_query.check_span(one_f, one_t, int(one_t[0]), int(one_t[0])).ok

    _, frame_idx, t_ns = _make_db(np.random.default_rng(24), 60, gaps=(30,))
    # Ending exactly on the last row before the break is continuous; reaching one row
    # past it is not — the boundary itself must not be off by one.
    assert clip_query.check_span(frame_idx, t_ns, int(t_ns[20]), int(t_ns[29])).ok
    assert not clip_query.check_span(frame_idx, t_ns, int(t_ns[20]), int(t_ns[30])).ok

    # A span landing wholly inside the break has no rows at all — a different verdict.
    inside = clip_query.check_span(
        frame_idx, t_ns, int(t_ns[29]) + 1, int(t_ns[30]) - 1
    )
    assert not inside.ok and "no embedded frames" in inside.reason
    assert inside.start_row is None and inside.n_rows == 0

    assert not clip_query.check_span(frame_idx, t_ns, int(t_ns[20]), int(t_ns[10])).ok


def test_check_span_honors_an_explicit_split_gap():
    _, frame_idx, t_ns = _make_db(np.random.default_rng(25), 60, gaps=(30,))
    span = (int(t_ns[28]), int(t_ns[32]))
    assert not clip_query.check_span(frame_idx, t_ns, *span).ok
    # A threshold wider than the break bridges it; 0 disables splitting entirely.
    assert clip_query.check_span(frame_idx, t_ns, *span, _PERIOD * 60).ok
    assert clip_query.check_span(frame_idx, t_ns, *span, 0).ok


def test_run_frame_ranges_are_global_frames():
    _, frame_idx, t_ns = _make_db(np.random.default_rng(26), 60, gaps=(30,))
    assert clip_query.run_frame_ranges(frame_idx, t_ns) == [(0, 29), (80, 109)]
    # Row ranges and frame ranges describe the same runs, in different units.
    runs = clip_query.segment_runs(t_ns, clip_query.default_max_gap_ns(t_ns))
    assert [(a, b - 1) for a, b in runs] == [(0, 29), (30, 59)]


def test_a_gap_crossing_span_is_accepted_when_the_caller_does_not_need_a_path():
    """``require_contiguous=False`` is the asking method's answer, not this module's guess.

    The gaps stay populated either way. A surface that accepts such a clip still has to *say*
    the animal was not seen for part of it, and reading that off ``reason`` would have meant
    ``reason`` doing two jobs — "why this was refused" and "what is in it" — which is how a
    refusal message ends up rendered beside an accepted clip.
    """
    _, _, t_ns = _make_db(np.random.default_rng(3), 40, gaps=(20,))
    frame_idx = (t_ns // _PERIOD).astype(np.int32)

    strict = clip_query.check_span(
        frame_idx, t_ns, int(t_ns[10]), int(t_ns[30]), require_contiguous=True
    )
    assert not strict.ok
    assert "break in tracking" in strict.reason

    relaxed = clip_query.check_span(
        frame_idx, t_ns, int(t_ns[10]), int(t_ns[30]), require_contiguous=False
    )
    assert relaxed.ok
    assert relaxed.reason is None
    # The span still resolves to the same rows, and the break is still reportable.
    assert (relaxed.start_row, relaxed.end_row) == (10, 30)
    assert relaxed.gaps == strict.gaps
    assert relaxed.gap_note == strict.reason


def test_a_span_with_no_embedded_rows_is_refused_however_the_caller_asks():
    """The unconditional half. Contiguity is the method's to relax; *having rows at all* is
    not — a clip with nothing in it gives a pooling method nothing to pool, so relaxing the
    gap rule must not quietly relax this one too."""
    _, _, t_ns = _make_db(np.random.default_rng(4), 20)
    frame_idx = (t_ns // _PERIOD).astype(np.int32)
    after_the_end = int(t_ns[-1]) + _PERIOD * 100
    check = clip_query.check_span(
        frame_idx,
        t_ns,
        after_the_end,
        after_the_end + _PERIOD,
        require_contiguous=False,
    )
    assert not check.ok
    assert "not detected" in check.reason


def test_the_break_sentence_leads_with_a_count_that_cannot_round_to_nothing():
    """One formatter, and it has to read correctly at every cadence.

    Two decimals put a real break at a high embed rate at ``0.00 s`` — "a break in tracking of
    0.00 s" reads as no break — and ``1 frames`` is wrong at every cadence. Both were tolerable
    while this sentence only ever explained a refusal; it is now shown beside accepted clips
    too, so it is read far more often.
    """
    # A single dropped frame at a 250 fps embed cadence: 4 ms, which two decimals would
    # print as "0.00 s". Frames only, and singular.
    one = clip_query.Gap(0, 4_000_000, 0, 2)
    assert clip_query._gap_reason([one]) == (
        "crosses a break in tracking of 1 frame with no detection"
    )

    # A break long enough for the duration to mean something carries both, and pluralizes.
    many = clip_query.Gap(0, _PERIOD * 31, 0, 31)
    sentence = clip_query._gap_reason([many])
    assert "30 frames with no detection" in sentence
    assert "0.00 s" not in sentence
    assert " s (" in sentence
