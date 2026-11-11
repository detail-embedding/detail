"""Selectable-matcher tests (§16) — the alt similarity methods share the DTW span contract."""

import numpy as np
import pytest

from detail.analysis import retrieval

METHODS = ["dtw", "clip_pooled_mean", "clip_pooled_max", "keyframe"]


def _planted(rng, m=200, d=8, length=10, at=120):
    db = rng.normal(0, 1, (m, d)).astype(np.float32)
    query = rng.normal(0, 1, (length, d)).astype(np.float32)
    db[at : at + length] = query  # an exact copy of the query, planted
    return db, query, at, length


@pytest.mark.parametrize("method", METHODS)
def test_matcher_recovers_a_planted_region(method):
    rng = np.random.default_rng(0)
    db, query, at, length = _planted(rng)
    matches = retrieval.query_multi(
        db, [query], fusion="mean", band=0, threshold=2.0,
        min_separation=length, method=method,
    )  # fmt: skip
    assert matches
    top = matches[0]  # best (lowest score) should land on the planted copy
    assert top.start_row <= at + length - 1 and top.end_row >= at
    assert 0 <= top.start_row <= top.end_row


def test_pooled_mean_band0_is_exact_fixed_length():
    rng = np.random.default_rng(1)
    db, query, _at, length = _planted(rng)
    top = retrieval.query_multi(
        db, [query], fusion="mean", band=0, threshold=2.0,
        min_separation=length, method="clip_pooled_mean",
    )[0]  # fmt: skip
    assert top.end_row - top.start_row + 1 == length  # band=0 → a length-L window


def test_band_gives_variable_length_spans():
    rng = np.random.default_rng(4)
    db, query, _at, length = _planted(rng, length=10)
    top = retrieval.query_multi(
        db, [query], fusion="mean", band=4, threshold=2.0,
        min_separation=length, method="clip_pooled_mean",
    )[0]  # fmt: skip
    assert abs((top.end_row - top.start_row + 1) - length) <= 4


def test_every_curve_fn_obeys_the_span_contract():
    rng = np.random.default_rng(3)
    db, query, _at, _length = _planted(rng)
    dbn = retrieval._l2_normalize(db)
    qn = retrieval._l2_normalize(query)
    ends = np.arange(len(db))
    for name in retrieval._CURVE_FNS:
        # The registry carries each method's defining params, so the test no longer has to
        # infer `pool` from the id — which was the same string-sniffing the resolver did.
        fn, params = retrieval._resolve_curve_fn(name, None)
        cost, start = fn(dbn, qn, 0, params)
        finite = np.isfinite(cost)
        assert np.all(start[finite] >= 0)
        assert np.all(
            start[finite] <= ends[finite]
        )  # inclusive start never past the end


@pytest.mark.parametrize("method", ["clip_pooled_mean", "clip_pooled_max", "keyframe"])
def test_length_bounds_hold_a_matcher_that_cannot_narrow_its_own_search(method):
    """The post-hoc mask is documented as "the backstop for matchers that cannot narrow their
    own search" — and until now nothing exercised it as one.

    DTW honours a duration range inside the DP, so on the only production path that passes
    bounds the mask is a no-op and could have been deleted without a test noticing. The other
    matchers ignore the key entirely, so they are where the mask is the *whole* enforcement:
    without it a run whose span falls outside the declared range is returned anyway.
    """
    rng = np.random.default_rng(5)
    db, query, at, length = _planted(rng)
    band = 6
    wide = retrieval.fused_curve(db, [query], fusion="mean", band=band, method=method)[
        0
    ]
    assert np.isfinite(wide).any()

    # A range no span can satisfy must empty the curve for these matchers...
    impossible = retrieval.fused_curve(
        db,
        [query],
        fusion="mean",
        band=band,
        length_bounds=[(length + band + 5, length + band + 9)],
        method=method,
    )[0]
    assert not np.isfinite(impossible).any(), "out-of-range spans survived the mask"

    # ...while a range containing the planted span keeps it.
    kept, starts, _ = retrieval.fused_curve(
        db,
        [query],
        fusion="mean",
        band=band,
        length_bounds=[(length - 1, length + 1)],
        method=method,
    )
    reachable = np.isfinite(kept)
    assert reachable.any()
    spans = np.arange(len(db))[reachable] - starts[reachable] + 1
    assert np.all((spans >= length - 1) & (spans <= length + 1))


@pytest.mark.parametrize("select", ["medoid", "distinctive", "rare"])
def test_keyframe_select_modes_run_and_recover(select):
    rng = np.random.default_rng(5)
    db, query, at, length = _planted(rng)
    matches = retrieval.query_multi(
        db, [query], fusion="mean", band=0, threshold=2.0, min_separation=length,
        method="keyframe", method_params={"keyframe_select": select, "n_keyframes": 2},
    )  # fmt: skip
    assert matches
    assert matches[0].start_row <= at + length - 1 and matches[0].end_row >= at


def test_unknown_method_raises():
    rng = np.random.default_rng(2)
    db, query, _at, _length = _planted(rng)
    with pytest.raises(ValueError, match="unknown method"):
        retrieval.query_multi(
            db, [query], fusion="mean", band=0, threshold=2.0,
            min_separation=5, method="nope",
        )  # fmt: skip


def test_dtw_method_matches_the_legacy_path():
    # method="dtw" must be byte-identical to the original single-matcher behavior.
    rng = np.random.default_rng(6)
    db, query, _at, length = _planted(rng)
    a = retrieval.query_multi(
        db, [query], fusion="mean", band=2, threshold=2.0, min_separation=length,
    )  # fmt: skip
    b = retrieval.query_multi(
        db, [query], fusion="mean", band=2, threshold=2.0, min_separation=length,
        method="dtw",
    )  # fmt: skip
    assert a == b
