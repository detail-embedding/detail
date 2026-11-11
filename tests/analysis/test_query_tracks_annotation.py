"""The clip-annotation plumbing on ``query_tracks``: ``labeled=`` excludes already-labeled
spans and ``overlap_ns=`` is converted to rows per subject. Both default to off, so clip
query is unaffected (covered by test_clip_query.py)."""

import numpy as np

from detail.analysis import clip_query as core

_PERIOD = 33_000_000


def _db(pattern_at, dim=8, total=60):
    rng = np.random.default_rng(1)
    pattern = rng.standard_normal((5, dim)).astype(np.float32)
    emb = rng.standard_normal((total, dim)).astype(np.float32)
    for at in pattern_at:
        emb[at : at + 5] = pattern
    frame_idx = np.arange(total, dtype=np.int32)
    t_ns = (np.arange(total) * _PERIOD).astype(np.int64)
    return {("s1", 0): (emb, frame_idx, t_ns)}


def _query():
    return [
        {
            "recording_id": "s1",
            "track_id": 0,
            "start_t_ns": 10 * _PERIOD,
            "end_t_ns": 14 * _PERIOD,
        }
    ]


def test_labeled_excludes_the_neighborhood_of_a_match():
    databases = _db([10, 30])
    _examples, matches = core.query_tracks(
        databases,
        _query(),
        [("s1", 0)],
        fusion="min",
        band_ns=_PERIOD,
        threshold=0.05,
        min_separation=5,
        overlap_ns=0,
    )
    # The copy at row 30 is found (the example's own location at 10 is auto-excluded).
    assert any(28 <= m.start_frame <= 32 for m in matches)

    labeled = [
        {
            "recording_id": "s1",
            "track_id": 0,
            "start_t_ns": 30 * _PERIOD,
            "end_t_ns": 34 * _PERIOD,
        }
    ]
    _examples2, matches2 = core.query_tracks(
        databases,
        _query(),
        [("s1", 0)],
        fusion="min",
        band_ns=_PERIOD,
        threshold=0.05,
        min_separation=5,
        overlap_ns=0,
        labeled=labeled,
    )
    # With the span labeled, nothing near row 30 resurfaces.
    assert not any(26 <= m.start_frame <= 34 for m in matches2)
