"""Overlap-tolerant match extraction + the ``_overlap_rows`` helper (the clip-annotation
knob on ``retrieval``). ``overlap=None`` is the strict clip-query path; a value relaxes the
spacing to allow overlaps up to that many rows against both taken matches and excludes.
"""

import numpy as np
import pytest

from detail.analysis import retrieval


def test_overlap_rows_helper():
    assert retrieval._overlap_rows(0, 3, 2, 5) == 2  # [0,3] ∩ [2,5] = rows 2,3
    assert retrieval._overlap_rows(0, 3, 4, 6) == 0  # adjacent, no shared row
    assert retrieval._overlap_rows(0, 3, 5, 6) < 0  # disjoint
    assert retrieval._overlap_rows(0, 5, 1, 2) == 2  # [1,2] contained in [0,5]


def _two_candidates():
    """Two reachable ends whose spans (0,3) and (2,5) overlap by 2 rows; (0,3) ranks first."""
    norm = np.full(10, np.inf)
    norm[3], norm[5] = 0.01, 0.02
    start = np.arange(10)
    start[3], start[5] = 0, 2
    return norm, start


def test_overlap_boundary_against_taken_match():
    norm, start = _two_candidates()
    spans = lambda ms: sorted((m.start_row, m.end_row) for m in ms)
    # strict (overlap=None): the 2-row overlap drops the second
    assert spans(
        retrieval.extract_matches(norm, start, threshold=0.1, min_separation=0)
    ) == [(0, 3)]
    # overlap=2 tolerates exactly 2 rows → both kept
    assert spans(
        retrieval.extract_matches(
            norm, start, threshold=0.1, min_separation=0, overlap=2
        )
    ) == [(0, 3), (2, 5)]
    # overlap=1 does not (2 > 1) → second dropped
    assert spans(
        retrieval.extract_matches(
            norm, start, threshold=0.1, min_separation=0, overlap=1
        )
    ) == [(0, 3)]


def test_overlap_boundary_against_excludes():
    norm = np.full(10, np.inf)
    norm[5] = 0.01
    start = np.arange(10)
    start[5] = 2  # span (2,5), overlaps the exclude (0,3) by 2 rows
    ex = [(0, 3)]
    assert retrieval.extract_matches(
        norm, start, threshold=0.1, min_separation=0, excludes=ex, overlap=2
    )  # 2 not > 2 → kept
    assert not retrieval.extract_matches(
        norm, start, threshold=0.1, min_separation=0, excludes=ex, overlap=1
    )  # 2 > 1 → dropped


def test_negative_overlap_raises():
    with pytest.raises(ValueError, match="overlap must be >= 0"):
        retrieval.extract_matches(
            np.array([0.0]), np.array([0]), threshold=1.0, min_separation=0, overlap=-1
        )


def test_same_match_containment_clause():
    # identical / contained spans are the same match even when overlap exceeds their length
    assert retrieval._same_match((0, 3), (0, 3), overlap=100)  # identical
    assert retrieval._same_match(
        (2, 3), (0, 6), overlap=100
    )  # (2,3) contained in (0,6)
    assert not retrieval._same_match((0, 3), (5, 8), overlap=0)  # disjoint
    assert retrieval._same_match((0, 4), (3, 7), overlap=1)  # overlap 2 rows > 1
    assert not retrieval._same_match(
        (0, 4), (3, 7), overlap=2
    )  # overlap 2, not >2, not contained


def test_query_multi_reports_best_matching_example():
    # A multi-example query tags each match with the example it best matches (argmin).
    rng = np.random.default_rng(5)
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    db = rng.standard_normal((40, 8)).astype(np.float32)
    db[20:24] = b  # a copy of example b (index 1)
    matches = retrieval.query_multi(
        db, [a, b], fusion="min", band=0, threshold=0.05, min_separation=4
    )
    m = next(mm for mm in matches if abs(mm.start_row - 20) <= 1)
    assert m.example_idx == 1  # it matched example b, not a


def test_contained_candidate_excluded_even_when_overlap_exceeds_length():
    # An exclude that fully contains a candidate always drops it — else a labeled span whose
    # length is < overlap would let its own re-match through (the self-hide bug).
    norm = np.full(10, np.inf)
    norm[3] = 0.01
    start = np.arange(10)
    start[3] = 2  # span (2, 3), length 2
    assert not retrieval.extract_matches(
        norm, start, threshold=0.1, min_separation=0, excludes=[(0, 6)], overlap=100
    )
