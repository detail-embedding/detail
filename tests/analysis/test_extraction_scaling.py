"""Match extraction stays linear in trajectory rows, and returns exactly what it always did.

**The class.** A per-candidate scan over everything accepted so far is fine while candidates are
few, and quadratic the moment they are not. `extract_matches` had one, and §20's Query made every
finite row a candidate: its search stores a cutoff-free curve and `instances_at` extracts at the
**permissive** limit, so a change of mind about a threshold costs a slice rather than a search.
Nothing was wrong at the line — `clip_query` had used the same function for a year, because it
extracts at the *user's* threshold and only a small subset of rows is ever a candidate.

Measured before the fix, on the real shape (`min_separation` = the longest exemplar, so the match
count grows with length too):

    rows     30k     60k    120k    240k
    sep=120  0.16s   0.62s  2.33s   9.03s     <- 4x per doubling
    sep=20   0.85s   3.39s  13.45s  53.16s

At the length of a real recording that cost is re-paid on every cutoff commit, so a slider was
impossible rather than merely slow. `_SpanIndex` binary-searches the spans that could possibly
conflict instead of scanning all of them; after it, the same measurements are 0.04 / 0.08 / 0.16 /
0.33 s, a clean 2x per doubling.

**Two guards, because the fix has two ways to fail.** It could stop being linear (a future edit
reintroducing a full scan), and it could return something different (the index skipping a span it
should have tested). The second is the dangerous one: the extraction is what turns a cutoff into
instances, so a dropped suppression silently changes what a user's behavior output contains. So
the reference implementation of the scan lives here and every result is compared against it — the
same discipline `dtw_distance` gets as the readable form of `_dtw_numba`.
"""

import time

import numpy as np
import pytest

from detail.analysis import retrieval
from detail.analysis.retrieval import Match, _same_match, _too_close


def _reference(
    norm,
    start,
    *,
    threshold,
    min_separation,
    excludes=(),
    overlap=None,
    example_of=None,
    max_results=None,
):
    """The scan as it was before `_SpanIndex`: test every span taken so far, in order.

    Kept readable and obviously correct rather than fast — it is the thing the fast path is
    checked against, so it must be the definition, not a second optimization.
    """
    reachable = np.nonzero(np.isfinite(norm) & (norm <= threshold))[0]
    order = reachable[np.argsort(norm[reachable], kind="stable")]
    matches, taken = [], []
    for end in order:
        end = int(end)
        span = (int(start[end]), end)
        if overlap is None:
            if any(_too_close(*span, ex, min_separation) for ex in excludes):
                continue
            if any(_too_close(*span, t, min_separation) for t in taken):
                continue
        else:
            if any(_same_match(span, ex, overlap) for ex in excludes):
                continue
            if any(_same_match(span, t, overlap) for t in taken):
                continue
        matches.append(
            Match(
                span[0],
                end,
                float(norm[end]),
                int(example_of[end]) if example_of is not None else 0,
            )
        )
        taken.append(span)
        if max_results is not None and len(matches) >= max_results:
            break
    return matches


def _case(rng, n_rows):
    """One randomized extraction problem, covering the edges the happy path hides."""
    if n_rows == 0:
        return {
            "norm": np.zeros(0),
            "start": np.zeros(0, dtype=np.int64),
            "example_of": np.zeros(0, dtype=np.int64),
        }
    norm = rng.random(n_rows) * 2.0
    if rng.random() < 0.3:  # ties: the stable sort order has to survive identically
        norm = np.round(norm, 1)
    if rng.random() < 0.2:  # unreachable rows
        norm[rng.integers(0, n_rows, size=max(1, n_rows // 10))] = np.inf
    if rng.random() < 0.3:  # ragged spans, not one fixed length
        start = np.maximum(
            np.arange(n_rows) - rng.integers(1, max(2, n_rows // 4), size=n_rows), 0
        )
    else:
        span_len = int(rng.integers(1, max(2, n_rows // 4)))
        start = np.maximum(np.arange(n_rows) - span_len + 1, 0)
    return {
        "norm": norm,
        "start": start.astype(np.int64),
        "example_of": rng.integers(0, 3, size=n_rows).astype(np.int64),
    }


@pytest.mark.parametrize("seed", range(12))
def test_the_indexed_scan_returns_exactly_what_the_full_scan_returned(seed):
    """Randomized over both spacing rules, excludes, ties, infinities and truncation.

    Byte-identical, not merely equivalent-looking: `Match` is a namedtuple, so `==` compares
    the span, the score and the example index of every match, in order.
    """
    rng = np.random.default_rng(seed)
    for _ in range(60):
        arrays = _case(rng, int(rng.integers(0, 400)))
        n = len(arrays["norm"])
        excludes = tuple(
            (a, a + int(rng.integers(0, 30)))
            for a in rng.integers(0, max(1, n), size=int(rng.integers(0, 6)))
        )
        kw = {
            "threshold": float(rng.choice([2.0, 1.5, 1.0, 0.5])),
            "min_separation": int(rng.integers(0, 30)),
            "excludes": excludes,
            "overlap": None if rng.random() < 0.5 else int(rng.integers(0, 20)),
            "max_results": None if rng.random() < 0.8 else int(rng.integers(1, 20)),
            "example_of": arrays["example_of"],
        }
        assert retrieval.extract_matches(
            arrays["norm"], arrays["start"], **kw
        ) == _reference(arrays["norm"], arrays["start"], **kw)


def _extraction_seconds(n_rows: int, separation: int) -> float:
    rng = np.random.default_rng(0)
    # Beta(5, 2) on [0, 2] is the shape a real cost curve has: most rows poor, a thin good
    # tail. A uniform curve would suppress differently and time differently.
    norm = rng.beta(5.0, 2.0, size=n_rows) * 2.0
    start = np.maximum(np.arange(n_rows) - separation + 1, 0).astype(np.int64)
    # CPU time, best of three: wall time counts the moments this process waits for a core, and
    # the full suite runs beside seven other workers, so one wall-clock sample per size
    # failed some full runs under load.
    timings = []
    for _ in range(3):
        began = time.process_time()
        retrieval.extract_matches(
            norm, start, threshold=2.0, min_separation=separation
        )  # the permissive limit: every finite row is a candidate
        timings.append(time.process_time() - began)
    return min(timings)


def test_extraction_is_linear_in_rows_not_quadratic():
    """Quadrupling the rows must not multiply the time by ~16.

    Asserted as a *ratio* rather than an absolute time, so it means the same thing on a slow
    CI box as on a fast dev machine — the failure this catches is a change in the exponent,
    which no wall-clock budget can express. The bound is deliberately loose: linear is 4x,
    quadratic is 16x, and anything under 8x cannot be quadratic at this size.
    """
    small = _extraction_seconds(30_000, 120)
    large = _extraction_seconds(120_000, 120)
    # A fast machine can time 30k rows near the clock's resolution, which inflates the ratio;
    # floor the denominator so the assertion is about the algorithm, not the timer.
    ratio = large / max(small, 1e-3)
    assert ratio < 8.0, (
        f"extraction over 4x the rows took {ratio:.1f}x the time ({small:.3f}s -> "
        f"{large:.3f}s). Linear is ~4x and quadratic is ~16x, so this is the per-candidate "
        "scan coming back — see this module's docstring."
    )


def test_the_span_index_finds_every_span_a_full_scan_would():
    """The index's bound is exact, and this is the assertion that says so.

    `nearby` skips spans that *cannot* touch the query; if its arithmetic is off by one it
    skips one that can, and the only symptom is a match that should have been suppressed.
    """
    rng = np.random.default_rng(5)
    spans = [
        (int(a), int(a) + int(rng.integers(0, 40)))
        for a in rng.integers(0, 500, size=80)
    ]
    index = retrieval._SpanIndex(spans)
    for _ in range(500):
        a = int(rng.integers(-50, 550))
        query = (a, a + int(rng.integers(0, 40)))
        reach = int(rng.integers(0, 25))
        near = set(index.nearby(*query, reach))
        # every span the predicate would have accepted must have been offered to it
        should_see = {s for s in spans if _too_close(*query, s, reach)}
        assert should_see <= near


def test_an_added_span_is_visible_to_the_next_query():
    """The growing half of the index — the one the extraction loop actually exercises."""
    index = retrieval._SpanIndex()
    assert index.nearby(10, 20, 0) == []
    index.add((12, 18))
    assert (12, 18) in index.nearby(10, 20, 0)
    # ...and a long span added later widens the reach for everything after it
    index.add((0, 200))
    assert (0, 200) in index.nearby(190, 195, 0)
