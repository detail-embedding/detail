"""Window pooling's working memory is bounded, and bounding it changes no value (#581).

**The class: a sliding-window reduction built by stacking every window first.** A
``(windows, window, D)`` stack costs ``window`` times the data it reduces, and the reduction then
throws all but one row per window away. Nothing is wrong at the line where it is written: the
output is right, and a test on a fixture of a few hundred rows allocates a few megabytes.

**How it hid.** ``segmentation.build_features`` stacked a whole continuity block at once. At
behavior clustering's default window one long recording's stack runs to tens of GB, many
times the output it is pooled into. That is what took a laptop down under both
`codebook_viterbi` and `dtw_finch`, and two issues (#572,
#573) blamed the decode for it. ``dynamics.window_features`` had already been written with a
cumulative sum for exactly this reason, and said so; the fix reached the plot and not the fit.

**The rule.** ``build_features`` stacks at most ``_POOL_CHUNK_BYTES`` of windows at a time into a
preallocated output. Each window reduces independently, so the output is **bit-identical** to the
whole-block stack — no stored fit, label or pinned value moves. The first test holds that; the
second holds the bound, and fails against the whole-block stack.

Swept for the same shape (#581): ``retrieval._sliding_reduce``, since removed with the windowed
matchers, reduced a ``sliding_window_view``, which is a strided view and never copies, and its mean
path was already a cumsum — not an instance.
"""

import tracemalloc

import numpy as np
import pytest

from detail.analysis import segmentation


def _reference(trajectory, runs, window, pool, feat_stride):
    """The whole-block stack this replaced, kept as the definition the chunked one must match."""
    tn = segmentation._normalize(np.asarray(trajectory, dtype=np.float64))
    feats = []
    for starts in segmentation._window_starts(runs, window, feat_stride):
        wins = np.stack([tn[s : s + window] for s in starts])
        if pool == "mean":
            feats.append(wins.mean(axis=1))
        elif pool == "max":
            feats.append(wins.max(axis=1))
        else:
            feats.append(wins.reshape(len(starts), -1))
    return np.vstack(feats)


@pytest.mark.parametrize("pool", ["mean", "max", "flatten"])
@pytest.mark.parametrize("feat_stride", [1, 3])
def test_chunking_changes_no_value(monkeypatch, pool, feat_stride):
    """Bit-identical, not close: a chunk boundary inside a run, several runs, a stride."""
    rng = np.random.default_rng(0)
    trajectory = rng.normal(size=(900, 12))
    runs = [
        (0, 400),
        (400, 405),
        (405, 900),
    ]  # the middle run is shorter than the window
    window = 9
    # A handful of windows per chunk, so every run is split several times.
    monkeypatch.setattr(segmentation, "_POOL_CHUNK_BYTES", 7 * window * 12 * 8)

    got, rows, xruns = segmentation.build_features(
        trajectory, runs, window=window, pool=pool, feat_stride=feat_stride
    )

    assert np.array_equal(got, _reference(trajectory, runs, window, pool, feat_stride))
    assert np.array_equal(
        rows, segmentation.feature_rows(runs, window=window, feat_stride=feat_stride)
    )
    assert xruns[-1][1] == len(got)


def test_an_empty_grid_keeps_its_width():
    """A trajectory shorter than the window has no windows; flatten still reports its width."""
    got, rows, xruns = segmentation.build_features(
        np.ones((3, 4)), [(0, 3)], window=5, pool="flatten"
    )
    assert got.shape == (0, 20) and len(rows) == 0 and xruns == []


def test_the_working_memory_does_not_grow_with_the_window(monkeypatch):
    """The whole-block stack here is 5,000 x 50 x 32 x 8 = 64 MB; bounded, the peak is the
    trajectory, its normalized copy, the output and one chunk — a few megabytes."""
    monkeypatch.setattr(segmentation, "_POOL_CHUNK_BYTES", 1024 * 1024)
    trajectory = np.random.default_rng(1).normal(size=(5000, 32))
    window = 50
    whole_stack = len(trajectory) * window * trajectory.shape[1] * 8

    tracemalloc.start()
    try:
        segmentation.build_features(
            trajectory, [(0, len(trajectory))], window=window, pool="mean"
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < whole_stack / 4, (
        f"pooling peaked at {peak / 1e6:.1f} MB, near the {whole_stack / 1e6:.0f} MB a "
        "whole-block window stack costs"
    )
