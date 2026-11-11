"""The temporal decode (§22) — the step that makes a frame-level labeling worth encoding.

Two things are pinned here, and they are different in kind.

**The compiled Viterbi equals the readable one, bit for bit.** ``decode.viterbi`` runs a numba
kernel and ``decode.viterbi_reference`` is the definition; a fast decode that quietly decodes
differently is worse than a slow one, and the difference would be invisible — both return a
plausible path. Same guarantee ``_dtw_numba.dtw_dp`` makes about its own loop.

**The decode earns its place.** An internal evaluation found `raw < min-run < Viterbi` on AMI,
homogeneity, completeness and boundary F1, at every codebook size and on both representations —
about forty ordered comparisons, all the same direction, and the most replicated effect in the
record. It is reproduced here on planted structure rather than cited, because a decode that did
not actually clean up a flickering labeling would pass every structural test in this file.
"""

import numpy as np
import pytest

from detail.analysis import decode


def _flickering(n_blocks=6, block_rows=40, k=4, flip=0.25, seed=0):
    """A clean state sequence, and a noisy observation of it.

    ``truth`` holds each state for a stretch; ``observed`` is that with a quarter of the frames
    knocked to a random other state — which is what a nearest-centroid labeling of real pose does,
    and what a decode exists to undo.
    """
    rng = np.random.default_rng(seed)
    truth = np.concatenate(
        [np.full(block_rows, i % k, dtype=np.int64) for i in range(n_blocks)]
    )
    observed = truth.copy()
    noise = rng.random(len(truth)) < flip
    observed[noise] = rng.integers(0, k, size=int(noise.sum()))
    return truth, observed


def _emission_from(labels: np.ndarray, k: int, confidence: float = 2.0) -> np.ndarray:
    """Log-likelihoods that favour each row's observed label, so the decode has evidence to weigh
    against the transition prior rather than a free choice."""
    emission = np.zeros((len(labels), k), dtype=np.float64)
    emission[np.arange(len(labels)), labels] = confidence
    return emission


def _purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of rows whose label matches the majority label of their true state."""
    correct = 0
    for state in np.unique(truth):
        rows = np.flatnonzero(truth == state)
        counts = np.bincount(labels[rows])
        correct += int(counts.max())
    return correct / len(truth)


# --- the kernel is the reference ------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_compiled_viterbi_matches_the_reference_exactly(k, seed):
    """Equality, not agreement-up-to-ties: the kernel and the reference both break ties toward the
    lowest state index, and that is part of the contract rather than an accident."""
    _truth, observed = _flickering(k=k, seed=seed)
    runs = [(0, 120), (120, len(observed))]
    emission = _emission_from(observed, k)
    transition = decode.fit_transition(observed, runs, k)

    fast = decode.viterbi(emission, runs, transition)
    slow = decode.viterbi_reference(emission, runs, transition)
    assert np.array_equal(fast, slow)


def test_the_two_agree_on_a_degenerate_transition():
    """A uniform transition matrix makes every path equally likely a priori, so the decode falls
    back to the emission alone — the case most likely to expose a tie-break difference.
    """
    _truth, observed = _flickering(k=3)
    runs = [(0, len(observed))]
    emission = _emission_from(observed, 3)
    uniform = np.log(np.full((3, 3), 1.0 / 3.0))

    assert np.array_equal(
        decode.viterbi(emission, runs, uniform),
        decode.viterbi_reference(emission, runs, uniform),
    )


def test_an_empty_input_decodes_to_nothing():
    assert decode.viterbi(np.zeros((0, 3)), [], np.zeros((3, 3))).shape == (0,)
    assert decode.viterbi(np.zeros((5, 3)), [], np.zeros((3, 3))).tolist() == [0] * 5


# --- blocks are not crossed -------------------------------------------------------------------


def test_the_decode_restarts_at_each_continuity_block():
    """A gap in tracking is an absence of evidence, not a transition. A decoder that ran across one
    would invent a transition where the animal was simply not seen — and the invented one would
    look exactly like a real one in the output."""
    k = 3
    # Two blocks that disagree completely: all state 0, then all state 2.
    observed = np.array([0] * 30 + [2] * 30, dtype=np.int64)
    runs = [(0, 30), (30, 60)]
    emission = _emission_from(observed, k)
    # A transition matrix that strongly prefers staying put, so a decode allowed to cross the
    # boundary would drag the first block's state into the second.
    transition = np.log(np.full((k, k), 0.01) + np.eye(k) * 0.97)

    labels = decode.viterbi(emission, runs, transition)
    assert set(labels[:30].tolist()) == {0}
    assert set(labels[30:].tolist()) == {2}


# --- the decode earns its place ---------------------------------------------------------------


def test_raw_then_min_run_then_viterbi_each_improve_on_the_last():
    """The record's most replicated effect, reproduced rather than cited.

    A decode that left a flickering labeling alone would satisfy every other test in this file.
    """
    k = 4
    truth, observed = _flickering(k=k, flip=0.3, seed=7)
    runs = [(0, len(observed))]
    emission = _emission_from(observed, k)
    transition = decode.fit_transition(observed, runs, k)

    raw = _purity(observed, truth)
    min_run = _purity(decode.smooth_min_run(observed, runs, min_rows=5), truth)
    viterbi = _purity(decode.viterbi(emission, runs, transition), truth)

    assert raw < min_run, f"min-run did not improve on raw ({raw:.3f} -> {min_run:.3f})"
    assert (
        min_run <= viterbi
    ), f"Viterbi did not match or beat min-run ({min_run:.3f} -> {viterbi:.3f})"
    assert viterbi > raw


def test_the_decode_collapses_the_flicker_into_runs():
    """The concrete reason this exists: without it, run-length encoding a raw labeling produces
    thousands of one-frame "instances" rather than the handful of stretches that are there.
    """
    k = 4
    truth, observed = _flickering(k=k, flip=0.3, seed=3)
    runs = [(0, len(observed))]
    emission = _emission_from(observed, k)
    transition = decode.fit_transition(observed, runs, k)

    raw_runs = len(decode.run_spans(observed, 0, len(observed)))
    decoded = decode.viterbi(emission, runs, transition)
    decoded_runs = len(decode.run_spans(decoded, 0, len(decoded)))
    true_runs = len(decode.run_spans(truth, 0, len(truth)))

    assert raw_runs > 5 * true_runs, "the fixture does not actually flicker"
    # Asserted as *moving toward the truth* rather than by some ratio: how far a decode smooths
    # depends on the emission confidence against the transition prior, so a fixed factor would be
    # a number invented here rather than a property of the decode. What must hold is the
    # direction — on this fixture it goes 103 runs -> 41 against a true 6.
    assert abs(decoded_runs - true_runs) < abs(raw_runs - true_runs) / 2
    assert decoded_runs < raw_runs


# --- min-run smoothing ------------------------------------------------------------------------


def test_min_run_leaves_a_clean_labeling_alone():
    """The negative case. A smoother that rewrote already-long runs would be changing the answer
    rather than cleaning it."""
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20, dtype=np.int64)
    runs = [(0, len(labels))]
    assert np.array_equal(decode.smooth_min_run(labels, runs, min_rows=5), labels)


def test_min_run_keeps_every_row_labeled():
    """Absorbing rather than discarding: coverage stays 1.0, so two labelings are compared on
    agreement and not on how much each declined to label."""
    _truth, observed = _flickering(flip=0.4, seed=11)
    runs = [(0, len(observed))]
    smoothed = decode.smooth_min_run(observed, runs, min_rows=6)

    assert len(smoothed) == len(observed)
    assert smoothed.min() >= 0
    for lo, hi, _label in decode.run_spans(smoothed, 0, len(smoothed)):
        assert hi - lo >= 6 or (lo == 0 and hi == len(smoothed))


# --- the transition matrix is estimated, not chosen --------------------------------------------


def test_the_transition_matrix_carries_the_dwell_time():
    """Counted row-to-row rather than run-to-run, so the diagonal *is* the smoothing strength —
    which is why the decode needs no hand-set smoothing parameter."""
    labels = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    transition = decode.fit_transition(labels, [(0, 100)], 2)

    assert transition.shape == (2, 2)
    # Each row is a log-probability distribution.
    assert np.allclose(np.exp(transition).sum(axis=1), 1.0)
    # Staying is far likelier than switching, which is what this fixture shows.
    assert transition[0, 0] > transition[0, 1]
    assert transition[1, 1] > transition[1, 0]


def test_an_unseen_transition_is_unlikely_rather_than_impossible():
    """Add-one smoothing. A zero would be ``-inf`` and would make that transition unreachable
    forever, which is a stronger claim than "it did not happen in this recording"."""
    labels = np.zeros(50, dtype=np.int64)
    transition = decode.fit_transition(labels, [(0, 50)], 3)
    assert np.isfinite(transition).all()
    assert transition[0, 1] < transition[0, 0]


def test_sigma_is_estimated_from_the_rows_it_describes():
    """It is the scale that turns a cosine distance into a likelihood. Choosing it by hand would
    make the decode's confidence a setting rather than a property of the data."""
    centroids = np.eye(3)
    tight = np.array([[1.0, 0.0, 0.0]] * 10)
    loose = np.array([[0.8, 0.6, 0.0]] * 10)
    labels = np.zeros(10, dtype=np.int64)

    assert decode.estimate_sigma_sq(tight, centroids, labels) == pytest.approx(0.0)
    assert decode.estimate_sigma_sq(loose, centroids, labels) > 0.1
    assert decode.estimate_sigma_sq(np.zeros((0, 3)), centroids, labels) == 1.0
