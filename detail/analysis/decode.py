"""Temporal decoding of a frame-level labeling (§22) — the step that makes one worth having.

A nearest-centroid labeling of a pose trajectory flickers: adjacent frames of one behavior land in
different codewords, and a raw run-length encoding of that turns ~110k rows into tens of thousands
of one-frame "instances". Something has to impose the fact that behavior persists, and *which*
something is the one thoroughly replicated result in this project's evaluations, which
found `raw < min-run < Viterbi` on AMI, homogeneity, completeness and boundary F1, at every codebook
size and on both representations, roughly forty ordered comparisons all pointing the same way.
Without a decode the frame path loses to the bout path outright; with one it ties.

Two decoders, in increasing order of what they know:

- ``smooth_min_run`` absorbs every run shorter than a floor into its longer neighbour. Cheap, and
  it keeps coverage at 1.0 — absorbing rather than discarding, so two methods are compared on
  agreement rather than on how much each declined to label.
- ``viterbi`` decodes an HMM whose states are the codewords, with a transition matrix estimated
  from the data itself. The self-transition probability *is* the smoothing strength, so nothing is
  chosen by hand.

Both restart at each **continuity block**: a gap in tracking is not a transition between behaviors,
it is an absence of evidence, and a decoder that ran across one would invent a transition where the
animal was simply not seen.

Lifted from the evaluation script where these were measured. §22 puts them here
because the decode belongs to the *method* — "a frame-level labeling flickers, and suppressing that
is the method's business, not a module-level cleanup pass".
"""

import heapq

import numpy as np


def gaussian_emission(
    features: np.ndarray, centroids: np.ndarray, sigma_sq: float
) -> np.ndarray:
    """Log-likelihood of every row under an isotropic Gaussian around each centroid.

    With both sides L2-normalized, ``||x - c||² = 2 - 2·cos``, so the log-likelihood is
    ``(cos - 1) / sigma²`` up to a constant the decode does not care about.
    """
    return (np.asarray(features) @ np.asarray(centroids).T - 1.0) / max(sigma_sq, 1e-9)


def estimate_sigma_sq(
    features: np.ndarray, centroids: np.ndarray, labels: np.ndarray
) -> float:
    """The spread of the fit rows around the codeword each was assigned to.

    Estimated rather than chosen: it is the scale that turns a cosine distance into a likelihood,
    and picking it by hand would make the decode's confidence a setting instead of a property of
    the data. Estimated on the **fit** rows only.
    """
    features = np.asarray(features)
    if not len(features):
        return 1.0
    residual = 2.0 - 2.0 * (features * np.asarray(centroids)[labels]).sum(axis=1)
    return float(np.mean(np.maximum(residual, 0.0)))


def fit_transition(labels: np.ndarray, runs, n_states: int) -> np.ndarray:
    """Row-normalized **log** transition matrix from a label sequence, add-one smoothed.

    Counted row-to-row rather than run-to-run, so the diagonal carries the dwell time — which is
    what the decode needs: the self-transition probability is the smoothing strength, estimated
    from the data rather than chosen.
    """
    counts = np.ones((n_states, n_states), dtype=np.float64)
    for lo, hi in runs:
        block = np.asarray(labels)[lo:hi]
        if len(block) < 2:
            continue
        np.add.at(counts, (block[:-1], block[1:]), 1.0)
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def viterbi(emission: np.ndarray, runs, log_transition: np.ndarray) -> np.ndarray:
    """Best state path per continuity block, compiled where numba is available.

    Takes the emission matrix rather than computing it, because more than one thing produces one —
    a Gaussian around a centroid, and a classifier's calibrated log-probabilities — and the decode
    is the same either way.

    The kernel is imported lazily (`_decode_numba`), so ``import detail`` still pulls no numba, and
    ``viterbi_reference`` below is what it must agree with **bit for bit**: a fast decode that
    quietly decodes differently is worse than a slow one.
    """
    emission = np.ascontiguousarray(np.asarray(emission, dtype=np.float64))
    log_transition = np.ascontiguousarray(np.asarray(log_transition, dtype=np.float64))
    bounds = np.asarray(
        [[int(lo), int(hi)] for lo, hi in runs], dtype=np.int64
    ).reshape(-1, 2)
    if not len(emission) or not len(bounds):
        return np.zeros(len(emission), dtype=np.int64)
    try:
        from detail.analysis import _decode_numba
    except ImportError:  # numba absent: correct, just slower
        return viterbi_reference(emission, runs, log_transition)
    return _decode_numba.viterbi_blocks(emission, bounds, log_transition)


def viterbi_reference(
    emission: np.ndarray, runs, log_transition: np.ndarray
) -> np.ndarray:
    """The readable form of :func:`viterbi`, and the definition the kernel is tested against.

    Kept rather than deleted for the reason ``_dtw_numba`` keeps its own: the compiled version is
    the one that runs and the plain one is the one that can be read, and a test pinning them equal
    is what makes the first trustworthy.
    """
    emission = np.asarray(emission, dtype=np.float64)
    out = np.zeros(len(emission), dtype=np.int64)
    n_states = emission.shape[1]
    for lo, hi in runs:
        if hi <= lo:
            continue
        block = emission[lo:hi]
        back = np.zeros((len(block), n_states), dtype=np.int32)
        delta = block[0].astype(np.float64)
        for t in range(1, len(block)):
            scored = delta[:, None] + log_transition
            best = np.argmax(scored, axis=0)
            back[t] = best
            delta = scored[best, np.arange(n_states)] + block[t]
        path = np.zeros(len(block), dtype=np.int64)
        path[-1] = int(np.argmax(delta))
        for t in range(len(block) - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        out[lo:hi] = path
    return out


def run_spans(labels: np.ndarray, lo: int, hi: int) -> list[tuple[int, int, int]]:
    """``(start, end_exclusive, label)`` runs within one continuity block."""
    block = np.asarray(labels)[lo:hi]
    if len(block) == 0:
        return []
    edges = [0, *(np.nonzero(np.diff(block) != 0)[0] + 1).tolist(), len(block)]
    return [(lo + a, lo + b, int(block[a])) for a, b in zip(edges, edges[1:])]


def smooth_min_run(labels: np.ndarray, runs, min_rows: int) -> np.ndarray:
    """Absorb every run shorter than ``min_rows`` into its longer neighbour, repeatedly.

    Absorbing rather than discarding keeps coverage at 1.0, so two labelings are compared on
    agreement rather than on how much each declined to label.

    Shortest-first over a linked list of runs with a lazily-deleted heap, because a raw
    nearest-centroid labeling of ~110k rows flickers into tens of thousands of runs and
    recomputing the run list after each absorption would be quadratic in that count.
    """
    out = np.array(labels, copy=True)
    for lo, hi in runs:
        spans = run_spans(out, lo, hi)
        if len(spans) < 2:
            continue
        start = [s[0] for s in spans]
        end = [s[1] for s in spans]
        label = [s[2] for s in spans]
        prev = list(range(-1, len(spans) - 1))
        nxt = [*range(1, len(spans)), -1]
        alive = [True] * len(spans)
        heap = [(end[i] - start[i], i) for i in range(len(spans))]
        heapq.heapify(heap)

        def absorb(target: int, victim: int) -> None:
            """Extend ``target`` over ``victim`` and unlink ``victim``."""
            start[target] = min(start[target], start[victim])
            end[target] = max(end[target], end[victim])
            alive[victim] = False
            if prev[victim] == target:
                nxt[target] = nxt[victim]
                if nxt[victim] != -1:
                    prev[nxt[victim]] = target
            else:
                prev[target] = prev[victim]
                if prev[victim] != -1:
                    nxt[prev[victim]] = target

        while heap:
            length, i = heapq.heappop(heap)
            if not alive[i] or end[i] - start[i] != length:
                continue  # stale entry: this run was absorbed or has since grown
            if length >= min_rows:
                break  # the shortest live run clears the floor, so every run does
            left, right = prev[i], nxt[i]
            if left == -1 and right == -1:
                break
            if left == -1:
                winner = right
            elif right == -1:
                winner = left
            else:
                winner = (
                    left
                    if (end[left] - start[left]) >= (end[right] - start[right])
                    else right
                )
            absorb(winner, i)
            # The winner may now touch a run carrying its own label; they are one run.
            for other in (prev[winner], nxt[winner]):
                if other != -1 and alive[other] and label[other] == label[winner]:
                    absorb(winner, other)
            heapq.heappush(heap, (end[winner] - start[winner], winner))

        for i in range(len(spans)):
            if alive[i]:
                out[start[i] : end[i]] = label[i]
    return out
