"""Temporally-weighted nearest-neighbour clustering of a pose trajectory (TW-FINCH).

Sarfraz et al., *Temporally-Weighted Hierarchical Clustering for Unsupervised Action Segmentation*
(CVPR 2021). Unlike everything else in this package it labels **rows, not bouts**: there is no
segmentation step and no distance matrix over segments, because segments *emerge* as runs of the
same label. That is the majority shape in the action-segmentation literature and the opposite of
our own pipeline, which cuts first and clusters the pieces.

The whole method is one modulated distance:

    W(i, j) = (1 - <x_i, x_j>) * (|t_i - t_j| / N)

so two rows are neighbours only when they are close **both** in pose and in time. Each row is
linked to its single nearest neighbour under ``W``; the connected components of that graph are the
first partition, and recursing on the cluster means of *both* the features and the timestamps gives
a hierarchy. Because each merge averages timestamps, clusters stay temporally coherent, which is
what gets action ordering for free where earlier methods needed a Mallows model or Viterbi.

**Per recording, and that is not an implementation detail.** The temporal term is defined within
one sequence, so cluster ids from two recordings mean nothing to each other. A clustering built this
way has a per-recording label space, and scoring it against a global-vocabulary clustering is not a
like-for-like comparison.

Pure compute over arrays: numpy only, no zarr, no paths, no Qt.
"""

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0.0, n, 1.0)


def _nearest_neighbors(
    feats: np.ndarray, times: np.ndarray, block: int = 512
) -> np.ndarray:
    """Each row's nearest neighbour under the temporally weighted distance.

    Computed in row blocks rather than as one N×N matrix: only the arg-min per row is needed, so
    materializing the full matrix would cost O(N²) memory for a quantity that is O(N). A 20k-row
    trajectory would be 3 GB as a dense matrix and is a few MB this way.
    """
    n = len(feats)
    span = float(n) if n else 1.0
    out = np.empty(n, dtype=np.int64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        # 1 - cosine on already-normalized rows, modulated by scaled time distance.
        feature = 1.0 - feats[start:stop] @ feats.T
        temporal = np.abs(times[start:stop, None] - times[None, :]) / span
        weighted = feature * temporal
        np.fill_diagonal(weighted[:, start:stop], np.inf)
        out[start:stop] = np.argmin(weighted, axis=1)
    return out


def _components(nearest: np.ndarray) -> np.ndarray:
    """Connected components of the 1-nearest-neighbour graph, as compact labels."""
    n = len(nearest)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        a, b = find(i), find(int(nearest[i]))
        if a != b:
            parent[a] = b
    roots = [find(i) for i in range(n)]
    order = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([order[r] for r in roots], dtype=np.int64)


def _cluster_means(
    feats: np.ndarray, times: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean feature and mean timestamp per cluster — what the next recursion level clusters."""
    k = int(labels.max()) + 1
    onehot = np.zeros((len(labels), k))
    onehot[np.arange(len(labels)), labels] = 1.0
    counts = onehot.sum(axis=0)[:, None]
    return (
        _normalize((onehot.T @ feats) / counts),
        (onehot.T @ times[:, None] / counts).ravel(),
    )


def tw_finch_hierarchy(trajectory: np.ndarray, times: np.ndarray) -> list[np.ndarray]:
    """Every TW-FINCH partition, finest first, each labeling all the original rows."""
    feats = _normalize(np.asarray(trajectory, dtype=np.float64))
    times = np.asarray(times, dtype=np.float64)
    if len(feats) < 2:
        return [np.zeros(len(feats), dtype=np.int64)]

    labels = _components(_nearest_neighbors(feats, times))
    levels = [labels]
    cur_f, cur_t = _cluster_means(feats, times, labels)
    while len(cur_f) > 1:
        step = _components(_nearest_neighbors(cur_f, cur_t))
        if len(set(step.tolist())) == len(cur_f):
            break  # nothing merged, so recursing again will not either
        labels = step[labels]
        levels.append(labels)
        cur_f, cur_t = _cluster_means(feats, times, labels)
    return levels


def _merge_to(
    feats: np.ndarray, times: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Merge the temporally closest pair of clusters until ``n_clusters`` remain.

    The paper's Algorithm 2: with each merge averaging timestamps, taking the pair with the
    smallest weighted distance keeps the result temporally consistent rather than merging two
    lookalike stretches from opposite ends of the recording.
    """
    labels = np.array(labels, copy=True)
    while int(labels.max()) + 1 > n_clusters:
        cf, ct = _cluster_means(feats, times, labels)
        span = float(len(feats))
        weighted = (1.0 - cf @ cf.T) * (np.abs(ct[:, None] - ct[None, :]) / span)
        np.fill_diagonal(weighted, np.inf)
        i, j = np.unravel_index(np.argmin(weighted), weighted.shape)
        lo, hi = (int(i), int(j)) if i < j else (int(j), int(i))
        labels[labels == hi] = lo
        labels[labels > hi] -= 1
    return labels


def tw_finch_labels(
    trajectory: np.ndarray,
    times: np.ndarray,
    *,
    n_clusters: int = 8,
    partition: int | None = None,
    max_rows: int = 6000,
) -> np.ndarray:
    """One label per row of ``trajectory``.

    ``max_rows`` bounds the cost: the nearest-neighbour search is O(rows²) in time however little
    memory it uses, so a long recording is evenly subsampled to at most this many rows, clustered,
    and every row then takes the label of the nearest retained row **in time**. Temporal nearest is
    the right fill because TW-FINCH's labels are temporally contiguous by construction. This
    mirrors ``max_segments`` on the bout path: a bound that is visible rather than a silent stall.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    n = len(trajectory)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    keep = (
        np.linspace(0, n - 1, int(max_rows)).round().astype(int)
        if n > max_rows
        else np.arange(n)
    )
    keep = np.unique(keep)
    feats = _normalize(trajectory[keep])
    kept_times = times[keep]

    levels = tw_finch_hierarchy(trajectory[keep], kept_times)
    if partition is not None:
        labels = levels[max(0, min(int(partition), len(levels) - 1))]
    else:
        usable = [lab for lab in levels if int(lab.max()) + 1 >= n_clusters]
        labels = _merge_to(
            feats, kept_times, usable[-1] if usable else levels[0], int(n_clusters)
        )
    if len(keep) == n:
        return labels
    # Nearest retained row in time — searchsorted on the (sorted) kept indices.
    pos = np.searchsorted(keep, np.arange(n))
    pos = np.clip(pos, 0, len(keep) - 1)
    left = np.clip(pos - 1, 0, len(keep) - 1)
    take_left = np.abs(keep[left] - np.arange(n)) <= np.abs(keep[pos] - np.arange(n))
    return labels[np.where(take_left, left, pos)]
