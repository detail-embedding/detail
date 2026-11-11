"""Steps more than one §22 cluster method shares, in one place.

The k-means level and its OpenMP cap, a frame's measured time, and every node's separation from unit
vectors live here rather than being written per method and drifting.
"""

import contextlib
from collections.abc import Iterator

import numpy as np


def kmeans_level(sample: np.ndarray, n_groups: int, seed: int) -> list[np.ndarray]:
    """One level: the sample's k-means groups, compacted to 0..K-1.

    The one place a cluster method runs scikit-learn's k-means, so it runs under
    `blas_safe_openmp` — see there for why that is not optional.
    """
    from sklearn.cluster import KMeans

    distinct = len(np.unique(sample, axis=0))
    groups = min(int(n_groups), distinct)
    if groups < 2:
        return [np.zeros(len(sample), dtype=np.int64)]
    with blas_safe_openmp():
        labels = (
            KMeans(n_clusters=groups, n_init=1, random_state=int(seed))
            .fit(sample)
            .labels_
        )
    return [np.unique(labels, return_inverse=True)[1].astype(np.int64)]


@contextlib.contextmanager
def blas_safe_openmp() -> Iterator[None]:
    """Hold OpenMP to no more threads than the loaded OpenBLAS runs with.

    scikit-learn's k-means runs one OpenMP thread per core and calls OpenBLAS's GEMM from every
    one of them at once. OpenBLAS is built for a fixed number of concurrent callers
    (``MAX_THREADS``, 24 in the Windows numpy/scipy wheels); past it, it prints "precompiled
    NUM_THREADS exceeded, adding auxiliary array" and on a many-core Windows box corrupted its heap
    (0xc0000374), killing the worker with no traceback. Only a sample large enough to occupy every
    core reaches it — k-means splits rows into 256-row chunks, one thread each — which is why a
    small study passed and a large one did not.

    OpenBLAS's own thread count is the ceiling: it never reports more than its build allows.
    Without OpenBLAS loaded (MKL, Accelerate) nothing is capped.
    """
    from threadpoolctl import threadpool_info
    from threadpoolctl import threadpool_limits

    blas_threads = [
        info["num_threads"]
        for info in threadpool_info()
        if info["internal_api"] == "openblas"
    ]
    if not blas_threads:
        yield
        return
    with threadpool_limits(limits=max(1, min(blas_threads)), user_api="openmp"):
        yield


def measured_frame_time(t_ns: np.ndarray, blocks) -> np.ndarray:
    """``(rows,) int64``: the time each analysed frame stands for, from the timestamps alone.

    A frame's time is the gap to the next analysed frame **inside its continuity block**, so a fit
    analysed at every *n*-th frame counts *n* frames of time per row and a tracking gap counts
    nothing. A block's last frame has no successor, so it takes the block's measured step (its
    median positive gap); a block of one frame takes the run's. No fps is consulted: an advertised
    rate is exactly what differs from the timestamps when it is wrong (CLAUDE.md, "Frame counts
    come from the timestamp map").
    """
    t = np.asarray(t_ns, dtype=np.int64)
    out = np.zeros(len(t), dtype=np.int64)
    steps_seen: list[np.ndarray] = []
    lasts: list[tuple[int, int]] = []
    for lo, hi in blocks:
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            continue
        steps = np.diff(t[lo:hi])
        out[lo : hi - 1] = np.maximum(steps, 0)
        positive = steps[steps > 0]
        if len(positive):
            steps_seen.append(positive)
            lasts.append((hi - 1, int(np.median(positive))))
        else:
            lasts.append((hi - 1, -1))
    pooled = np.concatenate(steps_seen) if steps_seen else np.zeros(0, np.int64)
    run_step = int(np.median(pooled)) if len(pooled) else 0
    for row, step in lasts:
        out[row] = step if step >= 0 else run_step
    return out


def vector_node_separations(tree, sample: np.ndarray, sample_leaf: np.ndarray):
    """Centre distance over within-child spread for every internal node, from unit vectors.

    Exact and additive, which is why it is cheap for every node at once: the mean cosine
    similarity of a node's members to its centroid is ``|S| / n``, where ``S`` is the sum of
    their unit vectors, and ``S`` of a node is the sum of its children's. So one scatter per leaf
    and one pass up the merges gives every node's centroid and spread.
    """
    leaves = tree.n_leaves
    vectors = np.asarray(sample, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms > 0, norms, 1.0)
    sums = np.zeros((tree.n_nodes, vectors.shape[1]), dtype=np.float64)
    counts = np.zeros(tree.n_nodes, dtype=np.float64)
    np.add.at(sums, np.asarray(sample_leaf, dtype=np.int64), vectors)
    np.add.at(counts, np.asarray(sample_leaf, dtype=np.int64), 1.0)
    for merge, (a, b) in enumerate(np.asarray(tree.children, dtype=np.int64)):
        sums[leaves + merge] = sums[a] + sums[b]
        counts[leaves + merge] = counts[a] + counts[b]
    length = np.linalg.norm(sums, axis=1)
    centre = sums / np.where(length > 0, length, 1.0)[:, None]
    spread = np.where(counts > 0, 1.0 - length / np.maximum(counts, 1.0), np.nan)
    out = np.full(tree.n_nodes, np.nan, dtype=np.float64)
    for merge, (a, b) in enumerate(np.asarray(tree.children, dtype=np.int64)):
        within = (spread[a] + spread[b]) / 2.0
        between = 1.0 - float(centre[a] @ centre[b])
        out[leaves + merge] = between / max(within, 1e-9)
    return out
