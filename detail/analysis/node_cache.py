"""The node cache's leaf-additive tables: every figure a node card sums, stored per leaf (§22).

The interpretation screen asks about **any** node of a fit's merge tree, and there can be
thousands. Everything that adds up (member time, members, time per recording, position counts,
descriptor histograms) is therefore kept per **leaf**, and a node's value is the sum over the
leaves under it: one fancy-index sum, exact, with no cut and no floor involved. What does not add
up is per node and computed when asked, from per-run sums kept here: a clip's typicality is a sum
over the leaf runs it is made of (``run_sums``), so it is exact for every node too.

Pure over arrays: numpy only, no zarr, no paths, no Qt. The builder in ``api/node_cache.py``
reads the inputs and ``storage/node_cache.py`` writes the result.
"""

from dataclasses import dataclass

import numpy as np

#: Position histogram resolution, per axis, over the frame-normalized unit square.
POSITION_BINS = 16
#: Dimensions a ``centroid`` method's per-run vector sums are kept in: the leading right singular
#: vectors of its fit sample. On a 768-dimension pose run, 64 keep each node's run order
#: close to the full vectors' for a twelfth of the bytes.
VECTOR_DIMS = 64


@dataclass(frozen=True)
class LeafTables:
    """The leaf-additive half of a node cache. Shapes are in ``storage/node_cache.py``."""

    recording_ids: tuple[str, ...]
    leaf_time_ns: np.ndarray  # (L,) int64
    leaf_members: np.ndarray  # (L,) int64
    leaf_rec_time_ns: np.ndarray  # (L, R) int64
    leaf_rec_pos: np.ndarray  # (L, R, B, B) uint32
    run_recording: np.ndarray  # (U,) int32
    run_subject: np.ndarray  # (U,) int64
    run_start_ns: np.ndarray  # (U,) int64
    run_end_ns: np.ndarray  # (U,) int64
    run_leaf: np.ndarray  # (U,) int32
    separation: np.ndarray  # (N,) float32, NaN on leaves
    #: Each run's mean frame-normalized position, NaN where unknown. ``None`` only for a cache
    #: written before these columns existed (read back as all-NaN).
    run_cx: np.ndarray | None = None  # (U,) float32
    run_cy: np.ndarray | None = None  # (U,) float32
    #: Members per run. ``None`` for a cache written before clips were listed from runs.
    run_rows: np.ndarray | None = None  # (U,) int32
    #: The run's last member's own end (inclusive, the §19 convention an instance ends on),
    #: where ``run_end_ns`` is the end of the time the run covers.
    run_last_ns: np.ndarray | None = None  # (U,) int64
    #: ``distance`` scoring: the sum of the run's members' distances.
    run_dist_sum: np.ndarray | None = None  # (U,) float64
    #: ``centroid`` scoring: the run's members' vectors summed, in ``VECTOR_DIMS`` dimensions.
    run_vec_sum: np.ndarray | None = None  # (U, K) float32
    #: ``centroid`` scoring: ``run_vec_sum`` summed per leaf, so a node's centroid is a sum.
    leaf_vec_sum: np.ndarray | None = None  # (L, K) float64


def leaf_tables(
    places,
    leaves: np.ndarray,
    n_leaves: int,
    separations: np.ndarray,
    positions: tuple[np.ndarray, np.ndarray] | None = None,
    scoring=None,
) -> LeafTables:
    """Build every leaf-additive table from one pass over the members.

    ``scoring`` is the method's ``ClipScoring``: whether runs of one leaf join (a bout method's
    never do, so each bout is its own run) and the per-member figures the runs sum. ``None``
    joins runs and sums nothing but their member counts.

    ``places`` is the method's ``MemberPlaces`` and ``leaves`` its ``member_leaves``, in the same
    index space. ``positions`` is ``(cx_norm, cy_norm)`` per member, ``NaN`` where the subject's
    position is unknown, or ``None`` for a run with no kinematics at all. Such a run gets an all
    zero histogram, and its reader says why rather than drawing an empty map as a finding.
    """
    leaves = np.asarray(leaves, dtype=np.int64)
    recordings = len(places.recording_ids)
    rec = np.asarray(places.recording_index, dtype=np.int64)
    time = np.asarray(places.time_ns, dtype=np.int64)

    leaf_time = np.zeros(n_leaves, dtype=np.int64)
    np.add.at(leaf_time, leaves, time)
    leaf_members = np.bincount(leaves, minlength=n_leaves).astype(np.int64)
    leaf_rec_time = np.zeros((n_leaves, recordings), dtype=np.int64)
    np.add.at(leaf_rec_time, (leaves, rec), time)

    pos = np.zeros(
        (n_leaves, recordings, POSITION_BINS, POSITION_BINS), dtype=np.uint32
    )
    if positions is not None:
        cx, cy = (np.asarray(v, dtype=np.float64) for v in positions)
        known = np.isfinite(cx) & np.isfinite(cy)
        bx = np.clip((cx[known] * POSITION_BINS).astype(np.int64), 0, POSITION_BINS - 1)
        by = np.clip((cy[known] * POSITION_BINS).astype(np.int64), 0, POSITION_BINS - 1)
        np.add.at(pos, (leaves[known], rec[known], by, bx), 1)

    join = True if scoring is None else bool(scoring.join)
    blocks = None if scoring is None else scoring.member_block_start
    runs = leaf_runs(places, leaves, join=join, block_start=blocks)
    run_cx, run_cy = run_positions(
        places, leaves, positions, join=join, block_start=blocks
    )
    sums = run_sums(places, leaves, n_leaves, scoring, join=join, block_start=blocks)
    return LeafTables(
        recording_ids=tuple(places.recording_ids),
        leaf_time_ns=leaf_time,
        leaf_members=leaf_members,
        leaf_rec_time_ns=leaf_rec_time,
        leaf_rec_pos=pos,
        run_recording=runs[0],
        run_subject=runs[1],
        run_start_ns=runs[2],
        run_end_ns=runs[3],
        run_leaf=runs[4],
        separation=np.asarray(separations, dtype=np.float32),
        run_cx=run_cx,
        run_cy=run_cy,
        **sums,
    )


def vector_basis(sample: np.ndarray, dims: int = VECTOR_DIMS) -> np.ndarray:
    """``(K, D)`` the leading right singular vectors of ``sample``, ``K = min(dims, D, S)``.

    From the eigenvectors of the ``(D, D)`` Gram matrix rather than an SVD of the sample, which
    would build an ``(S, D)`` left factor nothing needs (2.4 GB at 400,000 x 768). The same
    subspace and, vector by vector, the same directions up to sign, which every score here is
    invariant to: a clip and its node's centroid are projected onto the same basis.
    """
    sample = np.asarray(sample)
    gram = np.asarray(sample.T @ sample, dtype=np.float64)
    values, vectors = np.linalg.eigh(gram)
    keep = min(int(dims), sample.shape[1], max(sample.shape[0], 1))
    return vectors[:, ::-1][:, :keep].T.copy()


def run_sums(
    places,
    leaves: np.ndarray,
    n_leaves: int,
    scoring,
    *,
    join: bool = True,
    block_start: np.ndarray | None = None,
):
    """Per leaf run: its member count, and the figures a clip's typicality is summed from.

    ``{"run_rows", "run_last_ns", "run_dist_sum", "run_vec_sum", "leaf_vec_sum"}``, the last
    three ``None``
    unless ``scoring`` is of their kind. A ``centroid`` method's vectors are projected onto the
    leading ``VECTOR_DIMS`` right singular vectors of its ``vector_sample`` and summed block by
    block, so a study's vectors are never all in memory; a run never crosses a block, because a
    block is one unit's rows and a run never crosses a unit.
    """
    leaves = np.asarray(leaves, dtype=np.int64)
    out = {
        "run_rows": np.zeros(0, dtype=np.int32),
        "run_last_ns": np.zeros(0, dtype=np.int64),
        "run_dist_sum": None,
        "run_vec_sum": None,
        "leaf_vec_sum": None,
    }
    if not len(leaves):
        return out
    order, heads, tails, sorted_ = _run_bounds(
        places, leaves, join=join, block_start=block_start
    )
    out["run_rows"] = np.diff(np.append(heads, len(leaves))).astype(np.int32)
    out["run_last_ns"] = np.asarray(sorted_[4][tails], dtype=np.int64)
    kind = getattr(scoring, "kind", "none")
    if kind == "distance":
        distance = np.asarray(scoring.member_distance, dtype=np.float64)
        if len(distance) != len(leaves):
            raise ValueError(
                f"{len(distance):,} member distances for {len(leaves):,} members"
            )
        out["run_dist_sum"] = np.add.reduceat(distance[order], heads)
    elif kind == "centroid":
        basis = vector_basis(scoring.vector_sample)
        marks = np.zeros(len(leaves), dtype=np.int64)
        marks[heads] = 1
        run_of_member = np.empty(len(leaves), dtype=np.int64)
        run_of_member[order] = np.cumsum(marks) - 1
        sums = np.zeros((len(heads), basis.shape[0]), dtype=np.float64)
        covered = 0
        for lo, hi, vectors in scoring.member_vectors:
            runs = run_of_member[int(lo) : int(hi)]
            projected = np.asarray(vectors, dtype=np.float64) @ basis.T
            by_run = np.argsort(runs, kind="stable")
            ordered = runs[by_run]
            starts = np.concatenate([[0], np.flatnonzero(np.diff(ordered)) + 1])
            sums[ordered[starts]] += np.add.reduceat(projected[by_run], starts, axis=0)
            covered += int(hi) - int(lo)
        if covered != len(leaves):
            raise ValueError(
                f"member vectors covered {covered:,} of {len(leaves):,} members"
            )
        leaf_sums = np.zeros((int(n_leaves), basis.shape[0]), dtype=np.float64)
        np.add.at(leaf_sums, leaves[order][heads], sums)
        out["run_vec_sum"] = sums.astype(np.float32)
        out["leaf_vec_sum"] = leaf_sums
    return out


def covered_end(start_ns: np.ndarray, time_ns: np.ndarray) -> np.ndarray:
    """Where a member's covered time ends: its **start** plus its time (``MemberPlaces``).

    ``time_ns`` is the whole of what a member stands for, its span included: a frame's gap to the
    next frame, a bout's span plus one step. Adding it to the member's *end* counts a bout's span
    twice (a 0.37 s bout ran to 0.4 s past its last frame, over the next bout); for a frame the two
    agree, since its start is its end.
    """
    return np.asarray(start_ns, dtype=np.int64) + np.asarray(time_ns, dtype=np.int64)


def _run_bounds(
    places,
    leaves: np.ndarray,
    *,
    join: bool = True,
    block_start: np.ndarray | None = None,
):
    """The members sorted by unit and start, and where each maximal run of one leaf begins and
    ends in that order: ``(order, heads, tails, sorted columns)``. Shared by ``leaf_runs`` and
    ``run_positions``, so a run's position is over exactly the members its span covers.
    Without ``join`` every member is its own run (a bout method: two bouts never fuse).
    ``block_start`` marks the members that begin one of the method's continuity blocks; a run
    never crosses one, so a run is inside exactly the stretch the method's instances are cut
    from, whatever gap the method breaks at.
    """
    leaves = np.asarray(leaves, dtype=np.int64)
    rec = np.asarray(places.recording_index, dtype=np.int64)
    sub = np.asarray(places.subject, dtype=np.int64)
    start = np.asarray(places.start_ns, dtype=np.int64)
    end = np.asarray(places.end_ns, dtype=np.int64)
    time = np.asarray(places.time_ns, dtype=np.int64)
    order = np.lexsort((start, sub, rec))
    leaves, rec, sub, start, end, time = (
        a[order] for a in (leaves, rec, sub, start, end, time)
    )
    joins = (
        (rec[1:] == rec[:-1])
        & (sub[1:] == sub[:-1])
        & (leaves[1:] == leaves[:-1])
        & (start[1:] <= covered_end(start[:-1], np.maximum(time[:-1], 1)))
    )
    if block_start is not None:
        joins &= ~np.asarray(block_start, dtype=bool)[order][1:]
    if not join:
        joins = np.zeros_like(joins)
    heads = np.concatenate([[0], np.flatnonzero(~joins) + 1])
    tails = np.concatenate([heads[1:] - 1, [len(leaves) - 1]])
    return order, heads, tails, (leaves, rec, sub, start, end, time)


def run_positions(
    places,
    leaves: np.ndarray,
    positions: tuple[np.ndarray, np.ndarray] | None,
    *,
    join: bool = True,
    block_start: np.ndarray | None = None,
):
    """``(cx, cy)`` per leaf run, ``float32``: the mean of its members' known positions.

    A clip's position: a clip is one run, a stretch of one leaf, so its mean says where in the
    frame that stretch happened. ``NaN`` where none of its members has a position (no
    kinematics, or a tracking gap the whole way).
    """
    leaves = np.asarray(leaves, dtype=np.int64)
    if not len(leaves):
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty.copy()
    order, heads, _tails, _sorted = _run_bounds(
        places, leaves, join=join, block_start=block_start
    )
    if positions is None:
        nan = np.full(len(heads), np.nan, dtype=np.float32)
        return nan, nan.copy()
    cx, cy = (np.asarray(values, dtype=np.float64)[order] for values in positions)
    known = np.isfinite(cx) & np.isfinite(cy)
    counts = np.add.reduceat(known.astype(np.int64), heads)
    out = []
    for v in (cx, cy):
        sums = np.add.reduceat(np.where(known, v, 0.0), heads)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(counts > 0, sums / counts, np.nan).astype(np.float32))
    return out[0], out[1]


def leaf_runs(
    places,
    leaves: np.ndarray,
    *,
    join: bool = True,
    block_start: np.ndarray | None = None,
):
    """Maximal runs of one leaf per unit, as ``(recording, subject, start_ns, end_ns, leaf)``.

    What the labels ethogram is drawn from: map each run's leaf through the current labels, merge
    neighbours that land on one label, and the ethogram follows every label the moment it lands,
    with no reprojection. Consecutive members join when they share a unit and a leaf and the next
    one starts within its predecessor's own time (a frame's time is its gap to the next frame
    inside its block, so a tracking gap always breaks a run). ``end_ns`` is where the last
    member's covered time ends (``covered_end``): for a frame, the frame plus its time; for a bout,
    its last frame plus one frame step. So a run covers exactly its members' summed time.
    """
    leaves = np.asarray(leaves, dtype=np.int64)
    if not len(leaves):
        empty = np.zeros(0, dtype=np.int64)
        return (empty.astype(np.int32), empty, empty, empty, empty.astype(np.int32))
    _order, heads, tails, sorted_ = _run_bounds(
        places, leaves, join=join, block_start=block_start
    )
    leaves, rec, sub, start, end, time = sorted_
    return (
        rec[heads].astype(np.int32),
        sub[heads],
        start[heads],
        covered_end(start[tails], time[tails]),
        leaves[heads].astype(np.int32),
    )


def node_leaf_index(tree) -> list[np.ndarray]:
    """``leaves_under`` for every node at once, built bottom-up in one pass over the merges."""
    out: list[np.ndarray] = [
        np.array([leaf], dtype=np.int64) for leaf in range(tree.n_leaves)
    ]
    for a, b in np.asarray(tree.children, dtype=np.int64):
        out.append(np.sort(np.concatenate([out[int(a)], out[int(b)]])))
    return out


def member_positions(places, kinematics_by_unit: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(cx_norm, cy_norm)`` per member: the mean over its span, ``NaN`` where none is known.

    ``analysis/positions.instance_positions`` for members rather than instances, vectorised with
    running sums so a pose run's hundreds of thousands of frames cost one ``searchsorted`` each.
    A frame member's span is its own frame. ``kinematics_by_unit`` maps ``(recording_id,
    subject)`` to ``{t_ns, cx_norm, cy_norm}`` as ``api/run_preview.load_track_kinematics``
    returns them.
    """
    rec = np.asarray(places.recording_index, dtype=np.int64)
    sub = np.asarray(places.subject, dtype=np.int64)
    start = np.asarray(places.start_ns, dtype=np.int64)
    end = np.asarray(places.end_ns, dtype=np.int64)
    cx = np.full(len(rec), np.nan)
    cy = np.full(len(rec), np.nan)
    for (recording_id, subject), columns in kinematics_by_unit.items():
        if recording_id not in places.recording_ids:
            continue
        mask = (rec == places.recording_ids.index(recording_id)) & (sub == subject)
        if not mask.any():
            continue
        t = np.asarray(columns["t_ns"], dtype=np.int64)
        order = np.argsort(t, kind="stable")
        t = t[order]
        x = np.asarray(columns["cx_norm"], dtype=np.float64)[order]
        y = np.asarray(columns["cy_norm"], dtype=np.float64)[order]
        valid = np.isfinite(x) & np.isfinite(y)
        sums_x = np.concatenate([[0.0], np.cumsum(np.where(valid, x, 0.0))])
        sums_y = np.concatenate([[0.0], np.cumsum(np.where(valid, y, 0.0))])
        counts = np.concatenate([[0], np.cumsum(valid)])
        lo = np.searchsorted(t, start[mask], side="left")
        hi = np.searchsorted(t, end[mask], side="right")
        n = counts[hi] - counts[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            cx[mask] = np.where(n > 0, (sums_x[hi] - sums_x[lo]) / n, np.nan)
            cy[mask] = np.where(n > 0, (sums_y[hi] - sums_y[lo]) / n, np.nan)
    return cx, cy
