"""Measure rows over a clustering run's node cache — the node source of a comparison (§24).

The §19 artifact collapses unlabelled nodes by design, and the difference screen (§22) exists to
find the nodes nobody has labelled yet. So this reads the run's **node cache** instead: the
leaf-additive tables, where every node's time per recording is one index-and-sum, and the leaf
runs, where a node's bouts are its runs merged wherever adjacent. No frame view, no video, seconds
rather than minutes — and the rows come out in ``measures.COLUMNS``' own schema, one ``label`` per
candidate node (``node:<id>``), so ``compare.observations``, the summaries, the tables and the
figures consume them unchanged.

Four measures, from the registry so their units and names render as every other measure's do:
``time_fraction`` (the ranking family), ``rate``, ``total_time``, ``duration_mean``. The
denominator is the recording's analysed time — the root's member time in that recording — which is
what makes a share of time mean the same thing across recordings of different lengths.

A **candidate** is a node with at least ``candidate_floor_s`` of member time over the run, the
root excluded: a smaller cluster is too small to be judged from its clips, every candidate widens
the family the floor is taken over, and the root is every recording's whole time. The cache has no
floor of its own since every cluster has clips (§22), so the comparison declares this one. The hierarchy travels beside the rows (:func:`hierarchy_rows`) so a
ranking can be suppressed over it.

Pure over a ``NodeView``: numpy only, no disk, no Qt.
"""

from collections.abc import Callable, Sequence

import numpy as np

from detail.analysis import measures as measures_lib

NODE_PREFIX = "node:"
NS_PER_S = 1_000_000_000

#: The measures the node source emits, in registry order; ``time_fraction`` is the ranking family.
MEASURES = ("time_fraction", "rate", "total_time", "duration_mean")
RANKING_MEASURE = "time_fraction"

HIERARCHY_COLUMNS = ["label", "node", "parent", "depth", "ancestors", "time_s", "share"]

#: Nodes per progress report. A row build is seconds, which is long enough to want a bar.
_REPORT_EVERY = 64


def node_label(node: int) -> str:
    return f"{NODE_PREFIX}{int(node)}"


def node_of(label: str) -> int | None:
    """The node id a label names, or ``None`` for a label that is not a node's."""
    if not isinstance(label, str) or not label.startswith(NODE_PREFIX):
        return None
    try:
        return int(label[len(NODE_PREFIX) :])
    except ValueError:
        return None


def candidate_nodes(view, floor_s: float) -> np.ndarray:
    """Every node with at least ``floor_s`` of member time, the root excluded, ascending."""
    times = view.node_times()
    floor_ns = float(floor_s) * NS_PER_S
    nodes = np.flatnonzero(times >= floor_ns)
    return nodes[nodes != view.root].astype(np.int64)


def _row(recording_id, label, window, measure_id, value, denom, n) -> dict:
    row = {
        "recording_id": recording_id,
        "subject": "",
        "label": label,
        "label_to": "",
        # A node's figures count its whole subtree, never a remainder (#24): nothing to rename.
        "label_display": label,
        "label_is_remainder": False,
        "window_id": window[0],
        "window_start_t_ns": window[1],
        "window_end_t_ns": window[2],
        "measure": measure_id,
        "value": "" if value is None else value,
        "denominator_ns": "" if denom is None else denom,
        "n": n,
    }
    assert (
        list(row) == measures_lib.COLUMNS
    ), "the node source writes the measures schema"
    return row


def _bout_starts(view, nodes: np.ndarray, n_recordings: int) -> np.ndarray:
    """``(len(nodes), R)`` bouts per node and recording: the node's leaf runs, merged where one
    starts no later than the previous one ends within the same unit."""
    units, start, end, leaf = view.unit_runs()
    n_runs = len(leaf)
    rec_of_run = np.zeros(n_runs, dtype=np.int64)
    unit_id = np.zeros(n_runs, dtype=np.int64)
    for k, ((rec, _sub), (lo, hi)) in enumerate(units.items()):
        rec_of_run[lo:hi] = rec
        unit_id[lo:hi] = k
    same_unit = np.r_[False, unit_id[1:] == unit_id[:-1]]
    touching = np.r_[False, start[1:] <= end[:-1]]
    n_leaves = view.tree.n_leaves
    bouts = np.zeros((len(nodes), n_recordings), dtype=np.int64)
    for i, node in enumerate(nodes):
        under = np.zeros(n_leaves, dtype=bool)
        under[view.leaves_under(int(node))] = True
        member = under[leaf]
        previous = np.r_[False, member[:-1]]
        starts = member & ~(previous & same_unit & touching)
        bouts[i] = np.bincount(rec_of_run[starts], minlength=n_recordings)
    return bouts


def _recording_spans(view, n_recordings: int) -> list[tuple[int, int]]:
    """Each recording's ``[first run start, last run end)``, ``(0, 0)`` where it has no runs."""
    units, start, end, _leaf = view.unit_runs()
    spans = [(None, None)] * n_recordings
    for (rec, _sub), (lo, hi) in units.items():
        if hi <= lo:
            continue
        lo_ns, hi_ns = int(start[lo:hi].min()), int(end[lo:hi].max())
        have = spans[rec]
        spans[rec] = (
            lo_ns if have[0] is None else min(have[0], lo_ns),
            hi_ns if have[1] is None else max(have[1], hi_ns),
        )
    return [(0, 0) if s[0] is None else s for s in spans]


def node_rows(
    view,
    *,
    floor_s: float,
    measures: Sequence[str] = MEASURES,
    recordings: Sequence[str] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Tidy measure rows for every candidate node, per recording, whole-recording window.

    ``measures`` must be drawn from :data:`MEASURES`. ``recordings`` restricts the recordings
    measured; unset measures every recording in the cache. ``progress_cb(done, total)`` reports
    per block of nodes and ``stop_check()`` is polled at the same boundary, returning the rows
    built so far — the pass is seconds on a real run, and a silent pass reads as a hang.
    """
    unknown = sorted(set(measures) - set(MEASURES))
    if unknown:
        raise measures_lib.MeasureError(
            f"the node source cannot measure {unknown}; it offers {list(MEASURES)}"
        )
    recording_ids = list(view.recording_ids)
    keep = None if recordings is None else set(recordings)
    columns = [r for r, rid in enumerate(recording_ids) if keep is None or rid in keep]
    nodes = candidate_nodes(view, floor_s)
    total = len(nodes)
    rec_time = view.leaf_recording_time_ns().astype(np.float64)
    denominator = rec_time.sum(axis=0)
    spans = _recording_spans(view, len(recording_ids))
    bouts = _bout_starts(view, nodes, len(recording_ids))

    rows: list[dict] = []
    for i, node in enumerate(nodes):
        if i % _REPORT_EVERY == 0:
            if stop_check is not None and stop_check():
                break
            if progress_cb is not None:
                progress_cb(i, total)
        label = node_label(int(node))
        time_ns = rec_time[view.leaves_under(int(node))].sum(axis=0)
        for r in columns:
            denom = float(denominator[r])
            if denom <= 0:
                # No analysed time in this recording is no measurement, not a zero share.
                continue
            window = ("whole", spans[r][0], spans[r][1])
            t = float(time_ns[r])
            n = int(bouts[i, r])
            for measure_id in measures:
                if measure_id == "time_fraction":
                    value = t / denom
                elif measure_id == "rate":
                    value = n / (denom / NS_PER_S)
                elif measure_id == "total_time":
                    value = t
                else:  # duration_mean: declined where there is no bout to have a duration
                    value = t / n if n else None
                rows.append(
                    _row(recording_ids[r], label, window, measure_id, value, denom, n)
                )
    if progress_cb is not None:
        progress_cb(total, total)
    return rows


def hierarchy_rows(view, floor_s: float) -> list[dict]:
    """One row per candidate node: its label, parent, depth, the candidate ancestors above it,
    its member time and share — what ``nodes.csv`` holds and nesting suppression reads.
    """
    nodes = candidate_nodes(view, floor_s)
    candidates = set(int(n) for n in nodes)
    times = view.node_times()
    run_time = float(times[view.root]) if len(times) else 0.0
    rows = []
    for node in nodes:
        node = int(node)
        above = [a for a in view.ancestors(node) if a in candidates]
        rows.append(
            {
                "label": node_label(node),
                "node": node,
                "parent": int(view.summary(node).parent),
                "depth": len(view.ancestors(node)),
                "ancestors": "/".join(node_label(a) for a in above),
                "time_s": float(times[node]) / NS_PER_S,
                "share": float(times[node]) / run_time if run_time else 0.0,
            }
        )
    return rows


def ancestors_from_rows(rows: Sequence[dict]) -> dict[str, set[str]]:
    """``{label: {ancestor labels}}`` from hierarchy rows, as written or as read back."""
    out: dict[str, set[str]] = {}
    for row in rows:
        above = str(row.get("ancestors") or "")
        out[str(row["label"])] = set(above.split("/")) - {""} if above else set()
    return out
