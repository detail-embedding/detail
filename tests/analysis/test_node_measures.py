"""The node source of a comparison is leaf-additive (§24 *The node source*).

Every row the node source emits must equal the figure computed directly over the node's leaves
and runs — the guard §22 keeps over the node cache, reused over these rows. A row that drifted
from the cache would be a plausible share of time for a cluster that spends its time elsewhere,
and nothing downstream (the observations, the ranking, the floor) could tell.

Built on a synthetic ``NodeView`` — four leaves, three merges, two recordings — so each number
is checkable by hand, and so the API's node-source tests can patch the same view in place of a
fitted run.
"""

import numpy as np
import pytest

from detail.analysis import measures as measures_lib
from detail.analysis import node_cache as tables_lib
from detail.analysis import node_measures as nm
from detail.storage import node_cache as store

NS = 1_000_000_000


def synthetic_view(recording_ids=("r1", "r2")):
    """Four leaves merged (0,1)->4, (2,3)->5, (4,5)->6, over the given recordings.

    Recording 0 holds leaf 0 [0, 10 s) touching leaf 1 [10, 20 s) — one bout under node 4 —
    and leaf 2 [30, 40 s). Recording 1 holds leaf 0 twice with a gap, [0, 5 s) and [7, 9 s),
    and leaf 3 [9, 12 s). Every later recording repeats recording 1's runs with leaf 0 scaled.
    """
    from detail.api.node_view import NodeView
    from detail.core.dataclasses import ClusterTree

    tree = ClusterTree(
        children=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
        height=np.array([1.0, 2.0, 3.0]),
        size=np.array([1, 1, 1, 1, 2, 2, 4], dtype=np.int32),
        n_rows=np.array([1, 1, 1, 1, 2, 2, 4], dtype=np.int32),
        leaf_of_row=np.array([0, 1, 2, 3], dtype=np.int32),
        rung_merges=(0,),
    )
    R = len(recording_ids)
    rec_time = np.zeros((4, R), dtype=np.int64)
    rec_time[:, 0] = [10 * NS, 10 * NS, 10 * NS, 0]
    for r in range(1, R):
        rec_time[:, r] = [7 * NS * r, 0, 0, 3 * NS]
    run_recording, run_start, run_end, run_leaf = (
        [0, 0, 0],
        [0, 10, 30],
        [10, 20, 40],
        [0, 1, 2],
    )
    for r in range(1, R):
        run_recording += [r, r, r]
        run_start += [0, 7, 9]
        run_end += [5, 9, 12]
        run_leaf += [0, 0, 3]
    tables = tables_lib.LeafTables(
        recording_ids=tuple(recording_ids),
        leaf_time_ns=rec_time.sum(axis=1),
        leaf_members=(rec_time.sum(axis=1) // NS).astype(np.int64),
        leaf_rec_time_ns=rec_time,
        leaf_rec_pos=np.zeros((4, R, 16, 16), dtype=np.uint32),
        run_recording=np.array(run_recording, dtype=np.int32),
        run_subject=np.zeros(len(run_leaf), dtype=np.int64),
        run_start_ns=np.array(run_start, dtype=np.int64) * NS,
        run_end_ns=np.array(run_end, dtype=np.int64) * NS,
        run_leaf=np.array(run_leaf, dtype=np.int32),
        separation=np.array([np.nan] * 4 + [0.4, 0.3, 0.2], dtype=np.float32),
    )
    cache = store.NodeCache(leaves=tables, built_from={"fit": "fit.npz", "fit_size": 1})
    return NodeView(cache, tree)


def _by(rows, label, recording, measure):
    (row,) = [
        r
        for r in rows
        if r["label"] == label
        and r["recording_id"] == recording
        and r["measure"] == measure
    ]
    return row


def test_rows_follow_the_measures_schema():
    rows = nm.node_rows(synthetic_view(), floor_s=1.0)
    assert rows and all(list(row) == measures_lib.COLUMNS for row in rows)
    assert {row["window_id"] for row in rows} == {"whole"}
    assert {row["measure"] for row in rows} == set(nm.MEASURES)


def test_every_row_is_the_sum_over_the_nodes_leaves():
    view = synthetic_view()
    rows = nm.node_rows(view, floor_s=1.0)
    rec_time = view.leaf_recording_time_ns()
    for node in nm.candidate_nodes(view, 1.0):
        for r, recording in enumerate(view.recording_ids):
            direct = rec_time[view.leaves_under(int(node)), r].sum()
            denominator = rec_time[:, r].sum()
            label = nm.node_label(int(node))
            assert _by(rows, label, recording, "total_time")["value"] == direct
            assert _by(rows, label, recording, "time_fraction")[
                "value"
            ] == pytest.approx(direct / denominator)
            assert (
                _by(rows, label, recording, "total_time")["denominator_ns"]
                == denominator
            )


def test_adjacent_runs_under_one_node_are_one_bout():
    """Leaf 0 then leaf 1, touching, is one bout of node 4 and one bout each of its leaves; a
    gap between two runs of one leaf keeps them two bouts."""
    rows = nm.node_rows(synthetic_view(), floor_s=1.0)
    assert _by(rows, "node:4", "r1", "rate")["n"] == 1
    assert _by(rows, "node:0", "r1", "rate")["n"] == 1
    assert _by(rows, "node:1", "r1", "rate")["n"] == 1
    assert _by(rows, "node:4", "r2", "rate")["n"] == 2
    assert _by(rows, "node:0", "r2", "rate")["n"] == 2
    # 2 bouts over 10 s of analysed time in r2 (7 s of leaf 0 and 3 s of leaf 3)
    assert _by(rows, "node:4", "r2", "rate")["value"] == pytest.approx(2 / 10)
    assert _by(rows, "node:4", "r2", "duration_mean")["value"] == pytest.approx(
        7 * NS / 2
    )


def test_a_node_with_no_bout_declines_a_duration_rather_than_reporting_zero():
    rows = nm.node_rows(synthetic_view(), floor_s=1.0)
    assert _by(rows, "node:1", "r2", "duration_mean")["value"] == ""
    assert _by(rows, "node:1", "r2", "total_time")["value"] == 0


def test_the_root_is_excluded_and_the_floor_applies():
    view = synthetic_view()
    assert list(nm.candidate_nodes(view, 1.0)) == [0, 1, 2, 3, 4, 5]
    assert list(nm.candidate_nodes(view, 15.0)) == [0, 4]
    rows = nm.node_rows(view, floor_s=15.0)
    assert {row["label"] for row in rows} == {"node:0", "node:4"}


def test_a_recording_filter_measures_only_those_recordings():
    rows = nm.node_rows(synthetic_view(), floor_s=1.0, recordings=["r2"])
    assert {row["recording_id"] for row in rows} == {"r2"}


def test_an_unknown_measure_is_refused_naming_the_four():
    with pytest.raises(measures_lib.MeasureError, match="time_fraction"):
        nm.node_rows(synthetic_view(), floor_s=1.0, measures=["transitions"])


def test_hierarchy_rows_name_candidate_ancestors_only():
    view = synthetic_view()
    rows = nm.hierarchy_rows(view, 1.0)
    assert [list(row) for row in rows] == [nm.HIERARCHY_COLUMNS] * len(rows)
    by_label = {row["label"]: row for row in rows}
    assert by_label["node:0"]["ancestors"] == "node:4"  # the root is no candidate
    assert by_label["node:4"]["ancestors"] == ""
    assert by_label["node:0"]["parent"] == 4 and by_label["node:4"]["parent"] == 6
    ancestors = nm.ancestors_from_rows(rows)
    assert ancestors["node:0"] == {"node:4"} and ancestors["node:4"] == set()
    # Read back through text, as a CSV reader hands it over.
    assert nm.ancestors_from_rows([{"label": "x", "ancestors": "a/b"}]) == {
        "x": {"a", "b"}
    }


def test_node_labels_round_trip_and_everything_else_is_not_a_node():
    assert nm.node_of(nm.node_label(1192)) == 1192
    assert nm.node_of("rearing") is None and nm.node_of("node:x") is None


def test_progress_is_reported_and_a_stop_stops():
    view = synthetic_view()
    seen = []
    nm.node_rows(
        view, floor_s=1.0, progress_cb=lambda done, total: seen.append((done, total))
    )
    assert seen[0] == (0, 6) and seen[-1] == (6, 6)
    assert nm.node_rows(view, floor_s=1.0, stop_check=lambda: True) == []
