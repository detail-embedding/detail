"""Behavior-query eval metrics CSV (§16)."""

from dataclasses import fields

import pytest

from detail.storage import query_eval_csv as q


def _row(**kw):
    data = dict(
        embed_run_id="emb_01",
        behavior="groom",
        behavior_type="state",
        n_recordings=3,
        fusion="min",
        band_s=0.5,
        iou_threshold=0.5,
        point_window_s=0.5,
        n_exemplars=1,
        seed=0,
        trajectory_split_gap_s=None,
        max_results=None,
        n_gt=10,
        n_pred=12,
        tp=8,
        fp=4,
        fn=2,
        ap=0.83,
        precision=0.66,
        recall=0.8,
        best_f1=0.75,
        best_f1_threshold=0.12,
    )
    data.update(kw)
    return q.MetricRow(**data)


def test_columns_match_dataclass():
    assert q.COLUMNS[0] == "embed_run_id"
    assert {f.name for f in fields(q.MetricRow)} == set(q.COLUMNS)


def test_all_row_dataclasses_match_their_columns():
    # every row schema's fields == its COLUMNS list (a write/read never drops a field)
    assert {f.name for f in fields(q.PredictionRow)} == set(q.PREDICTION_COLUMNS)
    assert {f.name for f in fields(q.GroundTruthRow)} == set(q.GROUND_TRUTH_COLUMNS)
    assert {f.name for f in fields(q.FoldMetricRow)} == set(q.FOLD_COLUMNS)


def test_write_read_roundtrip(tmp_path):
    p = tmp_path / "m.csv"
    q.write_metrics_csv(p, [_row(), _row(behavior="rear")])
    rows = q.read_metrics_csv(p)
    assert [r.behavior for r in rows] == ["groom", "rear"]
    # empty cells (None) round-trip; typed columns come back int/float
    assert rows[0].trajectory_split_gap_s is None and rows[0].max_results is None
    assert rows[0].tp == 8 and rows[0].ap == 0.83


def test_append_grows_the_table(tmp_path):
    p = tmp_path / "m.csv"
    q.append_metrics(p, [_row()])
    q.append_metrics(p, [_row(behavior="rear")])
    assert [r.behavior for r in q.read_metrics_csv(p)] == ["groom", "rear"]


def test_write_is_atomic(tmp_path):
    p = tmp_path / "m.csv"
    q.write_metrics_csv(p, [_row()])
    assert not (p.parent / (p.name + ".tmp")).exists()


def test_read_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("embed_run_id,behavior\nemb_01,groom\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        q.read_metrics_csv(p)


def test_pr_points_written(tmp_path):
    p = tmp_path / "pr.csv"
    q.write_pr_points(p, [(0.5, 1.0), (1.0, 0.66)])
    assert p.read_text(encoding="utf-8-sig").splitlines()[0] == "recall,precision"


def test_read_pr_points_roundtrip(tmp_path):
    p = tmp_path / "pr.csv"
    q.write_pr_points(p, [(0.5, 1.0), (1.0, 0.66)])
    assert q.read_pr_points(p) == [(0.5, 1.0), (1.0, 0.66)]


def test_predictions_roundtrip_and_nullable(tmp_path):
    p = tmp_path / "predictions.csv"
    rows = [
        q.PredictionRow("groom", "s1", 0, 1, "tp", 0.31, 100, 200, 0.71, None, 90, 210),
        q.PredictionRow(
            "groom", "s1", 0, 2, "fp", 0.33, 300, 400, None, None, None, None
        ),
    ]
    q.write_predictions_csv(p, rows)
    back = q.read_predictions_csv(p)
    assert [r.outcome for r in back] == ["tp", "fp"]
    assert back[0].iou == 0.71 and back[0].matched_gt_start_t_ns == 90
    assert back[1].iou is None and back[1].matched_gt_start_t_ns is None  # "" -> None
    assert not (p.parent / (p.name + ".tmp")).exists()  # atomic (no temp left)


def test_ground_truth_roundtrip_keeps_empty_outcome(tmp_path):
    p = tmp_path / "ground_truth.csv"
    rows = [
        q.GroundTruthRow("groom", "s1", 0, 100, 200, "scored", "matched"),
        q.GroundTruthRow("groom", "s1", 0, 300, 400, "scored", "fn"),
        q.GroundTruthRow("groom", "s1", 0, 500, 600, "exemplar", ""),
    ]
    q.write_ground_truth_csv(p, rows)
    back = q.read_ground_truth_csv(p)
    # a plain str field's empty cell stays "" (not None — that distinction is the
    # exemplar / unscored marker)
    assert [r.outcome for r in back] == ["matched", "fn", ""]
    assert back[2].role == "exemplar"


def test_predictions_read_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("behavior,recording_id\ngroom,s1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        q.read_predictions_csv(p)


# The literal header a pre-folds sweep CSV was written with — spelled out rather than
# derived from COLUMNS so it stays a real historical file as the schema grows.
_PRE_FOLDS_HEADER = (
    "embed_run_id,behavior,behavior_type,n_recordings,fusion,band_s,iou_threshold,"
    "point_window_s,n_exemplars,seed,max_gap_s,max_results,n_gt,n_pred,tp,fp,fn,ap,"
    "precision,recall,best_f1,best_f1_threshold"
)
_PRE_FOLDS_ROW = (
    "emb_01,groom,state,3,min,0.5,0.5,0.5,1,0,1.25,,10,12,8,4,2,0.83,0.66,0.8,0.75,0.12"
)


def test_metrics_read_tolerates_pre_folds_csv(tmp_path):
    # An older sweep CSV lacks the cross-validation, matcher and null columns; they fall
    # back to defaults so an accumulated sweep keeps reading.
    p = tmp_path / "old.csv"
    p.write_text(_PRE_FOLDS_HEADER + "\n" + _PRE_FOLDS_ROW + "\n", encoding="utf-8")
    rows = q.read_metrics_csv(p)
    assert rows[0].ap == 0.83 and rows[0].n_folds == 1 and rows[0].ap_std == 0.0
    # A CSV predating the null columns has no null recorded → None (not a fabricated 0.0).
    assert rows[0].method == "dtw" and rows[0].ap_null is None


def test_metrics_read_keeps_a_renamed_columns_value(tmp_path):
    """`max_gap_s` became `trajectory_split_gap_s` once a second, opposite gap setting
    existed. An old file must keep its value, not silently revert to the default."""
    p = tmp_path / "old.csv"
    p.write_text(_PRE_FOLDS_HEADER + "\n" + _PRE_FOLDS_ROW + "\n", encoding="utf-8")
    assert q.read_metrics_csv(p)[0].trajectory_split_gap_s == 1.25


def test_overlap_csv_roundtrip(tmp_path):
    p = tmp_path / "overlap.csv"
    rows = [
        q.OverlapRow("groom", "state", 0.1, 10, 40, 8, 0.8, 0.3, 0.5, 0.2, 0.6, 0.15, 0.45),
        q.OverlapRow("groom", "state", 0.5, 10, 40, 3, 0.3, 0.1, 0.2, 0.075, 0.2, 0.05, 0.15),
    ]  # fmt: skip
    q.write_overlap_csv(p, rows)
    back = q.read_overlap_csv(p)
    assert back == rows
    assert back[0].ap_lift == 0.45 and back[1].iou == 0.5


def test_folds_csv_roundtrip(tmp_path):
    p = tmp_path / "folds.csv"
    rows = [
        q.FoldMetricRow("groom", 0, 3, 3, 3, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.1),
        q.FoldMetricRow("groom", 1, 3, 4, 2, 2, 1, 0.5, 0.5, 0.75, 0.6, 0.2),
    ]
    q.write_folds_csv(p, rows)
    back = q.read_folds_csv(p)
    assert [r.fold for r in back] == [0, 1] and back[1].ap == 0.5
