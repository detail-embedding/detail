"""The query-evaluation result tables sort their numeric columns by value (#35).

Rank, the clip span, Folds and the per-fold counts were built as `QStandardItem(str(x))`, so
on a sortable table Rank ordered 1, 10, 2 and the span column put `10:05` before `2:05`. The
package rule `a_numeric_cell_sorts_on_its_value` did not see those spellings (it read only
f-strings with a numeric spec and `str()` around a counting call); it reads them now, and
these tests drive the real tables a user clicks.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QTableView  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402

S = 1_000_000_000


def _metric(behavior, n_folds):
    from detail.storage import query_eval_csv as qc

    return qc.MetricRow(
        "emb_01", behavior, "state", 1, "min", 0.5, 0.5, 0.5, 1, 0, None, None,
        3, 4, 3, 1, 0, 0.8, 0.75, 1.0, 0.85, 0.1, n_folds=n_folds, ap_std=0.05,
    )  # fmt: skip


def _query_eval_run(tmp_path):
    """A query-eval run whose numbers sort differently as text and as values: Folds 2/10/3,
    fold indices 0–11, and three clips ranked 10, 2, 1 at 2:05, 10:05 and 0:05."""
    from detail.storage import analysis_runs as ar
    from detail.storage import query_eval_csv as qc

    proj = Experiment.create("p", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_01")
    adir.mkdir(parents=True, exist_ok=True)
    behaviors = (("groom", 2), ("rear", 10), ("sniff", 3))
    qc.write_metrics_csv(adir / "metrics.csv", [_metric(b, n) for b, n in behaviors])
    qc.write_folds_csv(
        adir / "folds.csv",
        [
            qc.FoldMetricRow("groom", f, 12 - f, 4, 3, 1, f, 0.8, 0.75, 1.0, 0.85, 0.1)
            for f in range(12)
        ],
    )
    qc.write_predictions_csv(
        adir / "predictions.csv",
        [
            qc.PredictionRow(
                "groom", "s1", 0, rank, "tp", 0.5, start * S, (start + 1) * S, 0.8,
                None, start * S, (start + 1) * S,
            )
            for rank, start in ((10, 125), (2, 605), (1, 5))
        ],
    )  # fmt: skip
    qc.write_ground_truth_csv(
        adir / "ground_truth.csv",
        [qc.GroundTruthRow("groom", "s1", 0, 5 * S, 6 * S, "scored", "matched")],
    )
    for f in range(12):
        qc.write_pr_points(
            adir / "pr" / "groom" / f"fold_{f:02d}.csv", [(0.5, 1.0), (1.0, 0.75)]
        )
    ar.write_analysis_yaml(
        adir,
        {
            "id": "an_01",
            "module_id": "query_eval",
            "recording_ids": ["recording_001"],
            "status": "complete",
            "embed_run_id": "emb_01",
            "track_id": 0,
            "params": {"fusion": "min", "band_s": 0.5, "iou_threshold": 0.5,
                       "n_exemplars": 1, "seed": 0, "folds": 2},
            "behaviors": [
                {"behavior": b, "behavior_type": "state", "n_recordings": 1,
                 "n_folds": n, "skipped": None}
                for b, n in behaviors
            ]
            + [{"behavior": "dig", "behavior_type": "state", "n_recordings": 0,
                "n_folds": 0, "skipped": "no eligible recordings"}],
            "artifacts": [
                {"path": "metrics.csv", "kind": "query_eval_metrics", "label": "metrics"}
            ],
        },
    )  # fmt: skip
    return proj


def _table(widget, first_header: str) -> QTableView:
    return next(
        t
        for t in widget.findChildren(QTableView)
        if t.model().horizontalHeaderItem(0)
        and t.model().horizontalHeaderItem(0).text() == first_header
    )


def _column(table: QTableView, header: str, order=Qt.SortOrder.AscendingOrder):
    """Click-sort ``table`` on ``header`` as a user would, and read that column top to bottom."""
    model = table.model()
    col = next(
        c
        for c in range(model.columnCount())
        if model.horizontalHeaderItem(c).text() == header
    )
    table.sortByColumn(col, order)
    return [model.index(r, col).data() for r in range(model.rowCount())]


def test_folds_and_the_per_fold_counts_sort_by_value(qapp, tmp_path):
    from detail.gui.pages import AnalysisRunDetailPage

    page = AnalysisRunDetailPage(_query_eval_run(tmp_path), "an_01")
    metrics = _table(page, "Behavior")
    # The skipped behavior's blank sorts below every scored one rather than among them.
    assert _column(metrics, "Folds") == ["—", "2", "3", "10"]

    folds = _table(page, "Fold")
    assert _column(folds, "Fold") == [str(f) for f in range(12)]
    # n_gt runs 12 down to 1 by fold, so as text it would start "1", "10", "11", "12".
    assert _column(folds, "n_gt") == [str(n) for n in range(1, 13)]


def test_the_clip_table_sorts_rank_and_time_by_value(qapp, tmp_path):
    from detail.api.run_preview import QueryEvalPreview
    from detail.gui.modules.analysis_query_eval import _QueryEvalExploreDialog

    proj = _query_eval_run(tmp_path)
    by_behavior = {
        m.behavior: m for m in QueryEvalPreview(proj.dir).read_metrics("an_01")
    }
    dialog = _QueryEvalExploreDialog(
        proj, proj.analysis_record("an_01"), by_behavior, "groom"
    )
    clips = _table(dialog, "Rank")
    assert _column(clips, "Rank") == ["1", "2", "10"]
    starts = [span.split("–")[0] for span in _column(clips, "Span")]
    assert starts == ["0:05.000", "2:05.000", "10:05.000"]
