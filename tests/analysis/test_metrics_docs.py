"""Anti-drift for the Metrics doc page (scripts/gen_docs_metrics.py).

The page's metric table comes from ``analysis.metrics_docs``, and its matching rule quotes config
defaults. These tests pin the described metrics to the ``EvalMetrics`` dataclass and the matching
knobs to ``QueryEvalConfig``, so a renamed field breaks here rather than shipping a stale page.
"""

from dataclasses import fields

from detail.analysis.metrics_docs import describe_metrics
from detail.analysis.query_eval import EvalMetrics
from detail.configs.query_eval import QueryEvalConfig


def test_metrics_are_real_evalmetrics_fields():
    names = {f.name for f in fields(EvalMetrics)}
    docs = describe_metrics()
    assert docs
    for doc in docs:
        assert doc.name in names, doc.name
        assert doc.label and doc.measures


def test_headline_scores_are_documented():
    documented = {doc.name for doc in describe_metrics()}
    assert {"ap", "recall", "precision", "best_f1"} <= documented


def test_matching_knobs_are_config_fields():
    for name in ("iou_threshold", "point_window_s"):
        assert name in QueryEvalConfig.model_fields
