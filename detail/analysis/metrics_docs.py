"""Single source for the Behavior-query eval doc page (§7).

One line per metric the behavior-query eval reports. The names are ``EvalMetrics`` fields, so
``describe_metrics`` raises if one drifts, and the anti-drift test pins the described set against
the dataclass. A second eval module documents its metrics the same way (its own metrics dataclass +
describe function + Evaluation-section page).
"""

from dataclasses import dataclass, fields

from detail.analysis.query_eval import EvalMetrics

# (EvalMetrics field, display label, what it measures). Order: scores first, then counts.
_METRICS = [
    (
        "ap",
        "AP",
        "area under the precision-recall curve over the ranked predictions; the headline "
        "retrieval score",
    ),
    ("recall", "Recall", "share of ground-truth instances that a prediction matched"),
    (
        "precision",
        "Precision",
        "share of predictions that matched a ground-truth instance",
    ),
    (
        "best_f1",
        "Best F1",
        "the best balance of precision and recall across the score threshold",
    ),
    ("tp", "TP", "predictions that matched a ground-truth instance"),
    ("fp", "FP", "predictions that matched none"),
    ("fn", "FN", "ground-truth instances that no prediction matched"),
]


@dataclass(frozen=True)
class MetricDoc:
    """One reported metric described for the docs page."""

    name: str  # EvalMetrics field name
    label: str
    measures: str


def describe_metrics() -> list[MetricDoc]:
    """Per-metric doc metadata; the names are validated against EvalMetrics."""
    valid = {f.name for f in fields(EvalMetrics)}
    docs: list[MetricDoc] = []
    for name, label, measures in _METRICS:
        if name not in valid:
            raise ValueError(
                f"metric {name!r} is not an EvalMetrics field "
                "(analysis.metrics_docs._METRICS is out of sync with query_eval)"
            )
        docs.append(MetricDoc(name=name, label=label, measures=measures))
    return docs
