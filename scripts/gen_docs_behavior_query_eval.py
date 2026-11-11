"""mkdocs-gen-files — generate the Behavior-query eval page, figure and all, from code (§7).

The one eval page under the Evaluation section today; a second eval module gets its own sibling
page the same way (its metrics single-source + a gen script). The metric table comes from
``analysis.metrics_docs.describe_metrics`` (validated against ``EvalMetrics``); the matching
defaults come from ``QueryEvalConfig``; and the figure is rendered at build time by running the
**real** matcher (``analysis.retrieval``) and the **real** scoring (``analysis.query_eval``) on the
committed Querying-methods fixture. So a change to the matcher, the metrics, or a default re-renders
the page. The only prose is the hand-written ``behavior-query-eval-intro.md``.
"""

import io
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import mkdocs_gen_files

from detail.analysis import query_eval, retrieval
from detail.analysis.metrics_docs import describe_metrics
from detail.configs.describe import describe_config
from detail.configs.query_eval import QueryEvalConfig

_DOCS = Path(__file__).resolve().parent.parent / "docs"
_FIXTURE = (
    _DOCS / "assets" / "query-methods" / "fixture.npz"
)  # shared with Querying methods
_GEN = "assets/behavior-query-eval/generated"
_GREEN, _RED, _ORANGE = "#009E73", "#D55E00", "#E69F00"

_DEFAULTS = {
    name: field.default for name, field in QueryEvalConfig.model_fields.items()
}
_FIELDS = {field["name"]: field for field in describe_config(QueryEvalConfig)}


def _rows(seconds: float, cadence_ns: int) -> int:
    return max(1, round(seconds * 1e9 / cadence_ns))


def _default(name: str) -> str:
    field = _FIELDS[name]
    value = field["default"]
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    unit = f" {field['unit']}" if field["unit"] else ""
    return f"`{value}`{unit}"


_EXAMPLE_METHOD = (
    "dtw"  # a windowed matcher, to illustrate span→GT matching + the metrics
)


def _example_matches(window, query, cadence_ns):
    """A windowed matcher's predicted spans on the fixture, using the config default knobs."""
    matches = retrieval.query_multi(
        window,
        [query],
        fusion=_DEFAULTS["fusion"],
        band=_rows(_DEFAULTS["band_s"], cadence_ns),
        threshold=2.0,  # no score cutoff — keep the best few, misses included
        min_separation=len(query),
        max_results=None,
        method=_EXAMPLE_METHOD,
    )
    return [("ex", m.start_row, m.end_row, m.score) for m in matches]


def _render_figure() -> tuple[bytes, dict]:
    data = np.load(_FIXTURE, allow_pickle=False)
    window = data["window"].astype(np.float32)
    query = data["query"].astype(np.float32)
    intervals = [(int(a), int(b)) for a, b in data["intervals"]]
    cadence_ns = int(data["cadence_ns"])
    threshold = _DEFAULTS["iou_threshold"]

    preds = _example_matches(window, query, cadence_ns)
    results = query_eval.match_ranked_detailed(
        preds, {"ex": intervals}, mode="state", iou_threshold=threshold, window_ns=0
    )
    flags = [r.is_tp for r in results]
    n_gt = len(intervals)
    ap, _points = query_eval.average_precision(flags, n_gt)
    tp = sum(flags)
    fp = len(flags) - tp
    fn = n_gt - tp
    matched = {r.matched_gt for r in results if r.is_tp}
    stats = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / n_gt,
        "precision": tp / len(flags),
        "ap": ap,
    }

    fig, ax = plt.subplots(figsize=(11, 2.9))
    fig.subplots_adjust(top=0.66, bottom=0.22, left=0.11, right=0.98)
    for (_g, start, end, _s), result in zip(preds, results):
        ax.plot(
            [start, end],
            [1, 1],
            lw=11,
            solid_capstyle="butt",
            color=_GREEN if result.is_tp else _RED,
        )
        if result.is_tp:
            center = (start + end) / 2
            gt = intervals[result.matched_gt]
            ax.plot([center, sum(gt) / 2], [0.9, 0.1], color="#aaa", lw=0.8, ls=":")
    for j, (start, end) in enumerate(intervals):
        ax.plot([start, end], [0, 0], lw=11, solid_capstyle="butt", color=_ORANGE)
        if j not in matched:
            ax.text(
                (start + end) / 2,
                -0.34,
                "FN",
                color=_RED,
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
            )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        ["ground truth", _EXAMPLE_METHOD], fontfamily="monospace", fontsize=9
    )
    ax.set_ylim(-0.6, 1.5)
    ax.set_xlim(0, len(window))
    ax.set_xlabel("frame")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)
    axis = ax.get_xaxis_transform()
    ax.text(
        0,
        1.42,
        f"{_EXAMPLE_METHOD} predictions matched to ground truth at IoU ≥ {threshold:g}",
        transform=axis,
        fontsize=10.5,
        fontweight="bold",
    )
    ax.text(
        0,
        1.24,
        f"TP {tp}  ·  FP {fp}  ·  FN {fn}      recall {stats['recall']:.2f}  ·  "
        f"precision {stats['precision']:.2f}  ·  AP {ap:.3f}",
        transform=axis,
        fontsize=9.5,
        fontfamily="monospace",
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color=_GREEN, lw=6, label="true positive"),
            Line2D([0], [0], color=_RED, lw=6, label="false positive"),
            Line2D([0], [0], color=_ORANGE, lw=6, label="ground truth"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
        ncol=3,
        bbox_to_anchor=(1.0, 1.16),
    )
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    return buffer.getvalue(), stats


_png, _stats = _render_figure()
with mkdocs_gen_files.open(f"{_GEN}/matching.png", "wb") as handle:
    handle.write(_png)

_table = ["## The metrics", "", "| Metric | What it measures |", "| --- | --- |"]
for metric in describe_metrics():
    _table.append(f"| **{metric.label}** (`{metric.name}`) | {metric.measures} |")

_matching = [
    "## Matching",
    "",
    "A prediction is scored against the ground truth by a greedy, one-to-one match, best-scoring "
    "predictions first, so each ground-truth instance is matched at most once.",
    "",
    "- A **state** annotation (a behavior with duration) is matched by **temporal IoU**: a "
    f"prediction counts when its overlap with the annotation is at least `iou_threshold` (default {_default('iou_threshold')}).",
    "- A **point** annotation (an instant) is matched when a prediction's midpoint falls within "
    f"`point_window_s` (default {_default('point_window_s')}) of it.",
]

_caption = (
    f"`{_EXAMPLE_METHOD}`'s predicted spans matched to the ground-truth intervals on the example "
    "from [Querying methods](querying-methods.md), at the default IoU. Green is a true positive, red "
    "a false positive, orange the ground truth, and a missed interval is marked FN. Those counts "
    f"give the recall ({_stats['recall']:.2f}), precision ({_stats['precision']:.2f}), and AP "
    f"({_stats['ap']:.3f}) shown. This span-matching path is how the windowed matchers are scored; "
    "the per-frame classifiers integrate a threshold-free per-frame AP instead."
)

lines = [
    (_DOCS / "behavior-query-eval-intro.md").read_text(encoding="utf-8-sig").rstrip(),
    "",
    *_matching,
    "",
    *_table,
    "",
    "## On one example",
    "",
    f"![{_EXAMPLE_METHOD} predictions scored against ground truth]({_GEN}/matching.png)",
    "",
    f"*{_caption}*",
]

with mkdocs_gen_files.open("behavior-query-eval.md", "w") as handle:
    handle.write("\n".join(lines) + "\n")
