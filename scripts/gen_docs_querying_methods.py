"""mkdocs-gen-files — generate the Querying methods page, figure and all, from code (§7).

The method table comes from ``analysis.query_methods_docs.describe_query_methods`` (the single
source), and the figure is rendered at build time by running the **real matchers**
(``detail.analysis.retrieval``) on a committed fixture: one real trajectory slice, one real
example clip, and the behavior intervals (``docs/assets/query-methods/fixture.npz``). The knob
values come from ``QueryEvalConfig`` defaults, so the picture shows what the tool does out of the
box, and a matcher or default change re-renders it. The method order is read from the config's
``method`` field, so a new method appears in the table and figure with no edit here. The only prose
is the hand-written ``querying-methods-intro.md`` (before) and ``querying-methods-controls.md``
(after).
"""

import io
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mkdocs_gen_files

from detail.analysis import retrieval
from detail.analysis.query_methods_docs import (
    describe_query_controls,
    describe_query_methods,
)
from detail.configs.query_eval import QueryEvalConfig

_DOCS = Path(__file__).resolve().parent.parent / "docs"
_FIXTURE = _DOCS / "assets" / "query-methods" / "fixture.npz"
_GEN = "assets/query-methods/generated"  # site path for the rendered figure
_ORANGE = "#E69F00"

_DEFAULTS = {
    name: field.default for name, field in QueryEvalConfig.model_fields.items()
}


def _rows(seconds: float, cadence_ns: int) -> int:
    return max(1, round(seconds * 1e9 / cadence_ns))


def _spans_for(method: str, window, query, cadence_ns) -> list[tuple[int, int]]:
    """The spans one windowed matcher returns on the fixture, using the config's defaults."""
    params = (
        {
            "keyframe_select": _DEFAULTS["keyframe_select"],
            "n_keyframes": _DEFAULTS["n_keyframes"],
        }
        if method == "keyframe"
        else {}
    )
    matches = retrieval.query_multi(
        window,
        [query],
        fusion=_DEFAULTS["fusion"],
        band=_rows(_DEFAULTS["band_s"], cadence_ns),
        threshold=2.0,  # no score cutoff — keep each method's best few, misses included
        min_separation=len(query),
        max_results=6,
        method=method,
        method_params=params,
    )
    return [(m.start_row, m.end_row) for m in matches[:4]]


def _render_figure() -> bytes:
    data = np.load(_FIXTURE, allow_pickle=False)
    window = data["window"].astype(np.float32)
    query = data["query"].astype(np.float32)
    intervals = [(int(a), int(b)) for a, b in data["intervals"]]
    cadence_ns = int(data["cadence_ns"])
    score, _ = retrieval.frame_scores(window, [query], fusion=_DEFAULTS["fusion"])
    # Only the windowed matchers return spans; the per-frame classifiers score every frame
    # (that is the top similarity panel), so they get no lane in the spans figure.
    method_ids = [
        doc.method_id
        for doc in describe_query_methods()
        if doc.method_id in retrieval._CURVE_FNS
    ]
    lanes = [(m, _spans_for(m, window, query, cadence_ns)) for m in method_ids]

    x = np.arange(len(window))
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 4.6), height_ratios=[2, 3.2], sharex=True
    )
    for lo, hi in intervals:
        top.axvspan(lo, hi, color=_ORANGE, alpha=0.28, lw=0)
    top.plot(x, score, color="#0072B2", lw=0.9)
    top.set_ylabel("similarity")
    top.set_yticks([0, 0.3, 0.6])
    top.set_title("Per-frame similarity to the example clip", fontsize=9.5, loc="left")

    for lo, hi in intervals:
        bottom.axvspan(lo, hi, color=_ORANGE, alpha=0.28, lw=0)
    for i, (_, spans) in enumerate(lanes):
        for lo, hi in spans:
            bottom.plot([lo, hi], [i, i], lw=9, solid_capstyle="butt", color="#333333")
    bottom.set_yticks(range(len(lanes)))
    bottom.set_yticklabels([m for m, _ in lanes], fontfamily="monospace", fontsize=9)
    bottom.set_ylim(-0.6, len(lanes) - 0.4)
    bottom.invert_yaxis()
    bottom.set_xlabel("frame")
    bottom.set_title(
        "Spans each method returns   (orange band = actual behavior)",
        fontsize=9.5,
        loc="left",
    )
    for ax in (top, bottom):
        ax.margins(x=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    return buffer.getvalue()


with mkdocs_gen_files.open(f"{_GEN}/methods.png", "wb") as handle:
    handle.write(_render_figure())

_caption = (
    "Each windowed matcher run on the same real recording slice against one example clip; the "
    "orange band is the actual behavior (climb wall). They return roughly constant-length windows "
    "near the example's duration; some matches land off the behavior, which the methods do return. "
    "The per-frame classifier (soft_knn) returns no spans — its view is the per-frame similarity in "
    "the top panel."
)

_methods = describe_query_methods()

_table = [
    "## Methods",
    "",
    "| Method | What it does | Match length |",
    "| --- | --- | --- |",
]
for method in _methods:
    _table.append(
        f"| `{method.method_id}` | {method.summary} | {method.match_length} |"
    )

_how = ["## How each method works", ""]
for method in _methods:
    _how += [f"### `{method.method_id}`", "", method.description, ""]

_controls = [
    "## Controls",
    "",
    "Defaults come from the config; full bounds and help are in the "
    "[Parameters](reference/parameters.md) reference.",
    "",
    "| Control | Default | Applies to | Sets |",
    "| --- | --- | --- | --- |",
]
for control in describe_query_controls():
    _controls.append(
        f"| `{control.name}` | {control.default} | {control.applies_to} | {control.sets} |"
    )

lines = [
    (_DOCS / "querying-methods-intro.md").read_text(encoding="utf-8-sig").rstrip(),
    "",
    *_table,
    "",
    "## What each method returns",
    "",
    f"![Spans each querying method returns]({_GEN}/methods.png)",
    "",
    f"*{_caption}*",
    "",
    *_how,
    *_controls,
]

with mkdocs_gen_files.open("querying-methods.md", "w", encoding="utf-8-sig") as handle:
    handle.write("\n".join(lines) + "\n")
