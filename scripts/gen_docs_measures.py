"""mkdocs-gen-files — generate the Measures reference page from the registry (§23).

Every column comes off ``MeasureSpec``, which already carries a ``label``, a ``description``, a
``unit``, a ``rule`` and its ``requires`` — so there is no second source to drift from the first,
the way ``metrics_docs`` needs one for ``EvalMetrics`` fields that have no text of their own. A
measure registered later appears here with no edit.

The hand-written prose is ``measures-intro.md`` (before the table).
"""

from pathlib import Path

import mkdocs_gen_files

from detail.analysis import location_measures
from detail.analysis.measures import CONDITIONS, MEASURES

_DOCS = Path(__file__).resolve().parent.parent / "docs"

#: How a `unit` reads to someone who did not write the registry.
_UNITS = {
    "ns": "duration",
    "1/s": "per second",
    "fraction": "share of time",
    "": "count",
    "ratio": "ratio (no unit)",
}

#: What each `rule` means for which instances a measure sees.
_RULES = {
    "onset": "counts a bout in the window that contains its start",
    "overlap": "counts the part of each bout that falls inside the window",
    "pair": "reads a bout together with its neighbour in the sequence, and counts it in the "
    "window where the first of the two starts",
}


def _rows() -> list[str]:
    out = []
    for measure_id, spec in MEASURES.items():
        needs = ", ".join(sorted(spec.requires)) if spec.requires else "—"
        out.append(
            f"| **{spec.label}** | `{measure_id}` | {_UNITS.get(spec.unit, spec.unit)} | "
            f"{spec.description} | {needs} |"
        )
    return out


lines = [(_DOCS / "measures-intro.md").read_text(encoding="utf-8-sig").rstrip(), ""]
lines += [
    "## The measures",
    "",
    "| Measure | Id | Unit | What it reports | Needs |",
    "| --- | --- | --- | --- | --- |",
    *_rows(),
    "",
    "The **id** is what `--measure` takes and what the `measure` column of `stats.csv` and "
    "`measures.csv` holds; the name is what every screen shows.",
    "",
    "**Transitions** answers per *ordered pair* — one row for each behavior that follows each "
    "other behavior, with the from in the `label` column and the to in `label_to`. A vocabulary "
    "of twenty behaviors is up to four hundred of those, so they are reported as rows and not "
    "drawn as four hundred figures; the table and `stats.csv` carry them all.",
    "",
    "## What a measure needs",
    "",
    "A measure that a behavior result cannot answer is greyed out where it is offered, with the "
    "reason attached, and produces no rows if you ask for it anyway — rather than a column of "
    "zeros that reads like a result.",
    "",
    "| Needs | Withheld when |",
    "| --- | --- |",
]
lines += [f"| `{name}` | {why} |" for name, (why, _check) in sorted(CONDITIONS.items())]
lines += [
    "",
    "## How a window is counted",
    "",
    "| Rule | Which bouts it sees |",
    "| --- | --- |",
]
lines += [f"| `{name}` | {what} |" for name, what in _RULES.items()]
lines += [
    "",
    "## Movement and body shape",
    "",
    "A [comparison](concepts.md#comparison) over a location-export run measures these "
    "instead, one value for each recording and animal. They are read from the run's "
    "per-frame tables, so they need a location export that measured the body and a distance "
    "calibration on every recording.",
    "",
    "| Measure | Id | Unit | What it reports |",
    "| --- | --- | --- | --- |",
]
lines += [
    f"| **{spec.label}** | `{measure_id}` | {_UNITS.get(spec.unit, spec.unit)} | "
    f"{spec.description} |"
    for measure_id, spec in location_measures.MEASURES.items()
]

with mkdocs_gen_files.open("measures.md", "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
