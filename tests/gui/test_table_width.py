"""A table fills its pane or hugs its rows — never leaves dead grid between.

The class: **Qt's default sizes columns to their content and says nothing about the
remainder.** A table that neither stretches nor is capped leaves the leftover blank, and the
wider the window the worse it gets. Measured at 1400px across this package: ten tables trailed
between 60px and 1265px of empty grid on every row, with `MetadataPage`, `BenchmarkDetailPage`
and `NameSyllablesDialog` over a thousand pixels each.

The rule was already written down (`gui_design` §3) and already had a helper. What it did not
have was a sweep: it was applied at the one site that reported it — `AuditTab`'s two summary
tables — and its only guard was hard-coded to those two, so the twenty-four raw `QTableView` /
`QTableWidget` constructions that bypass the helpers were never looked at.

Every table therefore does one of three things, and this scan makes a new one say which:
stretch its last column (`tables.fill_width`), hand one column the slack
(`tables.give_slack_to`), cap itself to its content (`fit_table_height` / `fit_table_width`),
or come from `pages._table`, which already stretches.
"""

import ast
import pathlib

import pytest

_GUI = pathlib.Path(__file__).resolve().parents[2] / "detail" / "gui"

#: Constructions that legitimately do neither, with the reason. A table inside a fixed-width
#: column has no leftover to spread, and one whose columns the user drags is theirs to size.
EXEMPT = {
    "gui/pages.py:_table": "the helper itself — it stretches its last data column",
    # `requirements.py`'s counterexamples are widgets built to be *wrong* — each one is the
    # proof that a structural requirement can fail. A scan for non-conforming tables finding
    # them is the scan working; they are never shown to anyone.
    "gui/requirements.py:_an_editable_table": "a counterexample — deliberately non-conforming",
    "gui/requirements.py:_a_table_nobody_sized": (
        "a counterexample — a table nobody called `fit_table_height` on, which is the whole "
        "point of it"
    ),
    "gui/requirements.py:_a_table_that_opens_nothing_but_looks_like_it_does": (
        "a counterexample — deliberately non-conforming"
    ),
    "gui/requirements.py:_a_navigable_table_with_no_cue": (
        "a counterexample — deliberately non-conforming"
    ),
}

_SATISFIED = (
    "fill_width(",
    "setStretchLastSection(True)",
    "give_slack_to(",
    "fit_table_width(",
    "fit_table_height(",
    "_fit_table_height(",
)


def _functions_building_tables():
    """Every function that constructs a table, with its source."""
    out = []
    for py in sorted(_GUI.rglob("*.py")):
        text = py.read_text(encoding="utf-8-sig")
        if "QTableView()" not in text and "QTableWidget(" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.get_source_segment(text, node) or ""
            if "QTableView()" in body or "QTableWidget(" in body:
                rel = py.relative_to(_GUI.parent).as_posix()
                out.append((f"{rel}:{node.name}", body))
    return out


def test_every_table_either_fills_its_pane_or_hugs_its_rows():
    builders = _functions_building_tables()
    assert len(builders) > 10, "the scan found almost nothing; it has stopped working"
    missing = [
        name
        for name, body in builders
        if name not in EXEMPT and not any(call in body for call in _SATISFIED)
    ]
    assert not missing, (
        "these build a table that neither stretches nor is capped, so it leaves dead grid "
        f"to the right of every row: {missing}. Call `tables.fill_width(table)` to spread it, "
        "or `fit_table_width` to hug its content."
    )


def test_the_scan_would_catch_a_bare_table():
    """Guard the guard: a scan that matches nothing passes vacuously forever."""
    body = "def build():\n    table = QTableView()\n    return table\n"
    assert "QTableView()" in body
    assert not any(call in body for call in _SATISFIED)


@pytest.mark.parametrize("name", sorted(EXEMPT))
def test_every_exemption_still_names_a_real_builder(name):
    """An exemption for a function that no longer builds a table is a claim about nothing."""
    assert name in {n for n, _ in _functions_building_tables()}
