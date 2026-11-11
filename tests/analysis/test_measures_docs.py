"""The measures reference page is generated from the registry, and covers all of it.

**The class.** A reference table hand-written beside a registry is complete on the day it is
written and not afterwards. §23 has eleven measures, each already carrying a `label`, a
`description`, a `unit`, a `rule` and its `requires` — everything a page needs — so the generator
reads those rather than a second list. What can still drift is the *translation* tables the
generator owns: a measure registered with a new `unit` or a new `rule` would render its raw
registry token to a reader, or vanish from a section. That is what this pins.

`scripts/gen_docs_measures.py` is the page; `docs/measures-intro.md` is the only hand-written
prose in it.
"""

import pathlib
import re

from detail.analysis.measures import CONDITIONS, MEASURES

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "gen_docs_measures.py"
)


def _tables() -> tuple[dict, dict]:
    """The generator's `_UNITS` and `_RULES`, without importing mkdocs_gen_files."""
    namespace: dict = {}
    source = _SCRIPT.read_text(encoding="utf-8-sig")
    body = source[source.index("_UNITS = {") : source.index("def _rows(")]
    exec(body, namespace)
    return namespace["_UNITS"], namespace["_RULES"]


def test_every_unit_in_the_registry_reads_as_english():
    """A measure registered with a unit the page cannot name renders `1/s` at a reader."""
    units, _rules = _tables()
    unknown = sorted({spec.unit for spec in MEASURES.values()} - set(units))

    assert (
        not unknown
    ), f"units with no reader-facing name on the measures page: {unknown}"


def test_every_rule_in_the_registry_is_explained():
    """The window-counting section is the part a reader needs to interpret a windowed number."""
    _units, rules = _tables()
    unknown = sorted({spec.rule for spec in MEASURES.values()} - set(rules))

    assert not unknown, f"window rules the measures page does not explain: {unknown}"


def test_every_declared_condition_is_on_the_page():
    """`requires` names a key of `CONDITIONS`, and the page's second table is that dict — so
    this holds by construction. Asserted anyway, because the construction is one line away from
    becoming a hand-written list."""
    declared = {c for spec in MEASURES.values() for c in spec.requires}

    assert declared <= set(CONDITIONS), sorted(declared - set(CONDITIONS))


def test_the_page_is_generated_rather_than_written():
    """Guard the guard: the checks above are about a *generator*. A hand-written `docs/measures.md`
    would leave them passing while the page went stale — the exact failure they exist to prevent.
    """
    docs = pathlib.Path(__file__).resolve().parents[2] / "docs"

    assert not (
        docs / "measures.md"
    ).exists(), (
        "docs/measures.md is checked in, so it no longer comes from the registry"
    )
    assert (docs / "measures-intro.md").is_file()
    assert "MEASURES" in _SCRIPT.read_text(encoding="utf-8-sig")


def test_the_generator_reads_every_measure():
    """No filter between the registry and the table."""
    source = _SCRIPT.read_text(encoding="utf-8-sig")

    assert re.search(
        r"for measure_id, spec in MEASURES\.items\(\):", source
    ), "the generator no longer walks the whole registry"
    assert len(MEASURES) >= 11, f"only {len(MEASURES)} measures; the walk covers little"
