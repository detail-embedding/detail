"""A source scan without a negative case cannot be told from a scan that stopped matching.

**The class.** A guard that searches the source for a forbidden shape passes in two different
worlds: the package is clean, or the pattern no longer matches anything. Those look identical
from the outside — one green test — and the second is worth nothing while costing wall-clock on
every run. CLAUDE.md has said *"a scan that matches nothing today passes vacuously forever, so
add the negative case"* since the idiom was written, and the step was the one most often skipped.

**How it hid.** Because it is a property of the *test*, not of the package. Nothing in the suite
looked at the suite. Measured 2026-09-11, of 23 source-scanning guard files **11 had no negative
case at all** — among them `tests/storage/test_path_serialization.py`, which CLAUDE.md names as
"the idiom" and which four other guards were copied from. Fed the spellings a person would
actually write it caught `str(path.relative_to(base))` and missed an f-string, a `%`-format, a
call in the receiver, and a `str(` wrapped onto two lines. It had never been watched reject
anything, and neither had the four scans modelled on it.

**The rule** (CLAUDE.md, *Fixing a bug means fixing its class*, step 4): a scan ships with a test
that feeds it a violation and asserts it fails. The remaining files are listed below with why
not yet, the same claim-with-a-reason shape as `UNIT_EXEMPTIONS` and `requirements.NOT_YET_SWEPT`
— and the ceiling only ever comes down.
"""

import ast
import importlib.util
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("_tiers", REPO / "tests" / "tiers.py")
tiers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiers)

#: A test function whose name says it plants a violation. Matched on the *name* because that is
#: what a reader scanning the file sees, and because a negative case that is not obviously one
#: does not serve the person who next doubts the guard.
_NEGATIVE_CASE = re.compile(
    r"would_catch|would_fail|would_flag|would_notice|catches|rejects|detects"
    r"|counterexample|negative|not_vacuous",
    re.I,
)

#: Source-scanning guards with no negative case, and why not yet. **An entry is a claim**: a file
#: that has since grown one, or been deleted, fails rather than sitting here excusing nothing.
SCANS_WITHOUT_A_NEGATIVE_CASE: dict[str, str] = {
    "tests/analysis/test_no_unfiltered_behavior_streams.py": "label-leakage scan; needs a fixture that plants a prediction read as ground truth",
    "tests/api/test_wrapper_signatures.py": "compares signatures rather than matching a pattern — a negative case means a hand-written divergent wrapper",
    "tests/gui/test_empty_states.py": "data-driven over index pages; a negative case means a page whose notice names no action",
    "tests/gui/test_spin_widgets.py": "scans for bare QSpinBox; a negative case means a planted one in a built form",
    "tests/gui/test_window_state.py": "scans for save/restoreGeometry; a negative case means a package with the calls removed",
    "tests/pipeline/test_producer_parity.py": "registry parity rather than a source pattern",
    "tests/test_console.py": "scans for encoding-unsafe output; a negative case means a planted glyph under a narrow locale",
    "tests/test_docs_map.py": "checks docs structure; a negative case means a planted orphan page",
}

#: The ratchet. Lowered when a scan grows a negative case; never raised. A literal, not
#: ``len(...)`` — derived from the dict it caps, the assertion is true by construction and could
#: never fail, which is the mistake ``requirements.BACKLOG_CEILING`` documents.
CEILING = 8


def _source_scanning_guards() -> dict[str, list[str]]:
    """Guard-tier files that read the source tree, mapped to their test function names."""
    found = {}
    for rel in sorted(set(tiers.GUARD_FILES)):
        path = REPO / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "ast.parse" not in text and 'rglob("*.py")' not in text:
            continue
        found[rel] = [
            node.name
            for node in ast.walk(ast.parse(text))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    return found


def test_every_source_scan_has_a_negative_case_or_is_declared():
    scans = _source_scanning_guards()
    undeclared = sorted(
        rel
        for rel, names in scans.items()
        if not any(_NEGATIVE_CASE.search(n) for n in names)
        and rel not in SCANS_WITHOUT_A_NEGATIVE_CASE
    )
    assert not undeclared, (
        f"source scans with no negative case and no entry: {undeclared}. A scan that matches "
        f"nothing passes vacuously forever, and nothing else in the suite can tell that apart "
        f"from a clean package. Add a test that plants a violation and watch it fail, or add an "
        f"entry to SCANS_WITHOUT_A_NEGATIVE_CASE saying why not yet."
    )


def test_no_declaration_excuses_nothing():
    """An entry for a file that has grown a negative case — or been deleted — is a stale claim.

    The same shape as `test_test_tiers`'s check on `FILE_TIER`, and the reason nine dialog
    exemptions rotted before anything compared them to reality.
    """
    scans = _source_scanning_guards()
    stale = []
    for rel in SCANS_WITHOUT_A_NEGATIVE_CASE:
        if rel not in scans:
            stale.append(f"{rel} (no longer a source-scanning guard, or deleted)")
        elif any(_NEGATIVE_CASE.search(n) for n in scans[rel]):
            stale.append(
                f"{rel} (now has a negative case — delete the entry, lower CEILING)"
            )
    assert not stale, "\n".join(stale)


def test_the_backlog_only_shrinks():
    assert len(SCANS_WITHOUT_A_NEGATIVE_CASE) <= CEILING, (
        f"{len(SCANS_WITHOUT_A_NEGATIVE_CASE)} scans lack a negative case against a ceiling of "
        f"{CEILING}. The ceiling is a ratchet: give the new scan a negative case rather than "
        f"raising it."
    )


def test_there_are_source_scans_to_check():
    """Guard the guard — this file is itself a scan, and inherits the failure it exists to name.

    If `_source_scanning_guards` stopped recognising a scan (the marker strings change, the tier
    table moves) it would return nothing and every assertion above would pass over an empty set.
    """
    scans = _source_scanning_guards()
    assert (
        len(scans) >= 20
    ), f"only {len(scans)} source-scanning guards found; the walk broke"
    with_case = [
        r for r, n in scans.items() if any(_NEGATIVE_CASE.search(x) for x in n)
    ]
    assert len(with_case) >= 10, (
        f"only {len(with_case)} scans read as having a negative case; the name pattern has "
        f"probably stopped matching rather than the suite having regressed"
    )
