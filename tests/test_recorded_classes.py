"""Every guard the rulebook cites is a file that exists, holding the name it names.

**The class.** A pointer into a tree that moves, checked by nobody. The *Recorded classes* table
is the project's index of bugs that must not come back: each row names the guard
enforcing it, and that name is the only route from the rule to the code. When a test file is
renamed or split, the row keeps naming the old one — and a citation is not an import, so nothing
breaks, no run turns red, and the row goes on reading exactly as it did when it was true.

**How it hid.** `tests/gui/test_descriptor_panel.py` was deleted in `618aeb9` and its coverage
moved into `test_measurements.py`. The rule stayed enforced the whole time; only the pointer
died. That is the quiet version, and the dangerous one is its twin: a row whose guard was
deleted *without* a replacement reads identically. From the table you cannot tell a moved guard
from a removed one, which is the whole reason the table is worth checking.

**The rule.** A citation resolves — the path is a real file, and a ``::name`` suffix is a real
test in it. Template placeholders (`tests/{package}/test_{module}.py`) are prose, not citations,
and are skipped by shape rather than by name, so a new one is skipped the day it is written.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent

# A citation names a file; anything with a brace is prose showing the layout, and a bare
# directory is a pointer at the tree rather than into it.
_CITATION = re.compile(r"`(tests/[^`]+)`")

#: The files whose ``tests/…`` backticks are citations. The *Recorded classes* table moved to
#: ``docs/recorded_classes.md`` when it reached a quarter of a file that is loaded into every
#: session; this follows it rather than narrowing, because the class — a pointer into a tree that
#: moves — is the same wherever the pointer is written. `CLAUDE.md` stays in the list: it cites
#: guards outside the table too, and dropping it would leave those unchecked.
_SOURCES = ("CLAUDE.md", "docs/recorded_classes.md")


def _cited_in(name: str) -> set[str]:
    """The citations one file makes."""
    text = (REPO / name).read_text(encoding="utf-8-sig")
    real = {c for c in _CITATION.findall(text) if c.endswith(".py") or "::" in c}
    return {c for c in real if "{" not in c}


def _citations() -> set[str]:
    return set().union(*(_cited_in(name) for name in _SOURCES))


def _row_guards() -> set[str]:
    """Citations from the **Guard column** of `docs/recorded_classes.md`'s table, only.

    These are the files that owe a class a write-up. Every other citation in the tree names a guard
    for some other reason — a fixture module, an idiom to copy — and owes nothing.
    """
    text = (REPO / "docs" / "recorded_classes.md").read_text(encoding="utf-8-sig")
    guards: set[str] = set()
    for line in text.split("\n"):
        if not line.startswith("| ") or line.startswith(("| ---", "| Rule")):
            continue
        cells = line.strip().strip("|").split("|")
        if len(cells) < 2:
            continue
        # The Guard column by position, not "the last one": a third column records when each
        # guard was last shown able to fail, and its evidence may cite tests too.
        guards |= {c for c in _CITATION.findall(cells[1]) if "{" not in c}
    return guards


def test_every_cited_guard_exists() -> None:
    dead = []
    for citation in sorted(_citations()):
        path, _, name = citation.partition("::")
        source = REPO / path
        if not source.is_file():
            dead.append(f"{citation} — no such file")
            continue
        if name and f"def {name}" not in source.read_text(encoding="utf-8-sig"):
            dead.append(f"{citation} — {path} holds no `{name}`")
    assert not dead, (
        "the rulebook cites guards that are not there. A row in *Recorded classes* is the only "
        "route from a rule to the code enforcing it, and a stale one reads exactly like a live "
        "one — so repoint it at the guard that covers the class now, or say the class is "
        "unguarded:\n  " + "\n  ".join(dead)
    )


def test_there_are_citations_to_check() -> None:
    """The floor is on the number actually checked, so the sweep cannot pass over an empty set.

    A regex that stops matching — the table reformatted, the backticks dropped — makes the check
    above vacuously true, and this file is the one place where that failure would be silent.
    """
    assert len(_citations()) >= 40


def test_every_source_contributes_citations() -> None:
    """Both files are really read, and both are really still in the list.

    The table alone clears the floor by a wide margin, so once it moved out of `CLAUDE.md` a reader
    that stopped opening one of the two files would still pass `>= 40`. That is this file's own
    class one level up: a check true for a reason other than the one it claims.

    **Two failure modes, and iterating `_SOURCES` sees only one of them.** "A listed file
    contributed nothing" is caught by the loop; "a file was dropped from the list" is not, because
    the thing to complain about is no longer in the list being walked. The first version of this
    test passed with `CLAUDE.md` removed from `_SOURCES` — checked by running that mutation, which
    is the only way that kind of hole shows up. So both names are asserted, not derived.
    """
    required = {"CLAUDE.md", "docs/recorded_classes.md"}
    assert required <= set(_SOURCES), (
        f"{sorted(required - set(_SOURCES))} dropped out of the citation sources. The rulebook "
        "cites guards outside the table and the table holds the rest, so reading one without the "
        "other leaves real citations unchecked while the floor goes on passing."
    )
    empty = [name for name in _SOURCES if not _cited_in(name)]
    assert not empty, (
        f"{empty} contributed no citations, so the sweep is not really reading them — the floor "
        "is being cleared by the other file alone"
    )


def test_every_cited_guard_carries_a_module_docstring() -> None:
    """Step 6 puts the canonical write-up in the guard, and the index holds a rule and a pointer.

    Backwards is the failure: the write-up lives only in `docs/recorded_classes.md` and the file that
    enforces the class says nothing about it. Nothing breaks — the row reads well, the guard passes —
    but the next person to open that test has no idea what it is for, and the *one* copy of the
    reasoning is in a document that cannot fail.

    Six guards were in that state when this was written. Three had a one-line docstring
    (`tests/gui/test_busy.py`, which three rows cite, `tests/gui/test_workers.py`, and
    `tests/gui/conftest.py`) and three carried a docstring about something else entirely
    (`tests/test_cli.py`, `tests/storage/test_behavior_output.py`, and the since-retired
    `tests/gui/test_pedigree.py`).

    Presence, not length: "long enough" is a proxy, and this repo has been bitten by proxies. A
    docstring that exists and is wrong is a different problem, and reading the pair is the only thing
    that finds it — `_SUBSTANTIVE` is only a floor against a bare one-liner standing in for a
    write-up.

    **Scoped to the rows' own Guard column, not to every citation.** The first version read
    `_citations()`, which also holds the guards `CLAUDE.md` names in passing — and immediately flagged
    `tests/conftest.py`, cited there as "the shared fixtures live here". That file owes nobody a
    bug-class write-up, and demanding one would have been a guard asserting the wrong property about
    the wrong files.
    """
    import ast

    #: A module docstring shorter than this is a label, not a write-up of a bug class. Set just above
    #: the longest one-liner that was standing in for one when this guard was written (76 chars).
    _SUBSTANTIVE = 120

    thin = []
    for citation in sorted(_row_guards()):
        path = citation.partition("::")[0]
        source = REPO / path
        if not source.is_file():
            continue  # test_every_cited_guard_exists owns that failure
        try:
            doc = ast.get_docstring(ast.parse(source.read_text(encoding="utf-8-sig")))
        except SyntaxError:  # pragma: no cover - a broken test file fails elsewhere
            continue
        if doc is None:
            thin.append(f"{path} — no module docstring")
        elif len(doc) < _SUBSTANTIVE:
            thin.append(f"{path} — {len(doc)} chars, a label rather than a write-up")
    assert not thin, (
        "these guards are cited by a recorded class whose write-up is not in them. The guard's "
        "module docstring is the canonical copy, because it cannot drift from the thing enforcing "
        "the rule; the index row is a rule and a pointer:\n  " + "\n  ".join(thin)
    )


def test_the_docstring_check_would_catch_a_label() -> None:
    """Guard the guard: a one-line docstring must not pass as a write-up."""
    import ast

    label = ast.get_docstring(ast.parse('"""Shared GUI test fixtures."""\n'))
    assert label is not None and len(label) < 120, (
        "the floor no longer rejects the one-liner that was standing in for a write-up, so it would "
        "not reject a new one either"
    )


def test_a_dead_citation_would_be_caught() -> None:
    """Guard the guard: the shape the sweep exists to reject."""
    path, _, name = "tests/gui/test_descriptor_panel.py".partition("::")
    assert not (REPO / path).is_file(), "the deleted file came back; pick another"


def test_the_row_reader_reads_the_guard_column() -> None:
    """Guard the guard: the table grew a third column, and a reader keyed on "the last cell" then
    reads the evidence instead — which cites a few tests, so the docstring check would go on
    passing over a handful of files while skipping every real guard."""
    guards = _row_guards()
    assert "tests/gui/test_frame_source.py" in guards, (
        "the first row's guard is not among the row guards, so the reader is not reading the "
        "Guard column"
    )
    assert len(guards) >= 60, f"only {len(guards)} row guards read — the wrong column"
