"""The guard on ``tests/impact.py``.

**The class: a selection mechanism that silently stops selecting.** It is the same shape
``tests/test_test_tiers.py`` exists for, one table over. Nothing fails when a selector
narrows too far — the tests it dropped simply do not run, the command is green in less
time, and the run *looks* better than the one that would have caught the regression. The
only signature is that a file stopped appearing in a list nobody reads.

So the rules this asserts are the ones whose failure is invisible:

- the fallbacks really do fall back — an unrecognised path, a gate file and an empty
  change list each select **everything**, because each of them means "I do not know";
- the guards ride along, since they are what the heuristic is known to miss;
- a selection never names a file that is not collected, which would fail the run for a
  reason that has nothing to do with the change;
- and every hand-written ``COUPLED`` entry still earns its line, the standard
  ``tiers.UNIT_EXEMPTIONS`` is held to — an entry that adds nothing the derived rules
  already found is an entry nobody will question later.

Each test ends in one ``assert not offenders`` with the offenders named, so a failure says
what to delete rather than that a count moved.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_TIERS_MODULE = "detail_test_tiers"


def _load(name: str, filename: str):
    """Load a tests/ table by path, under the key every other reader uses."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, pathlib.Path(__file__).parent / filename
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


tiers = _load(_TIERS_MODULE, "tiers.py")
impact = _load("detail_test_impact", "impact.py")


@pytest.fixture(scope="module")
def collected() -> set[str]:
    """Every collected test file, repo-relative — what a selection may legally name."""
    return {tiers.relpath(path) for path in tiers.test_files()}


# --------------------------------------------------------------------------- #
# The fallbacks fall back.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "changed, why",
    [
        ([], "an empty change list knows nothing"),
        (["setup.cfg"], "a path no rule recognises"),
        (["tests/tiers.py"], "the tier table is the gate's own moving part"),
        (["tests/conftest.py"], "the hook that applies the tiers"),
        (["pyproject.toml"], "the marker declarations"),
        (["tests/impact.py"], "this selector itself"),
        ([".github/workflows/ci.yml"], "the workflow"),
        (["detail/gui/scrubber.py", "setup.cfg"], "one unknown path poisons the batch"),
    ],
)
def test_a_change_that_is_not_understood_selects_everything(changed, why):
    assert impact.select(changed) is None, (
        f"{changed} narrowed the run, but {why} — so the honest answer is the whole "
        "suite. A selector that guesses here is one that stops selecting silently."
    )


def test_a_helper_module_under_tests_is_not_narrowed_to_itself():
    """``tests/gui/subjects.py`` is somebody's fixture, and we cannot tell whose."""
    assert impact.select(["tests/gui/subjects.py"]) is None


# --------------------------------------------------------------------------- #
# What a selection contains.
# --------------------------------------------------------------------------- #


def test_every_selected_path_is_a_collected_test_file(collected):
    offenders = {}
    for changed in (
        ["detail/gui/scrubber.py"],
        ["detail/storage/video.py"],
        ["detail/api/recording.py"],
        ["tests/gui/test_scrubber.py"],
        ["docs/cli.md"],
    ):
        selected = impact.select(changed) or []
        stray = sorted(set(selected) - collected)
        if stray:
            offenders[str(changed)] = stray
    assert not offenders, (
        "a selection named a file pytest will not collect, which fails the run for a "
        "reason unrelated to the change:\n"
        + "\n".join(f"{k}: {v}" for k, v in offenders.items())
    )


def test_the_guards_ride_along_with_every_selection(collected):
    guards = {rel for rel in collected if tiers.is_guard(rel)}
    assert guards, "no guards found — the fixture, not the rule, is broken"
    missing = {}
    for changed in (
        ["detail/gui/scrubber.py"],
        ["docs/cli.md"],
        ["tests/gui/test_busy.py"],
    ):
        absent = guards - set(impact.select(changed) or [])
        if absent:
            missing[str(changed)] = sorted(absent)[:5]
    assert not missing, (
        "guards are what this heuristic is known to miss — measured over 400 commits, the "
        "co-edited files a mirror cannot see are guards. They run with every selection:\n"
        + "\n".join(f"{k}: {v}" for k, v in missing.items())
    )


def test_dropping_the_guards_is_possible_and_visibly_smaller():
    """The tight-loop escape hatch exists, and is not quietly the same thing."""
    with_guards = impact.select(["detail/gui/scrubber.py"])
    without = impact.select(["detail/gui/scrubber.py"], guards=False)
    assert without is not None and with_guards is not None
    assert set(without) < set(with_guards), "--no-guards selected the same set"


def test_a_source_change_selects_its_own_mirror():
    selected = impact.select(["detail/gui/scrubber.py"], guards=False) or []
    assert "tests/gui/test_scrubber.py" in selected


def test_the_mention_scan_sees_the_parent_import_form():
    """``from detail.analysis import behavior_labels`` never writes the dotted path.

    A substring scan for ``detail.analysis.behavior_labels`` therefore reads that file as
    unrelated. 650 (module, test file) pairs in this suite are coupled only this way, so
    the spelling is not an edge case — it is most of them.

    **The fixture is the whole test.** The first version of this asserted the same property
    about ``detail.gui.busy`` and passed against a scan with the parent form deleted,
    because ``tests/gui/test_busy.py`` happens to contain the dotted string too — both code
    paths found it, so the check could not fail. The `assert ... not in` below is what
    keeps that from happening again: it fails the day this file stops being a case where
    the two spellings diverge, rather than silently going vacuous.
    """
    subject = "tests/analysis/test_behavior_labels_stream.py"
    source = (REPO / subject).read_text(encoding="utf-8-sig")
    assert "detail.analysis.behavior_labels" not in source, (
        f"{subject} now spells the dotted path, so it no longer distinguishes a scan that "
        "handles the parent-import form from one that does not — pick another subject"
    )
    hits = impact.mentions("detail.analysis.behavior_labels")
    assert subject in hits, (
        "the parent-module import form was not matched, so the scan silently misses the "
        "650 couplings in this suite that are spelled that way"
    )


# --------------------------------------------------------------------------- #
# The table is claims, not decoration.
# --------------------------------------------------------------------------- #


def test_no_coupled_entry_points_at_a_missing_file(collected):
    offenders = []
    for source, extras in sorted(impact.COUPLED.items()):
        if not (REPO / source).is_file():
            offenders.append(f"{source}: the source file is gone")
        offenders.extend(
            f"{source} -> {extra}: not a collected test file"
            for extra in extras
            if extra not in collected
        )
    assert not offenders, (
        "a COUPLED entry naming something that no longer exists is a line nobody will "
        "question. Delete these:\n" + "\n".join(offenders)
    )


def test_every_coupled_entry_still_adds_something_the_rules_miss():
    """The ``UNIT_EXEMPTIONS`` standard: an entry that excuses nothing must fail.

    If the mirror and the mention scan already find everything a COUPLED entry lists, the
    entry is inert — and an inert entry is indistinguishable from a live one at a glance,
    so it survives every future edit of this table on the strength of looking deliberate.
    """
    inert = []
    for source, extras in sorted(impact.COUPLED.items()):
        if not (REPO / source).is_file():
            continue  # the missing-file test above owns this case
        derived = impact.mentions(impact.module_path(source))
        mirrored = impact.mirror(source)
        if mirrored:
            derived.add(mirrored)
        if not set(extras) - derived:
            inert.append(f"{source}: {list(extras)} are all found without it")
    assert not inert, (
        "these COUPLED entries add nothing the derived rules do not already select:\n"
        + "\n".join(inert)
    )


# --------------------------------------------------------------------------- #
# Guard the guard.
# --------------------------------------------------------------------------- #


def test_the_fallback_and_the_narrowing_cannot_both_be_degenerate():
    """The pair no single degenerate implementation can satisfy.

    Every *fallback* assertion above is satisfied by a `select` that returns ``None``
    unconditionally, and every *narrowing* assertion by one that always returns a list. So
    neither family proves anything on its own — asserted together, one of them fails for
    each of those two ways of being wrong. This is the check that makes the rest load
    bearing, and it is why the first draft of this file (which had only the fallbacks) was
    not enough.

    It also replaces a check that could not fail: emptying an ``EVERYTHING`` table and
    asserting a gate file then narrowed. It never did, because the structural rules catch
    those paths anyway — which is how that table was found to be excusing nothing, and
    deleted.
    """
    assert (
        impact.select(["pyproject.toml"]) is None
    ), "the fallback stopped falling back"
    narrowed = impact.select(["detail/gui/scrubber.py"])
    assert (
        narrowed is not None
    ), "nothing narrows any more — select always says everything"
    assert len(narrowed) < len(tiers.test_files()), (
        "a 'narrowed' selection that names every test file is the whole suite wearing a "
        "list, and every assertion in this module would pass against it"
    )


def test_the_coupled_check_would_catch_an_inert_entry(monkeypatch):
    """An entry listing a file the mirror already finds must be reported."""
    monkeypatch.setattr(
        impact,
        "COUPLED",
        {"detail/gui/scrubber.py": ("tests/gui/test_scrubber.py",)},
    )
    inert = []
    for source, extras in impact.COUPLED.items():
        derived = impact.mentions(impact.module_path(source))
        mirrored = impact.mirror(source)
        if mirrored:
            derived.add(mirrored)
        if not set(extras) - derived:
            inert.append(source)
    assert inert == ["detail/gui/scrubber.py"], (
        "the inert-entry check did not flag an entry naming only the mirror, so it would "
        "not flag a real one either"
    )


# --- the docstring's own evidence ------------------------------------------- #
_QUOTED_SELECTION = re.compile(r"(\d+)(?:\s+test files)? for ``(detail/[^`]+)``")


def _quoted_counts(doc: str) -> dict[str, int]:
    """The per-file selection counts a worked example states, keyed by source path.

    Pure, so the negative case can feed it the numbers that actually shipped.
    """
    return {source: int(count) for count, source in _QUOTED_SELECTION.findall(doc)}


def test_the_worked_example_in_the_docstring_is_still_true():
    """The four numbers `impact.py` quotes about itself are derived, not remembered.

    All four had drifted — 3/8/51 out of 361, against 4/5/19 out of 387 — and two had moved
    *down*, which rules out "the suite just grew". Nothing checked them, because a stale number
    in a docstring reads exactly like a fresh one. That docstring is where the heuristic's
    usefulness is quantified, so those numbers are the argument for it, and an argument built on
    four wrong numbers is not one.

    Exact rather than a tolerance band: each is one ``select`` call away, so there is no reason
    to accept any drift at all.
    """
    doc = impact.__doc__ or ""
    quoted = _quoted_counts(doc)
    assert len(quoted) == 3, (
        "the worked example no longer quotes three per-file selections, which is what this "
        f"guard reads as the argument for the heuristic; it found {sorted(quoted)}"
    )

    wrong = {
        source: (stated, len(impact.select({source}, guards=False)))
        for source, stated in quoted.items()
        if stated != len(impact.select({source}, guards=False))
    }
    assert not wrong, (
        f"the worked example's per-file counts have drifted (stated, actual): {wrong}. "
        "Each is one impact.select call away."
    )

    total = re.search(r"out of about (\d+)\.", doc)
    assert total is not None, (
        "the worked example no longer says roughly how many test files it is selecting out of, "
        "which is the number that makes the other three mean anything"
    )
    actual = len(tiers.test_files())
    assert _roughly(int(total.group(1)), actual), (
        f"the worked example says 'out of about {total.group(1)}' where the suite collects "
        f"{actual} test files — more than {_ROUGHLY:.0%} off, so it no longer says how big a "
        "slice the selections are"
    )


#: How far the docstring's suite size may drift before it stops conveying the scale.
#:
#: The three per-file counts are checked exactly, because they are the argument. The total was
#: checked exactly too, and that made every PR that adds or removes *any* test file edit this
#: one line: two open PRs both bump N to N+1, so the second either conflicts textually or lands
#: with the wrong count and reddens main. It happened twice in one day. The total only says
#: "a handful out of a few hundred"; a band keeps that honest without making it a merge hazard.
_ROUGHLY = 0.25


def _roughly(stated: int, actual: int) -> bool:
    return abs(stated - actual) <= _ROUGHLY * actual


def test_the_suite_size_band_would_catch_a_wrong_scale():
    """Guard the guard: the band accepts ordinary growth and rejects a size that misleads."""
    assert _roughly(400, 416), "a round figure near the real size must pass"
    assert _roughly(400, 460), "a few dozen new files must not force an edit"
    assert not _roughly(100, 416), "a quarter of the real size says a different thing"
    assert not _roughly(400, 900), "a suite that doubled has outgrown the sentence"


def test_the_worked_example_check_would_catch_a_stale_number():
    """Guard the guard, on the sentence that really shipped.

    Without this the check above passes in two worlds — the numbers are right, or
    `_QUOTED_SELECTION` stopped matching — and they are one green test from outside.
    """
    shipped = (
        "A mirror-plus-mention heuristic answers usefully for the same changes: 3 test files "
        "for ``detail/gui/scrubber.py``, 8 for ``detail/storage/video.py``, 51 for "
        "``detail/api/recording.py``, out of 361."
    )
    quoted = _quoted_counts(shipped)
    assert len(quoted) == 3, (
        "the reader must find all three selections in the shipped wording, or it is the reader "
        f"that is broken rather than the numbers; it found {sorted(quoted)}"
    )
    # The shipped counts drift back into truth by chance (video.py's 8 did, in the change
    # that made the map count shown frames), so staleness is made certain rather than
    # assumed: no live selection is negative.
    stale = {source: -stated for source, stated in quoted.items()}
    drifted = sorted(
        source
        for source, stated in stale.items()
        if stated != len(impact.select({source}, guards=False))
    )
    assert len(drifted) == 3, (
        "every one of the three counts is wrong, so a reader that flags fewer is not "
        f"comparing anything; it flagged {drifted}"
    )
