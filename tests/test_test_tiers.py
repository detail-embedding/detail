"""Every test file says what it costs, or a fast gate is a lie about what it ran.

**The class:** *a selection rule that silently stops selecting.* The gate below the merge
button no longer runs the whole suite — it runs the tiers a stage asks for. That is only
safe while every test is in a tier and every tier means what it says. Both halves fail
quietly:

- A test file added to a directory the table does not cover is selected by *no* tier. It
  is not skipped, not reported, not red — it simply never runs again, and nothing about
  the run says so. This is the same shape as the docs-nav bug in CLAUDE.md's *Recorded
  classes* table: a file that is neither in the nav nor excluded, with no third state.
- A test that spawns processes, encodes video, or JIT-compiles a kernel, sitting in the
  ``unit`` tier, does not fail — it makes the tier slow, and a gate people wait four minutes
  for becomes a gate people skip. The tier's promise is the
  thing under test, and nothing enforces a promise about a *time*.

Neither is visible to a behavioural test: the suite is green in both cases. So this
scans instead, over ``tests/tiers.py`` — the one table that assigns a cost tier by
directory, records the per-file exceptions, and names the guards.

**The rule.** Every test file resolves to exactly one cost tier (``unit`` / ``gui`` /
``slow``). A ``guard`` is never ``slow``, because guards run at every stage and a stage
that has to skip its guards has none. A ``unit``-tier file carries no signature of a
heavier tier — a real ``subprocess`` call, a ``LocalExecutor``, a ``cv2.VideoWriter``,
an import of Qt / torch / sklearn / numba — unless ``tiers.UNIT_EXEMPTIONS`` says in
words why that signature costs nothing there. The exemptions are claims, not opt-outs:
an entry that no longer trips the scan fails, so a stale excuse cannot accumulate.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

_TIERS_PATH = pathlib.Path(__file__).parent / "tiers.py"


def _load_tiers():
    """Load ``tests/tiers.py`` by path.

    Not ``import tests.tiers``: a *different* ``tests`` package is installed in
    site-packages by one of our dependencies and wins the name, and ``import tiers``
    depends on ``tests/`` being on ``sys.path``, which ``--import-mode=importlib``
    does not guarantee. The file's location relative to this one is the only fact
    that is always true.
    """
    name = "detail_test_tiers"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _TIERS_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


tiers = _load_tiers()


# --------------------------------------------------------------------------- #
# Coverage: no test file is outside the table.
# --------------------------------------------------------------------------- #


def test_every_test_file_resolves_to_exactly_one_cost_tier():
    uncovered = []
    for path in tiers.test_files():
        rel = tiers.relpath(path)
        tier = tiers.cost_tier(rel)
        if tier not in tiers.COST_TIERS:
            uncovered.append(f"{rel}: resolved to {tier!r}")
    assert not uncovered, (
        "every test file must resolve to one of "
        f"{tiers.COST_TIERS} through tests/tiers.py — add the directory to "
        "DIRECTORY_TIER or the file to FILE_TIER:\n" + "\n".join(uncovered)
    )


def test_no_guard_runs_in_the_slow_tier():
    misplaced = [
        rel for rel in sorted(tiers.GUARD_FILES) if tiers.cost_tier(rel) == "slow"
    ]
    assert not misplaced, (
        "guards run at every stage of the gate, so a guard in the slow tier is a guard "
        "that gets skipped. Split the slow part out, or drop the guard claim:\n"
        + "\n".join(misplaced)
    )


# --------------------------------------------------------------------------- #
# Misapplication: a unit-tier file that is not actually cheap.
# --------------------------------------------------------------------------- #


def test_unit_tier_files_carry_no_heavier_signature():
    offenders = []
    for path in tiers.test_files():
        rel = tiers.relpath(path)
        if tiers.cost_tier(rel) != "unit" or rel in tiers.UNIT_EXEMPTIONS:
            continue
        found = tiers.heavy_signatures(path.read_text(encoding="utf-8-sig"))
        if found:
            why = ", ".join(
                f"{sig} ({reason})" for sig, reason in sorted(found.items())
            )
            offenders.append(f"{rel}: {why}")
    assert not offenders, (
        "these files are in the unit tier — the cheapest one, run at every stage of the "
        "push — but they pay for something heavier. Move them to FILE_TIER as 'slow' "
        "or 'gui', or add a tiers.UNIT_EXEMPTIONS entry saying why the signature costs "
        "nothing here:\n" + "\n".join(offenders)
    )


def test_every_gui_tier_file_actually_touches_qt():
    """The gui job is gated on a change under `detail/gui/`, so it is the one tier that
    does not always run. A file tiered `gui` with no connection to Qt is therefore a file
    skipped on most PRs for nothing."""
    strays = []
    for path in tiers.test_files():
        rel = tiers.relpath(path)
        if tiers.cost_tier(rel) != "gui":
            continue
        if not tiers.touches_qt(path.read_text(encoding="utf-8-sig")):
            strays.append(rel)
    assert not strays, (
        "these files are tiered 'gui' but never touch Qt, so the gui paths-filter skips "
        "them on every PR that does not change the GUI — give them a FILE_TIER entry of "
        "'unit' so they run in the tier that always does:\n" + "\n".join(strays)
    )


# --------------------------------------------------------------------------- #
# The table is claims, not exemptions: a stale entry fails.
# --------------------------------------------------------------------------- #


def test_no_table_entry_points_at_a_missing_file():
    known = {tiers.relpath(p) for p in tiers.test_files()}
    stale = sorted(
        (set(tiers.FILE_TIER) | set(tiers.GUARD_FILES) | set(tiers.UNIT_EXEMPTIONS))
        - known
    )
    assert not stale, (
        "tests/tiers.py names files that no longer exist — delete the entries so the "
        "table cannot excuse something that isn't there:\n" + "\n".join(stale)
    )


def test_no_file_tier_entry_merely_restates_its_directory():
    redundant = []
    for rel, tier in sorted(tiers.FILE_TIER.items()):
        parent = str(pathlib.PurePosixPath(rel).parent)
        inherited = None
        for candidate in (
            pathlib.PurePosixPath(parent),
            *pathlib.PurePosixPath(parent).parents,
        ):
            inherited = tiers.DIRECTORY_TIER.get(str(candidate))
            if inherited is not None:
                break
        if inherited == tier:
            redundant.append(f"{rel}: {tier!r} is already the tier for {parent}/")
    assert not redundant, (
        "FILE_TIER is for exceptions. An entry that agrees with its directory hides the "
        "day the directory's default changes:\n" + "\n".join(redundant)
    )


def test_every_unit_exemption_is_still_tripped_by_the_scan():
    unused = []
    for rel, reason in sorted(tiers.UNIT_EXEMPTIONS.items()):
        path = tiers.REPO / rel
        if not path.exists():
            continue  # covered by the missing-file test above
        if tiers.cost_tier(rel) != "unit":
            unused.append(f"{rel}: no longer in the unit tier ({reason})")
        elif not tiers.heavy_signatures(path.read_text(encoding="utf-8-sig")):
            unused.append(f"{rel}: no heavy signature left to excuse ({reason})")
    assert not unused, (
        "an exemption that excuses nothing is an exemption nobody will question. Delete "
        "these:\n" + "\n".join(unused)
    )


#: The gate, exactly as `.github/workflows/ci.yml` runs it.
SELECTIONS = ("guard", "unit and not guard", "gui and not guard", "slow")


def test_the_four_gate_selections_partition_the_suite():
    """`-m guard`, `-m "unit and not guard"`, `-m "gui and not guard"` and `-m slow` are
    the whole gate, so every test has to be selected by exactly one of them. A file
    matched by none never runs again and nothing says so; one matched by two is paid for
    twice, in two jobs, on every PR."""
    wrong = {}
    for path in tiers.test_files():
        rel = tiers.relpath(path)
        cost, guard = tiers.cost_tier(rel), tiers.is_guard(rel)
        selected = [
            name
            for name, hit in zip(
                SELECTIONS,
                (
                    guard,
                    cost == "unit" and not guard,
                    cost == "gui" and not guard,
                    cost == "slow",
                ),
            )
            if hit
        ]
        if len(selected) != 1:
            wrong[rel] = selected
    assert not wrong, (
        "these files are not selected exactly once by the gate's four -m expressions:\n"
        + "\n".join(
            f"{rel}: {sel or 'selected by nothing'}" for rel, sel in wrong.items()
        )
    )


def test_the_markers_reach_the_collected_tests(request):
    """The table is inert until something reads it, and nothing did until this hook.

    Without ``pytest_collection_modifyitems`` in ``tests/conftest.py`` every ``-m``
    expression above selects zero tests — and pytest's exit code for an empty selection
    is 5, so *every* job in the CI matrix fails the same way and the tiering looks broken
    rather than unwired. This file is ``unit`` + ``guard`` per the table; if the markers
    stopped being applied, that would be the visible signature.
    """
    names = {marker.name for marker in request.node.iter_markers()}
    rel = tiers.relpath(pathlib.Path(__file__))
    assert tiers.cost_tier(rel) in names, (
        f"{rel} is tiered {tiers.cost_tier(rel)!r} but collected without that marker — "
        "tests/conftest.py's collection hook is not applying tests/tiers.py"
    )
    assert ("guard" in names) == tiers.is_guard(rel)


def test_the_markers_are_declared_in_pyproject():
    declared = set(
        re.findall(
            r"^\s*\"([a-z_]+):",
            (tiers.REPO / "pyproject.toml").read_text(encoding="utf-8-sig"),
            re.M,
        )
    )
    expected = set(tiers.COST_TIERS) | {tiers.GUARD}
    assert expected <= declared, (
        "pytest runs with --strict-markers, so every tier must be declared in the "
        f"[tool.pytest.ini_options] markers block. Missing: {sorted(expected - declared)}"
    )


# --------------------------------------------------------------------------- #
# Guard the guard: the scan has to be able to catch one.
# --------------------------------------------------------------------------- #


def test_the_tier_scan_would_catch_an_uncovered_root():
    """Each configured test root carries a default, so a *new file* under one is
    always selected — that is the point of deriving the tier from the path rather than
    from a marker somebody remembers to write. What is not covered by a default is a new
    **root**: add one to `testpaths` and its files resolve to nothing, which is the
    runs-nowhere-and-says-nothing failure this table exists to prevent."""
    assert tiers.cost_tier("tests/newarea/test_something.py") == "unit"
    assert tiers.cost_tier("checks/test_something.py") is None
    assert tiers.cost_tier("somewhere/else/test_x.py") is None


def test_the_collected_roots_come_from_testpaths():
    """...which only works while the roots this table walks are the roots pytest walks."""
    configured = [tiers.relpath(root) for root in tiers.testpath_roots()]
    expected = ["tests"]
    assert configured == expected, configured


def test_the_tier_scan_would_catch_a_spawn_in_the_unit_tier():
    found = tiers.heavy_signatures(
        "import subprocess\n"
        "def test_x():\n"
        "    subprocess.run(['echo', 'hi'], check=True)\n"
    )
    assert "subprocess.run" in found


def test_the_tier_scan_would_catch_a_qt_import_in_the_unit_tier():
    assert "PyQt6" in tiers.heavy_signatures("from PyQt6.QtWidgets import QLabel\n")
    assert "numba" in tiers.heavy_signatures("import numba\n")
    assert "cv2.VideoWriter" in tiers.heavy_signatures(
        "import cv2\ndef test_x():\n    w = cv2.VideoWriter('a.mp4', 0, 1, (2, 2))\n"
    )


def test_the_qt_check_sees_an_indirect_dependency():
    """Guard the guard: a grep for "PyQt6" reads the two parity files as Qt-free, because
    they import their widget inside the test body and take the `qapp` fixture."""
    assert tiers.touches_qt("from PyQt6.QtWidgets import QLabel\n")
    assert tiers.touches_qt("def test_x(qapp):\n    pass\n")
    assert tiers.touches_qt(
        "def test_x(qapp):\n    from detail.gui.clusterers import ClusterFitForm\n"
    )
    assert tiers.touches_qt("def test_x():\n    import detail.gui.pages as mod\n")
    assert not tiers.touches_qt(
        '"""A docstring naming PyQt6 and detail.gui."""\nimport subprocess\n'
    )


def test_the_scan_reads_the_ast_not_the_text():
    """A monkeypatch *target* is a string, not a call — flagging it would make the
    exemption list a list of false alarms, and a list of false alarms gets ignored."""
    assert not tiers.heavy_signatures(
        "def test_x(monkeypatch):\n"
        "    monkeypatch.setattr('detail.cli.subprocess.run', lambda *a, **k: None)\n"
        "    # torch and PyQt6 named only in a comment and a docstring\n"
        '    """mentions torch, sklearn, PyQt6"""\n'
    )


def test_the_guard_placement_check_would_catch_a_slow_guard(monkeypatch):
    monkeypatch.setattr(
        tiers, "GUARD_FILES", frozenset({"tests/pipeline/test_detection.py"})
    )
    assert tiers.cost_tier("tests/pipeline/test_detection.py") == "slow"
    misplaced = [
        rel for rel in sorted(tiers.GUARD_FILES) if tiers.cost_tier(rel) == "slow"
    ]
    assert misplaced == ["tests/pipeline/test_detection.py"]
