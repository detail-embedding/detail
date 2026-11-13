"""How many real executor runs the suite starts, as a ratchet that only goes down.

**The class: the same expensive work done again.** `docs/how_we_work.md` R10 names it —
*"a wall-clock assertion on the machine that also runs the suite is a flake, and the defect is not
'slow' but 'done again'"*. That is exactly what made this suite unusable: 28 tests in
`tests/pipeline/test_pose_clustering_module.py` each started a real `pose_clustering` analysis with
identical arguments, recomputing one deterministic fit 28 times at ~25s each — **12 of the suite's
36 minutes in one file** (#534).

Nothing caught it, and the two instruments you would reach for first cannot:

- **Test count.** #534 removed 12 minutes and deleted **zero** tests. A count ceiling would neither
  have rewarded the fix nor detected the problem.
- **Wall clock.** Not assertable here. A timing assertion on the box that also runs the suite is a
  flake by construction (R10 again), and #574 is a live example of a test that asserts on elapsed
  time and fails under parallel load.

What *is* deterministic is the number of real runs a module starts. A run is
``run_analysis`` / ``run_processing`` — an executor subprocess, seconds to tens of seconds. One per
module needs no explanation. Several means either the module genuinely exercises several distinct
pipelines, or it is recomputing something a shared fixture should build once — and the only way to
tell is for the number to be written down and to have to come down.

**A ratchet, like ``tests/test_rulebook.py``'s ``LINE_CEILING``.** Lowered as work lands; raised only
as a deliberate, visible act. `tests/conftest.py` is deliberately absent: a run started by a shared
fixture is the *fix*, not the cost.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

#: A module may start this many real executor runs. Absent from the table means **at most one**, so a
#: new file cannot quietly add ten. Numbers come down as fixtures are shared; `test_slack_is_lowered`
#: is what makes them come down rather than sit here as headroom.
REAL_RUN_CEILING: dict[str, int] = {
    # Exercises the run lifecycle itself — launch, reattach, retry, cancel — which is a different
    # pipeline each time rather than one artifact rebuilt. Unexamined for sharing; the highest number
    # here and the first place to look.
    "tests/api/test_experiment.py": 14,
    "tests/pipeline/test_location_analysis.py": 6,
    "tests/api/test_analysis.py": 4,
    # The free-space check (#743) fires before a run would start, and every launch here stubs
    # `spawn_coordinator`: none of the nine is a real executor run. Counted because the scan is
    # static, which is the right default; a real run added here would show in its timing. Three
    # of the nine arrived with the launch-progress tests: the report of a launch (one
    # parametrized call covers one recording and two), the refusal's report, and the retry's
    # setup launch.
    "tests/api/test_run_space.py": 9,
    # Which executor `run_analysis` picks for a remote `compute`, and the two refusals before it:
    # both executors are replaced by a fake whose `submit` returns at once, so none of the four
    # is a real executor run. Counted because the scan is static.
    "tests/api/test_remote_analysis.py": 4,
    # Two apiece: one per method the module registers, which is the thing under test.
    "tests/pipeline/test_temporal_clustering_module.py": 2,
}

#: Not a test module: a run started here is shared by every test that asks for the fixture, which is
#: what this whole check exists to encourage.
EXEMPT = frozenset({"tests/conftest.py"})

_RUN_CALLS = frozenset({"run_analysis", "run_processing"})


def real_runs_per_file() -> dict[str, int]:
    """Count ``run_analysis`` / ``run_processing`` calls in each test module.

    On the AST, not the text, so a name in a docstring or a ``monkeypatch`` target spelled as a
    string is not a hit — the same reason ``tiers.heavy_signatures`` parses rather than greps.
    """
    counts: dict[str, int] = {}
    for path in sorted((REPO / "tests").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel in EXEMPT or "__pycache__" in rel:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:  # a file mid-edit is not this check's business
            continue
        n = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _RUN_CALLS
        )
        if n:
            counts[rel] = n
    return counts


def test_no_module_starts_more_real_runs_than_its_ceiling() -> None:
    over = {
        rel: (n, REAL_RUN_CEILING.get(rel, 1))
        for rel, n in real_runs_per_file().items()
        if n > REAL_RUN_CEILING.get(rel, 1)
    }
    assert not over, (
        "these modules start more real executor runs than their ceiling allows:\n"
        + "\n".join(f"  {rel}: {n} > {cap}" for rel, (n, cap) in sorted(over.items()))
        + "\n\nA real run is an executor subprocess, seconds to tens of seconds. Build it once in a "
        "shared fixture and copy the result per test (see `tests/conftest.py::_temporal_fit_template`), "
        "or raise the ceiling here and say why several distinct runs are the thing under test."
    )


def test_slack_is_lowered_rather_than_kept_as_headroom() -> None:
    """The ratchet. An entry above what the file actually starts is headroom for the next accretion.

    This is what makes the number a ratchet rather than a high-water mark: when a module's runs are
    consolidated into a fixture, the ceiling has to follow it down in the same change.
    """
    counts = real_runs_per_file()
    slack = {
        rel: (counts.get(rel, 0), cap)
        for rel, cap in REAL_RUN_CEILING.items()
        if counts.get(rel, 0) < cap
    }
    assert not slack, (
        "these ceilings are above what the file starts — lower them to the actual number:\n"
        + "\n".join(
            f"  {rel}: starts {n}, ceiling {cap}"
            for rel, (n, cap) in sorted(slack.items())
        )
    )


def test_no_ceiling_entry_points_at_a_missing_file() -> None:
    missing = sorted(rel for rel in REAL_RUN_CEILING if not (REPO / rel).exists())
    assert not missing, f"ceiling entries for files that no longer exist: {missing}"


def test_no_entry_merely_restates_the_default() -> None:
    """An entry of 1 excuses nothing — one run is the default and needs no table row."""
    pointless = sorted(rel for rel, cap in REAL_RUN_CEILING.items() if cap <= 1)
    assert not pointless, (
        "these entries allow what the default already allows, so they excuse nothing "
        f"and should be deleted: {pointless}"
    )


def test_the_check_would_catch_a_module_that_recomputed_a_fit() -> None:
    """Guard the guard: the shape #534 fixed must be rejected if it comes back.

    The pose-clustering module had 28 real runs against a ceiling of 2 (the file and the module were
    removed since; the temporal module, with the same ceiling, stands in). Without this, the check
    could be inverted or comparing the wrong way round and every test above would still pass.
    """
    counts = {"tests/pipeline/test_temporal_clustering_module.py": 28}
    over = {rel: n for rel, n in counts.items() if n > REAL_RUN_CEILING.get(rel, 1)}
    assert (
        over
    ), "the check would not flag the 28-run file this ratchet exists to prevent"

    fresh = {"tests/some/new_file.py": 2}
    assert any(
        n > REAL_RUN_CEILING.get(rel, 1) for rel, n in fresh.items()
    ), "a new module with two real runs and no entry must be flagged, not defaulted past"
