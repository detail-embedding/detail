#!/usr/bin/env python3
"""Print the pytest arguments worth running for the current change.

Prints a list of test files to stdout, so it composes directly:

    pytest -q $(python scripts/select_tests.py)

Printing **nothing** means "run the whole suite", which is what an unrecognised path, a
change to the gate's own files, or a clean tree all mean — and which that command line
does for free, because bare ``pytest -q`` is the full run. The reason is written to stderr
so a human can see which of those happened.

This is an edit-loop tool, not a gate --- see ``tests/impact.py`` for why that division is
the design rather than a shortcut. What runs everything is a full run someone starts by
hand: CI's PR jobs run tier selections (``.github/workflows/ci.yml``), not this script's,
and no hook runs the suite, so nothing this script prints ever narrows the gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    """Run git in the repo and return stdout, or "" if it failed or is absent.

    The convention: list form, explicit cwd, never
    ``check=True`` (the return code is inspected here), and a missing git binary is a
    reason to degrade rather than to raise.
    """
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=REPO,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def changed_files(against: str | None) -> list[str]:
    """Repo-relative paths this change touches.

    With ``against``, the diff of the branch against that ref plus anything uncommitted —
    a branch is not just its commits while you are still editing it. Without it, the
    working tree alone, which is the edit loop.
    """
    paths = set()
    if against:
        merge_base = _git("merge-base", "HEAD", against).strip()
        if merge_base:
            paths.update(_git("diff", "--name-only", merge_base, "HEAD").split())
    for line in _git("status", "--porcelain").splitlines():
        entry = line[3:].strip()
        if " -> " in entry:  # a rename reports "old -> new"; both sides matter
            old, _, new = entry.partition(" -> ")
            paths.update((old.strip(), new.strip()))
        elif entry:
            paths.add(entry)
    return sorted(paths)


def _load_impact():
    """Load ``tests/impact.py`` by path — the same reason conftest does not import it."""
    spec = importlib.util.spec_from_file_location(
        "impact", REPO / "tests" / "impact.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["impact"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--against",
        metavar="REF",
        help="also include everything this branch changed since REF (e.g. origin/main)",
    )
    parser.add_argument(
        "--no-guards",
        action="store_true",
        help="leave the guards out — for a tight loop only; run them before you push",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="list the changed files the selection was derived from",
    )
    args = parser.parse_args()

    changed = changed_files(args.against)
    if args.explain:
        print(f"changed files ({len(changed)}):", file=sys.stderr)
        for rel in changed:
            print(f"    {rel}", file=sys.stderr)

    impact = _load_impact()
    selected = impact.select(changed, guards=not args.no_guards)

    if selected is None:
        reason = (
            "no change detected"
            if not changed
            else "a change the table does not narrow"
        )
        print(f"select_tests: {reason} — run the whole suite", file=sys.stderr)
        return 0

    guards = "" if args.no_guards else " (guards included)"
    print(f"select_tests: {len(selected)} test files{guards}", file=sys.stderr)
    print(" ".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
