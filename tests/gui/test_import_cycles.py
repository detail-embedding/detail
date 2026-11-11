"""Every ``detail.gui.*`` module can be the **first** one imported.

The class: **a circular import that only fails in one import order**, so the whole suite is
green and a single file is unrunnable.

Two of these were live in this package at once, and both had the same shape — a shared name
pulled back out of a module that was still initializing:

1. ``gui/pages.py`` defined the status palette (``_MUTED`` and friends) at line 131 and imported
   ``detail.gui.modules`` at line 78 to register the module UIs. That reaches
   ``modules/analysis_query.py`` -> ``gui/confidence.py``, which imported the palette straight
   back out of ``pages`` — at line 78, before line 131 existed.
2. ``gui/confidence.py`` imported ``modules/_pydantic_form.py``. Python imports a package before
   its submodule, so that ran ``modules/__init__.py``, which imports *every* module UI to
   register them, one of which imports ``confidence`` — still on line 21 of itself.

Both raised ``ImportError: cannot import name ... from partially initialized module``, and both
were invisible to the suite: ``pytest tests/gui/test_pages.py`` alone was **31 failures**, all of
them this, while the full run passed. Nothing is wrong at the line where it is written, and the
only signal anyone looks at says fine.

The fixes were structural, not local: the palette moved to ``gui/status_color.py`` and the form
builder to ``gui/_pydantic_form.py``, both leaves. The rule is that a shared helper does not live
inside a package whose ``__init__`` imports the world.

**Why this is a subprocess check and not a source scan.** A scan has to model Python's import
semantics — parent packages run before submodules, ``from pkg import submodule`` is safe inside a
cycle while ``from mod import attribute`` is not — and the first version of this test got that
model wrong in both directions: it reported a clean tree while ``import detail.gui.confidence``
still raised, then reported a cycle for a package importing its own children. Importing each
module first in a fresh interpreter has no model to get wrong.

The 50 interpreters are spawned through a thread pool, which is what makes this a *guard*
(``tests/tiers.py``) rather than a slow-tier test: serially it is ~25s, pooled it is ~4s, and a
guard skipped on the PRs that touch ``detail/gui/`` is the one place a cycle is ever introduced.
The work is all in the child processes, so threads are the right pool — nothing here holds the
GIL, and each ``python -c import X`` is independent of every other.
"""

import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_GUI = _REPO / "detail" / "gui"


def _gui_modules() -> list[str]:
    """Every importable ``detail.gui.*`` module name, packages included."""
    names = []
    for py in sorted(_GUI.rglob("*.py")):
        parts = py.relative_to(_REPO).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def _import_first(module: str) -> subprocess.CompletedProcess:
    """Import ``module`` into a fresh interpreter, with nothing imported before it."""
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
    )


def test_every_gui_module_can_be_imported_first():
    modules = _gui_modules()
    workers = min(len(modules), (os.cpu_count() or 2) * 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_import_first, modules))

    offenders = []
    for module, result in zip(modules, results):
        if result.returncode != 0:
            last = [ln for ln in result.stderr.strip().splitlines() if ln.strip()]
            offenders.append(f"{module}: {last[-1] if last else 'failed'}")
    assert not offenders, (
        "these modules fail when imported first, which means a cycle that other import "
        "orders happen to hide:\n  " + "\n  ".join(offenders)
    )


def test_the_check_would_catch_a_cycle(tmp_path):
    """Guard the guard: with the tree clean the loop above passes vacuously forever.

    Builds the exact shape that shipped — a package whose ``__init__`` imports a child, and a
    child that reads a name back out of the still-initializing package.
    """
    pkg = tmp_path / "cyclic_probe"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from cyclic_probe import child  # noqa: F401\n", encoding="utf-8"
    )
    (pkg / "child.py").write_text("from cyclic_probe import NAME\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-c", "import cyclic_probe"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode != 0
    assert "partially initialized" in result.stderr, result.stderr[-400:]


def test_a_shared_helper_never_imports_a_hub():
    """The two helpers the cycles ran through must not reach back into the hubs.

    Not "imports nothing" — ``_pydantic_form`` legitimately uses ``gui/spin.py``, which imports
    nothing from ``detail`` at all, and a chain is not a cycle. The property that matters is
    narrower: a module *everyone* imports must not import ``pages`` (5000 lines that pull in the
    module registry) or ``detail.gui.modules`` (whose ``__init__`` imports every module UI).
    Those two are what closed both loops.

    The subprocess loop above is the complete check; this one names the rule, so a reviewer sees
    why the helper lives where it does rather than rediscovering it from an import error.
    """
    hubs = ("detail.gui.pages", "detail.gui.modules")
    for name in ("status_color", "_pydantic_form", "spin"):
        text = (_GUI / f"{name}.py").read_text(encoding="utf-8-sig")
        offending = [
            line
            for line in text.splitlines()
            if line.startswith(("import ", "from ")) and any(h in line for h in hubs)
        ]
        assert not offending, (
            f"{name}.py is imported by much of the GUI and now imports a hub: {offending}. "
            "That is how both shipped cycles formed."
        )


def test_opening_the_gui_does_not_import_matplotlib():
    """A plotting stack is not a startup cost.

    `detail/analysis/comparison_report.py` reads `stats.csv`, formats tables and names
    measurements — none of which needs matplotlib — and the comparison page imports it for those.
    A module-level `from detail.report import figures` there put ~160 ms of matplotlib and pyplot
    into every app launch, for surfaces that never draw. Measured against `origin/main`, which
    did not.

    `detail/report/figures.py` is legitimately a matplotlib module and is exempt; the rule is
    that reaching it must be a decision, not a side effect of importing a page.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, detail.gui.comparison;"
            " sys.exit(1 if 'matplotlib' in sys.modules else 0)",
        ],
        capture_output=True,
    )

    assert result.returncode == 0, (
        "importing the comparison page pulled in matplotlib — something on its import path "
        "reaches detail.report.figures at module scope"
    )
