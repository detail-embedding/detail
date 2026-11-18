"""Which tests a change needs — the edit-loop half of what ``tests/tiers.py`` answers.

``tiers.py`` says what a test *costs*, and the CI gate selects on that. This says which
tests a *change* touches, and it exists for the other half of the problem: an agent
iterating on one file runs the suite dozens of times, while the gate runs once. The machine
time is in the loop, not the gate.

**This is not a gate, and the distinction is the whole design.** The gate runs everything
and nothing narrows it; this only decides what is worth running *while you work*. That is
why it may use a heuristic at all — a wrong answer here costs one extra run, not a missed
regression. Anything that claims to be sufficient has to fail closed; this claims only to
be useful, so it does not.

Why it is not built on the import graph, which would be the obvious answer: every test in
this suite transitively imports nearly the whole package, on purpose.
``tests/conftest.py::_isolate_registries`` is autouse and calls ``_prepopulate_registries``,
which imports every registered backend module so that a subset run matches the full suite.
That is a property worth keeping, and it means a static import closure answers "run
everything" for most changes. A mirror-plus-mention
heuristic answers usefully for the same changes: 5 test files for ``detail/gui/scrubber.py``,
18 for ``detail/storage/video.py``, 21 for ``detail/api/recording.py``, out of about 530.

Those numbers are asserted against the live tree by
``tests/test_impact.py::test_the_worked_example_in_the_docstring_is_still_true``: the three
selections exactly, the suite size to within a quarter (an exact total made every PR that adds a
test file edit this line). They had all four drifted — 3/8/51/361 — and two had moved *down*,
which rules out "the suite just grew"; nothing checked them, because a stale number in a
docstring reads exactly like a fresh one.

The rules, in order:

1. A changed **test file** selects itself.
2. A changed **source file** selects its mirror (``detail/gui/scrubber.py`` →
   ``tests/gui/test_scrubber.py``) plus every test file that names its dotted module.
3. A path in ``COUPLED`` also selects what is written there — the cases where the mirror
   is not enough and no rule can derive the rest.
4. A file under ``tests/`` that pytest does **not** collect, and anything outside
   ``detail/``, selects the whole suite. That is what covers the gate's own moving parts —
   the tier table, both conftests, ``pyproject.toml``, the workflows, the hooks — without a
   list of them that could go stale.
5. Guards ride along with every selection unless asked not to, because they are what the
   heuristic misses. Backtested over 400 commits, the co-edited test files a mirror cannot
   see are ``tests/tiers.py``, ``tests/test_no_orphan_functions.py`` and
   ``tests/gui/test_ui_thread_blocking.py`` — guards, every one.

``tests/test_impact.py`` is the guard on this file.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from collections.abc import Iterable

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Loaded the way ``tests/conftest.py`` loads it, under the same ``sys.modules`` key, so
#: every reader sees one module object and a ``monkeypatch.setattr`` on its tables is
#: visible here. See that file for why it is not a plain ``import``.
_TIERS_MODULE = "detail_test_tiers"


def _load_tiers():
    """Load ``tests/tiers.py`` by path, sharing conftest's module object."""
    if _TIERS_MODULE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _TIERS_MODULE, pathlib.Path(__file__).parent / "tiers.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[_TIERS_MODULE] = module
        spec.loader.exec_module(module)
    return sys.modules[_TIERS_MODULE]


# --------------------------------------------------------------------------- #
# The tables.
# --------------------------------------------------------------------------- #

# The gate's own moving parts — `pyproject.toml`, `tests/conftest.py`, `tests/tiers.py`,
# this file, `tests/gui/conftest.py`, `.github/**`, `scripts/hooks/**` — all select the
# whole suite, and deliberately have no table of their own. They are already caught by the
# two structural rules in `select`: a path outside `detail/` is not something this can
# reason about, and a path under `tests/` that pytest does not collect is somebody's
# fixture and there is no telling whose. A list naming them as well would be a list every
# entry of which excuses nothing — the failure `tests/test_test_tiers.py` exists to catch
# one table over, and one this file had on its first draft. `tests/test_impact.py` pins
# each of those paths to "everything" so the behaviour cannot drift, without a table that
# could rot.

#: Repo-relative source paths (exact, or a directory prefix ending in "/") whose change
#: needs no test run beyond the guards. Documentation has guards of its own
#: (``tests/test_docs_nav.py``) and those ride along with every selection anyway.
DOCS_ONLY = frozenset(
    {
        "docs/",
        "README.md",
        "CLAUDE.md",
        "DESIGN_PLAN.md",
        "MASTER-TODO.md",
    }
)

#: Source path -> extra test files the mirror cannot derive. Each entry is a claim about
#: *why*, held to the standard ``tiers.UNIT_EXEMPTIONS`` is held to: an entry that selects
#: nothing the mirror would not have selected anyway fails, rather than sitting here
#: excusing nothing. Keep this short — it is the hand-maintained part, and every line in it
#: is a line that can go stale.
COUPLED: dict[str, tuple[str, ...]] = {
    # The shared frame widget is embedded by surfaces that are edited at different times,
    # and CLAUDE.md's "a widget shared by two pages" class is exactly the failure where a
    # change reaches one embedder and not the other. The mirror finds test_scrubber.py.
    "detail/gui/scrubber.py": ("tests/gui/test_audit_playback.py",),
    # Registration is a side effect of import, so a new registry member changes what every
    # parity sweep enumerates without naming any of them.
    "detail/core/registries.py": (
        "tests/configs/test_record_completeness.py",
        "tests/api/test_job_reattach.py",
        "tests/gui/test_model_availability.py",
    ),
    # Assignment is called by the keypoint dataset sampler and read back by the exporters,
    # neither of which the mirror or a mention of `pose_match` in their names reaches.
    "detail/pipeline/pose_match.py": (
        "tests/pipeline/test_keypoint_dataset.py",
        "tests/api/test_exporters.py",
    ),
}


# --------------------------------------------------------------------------- #
# Lookups.
# --------------------------------------------------------------------------- #


def _matches(rel: str, table: Iterable[str]) -> bool:
    """Whether a repo-relative path is named by an exact entry or a directory prefix."""
    return any(rel == entry or rel.startswith(entry) for entry in table)


def module_path(rel: str) -> str:
    """The dotted module a repo-relative source path imports as."""
    return ".".join(pathlib.PurePosixPath(rel).with_suffix("").parts)


def mirror(rel: str) -> str | None:
    """The mirrored test path for a source path, when that file exists.

    ``tests/`` mirrors ``detail/`` by convention (`docs/package_map.md`), so
    ``detail/gui/scrubber.py`` is tested by ``tests/gui/test_scrubber.py``. Returns
    ``None`` when the convention does not happen to hold, which is common and not an
    error — the mention scan below is what covers it.
    """
    parts = pathlib.PurePosixPath(rel).parts
    if len(parts) < 2 or parts[0] != "detail":
        return None
    candidate = "/".join(("tests", *parts[1:-1], f"test_{parts[-1]}"))
    return candidate if (REPO / candidate).is_file() else None


def _test_files() -> list[str]:
    """Every collected test file, repo-relative — the corpus the mention scan reads."""
    tiers = _load_tiers()
    return [tiers.relpath(path) for path in tiers.test_files()]


def mentions(module: str, corpus: Iterable[str] | None = None) -> set[str]:
    """Test files whose source names this dotted module.

    Both import spellings are in use here and only one of them contains the dotted path:
    ``from detail.gui.scrubber import FrameScrubber`` writes it, and
    ``from detail.gui import busy`` does not. So the parent form is matched separately.
    Reading the text rather than parsing imports is deliberate — ``monkeypatch.setattr``
    targets are strings, and they are exactly the couplings an AST walk cannot see.
    """
    parent, _, stem = module.rpartition(".")
    parent_form = (
        re.compile(
            rf"^\s*from\s+{re.escape(parent)}\s+import\s+.*\b{re.escape(stem)}\b",
            re.MULTILINE,
        )
        if parent
        else None
    )
    hits = set()
    for rel in _test_files() if corpus is None else corpus:
        try:
            source = (REPO / rel).read_text(encoding="utf-8-sig")
        except OSError:
            continue
        if module in source or (parent_form is not None and parent_form.search(source)):
            hits.add(rel)
    return hits


def select(changed: Iterable[str], *, guards: bool = True) -> list[str] | None:
    """Test files worth running for a changed-file list.

    Args:
        changed: repo-relative paths, as ``git status --porcelain`` reports them.
        guards: include every guard. Leave it on unless you are iterating tightly and
            intend to run them before you push — they are what this heuristic misses.

    Returns:
        Sorted repo-relative test paths, or ``None`` meaning the whole suite — which is
        what an unrecognised path, a gate file, or an empty change list all mean.
    """
    tiers = _load_tiers()
    changed = [rel for rel in changed if rel]
    if not changed:
        return None

    selected: set[str] = set()
    corpus = _test_files()
    known = set(corpus)

    for rel in changed:
        if _matches(rel, DOCS_ONLY):
            continue
        if rel.startswith("tests/"):
            # A changed test runs itself. One that is not collected — a helper, a fixture
            # module, `tiers.py`, either `conftest.py` — is somebody's dependency and there
            # is no telling whose, so it is the whole suite.
            if rel in known:
                selected.add(rel)
            else:
                return None
            continue
        if not rel.startswith("detail/") or not rel.endswith(".py"):
            # Everything else — `pyproject.toml`, the workflows, the hooks, a stray file at
            # the root — changes something this cannot reason about.
            return None
        mirrored = mirror(rel)
        if mirrored is not None:
            selected.add(mirrored)
        selected |= mentions(module_path(rel), corpus)
        selected.update(extra for extra in COUPLED.get(rel, ()) if extra in known)

    if guards:
        selected |= {rel for rel in corpus if tiers.is_guard(rel)}
    return sorted(selected)
