"""No module defines the same name twice (fail-closed source scan, the
``test_path_serialization.py`` idiom).

**The class: a second definition that silently replaces the first.** Python lets a module (or a
class body) bind a name twice, and only the last binding survives — no error, no warning, and the
first body is dead code that still reads as live. It arrives by merge: two branches each keep
their own copy of a function, git sees two non-overlapping hunks, and the result defines it
twice. The catch-up merge of ``main`` into ``remote-detail`` (2026-10-16) left four:
``_downstream_edges`` in ``api/recording.py``, ``with_alignment`` in
``transforms/constructors.py``, and the ``align_rotate`` / ``align_mirror`` transforms in
``transforms/basic.py`` — where the dead copies read ``aux["align_deg"]`` while the worker
writes ``aux["heading_deg"]``, so had the order been the other way round every aligned
embedding would have been embedded unturned, recorded as aligned. A fifth predated the merge on
both branches: ``ConfidenceStrip.resizeEvent``, whose dead first copy was the only one that
re-synced the *centre on cutoff* button.

It hides because the module imports and every test of the surviving copy passes. Only the
source says there were two.

**Test modules too, for definitions.** In ``tests/`` the cost is worse than dead code: a second
``def test_x`` means the first test never runs, and pytest reports nothing. The 2026-11-06 merge of
``main`` into ``remote-detail`` (#954) surfaced two, and the scan found two more: two different
draws' determinism tests in ``storage/test_behavior_audit.py`` under one name, and dead first
copies of the ``_training_job`` and ``_square_kf`` helpers in ``executor/test_remote_shepherd.py``
and ``gui/test_pages.py`` (kept: the copy every caller had been running). Registrations are **not**
scanned there: a test module registers a fake on purpose, and often inside a source string it
writes out for a subprocess, which the text scan cannot tell from code.

The same applies to registrations: ``register_transform`` (and every ``register_*``) writes a
dict entry, so a second ``@register_transform("align_rotate")`` replaces the first the same
way. Both are scanned.
"""

import ast
import collections
import re
from pathlib import Path

import detail

PACKAGE = Path(detail.__file__).parent
TESTS = Path(__file__).resolve().parent
_REGISTRATION = re.compile(r"@(register_[a-z_]+)\(\s*\"([^\"]+)\"\s*\)")


def duplicate_definitions(source: str) -> list[str]:
    """``name`` (or ``Class.name``) for every function or class defined twice in one scope."""
    found = []
    tree = ast.parse(source)

    def scan(body, prefix: str) -> None:
        counts = collections.Counter(
            node.name
            for node in body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        found.extend(f"{prefix}{name}" for name, n in counts.items() if n > 1)
        for node in body:
            if isinstance(node, ast.ClassDef):
                scan(node.body, f"{node.name}.")

    scan(tree.body, "")
    return found


def duplicate_registrations(source: str) -> list[str]:
    """Every ``@register_x("name")`` that appears more than once in one module — the same
    name in the same registry; one name in two registries (a transition curve and a detector both
    called ``speed``) is two entries, not a replacement."""
    counts = collections.Counter(_REGISTRATION.findall(source))
    return [f"{registry}:{name}" for (registry, name), n in counts.items() if n > 1]


def _offenders() -> list[str]:
    out = []
    for path in sorted(PACKAGE.rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        rel = path.relative_to(PACKAGE.parent).as_posix()
        out += [
            f"{rel}: {name} defined twice" for name in duplicate_definitions(source)
        ]
        out += [
            f"{rel}: {name!r} registered twice"
            for name in duplicate_registrations(source)
        ]
    for path in sorted(TESTS.rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        rel = path.relative_to(TESTS.parent).as_posix()
        out += [
            f"{rel}: {name} defined twice" for name in duplicate_definitions(source)
        ]
    return out


def test_no_module_defines_a_name_twice():
    offenders = _offenders()
    assert not offenders, (
        "a second definition silently replaces the first (usually a merge that kept both "
        "branches' copies); keep the one that is meant to be live and delete the other:\n"
        + "\n".join(offenders)
    )


def test_the_scan_would_catch_a_planted_duplicate():
    source = (
        "def edges():\n    return 1\n\n"
        "class Strip:\n"
        "    def resizeEvent(self):\n        pass\n"
        "    def resizeEvent(self):\n        pass\n\n"
        "def edges():\n    return 2\n\n"
        '@register_transform("align_rotate")\nclass A:\n    pass\n\n'
        '@register_transform("align_rotate")\nclass B:\n    pass\n'
    )
    assert sorted(duplicate_definitions(source)) == ["Strip.resizeEvent", "edges"]
    assert duplicate_registrations(source) == ["register_transform:align_rotate"]
    two_registries = (
        '@register_transition_curve("speed")\nclass S:\n    pass\n\n'
        '@register_transition_detector("speed")\nclass T:\n    pass\n'
    )
    assert duplicate_registrations(two_registries) == []


def test_the_scan_reads_every_module():
    assert (
        len(list(PACKAGE.rglob("*.py"))) > 200
    ), "the scan found almost nothing to read"
    assert (
        len(list(TESTS.rglob("*.py"))) > 400
    ), "the scan found almost no test modules to read"
