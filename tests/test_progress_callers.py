"""Every caller drives the progress callback the API offers.

**The class.** *A long operation whose only feedback is that it started and that it finished.*
CLAUDE.md's long-op rule has three halves: a long synchronous API method takes a ``progress``
callback, and **every caller drives it** — the CLI wires it to a bar, the GUI to its bar. This
guards the second half, the one that fails without anything failing: a call that drops the
callback runs correctly, returns the right thing, and shows the user nothing until it ends,
which is indistinguishable from a hang.

**How it hid.** Exporting a whole experiment (#916) was the case a user found: the Experiment
page's Export was a ``BackgroundCall`` around ``Experiment.export`` with a button reading
"Exporting…" for the whole of a 17 GB pack, the CLI's ``experiment export`` printed nothing until
it was done, and Home's Import was the same in reverse. Those methods took no ``progress`` at all
— the first half — and gained one in that change. Sweeping for the second half then found
``ImportBehaviorLabelsDialog`` calling ``import_behavior_labels`` without the ``progress`` it had
always taken: the API reported per recording and the dialog said "Importing…".

**The rule.** The methods covered are **derived from their signatures** — every public method of
``Experiment`` and ``Recording`` with a ``progress`` parameter — so one that gains a callback is
covered the day it does, with nothing here to update. Every call to one of those names in
``detail/cli*.py`` and ``detail/gui/`` passes ``progress=`` (or forwards ``**kwargs``).

**Where it stops.** It matches on the method's *name*, which is what lets it run without type
inference and is also its limit:

- A different object's method of the same name is checked too. ``export`` is the case: two
  hidden capabilities (§25) defined an ``export`` that took no callback, and their calls were
  listed in ``ALLOWED`` with that reason until both were removed. An entry that stops excusing a
  call fails.
- A call through a bound method held in a variable, ``getattr``, or a helper that forwards its
  arguments is not seen.
- It says nothing about a long method that takes **no** ``progress`` at all. That is the half
  ``Experiment.export`` was; ``Recording.export_locations`` still is (CLAUDE.md names it).
"""

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
DETAIL = REPO / "detail"
API_SOURCES = ("detail/api/experiment.py", "detail/api/recording.py")
API_CLASSES = {"Experiment", "Recording"}

#: ``(file, enclosing function) -> why that call passes no progress``. A claim: an entry that no
#: longer excuses a call fails, so a stale one cannot sit here the way a dropped keyword did.
ALLOWED: dict[tuple[str, str], str] = {}


def progress_methods() -> set[str]:
    """Public ``Experiment`` / ``Recording`` methods that take a ``progress`` parameter."""
    names = set()
    for rel in API_SOURCES:
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8-sig"))
        for cls in tree.body:
            if not (isinstance(cls, ast.ClassDef) and cls.name in API_CLASSES):
                continue
            for fn in cls.body:
                if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("_"):
                    continue
                params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
                if "progress" in params:
                    names.add(fn.name)
    return names


def calls_dropping_progress(source: str, names: set[str]) -> list[tuple[str, int, str]]:
    """``(enclosing function, line, method)`` for each call to ``names`` that passes no progress.

    Pure — source in, findings out — so the negative case can hand it a literal.
    """
    found = []

    def visit(node, enclosing: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            enclosing = node.name
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in names
        ):
            keywords = {k.arg for k in node.keywords}
            # `None` is a `**kwargs` forward, which may carry it.
            if "progress" not in keywords and None not in keywords:
                found.append((enclosing, node.lineno, node.func.attr))
        for child in ast.iter_child_nodes(node):
            visit(child, enclosing)

    visit(ast.parse(source), "<module>")
    return found


def _surface_files() -> list[pathlib.Path]:
    return sorted(DETAIL.glob("cli*.py")) + sorted((DETAIL / "gui").rglob("*.py"))


def _findings() -> dict[tuple[str, str], list[str]]:
    names = progress_methods()
    out: dict[tuple[str, str], list[str]] = {}
    for path in _surface_files():
        rel = path.relative_to(REPO).as_posix()
        for enclosing, line, method in calls_dropping_progress(
            path.read_text(encoding="utf-8-sig"), names
        ):
            out.setdefault((rel, enclosing), []).append(f"{rel}:{line} .{method}(")
    return out


def test_the_methods_covered_include_the_archive_operations():
    """Not vacuous: the derived set is the real one, archive export and import among it."""
    names = progress_methods()
    assert {"export", "import_archive", "add_video_part"} <= names, sorted(names)


def test_every_caller_passes_the_progress_the_api_offers():
    offenders = [
        site
        for key, sites in _findings().items()
        if key not in ALLOWED
        for site in sites
    ]
    assert not offenders, (
        "a call drops the progress callback its method takes, so it runs silent until it "
        "ends — wire it to the surface's bar (CLI `_StageProgress`, GUI `run_progress`):\n  "
        + "\n  ".join(offenders)
    )


def test_every_exemption_still_excuses_a_call():
    findings = _findings()
    stale = [f"{rel}::{fn}" for rel, fn in ALLOWED if (rel, fn) not in findings]
    assert not stale, f"ALLOWED entries that excuse nothing: {stale}"


def test_the_scan_would_catch_a_dropped_progress():
    source = (
        "def export_command(proj, out, bar, extra):\n"
        "    proj.export(out)\n"  # dropped
        "    proj.export(out, progress=bar)\n"  # wired
        "    proj.export(out, **extra)\n"  # forwarded; may carry it
        "    run_progress(w, lambda emit: proj.import_archive(a), done)\n"  # dropped
        "    proj.import_archive(a, progress=emit)\n"  # wired
    )
    found = calls_dropping_progress(source, {"export", "import_archive"})
    assert found == [
        ("export_command", 2, "export"),
        ("export_command", 5, "import_archive"),
    ]
