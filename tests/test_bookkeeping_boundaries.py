"""An isolation boundary may not span the bookkeeping *about* the work it isolates.

**The class.** A runner that records a failure and keeps going is right to exist — a
per-recording failure must not take down a whole batch (D5). What is not right is what
that handler is allowed to *cover*. In ``RunOrchestrator._run_recording`` one ``try`` held the
job launch, the run-record write, the progress pump and the status adoption, under a single
``except Exception`` that wrote ``status: failed``. So a transient Windows ``PermissionError`` on
the coordinator's own ``progress.json`` rename marked a healthy recording failed, orphaned its
still-running worker, and left ``recording.yaml`` permanently non-terminal (#289).

**Why a source scan and not only a behavioural test.** Nothing is wrong at the line where this
is written. The handler looks correct, the try block looks tidy, and the failure needs a
platform CI does not run plus a race that beats a twenty-attempt retry. ``test_atomic`` has
always asserted that the retry re-raises once exhausted; five separate boundaries then treated
that exception as a verdict on the work. Behavioural coverage of one of them says nothing about
the other four, and says nothing at all about the sixth written next month.

**The rule this enforces.** CLAUDE.md already says *"Keep `try` blocks minimal — only the line
that can raise, not the surrounding logic"* — the rule existed and had no enforcement. Concretely
here: **a swallowing `except Exception` must not have a bookkeeping call inside its `try`.**
Publishing progress belongs outside the boundary (or, better, cannot raise at all —
``core.progress.best_effort``); persisting an index is retried at the next boundary.
"""

import ast
from pathlib import Path

import detail

_SOURCE = Path(detail.__file__).parent

#: Calls that report on work rather than doing it. A swallowing boundary that spans one of these
#: can attribute a file-system hiccup to the job. Matched on the called name, so both
#: ``self._emit(...)`` and a bare ``emit(...)`` are seen.
_BOOKKEEPING = frozenset(
    {
        "_emit",
        "_emit_fold",
        "emit",
        "_write_record",
        "write_progress",
        "set_progress",
        "_write_progress_json",
        "report",
    }
)

#: ``(module relative to detail/, enclosing function)`` -> why it is allowed to span one.
#: Empty is the goal. An entry that no longer describes a real offender fails as stale, the same
#: contract ``tests/gui/test_ui_thread_blocking.py`` holds its allowlist to.
ALLOWED: dict[tuple[str, str], str] = {}


def _best_effort_names() -> frozenset[str]:
    """Functions carrying ``@…best_effort(…)`` — publishers that already cannot raise.

    Read from the source rather than listed here, so the scan and the fix cannot drift: delete
    a decorator and every boundary spanning that publisher lights up again on the same run. An
    allowlist entry would have gone quietly stale instead, which is the failure mode this whole
    file exists to catch one level down.
    """
    safe = set()
    for path in _SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                call = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = (
                    call.attr
                    if isinstance(call, ast.Attribute)
                    else getattr(call, "id", "")
                )
                if name == "best_effort":
                    safe.add(node.name)
    return frozenset(safe)


def _called_names(node: ast.AST) -> set[str]:
    """Every function name called anywhere under ``node`` — ``f()``, ``self.f()``, ``m.f()``."""
    names = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def _swallows(handler: ast.ExceptHandler) -> bool:
    """Whether a handler records-and-continues rather than re-raising.

    A handler that re-raises is not this class: the caller still sees the exception, so nothing
    is silently relabelled. ``dataset_export`` and ``imports`` both mark a status and re-raise,
    and are deliberately not offenders.
    """
    return not any(isinstance(n, ast.Raise) for n in ast.walk(handler))


def _attributes_failure(handler: ast.ExceptHandler) -> bool:
    """Whether the handler blames something — writes a failed status rather than just noting it.

    This is the half that makes the class what it is. A handler that swallows a bookkeeping
    error and **logs** it is the *fix* (``_pump``, ``_record_best_effort``,
    ``core.progress.best_effort``); a handler that swallows it and records ``status: failed`` is
    the bug, because the verdict reaches every downstream reader. Without this distinction the
    scan flags its own remedy, which is how the first draft of this file behaved.
    """
    for node in ast.walk(handler):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "fail" in node.value.lower():
                return True
        if isinstance(node, ast.Name) and "fail" in node.id.lower():
            return True
        if isinstance(node, ast.Attribute) and "fail" in node.attr.lower():
            return True
    return False


def _is_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:  # bare except
        return True
    return isinstance(handler.type, ast.Name) and handler.type.id in {
        "Exception",
        "BaseException",
    }


def _enclosing_functions(tree: ast.AST) -> dict[int, str]:
    """Line -> name of the innermost function containing it, so a finding names a site."""
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner[line] = node.name
    return owner


def offenders_in(
    source: str, safe: frozenset[str] = frozenset()
) -> set[tuple[str, str]]:
    """``(function, the bookkeeping name it spans)`` for each offending boundary in ``source``.

    ``safe`` names publishers that already cannot raise, so spanning one is not a finding.
    Exposed so the guard can be shown to catch something — a scan that matches nothing passes
    vacuously forever.
    """
    tree = ast.parse(source)
    owner = _enclosing_functions(tree)
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        broad = [
            h
            for h in node.handlers
            if _is_broad(h) and _swallows(h) and _attributes_failure(h)
        ]
        if not broad:
            continue
        spanned = set()
        for statement in node.body:
            spanned |= _called_names(statement)
        for name in sorted(spanned & _BOOKKEEPING - safe):
            found.add((owner.get(node.lineno, "<module>"), name))
    return found


def _violations() -> dict[tuple[str, str], str]:
    safe = _best_effort_names()
    found = {}
    for path in sorted(_SOURCE.rglob("*.py")):
        rel = path.relative_to(_SOURCE).as_posix()
        for function, name in offenders_in(path.read_text(encoding="utf-8-sig"), safe):
            found[(rel, function)] = name
    return found


def test_no_swallowing_boundary_spans_its_own_bookkeeping():
    new = set(_violations()) - set(ALLOWED)
    assert not new, (
        "an isolation boundary spans a call that only *reports on* the work it isolates, so a "
        "file-system hiccup there will be recorded as the work failing (#289). Move the "
        "reporting outside the try, or make it unable to raise "
        f"(core.progress.best_effort): {sorted(new)}"
    )


def test_the_allowlist_has_no_stale_entries():
    stale = set(ALLOWED) - set(_violations())
    assert (
        not stale
    ), f"ALLOWED entries are no longer offenders — remove them: {sorted(stale)}"


# --------------------------------------------------------------------------- #
# guard the guard
# --------------------------------------------------------------------------- #
def test_the_scan_catches_the_shape_it_was_written_for():
    """`_run_recording` as it stood when #289 was reported."""
    source = """
def _run_recording(self, sid):
    try:
        run = self._experiment.get_recording(sid).run(self._module_id, cfg)
        self._write_record()
        for event in run.progress():
            self._emit(event)
        run.wait()
    except Exception as exc:
        self._per_recording[sid] = {"status": "failed", "error": str(exc)}
"""
    assert offenders_in(source) == {
        ("_run_recording", "_emit"),
        ("_run_recording", "_write_record"),
    }


def test_the_scan_does_not_fire_on_a_boundary_that_re_raises():
    """`dataset_export` and `imports` mark a status and re-raise — the caller still sees it."""
    source = """
def run(job):
    try:
        _export(job)
        report(1)
    except Exception as exc:
        set_status("failed")
        raise
"""
    assert offenders_in(source) == set()


def test_the_scan_does_not_fire_on_work_alone():
    """The boundary itself is legitimate; only its span is the bug."""
    source = """
def _run_recording(self, sid):
    try:
        run = self._experiment.get_recording(sid).run(self._module_id, cfg)
    except Exception as exc:
        self._per_recording[sid] = {"status": "failed"}
    else:
        self._emit(run)
"""
    assert offenders_in(source) == set()


def test_the_scan_does_not_fire_on_a_narrow_except():
    """A named exception is a considered decision, not a catch-all verdict."""
    source = """
def run(job):
    try:
        _export(job)
        report(1)
    except FileNotFoundError:
        pass
"""
    assert offenders_in(source) == set()
