"""A form that rebuilds its Advanced rows calls `ConfigForm.finish()` again.

**The class.** *Settings built behind a hidden Advanced section.* A form whose Advanced rows
depend on another choice (the tracker, the trainer, the keypoint track method) clears and
re-adds them. The section is hidden at construction when it opens empty, and only
`ConfigForm.finish()` shows or hides it, so a rebuild that skips it leaves every new row
unreachable. Nothing fails: the widgets exist, `config()` reads them, and a test that checks
the widgets passes. Reported from the app on the keypoint track form, the only rebuild of three
that skipped it.

A source scan, the `test_ui_thread_blocking.py` idiom: every function in `detail/gui/` that
calls `.advanced().clear(` must also call `.finish()`.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.guard

_GUI = Path(__file__).resolve().parents[2] / "detail" / "gui"


def _offenders(source: str, name: str) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        text = ast.unparse(node)
        if ".advanced().clear(" in text and ".finish()" not in text:
            out.append(f"{name}::{node.name}")
    return out


def test_every_advanced_rebuild_calls_finish():
    offenders = []
    for path in sorted(_GUI.rglob("*.py")):
        offenders += _offenders(path.read_text(encoding="utf-8-sig"), path.name)
    assert not offenders, (
        "these rebuild Advanced rows without calling ConfigForm.finish(), so the section "
        f"stays hidden over the new rows: {offenders}"
    )


def test_the_scan_would_catch_a_rebuild_without_finish():
    broken = (
        "def _rebuild(self):\n"
        "    self._form.advanced().clear('knobs')\n"
        "    self._form.addRow('x', w, advanced='knobs')\n"
    )
    assert _offenders(broken, "m.py") == ["m.py::_rebuild"]
    fixed = broken + "    self._form.finish()\n"
    assert _offenders(fixed, "m.py") == []
