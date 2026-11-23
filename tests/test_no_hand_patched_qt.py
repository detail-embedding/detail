"""A test patches a Qt class through ``monkeypatch``, never by assigning to it.

**The class: a patch on a Qt class that outlives its test.** ``real = QMenu.popup; QMenu.popup =
fake; ...; QMenu.popup = real`` puts the method back only if the line that restores it runs, and
only if what it reads back is the method. A test that fails between the two leaves ``fake`` on the
class for every later test in the same process. ``monkeypatch.setattr`` restores at teardown
whether or not the test got that far.

Under the Qt binding before #107 the round trip was broken even when it ran: reading a method off
the class returned a plain built-in function rather than the method descriptor in its
``__dict__``, so the restored method no longer bound, and every later ``menu.popup(point)`` passed
the point as ``self``. PySide6 restores a working method (measured 2026-11-23), so that half of
the class is gone with the old binding; the half that remains is the restore that does not run.

**How it hid.** The test that does it passes, and so does every test in its own file. The
breakage lands on whichever test next calls the method *in the same process*: three
``tests/gui/test_label_panel.py`` tests failed on Linux CI's serial gui run, after
``tests/gui/test_cluster_workspace.py`` restored ``QMenu.popup`` by hand, and passed on every
local run, where ``--dist loadfile`` put the two files on different workers. A static method
(``QMessageBox.question``) happens to survive the round trip; it is the same shape, so it is
held to the same rule.

**The rule.** In ``tests/``, no statement assigns to ``Q<Name>.<attr>``. Use
``monkeypatch.setattr`` (or ``monkeypatch.context()`` to scope it to a block).
"""

from __future__ import annotations

import ast
import pathlib
import re

TESTS = pathlib.Path(__file__).resolve().parent

_QT_CLASS = re.compile(r"Q[A-Z]\w*")


def qt_attributes_assigned(sources: dict[str, str]) -> list[str]:
    """``file:line`` for every assignment to an attribute of a Qt class in ``sources``."""
    found = []
    for name, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and _QT_CLASS.fullmatch(target.value.id)
                ):
                    found.append(f"{name}:{node.lineno}")
    return sorted(found)


def test_no_test_assigns_to_a_qt_class():
    sources = {
        str(path.relative_to(TESTS)): path.read_text(encoding="utf-8-sig")
        for path in TESTS.rglob("*.py")
    }
    found = qt_attributes_assigned(sources)
    assert not found, (
        f"{found} assign to a Qt class's attribute; restoring it by hand leaves a method that "
        "no longer binds. Use monkeypatch.setattr (this module's docstring)."
    )


def test_the_scan_would_catch_each_way_a_qt_class_is_patched_by_hand():
    sources = {
        "method.py": "def t():\n    QMenu.popup = lambda self, *a: None\n",
        "static.py": "def t():\n    QMessageBox.question = staticmethod(f)\n",
        "restore.py": "def t():\n    real = QMenu.exec\n    QMenu.exec = real\n",
        "patched.py": "def t(m):\n    m.setattr(QMenu, 'popup', f)\n",
        "instance.py": "def t(menu):\n    menu.popup = f\n",
        "stub.py": "def t():\n    StubTrainer.calls = []\n",
        "lower.py": "def t():\n    Qt.x = 1\n",
    }
    assert qt_attributes_assigned(sources) == [
        "method.py:2",
        "restore.py:3",
        "static.py:2",
    ]
