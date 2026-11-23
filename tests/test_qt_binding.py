"""The Qt binding is named in one file, and only the GUI imports Qt.

**The class: a second spelling of the binding.** The GUI moved from PyQt6 (GPL-3.0) to PySide6
(LGPL-3.0) so that it can ship under a license that adds terms GPL forbids (#107). One import of
the old binding left anywhere — a widget, a test helper, a script — keeps the GPL dependency, and
in an environment that still has both installed it loads a second binding beside the first, where
the two cannot share a widget. An ``importorskip`` on the old name is worse: once it is
uninstalled, every test behind it **skips**, and the suite still reads green.

**The rule.** ``detail/gui/qt.py`` is the one module that imports ``PySide6`` or ``shiboken6``;
every other file imports Qt from it. No file imports ``PyQt5``, ``PyQt6``, ``PySide2`` or ``sip``,
by statement or by a string handed to ``__import__``, ``importlib.import_module`` or
``pytest.importorskip``. No test skips on a missing binding: in the GUI tier a missing binding is
an error. And CLAUDE.md's *No Qt imports outside gui/*: the backend packages import neither the
binding nor ``detail.gui``.

**And no wait that holds the GIL.** PySide6 keeps the GIL through ``QTest.qWait`` and through a
time-limited ``processEvents(flags, ms)``; PyQt6 released it. Every Python thread is frozen for
the length of such a wait, so a test waiting on a worker waits for a worker that cannot run: in
the first GUI-tier run after the switch, a decode thread finished only after the wait meant to
see it had ended. ``detail.gui.qt.qwait`` is the replacement.

**And no ``Signal(dict)``.** PySide6 converts a ``dict`` payload to Qt's string-keyed map; a dict
with int keys arrives as ``{}``, with one line on stderr and nothing raised. The Find bar's rings
(``{cluster id: colour}``) vanished that way. ``Signal(object)`` passes the dict through, as
PyQt6's ``pyqtSignal(dict)`` did.

**And no ``QSplitterHandle.splitter()`` in the app.** In PySide6 6.12 that read releases the
reference that keeps the splitter's Python side alive: the collector frees it while the C++
splitter is still on screen, a later lookup returns a wrapper with none of its attributes, and an
override Qt then calls on the freed object crashes the process — two GUI-tier workers died that
way. A handle keeps the splitter it was built for (``grip_splitter._GripHandle._owner``). Tests
may call it: one does, on purpose, to move ownership.

The predicate is :func:`violations`, a pure function of ``{path: source}``, so the negative cases
below hand it literal source.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The one module allowed to import the binding itself.
THE_SHIM = "detail/gui/qt.py"

#: Bindings, and their helper modules, that no file may import.
FORBIDDEN = ("PyQt5", "PyQt6", "PySide2", "sip")

#: The binding this repository uses, importable only by :data:`THE_SHIM`.
BINDING = ("PySide6", "shiboken6")

#: CLAUDE.md's *No Qt imports outside gui/*: these packages import no Qt at all.
QT_FREE_PACKAGES = (
    "detail/core/",
    "detail/storage/",
    "detail/pipeline/",
    "detail/models/",
    "detail/trackers/",
    "detail/transforms/",
    "detail/executor/",
    "detail/api/",
)

#: Calls whose first string argument names a module that will be imported (or skipped on).
_IMPORTING_CALLS = {"__import__", "import_module", "importorskip"}


def _top(module: str) -> str:
    return module.split(".")[0]


def _imported(tree: ast.AST) -> list[tuple[int, str, bool]]:
    """``(line, module, is_skip)`` for every module the source imports or skips on."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name, False) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module, False))
            # `from PyQt6 import sip` names the helper as an attribute, not in the module path.
            found += [
                (node.lineno, f"{node.module}.{alias.name}", False)
                for alias in node.names
            ]
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if name not in _IMPORTING_CALLS or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.append((node.lineno, first.value, name == "importorskip"))
    return found


#: Methods of sip's pointer type (`sip.voidptr`), which `QImage.bits()` returned under PyQt6. An
#: import scan cannot see them — the type arrives without being imported — so they are named.
_SIP_POINTER_METHODS = {"setsize", "asstring"}


def _gil_holding_waits(tree: ast.AST) -> list[tuple[int, str]]:
    """``(line, call)`` for every ``.qWait(...)`` and time-limited ``.processEvents(...)``."""
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr in _SIP_POINTER_METHODS:
            found.append(
                (node.lineno, f"sip's {node.func.attr}() (PyQt6's pointer type)")
            )
        elif node.func.attr == "qWait":
            found.append((node.lineno, "qWait"))
        elif (
            node.func.attr == "processEvents"
            and len(node.args) + len(node.keywords) >= 2
        ):
            found.append((node.lineno, "processEvents with a time limit"))
    return found


def _lossy_signals(tree: ast.AST) -> list[int]:
    """Lines declaring a ``Signal`` with a ``dict`` argument."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "Signal"
        and any(isinstance(a, ast.Name) and a.id == "dict" for a in node.args)
    ]


def _splitter_reads(tree: ast.AST) -> list[int]:
    """Lines calling ``.splitter()`` with no arguments — ``QSplitterHandle.splitter``."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "splitter"
        and not node.args
        and not node.keywords
    ]


def violations(sources: dict[str, str]) -> list[str]:
    """``path:line: why`` for every import or wait that breaks the rule in the module docstring."""
    found = []
    for path, source in sources.items():
        if path.startswith("detail/"):
            for line in _splitter_reads(ast.parse(source)):
                found.append(
                    f"{path}:{line}: QSplitterHandle.splitter() frees the splitter's Python "
                    "side in PySide6 — keep a reference instead"
                )
        for line in _lossy_signals(ast.parse(source)):
            found.append(
                f"{path}:{line}: Signal(dict) drops non-string keys in PySide6 — use object"
            )
        for line, call in _gil_holding_waits(ast.parse(source)):
            found.append(
                f"{path}:{line}: {call} — not in PySide6"
                if call.startswith("sip")
                else f"{path}:{line}: {call} holds the GIL under PySide6 — use detail.gui.qt.qwait"
            )
        seen = set()
        for line, module, is_skip in _imported(ast.parse(source)):
            parts = module.split(".")
            where = f"{path}:{line}"
            if where in seen:
                continue  # one report per import statement
            before = len(found)
            if any(part in FORBIDDEN for part in parts):
                found.append(f"{where}: imports {module} — use detail.gui.qt")
            elif _top(module) in BINDING and is_skip:
                found.append(
                    f"{where}: skips on {module} — a missing binding must fail"
                )
            elif _top(module) in BINDING and path != THE_SHIM:
                found.append(f"{where}: imports {module} — only {THE_SHIM} may")
            elif path.startswith(QT_FREE_PACKAGES) and (
                _top(module) in BINDING
                or module == "detail.gui"
                or module.startswith("detail.gui.")
            ):
                found.append(f"{where}: imports {module} — no Qt outside detail/gui/")
            if len(found) > before:
                seen.add(where)
    return found


def _repository_sources() -> dict[str, str]:
    sources = {}
    for root in ("detail", "tests", "scripts"):
        for path in sorted((REPO / root).rglob("*.py")):
            sources[path.relative_to(REPO).as_posix()] = path.read_text(
                encoding="utf-8"
            )
    return sources


def test_the_binding_is_named_in_one_file():
    sources = _repository_sources()
    assert THE_SHIM in sources, "the shim moved; update THE_SHIM"
    assert len(sources) > 500, "the scan found too few files to mean anything"
    assert violations(sources) == []


def test_the_shim_really_imports_the_binding():
    """Not vacuous: the one allowed site is where the binding actually comes from."""
    shim = (REPO / THE_SHIM).read_text(encoding="utf-8")
    modules = {_top(module) for _, module, _ in _imported(ast.parse(shim))}
    assert {"PySide6", "shiboken6"} <= modules


def test_the_scan_would_catch_the_old_binding():
    planted = {
        "detail/gui/pages.py": "from PyQt6.QtWidgets import QLabel\n",
        "detail/gui/lifetime.py": "from PyQt6 import sip\n",
        "tests/gui/test_x.py": "import PyQt6.QtCore\n",
        "scripts/x.py": "import sip\n",
        "tests/gui/test_y.py": '__import__("PyQt6.QtGui", fromlist=["QColor"])\n',
        "tests/gui/test_z.py": 'import importlib\nimportlib.import_module("PyQt5.QtCore")\n',
    }
    assert len(violations(planted)) == len(planted)


def test_the_scan_would_catch_a_direct_binding_import():
    planted = {
        "detail/gui/pages.py": "from PySide6.QtWidgets import QLabel\n",
        "detail/gui/lifetime.py": "import shiboken6\n",
        "tests/gui/test_x.py": "from PySide6 import QtCore\n",
    }
    assert len(violations(planted)) == len(planted)


def test_the_scan_would_catch_a_skip_on_a_missing_binding():
    planted = {
        "tests/gui/test_x.py": 'import pytest\npytest.importorskip("PyQt6")\n',
        "tests/gui/test_y.py": 'import pytest\npytest.importorskip("PySide6")\n',
    }
    assert len(violations(planted)) == len(planted)


def test_the_scan_would_catch_qt_in_a_backend_package():
    planted = {
        "detail/core/x.py": "from detail.gui.qt import QColor\n",
        "detail/api/y.py": "import detail.gui.workers\n",
        "detail/storage/z.py": "def f():\n    from detail.gui import qt\n",
    }
    assert len(violations(planted)) == len(planted)


def test_the_scan_would_catch_a_wait_that_holds_the_gil():
    planted = {
        "tests/gui/test_x.py": "from detail.gui.qt import QTest\nQTest.qWait(100)\n",
        "tests/gui/test_y.py": "def f(qapp, flags):\n    qapp.processEvents(flags, 50)\n",
        "detail/gui/z.py": "def f(app):\n    app.processEvents(maxtime=50, flags=0)\n",
    }
    assert len(violations(planted)) == len(planted)


def test_the_scan_would_catch_a_dict_signal():
    planted = {
        "detail/gui/x.py": "class F(QObject):\n    found = Signal(dict)\n",
        "detail/gui/y.py": "class F(QObject):\n    picked = QtCore.Signal(int, dict)\n",
    }
    assert len(violations(planted)) == len(planted)


def test_the_scan_would_catch_a_sip_pointer_method():
    planted = {
        "tests/gui/x.py": "bits = image.constBits()\nbits.setsize(n)\n",
        "detail/gui/y.py": "data = image.bits().asstring(n)\n",
    }
    assert len(violations(planted)) == len(planted)


def test_the_scan_would_catch_a_splitter_read():
    planted = {
        "detail/gui/x.py": "def sizeHint(self):\n    return self.splitter().strip_width()\n",
        "detail/gui/y.py": "def f(handle):\n    s = handle.splitter()\n",
    }
    assert len(violations(planted)) == len(planted)


def test_the_scan_leaves_the_allowed_shapes_alone():
    allowed = {
        THE_SHIM: "import shiboken6\nfrom PySide6.QtCore import Qt\n",
        "detail/gui/pages.py": "from detail.gui.qt import QLabel\n",
        "tests/gui/test_x.py": "from detail.gui.qt import QTest\n",
        "detail/core/x.py": '"""Mentions PyQt6 and sip in prose only."""\nimport math\n',
        "detail/cli.py": "def gui():\n    from detail.gui.__main__ import main\n",
        "tests/gui/test_w.py": "from detail.gui.qt import qwait\nqwait(100)\n",
        "tests/gui/test_v.py": "def f(qapp):\n    qapp.processEvents()\n",
        "detail/gui/w.py": "class F(QObject):\n    found = Signal(object)\n",
        "tests/gui/test_u.py": "def f(split):\n    split.handle(1).splitter()\n",
    }
    assert violations(allowed) == []
