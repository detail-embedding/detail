"""A dialog lets go of what it holds on ``finished``, never in ``closeEvent``.

**The class.** Cleanup a dialog does when it closes — taking itself off a list, saving its
geometry, releasing a tile source — written in ``closeEvent``. Esc, a **Close** button wired to
``reject`` and every ``accept``/``done`` close a ``QDialog`` through ``QDialog.done()``, which
since Qt 6.3 hides (and, under ``WA_DeleteOnClose``, deletes) the dialog **without sending a
close event**. Only the title bar's close and ``close()`` go through ``closeEvent``.

**How it hid.** The cluster window (``gui/cluster_window.py``) did exactly this; every test
closed it with ``window.close()``, which does send the event, so the suite was green while Esc
left the window's number on INTERPRET's tree, kept it — its montage and tile cache with it — in
the page's list until something pruned it, and never saved where it was (review, 2026-11-09).

**The rule.** A ``QDialog`` subclass in ``detail/gui`` does not define ``closeEvent``; it
connects its cleanup to ``finished`` (which every way of closing a dialog emits, the title bar
included: ``QDialog.closeEvent`` calls ``reject``). The behaviour is pinned through real keys in
``tests/gui/test_cluster_window.py::test_every_way_a_window_closes_lets_it_go``.

The sweep that found it (2026-11-09): ``cluster_window.ClusterWindow`` was the one dialog with a
``closeEvent``; ``clip_pane_window`` already let go on ``finished``; ``clip_popup``'s window and
the query-eval explorer clean up on ``destroyed`` under ``WA_DeleteOnClose``, which ``done()``
honours; the ``reject`` overrides (``benchmark``, ``behavior_query``, ``comparison``,
``keypoints_manager``) run for Esc, Cancel and the title bar alike.
"""

from __future__ import annotations

import ast
from pathlib import Path

GUI = Path(__file__).resolve().parents[2] / "detail" / "gui"

#: Qt's dialog classes; the app's own subclasses are found from these.
_QT_DIALOGS = {
    "QDialog",
    "QMessageBox",
    "QInputDialog",
    "QFileDialog",
    "QProgressDialog",
}


def _base_names(node: ast.ClassDef) -> set[str]:
    names = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def dialogs_with_close_event(sources: dict[str, str]) -> list[str]:
    """``file:Class`` for every dialog class in ``sources`` (``{name: text}``) that defines
    ``closeEvent``; dialog classes are found transitively from Qt's."""
    classes = []
    for name, text in sources.items():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ClassDef):
                classes.append((name, node))
    dialogs = set(_QT_DIALOGS)
    grew = True
    while grew:
        grew = False
        for _name, node in classes:
            if node.name not in dialogs and _base_names(node) & dialogs:
                dialogs.add(node.name)
                grew = True
    found = []
    for name, node in classes:
        if node.name in dialogs and node.name not in _QT_DIALOGS:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "closeEvent":
                    found.append(f"{name}:{node.name}")
    return sorted(found)


def test_no_dialog_cleans_up_in_close_event():
    sources = {
        str(path.relative_to(GUI)): path.read_text(encoding="utf-8-sig")
        for path in GUI.rglob("*.py")
    }
    found = dialogs_with_close_event(sources)
    assert not found, (
        f"{found} clean up in closeEvent, which Esc and done() never send: connect the cleanup "
        "to `finished` instead (this module's docstring)."
    )


def test_the_scan_would_catch_a_close_event_on_a_dialog_subclass():
    sources = {
        "base.py": "class Dialog(QDialog):\n    pass\n",
        "window.py": (
            "class Window(Dialog):\n"
            "    def closeEvent(self, event):\n"
            "        self.forget()\n"
        ),
        "deeper.py": (
            "class Deeper(Window):\n"
            "    def closeEvent(self, event):\n"
            "        pass\n"
        ),
        "main.py": (
            "class Main(QMainWindow):\n"
            "    def closeEvent(self, event):\n"
            "        pass\n"
        ),
    }
    assert dialogs_with_close_event(sources) == [
        "deeper.py:Deeper",
        "window.py:Window",
    ]
