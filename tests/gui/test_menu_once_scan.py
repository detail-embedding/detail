"""A menu built for one opening is opened through ``menus.open_once``, which deletes it.

**The class.** A ``QMenu`` built every time it opens and parented to a long-lived widget stays
alive after it closes — closing only hides a menu — so every opening adds a menu, its submenus
and its actions to the page for the page's life. **Labels ▾** did it on every click
(``label_panel.LabelField.show_list``), and so did the tree's right-click, INTERPRET's and
AUDIT's clip menus and CHECK's list menu (review, 2026-11-09).

**How it hid.** Nothing is wrong on screen: the menu opens, works and closes. The count of
``QMenu`` children only grows, which no behavioural test reads.

**The rule.** In ``detail/gui``, a function that builds a menu — a call to ``QMenu(…)``, a
``…Menu(…)`` subclass or a ``…_menu()`` builder — into a local, and opens that local with
``exec`` or ``popup``, goes through ``menus.open_once`` instead. A menu kept on ``self`` (a
button's ``setMenu``, **Show ▾**, **Export ▾**) is built once and is not this class. The
behaviour is pinned in ``tests/gui/test_label_panel.py::test_opening_labels_menu_leaves_no_menu_behind``.
"""

from __future__ import annotations

import ast
from pathlib import Path

GUI = Path(__file__).resolve().parents[2] / "detail" / "gui"


def _builds_a_menu(call: ast.AST) -> bool:
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    name = (
        func.id
        if isinstance(func, ast.Name)
        else func.attr if isinstance(func, ast.Attribute) else ""
    )
    return name.lower().endswith("menu")


def menus_opened_and_kept(sources: dict[str, str]) -> list[str]:
    """``file:function`` for every function in ``sources`` that opens a menu it built itself
    with ``exec`` / ``popup`` rather than ``open_once``."""
    found = []
    for name, text in sources.items():
        for func in ast.walk(ast.parse(text)):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            built = {
                target.id
                for node in ast.walk(func)
                if isinstance(node, ast.Assign) and _builds_a_menu(node.value)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            for node in ast.walk(func):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("exec", "popup")
                ):
                    receiver = node.func.value
                    if (
                        isinstance(receiver, ast.Name) and receiver.id in built
                    ) or _builds_a_menu(receiver):
                        found.append(f"{name}:{func.name}")
    return sorted(set(found))


def test_a_menu_built_per_opening_is_opened_once():
    sources = {
        str(path.relative_to(GUI)): path.read_text(encoding="utf-8-sig")
        for path in GUI.rglob("*.py")
        if path.name != "menus.py"
    }
    found = menus_opened_and_kept(sources)
    assert not found, (
        f"{found} open a menu they built with exec/popup, which leaves it alive under its "
        "parent: open it with `menus.open_once` (this module's docstring)."
    )


def test_the_scan_would_catch_each_way_a_menu_is_built_and_opened():
    sources = {
        "a.py": "def f(self, p):\n    menu = QMenu(self)\n    menu.exec(p)\n",
        "b.py": "def f(self, p):\n    m = self.labels_menu()\n    m.popup(p)\n",
        "c.py": "def f(self, p):\n    self.context_menu(1).exec(p)\n",
        "d.py": "def f(self, p):\n    menu = _LabelsMenu(self)\n    menu.exec(p)\n",
        "kept.py": "def f(self, p):\n    self.menu.exec(p)\n",
        "once.py": "def f(self, p):\n    menu = QMenu(self)\n    open_once(menu, p, wait=True)\n",
        "box.py": "def f(self):\n    box = QMessageBox(self)\n    box.exec()\n",
    }
    assert menus_opened_and_kept(sources) == ["a.py:f", "b.py:f", "c.py:f", "d.py:f"]
