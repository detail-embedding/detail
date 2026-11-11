"""Opening a menu built for one opening, so it is deleted when it closes.

**The class.** A ``QMenu`` built each time it opens — a right-click menu, **Labels ▾** — and
parented to a long-lived widget (so it shows on the right screen and dies with the page) is
kept alive by that parent: closing a menu only hides it. Every opening added another menu, with
its submenus and actions, to the page for as long as the page lived (review, 2026-11-09:
``label_panel``'s **Labels ▾**, the tree's right-click, INTERPRET's and AUDIT's clip menus and
CHECK's list menu).

**The rule.** A menu built per opening is opened through :func:`open_once`, which deletes it
once it has closed. A menu built once and kept (a button's ``setMenu``, **Show ▾**,
**Export ▾**) is not, and its owner opens it as before. Guarded by
``tests/gui/test_menu_once_scan.py``.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QMenu


def open_once(menu: QMenu, where: QPoint, *, wait: bool) -> None:
    """Open ``menu`` at ``where`` (global) and delete it once it has closed.

    ``wait``: run it modally (``exec``), returning once it closed; else ``popup`` it and return
    at once. Either way the deletion is deferred, so a chosen entry's slot — which runs after the
    menu hides — still finds it alive.
    """
    menu.aboutToHide.connect(menu.deleteLater)
    if wait:
        menu.exec(where)
    else:
        menu.popup(where)
