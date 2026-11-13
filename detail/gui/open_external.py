"""Opening something outside the app: a web page in the browser, a folder or file in the OS.

The one place in ``detail/gui`` that calls ``QDesktopServices.openUrl``
(``tests/gui/test_open_external_scan.py`` fails on any other). ``openUrl`` returns a ``bool``,
and ``False`` is silent: no exception, only a ``qt.qpa.services`` line in the terminal. It is
the normal answer inside WSL, which has no ``xdg-open`` browser or file manager, so there a
click falls back to the Windows side (``explorer.exe``). Anywhere else, and if that launch
fails too, the click says so, naming the link or path, so it is never a click that did
nothing.

The launch is spawned, not waited on: ``explorer.exe`` usually returns at once, but a click
must not freeze the window on the day it does not, so the child is reaped on a daemon thread
and killed after :data:`_LAUNCH_TIMEOUT_S`.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QGuiApplication
from PyQt6.QtWidgets import QMessageBox, QWidget

from detail.api import checks as api_checks

_LAUNCH_TIMEOUT_S = 30


def open_url(widget: QWidget | None, url: str) -> None:
    """Open web ``url`` in the user's browser, and say so when it cannot be opened."""
    qurl = QUrl(url)
    if qurl.isLocalFile():
        open_local_path(widget, Path(qurl.toLocalFile()), what="page")
        return
    if QDesktopServices.openUrl(qurl):
        return
    argv = api_checks.wsl_browser_argv(url)
    if argv is not None and _launch(argv):
        return
    show_link(widget, url)


def open_local_path(widget: QWidget | None, path: Path, what: str = "folder") -> None:
    """Open ``path`` in the OS's file manager or default handler (#475), and warn when it cannot.

    Inside WSL the fallback is Windows Explorer, on the ``wslpath`` translation of ``path``.
    """
    if QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
        return
    argv = api_checks.wsl_explorer_argv(path)
    if argv is not None and _launch(argv):
        return
    QMessageBox.warning(widget, f"Open {what}", f"Could not open a {what} for:\n{path}")


def _launch(argv: list[str]) -> bool:
    """Start ``argv`` without waiting on it; ``False`` when it could not start.

    The exit code is not read: ``explorer.exe`` exits 1 even when it opened what it was given.
    """
    try:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    threading.Thread(target=_reap, args=(child,), daemon=True).start()
    return True


def _reap(child: subprocess.Popen) -> None:
    try:
        child.wait(timeout=_LAUNCH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def link_message(widget: QWidget | None, url: str) -> tuple[QMessageBox, object]:
    """The message shown when no browser could be opened: the link as selectable text, and a
    **Copy link** button (returned beside the box) that puts it on the clipboard."""
    box = QMessageBox(widget)
    box.setWindowTitle("Open link")
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setText(
        f"No browser could be opened from here. Open this link yourself:\n\n{url}"
    )
    box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    copy = box.addButton("Copy link", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Close)
    return box, copy


def show_link(widget: QWidget | None, url: str) -> None:
    """Show :func:`link_message`, copying the link when **Copy link** closed it."""
    box, copy = link_message(widget, url)
    box.exec()
    if box.clickedButton() is copy:
        QGuiApplication.clipboard().setText(url)
