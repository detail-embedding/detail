"""``python -m detail.gui`` — launch the desktop GUI."""

import sys
import traceback
from importlib.resources import files

from detail.gui import ui_thread_gc
from detail.gui.app import MainWindow
from detail.gui.qt import QApplication, QIcon, QMessageBox
from detail.gui.style import apply_style

APP_NAME = "detail"


def _set_macos_menu_name(name: str) -> None:
    """Set the macOS menu-bar app name (the bold entry next to the Apple menu).

    That name is the bundle's ``CFBundleName``; Qt can't set it for an unbundled
    ``python`` process, so patch the bundle info dict via pyobjc. Must run *before*
    ``QApplication`` (which starts ``NSApplication`` and reads it). Best-effort: a
    no-op off macOS or when ``pyobjc-framework-Cocoa`` isn't installed.
    """
    try:
        from Foundation import NSBundle

        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        info["CFBundleName"] = name
    except Exception:  # noqa: BLE001 — cosmetic; never block launch on it
        pass


def configure_application(app: QApplication) -> None:
    """Apply the app's visual identity: Fusion style, name (window titles / taskbar),
    and the window/Dock icon. Split out from ``main`` so it's testable without
    ``app.exec()``."""
    apply_style(app)  # Fusion, one scrollbar look (gui/style.py, gui_design.md §3.1)
    app.setOrganizationName(
        APP_NAME
    )  # so a default QSettings() resolves a stable store
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)  # shown in window titles
    app.setDesktopFileName(APP_NAME)  # Linux: link to the .desktop entry
    icon_path = files("detail.gui") / "resources" / "app_icon.png"
    app.setWindowIcon(QIcon(str(icon_path)))  # Dock (macOS) / taskbar icon


def install_excepthook() -> None:
    """Tell the user about an exception that escapes a Qt slot, and keep the app running.

    A *single* unguarded read of something that vanished — a deleted experiment, a video root
    not registered on this machine — raises inside a slot or a Qt virtual override. The binding
    decides what happens next, so the app does not leave it to the binding: PyQt6, the binding
    before #107, aborted the process; PySide6 prints the traceback to a terminal a user who
    launched from an icon never sees, and carries on as if the action had worked. Neither is
    right for a desktop app: the user should be told, and left holding their work.

    This is an explicit isolation boundary (CLAUDE.md's carve-out for catching broadly).
    It is deliberately installed only by :func:`main`, never at import: under pytest the
    runner's own hook must keep failing tests loudly. ``KeyboardInterrupt`` passes
    straight through, and the traceback still goes to stderr, so nothing is hidden.
    """
    previous = sys.excepthook

    def hook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous(exc_type, exc, tb)
            return
        traceback.print_exception(exc_type, exc, tb)
        try:
            QMessageBox.critical(
                None,
                "Something went wrong",
                f"{exc_type.__name__}: {exc}\n\n"
                "The app is still running — your unsaved work is intact. "
                "Details were written to the terminal.",
            )
        except Exception:  # a dialog must never be the thing that kills the app
            pass

    sys.excepthook = hook


def main() -> int:
    _set_macos_menu_name(APP_NAME)  # before QApplication — see the helper's docstring
    app = QApplication(sys.argv)
    install_excepthook()  # before any window exists, so early failures are reported too
    configure_application(app)
    ui_thread_gc.install(app)  # the collector never runs on a worker thread
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
