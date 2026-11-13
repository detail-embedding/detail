"""``python -m detail.gui`` — launch the desktop GUI."""

import ctypes
import faulthandler
import os
import sys
import threading
import traceback
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox

from detail.gui import ui_thread_gc
from detail.gui.app import MainWindow
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


class _ErrorRelay(QObject):
    """Shows an error dialog on the UI thread, whichever thread asks for it.

    It lives on the thread that created it (the UI thread: :func:`install_excepthook` is called
    from :func:`main`), so a signal emitted from a worker thread is *queued* and the slot runs
    where widgets may be built. ``QMessageBox`` built on a worker thread is what the hook did
    before.
    """

    reported = pyqtSignal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.reported.connect(self._show)

    def _show(self, title: str, text: str) -> None:
        try:
            QMessageBox.critical(None, title, text)
        except Exception:  # a dialog must never be the thing that kills the app
            pass


def install_excepthook() -> None:
    """Report an exception that escapes a Qt slot instead of killing the app.

    PyQt6's default hook ends in ``qFatal()`` → SIGABRT, so a *single* unguarded read of
    something that vanished — a deleted experiment, a video root not registered on this
    machine — takes the whole window down mid-session, with no dialog and any unsaved
    edits lost. Nothing in a desktop app justifies that: the user should be told and left
    holding their work.

    This is an explicit isolation boundary (CLAUDE.md's carve-out for catching broadly).
    It is deliberately installed only by :func:`main`, never at import: under pytest the
    runner's own hook must keep failing tests loudly. ``KeyboardInterrupt`` passes
    straight through, and the traceback still goes to stderr, so nothing is hidden.

    **The hook runs in the thread that raised.** A worker's exception arrives here on the
    worker's thread, and a dialog built there is not a report but a second fault: Qt printed
    ``QBasicTimer::stop: Failed. Possibly trying to stop from a different thread`` ten times and
    ``QBackingStore::endPaint() called with active painter``, and the process died with a
    segmentation fault, in the session where a progress worker's ``ValueError`` was the thing
    being reported. So off the main thread the dialog is handed to the UI thread.
    """
    previous = sys.excepthook
    relay = _ErrorRelay()

    def hook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous(exc_type, exc, tb)
            return
        traceback.print_exception(exc_type, exc, tb)
        title = "Something went wrong"
        text = (
            f"{exc_type.__name__}: {exc}\n\n"
            "The app is still running — your unsaved work is intact. "
            "Details were written to the terminal."
        )
        if threading.current_thread() is threading.main_thread():
            relay._show(title, text)
        else:
            relay.reported.emit(title, text)

    sys.excepthook = hook


def enable_crash_traces() -> None:
    """Print every thread's Python stack when the process takes a fatal signal.

    A segmentation fault inside Qt otherwise ends as the shell's one line, ``Segmentation fault``,
    with nothing saying which of the app's threads was doing what; the pasted terminal then
    holds no more than the Qt warnings before it. Done in :func:`main`, not at import.

    A diagnostic, so it never blocks a launch: with no usable stderr (``pythonw``, a captured
    stream) ``faulthandler`` cannot attach and the app starts without it.
    """
    try:
        faulthandler.enable(all_threads=True)
    except (OSError, RuntimeError, ValueError):
        pass


#: What Qt's X11 platform plugin needs that a bare WSL Ubuntu does not have.
X11_PACKAGES = (
    "libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 "
    "libxcb-shape0 libxcb-xkb1 libxcb-util1 libxkbcommon-x11-0"
)


def running_under_wsl(osrelease: str = "/proc/sys/kernel/osrelease") -> bool:
    """Whether this Linux is WSL (its kernel release says ``microsoft``)."""
    try:
        return "microsoft" in Path(osrelease).read_text(encoding="utf-8-sig").lower()
    except OSError:  # not Linux, or no /proc: not WSL
        return False


def x11_plugin_loads() -> bool:
    """Whether Qt's ``xcb`` plugin and every library it links are present.

    Loading the plugin file is the test: a missing ``libxcb-cursor`` fails *there*, where Qt would
    otherwise abort at startup ("could not load the Qt platform plugin xcb").
    """
    from PyQt6.QtCore import QLibraryInfo

    plugin = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
    plugin = plugin / "platforms" / "libqxcb.so"
    try:
        ctypes.CDLL(str(plugin))
    except OSError:
        return False
    return True


def choose_qt_platform(
    environ, *, wsl: bool, x11_loads: Callable[[], bool]
) -> tuple[str | None, str | None]:
    """``(platform to use, a message for the person)``; ``None`` leaves Qt's own choice.

    **On WSL, X11 rather than Wayland.** WSLg offers both and Qt prefers Wayland, whose popup
    handling killed the app with a native stack overflow (no traceback, nothing printed) whenever a
    drop-down opened inside a dialog opened from another dialog: three crashes in a row in the New
    reference set editor, none of them while the same dialog ran on X11, Windows or the offscreen
    platform. An explicit ``QT_QPA_PLATFORM`` always wins, and so does having no X server to ask.
    When the X11 plugin cannot load, Wayland stays and the message says how to get X11.
    """
    if not wsl or environ.get("QT_QPA_PLATFORM") or not environ.get("DISPLAY"):
        return None, None
    if x11_loads():
        return "xcb", None
    return None, (
        "detail: this is WSL running Qt on Wayland, where drop-down menus inside dialogs have "
        "crashed the app. For the X11 display, run once:\n"
        f"  sudo apt install -y {X11_PACKAGES}\n"
        "(Set QT_QPA_PLATFORM=wayland to keep Wayland and hide this message.)"
    )


def main() -> int:
    enable_crash_traces()
    _set_macos_menu_name(APP_NAME)  # before QApplication — see the helper's docstring
    platform, message = choose_qt_platform(
        os.environ, wsl=running_under_wsl(), x11_loads=x11_plugin_loads
    )
    if platform is not None:
        os.environ["QT_QPA_PLATFORM"] = platform
    if message is not None:
        print(message, file=sys.stderr)
    app = QApplication(sys.argv)
    install_excepthook()  # before any window exists, so early failures are reported too
    configure_application(app)
    ui_thread_gc.install(app)  # the collector never runs on a worker thread
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
