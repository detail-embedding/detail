"""Opening a link or a path outside the app (``gui/open_external.py``): never a silent click.

``QDesktopServices.openUrl`` answers ``False`` when no handler is registered, and that is the
normal answer inside WSL, where a click on **Download Ollama** only printed
``qt.qpa.services: Unable to detect a web browser`` to the terminal. These tests stand in the
two things a machine decides: what Qt answers (``openUrl``) and whether this is WSL
(``commands._is_wsl``); the Windows launcher is a faked ``subprocess.Popen``, so nothing here
opens a browser. Each site that opens a web link is clicked through its real entry point, so a
site that stops routing through the helper fails here as well as in the source scan
(``test_open_external_scan.py``).
"""

from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QGuiApplication  # noqa: E402
from PyQt6.QtWidgets import QMessageBox  # noqa: E402

from detail.checks import commands as commands_module  # noqa: E402
from detail.gui import help_ui  # noqa: E402
from detail.gui import open_external  # noqa: E402
from detail.gui import pages as gui_pages  # noqa: E402

URL = "https://huggingface.co/settings/tokens"


class _Child:
    """A spawned launcher that has already exited."""

    def wait(self, timeout=None):
        return 1


@pytest.fixture
def machine(monkeypatch):
    """Qt finds no browser; ``wsl`` decides whether this is WSL. Records what was launched
    and every message box shown (its ``exec`` is stood in, so none blocks)."""
    seen = {"launched": [], "shown": [], "wsl": False, "click": None}
    monkeypatch.setattr(open_external.QDesktopServices, "openUrl", lambda url: False)
    monkeypatch.setattr(commands_module, "_is_wsl", lambda: seen["wsl"])
    monkeypatch.setattr(
        open_external.subprocess,
        "Popen",
        lambda argv, **kwargs: seen["launched"].append(argv) or _Child(),
    )

    def exec_(box):
        seen["shown"].append(box)
        if seen["click"] is not None:
            next(b for b in box.buttons() if b.text() == seen["click"]).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    return seen


def test_a_link_qt_cannot_open_goes_to_the_windows_browser_inside_wsl(qapp, machine):
    machine["wsl"] = True
    open_external.open_url(None, URL)
    assert machine["launched"] == [["explorer.exe", URL]]
    assert machine["shown"] == [], "a launched browser needs no message"


def test_a_link_qt_cannot_open_elsewhere_is_shown_to_copy(qapp, machine):
    open_external.open_url(None, URL)
    assert machine["launched"] == []
    [box] = machine["shown"]
    assert box.text().endswith(f"\n\n{URL}")
    assert (
        box.textInteractionFlags()
        & open_external.Qt.TextInteractionFlag.TextSelectableByMouse
    )
    assert [b.text() for b in box.buttons() if b.text() == "Copy link"] == ["Copy link"]


def test_copy_link_puts_the_link_on_the_clipboard(qapp, machine):
    QGuiApplication.clipboard().setText("")
    machine["click"] = "Copy link"
    open_external.open_url(None, URL)
    assert QGuiApplication.clipboard().text() == URL


def test_a_windows_launcher_that_cannot_start_still_shows_the_link(
    qapp, machine, monkeypatch
):
    machine["wsl"] = True

    def missing(argv, **kwargs):
        raise FileNotFoundError("explorer.exe")

    monkeypatch.setattr(open_external.subprocess, "Popen", missing)
    open_external.open_url(None, URL)
    [box] = machine["shown"]
    assert URL in box.text()


def test_a_link_qt_opens_needs_nothing_else(qapp, machine, monkeypatch):
    machine["wsl"] = True
    monkeypatch.setattr(open_external.QDesktopServices, "openUrl", lambda url: True)
    open_external.open_url(None, URL)
    assert machine["launched"] == [] and machine["shown"] == []


def test_a_file_url_is_opened_as_a_path(qapp, machine, monkeypatch, tmp_path):
    """The offline docs fall back to a ``file://`` URL when loopback cannot bind; Explorer
    needs that path translated, which is the local-path helper's job."""
    machine["wsl"] = True
    monkeypatch.setattr(
        open_external.api_checks,
        "wsl_explorer_argv",
        lambda path: ["explorer.exe", f"WIN:{path.name}"],
    )
    open_external.open_url(None, (tmp_path / "index.html").as_uri())
    assert machine["launched"] == [["explorer.exe", "WIN:index.html"]]


def test_a_launcher_that_hangs_is_killed_not_waited_on(qapp):
    calls = []

    class Hung:
        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if len(calls) == 1:
                raise subprocess.TimeoutExpired("explorer.exe", timeout)

        def kill(self):
            calls.append(("kill", None))

    open_external._reap(Hung())
    assert calls == [
        ("wait", open_external._LAUNCH_TIMEOUT_S),
        ("kill", None),
        ("wait", None),
    ]


# --------------------------------------------------------------------------- #
# Every site that opens a web link, through its real entry point
# --------------------------------------------------------------------------- #
def _click_link_button(monkeypatch):
    gui_pages._link_button("Create token ↗", URL).click()


def _documentation_page(monkeypatch):
    monkeypatch.setattr(gui_pages.docs_access, "needs_build", lambda: False)
    monkeypatch.setattr(
        gui_pages.docs_access, "local_url", lambda page="index", anchor=None: URL
    )
    page = gui_pages.DocumentationPage()
    page._open.click()


def _documentation_page_after_a_build(monkeypatch):
    page = gui_pages.DocumentationPage()
    page._on_built(URL)


def _help_popover_open_full_docs(monkeypatch):
    monkeypatch.setattr(help_ui.docs_access, "needs_build", lambda: False)
    monkeypatch.setattr(
        help_ui.docs_access, "local_url", lambda page="index", anchor=None: URL
    )
    button = help_ui.help_button("text-prompt")
    button.click()
    button._popover._open_full()


def _help_popover_link(monkeypatch):
    button = help_ui.help_button("text-prompt")
    button.click()
    button._popover._on_link(URL)


@pytest.mark.parametrize(
    "site",
    [
        _click_link_button,
        _documentation_page,
        _documentation_page_after_a_build,
        _help_popover_open_full_docs,
        _help_popover_link,
    ],
)
def test_every_web_link_falls_back_to_the_windows_browser_inside_wsl(
    qapp, machine, monkeypatch, site
):
    """Every site that opens a web link (Download Ollama, the sixth, went with Describe)."""
    machine["wsl"] = True
    site(monkeypatch)
    assert machine["launched"] == [["explorer.exe", URL]]
