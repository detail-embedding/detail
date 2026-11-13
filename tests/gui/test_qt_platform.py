"""Which Qt display platform the GUI uses on WSL (§ GUI launch).

**The class: a framework default that is wrong for one host, found only on that host.** On WSL
Qt prefers its Wayland platform, and the Wayland plugin's popup handling killed the app with a
native stack overflow whenever a drop-down opened inside a dialog opened from another dialog. It
happened three times in the New reference set editor, with no Python traceback and nothing
printed; the same dialog ran cleanly on Windows, on the offscreen platform, and (the proof) on
Qt's X11 platform under the same WSL. A deferral of the editor's own rebuilds (#668) did not
change it, because nothing in DETAIL's code was the cause.

**It hid because every test and every developer machine runs somewhere else**: CI is Linux
without a display, the dev boxes are macOS and Windows, and the one WSL box is the one where it
crashes. So the choice is a pure function of what the host says (is it WSL, is there an X server,
did the person choose a platform, does the X11 plugin load), and these tests give it each answer.
The last test goes through ``main()`` to the environment variable Qt reads, because the function
being right and never being called is the same bug.
"""

import os

import pytest

pytest.importorskip("PyQt6")

from detail.gui import __main__ as launcher  # noqa: E402


def _never():
    raise AssertionError("the X11 plugin was probed when nothing needed it")


def test_wsl_with_a_loadable_x11_plugin_uses_x11():
    environ = {"DISPLAY": ":0", "WAYLAND_DISPLAY": "wayland-0"}

    chosen = launcher.choose_qt_platform(environ, wsl=True, x11_loads=lambda: True)

    assert chosen == ("xcb", None)


def test_wsl_without_the_x11_libraries_stays_on_wayland_and_says_how_to_get_x11():
    chosen = launcher.choose_qt_platform(
        {"DISPLAY": ":0"}, wsl=True, x11_loads=lambda: False
    )

    platform, message = chosen
    assert (
        platform is None
    )  # Qt's own choice; not forced onto a plugin that cannot load
    assert "sudo apt install -y" in message
    assert "libxcb-cursor0" in message  # the library that was actually missing
    assert "QT_QPA_PLATFORM=wayland" in message  # and how to say "I know"


def test_a_platform_the_person_chose_always_wins_and_is_not_probed():
    for choice in ("wayland", "offscreen", "xcb"):
        environ = {"DISPLAY": ":0", "QT_QPA_PLATFORM": choice}
        assert launcher.choose_qt_platform(environ, wsl=True, x11_loads=_never) == (
            None,
            None,
        )


def test_off_wsl_nothing_is_changed_and_nothing_is_probed():
    """macOS, Windows and a plain Linux desktop keep Qt's own choice."""
    environ = {"DISPLAY": ":0"}

    assert launcher.choose_qt_platform(environ, wsl=False, x11_loads=_never) == (
        None,
        None,
    )


def test_without_an_x_server_to_ask_nothing_is_changed():
    assert launcher.choose_qt_platform({}, wsl=True, x11_loads=_never) == (None, None)


@pytest.mark.parametrize(
    "release, expected",
    [
        ("6.18.40.1-microsoft-standard-WSL2\n", True),
        ("5.15.153.1-Microsoft-standard-WSL2\n", True),
        ("6.8.0-45-generic\n", False),
    ],
)
def test_wsl_is_read_from_the_kernel_release(tmp_path, release, expected):
    path = tmp_path / "osrelease"
    path.write_text(release, encoding="utf-8")

    assert launcher.running_under_wsl(str(path)) is expected


def test_a_host_with_no_proc_is_not_wsl(tmp_path):
    assert launcher.running_under_wsl(str(tmp_path / "missing")) is False


def test_the_real_plugin_probe_answers_yes_or_no_and_never_raises():
    assert launcher.x11_plugin_loads() in (True, False)


def test_main_hands_the_choice_to_qt_before_the_application_exists(monkeypatch):
    """Qt reads ``QT_QPA_PLATFORM`` when ``QApplication`` is built, so it has to be set first."""
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(launcher, "running_under_wsl", lambda: True)
    monkeypatch.setattr(launcher, "x11_plugin_loads", lambda: True)
    seen = {}

    class Stop(Exception):
        pass

    def fake_application(argv):
        seen["platform"] = os.environ.get("QT_QPA_PLATFORM")
        raise Stop

    monkeypatch.setattr(launcher, "QApplication", fake_application)

    with pytest.raises(Stop):
        launcher.main()

    assert seen["platform"] == "xcb"


def test_main_prints_the_hint_when_x11_cannot_load(monkeypatch, capsys):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(launcher, "running_under_wsl", lambda: True)
    monkeypatch.setattr(launcher, "x11_plugin_loads", lambda: False)

    class Stop(Exception):
        pass

    def fake_application(argv):
        raise Stop

    monkeypatch.setattr(launcher, "QApplication", fake_application)

    with pytest.raises(Stop):
        launcher.main()

    assert "sudo apt install -y" in capsys.readouterr().err
    assert os.environ.get("QT_QPA_PLATFORM") is None  # Wayland left alone
