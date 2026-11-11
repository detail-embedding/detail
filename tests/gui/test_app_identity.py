"""The desktop app's identity: name + Dock/taskbar icon (detail/gui/__main__.py).

Exercises the ``configure_application`` seam (not ``main``, which blocks on
``app.exec()``) so the icon load + name-setting land in the GUI construction net.
"""

from importlib.resources import files

from detail.gui.__main__ import APP_NAME, configure_application


def test_configure_application_sets_name_and_icon(qapp):
    # Put the test name back after, so later tests keep the identity `qapp` gave them.
    before = qapp.applicationName()
    try:
        configure_application(qapp)
        assert qapp.applicationDisplayName() == APP_NAME
        assert qapp.applicationName() == APP_NAME
        assert not qapp.windowIcon().isNull()  # a real icon, not the blank default
    finally:
        qapp.setApplicationName(before)


def test_icon_resource_ships_with_the_package():
    icon = files("detail.gui") / "resources" / "app_icon.png"
    assert icon.is_file()


def test_macos_menu_name_hook_is_safe_off_platform():
    # The pyobjc patch is best-effort — importing/calling it must never raise, even
    # where Foundation is absent (Linux/Windows CI).
    from detail.gui.__main__ import _set_macos_menu_name

    _set_macos_menu_name(APP_NAME)  # no exception = pass
