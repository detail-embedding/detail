"""Every test process keeps its view settings in a store no other process can touch.

**The class.** ``_forget_view_preferences`` clears the settings store around every GUI test, and a
test that checks something is *remembered* writes a key and reads it back. Any store two processes
share lets one clear another's key between that write and that read. The store was first named per
xdist worker, which separated the workers of one run and nothing else: every concurrent run on the
machine — another worktree, another agent's session — has a ``gw0`` too, and Qt's test mode put
them all in one directory (macOS's native store ignores test mode altogether). The "remembers"
tests failed in full runs only, on whichever run lost the race. ``qapp`` now points the default
(INI) store into the process's own basetemp.
"""

from pathlib import Path


def test_the_settings_store_is_inside_this_processs_basetemp(qapp, tmp_path_factory):
    from PyQt6.QtCore import QSettings

    store = Path(QSettings().fileName()).resolve()
    assert store.is_relative_to(tmp_path_factory.getbasetemp().resolve()), (
        f"QSettings() writes {store}, outside this process's basetemp — a store another "
        "pytest run on the machine can clear mid-test"
    )


def test_renaming_the_app_does_not_move_the_store_out(qapp, tmp_path_factory):
    """``configure_application`` renames the app mid-test; the store must stay private."""
    from PyQt6.QtCore import QSettings

    before = qapp.applicationName()
    try:
        qapp.setApplicationName("detail")
        store = Path(QSettings().fileName()).resolve()
    finally:
        qapp.setApplicationName(before)
    assert store.is_relative_to(tmp_path_factory.getbasetemp().resolve()), store
