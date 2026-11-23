"""Every window a GUI test opens is destroyed at that test's end, by Qt, while it is whole.

**The class: a widget left for the cyclic collector**, written up on ``_windows_die_with_their_test``
in ``tests/gui/conftest.py``: a dropped window is cleared by the collector before Qt destroys it,
and its event filters and ``destroyed`` slots then run on objects with no attributes and functions
with no globals — a segfault and two "has no attribute" errors, each in a later test that never
touched the window.

- ``test_a_window_a_test_opens`` / ``test_is_gone_when_the_next_test_starts`` — the fixture runs, in
  order, in one file (``--dist loadfile`` keeps a file on one worker).
- ``test_only_windows_python_owns_are_destroyed`` — a top-level something else owns (a completer's
  popup) is left to its owner: deleting it here was a double free, a segfault in ``~QCompleter``.
"""

from __future__ import annotations

from detail.gui.qt import (
    QCompleter,
    QLineEdit,
    QWidget,
    address,
    is_deleted,
)

from tests.gui.conftest import destroy_new_windows

_OPENED: list[QWidget] = []


def test_a_window_a_test_opens(qapp):
    window = QWidget()
    window.show()
    _OPENED.append(window)  # kept alive past the test, the way a leaked cycle keeps one


def test_is_gone_when_the_next_test_starts(qapp):
    assert _OPENED, "runs after test_a_window_a_test_opens"
    assert is_deleted(_OPENED[0]), "the previous test's window outlived its test"


def test_only_windows_python_owns_are_destroyed(qapp):
    before = {address(w) for w in qapp.topLevelWidgets()}
    edit = QLineEdit()
    completer = QCompleter(["alpha", "beta"], edit)
    edit.setCompleter(completer)
    popup = completer.popup()  # a top-level the completer owns
    assert popup.isWindow()
    assert destroy_new_windows(qapp, before) == 1
    assert is_deleted(edit)
    del completer, popup, edit
