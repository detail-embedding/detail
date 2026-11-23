"""The shim's own promises: what `detail/gui/qt.py` adds on top of re-exporting PySide6.

The re-exports are exercised by every GUI test; these are the few behaviours the shim itself
owns and that no widget test reaches on its way (#107).
"""

from __future__ import annotations

import subprocess
import sys

from detail.gui.qt import (
    QObject,
    Signal,
    disconnect_if_connected,
    is_deleted,
    receivers,
)


def test_a_second_binding_already_loaded_is_refused_by_name():
    """Two bindings in one process cannot share a widget; pyqtgraph binds to whichever it finds
    first. With another one loaded, the shim refuses and says what to uninstall."""
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, types\n"
            "sys.modules['PyQt6'] = types.ModuleType('PyQt6')\n"
            "import detail.gui.qt\n",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode != 0
    assert "ImportError: PyQt6 is loaded alongside PySide6" in done.stderr
    assert "pip uninstall PyQt6" in done.stderr


class _Source(QObject):
    fired = Signal(int)


def test_receivers_counts_connections_of_a_bound_signal(qapp):
    source = _Source()
    assert receivers(source, source.fired) == 0
    source.fired.connect(lambda _value: None)
    assert receivers(source, source.fired) == 1


def test_disconnecting_what_is_not_connected_is_quiet_and_says_so(qapp):
    """PySide6 answers this with a RuntimeWarning, which `filterwarnings = error` would raise."""
    source = _Source()

    def slot(_value):
        pass

    assert disconnect_if_connected(source.fired, slot) is False
    source.fired.connect(slot)
    assert disconnect_if_connected(source.fired, slot) is True
    assert receivers(source, source.fired) == 0


def test_is_deleted_answers_the_question_it_names(qapp):
    """`shiboken6.isValid` answers the opposite question; the inversion lives in the shim."""
    source = _Source()
    assert not is_deleted(source)
    source.deleteLater()
    from detail.gui.qt import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert is_deleted(source)
