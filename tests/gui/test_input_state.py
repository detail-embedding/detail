"""No GUI test inherits a modifier key or a mouse button "held" by an earlier one.

The class and its history are in the docstring of ``conftest._no_input_held_from_an_earlier_test``,
the autouse fixture this guards. In short: a simulated input event carrying a modifier leaves
``QGuiApplication.keyboardModifiers()`` reading it until the next input event, a call with no
event of its own (``QTableView.selectRow``) reads that state, and the test that pays is
whichever one ``--dist loadfile`` runs next on that worker.

The pair below is order-dependent on purpose: the first leaves Shift held exactly as a real test
can, the second is the next test in the file (``loadfile`` keeps a file on one worker, in file
order) and must start with nothing held. With the fixture removed the second one fails.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QGuiApplication  # noqa: E402
from PyQt6.QtGui import QStandardItem  # noqa: E402
from PyQt6.QtGui import QStandardItemModel  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QAbstractItemView  # noqa: E402
from PyQt6.QtWidgets import QLineEdit  # noqa: E402
from PyQt6.QtWidgets import QTableView  # noqa: E402

from tests.gui.conftest import release_held_input  # noqa: E402

NONE = Qt.KeyboardModifier.NoModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier


def _hold_shift(qapp):
    """Leave Shift held the way a test can: a Shift key click, through QTest."""
    edit = QLineEdit()
    edit.show()
    qapp.processEvents()
    QTest.keyClick(edit, Qt.Key.Key_Right, SHIFT)
    edit.close()


def _table_of(rows: int) -> QTableView:
    view = QTableView()
    model = QStandardItemModel(rows, 1)
    for r in range(rows):
        model.setItem(r, 0, QStandardItem(str(r)))
    view.setModel(model)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    return view


def test_the_premise_a_held_shift_breaks_select_row(qapp):
    """Why this matters: with Shift held, ``selectRow`` selects nothing in a fresh table."""
    _hold_shift(qapp)
    assert (
        QGuiApplication.keyboardModifiers() == SHIFT
    ), "QTest no longer leaves Shift held"
    view = _table_of(3)
    view.selectRow(1)
    assert not view.selectionModel().hasSelection()


def test_the_next_test_starts_with_nothing_held(qapp):
    """Runs straight after the premise above, which left Shift held."""
    assert QGuiApplication.keyboardModifiers() == NONE
    assert QGuiApplication.mouseButtons() == Qt.MouseButton.NoButton
    view = _table_of(3)
    view.selectRow(1)
    assert view.selectionModel().hasSelection()


def test_release_held_input_clears_a_held_modifier_and_button(qapp):
    _hold_shift(qapp)
    edit = QLineEdit()
    edit.show()
    QTest.mousePress(edit, Qt.MouseButton.LeftButton, SHIFT)
    assert QGuiApplication.mouseButtons() == Qt.MouseButton.LeftButton
    release_held_input(qapp)
    assert QGuiApplication.keyboardModifiers() == NONE
    assert QGuiApplication.mouseButtons() == Qt.MouseButton.NoButton
    edit.close()
