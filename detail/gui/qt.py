"""The one module that names the Qt binding: every other file imports Qt from here.

The GUI is built on **PySide6**, The Qt Company's LGPL-3.0 binding. Naming the binding in
exactly one place means a change of binding is a change to this file, and
``tests/test_qt_binding.py`` holds the rest of the repository to it: no file imports
another binding, and nothing but this module imports ``PySide6`` or ``shiboken6``.

A name the GUI needs that is not here yet is added here, in its own module's list, rather
than imported from the binding at the point of use.

The object-lifetime questions the binding answers are wrapped as functions, so callers ask
in words rather than in the binding's vocabulary — and because ``shiboken6.isValid`` answers
the *opposite* question to the one most callers ask (:func:`is_deleted`).
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable

# A second binding already loaded means a library (pyqtgraph picks whichever binding it finds
# first) bound itself to it, and two bindings in one process cannot share a single widget.
# Refused here, by name, rather than left to fail as a type error deep inside a layout.
for _other in ("PyQt6", "PyQt5", "PySide2"):
    if _other in sys.modules:
        raise ImportError(
            f"{_other} is loaded alongside PySide6, which DETAIL's GUI uses. Uninstall it "
            f"from this environment (pip uninstall {_other}) and start again."
        )

import shiboken6
from PySide6.QtCore import QByteArray
from PySide6.QtCore import QCoreApplication
from PySide6.QtCore import QDataStream
from PySide6.QtCore import QEvent
from PySide6.QtCore import QEventLoop
from PySide6.QtCore import QIODevice
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtCore import QLineF
from PySide6.QtCore import QMetaMethod
from PySide6.QtCore import QModelIndex
from PySide6.QtCore import QObject
from PySide6.QtCore import QPoint
from PySide6.QtCore import QPointF
from PySide6.QtCore import QRect
from PySide6.QtCore import QRectF
from PySide6.QtCore import QRunnable
from PySide6.QtCore import QSettings
from PySide6.QtCore import QSize
from PySide6.QtCore import QStandardPaths
from PySide6.QtCore import QThread
from PySide6.QtCore import QThreadPool
from PySide6.QtCore import QTimer
from PySide6.QtCore import QUrl
from PySide6.QtCore import SIGNAL
from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
from PySide6.QtCore import SignalInstance
from PySide6.QtGui import QAction
from PySide6.QtGui import QActionGroup
from PySide6.QtGui import QBrush
from PySide6.QtGui import QCloseEvent
from PySide6.QtGui import QColor
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtGui import QCursor
from PySide6.QtGui import QDesktopServices
from PySide6.QtGui import QEnterEvent
from PySide6.QtGui import QFocusEvent
from PySide6.QtGui import QFont
from PySide6.QtGui import QFontMetrics
from PySide6.QtGui import QFontMetricsF
from PySide6.QtGui import QGuiApplication
from PySide6.QtGui import QIcon
from PySide6.QtGui import QImage
from PySide6.QtGui import QIntValidator
from PySide6.QtGui import QKeyEvent
from PySide6.QtGui import QKeySequence
from PySide6.QtGui import QLinearGradient
from PySide6.QtGui import QMouseEvent
from PySide6.QtGui import QPaintEvent
from PySide6.QtGui import QPainter
from PySide6.QtGui import QPainterPath
from PySide6.QtGui import QPalette
from PySide6.QtGui import QPen
from PySide6.QtGui import QPixmap
from PySide6.QtGui import QPolygonF
from PySide6.QtGui import QRegion
from PySide6.QtGui import QResizeEvent
from PySide6.QtGui import QShortcut
from PySide6.QtGui import QShowEvent
from PySide6.QtGui import QStandardItem
from PySide6.QtGui import QStandardItemModel
from PySide6.QtGui import QTextDocument
from PySide6.QtGui import QValidator
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton
from PySide6.QtWidgets import QAbstractItemView
from PySide6.QtWidgets import QAbstractScrollArea
from PySide6.QtWidgets import QAbstractSlider
from PySide6.QtWidgets import QAbstractSpinBox
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QBoxLayout
from PySide6.QtWidgets import QButtonGroup
from PySide6.QtWidgets import QCheckBox
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QCompleter
from PySide6.QtWidgets import QDialog
from PySide6.QtWidgets import QDialogButtonBox
from PySide6.QtWidgets import QDoubleSpinBox
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QFormLayout
from PySide6.QtWidgets import QFrame
from PySide6.QtWidgets import QGridLayout
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QHeaderView
from PySide6.QtWidgets import QInputDialog
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLayout
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QListWidget
from PySide6.QtWidgets import QListWidgetItem
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QMenu
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QPlainTextEdit
from PySide6.QtWidgets import QProgressBar
from PySide6.QtWidgets import QProxyStyle
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QRadioButton
from PySide6.QtWidgets import QScrollArea
from PySide6.QtWidgets import QScrollBar
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QSlider
from PySide6.QtWidgets import QSpacerItem
from PySide6.QtWidgets import QSpinBox
from PySide6.QtWidgets import QSplitter
from PySide6.QtWidgets import QSplitterHandle
from PySide6.QtWidgets import QStackedWidget
from PySide6.QtWidgets import QStyle
from PySide6.QtWidgets import QStyledItemDelegate
from PySide6.QtWidgets import QStyleFactory
from PySide6.QtWidgets import QStyleOption
from PySide6.QtWidgets import QStyleOptionSlider
from PySide6.QtWidgets import QStyleOptionViewItem
from PySide6.QtWidgets import QTabBar
from PySide6.QtWidgets import QTabWidget
from PySide6.QtWidgets import QTableView
from PySide6.QtWidgets import QTableWidget
from PySide6.QtWidgets import QTableWidgetItem
from PySide6.QtWidgets import QToolButton
from PySide6.QtWidgets import QToolTip
from PySide6.QtWidgets import QTreeWidget
from PySide6.QtWidgets import QTreeWidgetItem
from PySide6.QtWidgets import QTreeWidgetItemIterator
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QWidgetAction


def qwait(ms: int) -> None:
    """Process events for ``ms`` milliseconds, as ``QTest.qWait`` does, with the GIL released.

    PySide6 holds the GIL through ``QTest.qWait`` (and through ``processEvents`` with a time
    limit); PyQt6 released it. Holding it freezes every Python thread for the whole wait — a
    worker a test is waiting on cannot run a line of Python until the wait is over, so the wait
    can never see the result (measured 2026-11-23: a worker counting in a loop advanced 1 step
    in 300ms of ``qWait``, 267 in 300ms of ``QEventLoop.exec``). ``QEventLoop.exec`` releases
    it, so this runs one, then delivers the deferred deletes ``qWait`` would have.
    """
    loop = QEventLoop()
    QTimer.singleShot(max(int(ms), 0), loop.quit)
    loop.exec()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def is_deleted(obj: QObject) -> bool:
    """Whether the C++ object behind ``obj`` has been destroyed.

    The wrapper outlives it: the Python side of a destroyed ``QLabel`` is still an ordinary
    object, and any call that reaches Qt raises ``RuntimeError``. ``shiboken6.isValid`` asks
    the opposite question, so it is inverted here, once.
    """
    return not shiboken6.isValid(obj)


def delete(obj: QObject) -> None:
    """Destroy the C++ object behind ``obj`` now, synchronously (not ``deleteLater``)."""
    shiboken6.delete(obj)


def receivers(obj: QObject, signal: SignalInstance) -> int:
    """How many slots ``signal``, a signal of ``obj``, is connected to.

    What a cue derived from the wiring reads (``receivers(self, self.clicked) > 0``). PySide6's
    ``QObject.receivers`` takes the signal's signature, not the bound signal, so this finds it.
    """
    signature = QMetaMethod.fromSignal(signal).methodSignature().data().decode()
    return obj.receivers(SIGNAL(signature))


def disconnect_if_connected(
    signal: SignalInstance, slot: Callable[..., object] | None = None
) -> bool:
    """Disconnect ``slot`` (every slot, if None) from ``signal``; whether anything was.

    Disconnecting what is not connected is routine — a window closed twice, a row rebuilt before
    it was ever wired — and PySide6 answers it with a ``RuntimeWarning`` rather than an
    exception, which the test suite turns into an error. That one warning is the expected
    answer here, so it is the one silenced.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="libpyside: Failed to disconnect", category=RuntimeWarning
        )
        return signal.disconnect() if slot is None else signal.disconnect(slot)


def created_by_python(obj: QObject) -> bool:
    """Whether Python constructed ``obj`` — as opposed to Qt making it (a completer's popup).

    Not the same question as :func:`owned_by_python`, and the difference is the binding's: a
    read can move ownership to C++ (``QSplitterHandle.splitter()`` does, in PySide6 6.12), so a
    window a test built can stop being "owned by Python" without anything taking it over.
    """
    return shiboken6.createdByPython(obj)


def owned_by_python(obj: QObject) -> bool:
    """Whether Python owns ``obj`` — it has no Qt parent that will delete it."""
    return shiboken6.ownedByPython(obj)


def address(obj: QObject) -> int:
    """The address of the C++ object behind ``obj``: its identity across wrappers."""
    return shiboken6.getCppPointer(obj)[0]


__all__ = [
    "QAbstractButton",
    "QAbstractItemView",
    "QAbstractScrollArea",
    "QAbstractSlider",
    "QAbstractSpinBox",
    "QAction",
    "QActionGroup",
    "QApplication",
    "QBoxLayout",
    "QBrush",
    "QButtonGroup",
    "QByteArray",
    "QCheckBox",
    "QCloseEvent",
    "QColor",
    "QComboBox",
    "QCompleter",
    "QContextMenuEvent",
    "QCoreApplication",
    "QCursor",
    "QDataStream",
    "QDesktopServices",
    "QDialog",
    "QDialogButtonBox",
    "QDoubleSpinBox",
    "QEnterEvent",
    "QEvent",
    "QEventLoop",
    "QFileDialog",
    "QFocusEvent",
    "QFont",
    "QFontMetrics",
    "QFontMetricsF",
    "QFormLayout",
    "QFrame",
    "QGridLayout",
    "QGuiApplication",
    "QHBoxLayout",
    "QHeaderView",
    "QIODevice",
    "QIcon",
    "QImage",
    "QInputDialog",
    "QIntValidator",
    "QItemSelectionModel",
    "QKeyEvent",
    "QKeySequence",
    "QLabel",
    "QLayout",
    "QLineEdit",
    "QLineF",
    "QLinearGradient",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QMenu",
    "QMessageBox",
    "QModelIndex",
    "QMouseEvent",
    "QObject",
    "QPaintEvent",
    "QPainter",
    "QPainterPath",
    "QPalette",
    "QPen",
    "QPixmap",
    "QPlainTextEdit",
    "QPoint",
    "QPointF",
    "QPolygonF",
    "QProgressBar",
    "QProxyStyle",
    "QPushButton",
    "QRadioButton",
    "QRect",
    "QRectF",
    "QRegion",
    "QResizeEvent",
    "QRunnable",
    "QScrollArea",
    "QScrollBar",
    "QSettings",
    "QShortcut",
    "QShowEvent",
    "QSize",
    "QSizePolicy",
    "QSlider",
    "QSpacerItem",
    "QSpinBox",
    "QSplitter",
    "QSplitterHandle",
    "QStackedWidget",
    "QStandardItem",
    "QStandardItemModel",
    "QStandardPaths",
    "QStyle",
    "QStyleFactory",
    "QStyleOption",
    "QStyleOptionSlider",
    "QStyleOptionViewItem",
    "QStyledItemDelegate",
    "QTabBar",
    "QTabWidget",
    "QTableView",
    "QTableWidget",
    "QTableWidgetItem",
    "QTest",
    "QTextDocument",
    "QThread",
    "QThreadPool",
    "QTimer",
    "QToolButton",
    "QToolTip",
    "QTreeWidget",
    "QTreeWidgetItem",
    "QTreeWidgetItemIterator",
    "QUrl",
    "QVBoxLayout",
    "QValidator",
    "QWheelEvent",
    "QWidget",
    "QWidgetAction",
    "Qt",
    "Signal",
    "address",
    "created_by_python",
    "disconnect_if_connected",
    "delete",
    "is_deleted",
    "owned_by_python",
    "qwait",
    "receivers",
]
