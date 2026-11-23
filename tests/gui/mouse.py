"""A synthetic mouse event, built the one way Qt 6 has not deprecated.

``QMouseEvent(type, localPos, button, buttons, modifiers)`` — the constructor without a global
position — is deprecated in Qt 6. PyQt6 never said so; PySide6 raises a ``DeprecationWarning``,
which the suite's ``filterwarnings = error`` makes a failure (#107). The deprecated constructor
took the global position from ``QCursor.pos()``, so that is what :func:`mouse_event` passes:
the events a test sends are the ones it sent before.
"""

from __future__ import annotations

from detail.gui.qt import QCursor, QEvent, QMouseEvent, QPointF, Qt


def mouse_event(
    kind: QEvent.Type,
    local: QPointF,
    button: Qt.MouseButton,
    buttons: Qt.MouseButton,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> QMouseEvent:
    """A mouse event at ``local`` in the receiving widget's coordinates."""
    return QMouseEvent(kind, local, QPointF(QCursor.pos()), button, buttons, modifiers)
