"""Whether a Qt object is still there to be talked to.

A Python wrapper outlives the C++ object behind it: the Python side of a destroyed
``QLabel`` is an ordinary, readable object, and every attribute access that reaches
Qt raises ``RuntimeError: Internal C++ object (QLabel) already deleted.``
Anything that runs *later* than the code that started it — a queued signal, a timer,
a thread's result — has to ask before it touches a widget.

Kept in its own module, importing nothing but the Qt shim, so both the widgets
(``busy.py``) and the threads (``workers.py``) can share one answer without either
importing the other.
"""

from detail.gui.qt import (
    QObject,
    QWidget,
    is_deleted,
)


def alive(obj: QObject | None) -> bool:
    """Whether ``obj`` is a Qt object that still exists (C++ side not destroyed)."""
    return obj is not None and not is_deleted(obj)


def discard(widget: QWidget | None) -> None:
    """Take ``widget`` off the screen now, and delete it when the event loop next turns.

    ``deleteLater`` alone leaves the widget a visible child of its parent — still in its layout,
    still returned by ``findChildren`` — until deferred deletes run, which a bare
    ``processEvents`` does not do. A page that rebuilt its body that way showed the old
    "Edit scene geometry…" button (disabled) beside the new one, and a lookup by text found the
    old one first: 9 of 86 full runs of ``test_recording_detail_preview_follows_an_added_video_part``.
    Hidden and unparented, it is gone from the page at once; ``setParent(None)`` also takes it
    out of the parent's layout.
    """
    if not alive(widget):
        return
    widget.hide()
    widget.setParent(None)
    widget.deleteLater()
