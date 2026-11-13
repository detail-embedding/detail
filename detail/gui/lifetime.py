"""Whether a Qt object is still there to be talked to.

A PyQt wrapper outlives the C++ object behind it: the Python side of a destroyed
``QLabel`` is an ordinary, readable object, and every attribute access that reaches
Qt raises ``RuntimeError: wrapped C/C++ object of type QLabel has been deleted``.
Anything that runs *later* than the code that started it — a queued signal, a timer,
a thread's result — has to ask before it touches a widget.

Kept in its own module, importing nothing but PyQt6, so both the widgets
(``busy.py``) and the threads (``workers.py``) can share one answer without either
importing the other.
"""

from PyQt6 import sip
from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import (
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QWidget,
)


def alive(obj: QObject | None) -> bool:
    """Whether ``obj`` is a Qt object that still exists (C++ side not destroyed)."""
    return obj is not None and not sip.isdeleted(obj)


def tree_items(tree: QTreeWidget) -> list[QTreeWidgetItem]:
    """Every item of ``tree``, depth first in display order, collapsed children included.

    The one place a ``QTreeWidgetItemIterator`` is built, so none outlives this call. The
    iterator registers itself with the tree's model and its destructor unregisters itself
    without asking whether the model still exists. Held as a local beside the only reference
    to the tree — a test's frame — it is freed after the tree and writes into freed memory:
    ``corrupted double-linked list``, the process aborted at a later, unrelated free
    (``tests/gui/test_tree_items.py``).
    """
    items = []
    walk = QTreeWidgetItemIterator(tree)
    while walk.value() is not None:
        items.append(walk.value())
        walk += 1
    return items


def cleared(obj: QObject) -> bool:
    """Whether ``obj``'s Python instance state is gone while its C++ side still runs.

    The mirror of :func:`alive`. The cyclic collector *clears* every object in an unreachable
    cycle before it frees any, and for a sip wrapper that empties ``__dict__`` while the C++
    object is still installed as an event filter and still a slot's receiver. Whatever the
    collection then destroys first can send it events: a clip pane window closed by its
    player's ``destroyed`` hid its body, and ``Dialog.eventFilter`` raised ``AttributeError:
    no attribute '_body_widget'`` into the excepthook, blaming whichever test next collected.

    Ask it first in an ``eventFilter`` that reads instance state, and filter nothing when it
    says so. Only meaningful for an object that sets instance state in ``__init__`` — which is
    exactly the object that can be caught without it.
    """
    return not vars(obj)


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
