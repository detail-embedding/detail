"""Text on screen can be selected and copied.

Qt labels are not selectable by default, so by default nothing the app says can be quoted back.
The guidelines have asked for the opposite for a long time, *"so people can copy it (e.g. to
quote a label back when reporting a bug)"*.

**Pages get it; dialogs do not, and the obvious fix for that crashes.** `MainWindow._show` calls
this on every page, which is why a page's text is copyable. A dialog never passes through the
shell, so ~100 labels across ~30 dialogs are not — and a dialog is where the copyable text
usually is, because that is where the error message appears.

The near fix at the time, `sizing.finish_dialog`, was not a shared seam: **8 of 44 dialogs called
it**, and 38 inherited `QDialog` directly with no base to hook. Both halves have since been fixed
the other way round — every dialog now inherits `gui.dialog.Dialog`, which calls this from
`showEvent`, and `finish_dialog` is gone.

**Do not reach for an application-wide event filter.** It was tried, and it segfaults —
reproducibly, roughly one run in two of the suite. The reason is worth knowing, because nothing
about the Python is wrong:

    QCoreApplicationPrivate::sendThroughApplicationEventFilters(QObject*, QEvent*)
      sipQObject::eventFilter(QObject*, QEvent*)
        sip_api_call_method → sipWrapInstance → QObject::connect
          EXC_BAD_ACCESS ... (possible pointer authentication failure)

An application filter is offered *every* object's events, including posted ones bound for an
object that has since been destroyed. Before any of the filter's own code runs, sip must build a
Python wrapper for that pointer — and that is where it dies. No guard inside ``eventFilter`` can
help, because the crash happens upstream of its first line. The reach is the defect.

So this is a plain function, called at the seams where something is put on screen, and the
conformance grid (`detail/gui/requirements.py`) is what makes sure the seams call it.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QWidget


def make_text_selectable(widget: QWidget) -> None:
    """Let the user select and copy every label under ``widget``. Idempotent."""
    for label in widget.findChildren(QLabel):
        label.setTextInteractionFlags(
            label.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse
        )
