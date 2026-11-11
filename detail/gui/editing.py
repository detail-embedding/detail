"""Don't lose someone's edits when they navigate away.

A page with a Save button loses whatever was typed the moment you click something else,
and the metadata table is where that hurt most — people fill a column, click the next
recording, and it's gone. The fix isn't per-page: any page that can hold unsaved work
implements this protocol and the shell asks it *before* leaving.

**Where the check goes.** ``app.py::_show`` is the single funnel every page swap passes
through, which makes it the obvious hook — but it is called from ~39 sites, and by the
time it runs the replacement page is already built and the rail selection has already
moved. So the ask happens at the **navigation entry points** instead (``_navigate``,
``_switch_mode``, the ``_go_to_*`` drill-ins, the Home opens, and ``closeEvent``): early
enough that answering *Cancel* can genuinely leave you where you were.

**Pages, not dialogs.** A modal dialog already has Cancel/OK semantics and its own
discard path; sweeping it in here would double-prompt.

A page opts in by implementing the three methods. ``tests/gui/test_editing.py`` fails if a
content page grows a Save-like control without them, or if a navigation entry point forgets
to ask — the point being that the next editable surface inherits this instead of
rediscovering the bug.
"""

from typing import Protocol, runtime_checkable

from PyQt6.QtWidgets import QMessageBox


@runtime_checkable
class EditableSurface(Protocol):
    """A page that can hold edits the user has not committed."""

    def unsaved_summary(self) -> str | None:
        """What would be lost, or ``None`` when there is nothing pending.

        The string is shown in the prompt, so name the work ("3 changed cells"), not the
        widget.
        """

    def save_edits(self) -> None:
        """Commit the pending edits. Called when the user picks *Save*."""

    def discard_edits(self) -> None:
        """Drop the pending edits and return to the stored state."""


def confirm_leave(parent, page, *, title: str = "Unsaved changes") -> bool:
    """Ask about ``page``'s unsaved edits. ``True`` to proceed, ``False`` to stay put.

    A page that doesn't implement the protocol, or has nothing pending, passes straight
    through — so callers can ask unconditionally.

    *Save* commits and proceeds; a failing save (a duplicate name, a bad value) reports and
    **stays**, because proceeding would discard the very edits the user asked to keep.

    A pending-check that *raises* means the page's data went away underneath it — the
    experiment it reads was just deleted — so there is nothing left worth saving and we
    proceed. Failing this open matters: every navigation entry point asks here first, so
    an exception escaping this function leaves the dead page on screen and makes the
    window permanently unnavigable.
    """
    try:
        summary = _pending(page)
    except Exception:
        # An explicit isolation boundary: asking a page a *question* must never be able
        # to block navigation. The query has no side effects, so failing it open is safe
        # in every case, while failing it closed bricks the shell.
        return True
    if summary is None:
        return True
    choice = QMessageBox.question(
        parent,
        title,
        f"{summary}\n\nSave before leaving?",
        QMessageBox.StandardButton.Save
        | QMessageBox.StandardButton.Discard
        | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Save,
    )
    if choice == QMessageBox.StandardButton.Cancel:
        return False
    if choice == QMessageBox.StandardButton.Discard:
        page.discard_edits()
        return True
    try:
        page.save_edits()
    except (ValueError, OSError) as exc:
        QMessageBox.warning(parent, title, f"Could not save: {exc}")
        return False
    return True


def _pending(page) -> str | None:
    """``page``'s unsaved summary, or ``None`` when it has none / isn't editable."""
    if page is None or not hasattr(page, "unsaved_summary"):
        return None
    return page.unsaved_summary()
