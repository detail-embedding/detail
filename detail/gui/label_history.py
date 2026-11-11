"""INTERPRET's ⌘Z: one history of label changes for a run, shared by the page and its windows.

Every label-state write INTERPRET or one of its cluster windows makes — label, remove, rename,
merge, a recent label's key, AUDIT's *Move* / *Remove* on the label bar — is one :class:`Change`
here. ⌘Z in the page or in any of its windows undoes the most recent one, wherever it was made;
⇧⌘Z redoes it. A new change after an undo drops what could have been redone, as every editor's
history does.

**An undo is an inverse write, by id.** A change records what the run's label maps
(``AuthoredRun.label_maps``) were before and after its write, read by the write itself off the UI
thread (``label_change``); undoing writes the *before* side back (``AuthoredRun.restore_labels``)
and redoing the *after* side. So undoing a merge brings the merged label back under its own id —
and every audit answer recorded against it joins it again — where writing the old name back would
mint a new label. The page's own copy of the names (``names_before`` / ``names_after``) is what
it shows at once, the same way a label shows before its write lands.

**Where it stops.** The history is the page's, in memory: it is cleared when the run is re-fitted
(the tree changed, so a node id no longer means the same cluster), when a write fails (the run is
re-read, and what was recorded may no longer be true), and with the page. A snapshot or a finished
run greys Undo with the page's reason. The CLI's label commands are single explicit writes, so
there is nothing for it to undo (``docs/cli.md``, *Not in the CLI*).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

#: Why Undo is greyed with nothing to undo (AUDIT's wording, ``label_clean.NOTHING_TO_UNDO``).
NOTHING_TO_UNDO = "Nothing to undo"


@dataclass
class Change:
    """One change: what it did, in words, and how to put either side back.

    ``show(after)`` puts the page's on-screen copy at that side, at once, on the UI thread;
    ``write(after)`` writes that side to the run, off the UI thread, through the page's queue —
    which runs it after the change's own write, so what that write did is known by then.
    """

    text: str
    show: Callable[[bool], None]
    write: Callable[[bool], None]


def restore_side(written: tuple[dict, ...] | None, after: bool) -> tuple[dict, ...]:
    """``(labels, node_labels, membership_changed_at)`` to restore from what a label write did
    (``authored_run.label_change``): the *after* values for a redo, *before* for an undo. Empty
    when the write changed nothing or has not landed.

    The change times go back too, so undoing a misclick does not read as an edit of the label
    and set aside the review made before it."""
    if written is None:
        return {}, {}, {}
    index = 1 if after else 0
    return tuple({key: pair[index] for key, pair in part.items()} for part in written)


class LabelHistory:
    """The undo and redo stacks of one run's label changes."""

    def __init__(self) -> None:
        self._done: list[Change] = []
        self._undone: list[Change] = []

    def record(self, change: Change) -> None:
        self._done.append(change)
        self._undone.clear()

    def can_undo(self) -> bool:
        return bool(self._done)

    def can_redo(self) -> bool:
        return bool(self._undone)

    def next_undo(self) -> Change | None:
        return self._done[-1] if self._done else None

    def next_redo(self) -> Change | None:
        return self._undone[-1] if self._undone else None

    def undo(self) -> Change | None:
        if not self._done:
            return None
        change = self._done.pop()
        self._undone.append(change)
        return change

    def redo(self) -> Change | None:
        if not self._undone:
            return None
        change = self._undone.pop()
        self._done.append(change)
        return change

    def clear(self) -> None:
        self._done.clear()
        self._undone.clear()

    def __len__(self) -> int:
        return len(self._done)


def show_undo(button, history: LabelHistory, reason: str) -> None:
    """Grey ``button`` with ``reason`` (the run cannot be edited here) or with nothing to undo;
    else say exactly what it undoes, and that ⇧⌘Z redoes."""
    from detail.gui.sizing import set_available

    change = history.next_undo()
    set_available(button, not reason and change is not None, reason or NOTHING_TO_UNDO)
    if button.isEnabled():
        button.setToolTip(f"Undo: {change.text}. ⇧⌘Z redoes.")
