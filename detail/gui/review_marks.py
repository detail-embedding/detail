"""What a review page has marked and not yet read back, its undo history, and its ordered writes.

AUDIT reviews every clip of a scope — a cluster on a clustering run, one recording of a behavior on
a query run (§20, §22) — and every click marks a clip at once, on screen, then writes it in the
background through the run's write queue (``run_writes``), which outlives the page. Two pages do
this; this is the part they share, so neither holds a second copy of it:

- **Ordered writes.** :meth:`ReviewMarks.write` numbers each write and queues it on the run's queue,
  so a put-back lands after the take-out it undoes and a finish after every mark before it.
- **Marks not yet read back.** A read started before a write landed does not have it; laid back over
  that read (:meth:`ReviewMarks.overlay`), oldest first, the page never shows a mark taken back by a
  read that raced it. ``docs/recorded_classes.md`` records the bug a second copy of this caused.
- **Undo history.** Each change keeps every clip's mark before it (:meth:`ReviewMarks.remember`);
  what a mark *means* — a removal, a move to another label — stays the page's, so the page undoes
  a popped change through its own writes.

A mark is ``{clip key: edit or None}``: the edit record the page holds, or ``None`` for a clip put
back (:func:`apply_change`).
"""

from collections.abc import Callable

from detail.gui import run_writes


def apply_change(taken: dict, change: dict) -> None:
    """Lay ``{clip key: edit or None}`` over ``taken``: ``None`` puts the clip back."""
    for key, edit in change.items():
        if edit is None:
            taken.pop(key, None)
        else:
            taken[key] = edit


class ReviewMarks:
    """A review page's ordered writes, the marks they carry until read back, and its undo history.

    ``on_written()`` runs when the run's queue goes idle after one of this page's writes;
    ``on_failed(what, exc)`` when one fails — every write queued behind it is dropped by the queue,
    so the marks not yet read back are forgotten first.
    """

    def __init__(
        self,
        run,
        owner,
        *,
        on_written: Callable[[], None],
        on_failed: Callable[[str, BaseException], None],
    ) -> None:
        #: The run's write queue: shared with every page on this run and kept by the app.
        self._writes = run_writes.queue_for(run)
        self._owner = owner
        self._on_written = on_written
        self._on_failed = on_failed
        #: This review's changes, oldest first — what ⌘Z steps back through.
        self._history: list[dict] = []
        #: ``(write number, change)`` written and not yet seen by a read, oldest first.
        self._pending: list[tuple[int, dict]] = []
        self._queued = 0
        #: The number of the last write done — what a read started now has seen.
        self.landed = 0

    # -- undo history ---------------------------------------------------------------------
    def remember(self, text: str, rows: list, before: dict) -> None:
        """Push one change: what it did (``text``), the rows it marked, and each clip's mark
        before it (``{clip key: edit or None}``)."""
        self._history.append({"text": text, "rows": list(rows), "before": dict(before)})

    def pop(self) -> dict | None:
        """The last change, taken off the history, or ``None`` with none."""
        return self._history.pop() if self._history else None

    def last_text(self) -> str | None:
        """What the last change did, or ``None`` with nothing to undo."""
        return self._history[-1]["text"] if self._history else None

    def forget(self) -> None:
        """Clear the history — on Finish, and when another scope opens."""
        self._history = []

    # -- writes -----------------------------------------------------------------------------
    def write(self, work, what: str = "Not saved", change: dict | None = None) -> None:
        """Queue one write. ``change``: the marks it writes, laid over any read that lands
        before it does."""
        self._queued += 1
        number = self._queued
        if change:
            self._pending.append((number, change))

        def done(_result) -> None:
            self.landed = number
            if not self._writes.busy():
                self._on_written()

        def failed(exc) -> None:
            # Nothing queued will be written: the next read is the run as it stands.
            self._pending.clear()
            self._on_failed(what, exc)

        # The run's queue, not the page's: a queue held on a page lost every write after the one
        # in flight when the page was left (#24).
        self._writes.submit(
            work, owner=self._owner, done=done, failed=failed, title=what
        )

    def when_idle(self, then) -> None:
        """Run ``then`` once every queued write is done — at once if none are waiting."""
        self._writes.when_idle(then, owner=self._owner)

    def busy(self) -> bool:
        """Whether a write on this run — from this page or one since left — is not on disk."""
        return self._writes.busy()

    # -- reads ----------------------------------------------------------------------------
    def overlay(self, taken: dict, *, landed_before: int | None) -> dict:
        """``taken`` as read, with every mark not yet read back laid over it. ``landed_before``
        is :attr:`landed` when the read began: marks written by then are in it and are dropped;
        ``None`` keeps them all (a read the page did not start)."""
        if landed_before is not None:
            self._pending = [p for p in self._pending if p[0] > landed_before]
        out = dict(taken)
        for _n, change in self._pending:
            apply_change(out, change)
        return out
