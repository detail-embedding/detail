"""Count the filesystem round trips a piece of code makes — under one root, on every thread.

**The class.** A read-cost guard patches ``os.stat`` / ``io.open`` / ``os.scandir`` for the whole
process, because the code it measures may read from a worker thread and moving a read off the UI
thread does not make it free. Counted process-wide, though, the guard also counts every *other*
thread doing I/O — and a GUI test can leave one behind: a page opened on a run that is still
"running" keeps a progress poller reading that run's records every 0.25 s until the run ends,
which a fixture's fake run never does. ``test_page_read_cost`` measured those strays as the page
under test getting more expensive — 2.05 round trips per recording against a floor of 2.0 — in 5
of 86 full runs, only when ``--dist loadfile`` put ``test_run_detail.py`` earlier on the same
worker. Three copies of the counter existed (``test_page_read_cost``, ``test_read_cache``,
``test_dirscan``), all process-wide.

**The rule.** Count by *where* a call reads, not by which thread makes it: only paths under the
root the measured code was given. A stray from another test reads its own ``tmp_path``, which
is never under this one's. A call made on a file descriptor names no path and is not counted;
none of the measured seams use one.
"""

from __future__ import annotations

import collections
import contextlib
import io
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

KINDS = ("stat", "open", "scandir")


def _roots(root: Path) -> tuple[str, ...]:
    # Both spellings: macOS's tmp_path is a /var symlink into /private/var, and the code under
    # test may hold either form.
    spellings = {os.path.abspath(root), os.path.realpath(root)}
    return tuple(s.rstrip(os.sep) + os.sep for s in spellings)


@contextlib.contextmanager
def counting(
    root: Path, kinds: tuple[str, ...] = KINDS
) -> Iterator[collections.Counter]:
    """Count ``kinds`` of call on paths under ``root`` while the block runs, on any thread."""
    prefixes = _roots(root)
    seen: collections.Counter = collections.Counter()
    real = {"stat": os.stat, "open": io.open, "scandir": os.scandir}

    def spy(kind):
        func = real[kind]

        def wrapped(*args, **kwargs):
            target = args[0] if args else kwargs.get("path", kwargs.get("file"))
            if isinstance(target, (str, bytes, os.PathLike)):
                path = os.path.abspath(os.fsdecode(os.fspath(target)))
                if (path + os.sep).startswith(prefixes):
                    seen[kind] += 1
            return func(*args, **kwargs)

        return wrapped

    targets = {"stat": "os.stat", "open": "io.open", "scandir": "os.scandir"}
    with contextlib.ExitStack() as stack:
        for kind in kinds:
            stack.enter_context(patch(targets[kind], side_effect=spy(kind)))
        yield seen


def count_stats(root: Path, call) -> tuple[object, int]:
    """``(call(), number of os.stat calls under root)``.

    ``os.path.isfile`` and every ``pathlib`` predicate resolve ``os.stat`` at call time, so one
    patch sees both spellings; ``os.scandir`` does not go through it.
    """
    with counting(root, ("stat",)) as seen:
        result = call()
    return result, seen["stat"]
