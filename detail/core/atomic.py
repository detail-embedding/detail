"""Windows-safe atomic replace + retry for concurrently-accessed experiment files.

On Windows, ``os.replace`` (MoveFileEx) is not atomic against a concurrent open the
way POSIX ``rename`` is: while one process replaces a file, a concurrent reader's
``open`` — or the replace itself, when a reader holds the destination open — can hit
a transient sharing-violation / access-denied surfaced as ``PermissionError``
(WinError 32 / 5). Every experiment file a background worker rewrites while another
process polls it (``job.json``, ``run.yaml``, ``analysis.yaml``, ``recording.yaml``,
the per-worker progress files) is exposed on *both* sides — so both the replace and
the read must retry. A short bounded retry rides out the sub-millisecond window;
POSIX ``rename`` never enters it, so this is a no-op there.
"""

import contextlib
import os
import time
from collections.abc import Callable
from pathlib import Path

RETRY_ATTEMPTS = 20
RETRY_DELAY_S = 0.01
_LOCK_POLL_S = 0.005


def retry_on_lock(operation):
    """Run ``operation`` (a no-arg callable) and return its result, retrying the
    transient ``PermissionError`` a concurrent ``os.replace`` provokes on Windows.
    Re-raises once the attempts are exhausted (a genuine permission error still
    surfaces, just after the bounded wait)."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return operation()
        except PermissionError:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(RETRY_DELAY_S)


def replace(src: Path | str, dst: Path | str) -> None:
    """``os.replace(src, dst)`` with the Windows concurrent-access retry, after making
    ``src``'s contents durable.

    **The flush is what makes the swap safe to crash through.** A rename is journaled
    as metadata; the bytes written to ``src`` may still be in the OS cache. Lose power
    in that window and NTFS keeps the new name *and* the new length with the data never
    written, so the record reads back as that many zero bytes: not the old record, not
    the new one, and not parseable. That is how one interrupted distill run left its
    ``distill.yaml``, ``dataset.yaml`` and ``job.json`` all-NUL and took the Models and
    Datasets pages down with them. Flushing first means a crash yields the old file or
    the new one. A flush costs milliseconds, against a 0.5 s progress cadence for the
    busiest writer.

    A directory ``src`` (``storage/frame_view.py`` swaps a staged directory) is moved
    unflushed: there is no portable way to flush one, and that store is derived — it is
    rebuilt whenever its artifact is.
    """
    if not os.path.isdir(src):
        _flush_to_disk(src)
    retry_on_lock(lambda: os.replace(src, dst))


def _flush_to_disk(path: Path | str) -> None:
    """``fsync`` a closed file by path (``FlushFileBuffers`` on Windows, which needs a
    handle opened for writing)."""
    fd = retry_on_lock(lambda: os.open(path, os.O_RDWR))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_text(path: Path | str) -> str:
    """Read a concurrently-rewritten experiment file as text, with the Windows retry.

    Both hazards on these files have the same scope, so they belong on one seam rather
    than on each call. ``retry_on_lock(path.read_text)`` reads correctly on the dev box
    and takes the *locale* codec — cp1252 on Windows — because the bound method is
    invoked with no arguments; the file was written UTF-8, so an accented recording name
    comes back as mojibake and its undefined cp1252 bytes (0x81/0x8D/0x8F/0x90/0x9D)
    raise ``UnicodeDecodeError`` instead, from the reader of every ``recording.yaml``.
    ``utf-8-sig`` on the read side, per the encoding rule; ``replace`` above is the
    write-side twin.
    """
    return retry_on_lock(lambda: Path(path).read_text(encoding="utf-8-sig"))


# An atomic write keeps a *reader* from seeing a torn file; it does nothing for two writers
# doing read-modify-write on the same file, or for two processes allocating the next id from
# a directory scan. That needs mutual exclusion, and it is needed at more than one level —
# `recording.yaml`'s run lists and the experiment's `runs/` id counter have the identical
# hazard — so the primitive lives here rather than in whichever storage module hit it first.
if os.name == "nt":  # pragma: no cover - Windows is untested by CI

    def _acquire_lock(fd: int) -> None:
        # NOT msvcrt.LK_LOCK: it retries for only ~9s and then *fails* with
        # EDEADLOCK rather than waiting like fcntl.flock, and it holds the GIL
        # while it waits, so a contending thread in this process can never reach
        # its release. Poll the non-blocking variant instead — time.sleep drops
        # the GIL, and we wait as long as the POSIX branch does.
        import msvcrt

        while not _try_lock(fd):
            time.sleep(_LOCK_POLL_S)

    def _try_lock(fd: int) -> bool:
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _release_lock(fd: int) -> None:
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:

    def _acquire_lock(fd: int) -> None:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)

    def _try_lock(fd: int) -> bool:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _release_lock(fd: int) -> None:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


#: How often a lock taken with ``waiting`` tries again, and so how often ``waiting`` is called.
WAIT_POLL_S = 0.25


@contextlib.contextmanager
def file_lock(lock_path: Path | str, *, waiting: Callable[[], None] | None = None):
    """Hold an exclusive advisory lock on ``lock_path`` for the block.

    The lock file is a sibling marker, never the protected file itself — locking the file a
    writer is about to ``os.replace`` would protect the wrong inode.

    ``waiting``, when given, makes the wait visible and stoppable: the lock is tried without
    blocking and, while another holder has it, ``waiting()`` is called every
    :data:`WAIT_POLL_S` — to say so on a bar, and to raise (``OperationCancelled``) when the
    user cancels. A blocking ``flock`` held by another process can last minutes, and a thread
    parked in it can neither report nor be stopped.
    """
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    os.set_inheritable(fd, False)
    try:
        if waiting is None:
            _acquire_lock(fd)
        else:
            while not _try_lock(fd):
                waiting()
                time.sleep(WAIT_POLL_S)
    except BaseException:
        # Releasing a lock we never took raises in its own right (PermissionError on
        # Windows), which would mask this error and skip the close below.
        os.close(fd)
        raise
    try:
        yield
    finally:
        try:
            _release_lock(fd)
        finally:
            os.close(fd)
