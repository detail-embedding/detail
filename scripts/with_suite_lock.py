"""Run a command holding the machine-wide lock, so a full suite or a timing run has the box to itself.

Use it for anything that needs the whole machine or a quiet one:

    python scripts/with_suite_lock.py -- pytest -q -n auto --dist loadfile
    python scripts/with_suite_lock.py --who detail-48 -- python bench.py

Not for the edit loop. A selected run is small, and two of them side by side cost nothing.

**Why a lock and not a message.** Several agents can share one development machine. A full
suite takes every core for minutes, and a timing run is only a measurement if nothing else is
running. Coordinating by message worked with two sessions and failed with three: a gate
started into another session's benchmark, and a tier's timing was published several times too
high because it was measured under someone else's suite. A lock needs nobody to know who else
is there: whoever holds it has the box, and anyone else waits and is told who they are
waiting on.

**Why the path is fixed, not under ``TMPDIR``.** Sessions here give each run a private ``TMPDIR``
(and ``HOME``) so their temp roots and Qt settings do not collide. A lock under ``TMPDIR`` would
therefore be a different file for every run, and would serialise nothing. It lives at one path
per machine; ``DETAIL_SUITE_LOCK`` overrides it.

**Why ``%ProgramData%`` on Windows, not ``tempfile.gettempdir()``.** ``gettempdir()`` reads
``TMPDIR``, ``TEMP`` and ``TMP`` on Windows too, so it is the per-session directory under another
name, and it is per *user* besides. ``main`` used it while ``remote-detail`` had moved to
``%ProgramData%``, so on 2026-10-23 a main gate started into a remote-detail benchmark and doubled
its times: each branch held a lock, in a different file. ``machine_dir`` is the one answer, and
``tests/scripts/test_machine_dir.py`` pins it byte-for-byte on both branches, so a branch that moves
the lock fails its own guard rather than silently leaving the other.

**WSL and Windows on one box do not share a lock.** They compete for the same cores, so ideally
they would; they cannot through this lock. Measured 2026-10-23 on a ``/mnt/c`` file: a Windows
``msvcrt`` lock blocked WSL's ``fcntl.lockf`` but not its ``flock``, and neither WSL lock was seen
from Windows at all. A lock that holds in one direction serialises nothing, so WSL keeps POSIX's
``/tmp/detail-suite.lock`` inside the distro, a separate namespace from ``%ProgramData%``. Before a
timing run, look at the other side's holder note by hand (``\\\\wsl$\\<distro>\\tmp`` from Windows,
``/mnt/c/ProgramData`` from WSL).

**Why ``file_lock`` and not ``flock``.** The pre-push hook queued through ``flock(1)`` and fell
back to running unserialised when the binary was absent. It is absent on macOS, which was the
machine the queueing was written for. ``flock(2)`` is in the kernel regardless, and
``core/atomic.py::file_lock`` already wraps it (with the ``msvcrt`` branch for Windows), so this
borrows that seam rather than adding a second lock primitive. The lock is advisory on an open
descriptor, so the kernel releases it when the holder exits for any reason, including a kill. There
is no stale lock to time out.

**Who holds it.** On acquiring the lock, the holder writes one line to ``<lock>.holder``: who,
since when, and the command. A waiter prints that line after a couple of seconds and again every
minute, so a wait is never silent. The line is only read while the lock is held by someone,
because the holder rewrites it on taking the lock; a line left behind by a killed run is never
reported as current.
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from detail.core.atomic import file_lock


def machine_dir(os_name: str = os.name, environ: Mapping[str, str] = os.environ) -> str:
    """The directory every checkout and branch on this machine agrees on.

    Never derived from ``TMPDIR`` / ``TEMP`` / ``TMP`` (see the module docstring). The fallback for
    a stripped environment is the literal default, not ``gettempdir()``: a fallback that follows the
    session splits the lock exactly as the bug did. ``scripts/suite_progress.py`` carries the same
    function for its directory and ``tests/scripts/test_machine_dir.py`` holds the two equal.
    """
    if os_name == "nt":
        return environ.get("ProgramData") or r"C:\ProgramData"
    return "/tmp"


_DEFAULT_LOCK = Path(machine_dir()) / "detail-suite.lock"

#: How long a waiter stays quiet before saying who it is waiting on, then how often it repeats.
#: Environment-overridable so the test does not have to sit through the real delays.
_FIRST_NOTICE_S = float(os.environ.get("DETAIL_SUITE_LOCK_NOTICE_S", "2"))
_REPEAT_NOTICE_S = float(os.environ.get("DETAIL_SUITE_LOCK_REPEAT_S", "60"))


def lock_path() -> Path:
    """The machine-wide lock file: ``DETAIL_SUITE_LOCK`` if set, else the fixed default."""
    override = os.environ.get("DETAIL_SUITE_LOCK")
    return Path(override) if override else _DEFAULT_LOCK


def _holder_path(lock: Path) -> Path:
    return lock.with_name(lock.name + ".holder")


def _read_holder(lock: Path) -> str:
    try:
        return _holder_path(lock).read_text(encoding="utf-8-sig").strip()
    except OSError:
        return "an unnamed run"


def _announce_while_waiting(lock: Path, acquired: threading.Event) -> None:
    """Print who holds the lock until ``acquired`` is set, so a wait is never silent."""
    delay = _FIRST_NOTICE_S
    started = time.monotonic()
    while not acquired.wait(delay):
        waited = time.monotonic() - started
        print(
            f"with_suite_lock: waiting {waited:.0f}s on {_read_holder(lock)}",
            file=sys.stderr,
            flush=True,
        )
        delay = _REPEAT_NOTICE_S


def run_locked(command: list[str], *, who: str, lock: Path) -> int:
    """Run ``command`` while holding ``lock``; return its exit code."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    acquired = threading.Event()
    notifier = threading.Thread(
        target=_announce_while_waiting, args=(lock, acquired), daemon=True
    )
    start = time.monotonic()
    notifier.start()
    with file_lock(lock):
        acquired.set()
        waited = time.monotonic() - start
        if waited > 1.0:
            print(
                f"with_suite_lock: queued {waited:.0f}s behind another run",
                file=sys.stderr,
                flush=True,
            )
        holder = _holder_path(lock)
        since = datetime.now().isoformat(timespec="seconds")
        # Written only while the lock is held, so readers never see a half-written line from a
        # competing writer; a plain write is enough.
        holder.write_text(
            f"{who} since {since}: {' '.join(command)}\n", encoding="utf-8"
        )
        try:
            return subprocess.call(command)
        finally:
            try:
                holder.unlink()
            except OSError:
                pass  # a stale line is harmless: it is only read while someone holds the lock


def _default_who() -> str:
    return os.environ.get("DETAIL_LOCK_WHO") or f"{getpass.getuser()} pid {os.getpid()}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="with_suite_lock.py",
        description="Run a command while holding the machine-wide suite/timing lock.",
    )
    parser.add_argument(
        "--who",
        default=None,
        help="Name shown to anyone waiting (default: $DETAIL_LOCK_WHO, else user and pid).",
    )
    parser.add_argument(
        "--lock",
        default=None,
        help=f"Lock file (default: $DETAIL_SUITE_LOCK, else {_DEFAULT_LOCK}).",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given; put it after --")
    lock = Path(args.lock) if args.lock else lock_path()
    return run_locked(command, who=args.who or _default_who(), lock=lock)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
