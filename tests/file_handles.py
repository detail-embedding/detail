"""Asking the operating system whether this process holds a file open.

Here rather than private to one module because the question has two askers: the storage guard
that pins the release (`tests/storage/test_video_handle_release.py`) and the GUI test that moves
a video out from under an open recording page (`tests/gui/test_pages.py`).

**Why not just try to delete it.** On Windows an open handle refuses the delete, so `unlink`
failing *is* the signal there — and on Linux and macOS an open file unlinks happily, so a test
built on it is green on every box CI runs and on every box the bug does not reach. Asking the OS
for this process's open files is the Windows question asked in a way POSIX can answer.

**Why not psutil on Windows.** `psutil.Process().open_files()` is the right call on Linux (it
reads `/proc/self/fd`) and macOS, and it is unreliable on Windows: measured 2026-10-22, it reported
an open `FrameCursor`'s video as not held on one of two real parts tried. So on Windows the probe
asks the question the user's file manager asks — can it be renamed — and puts it back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def file_is_held(path: Path | str) -> bool:
    """Whether this process has ``path`` open, by the means each platform can answer.

    On Windows, a rename away and back: the move a user makes in Explorer, refused while any
    handle without delete-sharing is open. Elsewhere, ``psutil``'s list of this process's open
    files.
    """
    path = Path(path)
    if sys.platform == "win32":
        aside = path.with_name(path.name + ".held-probe")
        try:
            os.rename(path, aside)
        except PermissionError:
            return True
        os.rename(aside, path)
        return False
    import psutil

    target = os.path.realpath(path)
    return any(
        os.path.realpath(f.path) == target for f in psutil.Process().open_files()
    )
