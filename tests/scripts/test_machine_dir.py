"""Every checkout and branch on one machine takes the suite lock and writes progress in one place.

**The class: a machine-wide rendezvous path resolved by something per-session, or per-branch.**
The lock and the progress directory work only if every run on the box opens the same file. On
2026-10-23 ``main`` resolved the Windows lock through ``tempfile.gettempdir()``, which reads
``TMPDIR`` / ``TEMP`` / ``TMP`` and is per user, while ``remote-detail`` had moved to
``%ProgramData%`` three weeks before in a post-merge fix that never went upstream. A main gate
started into a remote-detail benchmark and doubled its times. Nothing failed: each run held a
lock, every run was green, and the only signature was two heavy runs on the machine at once.

It hid twice over. ``test_with_suite_lock.py``'s TMPDIR check skipped itself on Windows, and CI
runs Linux, where both branches already said ``/tmp``. So these tests call ``machine_dir`` with
the platform as an *argument*: the Windows answer is checked on every OS, and a revert to
``gettempdir()`` fails here on Linux too.

**The rule.** ``machine_dir`` is the one answer: ``%ProgramData%`` on Windows (literal
``C:\\ProgramData`` if the variable is missing, never the session's temp dir), ``/tmp`` elsewhere,
WSL included. WSL and Windows are deliberately separate namespaces; ``with_suite_lock``'s module
docstring has the measurement. ``suite_progress`` carries a copy, because it must import only the
standard library, so the copies are held equal here. This file is byte-identical on ``main`` and
``remote-detail``: a branch that moves the lock has to change its pins, which conflicts at the
next merge instead of drifting.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath

import pytest

from scripts import suite_progress

REPO = Path(__file__).resolve().parents[2]

#: The pins. Changing one is changing where every other checkout on the machine looks.
WINDOWS_DIR = r"C:\ProgramData"
POSIX_DIR = "/tmp"
LOCK_NAME = "detail-suite.lock"
PROGRESS_NAME = "detail-suite-progress"

#: What a session sets: a private temp dir under every name ``gettempdir()`` reads.
_SESSION_TEMP = r"C:\Users\someone\AppData\Local\Temp\claude\session-1"
_WINDOWS_ENVS = [
    pytest.param({"ProgramData": WINDOWS_DIR}, id="plain"),
    pytest.param(
        {
            "ProgramData": WINDOWS_DIR,
            "TMPDIR": _SESSION_TEMP,
            "TEMP": _SESSION_TEMP,
            "TMP": _SESSION_TEMP,
            "USERPROFILE": r"C:\Users\someone",
        },
        id="session-temp",
    ),
    pytest.param({"TEMP": _SESSION_TEMP, "TMP": _SESSION_TEMP}, id="no-ProgramData"),
]
_POSIX_ENVS = [
    pytest.param({}, id="plain"),
    pytest.param(
        {"TMPDIR": "/private/session-1", "HOME": "/home/someone"}, id="session"
    ),
]


def _lock_module():
    spec = importlib.util.spec_from_file_location(
        "with_suite_lock", REPO / "scripts" / "with_suite_lock.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RESOLVERS = [
    pytest.param(lambda: _lock_module().machine_dir, id="with_suite_lock"),
    pytest.param(lambda: suite_progress.machine_dir, id="suite_progress"),
]


@pytest.mark.parametrize("resolver", _RESOLVERS)
@pytest.mark.parametrize("environ", _WINDOWS_ENVS)
def test_windows_resolves_to_programdata_whatever_the_session_sets(resolver, environ):
    got = resolver()("nt", environ)
    assert PureWindowsPath(got) == PureWindowsPath(WINDOWS_DIR), (
        f"the Windows machine dir is {got!r}: a run whose session sets a different temp dir, or "
        "a checkout of the other branch, takes a different lock and nothing is serialised"
    )


@pytest.mark.parametrize("resolver", _RESOLVERS)
@pytest.mark.parametrize("environ", _POSIX_ENVS)
def test_posix_and_wsl_resolve_to_tmp_whatever_the_session_sets(resolver, environ):
    assert PurePosixPath(resolver()("posix", environ)) == PurePosixPath(POSIX_DIR)


def test_the_live_paths_are_the_pinned_ones(monkeypatch):
    """The module constants this machine actually uses, not only the function they come from."""
    monkeypatch.delenv("DETAIL_SUITE_LOCK", raising=False)
    monkeypatch.delenv("DETAIL_SUITE_PROGRESS", raising=False)
    if os.name == "nt":
        base = Path(os.environ.get("ProgramData") or WINDOWS_DIR)
    else:
        base = Path(POSIX_DIR)
    assert _lock_module().lock_path() == base / LOCK_NAME
    assert suite_progress.progress_dir() == base / PROGRESS_NAME


def test_the_guard_would_catch_a_lock_that_follows_the_session_temp_dir():
    """The negative case: the bug's own resolver, handed the same environment, is rejected."""

    def gettempdir_like(os_name, environ):
        if os_name == "nt":
            for name in ("TMPDIR", "TEMP", "TMP"):
                if name in environ:
                    return environ[name]
            return environ.get("ProgramData") or WINDOWS_DIR
        return environ.get("TMPDIR", POSIX_DIR)

    session = {"ProgramData": WINDOWS_DIR, "TEMP": _SESSION_TEMP}
    with pytest.raises(AssertionError):
        test_windows_resolves_to_programdata_whatever_the_session_sets(
            lambda: gettempdir_like, session
        )
    with pytest.raises(AssertionError):
        test_posix_and_wsl_resolve_to_tmp_whatever_the_session_sets(
            lambda: gettempdir_like, {"TMPDIR": "/private/session-1"}
        )
