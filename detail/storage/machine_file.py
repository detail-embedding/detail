"""One edit of a machine-config file — ``storage.yaml``, ``credentials.yaml``,
``remote_targets.yaml`` — read, changed and written back **to the path it was read from**.

Each of those files is named by a module global (``locations.STORAGE_PATH``,
``credentials.CREDENTIALS_PATH``, ``remote_targets.REMOTE_TARGETS_PATH``), and each writer was a
read-modify-write that looked the global up twice: once to read, once to write. When the global
moves between the two, the edit reads one file and overwrites another with it. Tests move it —
every test points the machine config at a throwaway directory and puts it back afterwards — and a
video-import worker thread that outlived its test read the test's ``storage.yaml``, then wrote it
over the developer's real one after the fixture had restored the path: every real video root on
the machine was replaced by one test alias (2026-11-06, the ``two-animal`` and ``first30s`` roots).

So :func:`editing` takes the path **once**, and the lock, the read and the write all use it. An
edit that straggles past a path change finishes on the file it started on.

Concurrency, which these files have too: an edit holds an ``O_CREAT | O_EXCL`` lock beside the
file (atomic on POSIX and Windows), so two writers cannot both read, both write and lose the first
one's change; a lock left by a killed process is taken over once it is clearly stale. The write is
a temp file **unique to this writer**, replaced in through ``core.atomic`` — a shared temp name
lets the second writer's replace find the first has already moved it.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml

from detail.core import atomic
from detail.core import yaml_load
from detail.storage import read_cache

_LOCK_ATTEMPTS = 500
_LOCK_RETRY_S = 0.01
_LOCK_STALE_S = 30.0  # a holder this old was killed; take the lock over


def read(path: Path | str) -> dict:
    """``path``'s mapping, or ``{}`` when the file is missing or empty.

    Raises:
        ValueError: The file holds something other than a mapping.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    data = yaml_load.safe_load(atomic.read_text(path))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping, got {type(data).__name__}")
    return data


@contextmanager
def editing(path: Path | str, *, mode: int | None = None) -> Iterator[dict]:
    """``path``'s mapping, written back to ``path`` when the block exits without raising.

    ``path`` is resolved here, once: pass the module global itself, and a change to it while the
    block runs cannot redirect the write. ``mode`` is applied to the written file (``0o600`` for
    credentials).
    """
    path = Path(path)
    with _exclusive(path):
        data = read(path)
        yield data
        _write(path, data, mode)


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    fd = None
    for _ in range(_LOCK_ATTEMPTS):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY | atomic.O_BINARY)
            break
        except FileExistsError:
            if _stale(lock):
                lock.unlink(missing_ok=True)
                continue
            time.sleep(_LOCK_RETRY_S)
    if fd is None:
        raise RuntimeError(
            f"could not lock {path} — another process has held it for more than "
            f"{_LOCK_ATTEMPTS * _LOCK_RETRY_S:.0f}s"
        )
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def _stale(lock: Path) -> bool:
    """Whether a lock file is old enough that its holder is presumed dead."""
    try:
        return time.time() - lock.stat().st_mtime > _LOCK_STALE_S
    except FileNotFoundError:
        return False  # released while we looked — just retry normally


def _write(path: Path, data: dict, mode: int | None) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
        if mode is not None:
            tmp.chmod(mode)
        # Through the cache's commit, so a cached parse of this file (`locations.read_storage`)
        # cannot outlive the write: a pinned read skips the `stat`, and a same-size rewrite
        # inside one tick of a coarse filesystem clock passes it.
        read_cache.replace_document(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # a failed replace must not leave litter behind
