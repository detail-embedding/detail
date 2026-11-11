"""Machine-local directory aliases in ``storage.yaml`` — the Tenet 6 escape hatch (Section 2).

An experiment file may hold no machine path. When one has to refer to a directory *outside* the
experiment, it stores ``{root: <alias>, rel: <path>}`` and the alias resolves here, against
``~/.config/detail/storage.yaml``. Moving a whole library is then one re-point of the alias, and
the experiment tar stays portable.

Two kinds of thing need this, under two keys:

- ``video_roots`` — reference-in-place recordings (§2), the original caller. ``video_roots.py``
  keeps its own named API over this core; every existing caller and the ``video-root`` CLI
  command are unchanged.
- ``export_roots`` — where a durable export run was told to write (#354). An export's destination
  is chosen by the user and lives outside the experiment, so the job and the run record carry an
  alias; putting the absolute path in either would land it in ``params.csv`` via
  ``runners.job_params`` and make ``detail experiment export`` refuse the experiment for good.

Generalised from ``video_roots`` on its second use rather than its third, because that module's
own docstring calls itself *"the local-only first step of the planned ``StorageLocation``
abstraction"* and its aliases *"an internal detail — the user picks a video, not a location"*.
"""

import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import yaml

from detail.core import atomic
from detail.storage import locations
from detail.storage import read_cache

_LOCK_ATTEMPTS = 500
_LOCK_RETRY_S = 0.01
_LOCK_STALE_S = 30.0  # a holder this old was killed; take the lock over


class MissingRootError(RuntimeError):
    """A referenced alias is not registered on this machine."""


def list_roots(key: str) -> dict[str, Path]:
    """Every registered alias -> absolute directory, under one ``storage.yaml`` key."""
    roots = locations.read_storage().get(key) or {}
    return {alias: Path(p).expanduser() for alias, p in roots.items()}


def root(key: str, alias: str) -> Path | None:
    """The directory an alias resolves to, or ``None`` if it isn't registered."""
    found = (locations.read_storage().get(key) or {}).get(alias)
    return None if found is None else Path(found).expanduser()


def register_root(key: str, alias: str, path: Path | str) -> None:
    """Register (or re-point) ``alias`` to ``path``, merged into storage.yaml atomically.

    Raises if the alias is empty or the path is not an existing directory — a typo or an
    unmounted drive should fail loudly at registration, not silently at run time.

    **Safe against concurrent registrations.** Importing a batch of videos runs the parts in
    parallel worker *processes*, each registering its folder, so this read-modify-write is
    contended. It runs under ``_exclusive`` — see there for why retrying alone doesn't do it.
    """
    if not alias:
        raise ValueError("root alias must be non-empty")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"root {resolved} is not an existing directory")
    with _exclusive():
        data = locations.read_storage_fresh()
        roots = dict(data.get(key) or {})
        roots[alias] = resolved.as_posix()
        data[key] = roots
        _write(data)


def resolve_ref(key: str, ref: dict, *, hint: str = "") -> Path:
    """Resolve a ``{root, rel}`` reference to an absolute path.

    Raises :class:`MissingRootError` when the alias isn't registered, carrying ``hint`` — the
    caller knows what the user should do about it and this module does not.  Does not check that
    the path exists; that is the caller's presence check.
    """
    alias = ref.get("root")
    rel = ref.get("rel")
    if not alias or rel is None:
        raise ValueError(f"invalid reference {ref!r}")
    found = root(key, alias)
    if found is None:
        raise MissingRootError(
            f"root {alias!r} is not registered on this machine"
            + (f"; {hint}" if hint else "")
        )
    return found / rel


def resolve_or_register(key: str, path: Path | str) -> str:
    """An alias whose directory contains ``path``, auto-registering it if none does.

    Reuses the **deepest** existing root that already holds it (things from one tree then share
    a location); otherwise registers the directory itself under an auto alias. The alias is an
    internal detail — the user picks a folder, not a location — which keeps the ``{root, rel}``
    model without any location UI.

    ``path`` is used as-is when it is a directory and its parent otherwise, so a caller can hand
    over either an export destination or a file inside one.
    """
    resolved = Path(path).expanduser().resolve()
    containing = [
        (alias, found)
        for alias, found in list_roots(key).items()
        if _is_within(resolved, found)
    ]
    if containing:
        return max(containing, key=lambda ar: len(ar[1].resolve().parts))[0]
    folder = resolved if resolved.is_dir() else resolved.parent
    alias = _auto_alias(key, folder)
    register_root(key, alias, folder)
    return alias


def _is_within(path: Path, within: Path) -> bool:
    try:
        path.resolve().relative_to(within.expanduser().resolve())
        return True
    except (ValueError, OSError):
        return False


def _auto_alias(key: str, folder: Path) -> str:
    """A stable, human-ish alias for an auto-registered folder (its name, deduped)."""
    existing = list_roots(key)
    base = folder.name or "roots"
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


@contextmanager
def _exclusive():
    """Hold an exclusive claim on storage.yaml across a read-modify-write.

    Retrying the merge is not enough on its own: two writers can both read, both write, and the
    second's verify still passes — the loser's alias is simply gone. The entry survives in the
    record's ``{root, rel}`` but resolves to nothing.

    ``O_CREAT | O_EXCL`` is the lock (atomic on POSIX and Windows alike). A lock left behind by a
    killed process is taken over once it is clearly stale, so a crash mid-write can't wedge every
    future one.
    """
    locations.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lock = locations.CONFIG_DIR / (locations.STORAGE_PATH.name + ".lock")
    fd = None
    for _ in range(_LOCK_ATTEMPTS):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if _stale(lock):
                lock.unlink(missing_ok=True)
                continue
            time.sleep(_LOCK_RETRY_S)
    if fd is None:
        raise RuntimeError(
            f"could not lock {locations.STORAGE_PATH} — another process has held it for "
            f"more than {_LOCK_ATTEMPTS * _LOCK_RETRY_S:.0f}s"
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


def _write(data: dict) -> None:
    """Atomic write, with a temp name **unique to this writer**.

    The house idiom is a fixed sibling ``*.tmp`` (CLAUDE.md), which is right where one process
    rewrites a file others only read. This file is different: parallel workers write it at once,
    and on a shared temp name the second ``os.replace`` finds the first has already moved it —
    ``FileNotFoundError`` mid-import.
    """
    locations.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = locations.STORAGE_PATH
    tmp = path.parent / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            yaml.dump(data, handle, sort_keys=False)
        atomic.replace(tmp, path)
        read_cache.forget_document(path)
    finally:
        tmp.unlink(missing_ok=True)  # a failed replace must not leave litter behind
