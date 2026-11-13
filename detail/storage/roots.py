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

from pathlib import Path

from detail.storage import locations
from detail.storage import machine_file


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
    contended, and the edit is ``machine_file.editing``: one lock, and one path for the read and
    the write.
    """
    if not alias:
        raise ValueError("root alias must be non-empty")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"root {resolved} is not an existing directory")
    with machine_file.editing(locations.STORAGE_PATH) as data:
        roots = dict(data.get(key) or {})
        roots[alias] = resolved.as_posix()
        data[key] = roots


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


def editing():
    """storage.yaml as a dict, written back when the block exits without raising.

    For a writer outside this module's ``{alias: path}`` keys — the experiment locations
    under ``storage:`` — so it takes the same lock as every other writer of the file, and
    writes back to the file it read (``machine_file``).
    """
    return machine_file.editing(locations.STORAGE_PATH)
