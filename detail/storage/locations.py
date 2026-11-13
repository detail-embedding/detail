"""Machine-local storage locations (Section 2) — the local-folder slice.

`~/.config/detail/storage.yaml` maps storage-alias names to paths on this
machine (never committed; never experiment state). Every ``type: local`` alias under
``storage:`` is an **experiment location**: a folder experiment directories live in and
are discovered from. The ``default`` alias always exists (its ``root`` falls back to
``DEFAULT_ROOT``); any other is one the user added — an external drive, a share — so
that a whole experiment can live there. Cloud backends and the full `StorageLocation`
abstraction (video_uri / fs_store) are still to come; a non-local alias is not listed.

**The file is parsed once, and again only when it changes on disk** (:func:`read_storage`).
Every lookup here and in ``roots.py`` goes through that one reader. They used to parse the
file on every call, which nobody would call slow: one small file. But the lookups sit under
per-item paths (a montage tile asks where the preview cache is twice per picture), and the
file grows by one alias per folder a video is imported from. Guard and write-up:
``tests/storage/test_machine_config_reads.py``.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path


from detail.core import atomic
from detail.core import yaml_load
from detail.storage import dirscan
from detail.storage import read_cache

# ``DETAIL_CONFIG_DIR`` relocates this machine's config (storage.yaml, credentials.yaml).
# Read at import, which is what makes it work for a **spawned worker**: the child imports
# fresh and picks up the parent's environment. An in-process caller that needs to redirect
# after import (the test isolation fixture) monkeypatches the constants instead — the two
# cover different cases and neither replaces the other.
CONFIG_DIR = Path(
    os.environ.get("DETAIL_CONFIG_DIR") or Path.home() / ".config" / "detail"
)
STORAGE_PATH = CONFIG_DIR / "storage.yaml"
# Default managed-workspace folder. Renamed from the legacy "SegProjects" as part
# of the DETAIL rename.
DEFAULT_ROOT = Path.home() / "Documents" / "DetailExperiments"
DEFAULT_LOCATION = "default"
_LOCATION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}")


class LocationNotConnectedError(FileNotFoundError):
    """A location is registered but its folder is not there — a drive not plugged in."""


@dataclass(frozen=True)
class ExperimentLocation:
    """A folder experiments live in, by its storage.yaml name."""

    name: str
    root: Path
    connected: bool

    @property
    def is_default(self) -> bool:
        return self.name == DEFAULT_LOCATION


def experiments_root() -> Path:
    """The local root holding experiment directories — the ``default`` alias root.

    Reads ``storage.yaml`` if present; otherwise falls back to ``DEFAULT_ROOT``. The
    directory is created if missing. Machine-global trees (``tools/``, ``roi_exports/``)
    live here too, whatever other locations exist.
    """
    configured = _storage().get(DEFAULT_LOCATION, {}).get("root")
    root = Path(configured).expanduser() if configured else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def configured_model_cache() -> Path | None:
    """The ``default`` alias's ``model_cache`` path from storage.yaml, or ``None``.

    ``None`` when unset — ``models.hf_access.model_cache_dir`` then falls back to
    its own default. Not created here; the caller owns creation, as with the
    weight-download path.
    """
    configured = _storage().get(DEFAULT_LOCATION, {}).get("model_cache")
    return Path(configured).expanduser() if configured else None


def experiment_locations() -> list[ExperimentLocation]:
    """Every experiment location on this machine: ``default`` first, then the rest by name.

    A location whose folder is missing is listed with ``connected=False`` rather than
    dropped or created — an unplugged drive is a state to show the user, and creating the
    folder would put an empty one on whatever disk now holds that drive letter.
    """
    found = [ExperimentLocation(DEFAULT_LOCATION, experiments_root(), True)]
    for name, entry in sorted(_storage().items()):
        if name == DEFAULT_LOCATION or not _is_local_location(entry):
            continue
        root = Path(entry["root"]).expanduser()
        found.append(ExperimentLocation(name, root, root.is_dir()))
    return found


def location_root(name: str) -> Path:
    """The folder of location ``name``, which must be connected.

    Raises:
        ValueError: No location has that name.
        LocationNotConnectedError: Its folder is missing (the drive is not plugged in).
    """
    for location in experiment_locations():
        if location.name != name:
            continue
        if not location.connected:
            raise LocationNotConnectedError(
                f"location {name!r} ({location.root.as_posix()}) is not connected — "
                "plug in the drive that holds it, or point the name at the folder's new "
                f"place with `detail experiment add-location {name!r} <folder>`"
            )
        return location.root
    known = ", ".join(repr(loc.name) for loc in experiment_locations())
    raise ValueError(f"no experiment location named {name!r} (known: {known})")


def location_of(
    experiment_dir: Path | str, *, among: list[ExperimentLocation] | None = None
) -> str | None:
    """The name of the location ``experiment_dir`` sits directly in, or ``None``.

    ``among`` is a list :func:`experiment_locations` already returned, for a caller asking
    about many experiments at once.
    """
    parent = _same_path_key(Path(experiment_dir).parent)
    for location in among if among is not None else experiment_locations():
        if _same_path_key(location.root) == parent:
            return location.name
    return None


def add_location(name: str, folder: Path | str) -> ExperimentLocation:
    """Register ``folder`` as experiment location ``name``, or re-point an existing name.

    Re-pointing is how a drive that came back under a different letter is found again.
    Only this machine's storage.yaml changes; no experiment file holds the path.

    Raises:
        ValueError: A bad name, ``default``, or a folder already registered under
            another name.
        NotADirectoryError: ``folder`` is not an existing directory.
    """
    name = name.strip()
    if not _LOCATION_NAME.fullmatch(name):
        raise ValueError(
            f"location name {name!r} must start with a letter or digit and use only "
            "letters, digits, spaces, '.', '_' or '-' (at most 64 characters)"
        )
    if name.lower() == DEFAULT_LOCATION:
        raise ValueError(
            "'default' is the built-in location; choose another name for this folder"
        )
    resolved = Path(folder).expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"{resolved} is not an existing folder")
    for location in experiment_locations():
        if location.name != name and _same_path_key(location.root) == _same_path_key(
            resolved
        ):
            raise ValueError(
                f"{resolved.as_posix()} is already the experiment location "
                f"{location.name!r}"
            )
    from detail.storage import roots  # roots imports this module for its paths

    with roots.editing() as data:
        storage = dict(data.get("storage") or {})
        entry = dict(storage.get(name) or {})
        entry.update(type="local", root=resolved.as_posix())
        storage[name] = entry
        data["storage"] = storage
    return ExperimentLocation(name, resolved, True)


def remove_location(name: str) -> None:
    """Stop listing location ``name`` here. Nothing on disk is moved or deleted.

    Raises:
        ValueError: ``default``, or no location has that name.
    """
    if name == DEFAULT_LOCATION:
        raise ValueError("the default location cannot be removed")
    from detail.storage import roots  # roots imports this module for its paths

    with roots.editing() as data:
        storage = dict(data.get("storage") or {})
        if not _is_local_location(storage.get(name)):
            raise ValueError(f"no experiment location named {name!r}")
        del storage[name]
        data["storage"] = storage


PREVIEW_DIR_ENV = "DETAIL_PREVIEW_DIR"
_DEFAULT_PREVIEW_CACHE = Path.home() / ".cache" / "detail" / "previews"


def configured_preview_cache() -> Path | None:
    """The ``default`` alias's ``preview_cache`` path from storage.yaml, or ``None``."""
    data = read_storage()
    configured = data.get("storage", {}).get("default", {}).get("preview_cache")
    return Path(configured).expanduser() if configured else None


def preview_cache_dir() -> Path:
    """Where this machine keeps its display-only video previews (§3 *Video preview cache*).

    Resolution order: the ``DETAIL_PREVIEW_DIR`` env var, then the ``default`` alias's
    ``preview_cache`` in storage.yaml, then ``~/.cache/detail/previews/`` — the same order
    ``models.hf_access.model_cache_dir`` uses for weights. Resolved on every call, so a spawned
    worker and a test that sets the variable both see the current value; what a call costs is
    an environment lookup and, past that, one ``stat`` of storage.yaml (:func:`read_storage`),
    never a parse of it. Not created here; the builder creates what it writes.
    """
    override = os.environ.get(PREVIEW_DIR_ENV)
    if override:
        return Path(override).expanduser()
    configured = configured_preview_cache()
    if configured is not None:
        return configured
    return _DEFAULT_PREVIEW_CACHE


def list_experiment_dirs() -> list[Path]:
    """Experiment directories (dirs holding a config.yaml) in every connected location."""
    return [
        path
        for location in experiment_locations()
        if location.connected
        for path in dirscan.dirs_containing(location.root, "config.yaml")
    ]


def _is_local_location(entry: object) -> bool:
    return (
        isinstance(entry, dict)
        and entry.get("type", "local") == "local"
        and bool(entry.get("root"))
    )


def _same_path_key(path: Path) -> str:
    """A comparison key for two spellings of one folder (case-folded where the OS is)."""
    try:
        path = path.resolve()
    except OSError:
        pass
    return os.path.normcase(str(path))


def _storage() -> dict:
    return read_storage().get("storage") or {}


def read_storage() -> dict:
    """This machine's ``storage.yaml`` as a dict; ``{}`` when there is no file.

    The one reader of the file. A call costs a ``stat``; the parse happens once and again only
    when the file's ``(mtime, size)`` changes (``read_cache.cached_document``), so a rewrite by
    this process, a worker or another session is seen by the next call. The result is the
    caller's own copy.

    A read-modify-write must not use this: see :func:`read_storage_fresh`.
    """
    try:
        return read_cache.cached_document(STORAGE_PATH, _parse_storage)
    except (FileNotFoundError, NotADirectoryError):
        return {}  # no file yet: every setting takes its default


def read_storage_fresh() -> dict:
    """``storage.yaml`` parsed from disk now, bypassing the cache.

    What a read-modify-write must merge into. Every writer of this file goes through
    ``machine_file.editing``, which reads it fresh under its own lock and commits through
    ``read_cache.replace_document``; this is the same read, by name, for a caller that is
    not writing. What it
    merges into must be what is on disk: a cached copy that missed another process's write
    would be written back over it, and that write's alias would be gone.
    """
    try:
        return _parse_storage()
    except (FileNotFoundError, NotADirectoryError):
        return {}


def _parse_storage() -> dict:
    return yaml_load.safe_load(atomic.read_text(STORAGE_PATH)) or {}
