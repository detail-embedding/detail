"""Machine-local storage locations (Section 2) — minimal slice.

`~/.config/detail/storage.yaml` maps storage-alias names to paths on this
machine (never committed; never experiment state). The ``default`` alias's ``root``
is where experiment directories live and are discovered — the managed workspace the
GUI Home screen lists. Only local-path resolution of ``default`` is implemented
here; the full `StorageLocation` abstraction (cloud backends, video_uri /
fs_store) is still to come.

**The file is parsed once, and again only when it changes on disk** (:func:`read_storage`).
Every lookup here and in ``roots.py`` goes through that one reader. They used to parse the
file on every call, which nobody would call slow: one small file. But the lookups sit under
per-item paths (a montage tile asks where the preview cache is twice per picture), and the
file grows by one alias per folder a video is imported from. Guard and write-up:
``tests/storage/test_machine_config_reads.py``.
"""

import os
from pathlib import Path

import yaml

from detail.core import atomic
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


def experiments_root() -> Path:
    """The local root holding experiment directories — the ``default`` alias root.

    Reads ``storage.yaml`` if present; otherwise falls back to ``DEFAULT_ROOT``
    (configuring a custom root via storage.yaml / Preferences comes later). The
    directory is created if missing.
    """
    root = _experiments_root_path()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _experiments_root_path() -> Path:
    """Where :func:`experiments_root` resolves, without creating it."""
    data = read_storage()
    configured = data.get("storage", {}).get("default", {}).get("root")
    return Path(configured).expanduser() if configured else DEFAULT_ROOT


def configured_model_cache() -> Path | None:
    """The ``default`` alias's ``model_cache`` path from storage.yaml, or ``None``.

    ``None`` when unset — ``models.hf_access.model_cache_dir`` then falls back to
    its own default. Not created here; the caller owns creation, as with the
    weight-download path.
    """
    data = read_storage()
    configured = data.get("storage", {}).get("default", {}).get("model_cache")
    return Path(configured).expanduser() if configured else None


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
    """Experiment directories under the managed root (dirs holding a config.yaml)."""
    return dirscan.dirs_containing(experiments_root(), "config.yaml")


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

    For the writer's read-modify-write, under its lock (``roots.register_root``). What it
    merges into must be what is on disk: a cached copy that missed another process's write
    would be written back over it, and that write's alias would be gone.
    """
    try:
        return _parse_storage()
    except (FileNotFoundError, NotADirectoryError):
        return {}


def _parse_storage() -> dict:
    return yaml.safe_load(atomic.read_text(STORAGE_PATH)) or {}
