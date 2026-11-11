"""Experiment directory: config.yaml read/write and path resolution (Sections 1-2).

A experiment is a portable directory — `config.yaml` (identity only) plus the
`recordings/ analysis/ models/ datasets/` tree. It contains no absolute paths, no
machine-specific config, and no credentials (Tenet 6, Tenet 14). Recordings are
discovered by scanning `recordings/`, never listed in config.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

from detail.core import atomic
from detail.storage import dirscan
from detail.storage import read_cache

SCHEMA_VERSION = 1
CONFIG_FILENAME = "config.yaml"
SUBDIRS = ("recordings", "analysis", "models", "datasets", "runs")


def create_experiment(
    parent_dir: Path | str, name: str, created_at: str | None = None
) -> Path:
    """Create a new experiment directory under ``parent_dir`` and return its path.

    Raises:
        ValueError: If ``name`` is empty.
        FileExistsError: If a experiment already exists at the target path.
    """
    if not name:
        raise ValueError("experiment name must be non-empty")
    experiment_dir = Path(parent_dir) / name
    if (experiment_dir / CONFIG_FILENAME).exists():
        raise FileExistsError(f"a experiment already exists at {experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (experiment_dir / sub).mkdir(exist_ok=True)
    if created_at is None:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_config(experiment_dir, {"name": name, "created_at": created_at})
    return experiment_dir


def _refuse_legacy_layout(exp_dir: Path) -> None:
    if (exp_dir / "sessions").is_dir() and not (exp_dir / "recordings").is_dir():
        raise ValueError(
            f"{exp_dir} uses the legacy 'sessions/' layout (before the "
            f"Session->Recording rename). It was written by a pre-release version and "
            f"has to be migrated before it can be opened."
        )


def _parse_config(exp_dir: Path, path: Path) -> dict:
    _refuse_legacy_layout(exp_dir)
    config = yaml.safe_load(atomic.read_text(path))
    if not isinstance(config, dict):
        raise ValueError(f"{path} is not a mapping")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema_version is {config.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    if not config.get("name"):
        raise ValueError(f"{path} has no 'name'")
    return config


def read_config(experiment_dir: Path | str) -> dict:
    """Read and validate a experiment's ``config.yaml``.

    Memoized against the file's mtime + size (``storage.read_cache``). ``Experiment.name``
    and ``.description`` are separate properties over this one file, so a surface that shows
    both reads it twice, and Home does that once per card — four round trips per experiment
    for a value that changes when someone renames it. The validation runs on the miss, so a
    malformed or legacy-layout experiment still raises on every call (an error is never
    cached).

    Raises:
        FileNotFoundError: If ``config.yaml`` is absent.
        ValueError: If it is malformed, has the wrong ``schema_version``, or has
            no ``name``.
    """
    exp_dir = Path(experiment_dir)
    path = exp_dir / CONFIG_FILENAME
    try:
        return read_cache.cached_document(path, lambda: _parse_config(exp_dir, path))
    except FileNotFoundError:
        # The legacy check is what turns "no config.yaml" into an actionable message, so it
        # runs on the failing path rather than costing two stats on every successful read.
        _refuse_legacy_layout(exp_dir)
        raise FileNotFoundError(f"no {CONFIG_FILENAME} in {experiment_dir}") from None


def write_config(experiment_dir: Path | str, config: dict) -> None:
    """Write ``config.yaml`` with ``schema_version`` forced to the literal first key (Tenet 7).

    Atomic, because it is rewritten while it is read — Home reads it for every experiment
    card — and with a temp name **unique to this writer**, because it has more than one:
    ``set_metadata`` from the GUI and ``next_recording_id`` from any process that adds a
    recording (a CLI import beside the GUI). On a shared ``config.yaml.tmp`` the second
    ``os.replace`` finds the first already moved it; ``roots._write`` is the same case.
    """
    if not config.get("name"):
        raise ValueError("config must have a non-empty 'name'")
    ordered = {"schema_version": SCHEMA_VERSION}
    for key, value in config.items():
        if key != "schema_version":
            ordered[key] = value
    path = Path(experiment_dir) / CONFIG_FILENAME
    tmp = path.parent / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(ordered, f, sort_keys=False)
        atomic.replace(tmp, path)
    finally:
        tmp.unlink(
            missing_ok=True
        )  # a failed write must not leave litter in the experiment


RECORDING_ID_PREFIX = "recording_"
_RECORDING_ID_DIGITS = 3


def next_recording_id(experiment_dir: Path | str) -> str:
    """Mint the next recording id (``recording_000``, ``recording_001``, …).

    The counter lives in ``config.yaml`` and only ever goes **up**, so an id is never
    reused. Deriving it from the highest existing id instead would hand a deleted
    recording's id to the next one, and every run record, export and analysis row that
    named it would silently point at a different animal.

    Absent from ``config.yaml`` (every experiment made before ids were generated) it
    starts at 0, and any id already on disk is skipped — so a hand-made ``recording_000``
    cannot be collided with. Zero-padded, so ``discover_recordings``' sort is creation
    order.
    """
    exp_dir = Path(experiment_dir)
    config = read_config(exp_dir)
    index = int(config.get("next_recording_index", 0))
    existing = set(discover_recordings(exp_dir))
    while True:
        recording_id = f"{RECORDING_ID_PREFIX}{index:0{_RECORDING_ID_DIGITS}d}"
        index += 1
        if recording_id not in existing:
            break
    config["next_recording_index"] = index
    write_config(exp_dir, config)
    return recording_id


def set_metadata(
    experiment_dir: Path | str,
    name: str | None = None,
    description: str | None = None,
) -> None:
    """Edit experiment identity in config.yaml. Only the given fields change.

    ``name`` is a display field decoupled from the directory — the folder is not
    renamed.

    Raises:
        ValueError: If ``name`` is given but empty.
    """
    config = read_config(experiment_dir)
    if name is not None:
        if not name:
            raise ValueError("experiment name must be non-empty")
        config["name"] = name
    if description is not None:
        config["description"] = description
    write_config(experiment_dir, config)


def discover_recordings(experiment_dir: Path | str) -> list[str]:
    """Return the sorted ids of recordings (subdirs of ``recordings/`` holding a recording.yaml)."""
    return dirscan.discover_ids(recordings_dir(experiment_dir), "recording.yaml")


def recordings_dir(experiment_dir: Path | str) -> Path:
    return Path(experiment_dir) / "recordings"


def analysis_dir(experiment_dir: Path | str) -> Path:
    return Path(experiment_dir) / "analysis"


def models_dir(experiment_dir: Path | str) -> Path:
    return Path(experiment_dir) / "models"


def datasets_dir(experiment_dir: Path | str) -> Path:
    return Path(experiment_dir) / "datasets"


# -- keypoints (§26): the path helpers land with their first caller, after this line --
def heading_definitions_path(experiment_dir: Path | str) -> Path:
    """Saved heading definitions (§14 *Egocentric alignment*), the stream-preset pattern."""
    return Path(experiment_dir) / "heading_definitions.yaml"


def examples_dir(experiment_dir: Path | str) -> Path:
    """Example sets (§26): ``examples/<id>/examples.yaml`` + ``frames/<sha>.png``. Audit sets
    (``purpose: audit``) share the directory; the purpose, not the folder, keeps them apart.
    """
    return Path(experiment_dir) / "examples"


def exports_dir(experiment_dir: Path | str) -> Path:
    """Export **run records**, not exported files — those go to the user's own folder (#354)."""
    return Path(experiment_dir) / "exports"


def recording_path(experiment_dir: Path | str, recording_id: str) -> Path:
    return recordings_dir(experiment_dir) / recording_id


def _is_absolute_any_os(relative: Path | str) -> bool:
    """True if ``relative`` is absolute under POSIX *or* Windows rules.

    ``Path.is_absolute()`` is OS-native: on Windows it returns False for a
    POSIX path like ``/Users/x`` (no drive), so the native check alone lets
    foreign-OS absolute paths slip past the Tenet 6 guard.
    """
    text = str(relative)
    return PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()


def experiment_path(experiment_dir: Path | str, relative: Path | str) -> Path:
    """Resolve a experiment-relative path to an absolute one.

    Raises:
        ValueError: If ``relative`` is absolute — experiment files store paths
            relative to the experiment root (Tenet 6).
    """
    if _is_absolute_any_os(relative):
        raise ValueError(
            f"path {str(relative)!r} is absolute; experiment files store paths "
            "relative to the experiment root (Tenet 6)"
        )
    return Path(experiment_dir) / Path(relative)
