"""Saved stream-import presets (Section 3) — an importer plus the config to drive it.

Working out how to read one lab's CSV is a one-off act of attention: which column is time,
what it holds, which of forty columns are wanted. Doing it again by eye for each of sixty
recordings is the same act sixty times, with sixty chances to differ. A **preset** is that
answer, named and saved once per experiment:

```yaml
schema_version: 1
presets:
  - name: dlc_pose
    importer: timeseries_csv
    config:
      time_column: t_sec
      time_kind: seconds
      channels:
        - {column: nose_x, name: nose_x}
        - {column: nose_y, name: nose_y}
```

**A preset holds an importer's config, not one importer's columns.** It is keyed by
`importer` and carries whatever that importer's ``config_model`` describes, so a preset for
an importer added later needs no change here. Validating it is the importer's job, through
the same path a direct import takes.

**It deliberately does not carry an offset.** Alignment is per *recording* — each video and
logger pair has its own skew — so baking one number into a reusable preset would apply one
recording's answer to all of them, silently and plausibly. `offset_ns` is an optional field
for the case where the rig genuinely is deterministic (a hardware trigger starting camera
and logger together); it defaults to absent, and the batch surface takes offsets per
recording instead.

A read-modify-write list, the same hazard as ``recording.yaml``'s ``streams:`` list, so
save/delete take the sibling ``.lock`` file (``core.atomic.file_lock``) and the write is
atomic (tmp + ``os.replace``). ``export_preset`` / ``import_preset`` move one entry to and
from a standalone file, for another experiment or a labmate.
"""

import contextlib
from pathlib import Path

import yaml

from detail.core import atomic
from detail.storage.user_yaml import read_user_yaml

SCHEMA_VERSION = 1
PRESETS_FILENAME = "stream_presets.yaml"

#: The charset a preset name may use. Not a directory name (unlike a stream's), but it is a
#: CLI argument and a YAML key, so the same restraint keeps it quotable and greppable.
_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _path(experiment_dir: Path | str) -> Path:
    return Path(experiment_dir) / PRESETS_FILENAME


@contextlib.contextmanager
def _lock(experiment_dir: Path | str):
    with atomic.file_lock(_path(experiment_dir).with_suffix(".yaml.lock")):
        yield


def validate_preset_name(name: str) -> None:
    """Raises: ValueError if ``name`` is empty or outside ``[A-Za-z0-9_-]``."""
    if not name:
        raise ValueError("preset name must be non-empty")
    if set(name) - _ALLOWED:
        raise ValueError(
            f"preset name {name!r} must contain only letters, digits, '-' and '_'"
        )


def list_presets(experiment_dir: Path | str) -> list[dict]:
    """All saved presets, oldest-saved first; ``[]`` if none exist yet."""
    path = _path(experiment_dir)
    if not path.is_file():
        return []
    data = yaml.safe_load(atomic.read_text(path)) or {}
    return list(data.get("presets", []))


def get_preset(experiment_dir: Path | str, name: str) -> dict:
    """Raises: ValueError if no preset named ``name`` exists."""
    presets = list_presets(experiment_dir)
    for preset in presets:
        if preset.get("name") == name:
            return preset
    known = ", ".join(sorted(p.get("name", "") for p in presets)) or "none saved"
    raise ValueError(f"no stream preset named {name!r} (have: {known})")


def _write(experiment_dir: Path | str, presets: list[dict]) -> None:
    path = _path(experiment_dir)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.dump(
            {"schema_version": SCHEMA_VERSION, "presets": presets},
            handle,
            sort_keys=False,
        )
    atomic.replace(tmp, path)


def check_new_name(experiment_dir: Path | str, name: str) -> None:
    """Refuse ``name`` the way :func:`save_preset` without ``overwrite`` would, before any
    work that the save is to follow.

    ``recording add-stream --save-preset`` saves after the import, never before (a preset is
    worth keeping once its mapping has read a file). Checked only then, a taken name failed
    the command after the stream was already added, and a re-run said the stream existed
    (#29).

    Raises:
        ValueError: An invalid name, or a preset by that name exists.
    """
    validate_preset_name(name)
    _refuse_taken(list_presets(experiment_dir), name)


def _refuse_taken(presets: list[dict], name: str) -> None:
    if any(p.get("name") == name for p in presets):
        raise ValueError(
            f"a stream preset named {name!r} already exists; choose another name, or "
            f"delete it first (`detail experiment delete-stream-preset {name}`)"
        )


def save_preset(
    experiment_dir: Path | str,
    name: str,
    importer: str,
    config: dict | None,
    *,
    offset_ns: int | None = None,
    overwrite: bool = False,
) -> None:
    """Save (or, with ``overwrite``, replace) a named importer setup.

    The ``config`` is stored as given and validated by the importer at *use* time, through
    the same path a direct import takes. Checking it here as well would put a second copy of
    each importer's rules in the storage layer, which is the drift this indirection avoids.

    Args:
        offset_ns: An alignment offset to apply on every use. Leave ``None`` unless the rig
            starts video and stream together, since alignment is otherwise per recording.

    Raises:
        ValueError: An invalid name, or the name exists and ``overwrite`` is False.
    """
    validate_preset_name(name)
    with _lock(experiment_dir):
        presets = list_presets(experiment_dir)
        if not overwrite:
            _refuse_taken(presets, name)
        entry = {"name": name, "importer": importer, "config": dict(config or {})}
        if offset_ns is not None:
            entry["offset_ns"] = int(offset_ns)
        kept = [p for p in presets if p.get("name") != name]
        _write(experiment_dir, kept + [entry])


def delete_preset(experiment_dir: Path | str, name: str) -> None:
    """Raises: ValueError if no preset named ``name`` exists."""
    with _lock(experiment_dir):
        presets = list_presets(experiment_dir)
        remaining = [p for p in presets if p.get("name") != name]
        if len(remaining) == len(presets):
            raise ValueError(f"no stream preset named {name!r}")
        _write(experiment_dir, remaining)


def export_preset(experiment_dir: Path | str, name: str, out_path: Path | str) -> None:
    """Write one saved preset to a standalone, shareable YAML file.

    Raises:
        ValueError: No preset named ``name`` exists.
    """
    preset = get_preset(experiment_dir, name)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.dump({"schema_version": SCHEMA_VERSION, **preset}, sort_keys=False),
        encoding="utf-8",
    )


def import_preset(
    experiment_dir: Path | str,
    in_path: Path | str,
    *,
    name: str | None = None,
    overwrite: bool = False,
) -> str:
    """Read a standalone preset file (as written by :func:`export_preset`) into this
    experiment; returns the name it was saved under.

    Raises:
        FileNotFoundError: ``in_path`` does not exist.
        ValueError: A malformed file, the wrong ``schema_version``, no resolvable name or
            importer, or (without ``overwrite``) the name already exists.
    """
    path = Path(in_path)
    if not path.is_file():
        raise FileNotFoundError(f"no stream preset file at {path}")
    data = read_user_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema_version is {data.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    resolved = name or data.get("name")
    if not resolved:
        raise ValueError(f"{path} has no 'name' and none was given")
    importer = data.get("importer")
    if not importer:
        raise ValueError(f"{path} has no 'importer'")
    config = data.get("config")
    if config is not None and not isinstance(config, dict):
        raise ValueError(f"{path} has a non-mapping 'config'")
    save_preset(
        experiment_dir,
        resolved,
        importer,
        config,
        offset_ns=data.get("offset_ns"),
        overwrite=overwrite,
    )
    return resolved
