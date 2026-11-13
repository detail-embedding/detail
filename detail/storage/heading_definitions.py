"""Saved heading definitions (§14 "Egocentric alignment"): ``heading_definitions.yaml``,
one file per experiment, the stream-preset pattern (§3).

A definition is a name plus two sets of skeleton point names, ``front`` and ``back``. It is
stored as given and checked against a skeleton at *use* time (`analysis.body_axis`), the
way a stream preset's config is checked by its importer; an embedding run copies the
resolved definition into its own record, so a definition can change or go without touching
any run.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import yaml

from detail.analysis.body_axis import validate_definition
from detail.core import atomic
from detail.core import yaml_load
from detail.storage import experiment as experiment_store

SCHEMA_VERSION = 1

_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _path(experiment_dir: Path | str) -> Path:
    return experiment_store.heading_definitions_path(experiment_dir)


@contextlib.contextmanager
def _lock(experiment_dir: Path | str):
    with atomic.file_lock(_path(experiment_dir).with_suffix(".yaml.lock")):
        yield


def validate_name(name: str) -> None:
    """Raises: ValueError if ``name`` is empty or outside ``[A-Za-z0-9_-]``."""
    if not name:
        raise ValueError("heading definition name must be non-empty")
    if set(name) - _ALLOWED:
        raise ValueError(
            f"heading definition name {name!r} must contain only letters, digits, '-' "
            "and '_'"
        )


def list_definitions(experiment_dir: Path | str) -> list[dict]:
    """Every saved definition, oldest first: ``{name, front, back}``; ``[]`` when none."""
    path = _path(experiment_dir)
    if not path.is_file():
        return []
    data = yaml_load.safe_load(atomic.read_text(path)) or {}
    return [dict(d) for d in data.get("definitions", [])]


def get_definition(experiment_dir: Path | str, name: str) -> dict:
    """Raises: ValueError if no definition named ``name`` exists."""
    definitions = list_definitions(experiment_dir)
    for definition in definitions:
        if definition.get("name") == name:
            return definition
    known = ", ".join(sorted(d.get("name", "") for d in definitions)) or "none saved"
    raise ValueError(f"no heading definition named {name!r} (have: {known})")


def _write(experiment_dir: Path | str, definitions: list[dict]) -> None:
    path = _path(experiment_dir)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.dump(
            {"schema_version": SCHEMA_VERSION, "definitions": definitions},
            handle,
            sort_keys=False,
        )
    atomic.replace(tmp, path)


def save_definition(
    experiment_dir: Path | str,
    name: str,
    front: list[str],
    back: list[str],
    *,
    overwrite: bool = False,
) -> dict:
    """Save (or, with ``overwrite``, replace) a named definition; returns it.

    Raises:
        ValueError: An invalid name or definition, or the name exists and ``overwrite`` is
            False.
    """
    validate_name(name)
    checked = validate_definition({"front": front, "back": back})
    entry = {"name": name, "front": checked["front"], "back": checked["back"]}
    with _lock(experiment_dir):
        definitions = list_definitions(experiment_dir)
        if any(d.get("name") == name for d in definitions) and not overwrite:
            raise ValueError(
                f"a heading definition named {name!r} already exists; pass overwrite to "
                "replace it"
            )
        kept = [d for d in definitions if d.get("name") != name]
        _write(experiment_dir, kept + [entry])
    return entry


def delete_definition(experiment_dir: Path | str, name: str) -> None:
    """Raises: ValueError if no definition named ``name`` exists."""
    with _lock(experiment_dir):
        definitions = list_definitions(experiment_dir)
        remaining = [d for d in definitions if d.get("name") != name]
        if len(remaining) == len(definitions):
            raise ValueError(f"no heading definition named {name!r}")
        _write(experiment_dir, remaining)


def export_definition(
    experiment_dir: Path | str, name: str, out_path: Path | str
) -> None:
    """Write one saved definition to a standalone, shareable YAML file.

    Raises:
        ValueError: No definition named ``name`` exists.
    """
    definition = get_definition(experiment_dir, name)
    Path(out_path).write_text(
        yaml.dump({"schema_version": SCHEMA_VERSION, **definition}, sort_keys=False),
        encoding="utf-8",
    )


def import_definition(
    experiment_dir: Path | str,
    in_path: Path | str,
    *,
    name: str | None = None,
    overwrite: bool = False,
) -> str:
    """Read a standalone definition file (as written by :func:`export_definition`) into
    this experiment; returns the name it was saved under.

    Raises:
        FileNotFoundError: ``in_path`` does not exist.
        ValueError: A malformed file, the wrong ``schema_version``, no name, or (without
            ``overwrite``) the name already exists.
    """
    path = Path(in_path)
    if not path.is_file():
        raise FileNotFoundError(f"no heading definition file at {path}")
    data = yaml_load.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a heading definition file")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version {data.get('schema_version')!r}; expected "
            f"{SCHEMA_VERSION}"
        )
    final = name or data.get("name")
    if not final:
        raise ValueError(f"{path} names no definition; pass a name")
    save_definition(
        experiment_dir,
        str(final),
        list(data.get("front") or []),
        list(data.get("back") or []),
        overwrite=overwrite,
    )
    return str(final)


def fits(definition: dict, names) -> str | None:
    """Why a definition cannot be used with a skeleton, or ``None`` when it fits."""
    try:
        validate_definition(definition, tuple(names))
    except ValueError as exc:
        return str(exc)
    return None
