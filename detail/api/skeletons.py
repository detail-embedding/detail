"""The skeleton library (§26 *The Keypoints manager*): machine-wide, shared by every
experiment, behind the top bar's **Keypoints manager** and the ``detail skeleton`` CLI group.

A **skeleton** is the points, the edges that join them, the left/right pairs and where they
were drawn (``core.skeleton.Skeleton``). The library holds the four built-in **presets**
(read-only entries whose ids are the preset keys) and the user's own skeletons (``sk_NN``,
``storage/skeleton_library.py``).

**Copy on use.** Whatever uses a skeleton — an example set, a keypoint run, a trained student —
holds its own copy plus ``skeleton_ref: {id, name, digest}`` (:func:`ref`). Nothing reads the
library to run, so a missing library never breaks anything.

**Dependents freeze a skeleton.** :func:`dependents` searches the experiments the app knows
(``storage.locations.list_experiment_dirs``) for records whose ``skeleton_ref.id`` names it.
While any exist, the skeleton's points, edges, pairs and layout cannot change and it cannot be
deleted: an edit is saved as a new skeleton (:func:`save_as_new`). Its name may still change,
since a name is not identity. Delete the dependents and the skeleton is free again. An
experiment the app does not know about is not searched; it keeps its own copies, so nothing in
it breaks.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from detail.core import atomic
from detail.core.skeleton import PRESETS, Skeleton
from detail.storage import locations
from detail.storage import skeleton_library as library

#: Where a dependent's record sits under an experiment, and what to call it. Each record that
#: copies a skeleton carries ``skeleton_ref``; the search reads only that field.
_DEPENDENT_GLOBS = (
    ("examples/*/examples.yaml", "example set"),
    ("recordings/*/keypoint_detect/*/detect.yaml", "keypoint detect run"),
    ("recordings/*/keypoint_track/*/track.yaml", "keypoint track run"),
    ("datasets/*/dataset.yaml", "keypoint dataset"),
    ("models/*/model.yaml", "trained model"),
)


def _entry(record: dict, *, builtin: bool) -> dict:
    return {
        "id": record["id"],
        "name": record["name"],
        "skeleton": Skeleton.from_dict(record["skeleton"]),
        "builtin": builtin,
        "created_at": record.get("created_at"),
        "derived_from": record.get("derived_from"),
    }


def _preset_entry(preset_id: str) -> dict:
    label, sk = PRESETS[preset_id]
    return {
        "id": preset_id,
        "name": label,
        "skeleton": sk,
        "builtin": True,
        "created_at": None,
        "derived_from": None,
    }


def skeletons() -> list[dict]:
    """Every skeleton: the presets first (read-only), then the library in id order. Each is
    ``{id, name, skeleton, builtin, created_at, derived_from}``."""
    out = [_preset_entry(p) for p in PRESETS]
    out.extend(_entry(r, builtin=False) for r in library.list_records())
    return out


def skeleton(skeleton_id: str) -> dict:
    """One skeleton, preset or library.

    Raises:
        FileNotFoundError: No such skeleton (naming those that exist).
    """
    if skeleton_id in PRESETS:
        return _preset_entry(skeleton_id)
    return _entry(library.read(skeleton_id), builtin=False)


def ref(skeleton_id: str) -> dict:
    """``{id, name, digest}``: what a copy records about where its skeleton came from."""
    entry = skeleton(skeleton_id)
    return {
        "id": entry["id"],
        "name": entry["name"],
        "digest": entry["skeleton"].digest(),
    }


def resolve_skeleton(source) -> Skeleton:
    """A ``Skeleton`` from a ``Skeleton``, its dict, a preset or library id, or a YAML file
    holding a skeleton (bare, or under ``skeleton:`` as an export writes it).

    Raises:
        ValueError: An unknown id, or a file without a skeleton in it.
    """
    if isinstance(source, Skeleton):
        return source
    if isinstance(source, dict):
        return Skeleton.from_dict(source)
    text = str(source)
    path = Path(text)
    if path.suffix.lower() in (".yaml", ".yml") and path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        body = data.get("skeleton", data) if isinstance(data, dict) else None
        if not isinstance(body, dict) or "names" not in body:
            raise ValueError(f"{path} holds no skeleton (no 'names')")
        return Skeleton.from_dict(body)
    try:
        return skeleton(text)["skeleton"]
    except FileNotFoundError:
        raise ValueError(
            f"no skeleton preset, library skeleton or file {text!r}; presets are "
            f"{list(PRESETS)}, the library has {library.discover() or 'none'}"
        ) from None


def skeleton_source_name(source) -> str | None:
    """The name ``source`` gives its skeleton, for a library entry made from it: a preset's or
    library entry's name, a file's stated ``name:`` (else its stem); ``None`` for a
    ``Skeleton`` or dict, which carry none."""
    if isinstance(source, (Skeleton, dict)):
        return None
    text = str(source)
    path = Path(text)
    if path.suffix.lower() in (".yaml", ".yml") and path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        stated = data.get("name") if isinstance(data, dict) else None
        return str(stated) if stated else path.stem
    try:
        return skeleton(text)["name"]
    except FileNotFoundError:
        return None


def create_skeleton(name: str, source) -> dict:
    """A new library skeleton from ``source`` (anything :func:`resolve_skeleton` takes).

    Raises:
        ValueError: An empty name, an invalid skeleton, or an unknown source.
    """
    return _entry(library.create(name, resolve_skeleton(source)), builtin=False)


def save_as_new(skeleton_id: str, name: str, source) -> dict:
    """An edit kept as a new skeleton, ``derived_from`` the one it started from: what an edit
    to a skeleton with dependents (or to a preset) becomes."""
    skeleton(skeleton_id)  # FileNotFoundError when the source is gone
    return _entry(
        library.create(name, resolve_skeleton(source), derived_from=skeleton_id),
        builtin=False,
    )


def duplicate_skeleton(skeleton_id: str, *, name: str | None = None) -> dict:
    """A new library skeleton holding a copy of this one (preset or library)."""
    entry = skeleton(skeleton_id)
    return save_as_new(skeleton_id, name or f"{entry['name']} copy", entry["skeleton"])


def dependents(skeleton_id: str) -> list[dict]:
    """Everything in the known experiments that copied this skeleton:
    ``{experiment, experiment_dir, kind, record}`` each, ``record`` the path under the
    experiment. Reads small YAML files only, but a whole workspace's worth: never call it on
    the UI thread."""
    return dependents_by_skeleton().get(skeleton_id, [])


def dependents_by_skeleton() -> dict[str, list[dict]]:
    """:func:`dependents` for every skeleton at once, ``{skeleton id: [...]}``: one pass over
    the workspace, which is what a screen listing the whole library wants."""
    out: dict[str, list[dict]] = {}
    for exp_dir in locations.list_experiment_dirs():
        exp_name = exp_dir.name
        config = exp_dir / "config.yaml"
        try:
            exp_name = (yaml.safe_load(atomic.read_text(config)) or {}).get(
                "name", exp_name
            )
        except (OSError, yaml.YAMLError):
            pass
        for pattern, kind in _DEPENDENT_GLOBS:
            for path in sorted(exp_dir.glob(pattern)):
                try:
                    data = yaml.safe_load(atomic.read_text(path)) or {}
                except (OSError, yaml.YAMLError):
                    continue
                found = (data.get("skeleton_ref") or {}).get("id")
                if found:
                    out.setdefault(str(found), []).append(
                        {
                            "experiment": str(exp_name),
                            "experiment_dir": exp_dir,
                            "kind": kind,
                            "record": path.relative_to(exp_dir).as_posix(),
                        }
                    )
    return out


def _describe(found: list[dict]) -> str:
    shown = [f"{d['kind']} {d['record']} ({d['experiment']})" for d in found[:5]]
    more = f" and {len(found) - 5} more" if len(found) > 5 else ""
    return ", ".join(shown) + more


def update_skeleton(
    skeleton_id: str, *, name: str | None = None, skeleton=None
) -> dict:
    """Rename a library skeleton and/or replace its points, edges, pairs and layout.

    Raises:
        ValueError: A preset; an empty name; an invalid skeleton; or a skeleton change while
            something uses it (the message lists what, and says to save it as new).
    """
    if skeleton_id in PRESETS:
        raise ValueError(
            f"{skeleton_id!r} is a built-in preset and cannot change; save it as new"
        )
    record = library.read(skeleton_id)
    if name is not None:
        if not str(name).strip():
            raise ValueError("a skeleton needs a name")
        record["name"] = str(name).strip()
    if skeleton is not None:
        new = resolve_skeleton(skeleton)
        new.validate()
        if new.to_dict() != record["skeleton"]:
            found = dependents(skeleton_id)
            if found:
                raise ValueError(
                    f"skeleton {record['name']!r} is used by {_describe(found)}; save the "
                    "change as a new skeleton"
                )
            record["skeleton"] = new.to_dict()
            record["digest"] = new.digest()
    library.write(record)
    return _entry(library.read(skeleton_id), builtin=False)


def delete_skeleton(skeleton_id: str) -> None:
    """Delete a library skeleton nothing uses.

    Raises:
        ValueError: A preset, or a skeleton something uses (listed).
        FileNotFoundError: No such skeleton.
    """
    if skeleton_id in PRESETS:
        raise ValueError(f"{skeleton_id!r} is a built-in preset and cannot be deleted")
    record = library.read(skeleton_id)
    found = dependents(skeleton_id)
    if found:
        raise ValueError(
            f"skeleton {record['name']!r} is used by {_describe(found)}; delete those first"
        )
    library.remove(skeleton_id)


def export_skeleton(
    skeleton_id: str, out_path: Path | str, *, overwrite: bool = False
) -> Path:
    """Write one skeleton to a standalone YAML file another machine can import.

    Raises:
        FileExistsError: ``out_path`` exists and ``overwrite`` is False.
    """
    entry = skeleton(skeleton_id)
    out = Path(out_path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": library.SCHEMA_VERSION,
        "name": entry["name"],
        "skeleton": entry["skeleton"].to_dict(),
    }
    tmp = out.parent / (out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.dump(body, handle, sort_keys=False)
    atomic.replace(tmp, out)
    return out


def import_skeleton(in_path: Path | str, *, name: str | None = None) -> dict:
    """Read a skeleton file (an export, or a bare ``names``/``edges``/``pairs`` mapping) into
    the library as a new skeleton.

    Raises:
        FileNotFoundError: No such file.
        ValueError: A file without a skeleton in it.
    """
    path = Path(in_path)
    if not path.is_file():
        raise FileNotFoundError(f"no skeleton file at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    stated = data.get("name") if isinstance(data, dict) else None
    return create_skeleton(name or stated or path.stem, path)
