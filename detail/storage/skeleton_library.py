"""The machine-wide skeleton library (§26 *The Keypoints manager*): the files behind
``api/skeletons.py``.

```
<model_cache_dir()>/skeletons/sk_03/
└── skeleton.yaml    schema_version first; id, name, created_at, the skeleton, its digest,
                     derived_from (when made by Save as new or Duplicate)
```

Beside the imported-model catalog (``models/imported.py``), so machines that share a model
cache share their skeletons. Discovered by scanning, like that catalog: no index file. Written
atomically (``*.tmp`` + ``core.atomic.replace``), since the Keypoints manager and a CLI command
may touch the same record. Ids are ``sk_NN``, reserved by ``mkdir`` without ``exist_ok`` so two
creators racing never share one.

A skeleton that arrives with a model (promote, restore, import — §12 *Keypoint models in the
catalog*) goes through :func:`ensure_in_library`: reused when an entry or preset has the same
digest, otherwise added under a name no entry holds. An existing entry is never edited.

Nothing here knows about dependents: whether a skeleton may change is the API's question
(``api/skeletons.dependents``), answered by searching the experiments that copied it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from detail.core import atomic
from detail.core.skeleton import PRESETS
from detail.core.skeleton import Skeleton
from detail.core.utils import utc_timestamp
from detail.models.hf_access import model_cache_dir
from detail.storage import dirscan

SCHEMA_VERSION = 1
FILENAME = "skeleton.yaml"
ID_PREFIX = "sk"


def library_dir() -> Path:
    return model_cache_dir() / "skeletons"


def skeleton_dir(skeleton_id: str) -> Path:
    return library_dir() / skeleton_id


def discover() -> list[str]:
    """Ids of the library's skeletons, in id order."""
    root = library_dir()
    if not root.is_dir():
        return []
    return dirscan.discover_ids(root, FILENAME)


def read(skeleton_id: str) -> dict:
    """One record; ``FileNotFoundError`` naming the library's skeletons when absent."""
    path = skeleton_dir(skeleton_id) / FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no skeleton {skeleton_id!r} in the library; it has {discover() or 'none'}"
        )
    return yaml.safe_load(atomic.read_text(path))


def list_records() -> list[dict]:
    """Every readable record; a torn file costs its row, not the listing (``dirscan``)."""
    return dirscan.read_each(discover(), read)


def write(record: dict) -> None:
    """Write ``record`` (``schema_version`` first, Tenet 7), atomically."""
    out = {"schema_version": SCHEMA_VERSION}
    out.update({k: v for k, v in record.items() if k != "schema_version"})
    path = skeleton_dir(record["id"]) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.dump(out, handle, sort_keys=False)
    atomic.replace(tmp, path)


def _reserve_id() -> str:
    root = library_dir()
    root.mkdir(parents=True, exist_ok=True)
    taken = []
    for child in root.iterdir():
        stem, _, num = child.name.rpartition("_")
        if stem == ID_PREFIX and num.isdigit():
            taken.append(int(num))
    n = max(taken, default=0) + 1
    while True:
        candidate = f"{ID_PREFIX}_{n:02d}"
        try:
            (root / candidate).mkdir()
            return candidate
        except FileExistsError:
            n += 1


def create(name: str, skeleton: Skeleton, *, derived_from: str | None = None) -> dict:
    """A new record holding ``skeleton``.

    Raises:
        ValueError: An empty name or an invalid skeleton.
    """
    skeleton.validate()
    if not str(name).strip():
        raise ValueError("a skeleton needs a name")
    record = {
        "id": _reserve_id(),
        "name": str(name).strip(),
        "created_at": utc_timestamp(),
        "skeleton": skeleton.to_dict(),
        "digest": skeleton.digest(),
    }
    if derived_from:
        record["derived_from"] = derived_from
    write(record)
    return read(record["id"])


def remove(skeleton_id: str) -> None:
    """Delete one skeleton's directory (idempotent)."""
    shutil.rmtree(skeleton_dir(skeleton_id), ignore_errors=True)


def find_by_digest(digest: str) -> dict | None:
    """``{id, name, digest}`` of the first preset, then library entry, whose skeleton has
    ``digest``; ``None`` when nothing on this machine holds it. The digest is recomputed from
    each stored skeleton rather than read from its record, so a hand-edited file cannot match
    on a stale hash."""
    for preset_id, (label, sk) in PRESETS.items():
        if sk.digest() == digest:
            return {"id": preset_id, "name": label, "digest": digest}
    for record in list_records():
        if Skeleton.from_dict(record["skeleton"]).digest() == digest:
            return {"id": record["id"], "name": record["name"], "digest": digest}
    return None


def unique_name(name: str) -> str:
    """``name``, or ``"name (2)"``, ``"name (3)"``… — the first no preset or entry holds."""
    base = str(name).strip()
    taken = {label for label, _sk in PRESETS.values()}
    taken.update(record["name"] for record in list_records())
    if base not in taken:
        return base
    n = 2
    while f"{base} ({n})" in taken:
        n += 1
    return f"{base} ({n})"


def ensure_in_library(skeleton: Skeleton, name: str) -> dict:
    """``{id, name, digest}`` of the library entry (or preset) holding ``skeleton``: the one
    with its digest when there is one, otherwise a new entry under :func:`unique_name`.
    Never edits an existing entry.

    Raises:
        ValueError: An invalid skeleton, or an empty name when a new entry is needed.
    """
    skeleton.validate()
    found = find_by_digest(skeleton.digest())
    if found is not None:
        return found
    if not str(name).strip():
        raise ValueError("a skeleton needs a name")
    record = create(unique_name(name), skeleton)
    return {"id": record["id"], "name": record["name"], "digest": record["digest"]}
