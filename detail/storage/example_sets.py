"""Example-set storage (§26 "Example sets"): the files behind ``core.examples``.

```
<experiment>/examples/ex_01/
├── examples.yaml       schema_version first; the skeleton copy, the frames, their marks, the digest
└── frames/
    ├── a1f3c9d2e0b4….png   the full decoded frame, lossless, named by its content hash
    └── 9c2e71….png
```

**Copied in**, so a run never re-seeks a video for its examples and a set survives a moved video
library. **Frames are immutable**: an image is written once under its content-hash name and never
rewritten; removing a frame from the set leaves the PNG while any run may list it. **Edited in
place, atomically** (``core.atomic.replace``: the labeling screen saves every click while a run may
be reading). There is no copy-on-write and no ``used/`` marker (#626 had both): a
``keypoint_detect`` run records the set's digest, the image names it used and a full copy of the
marks, so a later edit never changes a run and a run never depends on the set still existing.
**Deleting a set is refused while a run lists it**, found by scanning the run records
(``detect.yaml → example_set.id``), so there is nothing to go stale.

The **digest covers the images by hash** (``ExampleSet.digest``), so swapping an image under the
same name changes it. Audit sets (``purpose: audit``, §26 "Auditing") share this layout; the
purpose field keeps the two apart, not a folder.

Sets are discovered by scanning, like datasets, with no central list.
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import yaml

from detail.core import atomic
from detail.core.examples import PURPOSES, ExampleSet
from detail.core.skeleton import Skeleton
from detail.core.utils import utc_timestamp
from detail.storage import dirscan
from detail.storage import experiment as experiment_store
from detail.storage import id_allocator
from detail.storage import read_cache

SCHEMA_VERSION = 1
EXAMPLES_FILENAME = "examples.yaml"
FRAMES_DIR = "frames"
ID_PREFIX = "ex"
#: Hex digits of the content hash an image is named by; the full hash goes into the digest.
_NAME_HEX = 16


def set_dir(experiment_dir: Path | str, set_id: str) -> Path:
    return experiment_store.examples_dir(experiment_dir) / set_id


def discover_sets(experiment_dir: Path | str) -> list[str]:
    """Ids of the experiment's example sets (every purpose), in id order."""
    root = experiment_store.examples_dir(experiment_dir)
    return dirscan.discover_ids(root, EXAMPLES_FILENAME)


def read_set(experiment_dir: Path | str, set_id: str) -> ExampleSet:
    """Raises ``FileNotFoundError`` naming the sets that do exist."""
    path = set_dir(experiment_dir, set_id) / EXAMPLES_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no example set {set_id!r} in this experiment; it has "
            f"{discover_sets(experiment_dir) or 'none'}"
        )
    return ExampleSet.from_dict(yaml.safe_load(atomic.read_text(path)))


def list_sets(
    experiment_dir: Path | str, purpose: str | None = None
) -> list[ExampleSet]:
    """Every readable set, optionally of one ``purpose``; one torn file costs its row, not
    the listing (``dirscan``)."""
    records = dirscan.read_each(
        discover_sets(experiment_dir),
        lambda set_id: yaml.safe_load(
            atomic.read_text(set_dir(experiment_dir, set_id) / EXAMPLES_FILENAME)
        ),
    )
    sets = [ExampleSet.from_dict(r) for r in records]
    if purpose is not None:
        sets = [s for s in sets if s.purpose == purpose]
    return sets


def write_set(experiment_dir: Path | str, example_set: ExampleSet) -> None:
    """Write ``examples.yaml``, ``schema_version`` first (Tenet 7), with the digest over the
    marks and the images' own hashes, atomically."""
    out = example_set.to_dict(image_hashes=image_hashes(experiment_dir, example_set))
    path = set_dir(experiment_dir, example_set.id) / EXAMPLES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.dump(out, handle, sort_keys=False)
    atomic.replace(tmp, path)


def _reserve_id(root: Path) -> str:
    """Claim the next ``ex_NN`` directory under ``root``: two creators racing never share an
    id, and a deleted set's id is never handed out again, so a distill record's
    ``example_set.id`` cannot come to name another set (``storage/id_allocator.py``)."""
    return id_allocator.claim_dir(root, ID_PREFIX, digits=2)


def create_set(
    experiment_dir: Path | str,
    name: str,
    skeleton: Skeleton,
    *,
    purpose: str = "example",
) -> ExampleSet:
    """A new, empty set holding a copy of ``skeleton``.

    Raises:
        ValueError: An invalid skeleton (``Skeleton.validate``), an empty name, or a purpose
            outside ``core.examples.PURPOSES``.
    """
    skeleton.validate()
    if not str(name).strip():
        raise ValueError("an example set needs a name")
    if purpose not in PURPOSES:
        raise ValueError(f"example set purpose {purpose!r}; expected one of {PURPOSES}")
    example_set = ExampleSet(
        id=_reserve_id(experiment_store.examples_dir(experiment_dir)),
        name=str(name).strip(),
        purpose=purpose,
        created_at=utc_timestamp(),
        skeleton=Skeleton.from_dict(skeleton.to_dict()),
    )
    write_set(experiment_dir, example_set)
    return example_set


@contextlib.contextmanager
def removed_on_failure(experiment_dir: Path | str, set_id: str) -> Iterator[None]:
    """Delete the set ``set_id`` if the block raises (Ctrl-C included), then re-raise.

    For a set just made by :func:`create_set` and still being filled: an import that fails
    half-way must not leave a set behind that the user never asked for.
    """
    try:
        yield
    except BaseException:
        shutil.rmtree(set_dir(experiment_dir, set_id), ignore_errors=True)
        raise


def add_frame_image(experiment_dir: Path | str, set_id: str, rgb: np.ndarray) -> str:
    """Write ``rgb`` (RGB uint8) as a lossless PNG named by its content hash and return its
    path relative to the set, posix (``frames/<sha>.png``).

    Write-once: a picture already present under that name is left as it is, so an image is
    never rewritten and a name in the YAML pins its content.
    """
    ok, encoded = cv2.imencode(
        ".png", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    )
    if not ok:
        raise OSError("could not encode the frame image as PNG")
    data = encoded.tobytes()
    name = hashlib.sha256(data).hexdigest()[:_NAME_HEX] + ".png"
    frames = set_dir(experiment_dir, set_id) / FRAMES_DIR
    frames.mkdir(parents=True, exist_ok=True)
    path = frames / name
    if not path.is_file():
        tmp = frames / (name + ".tmp")
        tmp.write_bytes(data)
        atomic.replace(tmp, path)
    return f"{FRAMES_DIR}/{name}"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_hashes(experiment_dir: Path | str, example_set: ExampleSet) -> dict[str, str]:
    """``image -> sha256`` of each frame image present on disk, for ``ExampleSet.digest``.

    Memoized on each file's mtime (``read_cache``): the labeling screen saves on every click,
    and hashing twenty 1080p PNGs each time would be the cost of a click.
    """
    base = set_dir(experiment_dir, example_set.id)
    out: dict[str, str] = {}
    for frame in example_set.frames:
        path = base / frame.image
        if path.is_file():
            out[frame.image] = read_cache.cached_read(
                ("example_image_sha256", str(path)),
                path,
                lambda path=path: _file_sha256(path),
            )
    return out


def load_image(experiment_dir: Path | str, set_id: str, image: str) -> np.ndarray:
    """One frame image of a set as RGB uint8; raises ``FileNotFoundError`` naming the file."""
    path = set_dir(experiment_dir, set_id) / image
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"frame image missing or unreadable: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_images(
    experiment_dir: Path | str, example_set: ExampleSet
) -> list[np.ndarray]:
    """Every frame image of ``example_set``, in frame order."""
    return [
        load_image(experiment_dir, example_set.id, f.image) for f in example_set.frames
    ]


def runs_using(experiment_dir: Path | str, set_id: str) -> list[tuple[str, str]]:
    """``(recording_id, run_id)`` for every ``keypoint_detect`` run whose record names this
    set (``detect.yaml → example_set.id``, §26). Read from the run records, so nothing here
    has to be kept up to date by the stage."""
    root = experiment_store.recordings_dir(experiment_dir)
    if not root.is_dir():
        return []
    out = []
    for recording_dir in dirscan.child_dirs(root):
        for meta in sorted((recording_dir / "keypoint_detect").glob("*/detect.yaml")):
            data = yaml.safe_load(atomic.read_text(meta)) or {}
            ref = data.get("example_set") or {}
            if ref.get("id") == set_id:
                out.append((recording_dir.name, meta.parent.name))
    return out


def delete_set(experiment_dir: Path | str, set_id: str) -> None:
    """Delete a set no run lists. Refuses one a run recorded: the run keeps a copy of the
    marks, but its frames are read from here.

    Raises:
        FileNotFoundError: No such set (naming those that exist).
        ValueError: A ``keypoint_detect`` run lists the set.
    """
    read_set(experiment_dir, set_id)
    runs = runs_using(experiment_dir, set_id)
    if runs:
        names = ", ".join(f"{run} ({rec})" for rec, run in runs)
        raise ValueError(
            f"example set {set_id!r} is listed by keypoint runs {names}; it is kept so "
            "those runs stay reproducible"
        )
    shutil.rmtree(set_dir(experiment_dir, set_id))
