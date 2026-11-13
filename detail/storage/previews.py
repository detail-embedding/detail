"""The machine-local video preview cache: layout, keys, and its two stores (§3 *Video preview
cache*).

**Display only.** A proxy is a small copy of a video part that the clip montage and other
playback surfaces draw from; no pipeline stage reads it, and nothing in the experiment points
at it (``tests/storage/test_preview_isolation.py`` holds the first half). It lives outside the
experiment, under :func:`~detail.storage.locations.preview_cache_dir`, so export cannot include
it, clearing it cannot delete experiment data, and it reads at local-disk speed when the
experiment is on a network share. Each machine builds its own.

Layout::

    <cache>/
        index.json                          video path -> (size, mtime_ns, key): skips re-hashing
        proxies/<key>/                      one per source video file, swapped in whole
            proxy.mp4                       480x270 at most (aspect kept), 15 fps, H.264 or mp4v
            frames.zarr/src_frame (n,) int64   proxy frame i is source frame src_frame[i]
            masks/<tracking run>-<digest>.zarr  the tracked animals at proxy size

``<key>`` is the file's content fingerprint (``integrity.file_fingerprint``: size + head/tail
hash), so a copied-in part and the referenced original share one proxy, and a re-encoded file
gets a new one. Masks hang under their proxy because they are indexed by its frames; the
``<digest>`` is of the tracking and detection run records (:func:`masks_digest`), so a re-run
under a reused id is a different store rather than a stale one.

There is **no second timestamp map**: ``src_frame`` names a source frame, whose ``t_ns`` is the
part's own map (Tenet 8). The proxy container's clock is not a time reference.

``storage/`` imports no ``pycocotools`` (§9): the mask writer receives encoded ASCII counts and
the reader returns them verbatim; ``api/previews.py`` encodes and decodes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detail.core import atomic
from detail.core.utils import subject_key
from detail.storage import integrity
from detail.storage import locations
from detail.storage import zarr_store

SCHEMA_VERSION = 1
PROXY_FILE = "proxy.mp4"
FRAMES_STORE = "frames.zarr"
MASKS_DIR = "masks"
INDEX_FILE = "index.json"
PROXIES_DIR = "proxies"
#: Rows per chunk of the frame map: a 30-minute part at 15 fps is 27,000 proxy frames.
FRAME_CHUNK = 100_000
#: Rows per chunk of the numeric mask columns, and of the RLE column (§9's split).
MASK_ROW_CHUNK = 10_000
MASK_RLE_CHUNK = 1_000
#: Rows buffered before a flush — a whole multiple of both chunk sizes (Tenet 12).
FLUSH_ROWS = 10_000


def cache_dir() -> Path:
    """The preview cache root on this machine (not created)."""
    return locations.preview_cache_dir()


def proxy_key(fingerprint: dict) -> str:
    """The directory name of a source file's proxy: its size and head/tail hash."""
    return f"{int(fingerprint['size'])}-{str(fingerprint['sha256_ends'])[:20]}"


def proxy_dir(key: str) -> Path:
    return cache_dir() / PROXIES_DIR / key


def proxy_complete(folder: Path) -> bool:
    """Whether ``folder`` holds a whole proxy: the video *and* its frame map.

    A folder can exist without being whole — a clear that could not delete a file in use, or
    a mask build that recreated ``proxies/<key>/masks/`` after a clear — so existence is never
    the test.
    """
    return (folder / PROXY_FILE).is_file() and (
        folder / FRAMES_STORE / "zarr.json"
    ).is_file()


def masks_complete(path: Path) -> bool:
    """Whether a mask store is whole (its group metadata is written last of its files to go
    and is what ``zarr_store.open_group`` needs)."""
    return (path / "zarr.json").is_file()


def proxy_present(key: str) -> bool:
    """True once a whole proxy has been swapped into place."""
    return proxy_complete(proxy_dir(key))


def masks_digest(tracking_entry: dict, detection_entry: dict) -> str:
    """A short digest of the run records a mask store was built from.

    Read off ``recording.yaml``'s entries rather than the zarr stores, so asking whether masks
    are missing costs one stat. A run deleted and re-run under the same id has a new
    ``created_at``, so its masks are a new store, not a stale one.
    """
    fields = [
        str(tracking_entry.get("id")),
        str(tracking_entry.get("created_at")),
        str(tracking_entry.get("every_n_frames")),
        str(detection_entry.get("id")),
        str(detection_entry.get("created_at")),
    ]
    # Masks of a class-filtered run built before its assignment was paired through the filter
    # drew each animal on another detection's mask. Naming the class makes those another
    # store, rebuilt rather than trusted; an unfiltered run's digest, and masks, are unchanged.
    if tracking_entry.get("class_name"):
        fields.append(f"class={tracking_entry['class_name']}")
    return hashlib.sha1("|".join(fields).encode("utf-8")).hexdigest()[:12]


def masks_path(key: str, tracking_run_id: str, digest: str) -> Path:
    return proxy_dir(key) / MASKS_DIR / f"{tracking_run_id}-{digest}.zarr"


# --------------------------------------------------------------------------- #
# index: video path -> key, so a re-check stats the file instead of hashing it
# --------------------------------------------------------------------------- #
def _index_path() -> Path:
    return cache_dir() / INDEX_FILE


def _read_index() -> dict:
    try:
        return json.loads(atomic.read_text(_index_path())) or {}
    except FileNotFoundError:
        return {}  # never written, or a clear removed it between our check and our read
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}  # a cache index is disposable; a torn one is rebuilt by re-hashing


def _write_index(index: dict) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{INDEX_FILE}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(index, indent=0, sort_keys=True), encoding="utf-8")
    atomic.replace(tmp, path)


def key_for_video(video_path: Path | str, *, compute: bool = True) -> str | None:
    """The proxy key of a source video file.

    One ``stat`` when the index already knows the file at this size and mtime; otherwise the
    fingerprint is computed (two 1 MiB reads) and remembered. ``compute=False`` answers
    ``None`` instead of reading the file. Machine-local, so the index holds absolute paths —
    it is never part of an experiment.
    """
    path = Path(video_path)
    stat = path.stat()
    name = str(path.resolve())
    index = _read_index()
    known = index.get(name)
    if (
        known
        and int(known.get("size", -1)) == stat.st_size
        and int(known.get("mtime_ns", -1)) == stat.st_mtime_ns
    ):
        return str(known["key"])
    if not compute:
        return None
    key = proxy_key(integrity.file_fingerprint(path))
    index = _read_index()  # re-read: another builder may have written meanwhile
    index[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "key": key}
    _write_index(index)
    return key


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #
#: Index temp files younger than this may belong to a writer in flight (``key_for_video`` runs
#: outside the build lock); staging directories are only ever made under it.
STALE_INDEX_TMP_S = 3600.0

_LOCK_FILE = ".build.lock"
_held = threading.local()


@contextlib.contextmanager
def build_lock(waiting: Callable[[], None] | None = None) -> Iterator[None]:
    """Hold the cache's one build lock: every build and every clear runs under it.

    One lock for the whole cache, across threads *and* processes (``core.atomic.file_lock`` —
    ``flock`` on its own descriptor, so a second thread of the same process waits too). Builds
    are sequential anyway, so this costs nothing, and it is what makes the rest safe: two
    builds of the same file (a GUI and a CLI, or an old queue still stopping) cannot touch
    each other's staging, and a clear cannot delete a folder while a build is writing into it.
    Re-entrant within a thread, so a mask build may build its missing proxy first.

    ``waiting`` is called while another holder has the lock (``core.atomic.file_lock``): a build
    passes one that reports "waiting" on its bar and raises on cancel, so a build queued behind
    another process's is neither silent nor stuck. ``None`` waits blocking, as a clear does.
    """
    depth = getattr(_held, "depth", 0)
    if depth:
        _held.depth = depth + 1
        try:
            yield
        finally:
            _held.depth -= 1
        return
    root = cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    with atomic.file_lock(root / _LOCK_FILE, waiting=waiting):
        _held.depth = 1
        try:
            yield
        finally:
            _held.depth = 0


def staging_dir(final: Path) -> Path:
    """A fresh sibling ``<final>.<random>.tmp`` directory to build ``final`` in.

    The name is unique per call, not per process: two builds in one process (an old queue
    stopping while a new one starts) must not empty each other's staging."""
    tmp = final.with_name(f"{final.name}.{uuid.uuid4().hex}.tmp")
    tmp.mkdir(parents=True)
    return tmp


def swap_into_place(
    tmp: Path, final: Path, *, complete: Callable[[Path], bool]
) -> None:
    """Move a finished staging directory to ``final``, under the build lock.

    ``final`` already whole (``complete(final)``): this build is redundant and is dropped. ``final``
    present but **not** whole — a clear that could not finish, a mask build that recreated the
    folder after a clear — is removed first; treating it as built would block that part forever.
    A removal that fails (a file in use on Windows) raises, so the build is reported as failed
    rather than recorded as done.
    """
    if final.exists():
        if complete(final):
            shutil.rmtree(tmp, ignore_errors=True)
            return
        shutil.rmtree(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    atomic.replace(tmp, final)


def sweep_stale_staging() -> int:
    """Remove what interrupted builds left behind; returns how many entries went.

    Call under :func:`build_lock`. Staging directories are made only under the lock, so while
    it is held every ``*.tmp`` directory belongs to a build that died (a killed process, a
    power cut) and is removed whatever its age. Index temp files are written outside the lock,
    so only those older than :data:`STALE_INDEX_TMP_S` go.
    """
    root = cache_dir()
    removed = 0
    proxies = root / PROXIES_DIR
    if proxies.is_dir():
        candidates = list(proxies.glob("*.tmp")) + list(proxies.glob("*/masks/*.tmp"))
        for stale in candidates:
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
                removed += 1
    now = time.time()
    for stale in root.glob(f"{INDEX_FILE}.*.tmp") if root.is_dir() else []:
        with contextlib.suppress(OSError):
            if now - stale.stat().st_mtime > STALE_INDEX_TMP_S:
                stale.unlink()
                removed += 1
    return removed


def discard(tmp: Path) -> None:
    shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# frames.zarr — the proxy's frame map
# --------------------------------------------------------------------------- #
def write_frames(folder: Path, src_frame: np.ndarray, attrs: dict) -> None:
    """Write ``frames.zarr`` into a staging ``folder``: ``src_frame`` and the proxy's facts."""
    root = zarr_store.open_group(folder / FRAMES_STORE, mode="w")
    root.attrs["schema_version"] = SCHEMA_VERSION
    root.attrs.update(attrs)
    values = np.asarray(src_frame, dtype=np.int64)
    arr = root.create_array(
        "src_frame", shape=(0,), chunks=(FRAME_CHUNK,), dtype="int64"
    )
    for start in range(0, len(values), FRAME_CHUNK):
        _append(arr, values[start : start + FRAME_CHUNK])


@dataclass(frozen=True)
class ProxyFrames:
    """A built proxy's frame map and facts, read whole (it is small)."""

    key: str
    src_frame: np.ndarray
    width: int
    height: int
    source_width: int
    source_height: int
    codec: str

    @property
    def video(self) -> Path:
        return proxy_dir(self.key) / PROXY_FILE


def read_frames(key: str) -> ProxyFrames:
    """The frame map of a built proxy.

    Raises:
        FileNotFoundError: If the proxy has not been built on this machine.
    """
    if not proxy_present(key):
        raise FileNotFoundError(f"no preview built for {key} in {cache_dir()}")
    root = zarr_store.open_group(proxy_dir(key) / FRAMES_STORE, mode="r")
    attrs = dict(root.attrs)
    return ProxyFrames(
        key=key,
        src_frame=np.asarray(root["src_frame"][:], dtype=np.int64),
        width=int(attrs["width"]),
        height=int(attrs["height"]),
        source_width=int(attrs["source_width"]),
        source_height=int(attrs["source_height"]),
        codec=str(attrs.get("codec", "")),
    )


# --------------------------------------------------------------------------- #
# masks/<trk>-<digest>.zarr — one row per (proxy frame, animal)
# --------------------------------------------------------------------------- #
class MaskWriter:
    """Buffered writer for a mask store (Tenet 12): rows sorted by proxy frame, then animal.

    Columns: ``proxy_frame`` int64, ``object_id`` int32, ``bbox`` float32 ``(x0, y0, x1, y1)``
    normalized like the detection zarr's, ``mask_rle`` str — ASCII COCO-RLE counts at the
    proxy's ``(height, width)``.
    """

    def __init__(self, path: Path, attrs: dict) -> None:
        self._root = zarr_store.open_group(path, mode="w")
        self._root.attrs["schema_version"] = SCHEMA_VERSION
        self._root.attrs.update(attrs)
        create = self._root.create_array
        self._cols = {
            "proxy_frame": create(
                "proxy_frame", shape=(0,), chunks=(MASK_ROW_CHUNK,), dtype="int64"
            ),
            "object_id": create(
                "object_id", shape=(0,), chunks=(MASK_ROW_CHUNK,), dtype="int32"
            ),
            "bbox": create(
                "bbox", shape=(0, 4), chunks=(MASK_ROW_CHUNK, 4), dtype="float32"
            ),
            "mask_rle": create(
                "mask_rle", shape=(0,), chunks=(MASK_RLE_CHUNK,), dtype="str"
            ),
        }
        self._buf: dict[str, list] = {name: [] for name in self._cols}

    def add(self, proxy_frame: int, object_id: int, bbox, counts: str) -> None:
        self._buf["proxy_frame"].append(int(proxy_frame))
        self._buf["object_id"].append(int(object_id))
        self._buf["bbox"].append([float(v) for v in bbox])
        self._buf["mask_rle"].append(str(counts))
        if len(self._buf["proxy_frame"]) >= FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self._buf["proxy_frame"]:
            return
        _append(
            self._cols["proxy_frame"], np.asarray(self._buf["proxy_frame"], np.int64)
        )
        _append(self._cols["object_id"], np.asarray(self._buf["object_id"], np.int32))
        _append(
            self._cols["bbox"],
            np.asarray(self._buf["bbox"], np.float32).reshape(-1, 4),
        )
        _append(self._cols["mask_rle"], np.asarray(self._buf["mask_rle"], dtype=object))
        self._buf = {name: [] for name in self._cols}

    def close(self) -> None:
        self.flush()


@dataclass(frozen=True)
class MaskRows:
    """A mask store read whole: small (megabytes per hour), and read once per surface."""

    proxy_frame: np.ndarray
    object_id: np.ndarray
    bbox: np.ndarray
    mask_rle: np.ndarray
    width: int
    height: int
    attrs: dict

    def rows_for(self, object_id: int, lo: int, hi: int) -> np.ndarray:
        """Row positions of one animal on proxy frames ``lo..hi`` inclusive."""
        start = int(np.searchsorted(self.proxy_frame, int(lo), side="left"))
        stop = int(np.searchsorted(self.proxy_frame, int(hi), side="right"))
        rows = np.arange(start, stop)
        return rows[self.object_id[start:stop] == subject_key(object_id)]


def read_masks(path: Path) -> MaskRows:
    """A mask store's rows.

    Raises:
        FileNotFoundError: If the store has not been built.
    """
    if not masks_complete(path):
        raise FileNotFoundError(f"no preview masks at {path}")
    root = zarr_store.open_group(path, mode="r")
    attrs = dict(root.attrs)
    return MaskRows(
        proxy_frame=np.asarray(root["proxy_frame"][:], dtype=np.int64),
        object_id=np.asarray(root["object_id"][:], dtype=np.int32),
        bbox=np.asarray(root["bbox"][:], dtype=np.float32).reshape(-1, 4),
        mask_rle=np.asarray(root["mask_rle"][:]),
        width=int(attrs["width"]),
        height=int(attrs["height"]),
        attrs=attrs,
    )


# --------------------------------------------------------------------------- #
# size / clear
# --------------------------------------------------------------------------- #
def cache_size() -> int:
    """Bytes on disk under the preview cache (0 when it does not exist)."""
    root = cache_dir()
    total = 0
    if not root.is_dir():
        return 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue  # removed while walking (a concurrent clear or swap)
    return total


def clear_cache() -> int:
    """Delete every preview on this machine; returns the bytes freed.

    Removes only what this module writes (``proxies/`` and ``index.json``), never the
    configured folder itself — a user may have pointed it at a folder holding other things.
    Runs under :func:`build_lock`, so it waits for a build in flight to stop first.
    """
    root = cache_dir()
    with (
        build_lock()
    ):  # waits for a build in flight to stop: never delete under a writer
        before = cache_size()
        index = root / INDEX_FILE
        if index.is_file():
            index.unlink()
        if (root / PROXIES_DIR).exists():
            try:
                shutil.rmtree(root / PROXIES_DIR)
            except OSError as exc:
                raise OSError(
                    f"some previews could not be deleted (a file still in use?): {exc}. "
                    "What is left is rebuilt when next needed."
                ) from exc
        return before - cache_size()


def _append(arr, values: np.ndarray) -> None:
    n = arr.shape[0]
    arr.resize((n + len(values), *arr.shape[1:]))
    arr[n:] = values
