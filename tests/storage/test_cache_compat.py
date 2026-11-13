"""Derived data a user already has must stay valid across an update, unless invalidating it is
deliberate (CLAUDE.md, *Expensive derived data*; DESIGN_PLAN §3 *Video preview cache*,
*Compatibility contract*).

**The class: a change that silently makes every user rebuild an expensive cache.** The video
preview cache is keyed by a formula, not by a version: ``<key>`` is ``proxy_key`` of
``integrity.file_fingerprint`` (size + a hash of the file's first and last MiB), a mask store is
``<tracking run>-<masks_digest>``, and both live at fixed paths under ``preview_cache_dir()``.
Change any input to those (the hashed chunk size, a field in the digest, a folder name, the
default location) and every proxy on every machine stops being found. Nothing fails: the app
sees "missing", queues a build, and a user who updated the app watches hours of rebuilding,
per computer.

The opposite direction hides just as well. Nothing in a key records *how* a proxy was made, so a
change to the frame rate, the size box, the frame selection or the codec order is **not**
applied to proxies already built: old and new proxies mix, and the read side (which assumes the
current constants) draws the old ones wrongly. A parameter change therefore needs the same
deliberate procedure as a key change.

**Why a pin rather than a behavioural test.** Every behavioural test builds its cache fresh, so
it passes under any key formula and any parameters. The only thing that can see this class is a
test holding the values a user's disk already holds. The pins below are those values for fixed
inputs; a failure names the procedure rather than the line.

**When a pin fails.** If the change is intended, follow the procedure in DESIGN_PLAN §3
*Compatibility contract* (a format version, rebuild only the store that changed, a one-time
message, a release note), then update the pinned value in the same commit. Never update a pin
alone.

The node cache (``<run>/node_cache.zarr``, §22) is pinned too, with a gentler message: it is
rebuilt in seconds from the fit, and ``RULES`` exists to invalidate it, but a bump still has to
be a decision, with the banner and **Rebuild clips** that tell the user why.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from detail.api import previews as api
from detail.core import video_codecs
from detail.storage import integrity
from detail.storage import locations
from detail.storage import node_cache
from detail.storage import previews as store
from detail.storage import video
from detail.storage import zarr_store

PREVIEW_REBUILD = (
    "This change would make every user rebuild their video previews (about ten minutes per "
    "hour of video, per computer: hours on a big experiment), or would mix old and new "
    "previews without rebuilding. See CLAUDE.md 'Expensive derived data' and DESIGN_PLAN §3 "
    "'Compatibility contract'. If it is intended, follow that procedure, then update the "
    "pinned value here in the same commit."
)
NODE_CACHE_REBUILD = (
    "This change marks every existing node cache stale, so each clustering run's INTERPRET "
    "tab asks its user to Rebuild clips (seconds per run). That is allowed, but it must be "
    "deliberate: see DESIGN_PLAN §3 'Compatibility contract', record the bump in "
    "storage/node_cache.py's RULES list, then update the pinned value here."
)


# --------------------------------------------------------------------------- #
# pure helpers, so the negative case can hand them a literal
# --------------------------------------------------------------------------- #
def _sample_bytes(n: int) -> bytes:
    """Fixed, non-repeating bytes: the stand-in for a video file's content."""
    return (np.arange(n, dtype=np.uint64) * 2654435761 % 251).astype(np.uint8).tobytes()


def _differences(actual: dict, pinned: dict) -> list[str]:
    """Each pinned name whose current value differs, as ``name: pinned -> now``."""
    return [
        f"{name}: pinned {pinned[name]!r}, now {actual.get(name)!r}"
        for name in pinned
        if actual.get(name) != pinned[name]
    ]


def _assert_pinned(actual: dict, pinned: dict, message: str) -> None:
    changed = _differences(actual, pinned)
    assert not changed, message + "\n  " + "\n  ".join(changed)


# --------------------------------------------------------------------------- #
# the proxy key: what finds a built proxy
# --------------------------------------------------------------------------- #
#: ``proxy_key(file_fingerprint(path))`` for ``_sample_bytes(n)`` written to a file.
PINNED_KEYS = {
    1_000: "1000-3c11c677638f16e61937",  # smaller than one hashed chunk
    3 * (1 << 20) + 17: "3145745-ffb8c7eeb6f42a8a46d3",  # head and tail chunks
}


def _keys_for(tmp_path: pathlib.Path) -> dict:
    keys = {}
    for size in PINNED_KEYS:
        path = tmp_path / f"part_{size}.mp4"
        path.write_bytes(_sample_bytes(size))
        keys[size] = store.proxy_key(integrity.file_fingerprint(path))
    return keys


def test_the_proxy_key_of_a_file_is_unchanged(tmp_path):
    _assert_pinned(_keys_for(tmp_path), PINNED_KEYS, PREVIEW_REBUILD)


def test_the_proxy_key_ignores_the_middle_of_a_file(tmp_path):
    """Size + head/tail hash: a re-copy keeps its proxy, and the key never reads the whole
    file. A full-content hash would also change every key, and read gigabytes per part.
    """
    size = 3 * (1 << 20) + 17
    content = bytearray(_sample_bytes(size))
    content[size // 2] ^= 0xFF
    path = tmp_path / "middle_changed.mp4"
    path.write_bytes(bytes(content))
    key = store.proxy_key(integrity.file_fingerprint(path))
    assert key == PINNED_KEYS[size], PREVIEW_REBUILD


def test_the_key_check_would_catch_a_changed_fingerprint(tmp_path, monkeypatch):
    """Negative case: hash half the chunk, and the pin fails with the rebuild message."""
    monkeypatch.setattr(integrity, "_CHUNK", 1 << 19)
    with pytest.raises(AssertionError, match="rebuild their video previews"):
        _assert_pinned(_keys_for(tmp_path), PINNED_KEYS, PREVIEW_REBUILD)


# --------------------------------------------------------------------------- #
# the masks digest: what finds a built mask store
# --------------------------------------------------------------------------- #
_TRACKING = {
    "id": "trk_01",
    "created_at": "2026-10-01T12:00:00",
    "every_n_frames": 2,
    "detection_run_id": "det_01",
    "status": "complete",
    "tracker": "anything else in the entry is not part of the digest",
}
_DETECTION = {
    "id": "det_01",
    "created_at": "2026-09-30T08:00:00",
    "status": "complete",
    "model": "nor is this",
}
PINNED_DIGESTS = {
    "full entries": "0f825c6d9a12",
    "ids only": "427f0b2a5f20",  # missing fields hash as the string "None"
}


def _digests() -> dict:
    return {
        "full entries": store.masks_digest(_TRACKING, _DETECTION),
        "ids only": store.masks_digest({"id": "trk_02"}, {"id": "det_02"}),
    }


def test_the_masks_digest_of_a_run_is_unchanged():
    _assert_pinned(_digests(), PINNED_DIGESTS, PREVIEW_REBUILD)


def test_the_masks_digest_reads_only_its_five_fields():
    """A new field in a run record must not reach the digest by accident."""
    noisy_tracking = {**_TRACKING, "notes": "edited later", "status": "complete"}
    noisy_detection = {**_DETECTION, "notes": "edited later"}
    assert (
        store.masks_digest(noisy_tracking, noisy_detection)
        == PINNED_DIGESTS["full entries"]
    ), PREVIEW_REBUILD


# --------------------------------------------------------------------------- #
# how a proxy is made: not in its key, so a change mixes old and new
# --------------------------------------------------------------------------- #
PINNED_ENCODE = {
    "fps": 15,
    "max_wh": (480, 270),
    # The proxy goes through `storage/video.open_video_writer` (remote-detail's one writer):
    # H.264 through PyAV's linked libx264, else mp4v. It was cv2's `avc1` FourCC, which is the
    # same H.264 stream, so proxies already built stay valid and nothing is rebuilt.
    "preferred codec": "h264",
    "fallback codec": "mp4v",
    "size 1920x1080": (480, 270),
    "size 1280x1024": (338, 270),
    "size 640x480": (360, 270),
    "size 320x240": (320, 240),  # never enlarged
    "size 1001x777": (348, 270),  # even sides
    "size 2448x2048": (322, 270),
    # proxy_frame_selection over 600 frames: (frames kept, the first ten kept)
    "select 62.5 fps": (144, [0, 5, 9, 13, 17, 21, 25, 30, 34, 38]),
    "select 60 fps": (150, [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]),
    "select 30 fps": (300, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]),
    "select 10 fps": (600, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
}


def _encode() -> dict:
    found = {
        "fps": api.PROXY_FPS,
        "max_wh": tuple(api.PROXY_MAX_WH),
        "preferred codec": video_codecs.DEFAULT_VIDEO_CODEC,
        "fallback codec": next(
            c
            for c in video_codecs.VIDEO_CODECS
            if c != video_codecs.DEFAULT_VIDEO_CODEC
        ),
    }
    for name in PINNED_ENCODE:
        if name.startswith("size "):
            w, h = (int(v) for v in name.split()[1].split("x"))
            found[name] = api.proxy_size(w, h)
        elif name.startswith("select "):
            fps = float(name.split()[1])
            t_ns = np.round(np.arange(600) * 1e9 / fps).astype(np.int64)
            kept = api.proxy_frame_selection(t_ns)
            found[name] = (len(kept), kept[:10].tolist())
    return found


def test_the_proxy_encode_parameters_are_unchanged():
    _assert_pinned(_encode(), PINNED_ENCODE, PREVIEW_REBUILD)


def test_the_encode_check_would_catch_a_new_frame_rate(monkeypatch):
    """Negative case: 12 fps instead of 15 changes the rate and every selection."""
    monkeypatch.setattr(api, "PROXY_FPS", 12)
    changed = _differences(_encode(), PINNED_ENCODE)
    assert "fps: pinned 15, now 12" in changed
    # The three sources faster than 12 fps keep a different set of frames; a 10 fps one
    # keeps every frame either way.
    flagged = {line.split(":")[0] for line in changed}
    assert {"select 62.5 fps", "select 60 fps", "select 30 fps"} <= flagged
    assert "select 10 fps" not in flagged


# --------------------------------------------------------------------------- #
# where it lives and what the files are called
# --------------------------------------------------------------------------- #
PINNED_LAYOUT = {
    "default location": ".cache/detail/previews",
    "env var": "DETAIL_PREVIEW_DIR",
    "storage.yaml key": "preview_cache",
    "proxy folder": "proxies/1000-3c11c677638f16e61937",
    "proxy file": "proxies/1000-3c11c677638f16e61937/proxy.mp4",
    "frame map": "proxies/1000-3c11c677638f16e61937/frames.zarr",
    "mask store": "proxies/1000-3c11c677638f16e61937/masks/trk_01-0f825c6d9a12.zarr",
    "index": "index.json",
    "schema_version": 1,
}


def _layout(cache: pathlib.Path) -> dict:
    key = PINNED_KEYS[1_000]

    def rel(path: pathlib.Path) -> str:
        return path.relative_to(cache).as_posix()

    return {
        "default location": locations._DEFAULT_PREVIEW_CACHE.relative_to(
            pathlib.Path.home()
        ).as_posix(),
        "env var": locations.PREVIEW_DIR_ENV,
        "storage.yaml key": "preview_cache",
        "proxy folder": rel(store.proxy_dir(key)),
        "proxy file": rel(store.proxy_dir(key) / store.PROXY_FILE),
        "frame map": rel(store.proxy_dir(key) / store.FRAMES_STORE),
        "mask store": rel(
            store.masks_path(key, "trk_01", PINNED_DIGESTS["full entries"])
        ),
        "index": rel(cache / store.INDEX_FILE),
        "schema_version": store.SCHEMA_VERSION,
    }


def test_the_cache_layout_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv(locations.PREVIEW_DIR_ENV, str(tmp_path))
    _assert_pinned(_layout(tmp_path), PINNED_LAYOUT, PREVIEW_REBUILD)


def test_storage_yaml_names_the_pinned_key(tmp_path, monkeypatch):
    """The ``preview_cache`` key is read where storage.yaml is parsed, not at a constant."""
    monkeypatch.delenv(locations.PREVIEW_DIR_ENV, raising=False)
    target = tmp_path / "elsewhere"
    monkeypatch.setattr(
        locations,
        "read_storage",
        lambda: {"storage": {"default": {PINNED_LAYOUT["storage.yaml key"]: target}}},
    )
    assert locations.preview_cache_dir() == target, PREVIEW_REBUILD


# --------------------------------------------------------------------------- #
# the stores as written: names, dtypes, chunks, attrs
# --------------------------------------------------------------------------- #
PINNED_FRAMES = {
    "first attr": "schema_version",
    "attrs": [
        "schema_version",
        "width",
        "height",
        "source_width",
        "source_height",
        "fps",
        "codec",
        "n_source_frames",
    ],
    "src_frame": ("int64", (100_000,)),
}
PINNED_MASKS = {
    "first attr": "schema_version",
    "proxy_frame": ("int64", (10_000,)),
    "object_id": ("int32", (10_000,)),
    "bbox": ("float32", (10_000, 4)),
    "mask_rle": ("str", (1_000,)),
}


def _column(root, name: str) -> tuple[str, tuple]:
    arr = root[name]
    dtype = "str" if arr.dtype.kind in ("T", "U", "O") else str(arr.dtype)
    return dtype, tuple(arr.chunks)


def test_the_frame_map_store_is_unchanged(tmp_path):
    # The attrs _build_proxy writes, in its order.
    store.write_frames(
        tmp_path,
        np.arange(5, dtype=np.int64),
        {
            "width": 480,
            "height": 270,
            "source_width": 1920,
            "source_height": 1080,
            "fps": 15,
            "codec": "avc1",
            "n_source_frames": 20,
        },
    )
    root = zarr_store.open_group(tmp_path / store.FRAMES_STORE, mode="r")
    attrs = list(dict(root.attrs))
    found = {
        "first attr": attrs[0],
        "attrs": attrs,
        "src_frame": _column(root, "src_frame"),
    }
    _assert_pinned(found, PINNED_FRAMES, PREVIEW_REBUILD)


def test_the_mask_store_is_unchanged(tmp_path):
    path = tmp_path / "trk_01-0f825c6d9a12.zarr"
    writer = store.MaskWriter(path, {"tracking_run_id": "trk_01"})
    writer.add(0, 1, (0.1, 0.2, 0.3, 0.4), "abc")
    writer.close()
    root = zarr_store.open_group(path, mode="r")
    found = {"first attr": list(dict(root.attrs))[0]}
    for name in ("proxy_frame", "object_id", "bbox", "mask_rle"):
        found[name] = _column(root, name)
    _assert_pinned(found, PINNED_MASKS, PREVIEW_REBUILD)


def test_the_index_records_path_size_mtime_and_key(tmp_path, monkeypatch):
    """``index.json``: resolved path -> {size, mtime_ns, key}. Losing it costs a re-hash (two
    1 MiB reads per part), not a rebuild, but a reshaped entry that older builds misread would
    point a part at another part's proxy."""
    cache = tmp_path / "cache"
    monkeypatch.setenv(locations.PREVIEW_DIR_ENV, str(cache))
    part = tmp_path / "part.mp4"
    part.write_bytes(_sample_bytes(1_000))
    assert store.key_for_video(part) == PINNED_KEYS[1_000], PREVIEW_REBUILD
    index = json.loads((cache / store.INDEX_FILE).read_text(encoding="utf-8-sig"))
    stat = part.stat()
    assert index == {
        str(part.resolve()): {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "key": PINNED_KEYS[1_000],
        }
    }, PREVIEW_REBUILD


# --------------------------------------------------------------------------- #
# the node cache: cheap to rebuild, still a decision
# --------------------------------------------------------------------------- #
PINNED_NODE_CACHE = {
    "schema_version": 1,
    "rules": 2,
    "rules key": "rules",
    "built_from attr": "built_from",
    "dirname": "node_cache.zarr",
}


def _node_cache() -> dict:
    return {
        "schema_version": node_cache.SCHEMA_VERSION,
        "rules": node_cache.RULES,
        "rules key": node_cache.RULES_KEY,
        "built_from attr": node_cache.BUILT_FROM_ATTR,
        "dirname": node_cache.DIRNAME,
    }


def test_the_node_cache_version_is_unchanged():
    _assert_pinned(_node_cache(), PINNED_NODE_CACHE, NODE_CACHE_REBUILD)


def test_the_node_cache_check_would_catch_a_rules_bump(monkeypatch):
    monkeypatch.setattr(node_cache, "RULES", 3)
    with pytest.raises(AssertionError, match="Rebuild clips"):
        _assert_pinned(_node_cache(), PINNED_NODE_CACHE, NODE_CACHE_REBUILD)
