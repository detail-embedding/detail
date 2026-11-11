"""A cluster's clips as montage tiles from the preview cache (``api/clip_tiles.py``, §22).

Built on ``test_previews``'s tracked recording — a real VFR part, its timestamp map, and a tracking
run whose masks are built into the (isolated) preview cache — so every tile asserted on was decoded
from a proxy and drawn.

**Recorded class: work done on the way to an answer, thrown away, then asked for.** Reaching a
tile frame behind the decoder costs a seek to a keyframe and a decode of every frame up to it;
those frames were dropped and each was then asked for on a later tick, at a seek of its own.
Every picture was correct, so nothing failed: a node's tiles were slow to start playing,
and much slower on a busy machine. It hid because the tests asked
for a clip's frames in ascending order, the one order that never seeks. The rule: a tile source
that decodes on the way to a frame keeps what it decodes. The tests under *a short clip is drawn
whole* ask out of order, the way a playing montage does, and count seeks and decodes, never time.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from detail.api import clip_tiles, previews
from detail.api.clip_tiles import TileLook
from detail.storage import locations
from tests.api.test_previews import _timestamps, _tracked_recording


class _Experiment:
    """What the tile source asks of an experiment: a recording by id."""

    def __init__(self, recording):
        self._recording = recording

    def get_recording(self, recording_id):
        assert recording_id == self._recording.id
        return self._recording


def _tiles(recording, spans, tracking="track_01"):
    catalog = clip_tiles.TileCatalog(_Experiment(recording), {recording.id: tracking})
    clips = [clip_tiles.TileClip(recording.id, 0, a, b) for a, b in spans]
    return clip_tiles.ClipTiles(catalog, clips)


def test_a_tile_is_the_clip_at_its_offset(processed_recording, write_vfr_video):
    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    t = _timestamps(recording)
    tiles = _tiles(recording, [(int(t[10]), int(t[40]))])
    first = tiles.frame(0, 0, 64, True, look=TileLook(masks=True))
    assert first.shape == (64, 64, 3) and first.dtype == np.uint8
    reader = previews.PreviewReader.open(recording, 0, "track_01")
    with reader.clip(0, int(t[10]), int(t[40]), 64) as clip:
        offset = int(t[30]) - int(t[10])
        index = int(np.searchsorted(clip.t_ns, int(t[30]), side="right")) - 1
        assert np.array_equal(
            tiles.frame(0, offset, 64, True, look=TileLook(masks=True)),
            clip.frame_at(index),
        )
    tiles.close()


def test_drawing_tiles_parses_the_machine_config_once(
    processed_recording, write_vfr_video, monkeypatch
):
    """The preview cache is found the way a user's app finds it: through storage.yaml.

    Every other test here names the cache with ``DETAIL_PREVIEW_DIR``, which returns before
    the file is read. A user sets no such variable, and each tile picture then parsed the
    whole file twice (``tests/storage/test_machine_config_reads.py`` has the write-up). A
    count, not a time.
    """
    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    data = locations.read_storage()
    data.setdefault("storage", {}).setdefault("default", {})[
        "preview_cache"
    ] = previews.preview_cache_dir().as_posix()
    locations.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    locations.STORAGE_PATH.write_text(yaml.dump(data), encoding="utf-8")
    monkeypatch.delenv(locations.PREVIEW_DIR_ENV)
    parses = []
    real = locations._parse_storage
    monkeypatch.setattr(locations, "_parse_storage", lambda: parses.append(1) or real())

    t = _timestamps(recording)
    tiles = _tiles(recording, [(int(t[0]), int(t[20])), (int(t[20]), int(t[40]))])
    for clip, start in ((0, 0), (1, 20)):
        for i in range(start, start + 20, 2):
            offset = int(t[i]) - int(t[start])
            picture = tiles.frame(clip, offset, 48, True, look=TileLook(masks=True))
            assert picture.shape == (48, 48, 3)
    tiles.close()
    assert (
        len(parses) == 1
    ), f"{len(parses)} parses of storage.yaml for 20 tile pictures"


def test_a_clip_outside_the_filters_draws_its_animal_grey(
    processed_recording, write_vfr_video
):
    """Still playing, still outlined — in grey, not in the mask colour."""
    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    t = _timestamps(recording)
    tiles = _tiles(recording, [(int(t[10]), int(t[40]))])
    tint = np.array(previews.visualize.MASK_TINT)
    grey = np.array(clip_tiles.GREY)

    def near(tile, colour):
        return (np.abs(tile.astype(int) - colour).sum(axis=2) < 40).any()

    matching = tiles.frame(0, 0, 96, True, look=TileLook(masks=True))
    missing = tiles.frame(0, 0, 96, False, look=TileLook(masks=True))
    assert near(matching, tint) and not near(missing, tint)
    assert near(missing, grey)
    tiles.close()


def test_without_masks_a_tile_is_the_plain_footage_whatever_it_matches(
    processed_recording, write_vfr_video
):
    """The montage's Masks toggle off: no mask, no outline, and ``matching`` changes nothing.
    The drawn-tile cache is keyed on the flag, so asking again with masks on is not served the
    plain picture."""
    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    t = _timestamps(recording)
    tiles = _tiles(recording, [(int(t[10]), int(t[40]))])
    masked = tiles.frame(0, 0, 96, True, look=TileLook(masks=True))
    plain = tiles.frame(0, 0, 96, True, look=TileLook(masks=False))
    reader = previews.PreviewReader.open(recording, 0, "track_01")
    with reader.clip(0, int(t[10]), int(t[40]), 96, mask=False) as clip:
        assert np.array_equal(plain, clip.frame_at(0))
    assert not np.array_equal(plain, masked), "the fixture must draw a mask to remove"
    assert np.array_equal(
        tiles.frame(0, 0, 96, False, look=TileLook(masks=False)), plain
    )
    assert np.array_equal(
        tiles.frame(0, 0, 96, True, look=TileLook(masks=True)), masked
    )
    tiles.close()


def test_open_clips_are_capped(processed_recording, write_vfr_video, monkeypatch):
    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    monkeypatch.setattr(clip_tiles, "OPEN_CLIPS", 2)
    t = _timestamps(recording)
    tiles = _tiles(recording, [(int(t[i]), int(t[i + 5])) for i in range(0, 30, 5)])
    for index in range(len(tiles)):
        tiles.frame(index, 0, 48, True, look=TileLook(masks=True))
    assert len(tiles._open) == 2
    tiles.close()
    assert not tiles._open


def test_missing_says_proxy_then_masks_then_nothing(
    processed_recording, write_vfr_video
):
    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    experiment = _Experiment(recording)
    tracking = {recording.id: "track_01"}

    def missing():
        return clip_tiles.missing(experiment, tracking, [recording.id])

    assert missing() == {"proxy": [recording.id], "masks": []}
    recording.build_preview(masks=False)
    # The footage can play now; only the outline waits for the masks.
    assert missing() == {"proxy": [], "masks": [recording.id]}
    t = _timestamps(recording)
    tile = _tiles(recording, [(int(t[0]), int(t[5]))]).frame(
        0, 0, 32, True, look=TileLook(masks=True)
    )
    assert tile.shape == (32, 32, 3)
    recording.build_preview()
    assert missing() == {"proxy": [], "masks": []}


def test_tracking_runs_come_from_each_recordings_embedding_run(tmp_path):
    from detail.storage import experiment as experiment_store

    for rid, tracking in (("recording_001", "track_02"), ("recording_002", None)):
        folder = (
            experiment_store.recording_path(tmp_path, rid) / "embeddings" / "emb_01"
        )
        folder.mkdir(parents=True)
        meta = {"schema_version": 1, "tracking_run_id": tracking}
        (folder / "embedding.yaml").write_text(
            yaml.safe_dump(meta, sort_keys=False), encoding="utf-8"
        )
    found = clip_tiles.tracking_runs(
        tmp_path,
        {
            "recording_001": "emb_01",
            "recording_002": "emb_01",
            "recording_003": "emb_gone",
        },
    )
    assert found == {"recording_001": "track_02"}


def test_a_looping_tile_is_drawn_once_per_frame(
    processed_recording, write_vfr_video, monkeypatch
):
    """A second pass of the loop is served from what the first drew: no seek back, no decode."""
    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    t = _timestamps(recording)
    tiles = _tiles(recording, [(int(t[0]), int(t[20]))])
    first = [
        tiles.frame(0, int(t[i]) - int(t[0]), 48, True, look=TileLook(masks=True))
        for i in range(0, 20, 2)
    ]
    drawn = []
    real = previews.PreviewClip.frame_at
    monkeypatch.setattr(
        previews.PreviewClip,
        "frame_at",
        lambda self, offset: drawn.append(offset) or real(self, offset),
    )
    again = [
        tiles.frame(0, int(t[i]) - int(t[0]), 48, True, look=TileLook(masks=True))
        for i in range(0, 20, 2)
    ]
    assert drawn == [], "the second pass decoded again"
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    tiles.frame(
        0, 0, 48, False, look=TileLook(masks=True)
    )  # another colour is another picture
    assert drawn == [0]
    tiles.close()


def test_the_drawn_tiles_stay_under_their_budget(
    processed_recording, write_vfr_video, monkeypatch
):
    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    t = _timestamps(recording)
    tile_bytes = 32 * 32 * 3
    monkeypatch.setattr(clip_tiles, "CACHE_BYTES", 5 * tile_bytes)
    tiles = _tiles(recording, [(int(t[i]), int(t[i + 9])) for i in range(0, 40, 10)])
    for index in range(len(tiles)):
        for k in range(3):
            tiles.frame(
                index, int(t[k + 1]) - int(t[0]), 32, True, look=TileLook(masks=True)
            )
    assert tiles._bytes <= 5 * tile_bytes
    assert tiles._bytes == sum(
        f.nbytes for d in tiles._drawn.values() for f in d[-1].values()
    )
    tiles.close()


def test_montages_open_at_once_share_one_budget(
    processed_recording, write_vfr_video, monkeypatch
):
    """Review 2026-11-09: INTERPRET and five cluster windows each kept their own 192 MB of drawn
    tiles and 160 open clips. Every source that is drawing shares one pool now; one alone keeps
    the full budget, and a closed one gives its share back."""
    import weakref

    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    t = _timestamps(recording)
    tile_bytes = 32 * 32 * 3
    monkeypatch.setattr(clip_tiles, "_DRAWING", weakref.WeakSet())
    monkeypatch.setattr(clip_tiles, "CACHE_BYTES", 12 * tile_bytes)
    monkeypatch.setattr(clip_tiles, "CACHE_POOL_BYTES", 12 * tile_bytes)
    monkeypatch.setattr(clip_tiles, "CACHE_FLOOR_BYTES", 2 * tile_bytes)
    monkeypatch.setattr(clip_tiles, "OPEN_CLIPS", 4)
    monkeypatch.setattr(clip_tiles, "OPEN_POOL", 4)
    monkeypatch.setattr(clip_tiles, "OPEN_FLOOR", 1)
    spans = [(int(t[i]), int(t[i + 9])) for i in range(0, 40, 10)]
    assert clip_tiles.share() == (
        12 * tile_bytes,
        4,
    ), "nothing drawing: the full budget"

    def draw(tiles) -> None:
        for index in range(len(tiles)):
            for k in range(3):
                tiles.frame(index, int(t[k + 1]) - int(t[0]), 32, True, look=TileLook())

    montages = [_tiles(recording, spans) for _ in range(3)]
    for tiles in montages:
        draw(tiles)
    assert clip_tiles.share() == (4 * tile_bytes, 1)
    # The first two drew before the third joined; each comes down to its share at the next
    # picture it is asked for — a playing montage asks every tick.
    for tiles in montages[:2]:
        draw(tiles)
    for tiles in montages:
        assert tiles._bytes <= 4 * tile_bytes and len(tiles._open) <= 1
    montages[1].close()
    montages[2].close()
    assert clip_tiles.share() == (12 * tile_bytes, 4), "a closed one gives it back"
    montages[0].close()
    assert clip_tiles.share() == (12 * tile_bytes, 4)


# --- a short clip is drawn whole, in one forward pass ----------------------------------- #
# The class: **work done on the way to an answer, thrown away, then asked for.** A tile's clock
# keeps running while the montage's one worker serves the other tiles, so the frame a tile asks
# for next is rarely the one after its last; in a clip of half a second it is usually behind
# the decoder. Reaching it costs a seek to a keyframe and a decode of every frame up to it, and
# the frames passed were dropped — each of them then asked for on a later tick, at a seek of
# its own. Nothing was wrong at any line: every picture was correct, and on a quiet machine
# a node's tiles played soon enough. On a busy machine each seek costs more, and the same
# view filled in a picture per tile every few seconds, in tile order.
#
# How it hid: the looping test above asks for a clip's frames in ascending order, which is the
# one order that never seeks. The tests below ask the way a playing montage does — out of order
# — and count **seeks and decodes, never time**.
#
# The rule: a tile source that decodes on the way to a frame keeps what it decodes.


def _spied(monkeypatch):
    """``(drawn, seeks)``: every tile frame drawn, and every seek the proxy decoder makes."""
    from detail.storage import video as video_store

    drawn, seeks = [], []
    real_frame = previews.PreviewClip.frame_at
    real_seek = video_store.FrameCursor._seek_and_walk

    def frame_at(self, offset):
        drawn.append(int(offset))
        return real_frame(self, offset)

    def seek_and_walk(self, frame_idx):
        seeks.append(int(frame_idx))
        return real_seek(self, frame_idx)

    monkeypatch.setattr(previews.PreviewClip, "frame_at", frame_at)
    monkeypatch.setattr(video_store.FrameCursor, "_seek_and_walk", seek_and_walk)
    return drawn, seeks


def _clip_of(recording, n_frames: int, tile_px: int = 48):
    """``(start_ns, t_ns, tiles)`` of a clip of exactly ``n_frames`` tile frames (the proxy
    keeps 15 a second, so a span of source frames is not a count of them), with its tiles
    drawn by a clip of its own, in order."""
    t = _timestamps(recording)
    reader = previews.PreviewReader.open(recording, 0, "track_01")
    with reader.clip(0, int(t[0]), int(t[-1]), tile_px) as whole:
        proxy_t = np.array(whole.t_ns)
    assert (
        len(proxy_t) >= n_frames + 2
    ), "the fixture's video is too short for this clip"
    start, end = int(proxy_t[2]), int(proxy_t[2 + n_frames - 1])
    with reader.clip(0, start, end, tile_px) as clip:
        t_ns, tiles = np.array(clip.t_ns), list(clip)
    assert len(t_ns) == n_frames, "the fixture must give the clip the frames asked for"
    return start, end, t_ns, tiles


def _playing_order(n: int) -> list[int]:
    """Every frame of an ``n``-frame clip, in an order that goes back as often as forward —
    what a tile asks for when a pass over the other tiles takes longer than one of its frames.
    """
    order = [(k * 7 + 5) % n for k in range(n)] if n % 7 else list(range(n))[::-1]
    assert sorted(order) == list(range(n)), "the order must reach every frame"
    assert any(b < a for a, b in zip(order, order[1:])), "the order must go back"
    return order


@pytest.mark.parametrize("n_frames", [2, 3, 8, 15])
def test_a_short_clip_is_drawn_whole_on_its_second_ask(
    processed_recording, write_vfr_video, monkeypatch, n_frames
):
    """Its first picture is the one frame asked for; the next frame it lacks draws every frame
    it lacks, in order — two seeks for the clip however it is asked, and nothing drawn twice.
    """
    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    start, end, t_ns, reference = _clip_of(recording, n_frames)
    order = _playing_order(n_frames)
    tiles = _tiles(recording, [(start, end)])
    drawn, seeks = _spied(monkeypatch)
    look = TileLook(masks=True)

    first = tiles.frame(0, int(t_ns[order[0]]) - start, 48, True, look=look)
    assert drawn == [order[0]], "the first picture draws the one frame asked for"
    assert np.array_equal(first, reference[order[0]])

    second = tiles.frame(0, int(t_ns[order[1]]) - start, 48, True, look=look)
    rest = [i for i in range(n_frames) if i != order[0]]
    assert drawn == [order[0], *rest], "the second ask draws the rest, in order, once"
    assert np.array_equal(second, reference[order[1]])

    for _loop in range(2):
        for i in order:
            picture = tiles.frame(0, int(t_ns[i]) - start, 48, True, look=look)
            assert np.array_equal(picture, reference[i]), f"frame {i} is not frame {i}"
    assert len(drawn) == n_frames, "a frame was decoded again"
    assert len(seeks) <= 2, f"{len(seeks)} seeks for a clip of {n_frames} frames"
    assert tiles._bytes == n_frames * 48 * 48 * 3
    tiles.close()


def test_asked_frame_by_frame_the_same_clip_would_seek_on_every_step_back(
    processed_recording, write_vfr_video, monkeypatch
):
    """The negative case: with the whole-clip pass switched off the counts above are not met,
    so they measure the pass and not the fixture."""
    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    start, end, t_ns, _reference = _clip_of(recording, 15)
    order = _playing_order(15)
    monkeypatch.setattr(clip_tiles, "WHOLE_CLIP_FRAMES", 0)
    tiles = _tiles(recording, [(start, end)])
    drawn, seeks = _spied(monkeypatch)
    for i in order:
        tiles.frame(0, int(t_ns[i]) - start, 48, True, look=TileLook(masks=True))
    assert drawn == order, "frame by frame draws exactly what is asked"
    steps_back = sum(b < a for a, b in zip(order, order[1:]))
    assert len(seeks) == 1 + steps_back and len(seeks) > 2
    tiles.close()


@pytest.mark.parametrize("limit, whole", [(8, True), (7, False)])
def test_a_clip_longer_than_the_limit_is_drawn_frame_by_frame(
    processed_recording, write_vfr_video, monkeypatch, limit, whole
):
    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    start, end, t_ns, _reference = _clip_of(recording, 8)
    monkeypatch.setattr(clip_tiles, "WHOLE_CLIP_FRAMES", limit)
    tiles = _tiles(recording, [(start, end)])
    drawn, _seeks = _spied(monkeypatch)
    for i in (5, 2):
        tiles.frame(0, int(t_ns[i]) - start, 48, True, look=TileLook(masks=True))
    assert len(drawn) == (8 if whole else 2)
    tiles.close()


@pytest.mark.parametrize("room, whole", [(8, True), (7, False)])
def test_a_clip_that_does_not_fit_the_budget_is_drawn_frame_by_frame(
    processed_recording, write_vfr_video, monkeypatch, room, whole
):
    """Drawn whole it would push out the tiles beside it, or itself: the pass is for a clip
    whose every frame this source can keep."""
    import weakref

    _built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    start, end, t_ns, _reference = _clip_of(recording, 8)
    tile_bytes = 48 * 48 * 3
    monkeypatch.setattr(clip_tiles, "_DRAWING", weakref.WeakSet())
    monkeypatch.setattr(clip_tiles, "CACHE_BYTES", room * tile_bytes)
    monkeypatch.setattr(clip_tiles, "CACHE_POOL_BYTES", room * tile_bytes)
    monkeypatch.setattr(clip_tiles, "CACHE_FLOOR_BYTES", tile_bytes)
    tiles = _tiles(recording, [(start, end)])
    drawn, _seeks = _spied(monkeypatch)
    for i in (5, 2):
        tiles.frame(0, int(t_ns[i]) - start, 48, True, look=TileLook(masks=True))
    assert len(drawn) == (8 if whole else 2)
    assert tiles._bytes <= room * tile_bytes
    tiles.close()
