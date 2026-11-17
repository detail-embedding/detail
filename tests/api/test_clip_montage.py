"""The clip montage (``api.clip_montage``): a cluster's top clips, cropped, labelled, playing on
one clock.

**What is pinned, and why each.**

- **One clock, and each clip's own loop.** Every clip starts at once and a shorter one starts
  again the moment it ends (its last frame shown for one step), so no tile is ever still.
  Asserted as the frames each clip shows at each step — not that "a picture came back", which a
  montage stuck on frame 0 also passes.
- **The ticks.** Raw on draws the footage; Raw off draws on white; a trail without what it trails
  is refused. Asserted on pixels a tick must change. The defaults are the mask and its trail.
- **The mask on the footage is an outline**, so the animal's own pixels are untouched — asserted
  on the pixels inside the animal, which a tint changes and an outline does not.
- **The labels**, with Blind mode's letters; **which clips** (count, order).
- **The exports**, read back: a GIF's width and rate and an mp4's rate are what was asked; one GIF
  per clip, named for it; a CSV list that is a clip-query CSV with the montage's columns after.
- **Order and teardown**: drawing before ``prepare`` is refused rather than waited on (the port
  this replaced deadlocked there), and ``close`` drops the decode contexts and the reads.
"""

from __future__ import annotations

import math
import threading

import cv2
import numpy as np
import pytest
from PIL import Image

from detail.api import clip_montage as cm
from detail.api.animation import sample_offsets_ns
from detail.api.clip_views import ClipViewSettings
from tests.clip_fixtures import count_decodes, walking

STEP = int(1e9 / 30)


def _pass(montage) -> list[int]:
    """The montage times one pass samples at its own clock — what an mp4 export draws."""
    return sample_offsets_ns(montage.duration_ns, montage.fps)


def _clip(recording, rank, first, last, name="M01_day1"):
    return cm.MontageClip(
        rank=rank,
        recording_id=recording.id,
        recording_name=name,
        subject=0,
        embed_run_id="emb_01",
        start_t_ns=first * STEP,
        end_t_ns=last * STEP,
    )


def _montage(tmp_path, spans=((0, 4), (1, 2)), **kwargs):
    recording = walking(tmp_path)
    clips = [_clip(recording, i + 1, a, b) for i, (a, b) in enumerate(spans)]
    montage = cm.ClusterMontage(recording.dir.parent.parent, clips, **kwargs)
    montage.prepare()
    return montage


# --- the clock --------------------------------------------------------------------------- #
def test_a_shorter_clip_loops_from_its_start_until_the_longest_ends(tmp_path):
    """The short clip (frames 1..3, a 100 ms loop: 67 ms long plus one 33 ms step) sampled at
    0, 67 and 133 ms shows frames 1, 3 and — 133 mod 100 = 33 ms in — 2: it has started
    again rather than held its last frame."""
    montage = _montage(tmp_path, spans=((0, 4), (1, 3)))
    assert montage.fps == 15.0, "30 fps footage plays on a 15 fps clock"
    offsets = _pass(montage)
    assert len(offsets) == 3  # 0, 67 and 133 ms: the longest clip's 133 ms
    frames = [[montage.frame_at(i, t) for i in range(2)] for t in offsets]
    assert [f[0] for f in frames] == [0, 2, 4], "the long clip runs to its end"
    assert [f[1] for f in frames] == [1, 3, 2], "the short one has looped round"
    montage.close()


def test_every_clip_starts_together(tmp_path):
    """Synchronised to each clip's own start, not to the recording's clock."""
    montage = _montage(tmp_path, spans=((3, 4), (0, 1)))
    assert [montage.frame_at(i, 0) for i in range(2)] == [3, 0]
    montage.close()


# --- the ticks --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "layers",
    [
        cm.MontageLayers(raw=True, mask=False),
        cm.MontageLayers(raw=True, mask=True, keypoints=True, mask_trail=False),
        cm.MontageLayers(raw=False, mask=True, mask_trail=False),
        cm.MontageLayers(raw=False, mask=False, keypoints=True),
        cm.MontageLayers(raw=False, mask=True, keypoints=True, mask_trail=False),
        cm.DEFAULT_LAYERS,
        cm.MontageLayers(
            raw=True, mask=True, keypoints=True, mask_trail=True, keypoint_trail=True
        ),
        cm.MontageLayers(
            raw=False, mask=True, keypoints=True, mask_trail=True, keypoint_trail=True
        ),
    ],
    ids=lambda layers: "-".join(
        k for k in ("raw", "mask", "keypoints", "mask_trail", "keypoint_trail")
        if getattr(layers, k)
    ),
)  # fmt: skip
def test_every_tick_combination_draws_a_tile(tmp_path, layers):
    montage = _montage(tmp_path)
    tile = montage.tile(0, 4, layers, 48)
    assert tile.shape == (48, 48, 3) and tile.dtype == np.uint8
    white = (tile == 255).all(axis=2).mean()
    if layers.raw:
        assert white < 0.5, "Raw draws the footage, not a white page"
    else:
        assert white > 0.3, "Raw off draws on white"
        assert (tile != 255).any(axis=2).any(), "...and draws something on it"
    montage.close()


def test_each_tick_changes_the_picture(tmp_path):
    """A tick that is accepted and draws nothing different is the failure to catch."""
    montage = _montage(tmp_path)

    def tile(**kw):
        kw.setdefault("mask", False)
        kw.setdefault("mask_trail", False)
        return montage.tile(0, 4, cm.MontageLayers(**kw), 48).astype(int)

    plain = tile(raw=True)
    assert np.abs(tile(raw=True, mask=True) - plain).sum() > 0
    assert np.abs(tile(raw=True, keypoints=True) - plain).sum() > 0
    masks = tile(raw=False, mask=True)
    both = tile(raw=False, mask=True, keypoints=True)
    assert np.abs(both - masks).sum() > 0, "keypoints drawn over the masks"
    trailed = tile(raw=False, mask=True, mask_trail=True)
    assert np.abs(trailed - masks).sum() > 0, "the mask trail shows on white"
    raw_trail = tile(raw=True, mask=True, mask_trail=True)
    assert np.abs(raw_trail - tile(raw=True, mask=True)).sum() > 0, (
        "the mask trail shows on the footage too"
    )  # fmt: skip
    montage.close()


def test_nothing_drawn_and_orphan_trails_are_refused():
    with pytest.raises(ValueError, match="at least one"):
        cm.MontageLayers(raw=False, mask=False)
    with pytest.raises(ValueError, match="mask trail"):
        cm.MontageLayers(raw=True, mask=False, mask_trail=True)
    with pytest.raises(ValueError, match="keypoint trail"):
        cm.MontageLayers(raw=True, keypoint_trail=True)


def test_the_layers_name_their_flavour_and_their_read_in_the_clip_windows_words():
    """The three looks share ``clip_render``'s vocabulary, and the read a tile needs is the
    clip window's own settings object — never a mask tinted onto the footage (a tile outlines
    it at its own scale), the trails only when asked."""
    from detail.api import clip_render

    assert cm.MontageLayers(raw=True, mask=False).flavour == clip_render.RAW
    assert cm.MontageLayers(raw=True).flavour == clip_render.OVERLAY
    assert cm.MontageLayers(raw=True, mask=False, keypoints=True).flavour == (
        clip_render.OVERLAY
    )
    assert cm.DEFAULT_LAYERS.flavour == clip_render.MASK
    assert cm.DEFAULT_LAYERS.settings == ClipViewSettings(
        masks_on_frame=False, mask_trail=True
    )
    full = cm.MontageLayers(
        raw=True, keypoints=True, keypoint_trail=True, trail_points=("nose",)
    )
    assert full.settings == ClipViewSettings(
        masks_on_frame=False,
        points_on_frame=True,
        trail=True,
        trail_points=("nose",),
        mask_trail=True,
    )


RAW_OFF = [
    cm.DEFAULT_LAYERS,
    cm.MontageLayers(raw=False, mask=True, mask_trail=False),
    cm.MontageLayers(raw=False, mask=False, keypoints=True, keypoint_trail=True),
    cm.MontageLayers(
        raw=False, mask=True, keypoints=True, mask_trail=True, keypoint_trail=True
    ),
]


def _every_tile(montage, layers, size=48) -> list[np.ndarray]:
    """One whole pass, every clip's tile at every step."""
    return [
        montage.tile(i, montage.frame_at(i, offset), layers, size)
        for offset in _pass(montage)
        for i in range(len(montage.clips))
    ]


@pytest.mark.parametrize(
    "layers", RAW_OFF, ids=lambda layers: "+".join(
        k for k in ("mask", "keypoints", "mask_trail", "keypoint_trail")
        if getattr(layers, k)
    ),
)  # fmt: skip
def test_a_raw_off_tile_decodes_no_video_and_is_the_decoded_tile(
    tmp_path, monkeypatch, layers
):
    """Raw off shows the masks and keypoints on white, which are stored data: a whole pass
    opens no video, and every tile is pixel for pixel the one drawn from a decoded frame.
    The decoded pass is the counter's own check: it must see those decodes."""
    from detail.api.run_preview import ClipQueryPreview

    calls = count_decodes(monkeypatch)
    blank = _every_tile(_montage(tmp_path / "blank"), layers)
    assert calls == [], f"a Raw-off pass decoded video: {calls[:3]}"
    real = ClipQueryPreview.clip_frame
    monkeypatch.setattr(
        ClipQueryPreview,
        "clip_frame",
        lambda self, *a, **k: real(self, *a, **{**k, "footage": True}),
    )
    decoded = _every_tile(_montage(tmp_path / "decoded"), layers)
    assert calls, "the decoded pass decoded, so the counter sees a decode"
    assert len(blank) == len(decoded)
    for one, other in zip(blank, decoded):
        np.testing.assert_array_equal(one, other)


def test_raw_on_still_decodes_the_footage(tmp_path, monkeypatch):
    calls = count_decodes(monkeypatch)
    _every_tile(_montage(tmp_path), cm.MontageLayers(raw=True))
    assert any(seam == "FrameCursor.frame" for seam, _ in calls)


def test_a_pass_reads_each_clips_masks_once(tmp_path, monkeypatch):
    """A tile wants its frame's rows and the frames just before it (the mask trail), step
    after step, and each ask decompressed the same ``mask_rle`` chunk again. The rows come
    from a cached block instead — one read of the mask column per decode context for the
    whole pass."""
    from detail.storage.zarr_io import PartZarrReader

    montage = _montage(tmp_path)
    reads = []
    real = PartZarrReader.column
    monkeypatch.setattr(
        PartZarrReader,
        "column",
        lambda self, name: reads.append(name) or real(self, name),
    )
    tiles = _every_tile(montage, cm.DEFAULT_LAYERS)
    contexts = len(montage._shared.previews)
    assert len(tiles) > 2 * contexts
    assert reads.count("mask_rle") == contexts
    montage.close()


def test_without_its_video_file_a_raw_off_tile_still_draws(tmp_path):
    """Raw off needs nothing from the video, so a recording whose file is gone (a
    video-excluded export) still shows its masks; Raw on is the grey no-picture tile."""
    montage = _montage(tmp_path)
    (video,) = tmp_path.rglob("part_001.mp4")
    video.unlink()
    off = montage.tile(0, 4, cm.DEFAULT_LAYERS, 48)
    assert (off == 255).all(axis=2).mean() > 0.3 and (off != 255).any(axis=2).any()
    on = montage.tile(0, 4, cm.MontageLayers(raw=True), 48)
    assert (on == cm.NO_PICTURE).all(axis=2).mean() > 0.5
    montage.close()


def test_a_recording_with_no_recorded_size_is_decoded_by_the_montage(tmp_path):
    """The API refuses to size a picture it cannot know (``clip_frame(footage=False)``
    raises); the montage, as the caller, reads the footage for that clip and draws on.
    """
    from detail.storage import recording as recording_store

    recording = walking(tmp_path)
    data = recording_store.read_recording(recording.dir)
    del data["video"]["resolution"]
    recording_store.write_recording(recording.dir, data)
    montage = cm.ClusterMontage(
        recording.dir.parent.parent, [_clip(recording, 1, 0, 4)], workers=1
    )
    montage.prepare()
    assert montage._shared.prepared[montage.clips[0]].needs_footage
    tile = montage.tile(0, 4, cm.DEFAULT_LAYERS, 48)
    assert (tile == 255).all(
        axis=2
    ).mean() > 0.3, "the masks on white, sized by the footage"
    assert (tile != 255).any(axis=2).any()
    montage.close()


def test_a_raw_off_export_decodes_no_video(tmp_path, monkeypatch):
    montage = _montage(tmp_path)
    calls = count_decodes(monkeypatch)
    montage.export(tmp_path / "m.gif", cm.DEFAULT_LAYERS, ["a", "b"], max_width=96)
    montage.export_clip_gifs(
        tmp_path / "gifs", cm.DEFAULT_LAYERS, ["a", "b"], ["a", "b"], max_width=48
    )
    assert calls == []
    montage.close()


# --- ownership and order ------------------------------------------------------------------ #
def test_drawing_before_prepare_refuses_rather_than_waiting(tmp_path):
    """The latent deadlock: a tile asked for before ``prepare`` used to prepare from inside
    the pool task that wanted it, a nested blocking ``pool.map`` on the pool it was running
    on. Now it raises, at once, on the thread that asked."""
    recording = walking(tmp_path)
    clips = [_clip(recording, i + 1, 0, 4) for i in range(3)]
    montage = cm.ClusterMontage(recording.dir.parent.parent, clips)
    assert montage._shared.workers >= 1
    outcome: list = []

    def draw():
        try:
            montage.tile(0, 0, cm.DEFAULT_LAYERS, 32)
        except Exception as exc:  # noqa: BLE001 — the test records whatever came out
            outcome.append(exc)
        else:
            outcome.append(None)

    worker = threading.Thread(target=draw)
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), "tile() before prepare() is still waiting"
    assert isinstance(outcome[0], RuntimeError) and "prepare" in str(outcome[0])
    for ask in (
        lambda: montage.fps,
        lambda: montage.duration_ns,
        lambda: montage.frame_at(0, 0),
        lambda: montage.frame_span(0),
    ):
        with pytest.raises(RuntimeError, match="prepare"):
            ask()
    montage.close()


def test_close_drops_the_decode_contexts_and_the_reads(tmp_path):
    montage = _montage(tmp_path)
    montage.tile(0, 0, cm.DEFAULT_LAYERS, 32)
    assert montage._shared.previews and montage._shared.prepared
    montage.close()
    assert montage._shared.previews == [] and montage._shared.prepared == {}


def test_the_montage_sizes_its_threads_by_the_one_rule(tmp_path):
    from detail.core import cpu

    recording = walking(tmp_path)
    montage = cm.ClusterMontage(
        recording.dir.parent.parent, [_clip(recording, 1, 0, 4)]
    )
    assert montage._shared.workers == cpu.thread_workers()
    montage.close()


# --- the crop ---------------------------------------------------------------------------- #
def test_the_crop_is_a_square_around_the_animal_with_a_margin():
    boxes = np.array([[0.4, 0.4, 0.5, 0.5], [0.45, 0.4, 0.55, 0.5]])
    x0, y0, x1, y1 = cm.crop_box(boxes, (1000, 1000), min_bodies=0)
    assert x1 - x0 == y1 - y0 == round(150 * 1.3)
    assert abs((x0 + x1) / 2 - 475) <= 1 and abs((y0 + y1) / 2 - 450) <= 1


def test_a_still_animal_is_not_cropped_to_its_outline():
    boxes = np.array([[0.4, 0.4, 0.5, 0.5]] * 3)
    x0, _y0, x1, _y1 = cm.crop_box(boxes, (1000, 1000))
    assert x1 - x0 == round(cm.CROP_MIN_BODIES * 100)


def test_the_crop_stays_inside_the_frame():
    boxes = np.array([[0.95, 0.0, 1.0, 0.1]])
    x0, y0, x1, y1 = cm.crop_box(boxes, (640, 480))
    assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480


def test_an_untracked_clip_shows_the_whole_frame():
    assert cm.crop_box(np.full((4, 4), np.nan), (640, 480)) == (0, 0, 640, 480)


# --- labels and the cap ------------------------------------------------------------------ #
def test_a_label_is_rank_recording_and_start():
    clip = cm.MontageClip(
        3, "recording_007", "M01_day1", 0, "emb_01", 42_100_000_000, 0
    )
    assert cm.montage_label(clip) == "#3 · M01_day1 · 0:42.1"
    assert cm.montage_label(clip, "C") == "#3 · recording C · 0:42.1"


def test_blind_mode_letters_recordings_in_order_of_appearance():
    clips = [
        cm.MontageClip(i + 1, rec, f"name_{rec}", 0, "e", 0, 0)
        for i, rec in enumerate(["r9", "r2", "r9", "r5"])
    ]
    assert cm.montage_labels(clips, blind=True) == [
        "#1 · recording A · 0:00.0",
        "#2 · recording B · 0:00.0",
        "#3 · recording A · 0:00.0",
        "#4 · recording C · 0:00.0",
    ]
    assert cm.montage_labels(clips, blind=False)[1] == "#2 · name_r2 · 0:00.0"


def _fake_run(members, asked=None):
    class _Run:
        id = "an_07"

        def typical_members(self, node, limit, progress=None):
            if asked is not None:
                asked.append(limit)
            return members if limit is None else members[:limit]

        def state(self):
            class _State:
                @staticmethod
                def embed_run_for(recording_id):
                    return "emb_01"

            return _State()

        def node_names(self):
            return {"5": "rearing"}

    class _Experiment:
        @staticmethod
        def recording_names():
            return {f"r{i}": f"mouse {i}" for i in range(7)}

    return _Experiment(), _Run()


_SIXTY = [
    {
        "recording_id": f"r{i % 7}",
        "subject": 0,
        "start_t_ns": i,
        "end_t_ns": i + 1 + i % 5,
    }
    for i in range(60)
]


def test_the_defaults_are_twenty_five_clips_of_the_mask_and_its_trail():
    """The window, the CLI and the API open on the same thing."""
    assert cm.DEFAULT_CLIPS == 25
    assert (
        cm.DEFAULT_LAYERS
        == cm.MontageLayers()
        == cm.MontageLayers(
            raw=False, mask=True, keypoints=False, mask_trail=True, keypoint_trail=False
        )
    )
    assert cm.MontageLayers(mask=False, keypoints=True).mask_trail is False, (
        "an unset mask trail follows the mask, so a keypoints-only montage is not refused"
    )  # fmt: skip


def test_a_montage_shows_twenty_five_most_typical_unless_asked():
    asked = []
    experiment, run = _fake_run(_SIXTY, asked)
    clips = cm.montage_clips(experiment, run, 5)
    assert len(clips) == 25
    assert asked == [25], "the head of the ranking is all the default order needs"
    assert [c.rank for c in clips] == list(range(1, 26))
    assert [c.start_t_ns for c in clips] == list(range(25)), "most typical first"
    assert clips[0].recording_name == "mouse 0"
    more = cm.montage_clips(experiment, run, 5, limit=55)
    assert len(more) == 55, "no fifty cap: any count up to the cluster"


def test_least_typical_reads_every_member_and_starts_from_the_last():
    asked = []
    experiment, run = _fake_run(_SIXTY, asked)
    clips = cm.montage_clips(experiment, run, 5, limit=3, sort="least-typical")
    assert asked == [None], "the tail needs the whole ranking"
    assert [c.rank for c in clips] == [60, 59, 58]
    with pytest.raises(ValueError, match="unknown sort"):
        cm.montage_clips(experiment, run, 5, sort="shuffled")
    with pytest.raises(ValueError, match="at least 1"):
        cm.montage_clips(experiment, run, 5, limit=0)


@pytest.mark.parametrize("sort", list(cm.SORTS))
def test_every_order_sorts_and_keeps_each_clips_rank(sort):
    experiment, run = _fake_run(_SIXTY)
    members = cm.montage_members(experiment, run, 5)
    ordered = cm.sort_clips(members, sort)
    assert sorted(c.rank for c in ordered) == list(range(1, 61)), "a sort loses nothing"
    if sort == "typical":
        assert [c.rank for c in ordered] == list(range(1, 61))
    elif sort == "least-typical":
        assert [c.rank for c in ordered] == list(range(60, 0, -1))
    elif sort == "recording":
        keys = [(c.recording_name, c.start_t_ns) for c in ordered]
        assert keys == sorted(keys)
    elif sort == "random":
        assert [c.rank for c in ordered] != list(range(1, 61)), "a shuffle moves clips"
        assert ordered == cm.sort_clips(members, sort), "the same seed, the same order"
        assert ordered != cm.sort_clips(
            members, sort, seed=1
        ), "another node, another order"
    else:
        lengths = [c.end_t_ns - c.start_t_ns for c in ordered]
        assert lengths == sorted(lengths, reverse=True)
        tied = [c.rank for c in ordered if c.end_t_ns - c.start_t_ns == lengths[0]]
        assert tied == sorted(tied), "a tie is broken by typicality"
    head = cm.choose_clips(members, count=7, sort=sort)
    assert head == ordered[:7], "the count counts from the top of the order"


def test_choosing_clips_refuses_an_unknown_sort_or_no_clips():
    experiment, run = _fake_run(_SIXTY)
    members = cm.montage_members(experiment, run, 5)
    with pytest.raises(ValueError, match="unknown sort"):
        cm.choose_clips(members, sort="shuffled")
    with pytest.raises(ValueError, match="at least 1"):
        cm.choose_clips(members, count=0)


def test_an_origin_names_its_behaviour():
    assert cm.MontageOrigin("an_01", 7, "rearing").behavior_name == "rearing"
    assert cm.MontageOrigin("an_01", 7).behavior_name == "cluster 7"
    assert cm.MontageOrigin("an_01", 7, unit="behavior").behavior_name == "behavior 7"


# --- the grid ----------------------------------------------------------------------------- #
def test_an_export_frame_is_the_size_its_grid_says():
    """One grid formula: the export's picture is exactly the size ``grid_size`` says for the
    columns and tile ``export_grid`` chose, within the width asked, and the last tile sits
    where ``tile_origin`` puts it."""
    for count, max_width in ((5, 300), (1, 120), (50, None), (7, 640)):
        columns, tile_px = cm.export_grid(count, max_width)
        rows = math.ceil(count / columns)
        tiles = [np.full((40, 40, 3), 128, np.uint8)] * count
        picture = cm.compose(tiles, [""] * count, columns=columns, tile_px=tile_px)
        width, height = cm.grid_size(columns, rows, tile_px)
        assert picture.shape == (height, width, 3), (count, max_width)
        if max_width is not None:
            assert width <= max_width
        else:
            assert tile_px == cm.EXPORT_TILE_PX and width <= cm.EXPORT_MAX_WIDTH
        x, y = cm.tile_origin(count - 1, columns=columns, tile_px=tile_px)
        assert x + tile_px <= width and y + tile_px + cm.label_px(tile_px) <= height


def test_a_long_label_keeps_its_rank_and_time():
    text = cm.fit_label("#12 · a_very_long_recording_name · 1:02.3", len, 24)
    assert text.startswith("#12 · ") and text.endswith(" · 1:02.3") and "…" in text


# --- exports ----------------------------------------------------------------------------- #
def test_a_gif_export_has_the_asked_width_and_rate(tmp_path):
    montage = _montage(tmp_path)
    labels = cm.montage_labels(montage.clips, blind=False)
    path = montage.export(
        tmp_path / "m.gif", cm.MontageLayers(raw=True, mask=True), labels,
        max_width=120, fps=10,
    )  # fmt: skip
    with Image.open(path) as gif:
        assert gif.size[0] <= 120 and gif.size[0] >= 100
        assert gif.info["duration"] == 100
        assert gif.n_frames == 2  # 0 and 100 ms of a 133 ms pass at 10 fps
    montage.close()


def test_an_mp4_export_plays_at_the_montage_clock(tmp_path):
    montage = _montage(tmp_path)
    labels = cm.montage_labels(montage.clips, blind=True)
    path = montage.export(tmp_path / "m.mp4", cm.MontageLayers(raw=True), labels)
    cap = cv2.VideoCapture(str(path))
    try:
        assert round(cap.get(cv2.CAP_PROP_FPS)) == 15
        count = 0
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    assert count == len(_pass(montage))
    montage.close()


# --- the mask on the footage is an outline ----------------------------------------------- #
def _animal(montage, size):
    """The clip's animal in a ``size`` tile, as the filled mask on white shows it."""
    from detail.core.visualize import MASK_TINT

    filled = montage.tile(
        0, 4, cm.MontageLayers(raw=False, mask=True, mask_trail=False), size
    ).astype(int)
    return (np.abs(filled - np.asarray(MASK_TINT)).sum(axis=2) < 30), filled


def test_raw_on_outlines_the_animal_and_leaves_its_pixels_alone(tmp_path):
    """With the footage on, the animal's own pixels are the footage's: only a line *round*
    it is tinted. A tint over the animal (the clip window's look) fails the interior check.
    """
    from detail.core.visualize import MASK_TINT

    montage = _montage(tmp_path)
    size = 96
    animal, _filled = _animal(montage, size)
    assert animal.sum() > 50, "the fixture's animal fills part of the tile"
    interior = cv2.erode(animal.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert interior.any()
    plain = montage.tile(0, 4, cm.MontageLayers(raw=True, mask=False), size).astype(int)
    outlined = montage.tile(
        0, 4, cm.MontageLayers(raw=True, mask=True, mask_trail=False), size
    ).astype(int)
    changed = np.abs(outlined - plain).sum(axis=2) > 0
    assert not (changed & interior).any(), "the animal's interior was tinted"
    assert changed.any(), "...and the outline was drawn somewhere"
    near = cv2.dilate(animal.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert (changed & ~near).sum() == 0, "the line sits round the animal, not elsewhere"
    tinted = np.abs(outlined - np.asarray(MASK_TINT)).sum(axis=2) < 30
    assert (tinted & changed).sum() >= 0.8 * changed.sum(), "the line is the mask tint"
    montage.close()


def test_raw_off_fills_the_animal_with_the_tint(tmp_path):
    montage = _montage(tmp_path)
    animal, filled = _animal(montage, 96)
    assert animal.sum() > 50, "a filled tint on white, not an outline"
    assert (filled[~animal] > 200).all(axis=1).mean() > 0.6, "...on white"
    montage.close()


def test_the_outline_scales_with_the_tile():
    assert cm.outline_px(48) == 2 and cm.outline_px(96) == 2
    assert cm.outline_px(224) == 4
    assert cm.outline_px(640) == 10


@pytest.mark.parametrize("size", [32, 160])
def test_the_outline_is_visible_at_any_tile_size(tmp_path, size):
    montage = _montage(tmp_path)
    plain = montage.tile(0, 4, cm.MontageLayers(raw=True, mask=False), size)
    outlined = montage.tile(
        0, 4, cm.MontageLayers(raw=True, mask=True, mask_trail=False), size
    )
    ring = (np.abs(outlined.astype(int) - plain.astype(int)).sum(axis=2) > 0).sum()
    # A closed line round the animal at outline_px(size) wide covers at least its perimeter.
    assert ring >= 4 * cm.outline_px(size) * 4
    montage.close()


def test_the_mask_trail_on_the_footage_is_outlines_too(tmp_path):
    montage = _montage(tmp_path)
    size = 96
    animal, _filled = _animal(montage, size)
    interior = cv2.erode(animal.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    plain = montage.tile(0, 4, cm.MontageLayers(raw=True, mask=False), size).astype(int)
    trailed = montage.tile(0, 4, cm.MontageLayers(raw=True), size).astype(int)
    still = montage.tile(
        0, 4, cm.MontageLayers(raw=True, mask_trail=False), size
    ).astype(int)
    assert np.abs(trailed - still).sum() > 0, "the trail adds the past outlines"
    assert not (np.abs(trailed - plain).sum(axis=2) > 0)[interior].any(), (
        "and leaves the animal's own pixels alone"
    )  # fmt: skip
    montage.close()


# --- one GIF per clip --------------------------------------------------------------------- #
def test_each_clip_is_written_as_its_own_gif(tmp_path):
    # One thread: the bar is fed per frame there, which three frames can show; on the pool
    # the bar ticks every 0.1 s and these GIFs land inside one tick.
    montage = _montage(tmp_path, workers=1)
    clips = montage.clips
    stems = [cm.clip_file_stem(c) for c in clips]
    assert stems == ["1_M01_day1_0-00.0", "2_M01_day1_0-00.0"]
    events = []
    paths = montage.export_clip_gifs(
        tmp_path / "gifs",
        cm.DEFAULT_LAYERS,
        cm.montage_labels(clips, blind=False),
        stems,
        max_width=100,
        fps=10,
        progress=lambda *e: events.append(e),
    )
    assert [p.name for p in paths] == ["1_M01_day1_0-00.0.gif", "2_M01_day1_0-00.0.gif"]
    assert sorted(p.name for p in (tmp_path / "gifs").iterdir()) == [
        p.name for p in paths
    ]
    with Image.open(paths[0]) as gif:
        assert gif.size[0] == 100
        assert gif.info["duration"] == 100
        assert gif.n_frames == 2  # 0 and 100 ms of its own 133 ms
    with Image.open(paths[1]) as gif:
        assert gif.n_frames == 1  # a 33 ms clip: one picture
    fractions = [f for stage, f, _d in events if stage == "writing GIFs"]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert any(0 < f < 1 for f in fractions), "the bar moves within the batch"
    montage.close()


def test_a_blind_file_name_hides_the_recording():
    clip = cm.MontageClip(12, "recording_007", "M01 day1/x", 0, "e", 62_300_000_000, 0)
    assert cm.clip_file_stem(clip) == "12_M01_day1_x_1-02.3"
    assert cm.clip_file_stem(clip, "C") == "12_recording-C_1-02.3"


def test_a_stopped_batch_keeps_the_finished_gifs(tmp_path):
    from detail.core.progress import OperationCancelled

    montage = _montage(tmp_path, workers=1)
    written = []

    def progress(stage, fraction, detail):
        if stage == "writing GIFs" and "GIF 2" in detail:
            raise OperationCancelled()

    with pytest.raises(OperationCancelled):
        montage.export_clip_gifs(
            tmp_path / "gifs",
            cm.DEFAULT_LAYERS,
            ["a", "b"],
            ["one", "two"],
            max_width=64,
            fps=10,
            progress=progress,
            on_written=written.append,
        )
    assert [p.name for p in written] == ["one.gif"]
    assert sorted(p.name for p in (tmp_path / "gifs").iterdir()) == ["one.gif"]
    montage.close()


def test_one_gif_alone_still_moves_the_bar(tmp_path):
    """The cadence trap: with several GIFs the bar moves between them whatever happens inside
    one, so the single-GIF batch — a user ticking one clip — is the case that proves the bar
    is fed from inside the frame loop."""
    montage = _montage(tmp_path, spans=((0, 4),))
    events = []
    montage.export_clip_gifs(
        tmp_path / "one",
        cm.DEFAULT_LAYERS,
        ["a"],
        ["a"],
        max_width=64,
        fps=15,
        progress=lambda *e: events.append(e),
    )
    fractions = [f for stage, f, _d in events if stage == "writing GIFs"]
    assert any(0 < f < 1 for f in fractions), fractions
    montage.close()


def test_per_clip_gifs_written_side_by_side_match_one_at_a_time(tmp_path):
    """The montage's threads write the per-clip GIFs a clip each; the files are the ones a single
    thread writes, frame for frame, and the bar still only moves forward and ends at 1.
    """
    recording = walking(tmp_path)
    clips = [
        _clip(recording, i + 1, a, b)
        for i, (a, b) in enumerate([(0, 4), (1, 2), (0, 3), (2, 4)])
    ]
    root = recording.dir.parent.parent
    labels = cm.montage_labels(clips, blind=False)
    stems = [f"clip{i}" for i in range(len(clips))]
    written = {}
    for workers in (1, 3):
        montage = cm.ClusterMontage(root, clips, workers=workers)
        events, landed = [], []
        try:
            paths = montage.export_clip_gifs(
                tmp_path / f"w{workers}", cm.DEFAULT_LAYERS, labels, stems,
                max_width=80, fps=10, progress=lambda *e: events.append(e),
                on_written=landed.append,
            )  # fmt: skip
        finally:
            montage.close()
        assert [p.name for p in paths] == [
            f"{s}.gif" for s in stems
        ], "the clips' order"
        assert sorted(landed) == sorted(paths)
        fractions = [f for stage, f, _d in events if stage == "writing GIFs"]
        assert fractions == sorted(fractions) and fractions[-1] == 1.0
        written[workers] = []
        for path in paths:
            with Image.open(path) as gif:
                frames = []
                for k in range(gif.n_frames):
                    gif.seek(k)
                    frames.append(np.asarray(gif.convert("RGB")))
                written[workers].append((gif.info["duration"], np.stack(frames)))
    for (d1, one), (d3, three) in zip(written[1], written[3]):
        assert d1 == d3 and np.array_equal(one, three)


def test_a_stopped_side_by_side_batch_keeps_only_finished_gifs(tmp_path):
    """Cancelled mid-batch with several GIFs in flight: the finished ones stay (and were
    announced), the ones being written are removed, and nothing is half-written."""
    from detail.core.progress import OperationCancelled

    recording = walking(tmp_path)
    clips = [_clip(recording, i + 1, 0, 4) for i in range(6)]
    montage = cm.ClusterMontage(recording.dir.parent.parent, clips, workers=3)
    landed = []

    def progress(stage, fraction, detail):
        if stage == "writing GIFs" and fraction > 0.3:
            raise OperationCancelled()

    try:
        with pytest.raises(OperationCancelled):
            montage.export_clip_gifs(
                tmp_path / "gifs", cm.DEFAULT_LAYERS, ["l"] * 6,
                [f"g{i}" for i in range(6)], max_width=64, fps=30,
                progress=progress, on_written=landed.append,
            )  # fmt: skip
    finally:
        montage.close()
    on_disk = sorted(p.name for p in (tmp_path / "gifs").iterdir())
    assert on_disk == sorted(p.name for p in landed), "a file nobody was told about"
    assert len(on_disk) < 6, "the batch did not stop"
    for name in on_disk:
        with Image.open(tmp_path / "gifs" / name) as gif:
            assert gif.n_frames == 5, f"{name} is half-written"


def test_gifs_already_in_the_folder_are_refused_unless_replace(tmp_path):
    """A folder is picked, the names in it are not, so a second export into one folder wrote
    over the first's GIFs with nothing asked (#29). The GUI asks and passes ``replace``; the
    CLI refuses."""
    recording = walking(tmp_path)
    clips = [_clip(recording, i + 1, 0, 4) for i in range(2)]
    out = tmp_path / "gifs"
    out.mkdir()
    (out / "g1.gif").write_bytes(b"the first export")
    assert cm.existing_gifs(out, ["g0", "g1"]) == ["g1.gif"]
    montage = cm.ClusterMontage(recording.dir.parent.parent, clips, workers=1)
    try:
        with pytest.raises(FileExistsError, match="g1.gif"):
            montage.export_clip_gifs(
                out, cm.DEFAULT_LAYERS, ["l"] * 2, ["g0", "g1"], max_width=64, fps=10
            )
        assert (out / "g1.gif").read_bytes() == b"the first export"
        assert not (out / "g0.gif").exists(), "refused before anything was written"
        montage.export_clip_gifs(
            out, cm.DEFAULT_LAYERS, ["l"] * 2, ["g0", "g1"], max_width=64, fps=10,
            replace=True,
        )  # fmt: skip
    finally:
        montage.close()
    assert (out / "g1.gif").read_bytes() != b"the first export"


# --- the clip list ------------------------------------------------------------------------ #
def test_the_clip_list_is_a_clip_query_csv_with_the_montages_columns_after(tmp_path):
    import csv

    from detail.storage.clip_query_csv import COLUMNS, read_clip_csv

    montage = _montage(tmp_path)
    origin = cm.MontageOrigin(run_id="an_07", node=12, name="rearing")
    path = montage.export_clip_list(tmp_path / "clips.csv", origin)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
        header = list(rows[0].keys()) if rows else []
    assert header == cm.MONTAGE_LIST_COLUMNS == COLUMNS + list(cm.MONTAGE_LIST_EXTRA)
    first = rows[0]
    clip = montage.clips[0]
    assert first["kind"] == "query" and first["rank"] == "0", "an example, not a match"
    assert first["rank_typical"] == "1" and first["recording_name"] == "M01_day1"
    assert first["recording_id"] == clip.recording_id and first["subject"] == "0"
    assert int(first["start_t_ns"]) == clip.start_t_ns
    assert int(first["end_t_ns"]) == clip.end_t_ns
    assert float(first["end_s"]) == pytest.approx(clip.end_t_ns / 1e9, abs=1e-3)
    assert first["node"] == "12" and first["run_id"] == "an_07"
    assert (first["start_frame"], first["end_frame"]) == ("0", "4")
    back = read_clip_csv(path)
    assert [(r.recording_id, r.track_id, r.start_t_ns, r.end_t_ns) for r in back] == [
        (c.recording_id, c.subject, c.start_t_ns, c.end_t_ns) for c in montage.clips
    ]
    assert {r.query_name for r in back} == {"rearing"}
    assert [r.example_idx for r in back] == [0, 1]
    montage.close()


# --- what an export costs ------------------------------------------------------------------ #
def test_an_export_draws_each_frame_it_holds_once(tmp_path, monkeypatch):
    """Within one loop a clip's frames only go forward, so a frame sampled twice in a row is
    the tile already drawn: 60 samples a second over 30 fps footage ask for every frame twice, and each
    is drawn once."""
    montage = _montage(tmp_path, spans=((0, 4),))
    layers = cm.MontageLayers(raw=True)
    drawn = []
    real = montage.tile
    monkeypatch.setattr(montage, "tile", lambda *a: drawn.append(a[:2]) or real(*a))
    montage.export(tmp_path / "m.gif", layers, ["a"], max_width=70, fps=60)
    assert drawn, "the export drew nothing"
    assert len(set(drawn)) == len(drawn), f"a tile was drawn twice: {drawn}"
    montage.close()


def test_a_cold_prepare_parses_each_recording_once(tmp_path, monkeypatch):
    """Six clips of one recording, prepared on six threads with nothing cached: its
    ``recording.yaml`` is read and parsed **once**.

    Each clip's decode context reads its recording's record, and on a cold cache they all missed
    it together, and the repeated parses, sharing one interpreter, were most of a cold
    prepare. The read is slowed here so the threads
    overlap every time rather than when the scheduler happens to interleave them.
    """
    import time

    from detail.storage import read_cache
    from detail.storage import recording as recording_store

    recording = walking(tmp_path)
    clips = [_clip(recording, i + 1, i % 3, 4) for i in range(6)]
    montage = cm.ClusterMontage(recording.dir.parent.parent, clips, workers=6)
    recording_store.clear_recording_cache()
    read_cache.clear_read_cache()
    reads = []
    guard = threading.Lock()
    parse = recording_store._parse_recording

    def counted(path):
        with guard:
            reads.append(str(path))
        time.sleep(0.05)
        return parse(path)

    monkeypatch.setattr(recording_store, "_parse_recording", counted)
    try:
        montage.prepare()
    finally:
        montage.close()
    assert len(reads) == 1, f"recording.yaml read {len(reads)} times"


def test_least_typical_keeps_unranked_clips_last_as_the_clip_table_does():
    """A medoid method cannot rank a bout assigned after its fit (``NaN``). Reversing the rank
    put those first in *least typical*, while ``ClipTable.order("atypical")`` keeps them last:
    the montage and the cluster view would disagree about which clip is least typical.
    """
    from detail.api.node_view import ClipTable

    typicality = np.array([0.1, np.nan, 0.3, 0.2])
    n = len(typicality)
    clips = ClipTable(
        recording_ids=("recording_001",),
        recording=np.zeros(n, np.int32),
        subject=np.zeros(n, np.int64),
        start_ns=np.arange(n, dtype=np.int64) * 10,
        end_ns=np.arange(n, dtype=np.int64) * 10 + 5,
        cx=np.full(n, np.nan, np.float32),
        cy=np.full(n, np.nan, np.float32),
        members=np.ones(n, np.int64),
        typicality=typicality,
        leaf=np.zeros(n, np.int64),
    )
    rows = clips.rows(clips.order("typical"))
    montage = [
        cm.MontageClip(
            rank=rank,
            recording_id=r["recording_id"],
            recording_name="r",
            subject=0,
            embed_run_id="emb_01",
            start_t_ns=r["start_t_ns"],
            end_t_ns=r["end_t_ns"],
            ranked=cm._is_ranked(r["typicality"]),
        )
        for rank, r in enumerate(rows, start=1)
    ]
    least = [c.start_t_ns for c in cm.sort_clips(montage, "least-typical")]
    assert least == [int(clips.start_ns[i]) for i in clips.order("atypical")]
    assert least[-1] == 10, "the unranked clip is last"
    assert [c.rank for c in cm.sort_clips(montage, "typical")][-1] == 4
