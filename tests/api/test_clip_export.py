"""One clip, rendered for real — the file a player's **Export video…** writes.

Asserts the *content*, not that the call returned: which source frames landed in the mp4, and
that the `clip.csv` beside it says where they came from. A clip export that silently wrote the
wrong range is a plausible file, which is the failure mode a "did it return a path" test cannot
see — and the whole point of the export is the frames.

`write_indexed_video` / `frame_indices` / `make_part` come from `test_clip_render.py`, where the
greys encode their own source index so a decoded clip can be read back to the frames it holds.
"""

import csv

import pytest

from detail.api import clip_export, clip_render
from tests.api.test_clip_render import (
    frame_indices,
    make_part,
    write_indexed_video,
)

_N_FRAMES = 60


class _FakeScrub:
    def __init__(self, video, frames, track_id):
        self.part_frame_counts = [_N_FRAMES]
        self._parts = [make_part(video, frames, track_id=track_id)]


class _FakePreview:
    """Stands in for `ClipQueryPreview` over one single-part recording, 1 ms a frame.

    The same seam `export_behavior_clips` and `syllable_export` are tested through — the
    `preview=` parameter exists so a render can be driven without an experiment on disk.
    """

    def __init__(self, video, frames=range(_N_FRAMES), track_id=0):
        self._scrub = _FakeScrub(video, frames, track_id)

    def t_ns_to_frame(self, recording_id, embed_run_id, t_ns):
        return int(t_ns // 1_000_000)

    def frame_t_ns(self, recording_id, embed_run_id, frame):
        return int(frame) * 1_000_000

    def _scrub_ctx(self, recording_id, embed_run_id):
        return self._scrub, None


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "part.mp4"
    write_indexed_video(path, _N_FRAMES)
    return path


def _export(tmp_path, video, frames, *, preview=None, **kwargs):
    dest = tmp_path / "out"
    dest.mkdir(exist_ok=True)
    written = clip_export.export_clip(
        tmp_path,
        dest,
        recording_id="recording_001",
        frames=frames,
        embed_run_id="emb_01",
        preview=preview or _FakePreview(video),
        **kwargs,
    )
    return dest, written


def _rows(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_the_clip_holds_the_frames_that_were_asked_for(tmp_path, video):
    """The property: the file contains that span of the source and nothing else.

    Asserted on the ``raw`` flavour, because that is the one that leaves the source greys
    alone — an overlay tints them, and `frame_indices` reads the grey to recover the index.
    And inside frame 16, where those greys are still unique: past it they wrap modulo 256 and
    stop identifying anything, which the helper's own docstring says.
    """
    dest, written = _export(
        tmp_path, video, (4, 12), renders=(clip_render.RAW,), size=32
    )
    clip = dest / "recording_001_000004-000012.mp4"
    assert clip.is_file(), sorted(p.name for p in dest.iterdir())
    assert frame_indices(clip) == list(range(4, 13))
    assert clip in written and (dest / "clip.csv") in written


def test_the_csv_says_where_the_clip_came_from(tmp_path, video):
    """The data half. An mp4 alone cannot say which recording or which frames it is.

    Driven on track 3 against a part whose rows *are* track 3, so the subject column is checked
    against a render that actually happened rather than against an empty one.
    """
    dest, _written = _export(
        tmp_path,
        video,
        (10, 18),
        preview=_FakePreview(video, track_id=3),
        track_id=3,
    )
    (row,) = _rows(dest / "clip.csv")
    assert row["recording_id"] == "recording_001"
    assert row["subject"] == "3"
    assert (int(row["start_frame"]), int(row["end_frame"])) == (10, 18)
    assert row["video_overlay"] == "recording_001_000010-000018.mp4"
    assert float(row["dur_s"]) == pytest.approx(0.008, abs=1e-3)


def test_a_backwards_range_is_read_as_the_span_between_its_ends(tmp_path, video):
    """A drag from right to left is still a selection, not an empty one."""
    dest, _written = _export(tmp_path, video, (18, 10))
    assert (dest / "recording_001_000010-000018.mp4").is_file()


def test_the_clip_plays_at_the_rate_the_behavior_happened(tmp_path, video):
    """Not a nominal 30: `plan_job` derives fps from the span's real duration, so a clip of a
    slow bout is not silently sped up. 9 frames over 8 ms is far above the clamp, which is what
    makes the ceiling observable here."""
    dest, _written = _export(tmp_path, video, (10, 18))
    (row,) = _rows(dest / "clip.csv")
    assert float(row["fps"]) == 120.0  # clamped ceiling, not the 30.0 fallback


def test_several_flavours_cost_one_decode(tmp_path, video):
    """Raw and overlay from one pass, named apart so neither overwrites the other."""
    dest, _written = _export(
        tmp_path, video, (10, 14), renders=(clip_render.RAW, clip_render.OVERLAY)
    )
    (row,) = _rows(dest / "clip.csv")
    assert row["video_raw"] and row["video_overlay"]
    assert row["video_raw"] != row["video_overlay"]
    for name in (row["video_raw"], row["video_overlay"]):
        assert (dest / name).is_file()


def test_a_range_outside_the_video_writes_nothing(tmp_path, video):
    """Not a crash and not an empty file: nothing, so `write_export` can say so."""
    dest, written = _export(tmp_path, video, (_N_FRAMES + 10, _N_FRAMES + 20))
    assert written == []
    assert not list(dest.glob("*.mp4"))


def test_a_span_with_no_detections_writes_no_clip(tmp_path, video):
    """`render_part` only writes a frame that has a detection row, so a span the tracker never
    saw produces nothing rather than a file of dropped frames. `finish` unlinks it."""
    dest = tmp_path / "out"
    dest.mkdir()
    written = clip_export.export_clip(
        tmp_path,
        dest,
        recording_id="recording_001",
        frames=(10, 18),
        embed_run_id="emb_01",
        # Rows only for frames 40+, so nothing in the requested span has one.
        preview=_FakePreview(video, frames=range(40, _N_FRAMES)),
    )
    assert not list(dest.glob("*.mp4"))
    assert written == [
        dest / "clip.csv"
    ], "the index is still written, saying nothing landed"
