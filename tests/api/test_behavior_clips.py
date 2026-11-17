"""Clips from a §19 behavior artifact (api/behavior_clips.py) — one exporter, every producer.

The decode is pinned in ``test_clip_render``; what is left here is the orchestration the reported
bug was about. `cluster_export` and `syllable_export` did this for subjects that build hid (both
since removed), so the surfaced clustering path had no clip export at all — and the fix is written against the
**shared artifact** rather than a fourth module, so a producer added later inherits it.

The decode boundary is a test double over a real tiny mp4 (`_FakePreview`, defined here since
`test_cluster_export` was removed with the clusterers), which is a fake at a seam rather than a storage mock: everything on disk
is real.

**Tier: unit, measured rather than assumed.** This file does encode video, through a helper it
imports rather than a `cv2.VideoWriter` it names — the hole `tests/tiers.py` calls out, where the
signature scan cannot see the cost. So it was measured, on a 64px 16-frame clip, and it is
unit cost, so it stays in the tier that runs at every stage.
"""

import csv

import numpy as np
import pytest

from detail.api import behavior_clips, clip_render
from detail.storage import behavior_output as bo
from tests.api.test_clip_render import make_part, write_indexed_video

_N_FRAMES = 40


class _FakeScrub:
    def __init__(self, video, detected):
        self.part_frame_counts = [_N_FRAMES]
        self._parts = [make_part(video, detected)]


class _FakePreview:
    """Stands in for ClipQueryPreview: every recording is one part of the same tiny video."""

    def __init__(self, video, detected=range(_N_FRAMES)):
        self._scrub = _FakeScrub(video, detected)

    def _scrub_ctx(self, recording_id, embed_run_id):
        return self._scrub, None


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "part.mp4"
    write_indexed_video(path, _N_FRAMES)
    return path


def _instance(recording_id, source_label, start, end, *, label="", subject=0):
    """One bout, at frames `[start, end]` with times derived from them."""
    return bo.BehaviorInstance(
        recording_id,
        subject,
        int(start) * 1_000_000,
        int(end) * 1_000_000,
        int(start),
        int(end),
        source_label,
        label,
    )


def _rows(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _export(tmp_path, video, instances, **kwargs):
    dest = tmp_path / "out"
    dest.mkdir(exist_ok=True)
    written = behavior_clips.export_behavior_clips(
        tmp_path,
        dest,
        instances=instances,
        embed_run_ids={i.recording_id: "emb_01" for i in instances},
        preview=_FakePreview(video),
        **kwargs,
    )
    return dest, written


# --------------------------------------------------------------------------- #
# the artifact half — a clip per bout
# --------------------------------------------------------------------------- #
def test_a_clip_is_written_per_bout(tmp_path, video):
    rows = [_instance("r1", "c01", 2, 6), _instance("r1", "c02", 10, 14)]
    dest, _written = _export(tmp_path, video, rows)
    clips = sorted(p.name for p in dest.glob("*.mp4"))
    assert clips == ["c01_r1_000002-000006.mp4", "c02_r1_000010-000014.mp4"]


def test_the_cluster_comes_first_in_the_filename(tmp_path, video):
    """So a name-sorted folder groups a cluster together and plays it back to back.

    That is the whole reason clips land in one directory rather than a folder per cluster —
    reviewing a result in one pass, which is what the walkthrough asked for.
    """
    rows = [
        _instance("r2", "c02", 2, 4),
        _instance("r1", "c01", 20, 22),
        _instance("r1", "c01", 2, 4),
    ]
    dest, _written = _export(tmp_path, video, rows)
    names = sorted(p.name for p in dest.glob("*.mp4"))
    assert [n.split("_")[0] for n in names] == ["c01", "c01", "c02"]
    # Within a cluster, the zero-padded range puts them in time order.
    assert names[0] < names[1]


# --------------------------------------------------------------------------- #
# the data half — the rows behind the clips
# --------------------------------------------------------------------------- #
def test_clips_csv_names_the_bout_behind_every_file(tmp_path, video):
    """The "data required to recreate this video file" half, which is why this target
    declares `inputs` rather than `inputs_reason`."""
    rows = [_instance("r1", "c01", 2, 6, label="rearing")]
    dest, _written = _export(tmp_path, video, rows)
    table = _rows(dest / "clips.csv")
    assert len(table) == 1
    row = table[0]
    assert row["source_label"] == "c01"
    assert row["label"] == "rearing"
    assert row["recording_id"] == "r1"
    assert (row["start_frame"], row["end_frame"]) == ("2", "6")
    assert row["instance_id"] == rows[0].instance_id
    assert row["video_overlay"] == "c01_r1_000002-000006.mp4"


def test_clips_csv_row_count_equals_the_files_written(tmp_path, video):
    rows = [_instance("r1", "c01", i * 4, i * 4 + 2) for i in range(4)]
    dest, _written = _export(tmp_path, video, rows)
    assert len(_rows(dest / "clips.csv")) == len(list(dest.glob("*.mp4")))


def test_a_bout_whose_clip_could_not_be_written_keeps_its_row(tmp_path, video):
    """Its times and cluster are still true; one unreadable clip must not kill the export.

    The empty file is dropped and the column left blank — blank meaning "no clip", which is
    different from a name pointing at a file that is not there.
    """
    rows = [_instance("r1", "c01", 2, 6), _instance("r1", "c01", 100, 104)]
    dest, _written = _export(tmp_path, video, rows)
    table = _rows(dest / "clips.csv")
    assert len(table) == 2
    videos = [r["video_overlay"] for r in table]
    assert videos[0] and not videos[1]
    assert not (dest / "c01_r1_000100-000104.mp4").exists()


# --------------------------------------------------------------------------- #
# flavours and framing
# --------------------------------------------------------------------------- #
def test_every_requested_flavour_gets_its_own_column_and_file(tmp_path, video):
    rows = [_instance("r1", "c01", 2, 6)]
    dest, _written = _export(tmp_path, video, rows, renders=clip_render.RENDERS)
    table = _rows(dest / "clips.csv")
    for render in clip_render.RENDERS:
        assert table[0][f"video_{render}"] == f"c01_r1_000002-000006_{render}.mp4"
        assert (dest / table[0][f"video_{render}"]).is_file()


def test_the_full_field_of_view_is_the_default(tmp_path, video):
    """Clips in the full field of view were asked for by name, so that is the default.

    The fixture's frame is 64px square and the crop is 32px, so the two framings are
    distinguishable by the decoded size — assert the picture, not the flag.
    """
    import cv2

    rows = [_instance("r1", "c01", 2, 6)]
    dest, _written = _export(tmp_path, video, rows, size=64)
    cap = cv2.VideoCapture(str(dest / "c01_r1_000002-000006.mp4"))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    assert ok
    assert frame.shape[:2] == (64, 64), "the default is the whole frame, not a crop"


def test_the_crop_toggle_reaches_the_renderer(tmp_path, video):
    import cv2

    rows = [_instance("r1", "c01", 2, 6)]
    dest, _written = _export(tmp_path, video, rows, crop=True, size=32)
    cap = cv2.VideoCapture(str(dest / "c01_r1_000002-000006.mp4"))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    assert ok
    assert frame.shape[:2] == (32, 32)


# --------------------------------------------------------------------------- #
# scope and caps
# --------------------------------------------------------------------------- #
def test_max_per_cluster_caps_each_cluster_not_the_whole_export(tmp_path, video):
    """A cap on the total would take whole clusters off the end; the point is a sample of each."""
    rows = [_instance("r1", "c01", i * 3, i * 3 + 1) for i in range(4)]
    rows += [_instance("r1", "c02", 20 + i * 3, 21 + i * 3) for i in range(4)]
    dest, _written = _export(tmp_path, video, rows, max_per_cluster=2)
    table = _rows(dest / "clips.csv")
    per_cluster = {}
    for row in table:
        per_cluster[row["source_label"]] = per_cluster.get(row["source_label"], 0) + 1
    assert per_cluster == {"c01": 2, "c02": 2}


def test_no_instances_writes_nothing_at_all(tmp_path, video):
    dest, written = _export(tmp_path, video, [])
    assert written == []
    assert not (
        dest / "clips.csv"
    ).exists(), (
        "an index with no rows is worse than no index — it reads as an export that ran"
    )


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
def test_one_decode_serves_every_bout_of_a_part(tmp_path, video, monkeypatch):
    """The property that makes a whole-result export usable rather than minutes per clip.

    A per-bout seek re-decodes from the preceding keyframe, so it would produce identical
    clips and take orders of magnitude longer — only counting the opens can catch it.
    """
    import cv2

    opens = []
    real = cv2.VideoCapture

    def counting(*args, **kwargs):
        opens.append(args[0] if args else None)
        return real(*args, **kwargs)

    monkeypatch.setattr(cv2, "VideoCapture", counting)
    rows = [_instance("r1", "c01", i * 4, i * 4 + 2) for i in range(5)]
    _export(tmp_path, video, rows)
    assert len(opens) == 1, f"one decode per part, got {len(opens)}"


def test_two_subjects_in_one_recording_are_rendered_separately(tmp_path, video):
    """`render_part` draws one track, so the grouping key has to carry the subject.

    Grouping on the recording alone would draw one animal's mask on the other's clips —
    plausible output, no error, and wrong.
    """
    rows = [
        _instance("r1", "c01", 2, 6, subject=0),
        _instance("r1", "c01", 2, 6, subject=1),
    ]
    dest, _written = _export(tmp_path, video, rows)
    table = _rows(dest / "clips.csv")
    assert sorted(r["subject"] for r in table) == ["0", "1"]
    # Subject 1 has no detection on the span, so its clip is empty and dropped — and dropped by
    # its own name: one shared name deleted subject 0's clip too, leaving only clips.csv (#29).
    by_subject = {r["subject"]: r["video_overlay"] for r in table}
    assert by_subject["1"] == ""
    assert by_subject["0"] and (dest / by_subject["0"]).is_file()


def test_the_progress_callback_moves_within_a_single_recording(tmp_path, video):
    """A bar pinned to the recording boundary sits at 0 for a whole one-recording export.

    So the fixture is cut to **one** recording and one part: whatever still moves the bar is
    per clip, which is where the work actually is.
    """
    seen = []
    rows = [_instance("r1", "c01", i * 4, i * 4 + 2) for i in range(4)]
    _export(
        tmp_path,
        video,
        rows,
        progress=lambda stage, fraction, detail: seen.append(fraction),
    )
    intermediate = [f for f in seen if 0.0 < f < 1.0]
    assert intermediate, f"the bar never moved between start and finish: {seen}"
    assert seen[-1] == 1.0


def test_clip_playback_rate_follows_the_bout_not_a_nominal_thirty(tmp_path, video):
    """A bout's true rate, so it plays at the speed the behavior happened."""
    rows = [_instance("r1", "c01", 0, 30)]
    dest, _written = _export(tmp_path, video, rows)
    import cv2

    cap = cv2.VideoCapture(str(dest / "c01_r1_000000-000030.mp4"))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    # 30 frames over 30ms of t_ns is 1000 fps, clamped to the renderer's 120 ceiling.
    assert fps == pytest.approx(120.0, abs=1.0)


def test_instances_are_not_reordered_by_the_export(tmp_path, video):
    """clips.csv follows the artifact's order, so a caller that ranked its rows keeps the rank."""
    rows = [
        _instance("r1", "c02", 10, 12),
        _instance("r1", "c01", 2, 4),
        _instance("r1", "c02", 20, 22),
    ]
    dest, _written = _export(tmp_path, video, rows)
    table = _rows(dest / "clips.csv")
    assert [r["source_label"] for r in table] == ["c02", "c01", "c02"]
    assert [r["start_frame"] for r in table] == ["10", "2", "20"]


def test_a_mask_only_export_still_writes_its_index(tmp_path, video):
    """The mask alone is kept so a user can build a custom figure from it later — which only works if the rows saying *which bout* each mask is come with it."""
    rows = [_instance("r1", "c01", 2, 6)]
    dest, _written = _export(tmp_path, video, rows, renders=(clip_render.MASK,))
    table = _rows(dest / "clips.csv")
    assert table[0]["video_mask"] == "c01_r1_000002-000006.mp4"
    assert "video_overlay" not in table[0], "only the flavours asked for get a column"


def test_the_returned_paths_are_what_was_written(tmp_path, video):
    """The writer records these in export.yaml, so a path it never wrote would be deleted
    from a user's folder on the next re-export."""
    rows = [_instance("r1", "c01", 2, 6)]
    dest, written = _export(tmp_path, video, rows, renders=clip_render.RENDERS)
    assert sorted(p.name for p in written) == sorted(
        p.name for p in dest.iterdir() if p.is_file()
    )
    for path in written:
        assert path.is_file()


def test_a_bout_spanning_no_frames_is_still_indexed(tmp_path, video):
    rows = [_instance("r1", "c01", 5, 5)]
    dest, _written = _export(tmp_path, video, rows)
    assert len(_rows(dest / "clips.csv")) == 1


def test_the_export_is_deterministic(tmp_path, video):
    """Two runs of the same request write the same index — what the CLI/GUI parity rests on."""
    rows = [_instance("r1", "c01", 2, 6), _instance("r1", "c02", 10, 14)]
    first, _ = _export(tmp_path, video, rows)
    first_table = (first / "clips.csv").read_text(encoding="utf-8-sig")
    for path in first.iterdir():
        path.unlink()
    second, _ = _export(tmp_path, video, rows)
    assert (second / "clips.csv").read_text(encoding="utf-8-sig") == first_table


def test_an_unknown_render_is_refused_before_anything_is_written(tmp_path, video):
    rows = [_instance("r1", "c01", 2, 6)]
    with pytest.raises(ValueError, match="unknown clip render"):
        _export(tmp_path, video, rows, renders=("heatmap",))
    assert not list((tmp_path / "out").glob("*.mp4"))


def test_subjects_default_to_zero_when_the_artifact_records_none(tmp_path, video):
    """A §19 producer may leave `subject` as None; the renderer needs an int track id."""
    rows = [_instance("r1", "c01", 2, 6, subject=None)]
    dest, _written = _export(tmp_path, video, rows)
    assert (dest / "c01_r1_000002-000006.mp4").is_file()


def test_a_cluster_name_with_a_slash_cannot_escape_the_folder(tmp_path, video):
    """A cluster's name reaches a filename, and a user types cluster names."""
    rows = [_instance("r1", "../../etc/passwd", 2, 6)]
    dest, _written = _export(tmp_path, video, rows)
    written = list(dest.glob("*.mp4"))
    assert written, "the clip was not written at all"
    assert all(p.parent == dest for p in written)


def test_numpy_frame_values_survive_the_index(tmp_path, video):
    """Frames arrive as numpy ints from the artifact reader; csv must not write `np.int64(2)`."""
    row = _instance("r1", "c01", 2, 6)
    row.start_frame = np.int64(row.start_frame)
    row.end_frame = np.int64(row.end_frame)
    dest, _written = _export(tmp_path, video, [row])
    assert _rows(dest / "clips.csv")[0]["start_frame"] == "2"
