"""Exporting a montage's clips in several looks and formats at once (``api.montage_export``).

**What is pinned, and why each.**

- **The folder tree is the plan.** An export of two looks × two formats × both writes exactly
  the files ``ExportPlan.files()`` lists — a subfolder per look, ``montage.<ext>`` and one
  ``rank_recording_m-ss.t.<ext>`` per clip — and the summary's count is that number. Read back:
  each file's width, rate and frame count are what was asked.
- **Full size is the crop's own size**, not a default tile: a per-clip mp4 is as wide as its
  crop in the recording's pixels.
- **Cancel keeps the finished files and removes the one being written.**
- **The looks**: their names, folders and layers, the current ticks' name following the ticks,
  and the refusals.
- **The bar moves inside a single file**, on a one-clip, one-look, one-format export — the case
  a count of files cannot move.
"""

from __future__ import annotations

import cv2
import pytest
from PIL import Image

from detail.api import clip_montage as cm
from detail.api import montage_export as me
from detail.api.animation import GIF, MP4
from tests.api.test_clip_views import _walking

STEP = int(1e9 / 30)


def _clips(recording, spans=((0, 4), (1, 2))):
    return [
        cm.MontageClip(i + 1, recording.id, "mouse_07", 0, "emb_01", a * STEP, b * STEP)
        for i, (a, b) in enumerate(spans)
    ]


def _montage(tmp_path, spans=((0, 4), (1, 2)), **walking):
    recording = _walking(tmp_path, **walking)
    clips = _clips(recording, spans)
    return cm.ClusterMontage(recording.dir.parent.parent, clips, workers=1)


def _plan(montage, keys, formats, write=me.WRITE_BOTH, current=cm.DEFAULT_LAYERS):
    looks = me.resolve_looks(keys, current=current)
    return me.plan_for(montage.clips, looks, formats, write, blind=False)


# --- the looks ----------------------------------------------------------------------------- #
def test_the_fixed_looks_are_the_users_list_in_order():
    assert [v.label for v in me.VISUALS] == [
        "Raw footage",
        "Raw + mask outline",
        "Raw + mask outline + mask trail",
        "Raw + keypoints",
        "Raw + keypoints + keypoint trail",
        "Mask",
        "Mask + mask trail",
        "Keypoints",
        "Keypoints + keypoint trail",
        "Mask + keypoints + both trails",
    ]
    assert me.VISUALS_BY_KEY["raw"].layers() == cm.MontageLayers(
        raw=True, mask=False, keypoints=False, mask_trail=False
    )
    assert me.VISUALS_BY_KEY["raw-outline"].layers() == cm.MontageLayers(
        raw=True, mask=True, mask_trail=False
    )
    assert [v.key for v in me.VISUALS if v.needs_keypoints] == [
        "raw-keypoints",
        "raw-keypoints-trail",
        "keypoints",
        "keypoints-trail",
        "mask-keypoints-trails",
    ]


def test_the_current_look_is_named_for_the_ticks():
    assert me.describe_layers(cm.DEFAULT_LAYERS) == "Mask + Mask trail"
    assert (
        me.describe_layers(cm.MontageLayers(raw=True, mask=True, mask_trail=False))
        == "Raw + Mask"
    )
    trail = cm.MontageLayers(
        mask=False, keypoints=True, keypoint_trail=True, trail_points=("nose", "tail")
    )
    assert me.describe_layers(trail) == "Keypoints + Keypoint trail: nose, tail"
    many = cm.MontageLayers(
        mask=False, keypoints=True, keypoint_trail=True, trail_points=tuple("abcd")
    )
    assert me.describe_layers(many, 8) == "Keypoints + Keypoint trail: 4 of 8 points"
    (look,) = me.resolve_looks([me.CURRENT], current=cm.DEFAULT_LAYERS)
    assert look.name == "Mask + Mask trail"
    assert look.folder == "current_Mask_Mask_trail"
    assert look.layers == cm.DEFAULT_LAYERS


def test_looks_come_in_the_dialogs_order_with_one_folder_each():
    looks = me.resolve_looks(
        ["mask", me.CURRENT, "raw-outline"], current=cm.DEFAULT_LAYERS
    )
    assert [look.folder for look in looks] == [
        "current_Mask_Mask_trail",
        "Raw_mask_outline",
        "Mask",
    ]


def test_trail_points_reach_every_keypoint_trail():
    current = cm.MontageLayers(mask=False, keypoints=True, keypoint_trail=True)
    looks = me.resolve_looks(
        [me.CURRENT, "keypoints-trail", "raw-outline"],
        current=current,
        trail_points=["nose"],
    )
    # In the dialog's order: current, Raw + mask outline, Keypoints + keypoint trail.
    assert [look.layers.trail_points for look in looks] == [("nose",), None, ("nose",)]


@pytest.mark.parametrize(
    "keys, kwargs, match",
    [
        ([], {}, "at least one visual"),
        (["sparkles"], {}, "unknown visual"),
        (["mask", "mask"], {}, "once"),
        ([me.CURRENT], {}, "ticks"),
    ],
)
def test_bad_looks_are_refused(keys, kwargs, match):
    with pytest.raises(ValueError, match=match):
        me.resolve_looks(keys, **kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"kind": "webm"}, "mp4 or gif"),
        ({"kind": GIF}, "width"),
        ({"kind": MP4, "width": 10}, "out of range"),
        ({"kind": MP4, "fps": 0.0}, "out of range"),
    ],
)
def test_bad_formats_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        me.ExportFormat(**kwargs)


def test_the_default_formats_are_a_full_size_mp4_and_a_640_px_gif():
    assert me.DEFAULT_FORMATS == (
        me.ExportFormat(MP4, None, None),
        me.ExportFormat(GIF, 640, 12.0),
    )
    assert me.DEFAULT_WRITE == me.WRITE_BOTH
    assert me.ExportFormat(MP4).resolved_fps(29.97) == 29.97, "the footage's own rate"
    assert me.ExportFormat(GIF, 640).resolved_fps(29.97) == 12.0


# --- the plan ------------------------------------------------------------------------------ #
def test_the_plan_lists_every_file_and_the_summary_counts_them(tmp_path):
    montage = _montage(tmp_path)
    plan = _plan(montage, [me.CURRENT, "raw-outline"], me.DEFAULT_FORMATS)
    names = [f.relpath for f in plan.files()]
    assert names == [
        f"{folder}/{stem}.{ext}"
        for folder in ("current_Mask_Mask_trail", "Raw_mask_outline")
        for ext in ("mp4", "gif")
        for stem in ("montage", "1_mouse_07_0-00.0", "2_mouse_07_0-00.0")
    ]
    assert me.summary_text(plan) == (
        "Will write 2 visuals × 2 formats × (1 montage + 2 clips) = 12 files into a new "
        "subfolder per visual"
    )
    only = _plan(montage, ["mask"], me.DEFAULT_FORMATS[1:], me.WRITE_MONTAGE)
    assert me.summary_text(only).startswith(
        "Will write 1 visual × 1 format × 1 montage = 1 file"
    )
    assert len(only.files()) == 1


def test_a_plan_refuses_what_it_cannot_write(tmp_path):
    montage = _montage(tmp_path)
    looks = me.resolve_looks(["mask"])
    with pytest.raises(ValueError, match="at least one format"):
        me.plan_for(montage.clips, looks, (), blind=False)
    with pytest.raises(ValueError, match="no clips"):
        me.plan_for([], looks, me.DEFAULT_FORMATS, blind=False)
    with pytest.raises(ValueError, match="same file name"):
        me.ExportPlan(
            looks, me.DEFAULT_FORMATS, me.WRITE_BOTH, tuple(montage.clips),
            ("a", "b"), ("same", "same"),
        )  # fmt: skip


def test_blind_names_hide_the_recording(tmp_path):
    montage = _montage(tmp_path)
    plan = me.plan_for(
        montage.clips, me.resolve_looks(["mask"]), me.DEFAULT_FORMATS, blind=True
    )
    assert plan.stems == ("1_recording-A_0-00.0", "2_recording-A_0-00.0")
    assert plan.labels[0] == "#1 · recording A · 0:00.0"


# --- full size ----------------------------------------------------------------------------- #
def test_full_size_is_each_crops_own_size_in_the_recording(tmp_path):
    montage = _montage(tmp_path)
    native = me.native_tile_px(montage)
    boxes = montage.clip_boxes(0)
    x0, y0, x1, y1 = cm.crop_box(boxes, (64, 48))
    assert native[montage.clips[0]] == max(x1 - x0, y1 - y0)
    plan = _plan(montage, ["mask"], [me.ExportFormat(MP4)], me.WRITE_CLIPS)
    first = plan.files()[0]
    width, _height, tile = me.file_size(first, plan, native)
    assert tile == native[montage.clips[0]]
    assert width == tile + 2 * cm.GAP_PX


def test_a_full_size_montage_stays_within_the_widest_frame(tmp_path):
    montage = _montage(tmp_path)
    clips = list(montage.clips) * 1
    native = {c: 5000 for c in clips}
    plan = _plan(montage, ["mask"], [me.ExportFormat(MP4)], me.WRITE_MONTAGE)
    width, _h, tile = me.file_size(plan.files()[0], plan, native)
    assert width <= cm.EXPORT_MAX_WIDTH
    assert tile < 5000


# --- writing ------------------------------------------------------------------------------- #
def _frames(path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        count = 0
        while cap.read()[0]:
            count += 1
        return count
    finally:
        cap.release()


def test_two_looks_two_formats_and_both_write_the_planned_tree(tmp_path):
    montage = _montage(tmp_path)
    formats = (me.ExportFormat(MP4, 120, 15.0), me.ExportFormat(GIF, 96, 10.0))
    plan = _plan(montage, [me.CURRENT, "raw-outline"], formats)
    out = tmp_path / "export"
    heard = []
    events = []
    written = me.export_clips(
        montage,
        plan,
        out,
        progress=lambda *e: events.append(e),
        on_written=heard.append,
    )
    expected = [out / f.relpath for f in plan.files()]
    assert written == heard == expected
    on_disk = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.*"))
    assert on_disk == sorted(f.relpath for f in plan.files())
    assert len(on_disk) == 12 == int(me.summary_text(plan).split(" = ")[1].split()[0])
    for path in expected:
        if path.suffix == ".gif":
            with Image.open(path) as gif:
                assert gif.size[0] == 96
                assert gif.info["duration"] == 100
                # 10 fps over the montage's 133 ms, the first clip's 133 ms, the second's 33.
                want = 1 if path.name.startswith("2_") else 2
                assert gif.n_frames == want, path.name
        else:
            cap = cv2.VideoCapture(str(path))
            try:
                assert round(cap.get(cv2.CAP_PROP_FPS)) == 15
                assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 120
            finally:
                cap.release()
            want = 1 if path.name.startswith("2_") else 3
            assert _frames(path) == want, path.name
    fractions = [f for stage, f, _d in events if stage == "exporting"]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0


def test_one_file_alone_still_moves_the_bar(tmp_path):
    """The cadence trap: one look, one format, one clip — only the frame loop can move it."""
    full = _montage(tmp_path)
    montage = full.subset(full.clips[:1])
    plan = _plan(montage, ["mask"], [me.ExportFormat(GIF, 64, 30.0)], me.WRITE_CLIPS)
    events = []
    me.export_clips(
        montage, plan, tmp_path / "one", progress=lambda *e: events.append(e)
    )
    inner = [f for stage, f, _d in events if stage == "exporting" and 0 < f < 1]
    assert inner, events
    assert [s for s, _f, _d in events][0] == "preparing clips", "phases are named"


def test_a_cancelled_export_keeps_finished_files_and_removes_the_partial(tmp_path):
    from detail.core.progress import OperationCancelled

    montage = _montage(tmp_path)
    plan = _plan(montage, ["mask"], [me.ExportFormat(MP4, 64, 30.0)], me.WRITE_BOTH)
    heard = []
    second = []
    partial = []
    out = tmp_path / "stopped"

    def progress(stage, fraction, detail):
        if stage == "exporting" and "file 2 of" in detail:
            second.append(fraction)
            if len(second) == 3:  # the clip's mp4 is open and holds a frame or two
                partial.append((out / "Mask" / "1_mouse_07_0-00.0.mp4").exists())
                raise OperationCancelled()

    with pytest.raises(OperationCancelled):
        me.export_clips(montage, plan, out, progress=progress, on_written=heard.append)
    assert partial == [
        True
    ], "the cancel landed while the second file was being written"
    assert [p.relative_to(out).as_posix() for p in heard] == ["Mask/montage.mp4"]
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.*")) == [
        "Mask/montage.mp4"
    ], "the partly written clip file was removed"


def test_a_full_size_mp4_is_as_wide_as_the_crop(tmp_path):
    montage = _montage(tmp_path)
    plan = _plan(montage, ["raw"], [me.ExportFormat(MP4, None, 15.0)], me.WRITE_CLIPS)
    native = me.native_tile_px(montage)
    written = me.export_clips(montage, plan, tmp_path / "full")
    cap = cv2.VideoCapture(str(written[0]))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        cap.release()
    want = native[montage.clips[0]] + 2 * cm.GAP_PX
    assert width == want // 2 * 2, "H.264 keeps an even width"


# --- the estimate -------------------------------------------------------------------------- #
def test_the_estimate_counts_the_planned_files_and_grows_with_them(tmp_path):
    montage = _montage(tmp_path)
    native = me.native_tile_px(montage)
    one = _plan(montage, ["mask"], me.DEFAULT_FORMATS[1:], me.WRITE_MONTAGE)
    many = _plan(montage, [me.CURRENT, "raw"], me.DEFAULT_FORMATS)
    small = me.estimate_export(montage, one, native)
    large = me.estimate_export(montage, many, native)
    assert small.files == 1 and large.files == len(many.files()) == 12
    assert 0 < small.bytes < large.bytes and 0 < small.seconds < large.seconds
    text = me.summary_text(many, large)
    assert (
        "= 12 files" in text and " MB" in text and ("min" in text or "minute" in text)
    )


def test_footage_and_trails_cost_more_than_masks_alone():
    costs = me.DEFAULT_COSTS
    masks = costs.tile_s(cm.MontageLayers(mask_trail=False), 224)
    assert costs.tile_s(cm.DEFAULT_LAYERS, 224) > masks, "a trail costs something"
    assert costs.tile_s(cm.MontageLayers(raw=True, mask=False), 224) > masks
    assert costs.tile_s(cm.DEFAULT_LAYERS, 640) > costs.tile_s(cm.DEFAULT_LAYERS, 224)
    assert costs.encode_s_per_mpix("gif", True) > costs.encode_s_per_mpix("gif", False)


def test_sizes_and_times_read_as_approximate():
    assert me.about_size(180e6) == "about 180 MB"
    assert me.about_size(2.5e6) == "about 2.5 MB"
    assert me.about_size(1.4e9) == "about 1.4 GB"
    assert me.about_size(3e5) == "under 1 MB"
    assert me.about_time(20) == "under a minute"
    assert me.about_time(125) == "about 2 min"
    assert me.about_time(4200) == "about 1 h 10 min"


# --- keypoints, and the scripted twin -------------------------------------------------- #
def test_keypoints_are_found_only_where_the_embedding_read_them(tmp_path):
    with_points = _montage(tmp_path / "a")
    without = _montage(tmp_path / "b", keypoints=False)
    assert me.keypoints_available(with_points.experiment_dir, with_points.clips)
    assert not me.keypoints_available(without.experiment_dir, without.clips)


def test_the_scripted_twin_writes_the_windows_defaults(tmp_path):
    recording = _walking(tmp_path)
    members = [
        {"recording_id": recording.id, "subject": 0, "start_t_ns": a * STEP,
         "end_t_ns": b * STEP}
        for a, b in ((0, 4), (1, 2))
    ]  # fmt: skip

    class _Run:
        id = "an_07"

        def typical_members(self, node, limit, progress=None):
            return members if limit is None else members[:limit]

        def state(self):
            class _State:
                @staticmethod
                def embed_run_for(recording_id):
                    return "emb_01"

            return _State()

    class _Experiment:
        dir = recording.dir.parent.parent

        @staticmethod
        def recording_names():
            return {recording.id: "mouse_07"}

    out = tmp_path / "scripted"
    written = me.export_cluster_clips(_Experiment(), _Run(), 5, out)
    assert sorted(p.relative_to(out).as_posix() for p in written) == sorted(
        f"current_Mask_Mask_trail/{stem}.{ext}"
        for ext in ("mp4", "gif")
        for stem in ("montage", "1_mouse_07_0-00.0", "2_mouse_07_0-00.0")
    )
