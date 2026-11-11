"""The machine-local video preview cache (§3 *Video preview cache*): build, keys, schema, reads.

Edges taken from the schemas it consumes (``schema-edge-tests``): a variable-rate part (no single
fps maps frame to time), a source slower than the proxy rate, a strided tracking run, an
unassigned detection, an animal in two fragments on one frame, a part whose video is not on this
machine, the same file under two paths, and a run re-made under a reused id.

**Three classes guarded here, each a build that could not tell the user what was wrong.**

- *A blocking wait on a lock another process holds, in a worker whose only channel is its progress
  callback.* ``flock`` parks the thread: no report (the bar sat at 0%), and a cancel — which lands
  at the next report — never lands, so the GUI queue's bounded shutdown returned with the thread
  still running. The rule: a build takes the cache lock through ``build_lock(waiting=…)``, which
  polls and reports "Waiting for another preview build" and raises on cancel. Tested against a lock
  held by another **process** (``held_build_lock``), the case that blocks.
- *A writer that cannot report a failure, trusted as complete.* ``cv2.VideoWriter.write`` returns
  nothing useful on a full disk; a short proxy swapped into place is never rebuilt. The rule: count
  the written file's frames from its packets before the swap.
- *One part's build reading every part.* A mask build resolved every part's video, so another
  part's missing video root failed it; and a mask task after its part's failed proxy decoded the
  whole part again to fail the same way. The rule: a task reads its own part, and a part's failure
  is reported once.
"""

from __future__ import annotations

import shutil

import av
import numpy as np
import pytest

from detail.api import previews
from detail.api.experiment import Experiment
from detail.core import progress as core_progress
from detail.storage import previews as store
from detail.storage import recording as recording_store
from detail.storage import video
from detail.storage import zarr_store


def _decode_all(path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [f.to_ndarray(format="bgr24") for f in container.decode(video=0)]


@pytest.fixture
def vfr_recording(tmp_path, write_vfr_video):
    """An experiment with one recording whose one part is a 60-frame VFR video (10 ms then
    30 ms spacing), imported the real way."""
    source = write_vfr_video(tmp_path / "src" / "clip.mp4", n_frames=60)
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("first")
    recording.add_video_part(source)
    recording.source_path = source
    return experiment, recording


def _timestamps(recording) -> np.ndarray:
    data = recording_store.read_recording(recording.dir)
    return recording_store.load_active_timestamps(recording.dir, data, 0).values


def _key(recording) -> str:
    part = recording.video_parts[0]
    return store.key_for_video(
        recording_store.resolve_part_video_path(recording.dir, part)
    )


# --------------------------------------------------------------------------- #
# frame selection
# --------------------------------------------------------------------------- #
def test_selection_keeps_one_frame_per_fifteenth_of_a_second_at_60fps():
    t_ns = (np.arange(120) * 1_000_000_000 / 60).round().astype(np.int64)
    kept = previews.proxy_frame_selection(t_ns)
    assert kept.tolist() == list(range(0, 120, 4))


def test_selection_keeps_every_frame_of_a_source_slower_than_the_proxy():
    t_ns = np.arange(30, dtype=np.int64) * 100_000_000  # 10 fps
    assert previews.proxy_frame_selection(t_ns).tolist() == list(range(30))


def test_selection_follows_time_not_frame_count_on_a_variable_rate_part():
    t_ns = np.concatenate([np.arange(30) * 10, 300 + np.arange(30) * 30]) * 1_000_000
    kept = previews.proxy_frame_selection(t_ns.astype(np.int64))
    times = t_ns[kept]
    # one frame per slot, and no slot skipped while there were frames in it
    slots = times * 15 // 1_000_000_000
    assert len(set(slots.tolist())) == len(kept)
    assert set(slots.tolist()) == set((t_ns * 15 // 1_000_000_000).tolist())


def test_proxy_size_keeps_aspect_and_never_enlarges():
    assert previews.proxy_size(1920, 1080) == (480, 270)
    assert previews.proxy_size(1280, 1024) == (338, 270)
    assert previews.proxy_size(64, 48) == (64, 48)


# --------------------------------------------------------------------------- #
# proxy
# --------------------------------------------------------------------------- #
def test_proxy_frame_map_matches_the_timestamp_map(vfr_recording, write_vfr_video):
    _experiment, recording = vfr_recording
    assert recording.build_preview().built == 1
    key = _key(recording)
    root = zarr_store.open_group(store.proxy_dir(key) / store.FRAMES_STORE, mode="r")
    assert next(iter(root.attrs)) == "schema_version"
    assert root.attrs["schema_version"] == 1
    assert list(root.array_keys()) == ["src_frame"]
    assert root["src_frame"].dtype == np.int64
    assert root["src_frame"].chunks == (store.FRAME_CHUNK,)
    src = root["src_frame"][:]
    timestamps = _timestamps(recording)
    assert src.tolist() == previews.proxy_frame_selection(timestamps).tolist()
    assert root.attrs["n_source_frames"] == len(timestamps)
    assert (root.attrs["width"], root.attrs["height"]) == (64, 48)

    decoded = _decode_all(store.proxy_dir(key) / store.PROXY_FILE)
    assert len(decoded) == len(src)
    # Each proxy frame is the picture of the source frame it says it is.
    for proxy_frame, source_frame in enumerate(src):
        want = write_vfr_video.frame_bgr(int(source_frame), 48, 64)[0, 0].astype(int)
        got = decoded[proxy_frame][24, 32].astype(int)
        assert np.abs(got - want).max() <= 12, (proxy_frame, source_frame)


def test_a_built_proxy_is_not_missing_and_is_not_rebuilt(vfr_recording):
    _experiment, recording = vfr_recording
    assert [t.kind for t in recording.missing_previews()] == ["proxy"]
    recording.build_preview()
    assert recording.missing_previews() == []
    assert recording.build_preview().built == 0


def test_the_same_file_under_another_path_shares_one_proxy(vfr_recording, tmp_path):
    experiment, recording = vfr_recording
    recording.build_preview()
    copy = tmp_path / "elsewhere" / "renamed.mp4"
    copy.parent.mkdir()
    shutil.copy2(recording.source_path, copy)
    assert store.key_for_video(copy) == _key(recording)
    second = experiment.add_recording("second")
    second.add_video_part(copy, storage="reference")
    assert second.missing_previews() == []


def test_a_changed_file_gets_a_new_key(vfr_recording, tmp_path, write_vfr_video):
    _experiment, recording = vfr_recording
    other = write_vfr_video(tmp_path / "other.mp4", n_frames=40)
    assert store.key_for_video(other) != store.key_for_video(recording.source_path)


def test_clear_then_rebuild(vfr_recording):
    _experiment, recording = vfr_recording
    recording.build_preview()
    size = previews.preview_cache_size()
    assert size > 0
    assert previews.clear_preview_cache() == size
    assert previews.preview_cache_size() == 0
    assert previews.preview_cache_dir().is_dir()  # the folder itself is kept
    assert [t.kind for t in recording.missing_previews()] == ["proxy"]
    assert recording.build_preview().built == 1
    assert recording.missing_previews() == []


def test_a_part_whose_video_is_not_here_is_skipped(vfr_recording):
    _experiment, recording = vfr_recording
    part = recording.video_parts[0]
    recording_store.resolve_part_video_path(recording.dir, part).unlink()
    assert recording.missing_previews() == []
    with pytest.raises(previews.PreviewMissingError, match="not on this machine"):
        previews.PreviewReader.open(recording, 0)


def test_a_failed_build_leaves_nothing_behind(vfr_recording):
    _experiment, recording = vfr_recording

    def cancel(stage, fraction, detail):
        if fraction > 0:
            raise core_progress.OperationCancelled(stage)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core_progress, "PROGRESS_INTERVAL_S", 0.0)
        with pytest.raises(core_progress.OperationCancelled):
            recording.build_preview(progress=cancel)
    proxies = previews.preview_cache_dir() / store.PROXIES_DIR
    assert [p.name for p in proxies.iterdir()] == []
    assert [t.kind for t in recording.missing_previews()] == ["proxy"]


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
def test_progress_moves_within_a_single_part(vfr_recording, monkeypatch):
    _experiment, recording = vfr_recording
    monkeypatch.setattr(core_progress, "PROGRESS_INTERVAL_S", 0.0)
    events = []
    recording.build_preview(progress=lambda *e: events.append(e))
    fractions = [f for _s, f, _d in events]
    assert any(0 < f < 1 for f in fractions), fractions
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert events[0][2] == "proxy · first · part 1 of 1"


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #
def _with_video(built, recording, write_vfr_video, n_frames=60):
    """Give a ``processed_recording`` recording a real 40x40 VFR part and its map."""
    path = write_vfr_video(
        recording.dir / "video" / "part_001.mp4", n_frames, width=40, height=40
    )
    t_ns = video.generate_timestamps(path, 50.0)
    recording_store.write_timestamps_csv(
        recording.dir / "video" / "timestamps" / "generated" / "part_001.csv", t_ns
    )
    return path


def _tracked_recording(processed_recording, write_vfr_video):
    built = processed_recording
    sq = built.square
    entries = []
    for frame in range(60):
        c0 = 2 + frame // 4
        entries.append(
            ((frame, 0), (c0 / 40, 0.1, (c0 + 10) / 40, 0.4), sq(4, 16, c0, c0 + 10), 0)
        )
    entries.append(((0, 1), (0.6, 0.6, 0.9, 0.9), sq(24, 36, 24, 36), 1))
    entries.append(((0, 2), (0.0, 0.8, 0.1, 0.9), sq(32, 36, 0, 4), -1))
    entries.append(((8, 1), (0.0, 0.0, 0.1, 0.1), sq(0, 4, 0, 4), 0))
    entries.sort(key=lambda e: e[0])
    recording = built.add_recording(
        "recording_002",
        detections=[(f, i, box, mask) for (f, i), box, mask, _oid in entries],
        object_ids=[oid for *_rest, oid in entries],
    )
    _with_video(built, recording, write_vfr_video)
    return built, recording, entries


def test_masks_schema_and_rows(processed_recording, write_vfr_video):
    built, recording, entries = _tracked_recording(processed_recording, write_vfr_video)
    kinds = [(t.kind, t.tracking_run_id) for t in recording.missing_previews()]
    assert kinds == [("proxy", None), ("masks", "track_01")]
    assert recording.build_preview().built == 2
    key = _key(recording)
    data = recording_store.read_recording(recording.dir)
    entry = data["tracking_runs"][0]
    detection = data["detection_runs"][0]
    path = store.masks_path(key, "track_01", store.masks_digest(entry, detection))
    root = zarr_store.open_group(path, mode="r")
    assert next(iter(root.attrs)) == "schema_version"
    assert root.attrs["tracking_run_id"] == "track_01"
    assert root.attrs["built_from"] == store.masks_digest(entry, detection)
    assert sorted(root.array_keys()) == ["bbox", "mask_rle", "object_id", "proxy_frame"]
    assert (root["proxy_frame"].dtype, root["proxy_frame"].chunks) == (
        np.int64,
        (store.MASK_ROW_CHUNK,),
    )
    assert (root["object_id"].dtype, root["object_id"].chunks) == (
        np.int32,
        (store.MASK_ROW_CHUNK,),
    )
    assert (root["bbox"].dtype, root["bbox"].chunks) == (
        np.float32,
        (store.MASK_ROW_CHUNK, 4),
    )
    assert root["mask_rle"].chunks == (store.MASK_RLE_CHUNK,)
    assert root["mask_rle"].dtype.kind == "T"  # numpy StringDType, zarr v3 "str"

    rows = store.read_masks(path)
    frames = store.read_frames(key)
    # animal 0 on every proxy frame, animal 1 only where proxy frame 0 is source frame 0,
    # the unassigned row nowhere
    assert set(rows.object_id.tolist()) == {0, 1}
    assert (rows.object_id == 0).sum() == len(frames.src_frame)
    assert rows.proxy_frame[rows.object_id == 1].tolist() == [0]
    assert np.all(np.diff(rows.proxy_frame) >= 0)
    from detail.core.utils import decode_mask

    for row in np.flatnonzero(rows.object_id == 0):
        source = int(frames.src_frame[rows.proxy_frame[row]])
        want = np.zeros((40, 40), bool)
        for (f, _i), _box, mask, oid in entries:
            if f == source and oid == 0:
                want |= mask.astype(bool)
        got = decode_mask(str(rows.mask_rle[row]), (40, 40))
        assert (got == want).all(), source


def test_a_rerun_under_the_same_id_is_missing_again(
    processed_recording, write_vfr_video
):
    built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    data = recording_store.read_recording(recording.dir)
    data["tracking_runs"][0]["created_at"] = "2030-01-01T00:00:00Z"
    recording_store.write_recording(recording.dir, data)
    assert [(t.kind, t.tracking_run_id) for t in recording.missing_previews()] == [
        ("masks", "track_01")
    ]


def test_build_without_masks_builds_the_proxy_only(
    processed_recording, write_vfr_video
):
    built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    assert recording.build_preview(masks=False).built == 1
    assert [t.kind for t in recording.missing_previews()] == ["masks"]


def test_phase_messages_advance_from_proxy_to_masks(
    processed_recording, write_vfr_video, monkeypatch
):
    built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    monkeypatch.setattr(core_progress, "PROGRESS_INTERVAL_S", 0.0)
    events = []
    recording.build_preview(progress=lambda *e: events.append(e))
    phases = []
    for _stage, _fraction, detail in events:
        if not phases or phases[-1] != detail:
            phases.append(detail)
    assert phases == [
        "proxy · recording_002 · part 1 of 1 · 1 of 2",
        "masks track_01 · recording_002 · part 1 of 1 · 2 of 2",
        "2 built",
    ]


def test_experiment_builds_every_recording_under_one_bar(
    processed_recording, write_vfr_video
):
    built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    _with_video(built, built.recording, write_vfr_video)
    experiment = built.experiment
    missing = experiment.missing_previews()
    assert {t.recording_id for t in missing} == {"recording_001", "recording_002"}
    assert experiment.missing_previews(["recording_002"]) == [
        t for t in missing if t.recording_id == "recording_002"
    ]
    events = []
    assert experiment.build_previews(progress=lambda *e: events.append(e)).built == len(
        missing
    )
    assert experiment.missing_previews() == []
    details = " ".join(d for _s, _f, d in events)
    assert "recording_001" in details and "recording_002" in details


# --------------------------------------------------------------------------- #
# reading tiles
# --------------------------------------------------------------------------- #
def test_reader_draws_a_clip_at_tile_size(processed_recording, write_vfr_video):
    built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    reader = previews.PreviewReader.open(recording, 0, "track_01")
    timestamps = _timestamps(recording)
    assert reader.proxy_t_ns.tolist() == timestamps[reader.frames.src_frame].tolist()
    start, end = int(timestamps[10]), int(timestamps[40])
    with reader.clip(0, start, end, tile_px=96) as clip:
        assert clip.proxy_frames[0] == reader.proxy_frame_at(start)
        assert clip.t_ns[0] <= start
        tiles = list(clip)
        assert len(tiles) == len(clip)
        assert all(t.shape == (96, 96, 3) and t.dtype == np.uint8 for t in tiles)
        # the crop follows the animal's boxes, not the whole frame
        x0, y0, x1, y1 = clip.crop
        assert (x1 - x0, y1 - y0) != (40, 40) or x0 > 0
        # the outline is drawn: some pixels carry the mask tint
        tint = np.array(previews.visualize.MASK_TINT)
        assert (np.abs(tiles[-1].astype(int) - tint).sum(axis=2) < 40).any()
        # a jump lands on the same picture the walk drew
        assert np.array_equal(clip.frame_at(3), tiles[3])
        assert np.array_equal(clip.frame_at(len(clip) - 1), tiles[-1])


def test_reader_without_footage_draws_on_white(processed_recording, write_vfr_video):
    built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    reader = previews.PreviewReader.open(recording, 0, "track_01")
    timestamps = _timestamps(recording)
    clip = reader.clip(0, int(timestamps[0]), int(timestamps[5]), 64, footage=False)
    tile = clip.frame_at(0)
    assert (tile == 255).all(axis=2).any()
    assert not (tile == 255).all()


def test_reader_shows_footage_while_masks_build(processed_recording, write_vfr_video):
    """Masks asked for and not built yet: the reader opens footage-only and says so, so a
    montage can play while they build (review finding 7)."""
    built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview(masks=False)
    reader = previews.PreviewReader.open(recording, 0, "track_01")
    assert reader.masks is None and reader.masks_missing
    timestamps = _timestamps(recording)
    with reader.clip(0, int(timestamps[0]), int(timestamps[20]), 32) as clip:
        assert clip.crop == (0, 0, 40, 40)  # no boxes yet: the whole frame
        assert clip.frame_at(1).shape == (32, 32, 3)
    assert not previews.PreviewReader.open(recording, 0).masks_missing
    recording.build_preview()
    assert not previews.PreviewReader.open(recording, 0, "track_01").masks_missing


def test_reader_still_refuses_a_missing_proxy(vfr_recording):
    _experiment, recording = vfr_recording
    with pytest.raises(previews.PreviewMissingError, match="no preview built"):
        previews.PreviewReader.open(recording, 0)


def test_a_recording_with_no_video_needs_nothing(tmp_path):
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("metadata only")
    assert recording.missing_previews() == []
    assert recording.build_preview().built == 0


# --------------------------------------------------------------------------- #
# the build lock, held by another process
# --------------------------------------------------------------------------- #
def test_a_build_waiting_on_another_process_reports_and_can_be_cancelled(
    vfr_recording, held_build_lock
):
    """A CLI build and a GUI build share one cache lock across processes. The one that waits
    says so on its bar, and a cancel reaches it while it waits: a blocking ``flock`` sat at 0%
    with nothing on the bar and swallowed every cancel until the other build finished.
    """
    _experiment, recording = vfr_recording
    waits = []

    def progress(stage, fraction, detail):
        if detail.startswith("Waiting for another preview build"):
            waits.append(detail)
            if len(waits) == 3:
                raise core_progress.OperationCancelled(stage)

    import threading

    raised = []

    def build():
        try:
            recording.build_preview(progress=progress)
        except core_progress.OperationCancelled as exc:
            raised.append(exc)

    thread = threading.Thread(target=build, daemon=True)
    thread.start()
    thread.join(15)
    stuck = thread.is_alive()
    held_build_lock.release()  # a blocked build gets the lock now, so the thread ends
    thread.join(15)
    assert not stuck, "the build sat in the lock: no report, no cancel"
    assert raised and len(waits) == 3
    assert [t.kind for t in recording.missing_previews()] == ["proxy"]


def test_a_build_runs_once_the_other_process_lets_go(vfr_recording, held_build_lock):
    _experiment, recording = vfr_recording
    events = []

    def progress(stage, fraction, detail):
        events.append(detail)
        if len(events) == 2:
            held_build_lock.release()

    assert recording.build_preview(progress=progress).built == 1
    assert any(d.startswith("Waiting for another preview build") for d in events)


# --------------------------------------------------------------------------- #
# one part's build reads one part
# --------------------------------------------------------------------------- #
def _add_unreachable_part(recording):
    """A second part referenced under a video root this machine has never registered."""
    data = recording_store.read_recording(recording.dir)
    files = data["video"]["files"]
    other = {k: v for k, v in files[0].items() if k != "file"}
    other["id"] = "part_002"
    other["source"] = {"root": "not_on_this_machine", "rel": "elsewhere.mp4"}
    files.append(other)
    recording_store.write_recording(recording.dir, data)


def test_a_part_s_masks_build_while_another_part_s_video_is_not_here(
    processed_recording, write_vfr_video
):
    """Masks for part 1 need part 1's detections and map, not part 2's video: resolving every
    part failed part 1's build on part 2's missing video root."""
    built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    _add_unreachable_part(recording)
    tasks = [t for t in recording.missing_previews() if t.part_index == 0]
    assert [t.kind for t in tasks] == ["proxy", "masks"]
    report = previews.build_tasks(built.experiment.dir, tasks)
    assert report.ok, report.failed
    assert report.built == 2


def test_a_part_whose_proxy_failed_has_its_masks_skipped(
    processed_recording, write_vfr_video, monkeypatch
):
    """The masks are drawn onto the proxy, so a mask build after a failed proxy decodes the
    whole part again to fail the same way. Said once, as skipped, with the reason."""
    built, recording, _ = _tracked_recording(processed_recording, write_vfr_video)
    calls = []
    real = previews._build_proxy

    def broken(*args, **kwargs):
        calls.append(args[1])
        raise OSError("disk unreadable")

    monkeypatch.setattr(previews, "_build_proxy", broken)
    report = recording.build_preview()
    assert calls == [0], "the mask build did not try the proxy again"
    (proxy, why_proxy), (masks, why_masks) = report.failed
    assert (proxy.kind, masks.kind) == ("proxy", "masks")
    assert "disk unreadable" in why_proxy
    assert why_masks.startswith("skipped: this part's proxy failed")
    assert "disk unreadable" in why_masks
    monkeypatch.setattr(previews, "_build_proxy", real)
    assert recording.build_preview().ok


# --------------------------------------------------------------------------- #
# a proxy cut short is not kept
# --------------------------------------------------------------------------- #
def test_a_proxy_the_writer_cut_short_is_not_kept(vfr_recording, monkeypatch):
    """``VideoWriter.write`` raises nothing on a full disk. A short file marked complete is
    never rebuilt, so the written file is counted before it goes into place."""
    _experiment, recording = vfr_recording
    real = video.open_video_writer

    def dropping(path, *args, **kwargs):
        writer, codec = real(path, *args, **kwargs)
        count = [0]

        class Short:
            def write(self, frame):
                count[0] += 1
                if count[0] <= 5:  # the disk fills after five frames
                    writer.write(frame)

            def release(self):
                writer.release()

        return Short(), codec

    monkeypatch.setattr(video, "open_video_writer", dropping)
    report = recording.build_preview()
    ((task, why),) = report.failed
    assert task.kind == "proxy"
    assert "the preview holds 5 of the" in why and "disk" in why
    proxies = previews.preview_cache_dir() / store.PROXIES_DIR
    assert [p.name for p in proxies.iterdir()] == []
    assert [t.kind for t in recording.missing_previews()] == ["proxy"]


def test_a_part_s_proxy_is_sized_from_its_own_video(vfr_recording):
    """The recording's resolution is one number for every part; a part of another shape got a
    proxy stretched to it."""
    _experiment, recording = vfr_recording
    data = recording_store.read_recording(recording.dir)
    data["video"]["resolution"] = [1920, 1080]  # not this part's shape
    recording_store.write_recording(recording.dir, data)
    recording.build_preview()
    frames = store.read_frames(_key(recording))
    width, height = video.probe_video(recording.source_path)["resolution"]
    assert (frames.width, frames.height) == previews.proxy_size(width, height)
    assert (frames.source_width, frames.source_height) == (width, height)
