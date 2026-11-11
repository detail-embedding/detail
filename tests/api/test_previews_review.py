"""The preview cache under the conditions a review of #952 named (§3 *Video preview cache*).

Each case is a failure the first version had and nothing tested: a folder that exists without
being whole, two builds or a build and a clear at the same moment, one bad part in a batch,
leftovers from a killed build, a frame with no timestamp, and the mask edges — a strided run,
a source larger than the proxy, an animal in two fragments.
"""

from __future__ import annotations

import json
import shutil
import threading
from fractions import Fraction

import numpy as np
import pytest

from detail.api import previews
from detail.api.experiment import Experiment
from detail.core import progress as core_progress
from detail.core.utils import decode_mask
from detail.storage import previews as store
from detail.storage import recording as recording_store
from detail.storage import video


@pytest.fixture
def one_recording(tmp_path, write_vfr_video):
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.add_recording("first")
    recording.add_video_part(write_vfr_video(tmp_path / "src.mp4", n_frames=60))
    return experiment, recording


def _key(recording) -> str:
    part = recording.video_parts[0]
    return store.key_for_video(
        recording_store.resolve_part_video_path(recording.dir, part)
    )


def _proxies():
    return previews.preview_cache_dir() / store.PROXIES_DIR


# --------------------------------------------------------------------------- #
# finding 1: a folder that exists is not a folder that is whole
# --------------------------------------------------------------------------- #
def test_a_half_deleted_proxy_folder_is_rebuilt_not_trusted(one_recording):
    """A clear that could not finish leaves ``proxies/<key>/`` without its frame map; that
    folder must not count as built, or the part never gets a preview again."""
    _experiment, recording = one_recording
    recording.build_preview()
    key = _key(recording)
    shutil.rmtree(store.proxy_dir(key) / store.FRAMES_STORE)
    assert [t.kind for t in recording.missing_previews()] == ["proxy"]
    recording.build_preview()
    assert store.proxy_present(key)
    assert recording.missing_previews() == []


def test_a_folder_holding_only_masks_is_rebuilt(one_recording):
    """A mask build that recreated ``proxies/<key>/masks/`` after a clear leaves a folder
    with no proxy in it."""
    _experiment, recording = one_recording
    key = _key(recording)
    (store.proxy_dir(key) / store.MASKS_DIR / "trk-x.zarr").mkdir(parents=True)
    recording.build_preview()
    assert store.proxy_present(key)


def test_clear_waits_for_a_build_in_flight(one_recording, monkeypatch):
    """Clear deletes nothing while a build is writing; it waits for it, then clears."""
    _experiment, recording = one_recording
    monkeypatch.setattr(core_progress, "PROGRESS_INTERVAL_S", 0.0)
    inside, release = threading.Event(), threading.Event()

    def hold(_stage, fraction, _detail):
        if 0 < fraction < 1 and not inside.is_set():
            inside.set()
            assert release.wait(10)

    builder = threading.Thread(target=lambda: recording.build_preview(progress=hold))
    builder.start()
    assert inside.wait(10)
    clearer = threading.Thread(target=previews.clear_preview_cache)
    clearer.start()
    clearer.join(0.5)
    still_waiting = clearer.is_alive()
    release.set()
    builder.join(10)
    clearer.join(10)
    assert still_waiting, "clear ran while a build was writing into the cache"
    assert not _proxies().exists()
    assert [t.kind for t in recording.missing_previews()] == ["proxy"]
    assert recording.build_preview().built == 1


# --------------------------------------------------------------------------- #
# finding 2: two builds at once
# --------------------------------------------------------------------------- #
def test_staging_names_are_unique_per_build(tmp_path):
    """Two builds in one process must not share (and empty) one staging folder."""
    final = tmp_path / "proxies" / "k"
    first, second = store.staging_dir(final), store.staging_dir(final)
    assert first != second and first.is_dir() and second.is_dir()
    assert first.name.endswith(".tmp") and second.name.endswith(".tmp")


def test_two_builds_of_one_file_at_once(one_recording):
    """Builders of the same key at the same moment — an old queue stopping while a new one
    starts, the GUI beside the CLI — all finish, and one proxy results."""
    _experiment, recording = one_recording
    errors = []

    def build():
        try:
            previews.build_proxy(recording, 0)
        except Exception as exc:  # noqa: BLE001 — collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=build) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
    key = _key(recording)
    assert store.proxy_present(key)
    assert sorted(p.name for p in _proxies().iterdir()) == [key]


def test_concurrent_index_writes_do_not_collide(tmp_path, write_vfr_video):
    """``key_for_video`` runs outside the build lock (a page opening a reader beside the
    queue), so its temp file must be unique per write."""
    videos = [write_vfr_video(tmp_path / f"v{i}.mp4", n_frames=4 + i) for i in range(8)]
    index = store.cache_dir() / store.INDEX_FILE
    errors = []

    def keys(path):
        try:
            for _ in range(15):
                store.key_for_video(path)
                index.unlink(missing_ok=True)  # so the next call writes again
        except Exception as exc:  # noqa: BLE001 — collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=keys, args=(v,)) for v in videos]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []


# --------------------------------------------------------------------------- #
# finding 4: a frame with no timestamp
# --------------------------------------------------------------------------- #
class _Frame:
    def __init__(self, pts):
        self.pts = pts

    def to_ndarray(self, **_kwargs):
        return np.zeros((4, 4, 3), np.uint8)


class _Container:
    def __init__(self, pts_list):
        stream = type("S", (), {})()
        stream.time_base = Fraction(1, 1000)
        stream.start_time = 0
        stream.thread_type = None
        self.streams = type("Streams", (), {"video": [stream]})()
        self._pts = pts_list

    def decode(self, _stream):
        return (_Frame(p) for p in self._pts)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_frame_with_no_timestamp_mid_file_is_an_error(tmp_path, monkeypatch):
    """Ending the read there handed back a short proxy as if it were the whole part."""
    path = tmp_path / "v.mp4"
    path.write_bytes(b"x")
    monkeypatch.setattr(video.av, "open", lambda _p: _Container([0, 10, None, 30]))
    timestamps = np.array([0, 10, 20, 30], dtype=np.int64) * 1_000_000
    read = video.read_selected_frames(path, np.arange(4), timestamps=timestamps)
    assert [idx for idx, _ in [next(read), next(read)]] == [0, 1]
    with pytest.raises(ValueError, match="carries no timestamp"):
        next(read)


# --------------------------------------------------------------------------- #
# finding 5: one bad part does not stop the batch
# --------------------------------------------------------------------------- #
@pytest.fixture
def one_broken(tmp_path, write_vfr_video):
    experiment = Experiment.create("p", tmp_path)
    broken = experiment.add_recording("broken")
    broken.add_video_part(write_vfr_video(tmp_path / "a.mp4"))
    good = experiment.add_recording("good")
    good.add_video_part(write_vfr_video(tmp_path / "b.mp4", n_frames=40))
    part = recording_store.resolve_part_video_path(broken.dir, broken.video_parts[0])
    part.write_bytes(b"not a video" * 1000)
    return experiment, broken, good


def test_a_failing_part_is_reported_and_the_batch_goes_on(one_broken):
    experiment, broken, good = one_broken
    report = experiment.build_previews()
    assert report.built == 1 and not report.ok
    assert [(t.recording_id, t.kind) for t, _why in report.failed] == [
        (broken.id, "proxy")
    ]
    assert good.missing_previews() == []
    assert [t.kind for t in broken.missing_previews()] == ["proxy"]


def test_a_cancel_still_stops_the_whole_batch(one_broken, monkeypatch):
    """The per-part boundary must not swallow a cancel: that is the GUI's Stop."""
    experiment, _broken, _good = one_broken
    monkeypatch.setattr(core_progress, "PROGRESS_INTERVAL_S", 0.0)

    def cancel(stage, fraction, detail):
        if fraction > 0:
            raise core_progress.OperationCancelled(stage)

    with pytest.raises(core_progress.OperationCancelled):
        experiment.build_previews(progress=cancel)


# --------------------------------------------------------------------------- #
# finding 6: leftovers of a killed build
# --------------------------------------------------------------------------- #
def test_staging_left_by_a_killed_build_is_swept(one_recording):
    _experiment, recording = one_recording
    key = _key(recording)
    stale = [
        _proxies() / f"{key}.dead.tmp",
        store.proxy_dir(key) / store.MASKS_DIR / "trk-x.zarr.dead.tmp",
    ]
    for folder in stale:
        folder.mkdir(parents=True)
        (folder / "junk").write_bytes(b"x" * 1000)
    recording.build_preview()
    assert [folder.exists() for folder in stale] == [False, False]


def test_a_fresh_index_temp_file_is_left_alone(one_recording):
    """Index temp files are written outside the lock, so only old ones are someone's
    leftovers; a young one may be a write in flight."""
    _experiment, recording = one_recording
    young = store.cache_dir() / f"{store.INDEX_FILE}.inflight.tmp"
    young.parent.mkdir(parents=True, exist_ok=True)
    young.write_text("{}", encoding="utf-8")
    recording.build_preview()
    assert young.exists()


# --------------------------------------------------------------------------- #
# finding 8: masks — a strided run, a larger source, fragments under resize
# --------------------------------------------------------------------------- #
H, W = 360, 640  # the source and its masks; the proxy is 480x270


def _big_tracked_recording(processed_recording, write_vfr_video, *, every_n=1):
    """A 640x360 recording: animal 0 a box moving right, animal 1 a box in two fragments,
    with detection rows only on source frames that are multiples of ``every_n``."""
    built = processed_recording
    entries = []
    for frame in range(0, 60, every_n):
        c0 = 40 + frame * 4
        body = np.zeros((H, W), np.uint8)
        body[60:180, c0 : c0 + 120] = 1
        entries.append(((frame, 0), (c0 / W, 60 / H, (c0 + 120) / W, 180 / H), body, 0))
        left = np.zeros((H, W), np.uint8)
        left[220:300, 400:460] = 1
        right = np.zeros((H, W), np.uint8)
        right[220:300, 470:530] = 1  # a 10 px gap: two fragments of one animal
        entries.append(((frame, 1), (400 / W, 220 / H, 460 / W, 300 / H), left, 1))
        entries.append(((frame, 2), (470 / W, 220 / H, 530 / W, 300 / H), right, 1))
    recording = built.add_recording(
        "big",
        detections=[(f, i, box, mask) for (f, i), box, mask, _oid in entries],
        object_ids=[oid for *_rest, oid in entries],
    )
    path = write_vfr_video(
        recording.dir / "video" / "part_001.mp4", 60, width=W, height=H
    )
    recording_store.write_timestamps_csv(
        recording.dir / "video" / "timestamps" / "generated" / "part_001.csv",
        video.generate_timestamps(path, 50.0),
    )
    data = recording_store.read_recording(recording.dir)
    data["video"]["resolution"] = [W, H]
    data["tracking_runs"][0]["every_n_frames"] = every_n
    recording_store.write_recording(recording.dir, data)
    job = recording.dir / "detections" / "det_01" / "job.json"
    envelope = json.loads(job.read_text(encoding="utf-8-sig"))
    envelope["mask_resolution"] = [H, W]
    job.write_text(json.dumps(envelope), encoding="utf-8")
    return recording


def _built_masks(recording):
    key = _key(recording)
    data = recording_store.read_recording(recording.dir)
    path = store.masks_path(
        key,
        "track_01",
        store.masks_digest(data["tracking_runs"][0], data["detection_runs"][0]),
    )
    return store.read_frames(key), store.read_masks(path)


def test_masks_are_shrunk_from_a_larger_source(processed_recording, write_vfr_video):
    """Masks at 640x360, proxy 480x270: each proxy mask is the source mask at 3/4, and the
    two fragments of animal 1 are one row covering both, with their boxes unioned."""
    recording = _big_tracked_recording(processed_recording, write_vfr_video)
    recording.build_preview()
    frames, rows = _built_masks(recording)
    assert (rows.width, rows.height) == (480, 270)
    for proxy_frame, source in enumerate(frames.src_frame):
        mine = np.flatnonzero(rows.proxy_frame == proxy_frame)
        assert rows.object_id[mine].tolist() == [0, 1]  # one row per animal
        body = decode_mask(str(rows.mask_rle[mine[0]]), (270, 480))
        c0 = 40 + int(source) * 4
        want = np.zeros((270, 480), bool)
        want[45:135, round(c0 * 0.75) : round((c0 + 120) * 0.75)] = True
        assert (body ^ want).sum() <= 2 * 90, int(source)  # one pixel of edge each side
        pair = decode_mask(str(rows.mask_rle[mine[1]]), (270, 480))
        assert pair[166:224, 301:344].all() and pair[166:224, 354:396].all()
        np.testing.assert_allclose(
            rows.bbox[mine[1]], [400 / W, 220 / H, 530 / W, 300 / H], atol=1e-6
        )


@pytest.mark.parametrize("every_n", [2, 4])
def test_a_strided_run_has_its_animal_on_every_proxy_frame(
    processed_recording, write_vfr_video, every_n
):
    """Rows exist on every ``every_n``-th source frame only; each proxy frame takes the
    nearest processed one (``round(src / n) * n``, clamped to the last processed frame).
    """
    recording = _big_tracked_recording(
        processed_recording, write_vfr_video, every_n=every_n
    )
    recording.build_preview()
    frames, rows = _built_masks(recording)
    body = np.flatnonzero(rows.object_id == 0)
    assert rows.proxy_frame[body].tolist() == list(range(len(frames.src_frame)))
    last = 59 - (59 % every_n)
    for row in body:
        source = int(frames.src_frame[rows.proxy_frame[row]])
        nearest = min(int(round(source / every_n)) * every_n, last)
        got = decode_mask(str(rows.mask_rle[row]), (270, 480))
        left_edge = int(np.flatnonzero(got.any(axis=0))[0])
        assert abs(left_edge - round((40 + nearest * 4) * 0.75)) <= 1, (source, nearest)
