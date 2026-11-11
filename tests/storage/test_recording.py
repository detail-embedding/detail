"""``storage/recording.py``: ``recording.yaml``'s writer and reader, the per-recording cache
behind every ``Recording`` property, and the timestamp maps.

**One class guarded here** (``docs/recorded_classes.md``): *a memo whose miss is check,
compute, store*. ``read_recording`` keeps its own mtime-keyed cache, and its miss path was
"look, parse, store" — so every thread that missed one record before the first store parsed it
too. A montage preparing six clips of one recording on six threads parsed its 35 KB
``recording.yaml`` six times, each parse inflated to ~160 ms by the other threads holding the
interpreter, which was most of a cold prepare. The miss path now takes
``read_cache.computing(("recording", key))`` and re-checks under it, so threads that miss one
record together wait for a single parse; a thread wanting another record is not held. The
guard (``test_threads_missing_one_record_together_parse_it_once``) holds six threads inside the
read at once, so a racing cache counts 6 every time rather than when the scheduler happens to
interleave; ``tests/storage/test_read_cache.py`` is the canonical write-up of the class and
applies the same rendezvous to the two shared caches.
"""

import os

import numpy as np
import pytest
import yaml

from detail.core import atomic
from detail.storage import experiment, recording


def test_part_stem_prefers_id_then_falls_back_to_file():
    assert (
        recording.part_stem({"id": "part_002", "file": "video/part_002.mp4"})
        == "part_002"
    )
    assert recording.part_stem({"file": "video/part_005.mp4"}) == "part_005"  # legacy
    assert (
        recording.part_stem({"id": "part_001", "source": {"root": "r", "rel": "a.mp4"}})
        == "part_001"
    )


def test_resolve_part_video_path_copied_is_recording_relative(tmp_path):
    part = {"id": "part_001", "file": "video/part_001.mp4"}
    assert (
        recording.resolve_part_video_path(tmp_path, part)
        == tmp_path / "video/part_001.mp4"
    )


def test_resolve_part_video_path_reference_goes_through_the_root(tmp_path, monkeypatch):
    from detail.storage import video_roots

    root = tmp_path / "nas"
    root.mkdir()
    monkeypatch.setattr(
        video_roots, "video_root", lambda alias: root if alias == "nas" else None
    )
    part = {"id": "part_001", "source": {"root": "nas", "rel": "day3/m27.mp4"}}
    assert (
        recording.resolve_part_video_path(tmp_path, part) == root / "day3" / "m27.mp4"
    )


def test_create_recording_builds_structure(experiment_dir):
    sdir = recording.create_recording(
        experiment_dir, "recording_009", created_at="2026-01-01T00:00:00Z"
    )
    assert sdir == experiment.recording_path(experiment_dir, "recording_009")
    assert (sdir / "recording.yaml").is_file()
    for sub in ("video/timestamps/generated", "detections", "tracking", "embeddings"):
        assert (sdir / sub).is_dir()

    data = recording.read_recording(sdir)
    assert data["id"] == "recording_009"
    assert data["name"] == "recording_009"
    assert data["created_at"] == "2026-01-01T00:00:00Z"
    assert data["schema_version"] == 1


def test_recording_schema_version_is_first_key(experiment_dir):
    sdir = recording.create_recording(experiment_dir, "s1")
    first_line = (
        (sdir / "recording.yaml").read_text(encoding="utf-8-sig").splitlines()[0]
    )
    assert first_line.startswith("schema_version:")


def test_create_recording_twice_raises(experiment_dir):
    recording.create_recording(experiment_dir, "s1")
    with pytest.raises(FileExistsError, match="already exists"):
        recording.create_recording(experiment_dir, "s1")


def test_create_recording_is_discoverable(experiment_dir):
    recording.create_recording(experiment_dir, "s1")
    assert experiment.discover_recordings(experiment_dir) == ["s1"]


def test_read_recording_validates(tmp_path):
    with pytest.raises(FileNotFoundError, match="recording.yaml"):
        recording.read_recording(tmp_path)
    (tmp_path / "recording.yaml").write_text(
        yaml.dump({"schema_version": 2, "id": "s"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema_version"):
        recording.read_recording(tmp_path)
    (tmp_path / "recording.yaml").write_text(
        yaml.dump({"schema_version": 1}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="'id'"):
        recording.read_recording(tmp_path)


def test_read_recording_accepts_conftest_fixture(recording_dir):
    data = recording.read_recording(recording_dir)
    assert data["id"] == "recording_001"
    assert data["video"]["fps_actual"] == 30.0


def test_write_recording_forces_schema_version_first(tmp_path):
    recording.write_recording(tmp_path, {"id": "s", "schema_version": 7, "name": "n"})
    text = (tmp_path / "recording.yaml").read_text(encoding="utf-8-sig")
    assert text.splitlines()[0].startswith("schema_version: 1")
    assert recording.read_recording(tmp_path)["schema_version"] == 1


# --- read_recording cache (perf; mtime-invalidated) --------------------------- #
def test_read_recording_caches_repeated_reads(recording_dir, monkeypatch):
    """A second read of an unchanged file skips the parse (the batch-over-N-recordings
    hot path); only the mtime stat runs."""
    recording.clear_recording_cache()
    calls = {"n": 0}
    real = recording.Path.read_text

    def counting(self, *a, **k):
        if self.name == recording.RECORDING_FILENAME:
            calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(recording.Path, "read_text", counting)
    first = recording.read_recording(recording_dir)
    second = recording.read_recording(recording_dir)
    assert calls["n"] == 1  # second read hit the cache
    assert first == second


def test_read_recording_returns_independent_copy(recording_dir):
    """The cached dict is never handed out directly — mutating one read can't leak
    into the next (or corrupt the cache)."""
    a = recording.read_recording(recording_dir)
    a["injected"] = "boom"
    a["video"]["fps_actual"] = -1
    b = recording.read_recording(recording_dir)
    assert "injected" not in b
    assert b["video"]["fps_actual"] == 30.0


def test_write_recording_invalidates_cache(recording_dir):
    recording.read_recording(recording_dir)  # warm the cache
    data = recording.read_recording(recording_dir)
    data["name"] = "renamed"
    recording.write_recording(recording_dir, data)
    assert recording.read_recording(recording_dir)["name"] == "renamed"


def test_read_recording_reparses_on_external_mtime_change(tmp_path):
    """A background worker (another process) rewriting recording.yaml bumps the mtime,
    so the cache re-reads even without going through write_recording."""
    recording.write_recording(tmp_path, {"id": "s", "name": "one"})
    assert recording.read_recording(tmp_path)["name"] == "one"
    p = tmp_path / "recording.yaml"
    p.write_text(
        yaml.dump({"schema_version": 1, "id": "s", "name": "two"}), encoding="utf-8"
    )
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))  # force a later mtime
    assert recording.read_recording(tmp_path)["name"] == "two"


# --- concurrent-write lock retry (Windows os.replace race, §3) -------------- #
# The retry helper itself is covered in tests/core/test_atomic.py; these check
# that recording read/write route through it.
def test_read_recording_retries_transient_lock(recording_dir, monkeypatch):
    """The GUI thread reading recording.yaml rides out the PermissionError the import
    QThread's os.replace provokes on Windows, instead of crashing the app."""
    real_text = (recording_dir / "recording.yaml").read_text(encoding="utf-8-sig")
    monkeypatch.setattr(atomic.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky_read(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "locked by import worker")
        return real_text

    monkeypatch.setattr(recording.Path, "read_text", flaky_read)
    assert recording.read_recording(recording_dir)["id"] == "recording_001"
    assert calls["n"] == 3


def test_write_recording_retries_replace_on_lock(recording_dir, monkeypatch):
    """The import worker's os.replace rides out the reverse race — a GUI reader
    holding the destination open when the worker swaps the temp file in."""
    real_replace = atomic.os.replace
    monkeypatch.setattr(atomic.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "reader holds the destination")
        return real_replace(src, dst)

    monkeypatch.setattr(atomic.os, "replace", flaky_replace)
    recording.write_recording(recording_dir, {"id": "recording_001", "name": "renamed"})
    assert calls["n"] == 2
    assert recording.read_recording(recording_dir)["name"] == "renamed"


# --- timestamp maps --------------------------------------------------------- #
def test_timestamp_map_round_trip(tmp_path):
    t_ns = np.arange(5, dtype=np.int64) * 33_333_333
    csv_path = tmp_path / "part_001.csv"
    recording.write_timestamps_csv(csv_path, t_ns)

    tsmap = recording.TimestampMap.from_csv(csv_path)
    assert len(tsmap) == 5
    assert tsmap.t_ns(0) == 0
    assert tsmap.t_ns(2) == 2 * 33_333_333
    assert np.array_equal(tsmap.values, t_ns)


def test_timestamp_csv_header(tmp_path):
    csv_path = tmp_path / "part_001.csv"
    recording.write_timestamps_csv(csv_path, np.array([0, 100], dtype=np.int64))
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == "frame_idx,t_ns"
    assert lines[1] == "0,0"
    assert lines[2] == "1,100"


def test_timestamp_map_non_contiguous_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "frame_idx,t_ns\n0,0\n2,66\n", encoding="utf-8"
    )  # missing frame 1
    with pytest.raises(ValueError, match="not contiguous"):
        recording.TimestampMap.from_csv(csv_path)


def test_timestamp_map_empty(tmp_path):
    csv_path = tmp_path / "empty.csv"
    recording.write_timestamps_csv(csv_path, np.array([], dtype=np.int64))
    assert csv_path.read_text(encoding="utf-8-sig").splitlines() == [
        "frame_idx,t_ns"
    ]  # header only
    tsmap = recording.TimestampMap.from_csv(csv_path)
    assert len(tsmap) == 0
    with pytest.raises(IndexError, match="out of range"):
        tsmap.t_ns(0)


def test_timestamp_map_single_row(tmp_path):
    csv_path = tmp_path / "one.csv"
    recording.write_timestamps_csv(csv_path, np.array([42], dtype=np.int64))
    tsmap = recording.TimestampMap.from_csv(csv_path)
    assert len(tsmap) == 1
    assert tsmap.t_ns(0) == 42


def test_timestamp_map_out_of_range_raises(tmp_path):
    csv_path = tmp_path / "part_001.csv"
    recording.write_timestamps_csv(csv_path, np.array([0, 1, 2], dtype=np.int64))
    tsmap = recording.TimestampMap.from_csv(csv_path)
    with pytest.raises(IndexError, match="out of range"):
        tsmap.t_ns(3)
    with pytest.raises(IndexError, match="out of range"):
        tsmap.t_ns(-1)


def test_active_timestamp_resolution_and_load(recording_dir):
    data = recording.read_recording(recording_dir)
    # the fixture points part 0 at video/timestamps/generated/part_001.csv (active=generated)
    csv_path = recording.active_timestamp_csv(recording_dir, data, 0)
    assert csv_path == recording_dir / "video/timestamps/generated/part_001.csv"

    recording.write_timestamps_csv(csv_path, np.arange(4, dtype=np.int64) * 1000)
    tsmap = recording.load_active_timestamps(recording_dir, data, 0)
    assert tsmap.t_ns(3) == 3000


def test_active_timestamp_null_raises(recording_dir):
    data = recording.read_recording(recording_dir)
    data["video"]["files"][0]["timestamps"]["active"] = "user"  # user map is null
    with pytest.raises(ValueError, match="is null"):
        recording.active_timestamp_csv(recording_dir, data, 0)


# --- the one video source, and the legacy-run guard ------------------------ #
def test_video_files_returns_the_recordings_own_parts():
    data = {"video": {"files": [{"index": 0, "file": "video/part_001.mp4"}]}}
    assert recording.video_files(data)[0]["file"] == "video/part_001.mp4"


def test_legacy_downsampling_run_is_refused_not_misread():
    """A run produced on downsampled video counts *its* frames, so reading it against the
    recording's video resolves the wrong ones — silently. It must refuse instead."""
    with pytest.raises(ValueError, match="no longer exists"):
        recording.raise_on_legacy_downsampling(
            {"id": "det_01", "downsampling_run_id": "ds_01"}
        )


def test_a_normal_run_passes_the_legacy_guard():
    # Both shapes a current record can have: the field absent, or present and null.
    recording.raise_on_legacy_downsampling({"id": "det_01", "every_n_frames": 4})
    recording.raise_on_legacy_downsampling(
        {"id": "det_01", "downsampling_run_id": None}
    )
    recording.raise_on_legacy_downsampling(None)


def test_load_active_timestamps_cached_reads_the_map_once(recording_dir):
    data = recording.read_recording(recording_dir)
    recording.write_timestamps_csv(
        recording.active_timestamp_csv(recording_dir, data, 0),
        np.arange(3, dtype=np.int64) * 1000,
    )
    first = recording.load_active_timestamps_cached(recording_dir, data, 0)
    assert first.t_ns(2) == 2000
    # Same object back: an immutable written map, keyed on its mtime.
    assert recording.load_active_timestamps_cached(recording_dir, data, 0) is first


def test_set_name_updates_name(experiment_dir):
    sdir = recording.create_recording(experiment_dir, "recording_001")
    recording.set_name(sdir, "Mouse A")
    assert recording.read_recording(sdir)["name"] == "Mouse A"
    assert recording.read_recording(sdir)["id"] == "recording_001"  # id unchanged


def test_set_name_empty_raises(experiment_dir):
    sdir = recording.create_recording(experiment_dir, "recording_001")
    with pytest.raises(ValueError, match="non-empty"):
        recording.set_name(sdir, "")


def _two_part_recording(experiment_dir):
    """Recording with 2 parts at fps=1: part_001 = 3 frames @ [0,1,2]s, part_002 =
    2 frames @ [3,4]s (global). Returns the recording dir + timestamps dir."""
    sdir = recording.create_recording(experiment_dir, "s")
    ts_dir = sdir / "video" / "timestamps" / "generated"
    ts_dir.mkdir(parents=True, exist_ok=True)
    recording.write_timestamps_csv(
        ts_dir / "part_001.csv", np.array([0, 10**9, 2 * 10**9], dtype=np.int64)
    )
    recording.write_timestamps_csv(
        ts_dir / "part_002.csv", np.array([3 * 10**9, 4 * 10**9], dtype=np.int64)
    )
    data = recording.read_recording(sdir)
    data["video"] = {
        "storage": "default",
        "fps_container": 1.0,
        "fps_actual": 1.0,
        "fps_source": "container",
        "resolution": [10, 10],
        "total_duration_s": 5.0,
        "files": [
            {
                "index": 0,
                "file": "video/part_001.mp4",
                "timestamps": {
                    "generated": "video/timestamps/generated/part_001.csv",
                    "user": None,
                    "active": "generated",
                },
            },
            {
                "index": 1,
                "file": "video/part_002.mp4",
                "timestamps": {
                    "generated": "video/timestamps/generated/part_002.csv",
                    "user": None,
                    "active": "generated",
                },
            },
        ],
    }
    recording.write_recording(sdir, data)
    return sdir, ts_dir


def test_reorder_video_parts_reindexes_and_reoffsets(experiment_dir):
    sdir, ts_dir = _two_part_recording(experiment_dir)
    recording.reorder_video_parts(sdir, [1, 0])

    files = recording.read_recording(sdir)["video"]["files"]
    assert [f["file"] for f in files] == ["video/part_002.mp4", "video/part_001.mp4"]
    assert [f["index"] for f in files] == [0, 1]
    # part_002 (2 frames) now first → offset 0; part_001 (3 frames) after 2s → +2e9
    moved_first = recording.TimestampMap.from_csv(ts_dir / "part_002.csv").values
    moved_second = recording.TimestampMap.from_csv(ts_dir / "part_001.csv").values
    assert list(moved_first) == [0, 10**9]
    assert list(moved_second) == [2 * 10**9, 3 * 10**9, 4 * 10**9]


def test_reorder_video_parts_rejects_non_permutation(experiment_dir):
    sdir, _ = _two_part_recording(experiment_dir)
    with pytest.raises(ValueError, match="permutation"):
        recording.reorder_video_parts(sdir, [0, 0])


# --------------------------------------------------------------------------- #
# the cost of a cached read
# --------------------------------------------------------------------------- #
def _count_stats(call):
    """``(result, number of stat syscalls)`` for ``call``.

    ``os.path.isfile`` and every ``pathlib`` predicate resolve ``os.stat`` at call time, so
    one patch sees both spellings (the idiom in ``tests/storage/test_dirscan.py``).
    """
    from unittest.mock import patch

    real = os.stat
    with patch("os.stat", side_effect=real) as spy:
        result = call()
    return result, spy.call_count


def test_a_cached_read_costs_one_round_trip(recording_dir):
    """The existence probe and the cache's mtime are the same ``stat``.

    The class: **a cache probe that pays two round trips to learn one fact.** Nothing is
    wrong at the line where it is written and no behavioural test can see it — the answer is
    identical either way. It only shows as cost, and only at the scale and latency where the
    read is repeated: this is read once per recording by every listing there is, and it was a
    third of the project rail's whole cost on a workspace held on a network share (#316).
    """
    recording.read_recording(recording_dir)  # warm the mtime cache

    _, stats = _count_stats(lambda: recording.read_recording(recording_dir))

    assert stats == 1, (
        f"{stats} stat calls for one cached recording.yaml — the presence check and the "
        "cache's mtime are one stat, not two"
    )


def test_the_count_would_see_the_shape_this_replaced(recording_dir):
    """Guard the guard: the probe the fix removed costs exactly the stat it asserts about."""
    path = recording_dir / recording.RECORDING_FILENAME
    _, stats = _count_stats(lambda: (path.is_file(), path.stat()))
    assert stats == 2


def test_reading_a_directory_named_like_the_record_is_absent(tmp_path):
    """A regular-file check the single stat still has to make.

    ``is_file()`` refused a *directory* of that name; taking the mtime off a bare ``stat``
    would let one through and move the failure to ``read_text``, as an ``IsADirectoryError``
    from inside the parse rather than the absence this reports.
    """
    (tmp_path / recording.RECORDING_FILENAME).mkdir()

    with pytest.raises(FileNotFoundError, match=recording.RECORDING_FILENAME):
        recording.read_recording(tmp_path)


def test_a_pinned_reader_still_sees_a_real_write(experiment_dir):
    """Through the **real** writer, not a stand-in for it.

    ``read_cache.pinned()`` skips the ``stat`` that would notice a change, which is safe only
    because every writer here drops its own cache entry. The companion test in
    ``test_read_cache.py`` calls ``clear_read_cache()`` to stand in for that — which asserts
    the pin's half and quietly assumes the writer's. This drives ``write_recording`` itself,
    so the assumption is the thing under test: a rename mid-burst is visible to the next read.
    """
    from detail.storage import read_cache

    sdir = recording.create_recording(experiment_dir, "recording_001", name="before")
    recording.read_recording(sdir)  # warm

    with read_cache.pinned():
        assert recording.read_recording(sdir)["name"] == "before"
        recording.set_name(sdir, "after")
        assert recording.read_recording(sdir)["name"] == "after"

    assert recording.read_recording(sdir)["name"] == "after"


def test_threads_missing_one_record_together_parse_it_once(recording_dir, monkeypatch):
    """A montage prepares its clips on several threads, and on a cold cache the clips that
    share a recording all missed its ``recording.yaml`` at once: each one read and parsed it
    (~160 ms a parse on a real 35 KB record while the threads fought over the interpreter). One
    parse per change of the file, however many threads miss it (``read_cache.computing``).
    """
    from pathlib import Path

    from tests.storage.test_read_cache import _miss_together

    recording.clear_recording_cache()
    real = atomic.read_text
    holder = {}

    def read_text(path, *a, **k):
        if Path(path).name == recording.RECORDING_FILENAME:
            return holder["computation"]()
        return real(path, *a, **k)

    def call(computation):
        holder["computation"] = computation
        return recording.read_recording(recording_dir)

    monkeypatch.setattr(atomic, "read_text", read_text)
    text = real(recording_dir / recording.RECORDING_FILENAME)
    assert _miss_together(6, call, lambda: text) == 1
