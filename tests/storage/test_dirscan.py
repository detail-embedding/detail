"""A listing costs one enumeration plus one probe per child, not two probes.

**The class.** *A cost model that assumes local disk, applied to a workspace that can be
configured onto a network share.* An O(N) sweep of individually-cheap reads is invisible on
an SSD and seconds over SMB, and the guard that would normally catch a slow seam
(``tests/gui/test_ui_thread_blocking.py``) excludes those reads *precisely because each one
is small*.

The instance: every ``discover_*`` in ``detail/storage`` spelled its listing as
``iterdir()`` + ``is_dir()`` + ``is_file()``. ``iterdir`` yields bare paths, so ``is_dir()``
is a fresh ``stat`` per child on top of the probe — two round trips per recording where
one will do. On an SMB share a ``stat`` is slow, so Home spent long enough on the UI
thread listing recordings it only wanted to *count* that Windows retitled the window
"detail not responding".

**Why the count and not a timer.** The defect is a number of round trips; a wall-clock
assertion on a local SSD measures nothing (both shapes finish in a millisecond) and would be
flaky besides. Count the syscall. ``os.scandir`` answers ``is_dir`` out of the directory
entry itself, so it does not appear in this count at all — which is the whole point.
"""

import os
from pathlib import Path

from detail.storage import dirscan, experiment as experiment_store
from tests.round_trips import count_stats

N_RECORDINGS = 60


def _recordings_tree(root: Path, n: int = N_RECORDINGS) -> Path:
    sdir = root / "recordings"
    for i in range(n):
        (sdir / f"recording_{i:03d}").mkdir(parents=True)
        (sdir / f"recording_{i:03d}" / "recording.yaml").write_text(
            "schema_version: 1\n", encoding="utf-8"
        )
    return sdir


# --------------------------------------------------------------------------- #
# the cost
# --------------------------------------------------------------------------- #
def test_discovering_recordings_probes_each_child_once(tmp_path):
    _recordings_tree(tmp_path)
    ids, stats = count_stats(
        tmp_path, lambda: experiment_store.discover_recordings(tmp_path)
    )

    assert len(ids) == N_RECORDINGS
    # One probe per child, plus a small constant for the root. The rejected shape is 2N.
    assert stats <= N_RECORDINGS + 4, (
        f"{stats} stat calls for {N_RECORDINGS} recordings — a listing costs one probe per "
        "child, not two. Route it through storage/dirscan."
    )


def test_the_count_would_see_the_shape_this_replaced(tmp_path):
    """Guard the guard: a counter that sees nothing passes vacuously forever.

    The reverted body, run through the same counter. If this does not report ~2N, the
    assertion above is not measuring what it claims to.
    """
    sdir = _recordings_tree(tmp_path)

    def the_old_way():
        return sorted(
            p.name
            for p in sdir.iterdir()
            if p.is_dir() and (p / "recording.yaml").is_file()
        )

    ids, stats = count_stats(tmp_path, the_old_way)
    assert len(ids) == N_RECORDINGS
    assert stats >= 2 * N_RECORDINGS
    # ...and it is over the budget the real listing is held to.
    assert stats > N_RECORDINGS + 4


# --------------------------------------------------------------------------- #
# ...bought without changing the answer
# --------------------------------------------------------------------------- #
def test_the_listing_is_unchanged_by_the_cheaper_enumeration(tmp_path):
    sdir = _recordings_tree(tmp_path, n=5)
    (sdir / "half_written").mkdir()  # a directory with no record is not a recording
    (sdir / "stray.txt").write_text("", encoding="utf-8")  # nor is a file
    (
        sdir / "recording_002" / "recording.yaml"
    ).unlink()  # nor is one whose record is gone

    assert experiment_store.discover_recordings(tmp_path) == [
        "recording_000",
        "recording_001",
        "recording_003",
        "recording_004",
    ]


def test_an_absent_directory_lists_empty_rather_than_raising(tmp_path):
    """Every caller scans a section directory that only exists once written into."""
    assert dirscan.discover_ids(tmp_path / "never_created", "x.yaml") == []
    assert dirscan.child_dirs(tmp_path / "never_created") == []
    assert dirscan.dirs_containing(tmp_path / "never_created", "x.yaml") == []
    # A *file* where a directory was expected is the same answer, not an OSError.
    (tmp_path / "a_file").write_text("", encoding="utf-8")
    assert dirscan.child_dirs(tmp_path / "a_file") == []


def test_child_dirs_are_directories_only(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "c.txt").write_text("", encoding="utf-8")
    assert [p.name for p in dirscan.child_dirs(tmp_path)] == ["a", "b"]
