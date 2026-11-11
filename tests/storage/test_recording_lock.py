"""recording_lock: mutual exclusion and collision-free run-id reservation.

These guard running several embeddings concurrently on one recording (pinned to different GPUs): the
recording.yaml read-modify-write must serialize so ids don't collide and records aren't lost. Each
``recording_lock`` acquires its own open file description, so ``fcntl.flock`` serializes concurrent
holders — the same kernel lock that serializes separate processes — which these thread-based tests
exercise directly. Reverting the lock makes both fail (lost updates / colliding ids).
"""

import threading
import time

import pytest

from detail.api.experiment import Experiment
from tests.concurrency import join_all
from detail.core import atomic
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store


def test_recording_lock_serializes_concurrent_writers(tmp_path):
    counter = tmp_path / "counter.txt"
    counter.write_text("0", encoding="utf-8")

    def increment() -> None:
        with recording_store.recording_lock(tmp_path):
            value = int(counter.read_text(encoding="utf-8-sig") or "0")
            time.sleep(
                0.01
            )  # widen the window; without the lock, updates would be lost
            counter.write_text(str(value + 1), encoding="utf-8")

    threads = [threading.Thread(target=increment) for _ in range(12)]
    for t in threads:
        t.start()
    join_all(threads)
    assert int(counter.read_text(encoding="utf-8-sig")) == 12  # every increment landed


def test_failed_acquire_propagates_and_closes_the_fd(tmp_path, monkeypatch):
    """A failed acquire must not be followed by a release of a lock we never took.

    Releasing an unheld lock raises in its own right (PermissionError on Windows), which
    would replace the real error and skip the close, leaking a descriptor per attempt.
    """
    closed = []
    real_close = atomic.os.close

    def spy_close(fd):
        closed.append(fd)
        return real_close(fd)

    def boom(fd):
        raise OSError("cannot acquire")

    monkeypatch.setattr(atomic, "_acquire_lock", boom)
    monkeypatch.setattr(
        atomic, "_release_lock", lambda fd: pytest.fail("released unheld lock")
    )
    monkeypatch.setattr(atomic.os, "close", spy_close)

    with pytest.raises(OSError, match="cannot acquire"):  # the real error, unmasked
        with recording_store.recording_lock(tmp_path):
            pytest.fail("body must not run")
    assert len(closed) == 1  # the descriptor was closed, not leaked


def test_reserve_run_allocates_distinct_ids_under_concurrency(tmp_path):
    recording = Experiment.create("proj", tmp_path).create_recording("recording_001")

    def reserve() -> None:
        recording._reserve_run("stress", "x", lambda rid: {"id": rid})

    threads = [threading.Thread(target=reserve) for _ in range(10)]
    for t in threads:
        t.start()
    join_all(threads)

    recording_store.clear_recording_cache()
    sdir = experiment_store.recording_path(recording._experiment_dir, "recording_001")
    ids = [r["id"] for r in recording_store.read_recording(sdir).get("stress", [])]
    assert len(ids) == 10  # every reservation's record survived (no lost update)
    assert len(set(ids)) == 10  # and every id is distinct (no collision)
