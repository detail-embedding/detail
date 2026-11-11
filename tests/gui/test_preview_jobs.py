"""The app-owned preview queue and the Configuration page's "Preview cache" line (§3).

**A class guarded here: a worker waiting on a cross-process lock with a blocking call.** The
queue's build reports and is cancelled only through its progress callback, so a build parked in a
blocking ``flock`` behind another process's build (a CLI ``cache previews build``) showed nothing
and ignored the cancel, and ``shutdown`` returned with the thread still running — at quit, a live
thread torn down with the process. The build takes the cache lock through
``storage.previews.build_lock(waiting=…)``, which reports while it waits and raises on cancel;
the canonical write-up is ``tests/api/test_previews.py``'s.
"""

from __future__ import annotations

import pytest

from detail.api import previews
from detail.api.experiment import Experiment
from detail.gui import pages
from detail.gui import preview_jobs
from tests.gui.conftest import drain_until


@pytest.fixture
def two_recordings(tmp_path, write_vfr_video):
    experiment = Experiment.create("p", tmp_path)
    for name in ("first", "second"):
        recording = experiment.add_recording(name)
        recording.add_video_part(write_vfr_video(tmp_path / f"{name}.mp4", n_frames=30))
    return experiment


def test_queue_builds_every_recording_and_goes_idle(qapp, two_recordings):
    queue = preview_jobs.install(two_recordings)
    done, ticks, idle = [], [], []
    queue.recording_done.connect(done.append)
    queue.progress.connect(lambda p, s: ticks.append(s))
    queue.idle.connect(lambda: idle.append(True))
    queue.queue()
    assert drain_until(qapp, lambda: idle and not queue.busy)
    assert done == list(two_recordings.recording_ids)
    assert two_recordings.missing_previews() == []
    assert any(s.startswith("Previews: Building previews") for s in ticks)
    assert any(
        "1 waiting" in s for s in ticks
    )  # the second recording, while the first ran


def test_prioritise_moves_recordings_to_the_front(qapp, two_recordings):
    queue = preview_jobs.PreviewQueue(two_recordings)
    first, second = two_recordings.recording_ids
    queue.current = "busy elsewhere"  # hold the queue so nothing starts
    queue.queue([first, second, first])
    assert queue.pending == [first, second]
    queue.prioritise([second])
    assert queue.pending == [second, first]
    queue.prioritise(["third"])
    assert queue.pending == ["third", second, first]
    queue.cancel()
    assert queue.pending == []


def test_queue_recordings_only_reaches_the_installed_experiment(
    qapp, two_recordings, tmp_path
):
    queue = preview_jobs.install(two_recordings)
    queue.current = "hold"
    other = Experiment.create("other", tmp_path / "elsewhere")
    preview_jobs.queue_recordings(other, ["x"])
    assert queue.pending == []
    preview_jobs.queue_recordings(two_recordings, ["y"])
    assert queue.pending == ["y"]
    queue.close()
    queue.queue(["z"])
    assert queue.pending == []


def test_a_closed_queue_stops_its_build(qapp, two_recordings):
    queue = preview_jobs.install(two_recordings)
    queue.queue()
    queue.shutdown()
    assert drain_until(qapp, lambda: not queue.busy)
    assert queue.pending == []


def test_configuration_shows_the_cache_and_clears_in_place(qapp, two_recordings):
    two_recordings.build_previews()
    size = previews.preview_cache_size()
    panel = pages.PreviewCachePanel()
    assert panel._location.text() == str(previews.preview_cache_dir())
    assert drain_until(qapp, lambda: panel._clear.isEnabled())
    from detail.api.exporters import format_size

    assert panel._size_label.text() == format_size(size)

    panel._clear.click()  # first click asks, in place
    assert not panel._keep.isHidden()
    assert panel._clear.text() == f"Clear {format_size(size)}"
    assert previews.preview_cache_size() == size
    panel._keep.click()  # Keep backs out
    assert panel._keep.isHidden() and panel._clear.text() == "Clear"

    panel._clear.click()
    panel._clear.click()  # confirmed
    assert drain_until(qapp, lambda: panel._size_label.text() == "0 B")
    assert previews.preview_cache_size() == 0
    assert not panel._clear.isEnabled()  # nothing left to clear


# --------------------------------------------------------------------------- #
# review fixes
# --------------------------------------------------------------------------- #
def test_a_recording_queued_while_it_builds_runs_again(qapp, two_recordings):
    """A tracking run finishing while its recording is the one building: what it is
    missing was read when that build started, so it must run again after (finding 3)."""
    queue = preview_jobs.PreviewQueue(two_recordings)
    first = two_recordings.recording_ids[0]
    queue.current = first  # as if its build were in hand
    queue.queue([first])
    assert queue.pending == []
    assert queue._finish() == first
    assert queue.pending == [first]


def test_a_cancel_forgets_a_pending_rerun(qapp, two_recordings):
    queue = preview_jobs.PreviewQueue(two_recordings)
    first = two_recordings.recording_ids[0]
    queue.current = first
    queue.queue([first])
    queue.cancel()
    queue._finish()
    assert queue.pending == []


@pytest.fixture
def with_a_broken_part(two_recordings):
    from detail.storage import recording as recording_store

    broken = two_recordings.get_recording(two_recordings.recording_ids[0])
    part = recording_store.resolve_part_video_path(broken.dir, broken.video_parts[0])
    part.write_bytes(b"not a video" * 1000)
    return two_recordings, broken


def test_a_failed_build_is_signalled_and_the_queue_goes_on(qapp, with_a_broken_part):
    """A failed build used to vanish: the batch stopped and nothing said why (finding 5)."""
    experiment, broken = with_a_broken_part
    queue = preview_jobs.install(experiment)
    failures, done, idle = [], [], []
    queue.failed.connect(lambda rid, why: failures.append((rid, why)))
    queue.recording_done.connect(done.append)
    queue.idle.connect(lambda: idle.append(True))
    queue.queue()
    assert drain_until(qapp, lambda: idle and not queue.busy)
    assert [rid for rid, _why in failures] == [broken.id]
    assert "part 1 proxy" in failures[0][1]
    assert done == list(experiment.recording_ids)  # the second one still built
    assert experiment.missing_previews() == [
        t for t in experiment.missing_previews() if t.recording_id == broken.id
    ]


def test_the_window_shows_a_failed_build(qapp, with_a_broken_part):
    """The status bar keeps saying a build failed, with the reason on hover (finding 5)."""
    from detail.gui.app import MainWindow

    experiment, broken = with_a_broken_part
    window = MainWindow()
    window._activate_experiment(experiment)
    queue = window._preview_queue
    assert drain_until(qapp, lambda: not queue.busy and not queue.pending)
    assert drain_until(qapp, lambda: "failed" in window._preview_label.text())
    assert not window._preview_label.isHidden()
    assert window._preview_label.toolTip().startswith(f"{broken.name}: part 1 proxy:")
    queue.shutdown()


def test_install_stops_the_old_queue_before_returning(
    qapp, two_recordings, tmp_path, monkeypatch
):
    """Switching experiments must not leave the old queue's build running beside the new
    one's (finding 2)."""
    import threading

    started = threading.Event()

    def slow(recording, part_index, report=None, **_kwargs):
        started.set()
        for step in range(200):  # cancellable: it reports, as a real build does
            report(step / 200)
            threading.Event().wait(0.01)

    monkeypatch.setattr(previews, "build_proxy", slow)
    monkeypatch.setattr("detail.core.progress.PROGRESS_INTERVAL_S", 0.0)
    old = preview_jobs.install(two_recordings)
    old.queue()
    assert started.wait(10)
    worker = old._worker
    other = Experiment.create("other", tmp_path / "elsewhere")
    preview_jobs.install(other)
    assert worker.isFinished(), "the old build was still running after install returned"
    drain_until(qapp, lambda: not old.busy)


def test_a_build_waiting_on_another_process_says_so_and_stops_on_shutdown(
    qapp, two_recordings, held_build_lock
):
    """Another process (a CLI build) holds the cache's build lock. The queue's build shows that
    it is waiting, and shutting the queue down stops it there: a thread parked in a blocking
    ``flock`` could neither report nor be cancelled, so ``shutdown`` returned with it still
    running and the app tore down a live thread."""
    queue = preview_jobs.install(two_recordings)
    statuses = []
    queue.progress.connect(lambda _p, s: statuses.append(s))
    queue.queue()
    assert drain_until(qapp, lambda: any("Waiting for another" in s for s in statuses))
    worker = queue._worker
    queue.shutdown()
    assert worker.isFinished(), "the build waiting on the lock was still running"
    assert two_recordings.missing_previews(), "nothing was built behind the lock"
