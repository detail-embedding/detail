"""App-level import queue: shells up front, source overlay, worker, cancel.

**An armed state set without asking whether it applies** (#35). ``cancel()`` set the queue's stop
event whether or not anything was running. **Cancel imports** is enabled from the table's last
read, and the quit path decides from run records rather than from the queue, so both reach it
with nothing running; nothing cleared the event while idle, and the next ``submit`` handed it to
a new worker, which marked that batch ``cancelled`` at once — a video never copied, and no
message. A stop left set when a batch ended also stranded what was submitted after it as
"queued". The stop now applies only to a running batch and is cleared when that batch ends.
"""

import threading

import cv2
import numpy as np

from detail.api.imports import (
    RecordingImportSpec,
    create_shells,
)
from detail.api.experiment import Experiment


def _video(path, n=4, w=32, h=24):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), (i * 9) % 250 + 1, np.uint8))
    vw.release()
    return path


def test_submit_creates_shells_immediately(qapp, tmp_path):
    from detail.gui.import_queue import ImportQueue

    experiment = Experiment.create("p", tmp_path)
    queue = ImportQueue(experiment)
    queue.submit(
        [
            RecordingImportSpec(id="vid", video_parts=[_video(tmp_path / "v.mp4")]),
            RecordingImportSpec(id="empty", metadata={"animal_id": "M001"}),
        ]
    )
    # both recordings exist synchronously, before the worker imports the video
    assert {s.id for s in experiment.recordings} == {"vid", "empty"}
    assert experiment.get_recording("empty").import_status == "ready"


def test_source_for_shows_pending_video(qapp, tmp_path):
    from detail.gui.import_queue import ImportQueue

    queue = ImportQueue(Experiment.create("p", tmp_path))
    spec = RecordingImportSpec(
        id="s1", video_parts=[tmp_path / "clip.mp4", tmp_path / "clip2.mp4"]
    )
    queue._items["s1"] = spec  # white-box: a queued item
    assert queue.source_for("s1") == "clip.mp4  (+1)"
    assert queue.source_for("absent") is None


def test_worker_imports_created_shells(qapp, tmp_path):
    from detail.gui.import_queue import _ImportWorker

    experiment = Experiment.create("p", tmp_path)
    specs = [RecordingImportSpec(id="s1", video_parts=[_video(tmp_path / "a.mp4")])]
    create_shells(experiment, specs)
    assert experiment.get_recording("s1").import_status == "queued"
    worker = _ImportWorker(experiment, specs, {}, threading.Event(), receiver=None)
    done = []
    worker.item_done.connect(lambda sid, ok, err: done.append((sid, ok)))
    worker.run()  # synchronous (not start())
    assert done == [("s1", True)]
    assert experiment.get_recording("s1").import_status == "ready"


def test_queue_folds_every_batch_into_one_count(qapp, tmp_path, monkeypatch):
    """Through the real worker thread: the bar appears at submit, moves inside a recording,
    and a bout queued while another imports grows the same "N of M" rather than restarting it.
    """
    from detail.gui.qt import QEventLoop, QTimer

    from detail.core import progress as progress_mod
    from detail.gui.import_queue import ImportQueue

    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    monkeypatch.setenv("DETAIL_MAX_TRANSCODE_PARALLELISM", "1")
    queue = ImportQueue(Experiment.create("p", tmp_path))
    events = []
    queue.progress.connect(events.append)
    loop = QEventLoop()
    queue.idle.connect(loop.quit)

    queue.submit(
        [RecordingImportSpec(id="a", video_parts=[_video(tmp_path / "a.mp4")])]
    )
    assert (events[0].recordings_done, events[0].recordings_total) == (0, 1)
    queue.submit(
        [RecordingImportSpec(id="b", video_parts=[_video(tmp_path / "b.mp4")])]
    )
    QTimer.singleShot(60_000, loop.quit)  # a hang fails the asserts below, not the run
    loop.exec()

    assert all(
        e.recordings_total == 2 for e in events[1:]
    ), "a batch restarted the count"
    assert any(0 < e.fraction < 0.5 for e in events), "no movement inside the first"
    assert events[-1].recordings_done == 2 and events[-1].fraction == 1.0
    assert queue._fold is None, "the next submit would inherit this batch's clock"


def test_idle_when_empty(qapp, tmp_path):
    from detail.gui.import_queue import ImportQueue

    queue = ImportQueue(Experiment.create("p", tmp_path))
    assert queue.active is False
    queue.submit([])
    assert queue.active is False


def test_cancel_with_nothing_running_does_not_cancel_the_next_import(qapp, tmp_path):
    """#35 finding 7. **Cancel imports** is enabled from the table's last read, so a click can
    land after the queue went idle — and so can the quit path, which decides from run records
    rather than from the queue. `cancel()` set its event regardless; nothing cleared it while
    idle, and the next `submit` handed it to a new worker, which marked the batch `cancelled`
    at once: the video never copied, and nothing said so."""
    from detail.gui.qt import QEventLoop, QTimer

    from detail.gui.import_queue import ImportQueue

    experiment = Experiment.create("p", tmp_path)
    queue = ImportQueue(experiment)
    queue.cancel()  # nothing is running

    loop = QEventLoop()
    queue.idle.connect(loop.quit)
    queue.submit(
        [RecordingImportSpec(id="a", video_parts=[_video(tmp_path / "a.mp4")])]
    )
    QTimer.singleShot(60_000, loop.quit)  # a hang fails the assert below, not the run
    loop.exec()

    assert not queue.active, "the import never finished"
    assert experiment.get_recording("a").import_status == "ready"


def test_a_recording_added_after_a_cancel_still_imports(qapp, tmp_path, monkeypatch):
    """The sibling the same event left behind: a Cancel while a batch runs, then a submit
    before that batch ends. The batch ended with the event still set, so the new specs were
    neither drained nor marked — stuck "queued" for good. The stop is for what was queued
    when it was asked; what comes after it imports."""
    import threading

    from detail.gui.qt import QEventLoop, QTimer

    from detail.gui import import_queue as module
    from detail.gui.import_queue import ImportQueue

    gate, first_running = threading.Event(), threading.Event()
    real = module.import_pending

    def gated(experiment, specs, **kwargs):
        if specs[0].id == "a":
            first_running.set()
            assert gate.wait(30), "the test never released the first batch"
        return real(experiment, specs, **kwargs)

    monkeypatch.setattr(module, "import_pending", gated)
    experiment = Experiment.create("p", tmp_path)
    queue = ImportQueue(experiment)
    queue.submit(
        [RecordingImportSpec(id="a", video_parts=[_video(tmp_path / "a.mp4")])]
    )
    assert first_running.wait(30), "the first batch never started"
    queue.cancel()
    queue.submit(
        [RecordingImportSpec(id="b", video_parts=[_video(tmp_path / "b.mp4")])]
    )

    loop = QEventLoop()
    queue.idle.connect(loop.quit)
    gate.set()
    QTimer.singleShot(60_000, loop.quit)
    loop.exec()

    assert experiment.get_recording("a").import_status == "cancelled"
    assert experiment.get_recording("b").import_status == "ready"
