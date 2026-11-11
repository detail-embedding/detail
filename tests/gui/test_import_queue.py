"""App-level import queue: shells up front, source overlay, worker, cancel."""

import threading

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from detail.api.imports import (  # noqa: E402
    RecordingImportSpec,
    create_shells,
)
from detail.api.experiment import Experiment  # noqa: E402


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
    from PyQt6.QtCore import QEventLoop, QTimer

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
