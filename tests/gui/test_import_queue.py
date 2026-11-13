"""App-level import queue: shells up front, source overlay, worker, cancel.

Also the queue's lifetime, because the shell replaces it on every experiment switch: **a queue
dropped while its worker imports must not take the process with it.** Destroying a running
QThread is a Qt fatal error, and on Windows Qt terminates by fail-fast, which skips
faulthandler, so in the suite it showed as an xdist worker gone with no traceback ("node down:
Not properly terminated"). ``GuiWorker.start`` roots every started worker
(``workers._IN_FLIGHT``); the subprocess pair below pins that for the queue, and
``tests/gui/test_workers.py`` holds every other worker to it.
"""

import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from detail.api.imports import (  # noqa: E402
    RecordingImportSpec,
    create_shells,
)
from detail.api.experiment import Experiment  # noqa: E402
from tests import waits  # noqa: E402


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
    # The import itself is not this test's subject, but it is a real thread: join it, or it
    # runs on into the next test.
    waits.until(qapp, lambda: not queue.active, "the queued import to finish")


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


_REPO = Path(__file__).resolve().parents[2]

_DROP_WHILE_IMPORTING = """
import gc, os, pathlib, sys, tempfile, threading, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_FORCE_STDERR_LOGGING"] = "1"  # without a console, Windows Qt logs elsewhere
import cv2, numpy as np
from PyQt6.QtWidgets import QApplication
from detail.api.experiment import Experiment
from detail.api.imports import RecordingImportSpec
from detail.gui import import_queue, workers
app = QApplication([])
{unroot}
started, release = threading.Event(), threading.Event()

def blocked_import(*_args, **_kwargs):
    started.set()
    release.wait(30)

import_queue.import_pending = blocked_import  # held mid-import for as long as we like
root = pathlib.Path(tempfile.mkdtemp())
video = root / "v.mp4"
vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24))
for i in range(4):
    vw.write(np.full((24, 32, 3), i * 9 + 1, np.uint8))
vw.release()
queue = import_queue.ImportQueue(Experiment.create("p", root))
# The fatal window is between `start()` and the thread entering `run()`: once `run()` executes,
# its own frame holds the worker. Holding the GIL here keeps the thread out of `run()` until the
# queue is gone, so an unrooted worker is destroyed running every time rather than now and then.
sys.setswitchinterval(5)  # far longer than the drop takes; bounds any stall
queue.submit([RecordingImportSpec(id="vid", video_parts=[video])])
del queue  # what an experiment switch does to the old queue
gc.collect()
sys.setswitchinterval(0.005)
print("DROPPED", flush=True)
assert started.wait(30), "the import worker never started"
release.set()
deadline = time.monotonic() + 30
while workers._IN_FLIGHT and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.01)
assert not workers._IN_FLIGHT, "the worker was never released"
print("ALIVE", flush=True)
"""


def _drop_while_importing(unroot: str = "") -> subprocess.CompletedProcess:
    # A real interpreter, because the failure is the process ending: nothing in-process can catch
    # it, and on Windows it does not even reach faulthandler. Run from this checkout, so `-c`
    # imports this tree's `detail` and not whatever the editable install points at.
    return subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(_DROP_WHILE_IMPORTING.format(unroot=unroot)),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_queue_dropped_mid_import_does_not_end_the_process():
    done = _drop_while_importing()
    assert done.returncode == 0, done.stderr[-800:]
    assert "ALIVE" in done.stdout


def test_an_unrooted_worker_dropped_mid_import_really_does_end_it():
    """Guard the guard: the same script with ``GuiWorker.start`` put back to a bare
    ``QThread.start`` (the code before the root) must die, or the test above proves nothing.
    """
    done = _drop_while_importing(
        unroot="workers.GuiWorker.start = "
        "lambda self, *a, **k: workers.QThread.start(self, *a, **k)"
    )
    assert done.returncode != 0, (
        "dropping an unrooted running QThread no longer ends the process, so the test above "
        "no longer proves anything; re-derive it."
    )
    assert "Destroyed while thread" in done.stderr
    assert "ALIVE" not in done.stdout
