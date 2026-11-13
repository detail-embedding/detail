"""One bad read must not kill the app.

PyQt6's default excepthook ends in ``qFatal()`` → SIGABRT, so *any* exception escaping a Qt
slot takes the whole window down mid-session — no dialog, unsaved edits lost. Two layers are
tested here:

- the specific accessor that was raising where it documents a graceful answer
  (``Recording.video_available`` on a video root not registered on this machine), and
- the class-level guarantee that an escaping exception is reported rather than fatal.

The second needs a subprocess: an abort cannot be caught in-process, so the only honest
assertion is on the exit code of a real interpreter.
"""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.storage import locations, recording as recording_store  # noqa: E402
from detail.storage import video_roots  # noqa: E402


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """Point storage.yaml at a throwaway location so tests never touch ~/.config."""
    config_dir = tmp_path / "config"
    monkeypatch.setattr(locations, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(locations, "STORAGE_PATH", config_dir / "storage.yaml")
    return config_dir


def _recording_with_part(tmp_path, part: dict):
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.create_recording("recording_001")
    data = recording_store.read_recording(recording.dir)
    data["video"] = {"storage": "default", "files": [part]}
    recording_store.write_recording(recording.dir, data)
    return recording


# --------------------------------------------------------------------------- #
# the presence gate answers, whatever the reason
# --------------------------------------------------------------------------- #
def test_video_available_is_false_for_an_unregistered_root(isolated_storage, tmp_path):
    """A reference-in-place part whose root alias isn't registered on this machine —
    routine after a video-slim export is opened elsewhere. It used to raise
    MissingVideoRootError out of five Qt slots and abort the app."""
    recording = _recording_with_part(
        tmp_path,
        {"index": 0, "id": "part_001", "source": {"root": "nas", "rel": "a.mp4"}},
    )
    assert recording.video_available is False


def test_the_two_unreachable_cases_agree(isolated_storage, tmp_path):
    """A missing file and a missing root both mean "not reachable here", so the presence
    gate must not answer them differently — that asymmetry was the bug."""
    missing_file = _recording_with_part(
        tmp_path / "a", {"index": 0, "id": "part_001", "file": "video/part_001.mp4"}
    )
    (tmp_path / "nas").mkdir(parents=True, exist_ok=True)
    video_roots.register_video_root("nas", tmp_path / "nas")
    missing_root_file = _recording_with_part(
        tmp_path / "b",
        {"index": 0, "id": "part_001", "source": {"root": "nas", "rel": "gone.mp4"}},
    )
    unregistered = _recording_with_part(
        tmp_path / "c",
        {"index": 0, "id": "part_001", "source": {"root": "other", "rel": "a.mp4"}},
    )
    assert missing_file.video_available is False
    assert missing_root_file.video_available is False
    assert unregistered.video_available is False


def test_readers_that_need_pixels_still_fail_loudly(isolated_storage, tmp_path):
    """Only the *presence gate* softens. A reader resolving a path must still raise, or a
    run would silently skip a recording whose video is merely mis-registered."""
    with pytest.raises(video_roots.MissingVideoRootError):
        recording_store.resolve_part_video_path(
            tmp_path, {"source": {"root": "nope", "rel": "a.mp4"}}
        )


# --------------------------------------------------------------------------- #
# the class-level guarantee
# --------------------------------------------------------------------------- #
def test_the_excepthook_reports_instead_of_dying(monkeypatch):
    from detail.gui import __main__ as gui_main

    shown = []
    monkeypatch.setattr(
        gui_main.QMessageBox, "critical", staticmethod(lambda *a, **k: shown.append(a))
    )
    original = sys.excepthook
    try:
        gui_main.install_excepthook()
        sys.excepthook(ValueError, ValueError("no config.yaml in /gone"), None)
    finally:
        sys.excepthook = original

    assert shown, "an escaping exception must be reported to the user"
    assert "no config.yaml" in shown[0][2]


def test_keyboard_interrupt_is_not_swallowed(monkeypatch):
    from detail.gui import __main__ as gui_main

    seen = []
    original = sys.excepthook
    try:
        sys.excepthook = lambda *a: seen.append(a)  # stand in for the previous hook
        gui_main.install_excepthook()
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    finally:
        sys.excepthook = original
    assert seen, "Ctrl-C must reach the interpreter's own hook"


_SLOT_RAISES = """
import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication, QPushButton
app = QApplication([])
{install}
b = QPushButton("go")
b.clicked.connect(lambda: (_ for _ in ()).throw(ValueError("boom")))
b.click()
print("ALIVE")
"""


def _run(install: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SLOT_RAISES.format(install=install))],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_raising_slot_does_not_abort_the_process():
    """The real assertion: exit code of a real interpreter, since an abort can't be caught
    in-process."""
    done = _run(
        "from detail.gui.__main__ import install_excepthook\ninstall_excepthook()\n"
        "from PyQt6.QtWidgets import QMessageBox\n"
        "QMessageBox.critical = staticmethod(lambda *a, **k: None)"
    )
    assert done.returncode == 0, done.stderr[-800:]
    assert "ALIVE" in done.stdout
    assert "boom" in done.stderr  # ...and the traceback still reaches a developer


_WORKER_RAISES = """
import os, sys, threading
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtCore import QThread, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox
app = QApplication([])
from detail.gui.__main__ import install_excepthook
install_excepthook()
shown = []
QMessageBox.critical = staticmethod(
    lambda *a, **k: shown.append(threading.current_thread().name)
)

class Worker(QThread):
    def run(self):
        raise ValueError("boom in a worker")

worker = Worker()
worker.start()
worker.wait()
QTimer.singleShot(400, app.quit)
app.exec()
print("SHOWN", shown)
"""


def test_an_exception_in_a_worker_thread_is_reported_on_the_ui_thread():
    """The hook runs in the thread that raised, and a `QMessageBox` built on a worker thread
    is a second fault. A progress worker's `ValueError` was reported that way and Qt then
    printed `QBasicTimer::stop: Failed. Possibly trying to stop from a different thread`
    ten times, `QBackingStore::endPaint() called with active painter`, and the process died
    with a segmentation fault.

    Asserted on the thread the dialog is built on, since whether a wrong-thread widget
    crashes this interpreter on this box is not something a test can rely on.
    """
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_WORKER_RAISES)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-800:]
    assert "boom in a worker" in done.stderr  # the traceback still reaches a developer
    assert "SHOWN ['MainThread']" in done.stdout, done.stdout


def test_a_fatal_signal_prints_every_threads_python_stack():
    """Before this a native crash left the shell's `Segmentation fault` and nothing else."""
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import faulthandler, threading, time\n"
            "from detail.gui.__main__ import enable_crash_traces\n"
            "enable_crash_traces()\n"
            "threading.Thread(target=time.sleep, args=(5,), daemon=True, name='a-worker').start()\n"
            "time.sleep(0.2)\n"
            "faulthandler._sigsegv()\n",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode != 0
    assert "Fatal Python error: Segmentation fault" in done.stderr, done.stderr[-600:]
    assert (
        done.stderr.count("Thread 0x") >= 1 and "most recent call first" in done.stderr
    )


def test_enabling_crash_traces_never_blocks_a_launch_without_a_usable_stderr(
    monkeypatch,
):
    """`pythonw` has no stderr and a test's capture has no file descriptor: `faulthandler`
    cannot attach to either, and the app must still start."""
    import io

    from detail.gui import __main__ as gui_main

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    gui_main.enable_crash_traces()
    monkeypatch.setattr(sys, "stderr", None)
    gui_main.enable_crash_traces()


def test_without_the_hook_it_really_does_abort():
    """Guard the guard: if PyQt stopped aborting, the test above would pass vacuously."""
    done = _run("")
    assert done.returncode != 0, (
        "PyQt no longer aborts on an escaping slot exception, so the hook test above no "
        "longer proves anything — re-derive it."
    )
    assert "ALIVE" not in done.stdout
