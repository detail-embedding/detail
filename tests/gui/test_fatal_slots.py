"""One bad read must not kill the app, and must not pass unseen.

An exception escaping a Qt slot — or a Python override of a Qt virtual, such as ``paintEvent`` —
cannot cross back through C++. What the binding does with it decides what a user sees:

- **PyQt6** (the binding before #107) passed it to ``sys.excepthook``, whose default ended in
  ``qFatal()`` → SIGABRT: the window went down mid-session, unsaved edits lost.
- **PySide6** passes it to ``sys.excepthook`` and carries on. Nothing dies — and nothing tells the
  user either: the traceback goes to a terminal that a user who launched the app from an icon
  never sees, and the action they asked for silently did not happen.

So the app installs its own hook (``detail.gui.__main__.install_excepthook``) rather than relying
on the binding's default: the process survives **and** the user is told. Two layers are tested
here:

- the specific accessor that was raising where it documents a graceful answer
  (``Recording.video_available`` on a video root not registered on this machine), and
- the class-level guarantee that an escaping exception is reported to the user and is not fatal.

The second needs a subprocess: an abort cannot be caught in-process, so the honest assertion is on
the exit code and output of a real interpreter running a real event loop.
"""

import subprocess
import sys
import textwrap

import pytest

from detail.api.experiment import Experiment
from detail.storage import locations, recording as recording_store
from detail.storage import video_roots


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
from detail.gui.qt import QApplication, QMessageBox, QPushButton, QTimer, QWidget
app = QApplication([])
# The dialog the hook raises, recorded instead of shown: nothing can click it here.
QMessageBox.critical = staticmethod(lambda *a, **k: print("DIALOG", a[2], flush=True))
{install}


class Painted(QWidget):
    def paintEvent(self, event):
        raise ValueError("paint went wrong")


b = QPushButton("go")
b.clicked.connect(lambda: (_ for _ in ()).throw(ValueError("boom")))
w = Painted()
w.resize(40, 40)
# From the running event loop, as a user's click and a repaint reach them in the app.
QTimer.singleShot(0, b.click)
QTimer.singleShot(0, w.show)
QTimer.singleShot(300, app.quit)
app.exec()
print("ALIVE")
"""


def _run(install: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SLOT_RAISES.format(install=install))],
        capture_output=True,
        text=True,
        timeout=120,
    )


_INSTALL = "from detail.gui.__main__ import install_excepthook\ninstall_excepthook()\n"


def test_a_raising_slot_is_reported_and_the_app_carries_on():
    """The real assertion: a real interpreter, a real event loop, a slot and an override that
    raise. The process lives, the user is shown each error, and the traceback still reaches a
    developer's terminal."""
    done = _run(_INSTALL)
    assert done.returncode == 0, done.stderr[-800:]
    assert "ALIVE" in done.stdout
    dialogs = [line for line in done.stdout.splitlines() if line.startswith("DIALOG")]
    assert any("boom" in line for line in dialogs), done.stdout
    assert any("paint went wrong" in line for line in dialogs), done.stdout
    assert "boom" in done.stderr and "paint went wrong" in done.stderr


def test_without_the_hook_the_user_is_never_told():
    """Guard the guard: the binding's own default shows the user nothing, so the dialogs above
    are the hook's doing. If the binding ever started reporting on its own — or aborting again —
    this fails and the test above needs re-deriving."""
    done = _run("")
    assert "DIALOG" not in done.stdout
    assert done.returncode == 0 and "ALIVE" in done.stdout, (
        "the binding no longer carries on after an escaping exception; re-derive what the "
        "hook has to guarantee"
    )
