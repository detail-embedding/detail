"""GUI cancel affordances: a Cancel button on any active run + the quit prompt.

The bug behind these: a run launched from the GUI was orphaned (GUI quit) and
kept running, with no in-app way to stop it. These assert the button appears only
while a run is active and routes to the (kill-capable, reopen-safe) API, and that
the close prompt's three choices dispatch correctly.
"""

from detail.api.experiment import Experiment
from detail.gui import pages
from detail.gui.app import MainWindow
from detail.gui.pages import (
    DatasetDetailPage,
    ModelDetailPage,
    RunDetailPage,
)
from detail.gui.qt import QCloseEvent, QMessageBox, QPushButton
from detail.gui.workers import BackgroundCall
from detail.storage import datasets as datasets_store
from detail.storage import locations
from detail.storage import models as models_store
from detail.storage import runs as runs_store


class _SyncBackgroundCall(BackgroundCall):
    """Run the call inline so done/failed fire deterministically in the test
    thread (real signals; only the OS thread is skipped) — no lingering QThread
    to be torn down later and segfault a subsequent test's event loop."""

    def start(self) -> None:  # type: ignore[override]
        self.run()


def _button(widget, text: str) -> QPushButton | None:
    return next((b for b in widget.findChildren(QPushButton) if b.text() == text), None)


def _write_run(proj: Experiment, run_id: str, status: str) -> None:
    runs_store.write_run(
        proj.dir,
        run_id,
        {
            "id": run_id,
            "module_id": "detection",
            "recording_ids": [],
            "per_recording": {},
            "config": {},
            "status": status,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )


# --- RunDetailPage: Cancel shown only while active, routes to the API --------- #
def test_run_detail_shows_cancel_only_when_active(qapp, tmp_path):
    proj = Experiment.create("p", tmp_path)
    _write_run(proj, "run_001", "running")
    page = RunDetailPage(proj, "run_001")
    assert _button(page, "Cancel run") is not None
    assert _button(page, "Delete run") is None


def test_run_detail_shows_delete_when_terminal(qapp, tmp_path):
    proj = Experiment.create("p", tmp_path)
    _write_run(proj, "run_001", "complete")
    page = RunDetailPage(proj, "run_001")
    assert _button(page, "Cancel run") is None
    assert _button(page, "Delete run") is not None


def test_run_detail_cancel_invokes_api(qapp, tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    _write_run(proj, "run_001", "running")
    called = []
    monkeypatch.setattr(proj, "cancel_run", lambda rid: called.append(rid))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr(pages, "BackgroundCall", _SyncBackgroundCall)
    page = RunDetailPage(proj, "run_001")
    _button(page, "Cancel run").click()
    assert called == ["run_001"]


def test_run_detail_cancel_declined_is_noop(qapp, tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    _write_run(proj, "run_001", "running")
    called = []
    monkeypatch.setattr(proj, "cancel_run", lambda rid: called.append(rid))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )
    page = RunDetailPage(proj, "run_001")
    _button(page, "Cancel run").click()
    assert called == []


# --- Model / Dataset detail: Cancel shown only while active ------------------- #
def test_model_detail_cancel_invokes_api(qapp, tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(proj.dir, "model_001")
    mdir.mkdir(parents=True, exist_ok=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "m",
            "status": "running",
            "trainer_id": "t",
            "dataset_id": "d",
        },
    )
    called = []
    monkeypatch.setattr(proj, "cancel_model", lambda mid: called.append(mid))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr(pages, "BackgroundCall", _SyncBackgroundCall)
    page = ModelDetailPage(proj, "model_001")
    assert _button(page, "Delete model") is None
    _button(page, "Cancel model").click()
    assert called == ["model_001"]


def test_dataset_detail_cancel_invokes_api(qapp, tmp_path, monkeypatch):
    proj = Experiment.create("p", tmp_path)
    ddir = datasets_store.dataset_dir(proj.dir, "dataset_001")
    ddir.mkdir(parents=True, exist_ok=True)
    datasets_store.write_dataset_yaml(
        ddir,
        {
            "id": "dataset_001",
            "name": "d",
            "status": "running",
            "n_train": 0,
            "n_val": 0,
        },
    )
    called = []
    monkeypatch.setattr(proj, "cancel_dataset", lambda did: called.append(did))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr(pages, "BackgroundCall", _SyncBackgroundCall)
    page = DatasetDetailPage(proj, "dataset_001")
    assert _button(page, "Delete dataset") is None
    _button(page, "Cancel dataset").click()
    assert called == ["dataset_001"]


# --- MainWindow.closeEvent: prompt + the three dispatch paths ----------------- #
def _window(tmp_path, monkeypatch) -> MainWindow:
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    return MainWindow()


def test_close_no_experiment_accepts(qapp, tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()


def test_close_no_active_runs_accepts(qapp, tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    proj = Experiment.create("p", tmp_path)
    _write_run(proj, "run_001", "complete")
    window._experiment = proj
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()


def test_close_active_runs_dont_quit_ignores(qapp, tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    proj = Experiment.create("p", tmp_path)
    _write_run(proj, "run_001", "running")
    window._experiment = proj
    monkeypatch.setattr(MainWindow, "_confirm_quit", lambda self, active: "dont")
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()


def test_close_active_runs_leave_accepts_without_cancel(qapp, tmp_path, monkeypatch):
    window = _window(tmp_path, monkeypatch)
    proj = Experiment.create("p", tmp_path)
    _write_run(proj, "run_001", "running")
    window._experiment = proj
    called = []
    monkeypatch.setattr(proj, "cancel_run", lambda rid: called.append(rid))
    monkeypatch.setattr(MainWindow, "_confirm_quit", lambda self, active: "leave")
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    assert called == []


def test_close_active_runs_cancel_quit_cancels_then_accepts(
    qapp, tmp_path, monkeypatch
):
    window = _window(tmp_path, monkeypatch)
    proj = Experiment.create("p", tmp_path)
    _write_run(proj, "run_001", "running")
    window._experiment = proj
    called = []
    monkeypatch.setattr(proj, "cancel_run", lambda rid: called.append(rid))
    monkeypatch.setattr(MainWindow, "_confirm_quit", lambda self, active: "cancel_quit")
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    assert called == ["run_001"]


def test_close_active_analysis_run_cancel_quit_cancels(qapp, tmp_path, monkeypatch):
    # An in-flight analysis run (clip-query / location-export) must be listed in the
    # quit prompt and cancellable like the other families.
    from detail.storage import analysis_runs as ar

    window = _window(tmp_path, monkeypatch)
    proj = Experiment.create("p", tmp_path)
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, "an_01"),
        {"id": "an_01", "module_id": "clip_query", "status": "running"},
    )
    window._experiment = proj
    # Keyed by the navigation kind now, the way `api.active` keys everything else — the quit
    # guard used to invent its own family names alongside them.
    assert window._active_runs().get("analysis_run") == ["an_01"]
    called = []
    monkeypatch.setattr(proj, "cancel_analysis", lambda rid: called.append(rid))
    monkeypatch.setattr(MainWindow, "_confirm_quit", lambda self, active: "cancel_quit")
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    assert called == ["an_01"]


def test_confirm_quit_dialog_builds_and_defaults_to_dont(qapp, tmp_path, monkeypatch):
    # exercise the real dialog construction (a renamed role/method would raise);
    # a dismissed box has no clicked button → the safe default, "dont".
    window = _window(tmp_path, monkeypatch)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    assert window._confirm_quit({"processing": ["run_001"]}) == "dont"
