"""The cyclic collector runs on the UI thread only — in the app and in every GUI test process.

**The class: Qt objects torn down by whichever thread happened to allocate.** The write-up is
``detail/gui/ui_thread_gc.py``'s docstring: a worker thread crossing the allocation threshold ran
the collection, cleared the closures connected to a widget's ``destroyed`` while the C++ widget
waited in the UI thread's queue, and the UI thread then called a function with no globals —
``Fatal Python error: Segmentation fault`` in ``busy.py``'s ``cancel`` at ``line ???``, four crash
reports, and the three suite hangs that followed them.

What would bring it back, each pinned here:

- **The app not installing it.** ``test_the_app_collects_on_the_ui_thread_only``.
- **A test process not installing it** (the ``qapp`` fixture). ``test_a_gui_test_process_collects_on_the_ui_thread_only``.
- **A schedule that never collects**, so turning the automatic collector off leaks every cycle.
  ``test_a_due_collection_runs_and_frees_a_cycle``.

The probe is a background thread that allocates far past the threshold; the negative case shows
the same thread *does* collect when the automatic collector is on, so the check can fail.
"""

from __future__ import annotations

import gc
import threading

import pytest

from detail.gui import ui_thread_gc


def _collections_on_a_worker_thread() -> int:
    """How many collections a thread that allocates past every threshold runs itself."""
    ran: list[int] = []

    def on_gc(phase, info):
        if (
            phase == "start"
            and threading.current_thread() is not threading.main_thread()
        ):
            ran.append(info["generation"])

    def allocate():
        keep = []
        for _ in range(50 * gc.get_threshold()[0]):
            keep.append([])  # a container, so it counts toward the threshold
            if len(keep) > 1000:
                keep.clear()

    gc.callbacks.append(on_gc)
    try:
        thread = threading.Thread(target=allocate)
        thread.start()
        thread.join()
    finally:
        gc.callbacks.remove(on_gc)
    return len(ran)


def test_a_gui_test_process_collects_on_the_ui_thread_only(qapp):
    assert not gc.isenabled(), "the qapp fixture must install ui_thread_gc"
    assert _collections_on_a_worker_thread() == 0


def test_the_probe_sees_a_worker_thread_collect_when_the_collector_is_automatic(qapp):
    gc.enable()
    try:
        assert _collections_on_a_worker_thread() > 0
    finally:
        gc.disable()


def test_a_due_collection_runs_and_frees_a_cycle(qapp):
    import weakref

    class Node:
        pass

    gc.collect()
    node = Node()
    node.me = node
    dead = weakref.ref(node)
    del node
    assert ui_thread_gc.collect_due() == 0 and dead() is not None, "nothing was due yet"
    keep = [[] for _ in range(gc.get_threshold()[0] + 10)]  # gen 0 over its threshold
    assert dead() is not None, "the automatic collector is off; nothing collected it"
    ui_thread_gc.collect_due()
    assert dead() is None
    del keep


def test_the_app_collects_on_the_ui_thread_only(qapp, monkeypatch):
    from detail.gui import __main__ as entry

    installed = []

    class _Window:
        def show(self):
            pass

    class _App:
        def __init__(self, _argv):
            pass

        def exec(self):
            return 0

    monkeypatch.setattr(entry, "QApplication", _App)
    monkeypatch.setattr(entry, "MainWindow", _Window)
    monkeypatch.setattr(entry, "configure_application", lambda app: None)
    monkeypatch.setattr(entry, "install_excepthook", lambda: None)
    monkeypatch.setattr(entry, "_set_macos_menu_name", lambda name: None)
    monkeypatch.setattr(ui_thread_gc, "install", lambda app: installed.append(app))
    assert entry.main() == 0
    assert len(installed) == 1 and isinstance(installed[0], _App)


@pytest.fixture(autouse=True)
def _restore_collector_state():
    yield
    gc.disable()  # the qapp fixture's state, whatever a test did
