"""A window a test orphans is freed at that test's teardown, never later by a stray collection.

**The class: a widget tree whose lifetime the cyclic collector decides.** A GUI test shows a window
and returns without closing it; the window's wrapper is in a reference cycle, so nothing frees it
and Qt keeps it live and visible, still receiving events. The collector frees it on some later full
pass, triggered by any allocation, including one inside that window's own event handler. The
handler's frames hold the child wrappers rather than the top level, so the top level is collected
and Qt deletes the tree under the running code:

    Windows fatal exception: access violation
      File "detail/gui/flow.py", line 136 in _arrange          <- item.sizeHint() on a freed control
      File "detail/gui/flow.py", line 83 in heightForWidth
      File "detail/gui/scrubber.py", line 844 in heightForWidth
      File "detail/gui/image_view.py", line 442 in resizeEvent
      File "tests/gui/conftest.py", line 106 in drain_busy_calls

Three times in five full ``tests/gui`` runs on Windows when #821 measured it, twice in
``test_spatial_map.py``'s clip player and once as a freed ``cluster_strip`` timer lambda an earlier
file had left running. The xdist worker dies, and before ``--max-worker-restart=0`` the run then
hung near the end. ``FlowLayout`` was not the bug, only where a
freed widget was read first.

``tests/gui/conftest.py::_orphaned_windows_die_with_their_test`` runs that collection itself at
teardown, where nothing is mid-event, so what a test orphaned is freed there and never later. It
freezes what survives, to keep the pass cheap, and ``_what_a_module_froze_dies_with_it`` unfreezes
and collects once at each module's end, so a survivor that became garbage only later is freed with
its own file rather than leaking into the next one. The
subprocess pair below is the reproduction and its fix; the cross-test pair is the fixture being
wired; the module-scoped pair is the limit it must not cross — a window something still
references is not the collector's to free.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

_CONFTEST = Path(__file__).with_name("conftest.py")
_REPO = Path(__file__).resolve().parents[2]

_ORPHAN_THEN_COLLECT = """
import faulthandler, gc, os, pathlib
faulthandler.enable()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QApplication
from detail.gui import flow
from detail.gui.spatial_map import SpatialMap
import importlib.util

# By path: in a bare interpreter `tests` is not this repo's package.
_spec = importlib.util.spec_from_file_location("gui_conftest", {conftest!r})
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
collect_orphaned_windows = _conftest.collect_orphaned_windows

app = QApplication([])


class _Experiment:
    dir = pathlib.Path(".")


def a_gui_test():
    # What the tests do: show a window holding a clip player, and return without closing it.
    window = SpatialMap(_Experiment())
    window.resize(900, 240)
    window.show()
    placed = [
        {"instance_id": f"i{i}", "cx": x, "cy": y, "subject": 0,
         "t_ns": 1000 * (i + 1), "end_t_ns": 1000 * (i + 2)}
        for i, (x, y) in enumerate([(0.1, 0.1), (0.9, 0.9)])
    ]
    window.show_instances(
        {"rec_a": placed}, outlines={}, aspects={}, embed_run_ids={"rec_a": "emb_01"}
    )
    for _ in range(5):
        app.processEvents()
    block = window._player._block
    QApplication.postEvent(block, QResizeEvent(QSize(500, 300), block.size()))


class _CollectAfterFirst(list):
    # An allocation-triggered full collection, landing where the dumps did: mid-loop in _arrange.
    def __iter__(self):
        items = super().__iter__()
        yield next(items)
        gc.collect()
        yield from items


arrange = flow.FlowLayout._arrange


def arrange_then_collect(self, rect, *, apply):
    if not getattr(flow, "_collected", False):
        flow._collected = True
        self._items = _CollectAfterFirst(self._items)
    return arrange(self, rect, apply=apply)


flow.FlowLayout._arrange = arrange_then_collect
gc.disable()  # the one collection is the one placed above
a_gui_test()
{teardown}
app.processEvents()  # what the next teardown drain does
print("SURVIVED", flush=True)
"""


def _orphan_then_collect(teardown: str) -> subprocess.CompletedProcess:
    # A real interpreter: the failure is an access violation, which nothing in-process survives.
    # Run from this checkout, so `-c` imports this tree's `detail`.
    return subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(_ORPHAN_THEN_COLLECT)
            .replace("{teardown}", teardown)
            .replace("{conftest!r}", repr(str(_CONFTEST))),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_window_collected_at_teardown_cannot_be_collected_mid_event():
    done = _orphan_then_collect("collect_orphaned_windows()")
    assert done.returncode == 0, done.stderr[-1200:]
    assert "SURVIVED" in done.stdout


def test_without_the_teardown_the_collector_frees_it_mid_event():
    """Guard the guard: the same script with no collection at teardown must crash, or the test
    above proves nothing."""
    done = _orphan_then_collect("")
    assert done.returncode != 0, (
        "an orphaned window collected inside its own resize no longer crashes, so the test "
        "above no longer proves anything — re-derive it."
    )
    assert "SURVIVED" not in done.stdout
    assert "flow.py" in done.stderr, done.stderr[-1200:]


@pytest.fixture(scope="module", autouse=True)
def _no_automatic_collection():
    """Only the fixture's own collection may free the pair's window — an automatic pass between
    the two tests would make the second one pass without it."""
    import gc

    was_enabled = gc.isenabled()
    gc.disable()
    yield
    if was_enabled:
        gc.enable()


_LEFT_BEHIND: list = []


def test_a_test_that_orphans_a_window(qapp):
    import weakref

    from PyQt6.QtWidgets import QWidget

    window = QWidget()
    window.cycle = window  # what a closure connected back to its own widget amounts to
    window.show()
    _LEFT_BEHIND.append(weakref.ref(window))


def test_the_window_it_orphaned_is_gone_before_this_one_starts(qapp):
    if not _LEFT_BEHIND:
        pytest.skip("runs after test_a_test_that_orphans_a_window")
    assert (
        _LEFT_BEHIND.pop()() is None
    ), "the previous test's orphaned window outlived it"


@pytest.fixture(scope="module")
def a_module_window(qapp):
    from PyQt6.QtWidgets import QWidget

    window = QWidget()
    window.show()
    yield window
    window.deleteLater()


def test_a_wider_scoped_window_is_shared(a_module_window):
    a_module_window.setWindowTitle("shared")


def test_and_survives_the_tests_that_share_it(a_module_window):
    from PyQt6 import sip

    assert not sip.isdeleted(a_module_window), "a module-scoped window was destroyed"


def test_the_teardown_collection_does_not_rescan_the_heap():
    """A full pass over the whole heap per test measured 232 ms, 722 s of one ``tests/gui``
    run — a third of every worker. The helper freezes what survives, so the next pass scans
    only what the next test allocated (1.0 ms). Dropping the freeze fails nothing else; it just
    makes the suite slow, which is why it is pinned here rather than by a timing assertion.
    """
    import gc

    import importlib.util

    spec = importlib.util.spec_from_file_location("gui_conftest", _CONFTEST)
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)

    survivor = [object()]  # something certain to outlive the pass
    conftest.collect_orphaned_windows()
    assert (
        gc.is_tracked(survivor) and gc.get_freeze_count() > 0
    ), "survivors were not frozen"
    # By identity: `in` would call `==` on every object in the heap, arrays included.
    still_scanned = any(o is survivor for o in gc.get_objects())
    assert not still_scanned, "a survivor is still scanned by the next pass"


def test_what_the_per_test_pass_froze_is_collected_at_the_module_end():
    """The freeze's cost, paid back once per file: a survivor that becomes garbage only after it
    was frozen is never collected by the per-test pass, so it leaks with its timers running. A
    window ``test_dialog.py`` left that way started thread-pool work inside the next file's tests
    (``test_pool_work_does_not_cross_a_test_boundary``, 1 run in 2 on one xdist worker). The
    module-end pass unfreezes first, so it is freed with the module that leaked it.
    """
    import importlib.util
    import weakref

    spec = importlib.util.spec_from_file_location("gui_conftest", _CONFTEST)
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)

    class _Window:
        pass

    holder = [_Window()]
    holder[0].cycle = holder[
        0
    ]  # what a closure connected back to its own widget amounts to
    ref = weakref.ref(holder[0])
    conftest.collect_orphaned_windows()  # still referenced, so it survives and is frozen
    holder.clear()  # garbage only now, after the freeze

    conftest.collect_orphaned_windows()
    assert (
        ref() is not None
    ), "the per-test pass reaches frozen objects; this proves nothing"
    conftest.collect_what_the_module_froze()
    assert ref() is None, "a survivor frozen in one test outlived its module"
