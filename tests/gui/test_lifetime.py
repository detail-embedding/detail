"""An event filter whose Python state the cyclic collector has cleared (``detail/gui/lifetime.py``).

**The class.** The cyclic collector *clears* every object in an unreachable cycle before it frees
any of them, and for a sip wrapper that empties ``__dict__`` while the C++ object is still alive —
still installed as some widget's event filter. Whatever the collection destroys first can then
send that widget events, and an ``eventFilter`` that reads ``self._x`` raises ``AttributeError``
into the excepthook. It is the event-filter half of the recorded class *"a Qt object's C++ side
outliving the Python state it needs"*, and it had been fixed twice by hand (``BusyOverlay``,
``FrameScrubber``, each with its own ``getattr`` sentinel) while ``Dialog`` — the base of every
pop-up in the app — went on reading ``self._body_widget`` on every event.

**How it hid.** A ``ClipPaneWindow`` and the clip player it follows form a cycle, and the window
closes on the player's ``destroyed``. Collected, the player went first, closed the cleared window,
the window's body was hidden, and ``Dialog.eventFilter`` raised ``'ClipPaneWindow' object has no
attribute '_body_widget'`` — at whichever later test happened to trigger a collection, so it was
reported as a teardown error on ``test_dialog_space[ClusterProfileDialog]``, about half the time,
alone or in a full run. Under a probe that cleared each buildable dialog and closed it, all 60
raised it: the shape is the base class, not the clip window.

**The rule.** An ``eventFilter`` in ``detail/gui/`` that reads instance state asks
``lifetime.cleared(self)`` first and filters nothing when it is true. One predicate rather than a
sentinel attribute per site, so the guard is one shape a scan can check. A filter that keeps no
instance state at all is the exception, and must *not* ask: ``cleared`` is true for it always.

**Guards.** ``test_a_cleared_dialog_filters_nothing`` is the deterministic form of the report;
``test_a_collected_clip_pane_window_raises_nothing`` is the reported path itself;
``test_every_event_filter_asks_whether_it_was_cleared`` is the fail-closed scan over every filter,
so the next one is covered the day it is written.
"""

import ast
import gc
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QChildEvent, QEvent  # noqa: E402
from PyQt6.QtGui import QHideEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from detail.gui.dialog import Dialog  # noqa: E402
from detail.gui.lifetime import cleared  # noqa: E402

GUI = Path(__file__).resolve().parents[2] / "detail" / "gui"

#: Filters that read no instance state, so must not ask :func:`cleared` (it is always true for
#: them). Each is a claim: one whose class has come to assign ``self.x = …`` fails
#: `test_every_exemption_still_keeps_no_state`.
STATELESS_FILTERS = {
    "scroll.py::PageScroll": "every read goes to Qt (`widget()`, the scroll bar) or to a method",
}


def test_a_cleared_dialog_filters_nothing(qapp):
    dialog = Dialog()
    dialog.body().addWidget(QLabel("row"))
    dialog.show()
    qapp.processEvents()
    body = dialog._body_widget

    # Not vacuous: the filter is installed on the body and acts on what it is sent.
    dialog._refit_timer.stop()
    QApplication.sendEvent(body, QEvent(QEvent.Type.LayoutRequest))
    assert (
        dialog._refit_timer.isActive()
    ), "the body's filter did not run — nothing to clear"

    dialog.__dict__.clear()  # what the collector's clear does to a wrapper in a cycle
    assert cleared(dialog)
    child = QLabel()  # held: the event carries a bare pointer to it
    for event in (
        QHideEvent(),
        QChildEvent(QEvent.Type.ChildAdded, child),
        QEvent(QEvent.Type.LayoutRequest),
    ):
        QApplication.sendEvent(body, event)
    dialog.close()
    qapp.processEvents()
    # An escape is failed by `conftest._slot_exceptions_are_failures`.


def test_a_collected_clip_pane_window_raises_nothing(qapp, tmp_path):
    """The reported path: built as the dialog sweep builds it, dropped, and collected.

    Measured before the fix: the AttributeError escaped on 5 of 6 collections, so three rounds
    miss a regression about once in two hundred runs.
    """
    from detail.gui.clip_pane_window import ClipPaneWindow
    from tests.gui import subjects

    experiment = subjects.an_experiment_with_completed_runs(tmp_path)
    recording = experiment.get_recording("recording_001")
    for _ in range(3):
        window = subjects.build_dialog(ClipPaneWindow, experiment, recording)
        window.show()
        qapp.processEvents()
        del window
        gc.collect()
        qapp.processEvents()


def _event_filters(source: str):
    """``(class name, eventFilter def)`` for every filter defined in ``source``."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "eventFilter":
                    yield node, item


def _asks_cleared_first(method: ast.FunctionDef) -> bool:
    body = method.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]  # the docstring
    if not body or not isinstance(body[0], ast.If):
        return False
    test = body[0].test
    return (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Name)
        and test.func.id == "cleared"
        and [ast.unparse(a) for a in test.args] == ["self"]
        and isinstance(body[0].body[0], ast.Return)
    )


def _assigns_instance_state(cls: ast.ClassDef) -> bool:
    for node in ast.walk(cls):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                return True
    return False


def _unguarded(source: str, name: str) -> list[str]:
    return [
        f"{name}::{cls.name}"
        for cls, method in _event_filters(source)
        if f"{name}::{cls.name}" not in STATELESS_FILTERS
        and not _asks_cleared_first(method)
    ]


def test_the_scan_rejects_an_unguarded_filter_and_accepts_a_guarded_one():
    """The scan's own counterexamples: the pre-fix `Dialog.eventFilter` shape, and the fix."""
    unguarded = (
        "class D:\n"
        "    def eventFilter(self, watched, event):\n"
        "        if watched is self._body_widget:\n"
        "            pass\n"
        "        return False\n"
    )
    guarded = (
        "class D:\n"
        "    def eventFilter(self, watched, event):\n"
        '        """Doc."""\n'
        "        if cleared(self):\n"
        "            return False\n"
        "        return watched is self._body_widget\n"
    )
    assert _unguarded(unguarded, "x.py") == ["x.py::D"]
    assert _unguarded(guarded, "x.py") == []


def test_every_event_filter_asks_whether_it_was_cleared():
    files = sorted(GUI.rglob("*.py"))
    found = [
        f"{p.relative_to(GUI).as_posix()}::{cls.name}"
        for p in files
        for cls, _ in _event_filters(p.read_text(encoding="utf-8-sig"))
    ]
    # Ten when this was written; a walk that stops finding them passes vacuously.
    assert len(found) >= 10, f"only {len(found)} event filters found: {found}"
    bad = [
        site
        for p in files
        for site in _unguarded(
            p.read_text(encoding="utf-8-sig"), p.relative_to(GUI).as_posix()
        )
    ]
    assert not bad, (
        f"these event filters do not start with `if cleared(self): return False`: {bad}. A "
        f"filter's wrapper can be cleared by the cyclic collector while Qt still delivers to it, "
        f"and a read of `self._x` then raises out of the event loop. If the filter keeps no "
        f"instance state at all, list it in STATELESS_FILTERS with why instead."
    )


def test_every_exemption_still_keeps_no_state():
    """A stateless filter that gains state is an unguarded one; an entry naming no filter is stale."""
    problems = []
    for site, _reason in STATELESS_FILTERS.items():
        path, cls_name = site.split("::")
        classes = {
            cls.name: cls
            for cls, _ in _event_filters((GUI / path).read_text(encoding="utf-8-sig"))
        }
        if cls_name not in classes:
            problems.append(f"{site}: no such event filter")
        elif _assigns_instance_state(classes[cls_name]):
            problems.append(f"{site}: assigns instance state, so it must ask `cleared`")
    assert not problems, problems
