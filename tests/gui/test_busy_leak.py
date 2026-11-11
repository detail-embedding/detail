"""A background call started in one test must not deliver into the next one.

The class: **`run_busy` hands its result to a callback on the UI thread, so a test that starts
one and returns before the delivery lands leaves that callback queued for whichever later test
next pumps the event loop.** Several of those callbacks are modal — the metadata export ends in
`QMessageBox.information` — and a modal box in a headless run has nobody to dismiss it, so the
session stops. Not a failure: a stop, indistinguishable from a slow test.

It got all the way to CI and cost 1h43m before anyone cancelled it, and it is worth knowing why
it hid so well:

- **The test that causes it passes.** `test_metadata_page_export_csv` waited for its *output
  file*, which the worker writes before it emits `done`, so the loop could break with the
  delivery still queued. Nothing about that test looks wrong or fails.
- **The test that dies did nothing.** `test_recording_detail_button_rows_share_a_left_edge` hung
  because it calls `processEvents()` — which is not a mistake, it is how you lay out a widget.
- **It is timing-dependent and order-dependent.** On a fast machine the delivery slips in during
  the waiting loop anyway; it needs a slow runner *and* a random order that puts a pumping test
  after the leaking one. It has never reproduced on the dev box.

`tests/gui/conftest.py::_no_leaked_busy_calls` is the mechanism — modal dialogs never block, and
pending calls are drained at teardown — and this is its guard.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QMessageBox, QWidget  # noqa: E402

from detail.gui import busy  # noqa: E402
from detail.gui.busy import run_busy  # noqa: E402

from tests.gui.conftest import drain_busy_calls  # noqa: E402


def _leak(widget) -> list:
    """Start a call and return **without** pumping, exactly as the real leak did."""
    delivered: list = []
    run_busy(widget, lambda: "done", delivered.append, label="")
    return delivered


def test_a_pending_call_is_registered_until_it_is_delivered(qapp):
    """The negative case. Without something outstanding to drain, the drain below would pass
    vacuously forever."""
    widget = QWidget()
    delivered = _leak(widget)
    assert busy._active, "nothing was outstanding, so this file proves nothing"
    assert (
        delivered == []
    ), "it delivered immediately; the leak cannot be reproduced this way"
    drain_busy_calls(qapp)


def test_draining_delivers_the_pending_call(qapp):
    widget = QWidget()
    delivered = _leak(widget)

    outstanding = drain_busy_calls(qapp)

    assert outstanding == 1
    assert delivered == ["done"], "the drain did not deliver the queued result"
    assert not busy._active, "the drain left a call outstanding"


def test_the_teardown_leaves_nothing_for_the_next_test(qapp):
    """What the fixture actually promises. This test leaks deliberately and asserts nothing;
    `_no_leaked_busy_calls` has to clean up after it, and the assertion is in the *next* test.
    """
    _leak(QWidget())


def test_nothing_was_left_over_from_the_previous_test(qapp):
    assert not busy._active, "a call leaked across a test boundary"


def test_a_modal_callback_cannot_block_a_test(qapp):
    """The other half. Draining is no use if the queued callback is the thing that blocks — and
    the callback that actually hung CI was `QMessageBox.information`, reached through the
    metadata export's `run_busy` on_done.

    Unpatched, this call never returns and the run stops here. It returns because the fixture
    stubs it, which is the whole point.
    """
    widget = QWidget()
    seen: list = []

    def on_done(_result):
        seen.append(QMessageBox.information(widget, "Export metadata", "Wrote it"))

    run_busy(widget, lambda: None, on_done, label="")
    drain_busy_calls(qapp)
    assert seen, "the modal callback never ran"


def test_the_slot_exception_guard_is_declared_before_the_busy_drain():
    """Autouse teardown is LIFO, so a fixture declared *later* finishes *first*.

    `_slot_exceptions_are_failures` swaps in an excepthook that records exceptions escaping a
    Qt slot and fails the test for them. `_no_leaked_busy_calls` drains pending `run_busy`
    callbacks at teardown — which is exactly where a leaked callback raises. Declared after the
    drain fixture, the hook was restored and asserted *before* the drain ran, so an exception
    in that window went to the real excepthook and was swallowed: the precise class the guard
    exists to catch, in the one place most likely to produce it.

    Asserted on source order because that is what the ordering *is*: pytest gives autouse
    fixtures of the same scope their file order, and there is no public handle on the finalizer
    stack to interrogate instead.
    """
    import pathlib

    source = (pathlib.Path(__file__).parent / "conftest.py").read_text(
        encoding="utf-8-sig"
    )
    assert source.index("def _slot_exceptions_are_failures") < source.index(
        "def _no_leaked_busy_calls"
    ), "the slot-exception hook is torn down before the busy drain it needs to cover"


# --------------------------------------------------------------------------- #
# a raw BackgroundCall is drained too
# --------------------------------------------------------------------------- #
def test_a_raw_background_call_is_drained_like_a_run_busy_one(qapp):
    """`run_busy` registered its worker and the harness drained it; a `BackgroundCall` started
    *directly* — which several pages do — was drained by nothing.

    #263 roots such a call in `workers._IN_FLIGHT` so the cyclic collector cannot clear the
    closure it is about to be delivered into. This is the other half of the same fact: an
    un-drained call's *result* still lands in whichever later test next pumps the loop, and
    several of these callbacks are modal — which in a headless run stops the session rather than
    failing it.
    """
    import threading

    from detail.gui.workers import _IN_FLIGHT, BackgroundCall

    release = threading.Event()
    call = BackgroundCall(lambda: release.wait(5), receiver=None)
    delivered: list = []
    call.done.connect(delivered.append)

    call.start()
    assert (
        call in _IN_FLIGHT
    ), "a started call is not in the register the harness drains"

    release.set()
    outstanding = drain_busy_calls(qapp)

    assert outstanding >= 1, "the drain did not see the pending call"
    assert delivered, "the drain returned before the result landed"
    assert call not in _IN_FLIGHT


def test_the_register_does_not_double_count_a_run_busy_worker(qapp):
    """`run_busy` starts a `BackgroundCall`, so its worker is in *both* registers. Counting them
    as a sum reported two pending calls for one, which is what the drain's own guard caught.
    """
    from detail.gui import busy
    from detail.gui.workers import _IN_FLIGHT

    widget = QWidget()
    worker = run_busy(widget, lambda: "done", lambda _r: None, label="")

    assert worker in busy._active and worker in _IN_FLIGHT
    assert len(busy._active | _IN_FLIGHT) == 1

    drain_busy_calls(qapp)
