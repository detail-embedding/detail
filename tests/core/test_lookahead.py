"""``core/lookahead.py``: working ahead of a consumer without reordering it.

**The class.** A stage that decodes on one thread, renders on a pool and runs a GPU on a third is
only correct if the consumer sees exactly the serial loop's sequence: the same items, in the same
order, failing at the same item, and holding a bounded number of them. Each test below plants the
way that breaks — random per-item delays to scramble completion order, an exception at a known
item, an early ``break`` that would strand a thread, a producer that counts what is outstanding —
and asserts the serial answer.

**No item starts after a failure** (``test_no_item_starts_after_an_item_has_failed``). The
writer stages of detection, embedding and keypoint detect are ``ordered_map(write_batch, ...,
workers=1)``, so ``fn`` writes. The pool runs a window ahead, and the batch queued behind a
failed write ran on the freed worker before the consumer saw the error: a later batch on disk after
a gap, which resume never fills. It hid because it needs the consumer to be slow at that instant:
1 run in 20 alone, every run with 8 loops in parallel (keypoint detect's write-error test, #920).
The test holds the consumer still while item 1 fails, so the race is not left to the scheduler.
"""

import random
import threading
import time

import pytest

from detail.core import lookahead


def _lookahead_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("lookahead")]


def _jittered(x: int) -> int:
    time.sleep(random.Random(x).uniform(0, 0.004))
    return x * x


def test_ordered_map_keeps_input_order_when_completion_is_scrambled():
    got = list(lookahead.ordered_map(_jittered, range(200), workers=8, window=24))
    assert got == [x * x for x in range(200)]


def test_background_iter_yields_the_producers_sequence():
    assert list(lookahead.background_iter(iter(range(500)), depth=3)) == list(
        range(500)
    )


def test_the_two_composed_are_the_serial_loop():
    serial = [_jittered(x) for x in range(150)]
    piped = list(
        lookahead.ordered_map(
            _jittered,
            lookahead.background_iter(range(150), depth=4),
            workers=6,
            window=10,
        )
    )
    assert piped == serial


def test_an_fn_error_surfaces_at_its_item_after_every_earlier_result():
    def boom(x: int) -> int:
        if x == 37:
            raise ValueError("item 37")
        return _jittered(x)

    seen = []
    with pytest.raises(ValueError, match="item 37"):
        for value in lookahead.ordered_map(boom, range(100), workers=8, window=16):
            seen.append(value)
    assert seen == [x * x for x in range(37)]


def test_no_item_starts_after_an_item_has_failed():
    """A writer stage is ``ordered_map(write_batch, ..., workers=1)``: its ``fn`` writes. The
    consumer is held at its first result while item 1 fails, which leaves the freed worker
    every chance to start the items queued behind it; none may start. Before the fix the next
    queued batch was written after the failed one, a gap in the part store."""
    calls = []
    one_failed = threading.Event()
    later_started = threading.Event()

    def write(x: int) -> int:
        calls.append(x)
        if x == 1:
            one_failed.set()
            raise OSError("disk full")
        if x > 1:
            later_started.set()
        return x

    written = lookahead.ordered_map(write, range(6), workers=1, window=4)
    assert next(written) == 0
    assert one_failed.wait(5)
    assert not later_started.wait(0.3), f"started after the failure: {calls}"
    with pytest.raises(OSError, match="disk full"):
        next(written)
    assert calls == [0, 1]
    assert _lookahead_threads() == []


def test_a_source_error_surfaces_after_the_items_it_did_produce():
    """A video that ends early: every frame read before it is still delivered first."""

    def walk():
        yield from range(10)
        raise RuntimeError("video ended at frame 10")

    for source in (walk(), lookahead.background_iter(walk(), depth=2)):
        seen = []
        with pytest.raises(RuntimeError, match="ended at frame 10"):
            for value in lookahead.ordered_map(_jittered, source, workers=4, window=8):
                seen.append(value)
        assert seen == [x * x for x in range(10)]


def test_closing_early_leaves_no_thread_and_closes_the_source():
    closed = threading.Event()

    def walk():
        try:
            yield from range(10_000)
        finally:
            closed.set()

    before = _lookahead_threads()
    rows = lookahead.ordered_map(
        _jittered, lookahead.background_iter(walk(), depth=4), workers=6, window=12
    )
    for i, _ in enumerate(rows):
        if i == 20:
            break
    rows.close()
    assert closed.wait(2.0), "the source's finally (a VideoCapture release) never ran"
    deadline = time.monotonic() + 2.0
    while _lookahead_threads() != before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _lookahead_threads() == before


def test_a_consumer_error_also_stops_the_threads():
    before = _lookahead_threads()
    with pytest.raises(KeyError):
        for i, _ in enumerate(
            lookahead.ordered_map(
                _jittered,
                lookahead.background_iter(range(5000), depth=3),
                workers=4,
                window=8,
            )
        ):
            if i == 5:
                raise KeyError("consumer")
    deadline = time.monotonic() + 2.0
    while _lookahead_threads() != before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _lookahead_threads() == before


@pytest.mark.parametrize("depth, window", [(1, 1), (3, 8), (16, 96)])
def test_items_outstanding_never_exceed_depth_plus_window(depth, window):
    """Tenet 11: a 1080p frame is ~6 MB, so the bound is the memory bound.

    Outstanding = produced and not yet counted by the consumer; the consumer is slow so both
    buffers fill to their limit. Two more are allowed and no others: the item the producer holds
    while it waits on a full queue, and the result being handed over while the window refills.
    """
    lock = threading.Lock()
    produced = 0
    consumed = 0
    peak = 0

    def walk():
        nonlocal produced, peak
        for x in range(300):
            with lock:
                produced += 1
                peak = max(peak, produced - consumed)
            yield x

    for _ in lookahead.ordered_map(
        lambda x: x,
        lookahead.background_iter(walk(), depth=depth),
        workers=4,
        window=window,
    ):
        time.sleep(0.0005)
        with lock:
            consumed += 1
    assert consumed == 300
    assert peak <= depth + window + 2


@pytest.mark.parametrize(
    "call", ["background_iter", "ordered_map_workers", "ordered_map_window"]
)
def test_a_bound_below_one_is_refused(call):
    with pytest.raises(ValueError, match="at least 1"):
        if call == "background_iter":
            next(lookahead.background_iter(range(3), depth=0))
        elif call == "ordered_map_workers":
            next(lookahead.ordered_map(abs, range(3), workers=0, window=4))
        else:
            next(lookahead.ordered_map(abs, range(3), workers=2, window=0))
