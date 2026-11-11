"""Work ahead of a consumer without reordering it: the one seam for overlapping CPU prep with a GPU.

**The class this closes.** A stage that reads video and runs a model did every step on one thread
— decode, render, forward, write — so the GPU waited for the CPU and the CPU for the GPU
(#874), idle most of the time. The fix had already been written twice, as a
deque of futures a bounded window ahead of an in-order consumer: the embedding worker's mask-only
branch and the kinematics backfill, each spelled out in its own module. CLAUDE.md: two copies of a
shape can be tolerated; the third is when it becomes one mechanism. These two functions are that
mechanism.

- :func:`background_iter` runs a *sequential* producer (one forward video walk, Tenet 11) on its
  own thread into a bounded queue.
- :func:`ordered_map` maps a function over an iterable on a thread pool, a bounded window ahead,
  and yields results **in input order**.

Threads, not processes: cv2 decode / colour convert / resize / border, pycocotools' RLE decode and
torch all release the GIL, so the Python left between them is microseconds a frame. Processes would
pickle each decoded frame through a pipe and open one ``VideoCapture`` per worker, which seeks.

**Memory is bounded by construction** (Tenet 11): an item is held either in ``background_iter``'s
queue (at most ``depth``) or in ``ordered_map``'s window (at most ``window``), plus the one the
producer holds while the queue is full and the one being handed over: never more than
``depth + window + 2`` at once however long the video.

**Order and errors are the serial path's.** Results come back in the order the input produced
them, and an exception is raised at the position of the item that caused it: every earlier result
is yielded first. Closing either generator early (a ``break``, a consumer exception) stops and joins
its thread and closes the iterator it was reading, so no thread outlives the loop that started it
holding a ``VideoCapture``.

**Nothing starts after a failure.** Once ``fn`` raises for an item, ``ordered_map`` starts ``fn``
on no later item. With one worker that is the serial loop exactly: a writer stage
(``ordered_map(write_batch, ..., workers=1)``, in detection, embedding and keypoint detect) stops
at the first failed write. Before this, the batch already queued behind the failed one ran on the
freed worker before the consumer saw the error, so a later batch reached disk after a gap: a part
store whose rows skip frames, which resume (reading the store's length) then never fills.
"""

from __future__ import annotations

import concurrent.futures
import queue
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

#: How often a blocked producer re-checks whether its consumer has gone away.
_POLL_S = 0.05


class _Raised:
    """The producer's exception, carried through the queue to the consumer's thread."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


_DONE = object()


class _Abandoned(Exception):
    """An item ``ordered_map`` did not start because an earlier one had failed."""


def _close(iterator) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


def background_iter(iterable: Iterable[T], depth: int) -> Iterator[T]:
    """Yield ``iterable``'s items in order, produced on one background thread ``depth`` ahead.

    The producer is the only thread that ever touches ``iterable``, so a source that must be read
    in order by one reader (a ``cv2.VideoCapture``) stays exactly that. Its exception is re-raised
    here after every item before it has been yielded.

    Raises:
        ValueError: ``depth`` is below 1.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    items: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def _put(item) -> bool:
        while not stop.is_set():
            try:
                items.put(item, timeout=_POLL_S)
                return True
            except queue.Full:
                continue
        return False

    def _produce() -> None:
        source = iter(iterable)
        try:
            for item in source:
                if not _put(item):
                    return
            _put(_DONE)
        except (
            BaseException
        ) as exc:  # noqa: BLE001 — carried to the consumer and re-raised there
            _put(_Raised(exc))
        finally:
            _close(source)

    producer = threading.Thread(target=_produce, name="lookahead-producer", daemon=True)
    producer.start()
    try:
        while True:
            item = items.get()
            if item is _DONE:
                return
            if isinstance(item, _Raised):
                raise item.exc
            yield item
    finally:
        stop.set()
        producer.join()


def ordered_map(
    fn: Callable[[T], R], iterable: Iterable[T], workers: int, window: int
) -> Iterator[R]:
    """Yield ``fn(item)`` for each item of ``iterable``, in input order, computed on ``workers``
    threads at most ``window`` items ahead of the consumer.

    An exception from ``fn`` is raised at its item's position; one from ``iterable`` itself is
    raised after every item it did produce has been yielded — the order a serial loop would
    have failed in. Once ``fn`` has raised, it is started on no later item, so a side-effecting
    ``fn`` (a writer, ``workers=1``) stops where the serial loop would.

    Raises:
        ValueError: ``workers`` or ``window`` is below 1.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    source = iter(iterable)
    pending: deque[concurrent.futures.Future] = deque()
    source_error: BaseException | None = None
    exhausted = False
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="lookahead-worker"
    )
    failed = threading.Event()

    def _call(item):
        if failed.is_set():
            raise _Abandoned  # never surfaced: an earlier item's error is raised first
        try:
            return fn(item)
        except BaseException:
            failed.set()
            raise

    def _fill() -> None:
        nonlocal exhausted, source_error
        while not exhausted and len(pending) < window:
            try:
                item = next(source)
            except StopIteration:
                exhausted = True
                return
            except Exception as exc:  # raised in order, after the items before it
                source_error = exc
                exhausted = True
                return
            pending.append(pool.submit(_call, item))

    try:
        _fill()
        while pending:
            result = pending.popleft().result()
            _fill()
            yield result
        if source_error is not None:
            raise source_error
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        _close(source)
