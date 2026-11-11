"""mtime-invalidated cache for repeated reads of immutable run outputs.

The launch / selection surfaces read the same completed-run outputs — a tracking
run's track ids, a track set's largest mask — once per selected recording on every
selection change and every launch. Those outputs are immutable once the run is
complete, but each read re-opens a zarr; over many recordings that is the GUI lag
this removes. Cache the parsed result against a stamp file's mtime so a repeat is
a dict hit, and a rewrite (mtime change → the run was re-created) recomputes.
Thread-safe — the GUI resolves selections off the UI thread (see CLAUDE.md, "no
O(recordings) disk I/O on the UI thread").

:func:`cached_document` is the same idea one size down, for the small YAML records a page
build re-reads: a run record, an analysis record. Individually each is one file and nobody
would call it a slow seam — which is exactly how the cost hid. The project rail rebuilds
from every run record on **every navigation**, and on the SMB share these workspaces live
on a file read is ~12 ms, so an experiment with a few hundred analysis runs pays seconds to
move between two pages. ``recording.py`` has memoized ``recording.yaml`` this way for a
while; this is the same idiom, generalized, so the next small record a page build reads in
a loop does not grow a third private cache.
"""

import contextlib
import copy
import os
import threading
from collections.abc import Callable, Hashable
from collections.abc import Iterator
from pathlib import Path

_LOCK = threading.Lock()
_CACHE: dict[Hashable, tuple[int, object]] = {}
_DOCUMENTS: dict[str, tuple[tuple[int, int], object]] = {}
#: One lock per cached key, held while that key is computed (:func:`computing`).
_COMPUTING: dict[Hashable, threading.RLock] = {}

#: Depth of :func:`pinned` blocks on **this** thread (see the function's docstring for why
#: thread-local rather than global).
_PIN = threading.local()


def clear_read_cache() -> None:
    """Drop every cached read (tests; a hard experiment reload)."""
    with _LOCK:
        _CACHE.clear()
        _DOCUMENTS.clear()


def forget_document(path: Path) -> None:
    """Drop one :func:`cached_document` entry — for the writer that just replaced the file.

    The ``(mtime_ns, size)`` signature already catches a rewrite, with two exceptions a writer
    in this process can close for nothing: a read inside :func:`pinned`, which skips the
    ``stat``, and a same-size rewrite inside one tick of a coarse filesystem clock (re-pointing
    an alias to a path of equal length, milliseconds after a read).
    """
    with _LOCK:
        _DOCUMENTS.pop(os.path.abspath(path), None)


def computing(key: Hashable) -> threading.RLock:
    """The lock a cache holds while it computes ``key``: whoever misses first computes, and
    whoever misses the same key meanwhile waits for that result instead of computing its own.

    **A cache every caller races is a cache nobody has.** A memo whose miss is "check, then
    compute, then store" lets every thread that arrives before the first store compute too. A
    montage prepares its clips on several threads, and on a cold cache the clips that share a
    recording all missed its ``recording.yaml`` together: a 35 KB file parsed once per thread,
    each parse inflated from 64 ms to ~160 ms by the other threads holding the interpreter.
    ``ClipQueryPreview._scrub_ctx`` learned the same lesson for its decode contexts.

    Per key, never global: a thread computing one recording's record must not hold up a thread
    that wants another's. The caller re-checks its cache once it holds the lock, since the
    thread it waited for has normally just stored the answer. Re-entrant, so a computation that
    reaches its own key again (nothing does today) recomputes rather than deadlocks.
    """
    with _LOCK:
        lock = _COMPUTING.get(key)
        if lock is None:
            lock = _COMPUTING[key] = threading.RLock()
        return lock


def is_pinned() -> bool:
    """Whether this thread is inside a :func:`pinned` block."""
    return getattr(_PIN, "depth", 0) > 0


@contextlib.contextmanager
def pinned() -> Iterator[None]:
    """Trust the cache without re-validating it, for one synchronous burst of reads.

    A cached record costs one ``stat`` per *call*: the cache makes the parse free and leaves
    the round trip. That is right for a page that reads a record once, and wrong for a loop
    that wants several fields — every ``Recording`` property is its own
    ``read_recording(self.dir).get(...)``, so the recordings table read one
    ``recording.yaml`` **nine times per row**, seconds of frozen window on a workspace
    held on a network share. Nothing is wrong at the line where
    it is written; each property is obviously cheap, and only the loop is not.

    Inside this block a cache **hit** is returned without the ``stat``. A miss still reads, so
    a cold cache behaves exactly as it does outside. Every writer here drops its own entry
    (``write_recording``, ``clear_read_cache``), so a write inside a pin is seen by the read
    after it — the pin skips *validation*, it does not freeze the cache.

    **Thread-local, refcounted.** A pin on the UI thread must not change what a worker thread
    reads: these caches are shared, the GUI resolves selections off the UI thread, and a
    process-global flag would let one thread's convenience silently stale another's data.

    Scope it to a burst you want to be *internally consistent* — a table build, a selection
    resolve — never around a wait, a subprocess, or a user interaction. Within that scope it
    is also more correct than the unpinned form, which lets row 1 and row 60 come from
    different states of the world.
    """
    depth = getattr(_PIN, "depth", 0)
    _PIN.depth = depth + 1
    try:
        yield
    finally:
        _PIN.depth = depth


def cached_read(key: Hashable, stamp_path: Path, compute: Callable[[], object]):
    """``compute()`` for ``key``, cached until ``stamp_path``'s mtime changes.

    ``stamp_path`` is the file whose mtime signals the output changed — the run's
    completion marker for an otherwise-immutable output (a tracking run's
    ``track.yaml`` / ``tracks.zarr``). A missing ``stamp_path`` is computed but not
    cached, since the output may appear later. ``compute`` raising propagates
    without caching (an error is never memoized).

    The cached value is returned **by reference** (not copied) — a hit hands back the
    same object every caller shares, so a caller that mutates it corrupts every later
    read. Both current callers defend themselves (``track_ids`` wraps the result in
    ``list(...)``; ``largest_mask_px`` returns an ``int``); a mutating caller must copy.
    """
    try:
        mtime = stamp_path.stat().st_mtime_ns
    except FileNotFoundError:
        return compute()
    with _LOCK:
        hit = _CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    with computing(key):
        with _LOCK:
            hit = _CACHE.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        value = compute()
        with _LOCK:
            _CACHE[key] = (mtime, value)
    return value


def cached_document(path: Path, parse: Callable[[], object]) -> object:
    """``parse()`` for the file at ``path``, memoized against its ``(mtime_ns, size)``.

    For a small record a page build reads repeatedly. A ``stat`` is one round trip where a
    read is a round trip plus the transfer plus a YAML parse, so a hit is roughly free even
    on a network share — the cost that remains is the stat, and that is the cost of knowing
    the answer is still true.

    Every writer of these records replaces the file atomically (``core.atomic.replace``),
    which always moves the mtime, so a rewrite invalidates. ``size`` rides along as a second
    signal for the case a filesystem's timestamp granularity could hide two writes inside
    one tick.

    Returns a **deep copy**, unlike :func:`cached_read`: these are plain dicts that callers
    read-modify-write (``Experiment._fail_run_record`` and ``_reconcile_run`` both mutate the
    record and persist it), and one caller's edit must not become every later reader's
    record. Copying a parsed YAML dict is microseconds against a network read's
    milliseconds.

    Inside :func:`pinned` a hit is returned without the ``stat`` — see there for when that is
    the right trade and why it is thread-local.

    A missing file, or a ``parse`` that raises, propagates without caching.
    """
    # abspath, not resolve: normalizes a relative vs absolute spelling of the same file
    # (the signature keeps it correct either way; this bounds duplicate entries) without
    # resolve's symlink I/O, which on a share is another round trip.
    key = os.path.abspath(path)
    if is_pinned():
        with _LOCK:
            hit = _DOCUMENTS.get(key)
        if hit is not None:
            return copy.deepcopy(hit[1])  # the stat is what the pin skips
    stamp = path.stat()
    signature = (stamp.st_mtime_ns, stamp.st_size)
    with _LOCK:
        hit = _DOCUMENTS.get(key)
    if hit is not None and hit[0] == signature:
        return copy.deepcopy(hit[1])
    with computing(("document", key)):
        with _LOCK:
            hit = _DOCUMENTS.get(key)
        if hit is not None and hit[0] == signature:
            return copy.deepcopy(hit[1])
        value = parse()
        with _LOCK:
            _DOCUMENTS[key] = (signature, value)
    return copy.deepcopy(value)
