"""A bounded, thread-safe least-recently-used cache, counted in entries.

**The class this replaces.** A reader that serves a short span out of a larger block it already
decompressed wants to keep a few recent blocks; a playback surface that shows the frames before
the one on screen wants to keep a few hundred recent crops. Each was written as its own
``OrderedDict`` + ``Lock`` + ``move_to_end`` + ``popitem(last=False)`` drain loop, and a third
copy was on its way (the deleted montage window's rendered-tile cache, which had a byte-bounded
twin here until the window went). CLAUDE.md: two copies of a shape can be tolerated; the third is
when it becomes one mechanism. This is that mechanism.

Deliberately not :mod:`detail.storage.read_cache`: that store is keyed on a file's path and
mtime and is unbounded, which is right for small records a page re-reads and wrong for
megabyte blocks a montage would read for sixty recordings. Deliberately not
``functools.lru_cache``: the key is chosen by the caller, the value may be computed outside the
lock (a zarr read must not hold up every other block), and the cache is per instance, not per
process.

The compute happens outside the lock on purpose: two threads that miss the same key at once
both compute it, and the second ``put`` wins. For a read that is idempotent and cheap next to
holding every other reader, that is the right trade; a caller that must compute once per key
takes :func:`detail.storage.read_cache.computing` around the miss.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

_MISSING = object()


class CountLRU(Generic[K, V]):
    """At most ``capacity`` entries; a hit makes its entry the newest, a put past capacity drops
    the oldest. Every method takes the cache's own lock, so any thread may call any of them.

    Raises:
        ValueError: ``capacity`` is not a positive integer.
    """

    def __init__(self, capacity: int) -> None:
        if int(capacity) < 1:
            raise ValueError(f"an LRU needs a capacity of at least 1, not {capacity!r}")
        self._capacity = int(capacity)
        self._kept: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def get(self, key: K, default: V | None = None) -> V | None:
        """The value under ``key``, made the newest entry; ``default`` when it is not held."""
        with self._lock:
            found = self._kept.get(key, _MISSING)
            if found is _MISSING:
                return default
            self._kept.move_to_end(key)
            return found

    def put(self, key: K, value: V) -> None:
        """Hold ``value`` under ``key`` as the newest entry, dropping the oldest past capacity."""
        with self._lock:
            self._kept[key] = value
            self._kept.move_to_end(key)
            while len(self._kept) > self._capacity:
                self._kept.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._kept.clear()

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._kept

    def __len__(self) -> int:
        with self._lock:
            return len(self._kept)
