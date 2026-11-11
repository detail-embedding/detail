"""`core.lru.CountLRU`: the one bounded LRU the readers share (its module docstring is the
write-up). The edges a hand-rolled copy gets wrong: a hit must refresh recency, a re-put of a
held key must refresh it too rather than count twice, and the drain must be oldest-first.
"""

import threading

import pytest

from detail.core.lru import CountLRU


def test_a_put_past_capacity_drops_the_oldest_first():
    cache = CountLRU(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert "a" not in cache
    assert cache.get("b") == 2 and cache.get("c") == 3
    assert len(cache) == 2


def test_a_hit_makes_its_entry_the_newest():
    cache = CountLRU(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # "a" is now the newest, "b" the oldest
    cache.put("c", 3)
    assert "b" not in cache and "a" in cache and "c" in cache


def test_a_re_put_refreshes_rather_than_duplicating():
    cache = CountLRU(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)  # refresh, not a third entry
    assert len(cache) == 2
    cache.put("c", 3)
    assert "b" not in cache
    assert cache.get("a") == 10


def test_a_miss_returns_the_default_and_holds_nothing():
    cache = CountLRU(1)
    assert cache.get("missing") is None
    assert cache.get("missing", default="fallback") == "fallback"
    assert len(cache) == 0


def test_a_held_none_is_a_hit_not_a_miss():
    """A reader may legitimately cache "nothing here" (a frame with no mask)."""
    cache = CountLRU(1)
    cache.put("empty", None)
    assert "empty" in cache
    assert cache.get("empty", default="fallback") is None


def test_clear_empties_it():
    cache = CountLRU(3)
    for key in "abc":
        cache.put(key, key)
    cache.clear()
    assert len(cache) == 0 and "a" not in cache


@pytest.mark.parametrize("bad", [0, -1])
def test_a_capacity_below_one_is_refused(bad):
    with pytest.raises(ValueError, match="at least 1"):
        CountLRU(bad)


def test_many_threads_never_push_it_past_capacity():
    """Sixteen threads putting and getting at once: no exception escapes, and the bound holds
    at the end — the invariant a lock-free copy would break under the montage's thread pool.
    """
    cache = CountLRU(8)
    threads = 16
    start = threading.Barrier(threads)
    failures: list[BaseException] = []

    def hammer(seed: int) -> None:
        try:
            start.wait()
            for i in range(300):
                key = (seed * 7 + i) % 40
                cache.put(key, seed)
                cache.get((key + 1) % 40)
                _ = key in cache
        except BaseException as exc:  # a worker's failure must reach the assertion
            failures.append(exc)

    workers = [threading.Thread(target=hammer, args=(n,)) for n in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    assert not failures, failures
    assert all(not worker.is_alive() for worker in workers)
    assert len(cache) <= 8
