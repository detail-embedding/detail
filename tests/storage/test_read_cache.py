"""mtime-invalidated read cache (storage/read_cache.py), and the class it exists for.

**The class: a cached document re-validated once per field access.** The cache makes the *parse* free
and leaves the **round trip**, so a caller wanting several fields of one record pays per field. Every
`Recording` property is its own `read_recording`, and the recordings table asked one `recording.yaml`
for eight of them — **round trips per field, per recording, and seconds of frozen window** on a
network share.

Each property is obviously cheap; only the loop is not. That is why it survived review, and why a
wall clock is the wrong instrument: on a local disk the same loop is too fast to notice.

The rule: **count syscalls, assert the slope rather than a total, and pin the burst.**
`read_cache.pinned()` is thread-local and refcounted, and a hit inside it skips the `stat`
revalidation — so it is scoped to a synchronous burst, never around a wait or a user interaction. A
miss still reads, and every writer drops its own entry, so a pin skips validation rather than
freezing the cache.

`tests/gui/test_page_read_cost.py` holds the surface-level half, where the count is asserted against
a real page build.

**The second class here: a memo whose miss is check, compute, store.** Every thread that misses
before the first store computes too, so a cold cache under a thread pool does the work once per
thread. A montage prepares its clips on several threads, and the clips sharing a recording all
missed its `recording.yaml` together: one read and parse of a 35 KB file per thread, each inflated
to ~160 ms by the other threads holding the interpreter. `read_cache.computing(key)` is a per-key
lock the miss path takes and re-checks under (`cached_read`, `cached_document`, and
`recording.read_recording`'s own cache). The guards hold six threads inside the computation at
once, so a cache that lets them all compute counts 6 every time, not only when the scheduler
interleaves them; `tests/storage/test_recording.py` and
`tests/api/test_clip_montage.py::test_a_cold_prepare_parses_each_recording_once` apply the same
rendezvous to `read_recording` and to a real montage prepare.
"""

import os

from detail.storage import read_cache
from tests.round_trips import count_stats


def _bump_mtime(path):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def test_cached_read_memoizes_until_mtime_changes(tmp_path):
    read_cache.clear_read_cache()
    stamp = tmp_path / "run.yaml"
    stamp.write_text("x", encoding="utf-8")
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return [1, 2, 3]

    assert read_cache.cached_read("k", stamp, compute) == [1, 2, 3]
    assert read_cache.cached_read("k", stamp, compute) == [1, 2, 3]
    assert calls["n"] == 1  # second read is a cache hit — compute ran once

    _bump_mtime(stamp)  # the stamped output was rewritten
    assert read_cache.cached_read("k", stamp, compute) == [1, 2, 3]
    assert calls["n"] == 2  # mtime changed → recomputed


def test_cached_read_key_isolates_entries(tmp_path):
    read_cache.clear_read_cache()
    stamp = tmp_path / "run.yaml"
    stamp.write_text("x", encoding="utf-8")
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return calls["n"]

    assert read_cache.cached_read("a", stamp, compute) == 1
    assert read_cache.cached_read("b", stamp, compute) == 2  # distinct key → own entry
    assert read_cache.cached_read("a", stamp, compute) == 1  # still cached
    assert calls["n"] == 2


def test_cached_read_does_not_cache_missing_stamp(tmp_path):
    read_cache.clear_read_cache()
    stamp = tmp_path / "absent.yaml"  # never created
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return 7

    assert read_cache.cached_read("k", stamp, compute) == 7
    assert read_cache.cached_read("k", stamp, compute) == 7
    assert calls["n"] == 2  # a missing stamp is computed every time, never cached


def test_cached_read_does_not_memoize_errors(tmp_path):
    read_cache.clear_read_cache()
    stamp = tmp_path / "run.yaml"
    stamp.write_text("x", encoding="utf-8")
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("bad")

    for _ in range(2):
        try:
            read_cache.cached_read("k", stamp, boom)
        except ValueError:
            pass
    assert calls["n"] == 2  # the raise is not cached — it recomputes


def test_clear_read_cache_forces_recompute(tmp_path):
    read_cache.clear_read_cache()
    stamp = tmp_path / "run.yaml"
    stamp.write_text("x", encoding="utf-8")
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return 1

    read_cache.cached_read("k", stamp, compute)
    read_cache.clear_read_cache()
    read_cache.cached_read("k", stamp, compute)
    assert calls["n"] == 2  # cleared → recomputed even though mtime is unchanged


# --------------------------------------------------------------------------- #
# cached_document — the small records a page build re-reads
# --------------------------------------------------------------------------- #
def test_cached_document_parses_once_until_the_file_changes(tmp_path):
    read_cache.clear_read_cache()
    path = tmp_path / "run.yaml"
    path.write_text("status: running\n", encoding="utf-8")
    calls = {"n": 0}

    def parse():
        calls["n"] += 1
        return {"status": path.read_text(encoding="utf-8-sig").split(": ")[1].strip()}

    assert read_cache.cached_document(path, parse)["status"] == "running"
    assert read_cache.cached_document(path, parse)["status"] == "running"
    assert calls["n"] == 1

    path.write_text("status: complete\n", encoding="utf-8")  # the writer replaced it
    _bump_mtime(path)
    assert read_cache.cached_document(path, parse)["status"] == "complete"
    assert calls["n"] == 2


def test_a_caller_mutating_its_record_does_not_poison_the_cache(tmp_path):
    """The whole reason this returns a copy and ``cached_read`` does not.

    ``Experiment._fail_run_record`` and ``_reconcile_run`` both read a run record, mutate
    it, and write it back. Handing out the cached object would make the first of those the
    answer every later reader gets — a run that reads ``failed`` because somebody else was
    about to fail it, with nothing on disk saying so.
    """
    read_cache.clear_read_cache()
    path = tmp_path / "run.yaml"
    path.write_text("x", encoding="utf-8")

    first = read_cache.cached_document(
        path, lambda: {"status": "running", "per": {"a": 1}}
    )
    first["status"] = "failed"
    first["per"]["a"] = 99  # nested, so a shallow copy would not be enough

    second = read_cache.cached_document(path, lambda: {"unreachable": True})
    assert second == {"status": "running", "per": {"a": 1}}


def test_size_is_part_of_the_signature(tmp_path):
    """A second signal for a filesystem whose timestamp granularity could hide two
    writes inside one tick — the mtime is the primary one."""
    read_cache.clear_read_cache()
    path = tmp_path / "run.yaml"
    path.write_text("a", encoding="utf-8")
    assert read_cache.cached_document(path, lambda: "first") == "first"

    st = path.stat()
    path.write_text("aa", encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # same mtime, different size
    assert read_cache.cached_document(path, lambda: "second") == "second"


def test_forgetting_a_document_reaches_what_the_signature_cannot(tmp_path):
    """A writer drops its own entry: a same-size rewrite inside one clock tick has the old
    signature, and a pinned read never looks at the signature at all."""
    read_cache.clear_read_cache()
    path = tmp_path / "run.yaml"
    other = tmp_path / "other.yaml"
    path.write_text("a", encoding="utf-8")
    other.write_text("a", encoding="utf-8")
    assert read_cache.cached_document(path, lambda: "first") == "first"
    assert read_cache.cached_document(other, lambda: "other") == "other"

    st = path.stat()
    path.write_text("b", encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # same mtime, same size
    assert (
        read_cache.cached_document(path, lambda: "second") == "first"
    )  # the blind spot
    with read_cache.pinned():
        assert read_cache.cached_document(path, lambda: "second") == "first"
        read_cache.forget_document(path)
        assert read_cache.cached_document(path, lambda: "second") == "second"
    assert read_cache.cached_document(other, lambda: "never") == "other"  # only the one
    read_cache.forget_document(tmp_path / "never_read.yaml")  # nothing to drop is fine


def test_a_missing_file_raises_rather_than_caching(tmp_path):
    read_cache.clear_read_cache()
    try:
        read_cache.cached_document(tmp_path / "gone.yaml", lambda: "never")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing document must raise, not return a parse")


# --------------------------------------------------------------------------- #
# pinned(): trusting the cache for one burst of reads
# --------------------------------------------------------------------------- #


def test_a_pin_serves_a_hit_without_re_validating_it(tmp_path):
    """The class: **a cached document re-validated once per field access.**

    A hit costs one ``stat`` per *call*, so a caller wanting several fields of one record
    pays per field — every ``Recording`` property is its own read, and the recordings table
    asked one ``recording.yaml`` for eight of them, nine round trips a row. Nothing is wrong
    at the line where it is written; only the loop is.
    """
    read_cache.clear_read_cache()
    path = tmp_path / "run.yaml"
    path.write_text("a: 1", encoding="utf-8")
    read_cache.cached_document(path, lambda: {"a": 1})  # warm

    _, unpinned = count_stats(
        tmp_path,
        lambda: [read_cache.cached_document(path, lambda: {"a": 1}) for _ in range(8)],
    )

    def pinned_reads():
        with read_cache.pinned():
            return [
                read_cache.cached_document(path, lambda: {"a": 1}) for _ in range(8)
            ]

    values, pinned = count_stats(tmp_path, pinned_reads)

    assert unpinned == 8, "guard the guard: eight reads must cost eight stats unpinned"
    assert pinned == 0
    assert values == [{"a": 1}] * 8  # ...and it is the record, not a stale shape


def test_a_pin_still_reads_what_it_has_never_seen(tmp_path):
    """A miss inside a pin behaves exactly as outside — the pin skips validation, not the
    read. Otherwise a cold cache would serve nothing and the page would render empty."""
    read_cache.clear_read_cache()
    path = tmp_path / "run.yaml"
    path.write_text("a: 1", encoding="utf-8")

    with read_cache.pinned():
        assert read_cache.cached_document(path, lambda: {"a": 1}) == {"a": 1}


def test_a_write_inside_a_pin_is_seen_by_the_next_read(tmp_path):
    """The pin skips *validation*; it does not freeze the cache.

    Every writer drops its own entry, so a read after a write is a miss and goes to disk.
    This is the property that keeps a pin safe to wrap around a build that also writes —
    asserted rather than assumed, because the failure would be silent.
    """
    read_cache.clear_read_cache()
    path = tmp_path / "run.yaml"
    path.write_text("a: 1", encoding="utf-8")

    with read_cache.pinned():
        assert read_cache.cached_document(path, lambda: {"a": 1}) == {"a": 1}
        path.write_text("a: 2", encoding="utf-8")
        read_cache.clear_read_cache()  # what a writer does for its own entry
        assert read_cache.cached_document(path, lambda: {"a": 2}) == {"a": 2}


def test_a_pin_on_one_thread_does_not_reach_another(tmp_path):
    """Thread-local, or one thread's convenience silently stales another's data.

    These caches are shared and the GUI reads them off the UI thread — a process-global flag
    would mean a pinned table build on the UI thread stopped a worker noticing a file had
    changed underneath it. Driven through a **real** thread: a global flag would otherwise
    look identical here.
    """
    import threading

    read_cache.clear_read_cache()
    path = tmp_path / "run.yaml"
    path.write_text("a: 1", encoding="utf-8")
    read_cache.cached_document(path, lambda: {"a": 1})  # warm
    seen = {}

    def other() -> None:
        seen["pinned"] = read_cache.is_pinned()
        _, seen["stats"] = count_stats(
            tmp_path, lambda: read_cache.cached_document(path, lambda: {"a": 1})
        )

    with read_cache.pinned():
        assert read_cache.is_pinned()
        worker = threading.Thread(target=other)
        worker.start()
        worker.join()

    assert seen["pinned"] is False
    assert seen["stats"] == 1, "the other thread skipped validation — the pin is global"


def test_a_pin_unwinds_to_where_it_started():
    """Refcounted: a nested pin must not leave the outer scope unpinned on the way out, and
    the outermost must leave the thread unpinned even when its body raises."""
    assert not read_cache.is_pinned()
    with read_cache.pinned():
        with read_cache.pinned():
            assert read_cache.is_pinned()
        assert read_cache.is_pinned(), "the inner exit unpinned the outer block"
    assert not read_cache.is_pinned()

    try:
        with read_cache.pinned():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not read_cache.is_pinned(), "a raising body left the thread pinned"


# --- concurrent misses compute once ---------------------------------------------------------- #
def _miss_together(n, call, enter):
    """Run ``call`` on ``n`` threads released at once; ``enter()`` is what the computation calls.

    ``enter`` waits (briefly) for every thread to be inside the computation together, so a cache
    that lets concurrent misses all compute is caught every time rather than when the scheduler
    happens to interleave them. A cache that computes once has one thread inside, and the wait
    times out.
    """
    import threading

    inside = threading.Barrier(n)
    start = threading.Barrier(n)
    entered = []

    def computation():
        entered.append(1)
        try:
            inside.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass
        return enter()

    def worker():
        start.wait()
        call(computation)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return len(entered)


def test_concurrent_misses_of_one_read_compute_it_once(tmp_path):
    """**A cache every caller races is a cache nobody has**: check-compute-store let every thread
    that missed before the first store compute too (``read_cache.computing``)."""
    read_cache.clear_read_cache()
    stamp = tmp_path / "track.yaml"
    stamp.write_text("x", encoding="utf-8")
    computed = _miss_together(
        6, lambda compute: read_cache.cached_read(("k",), stamp, compute), lambda: 1
    )
    assert computed == 1


def test_concurrent_misses_of_one_document_parse_it_once(tmp_path):
    read_cache.clear_read_cache()
    path = tmp_path / "run.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    parsed = _miss_together(
        6, lambda parse: read_cache.cached_document(path, parse), lambda: {"a": 1}
    )
    assert parsed == 1


def test_misses_of_different_keys_do_not_wait_for_each_other(tmp_path):
    """Per key, never global: two keys missed together both compute at once."""
    import threading

    read_cache.clear_read_cache()
    stamps = [tmp_path / "a.yaml", tmp_path / "b.yaml"]
    for stamp in stamps:
        stamp.write_text("x", encoding="utf-8")
    keys = iter(range(2))
    pick = threading.Lock()

    def call(compute):
        with pick:
            index = next(keys)
        read_cache.cached_read(("k", index), stamps[index], compute)

    assert _miss_together(2, call, lambda: 1) == 2
