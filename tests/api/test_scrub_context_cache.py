"""A cache whose check and fill are two unguarded statements is a race, not a cache.

**The class.** Under one thread, "if the key is missing, build it" memoises perfectly. Under N
threads it does not memoise at all until the first build finishes: everyone who arrives during
that window also misses, and every one of them does the whole build. Nothing is wrong at the line
where it is written, nothing raises, and the symptom is only that the thing is slow — and slower
the more threads you give it, which is the opposite of what anyone looking at it expects.

**The instance.** `ClipQueryPreview._scrub_ctx` opens a recording's tracking metadata and
concatenates its timestamp map. Thumbnail strips start one pool task per clip at once and every
task lands here first, so a cold clustering page built the same context once per worker. Measured
on a real experiment: ten thumbnails across two recordings built it **ten** times.

**Why these assert a count and not a duration.** This machine routinely runs three or more agent
sessions, several driving the GUI suite, so a wall clock here measures whoever else is busy — the
first version of that measurement reported "44x" and was mostly contention; interleaved and taken
as a minimum it is ~5x. A construction count is a property of the code: the same on an idle box
and a loaded one, and it fails on exactly this bug. The repo already records the rule for the
network-share class — count the syscalls, do not time them.

**Why the builder is substituted here.** What changed is the guard around the build, not the
build, so these drive `_scrub_ctx` against a stand-in builder that sleeps. That is deliberate on
two counts: it makes the threads genuinely overlap (a real build that happens to be fast would
let the first thread finish before the others arrive, and the race would go unobserved), and it
keeps the test off video decoding entirely. The real builder is exercised by the callers of
`_scrub_ctx` throughout `tests/api` and `tests/gui`.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from detail.api.run_preview import ClipQueryPreview

KEY = ("recording_001", "emb_01")
OTHER = ("recording_002", "emb_01")


def _preview_with_a_counting_builder(tmp_path, delay=0.05, delays=None):
    """A preview whose context build is a slow stand-in that records every call.

    ``delays`` gives a per-key duration where one key has to be slow and another fast — which is
    what separates "B waited for A" from "B did its own work", the only way to see whether the
    lock is per key or shared.
    """
    preview = ClipQueryPreview(tmp_path)
    calls: list[tuple[str, str]] = []
    guard = threading.Lock()

    def build(recording_id, embed_run_id):
        key = (recording_id, embed_run_id)
        with guard:
            calls.append(key)
        # long enough that every waiting thread is genuinely inside the window
        time.sleep((delays or {}).get(key, delay))
        return (f"ctx-{recording_id}", None)

    preview._build_scrub_ctx = build
    return preview, calls


def _all_at_once(preview, key, workers):
    """`workers` threads asking for the same key, released together."""
    barrier = threading.Barrier(workers)

    def ask(_i):
        barrier.wait()  # without this the first caller wins and there is no race to observe
        return preview._scrub_ctx(*key)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(ask, range(workers)))


def test_concurrent_callers_build_one_context_once(tmp_path):
    """The number that must not scale with the thread count."""
    preview, calls = _preview_with_a_counting_builder(tmp_path)

    results = _all_at_once(preview, KEY, workers=8)

    assert len(calls) == 1, (
        f"8 concurrent callers built the context {len(calls)} times. The check and the fill "
        "belong inside one lock, or every thread arriving during a build repeats it."
    )
    assert (
        results == [("ctx-recording_001", None)] * 8
    ), "they must all get the same context"


def test_the_count_would_catch_an_unguarded_cache(tmp_path):
    """Guard the guard.

    Put the original race back and the count above has to move. Without this, a `_scrub_ctx`
    that had stopped being concurrent — or threads that never actually overlap — would satisfy
    the test above forever, which is how a guard passes while guarding nothing.
    """
    preview, calls = _preview_with_a_counting_builder(tmp_path)

    def unguarded(recording_id, embed_run_id):
        key = (recording_id, embed_run_id)
        if key not in preview._scrub:
            preview._scrub[key] = preview._build_scrub_ctx(recording_id, embed_run_id)
        return preview._scrub[key]

    preview._scrub_ctx = unguarded
    _all_at_once(preview, KEY, workers=8)

    assert len(calls) > 1, (
        "the unguarded version built it once, so these threads are not overlapping and the "
        "test above proves nothing"
    )


def test_a_later_caller_gets_the_cached_context_without_building(tmp_path):
    """The ordinary path: once built, nobody builds again."""
    preview, calls = _preview_with_a_counting_builder(tmp_path)

    first = preview._scrub_ctx(*KEY)
    second = preview._scrub_ctx(*KEY)

    assert first is second
    assert len(calls) == 1


def test_one_recording_does_not_block_a_question_about_another(tmp_path):
    """Per key, not one lock for all — and this is the reason the distinction exists.

    `_scrub_ctx` is reached from the **UI thread** too (`frame_rate` off a selection change,
    `recording_frame_count` when a player opens). Under a single shared lock, a pool worker
    building recording A would freeze the window for somebody asking about recording B. The two
    designs measured the same in throughput, so there is nothing traded away for it.

    Asserted as elapsed time deliberately, and with a wide margin: the claim is only that B does
    not wait out A's whole build, which stays true under any load — a machine busy enough to
    break it would have broken A's build too.
    """
    # KEY is slow, OTHER is instant. Without the split this cannot tell "B waited for A" from
    # "B did its own slow build" — the first version of this test could not, and failed on
    # correct code for that reason.
    preview, calls = _preview_with_a_counting_builder(
        tmp_path, delays={KEY: 1.0, OTHER: 0.0}
    )

    started = threading.Event()
    real_build = preview._build_scrub_ctx

    def build(recording_id, embed_run_id):
        if (recording_id, embed_run_id) == KEY:
            started.set()
        return real_build(recording_id, embed_run_id)

    preview._build_scrub_ctx = build

    slow = threading.Thread(target=lambda: preview._scrub_ctx(*KEY), daemon=True)
    slow.start()
    assert started.wait(timeout=5), "the first build never started"

    t0 = time.perf_counter()
    preview._scrub_ctx(*OTHER)
    waited = time.perf_counter() - t0
    slow.join(timeout=10)

    assert waited < 0.5, (
        f"asking about {OTHER[0]} waited {waited:.2f}s while {KEY[0]} was building — that is a "
        "shared lock, and on the UI thread it is a frozen window"
    )
    assert sorted(calls) == sorted([KEY, OTHER]), "each key built its own context, once"


def test_locks_are_made_per_key_on_demand(tmp_path):
    """The structural half: no lock exists until a key needs one, and then exactly one does."""
    preview, _calls = _preview_with_a_counting_builder(tmp_path)

    assert preview._scrub_locks == {}
    preview._scrub_ctx(*KEY)
    preview._scrub_ctx(*KEY)
    assert list(preview._scrub_locks) == [KEY]
