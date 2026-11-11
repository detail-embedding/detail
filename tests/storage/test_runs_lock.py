"""runs_lock: two processing runs launched together must not claim the same ``run_NNN``.

``next_run_id`` is max+1 over a scan of ``runs/``, so between one launch's scan and its write
another launch can scan and get the same id. Both then write ``run.yaml.tmp`` to the one path
and the loser's atomic replace fails with ``FileNotFoundError`` — its tmp already consumed.
Measured for real: six embedding runs started together, one died on exactly that.

Each ``runs_lock`` opens its own file description, so ``fcntl.flock`` serializes concurrent
holders — the same kernel lock that serializes separate processes — which these thread-based
tests exercise directly. Remove the lock from ``phase`` below and it fails.
"""

import threading
import time

from detail.storage import runs as runs_store
from tests.concurrency import join_all, rendezvous


def _reserve(experiment_dir, out: list, barrier: threading.Barrier) -> None:
    """Allocate + reserve the way ``run_processing`` does, under the lock."""
    rendezvous(barrier)  # maximize the overlap rather than hoping for it
    with runs_store.runs_lock(experiment_dir):
        run_id = runs_store.next_run_id(experiment_dir)
        time.sleep(0.01)  # widen the scan->write window the race lives in
        runs_store.write_run(
            experiment_dir, run_id, {"id": run_id, "module_id": "embedding"}
        )
    out.append(run_id)


def test_concurrent_reservations_get_distinct_ids(tmp_path):
    out: list[str] = []
    n = 8
    barrier = threading.Barrier(n)
    threads = [
        threading.Thread(target=_reserve, args=(tmp_path, out, barrier))
        for _ in range(n)
    ]
    for t in threads:
        t.start()
    join_all(threads)

    assert len(out) == n, "a reservation raised instead of returning an id"
    assert len(set(out)) == n, f"colliding run ids under concurrency: {sorted(out)}"
    assert sorted(out) == [f"run_{i:03d}" for i in range(1, n + 1)]
    # Every record actually landed — a collision loses one silently as well as raising.
    assert sorted(runs_store.discover_runs(tmp_path)) == sorted(out)


def test_the_race_is_real_without_the_lock(tmp_path):
    """Guard the guard: the same reservation without the lock must collide, or the test
    above is passing for a reason other than the lock."""
    out: list[str] = []
    n = 8
    barrier = threading.Barrier(n)

    def unlocked() -> None:
        rendezvous(barrier)
        run_id = runs_store.next_run_id(tmp_path)
        time.sleep(0.01)
        try:
            runs_store.write_run(
                tmp_path, run_id, {"id": run_id, "module_id": "embedding"}
            )
        except OSError:
            # The loser's tmp file was consumed by the winner's replace. Swallowed here
            # because *this* test wants the collision, and an escaping thread exception
            # is a warning, which the suite turns into an error.
            out.append("raised")
            return
        out.append(run_id)

    threads = [threading.Thread(target=unlocked) for _ in range(n)]
    for t in threads:
        t.start()
    join_all(threads)
    ids = [r for r in out if r != "raised"]
    collided = len(set(ids)) < n or "raised" in out
    assert collided, f"expected colliding ids or a failed write without the lock: {out}"


def test_runs_lock_creates_the_directory_it_guards(tmp_path):
    """The first launch in a fresh experiment locks before ``runs/`` exists."""
    with runs_store.runs_lock(tmp_path):
        pass
    assert (tmp_path / "runs").is_dir()


# --- the same hazard, one more registry ------------------------------------------------ #
# Two id counters, one bug (a third, `clusterers`, went with the clusterer chain). `runs`
# needed a lock (the record is written outside the directory claim); `analysis` claims the
# directory itself, which is atomic and needs none. What matters is that neither is a bare
# scan-then-write.


def test_concurrent_analysis_reservations_get_distinct_ids(tmp_path):
    from detail.storage import analysis_runs

    out: list[str] = []
    n = 8
    barrier = threading.Barrier(n)

    def reserve() -> None:
        rendezvous(barrier)
        out.append(analysis_runs.reserve_analysis_id(tmp_path))

    threads = [threading.Thread(target=reserve) for _ in range(n)]
    for t in threads:
        t.start()
    join_all(threads)
    assert len(set(out)) == n, f"colliding analysis ids: {sorted(out)}"
    # Each claim is its directory. `discover_analyses` deliberately does not see them yet —
    # it counts runs with a record — which is exactly why the claim scan cannot use it.
    assert all(analysis_runs.analysis_dir(tmp_path, r).is_dir() for r in out)
    assert analysis_runs.discover_analyses(tmp_path) == []


def test_the_analysis_race_is_real_without_the_claim(tmp_path):
    """Guard the guard: the unreserved scan must collide."""
    from detail.storage import analysis_runs

    out: list[str] = []
    n = 8
    barrier = threading.Barrier(n)

    def unreserved() -> None:
        rendezvous(barrier)
        run_id = analysis_runs.next_analysis_id(tmp_path)
        time.sleep(0.01)
        analysis_runs.analysis_dir(tmp_path, run_id).mkdir(parents=True, exist_ok=True)
        out.append(run_id)

    threads = [threading.Thread(target=unreserved) for _ in range(n)]
    for t in threads:
        t.start()
    join_all(threads)
    assert len(set(out)) < n, "expected colliding analysis ids without the claim"
