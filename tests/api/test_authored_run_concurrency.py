"""Two authored edits at once (§21).

§21 calls authored operations "strictly sequential — the user acts between operations, so two at
once is a bug rather than a concurrency problem to solve". That held for the *compute* operations it
was written about and was quietly false for state edits: every one runs off the UI thread through
``gui/busy.py``, so dragging the granularity slider twice, or naming while a granularity change is
still in flight, runs two ``_edit`` calls concurrently.

Two things break when they interleave, and both are silent:

- the state rewrite is read-modify-write, so the second writer overwrites what the first just did;
- ``_project`` rewrites the frame views, and two writers race on one zarr store — seen twice in five
  full-suite runs, once as ``an array exists ... at path 't_ns'`` and once as a vanished
  ``zarr.<uuid>.partial``, both from one call ``rmtree``-ing the staging directory the other was
  writing into.

These tests drive real concurrency rather than asserting a lock exists, because a lock held in the
wrong place would still pass the latter.
"""

import threading

import pytest

from detail.api import authored_run


class _Run:
    """The smallest thing ``_edit`` needs: a state on disk, and a projection that touches a file.

    Not a clustering run — the point is the funnel, not the module. A real fit would put a minute
    on each of these and test the same three lines.
    """

    kind = "test run"

    def __init__(self, tmp_path):
        self._dir = tmp_path
        self._path = tmp_path / "state.txt"
        self._path.write_text("", encoding="utf-8")
        self.projections = 0
        self.overlap = False
        self._projecting = False
        self.stale_marks = []

    # -- the four hooks a subclass supplies -------------------------------------------------- #
    def _run_dir(self):
        return self._dir

    def _read_state(self):
        return list(filter(None, self._path.read_text(encoding="utf-8-sig").split(",")))

    def _write_state(self, state):
        # A deliberately slow write, so an unserialized second edit has time to read the old
        # value and clobber this one — the race is real but narrow, and a test that never loses
        # is a test that proves nothing.
        text = ",".join(state)
        import time

        time.sleep(0.02)
        self._path.write_text(text, encoding="utf-8")

    def _log(self, entry):
        pass

    def _set_stale(self, stale):
        # The real one rewrites `analysis.yaml`; this run has none. Recorded rather than
        # ignored so the mark stays visible here too: an edit that ran without one is a
        # different edit from the one under test.
        self.stale_marks.append(stale)

    @property
    def artifact_stale(self):
        # The reader for the marks above, which `_edit` consults to decide what it may keep.
        return bool(self.stale_marks and self.stale_marks[-1])

    def _project(self, *, progress=None, reuse=False):
        if self._projecting:
            self.overlap = True
        self._projecting = True
        import time

        time.sleep(0.02)
        self._projecting = False
        self.projections += 1

    @property
    def status(self):
        return "open"

    # -- borrowed wholesale, which is what is under test ------------------------------------- #
    _edit = authored_run.AuthoredRun._edit
    _refuse_if_finalized = authored_run.AuthoredRun._refuse_if_finalized
    _refuse_if_read_only = authored_run.AuthoredRun._refuse_if_read_only
    editable = True
    id = "an_01"


def _append(name):
    def mutate(state):
        state.append(name)
        return {}

    return mutate


def test_two_edits_at_once_do_not_lose_one(tmp_path):
    """The lost update. Without the lock the two threads read the same state and the second write
    overwrites the first, so one of the names simply is not there afterwards."""
    run = _Run(tmp_path)
    threads = [
        threading.Thread(target=lambda n=n: run._edit("name", _append(n)))
        for n in ("rearing", "grooming")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(run._read_state()) == ["grooming", "rearing"]


def test_two_edits_at_once_never_project_concurrently(tmp_path):
    """The frame-view race, at the seam that causes it. Two reprojections writing one zarr store is
    what produced the intermittent `.zarr.tmp` failures; the flag catches an overlap directly
    rather than hoping the filesystem misbehaves during the test."""
    run = _Run(tmp_path)
    threads = [
        threading.Thread(target=lambda n=n: run._edit("name", _append(n)))
        for n in ("a", "b", "c", "d")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not run.overlap, "two projections ran at once"
    assert run.projections == 4


def test_edits_on_different_runs_do_not_block_each_other(tmp_path):
    """The lock is per run directory, not global: one person naming clusters must not hold up an
    unrelated run. Distinct locks is the assertion — timing would be flaky."""
    (tmp_path / "a").mkdir(exist_ok=True)
    (tmp_path / "b").mkdir(exist_ok=True)
    first, second = _Run(tmp_path / "a"), _Run(tmp_path / "b")
    assert authored_run._edit_lock(first._run_dir()) is not authored_run._edit_lock(
        second._run_dir()
    )
    assert authored_run._edit_lock(first._run_dir()) is authored_run._edit_lock(
        first._run_dir()
    )


def test_an_edit_that_raises_releases_the_lock(tmp_path):
    """Otherwise the first bad edit wedges the run forever, which is worse than the race."""
    run = _Run(tmp_path)

    def boom(state):
        raise ValueError("no")

    with pytest.raises(ValueError):
        run._edit("name", boom)

    run._edit("name", _append("after"))
    assert run._read_state() == ["after"]
