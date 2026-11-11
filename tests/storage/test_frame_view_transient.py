"""A frame-view write rides out the transient network-share replace, on every call.

WinError 32 is SMB-specific and a fixer on Linux/macOS cannot provoke it, so it is
injected: one ``PermissionError`` shaped like the production one, raised on the first
``Path.replace`` zarr makes internally whose destination matches a marker. That is
exactly where the real failure landed —
``zarr/storage/_local.py::_atomic_write`` → ``tmp_path.replace(path)``.

Two injection points, because before the fix they behaved differently and that difference
*is* the bug:

- ``zarr.json`` — written by ``open_group``, which was wrapped in ``retry_on_lock``, so a
  transient there was already ridden out. It is the positive control: a scan that fires
  here and nowhere else proves the harness works and would have made a fixer conclude
  there was no bug.
- a chunk file under ``observed`` — written by ``create_array``, which was **not**
  wrapped. This is the production failure, and it is the test that flips from raising to
  writing cleanly now that the retry lives on the store.

Both must survive, and a third point (the store-directory swap) covers the rename that is
ours rather than zarr's.
"""

import os
import pathlib

import numpy as np

from detail.storage import frame_view


def _write(run_dir):
    axis = np.arange(30_000, dtype=np.int64) * 1_000_000
    return frame_view.write_frame_view(
        run_dir,
        recording_id="recording_001",
        subject=0,
        t_ns=axis,
        observed=np.ones(axis.shape, dtype=bool),
        instances=[],
    )


def _inject(monkeypatch, matches) -> dict:
    """Raise one WinError-32-shaped ``PermissionError`` on the first atomic replace whose
    destination path satisfies ``matches``. Returns a dict whose ``fired`` says it happened —
    asserted by every test here, because an injection that never fires passes vacuously.
    """
    state = {"fired": False}

    def trip(source, target) -> None:
        if not state["fired"] and matches(str(target)):
            state["fired"] = True
            raise PermissionError(
                32,
                "The process cannot access the file because it is being used by "
                "another process",
                str(source),
                None,
                str(target),
            )

    # Both renames, because the two halves of the write use different ones: zarr's internal
    # `_atomic_write` is `Path.replace`, and our own store-directory swap is `os.replace`
    # via `core.atomic`. Patching one leaves the other's transient uninjectable, which is
    # how a scan comes back clean on a path nobody covered.
    real_path_replace = pathlib.Path.replace
    real_os_replace = os.replace

    def flaky_path_replace(self, target):
        trip(self, target)
        return real_path_replace(self, target)

    def flaky_os_replace(source, target, **kwargs):
        trip(source, target)
        return real_os_replace(source, target, **kwargs)

    monkeypatch.setattr(pathlib.Path, "replace", flaky_path_replace)
    monkeypatch.setattr(os, "replace", flaky_os_replace)
    return state


def test_group_metadata_write_survives_a_transient(tmp_path, monkeypatch):
    """The positive control — this one was covered before the fix, and must stay so."""
    state = _inject(monkeypatch, lambda target: target.endswith("zarr.json"))
    assert _write(tmp_path).is_dir()
    assert state["fired"], "the injection point was never reached"


def test_chunk_write_survives_a_transient(tmp_path, monkeypatch):
    """The production failure: an unwrapped ``create_array`` writing chunk data.

    Before the retry moved to the store this raised ``PermissionError`` part-way into a
    long run.
    """
    # A chunk file, not the array's own metadata: `observed/c/0` is what the production
    # stack died on, and it is written by the same `create_array` call.
    state = _inject(
        monkeypatch,
        lambda target: "observed" in target and not target.endswith("zarr.json"),
    )
    path = _write(tmp_path)
    assert state["fired"], "the injection point was never reached"
    view = frame_view.read_frame_view(tmp_path, "recording_001", 0)
    axis, mask, _ = view.arrays()
    assert axis.size == 30_000 and bool(mask.all())
    assert not list(path.parent.glob("*.tmp")), "a staged store was left behind"


def test_the_store_directory_swap_survives_a_transient(tmp_path, monkeypatch):
    """The one replace zarr does not make: our own rename of the finished store into place.

    Reached only on a *rewrite*, which is what an authored run does on every reprojection —
    a granularity change, a naming — while the GUI may hold the store open.
    """
    _write(tmp_path)
    state = _inject(monkeypatch, lambda target: target.endswith(".zarr"))
    assert _write(tmp_path).is_dir()
    assert state["fired"], "the injection point was never reached"
    assert frame_view.discover_frame_views(tmp_path) == [("recording_001", 0)]
