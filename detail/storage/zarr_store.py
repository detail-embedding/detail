"""The one seam every zarr store in this package is opened through.

zarr commits each metadata rewrite and each chunk file with its own
``tmp.replace(dst)`` inside :class:`zarr.storage.LocalStore`. On a network share that
rename-over-existing intermittently returns a transient ``PermissionError``
(WinError 5/32) — the same lock ``core.atomic`` already rides out for our own writers,
and which zarr's internal replace bypasses because it is not our call.

**The hazard was diagnosed once and mitigated per call site**, which is the bug this
module exists to end. ``storage/zarr_io.py`` wrapped each of its own ``create_array`` and
resize calls in ``atomic.retry_on_lock``; ``storage/frame_view.py``, written later,
inherited the diagnosis on its ``open_group`` and left its four ``create_array`` calls
bare — and those are the calls that write chunk data. A long run on an SMB share died
part-way, after hundreds of successful internal renames, having
made **zero** retry attempts on the one that failed. A per-call mitigation is complete on
the day it is written and incomplete the moment a second module writes a zarr.

So the retry lives on the store, which is the seam every write passes through — chunk
data, ``zarr.json``, an attrs rewrite, the metadata write behind a resize. Open a group
through :func:`open_group` and a module inherits the mitigation instead of re-deriving
it, whichever calls it goes on to make.

The one write a store cannot cover is a rename of the store *directory* — swapping a
staged copy into place, which is our own ``os.replace`` and belongs to
``core.atomic.replace``. ``tests/storage/test_zarr_seam.py`` scans for both.
"""

import asyncio
import os
from pathlib import Path

import zarr
import zarr.storage

from detail.core import atomic


def _check_read(root: Path, key: str, buf, byte_range) -> None:
    """Raise :class:`atomic.DamagedFileError` for a whole-file read of a file the OS lost.

    **Why on the store:** every read of a zarr passes through ``get`` / ``get_sync``, whichever
    array or group is asked for, so one check names the damaged file for every reader instead of
    each parser failing in its own words. Left alone, zarr hands a zero-filled ``zarr.json`` to
    ``json.loads`` as bytes, which sniffs leading NULs as UTF-32 and fails with ``'utf-32-be'
    codec can't decode byte 0x00`` and no path.

    An empty file is damaged anywhere in a store: zarr never writes one, and a chunk it did not
    write at all is a missing key, not an empty file. All-NUL is checked on ``zarr.json`` only,
    where it cannot be real. A zero-filled *chunk* is indistinguishable from data here (a
    stored array of zeros) and is left to the codec, and nothing is copied on the hot path:
    a chunk is checked by length alone.
    """
    if buf is None or byte_range is not None:
        return
    path = root / key
    if len(buf) == 0:
        raise atomic.DamagedFileError(atomic.damaged_file_message(path, "empty"))
    if key.endswith("zarr.json") and atomic.is_lost(buf.to_bytes()):
        raise atomic.DamagedFileError(
            atomic.damaged_file_message(path, f"all zero bytes ({len(buf)} of them)")
        )


class RetryingLocalStore(zarr.storage.LocalStore):
    """A ``LocalStore`` whose writes ride out the transient network-share replace, and whose
    reads name a file the OS lost.

    Both write entry points are covered: ``_set`` (behind ``set`` and
    ``set_if_not_exists``, which is where the codec pipeline writes every chunk) and
    ``set_sync``. Reads are not retried — a read never enters the rename window — but they are
    checked, by :func:`_check_read`, in both read entry points.
    """

    async def get(self, key, prototype=None, byte_range=None):
        buf = await super().get(key, prototype, byte_range)
        _check_read(self.root, key, buf, byte_range)
        return buf

    def get_sync(self, key, *, prototype=None, byte_range=None):
        buf = super().get_sync(key, prototype=prototype, byte_range=byte_range)
        _check_read(self.root, key, buf, byte_range)
        return buf

    async def _set(self, key: str, value, exclusive: bool = False) -> None:
        for attempt in range(atomic.RETRY_ATTEMPTS):
            try:
                return await super()._set(key, value, exclusive=exclusive)
            except PermissionError:
                if attempt == atomic.RETRY_ATTEMPTS - 1:
                    raise
                # `asyncio.sleep`, not `time.sleep`: this runs on zarr's event loop, and
                # blocking it would stall every other chunk write queued behind us.
                await asyncio.sleep(atomic.RETRY_DELAY_S)

    def set_sync(self, key: str, value) -> None:
        # Explicit two-argument `super()`: the zero-argument form does not resolve inside
        # a nested function.
        parent = super(RetryingLocalStore, self)
        atomic.retry_on_lock(lambda: parent.set_sync(key, value))


def check_intact(root: Path | str) -> None:
    """Raise :class:`atomic.DamagedFileError` if any zarr under ``root`` has lost its metadata.

    Reads each ``zarr.json`` under a ``*.zarr`` directory (a handful per part), never a chunk:
    cheap enough to run for every recording of a launch, on a network share too, which is what
    lets a damaged run be named *before* any worker starts. A ``root`` that does not exist is
    fine, since a run with nothing there yet is somebody else's error to report.
    """
    root = Path(root)
    is_store = root.name.endswith(".zarr")  # ``root`` may itself be one store
    for directory, _dirs, files in os.walk(root):
        inside = Path(directory).relative_to(root).parts
        if "zarr.json" not in files or not (
            is_store or any(p.endswith(".zarr") for p in inside)
        ):
            continue
        path = Path(directory, "zarr.json")
        data = atomic.retry_on_lock(path.read_bytes)
        if not data:
            raise atomic.DamagedFileError(atomic.damaged_file_message(path, "empty"))
        if atomic.is_lost(data):
            raise atomic.DamagedFileError(
                atomic.damaged_file_message(
                    path, f"all zero bytes ({len(data)} of them)"
                )
            )


def open_group(path: Path | str, *, mode: str):
    """Open the zarr group at ``path`` on a retrying store.

    ``mode`` is zarr's (``"r"`` / ``"a"`` / ``"w"``). The outer retry covers the open
    itself: ``mode="w"`` empties an existing store first, and that delete is exposed to
    the same sharing violation as the writes that follow it.
    """
    read_only = mode == "r"

    def _open():
        store = RetryingLocalStore(Path(path), read_only=read_only)
        return zarr.open_group(store=store, mode=mode)

    return atomic.retry_on_lock(_open)
