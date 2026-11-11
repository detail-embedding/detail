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
from pathlib import Path

import zarr
import zarr.storage

from detail.core import atomic


class RetryingLocalStore(zarr.storage.LocalStore):
    """A ``LocalStore`` whose writes ride out the transient network-share replace.

    Both write entry points are covered: ``_set`` (behind ``set`` and
    ``set_if_not_exists``, which is where the codec pipeline writes every chunk) and
    ``set_sync``. Reads are left alone — a read never enters the rename window.
    """

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
