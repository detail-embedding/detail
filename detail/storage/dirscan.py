"""Directory listing that costs one enumeration plus one probe per child.

Every ``discover_*`` in this package asks the same question — *which child directories of
``root`` hold ``<filename>``* — and nine of them spelled it the same way::

    p.name for p in root.iterdir() if p.is_dir() and (p / FILENAME).is_file()

``Path.iterdir`` yields bare paths, so ``is_dir()`` is a fresh ``stat`` per child on top of
the ``is_file()`` probe: **2N + 1** round trips where N + 1 will do. ``os.scandir`` answers
``is_dir`` out of the enumeration itself (Windows carries the attribute in the directory
entry; POSIX uses ``d_type``), and the probe alone already implies it — a plain file cannot
contain ``p / FILENAME``.

Locally the difference is invisible: a warm ``stat`` takes microseconds. On an SMB
share a ``stat`` takes milliseconds, so a listing of a large experiment takes over a
second, and Home built one per experiment on the UI thread until Windows painted the
window "not responding".

Halving the round trips does not make the listing free and is **not** a substitute for
keeping it off the UI thread — Home does both. What it buys is that every listing in the
app gets cheaper, including the ones that legitimately run there.
"""

import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

import yaml

from detail.core.ids import sorted_by_id

_log = logging.getLogger(__name__)

T = TypeVar("T")

#: What reading a record that exists but is broken raises (see :func:`read_each`).
UNREADABLE_RECORD_ERRORS = (FileNotFoundError, ValueError, yaml.YAMLError)


def _entries(root: Path | str) -> list[os.DirEntry]:
    """One enumeration of ``root``, or ``[]`` if it is absent or not a directory.

    Absent is not an error: every caller here is scanning a section directory that only
    exists once something has been written into it.
    """
    try:
        with os.scandir(os.fspath(root)) as scan:
            return list(scan)
    except (FileNotFoundError, NotADirectoryError):
        return []


def child_dirs(root: Path | str) -> list[Path]:
    """Subdirectories of ``root``, lexicographic, one level deep.

    For callers that want the directories themselves rather than the ones holding a
    record — an id allocator counting *claimed* directories, say.
    """
    return sorted(Path(e.path) for e in _entries(root) if e.is_dir())


def dirs_containing(root: Path | str, filename: str) -> list[Path]:
    """Child directories of ``root`` that hold ``filename``, lexicographic.

    The ``is_dir`` half is free (it comes out of the enumeration) and the probe is the
    only per-child syscall, so this is N + 1 round trips for N children.
    """
    return sorted(
        Path(e.path)
        for e in _entries(root)
        if os.path.isfile(os.path.join(e.path, filename))
    )


def discover_ids(root: Path | str, filename: str) -> list[str]:
    """Ids of ``root``'s child directories that hold ``filename``, in **id** order.

    The ``discover_*`` idiom: ``sorted_by_id`` rather than lexicographic, because these are
    generated ``<stem>_<n>`` ids and text order breaks at the zero-pad boundary
    (``detail.core.ids``).
    """
    return sorted_by_id(p.name for p in dirs_containing(root, filename))


def read_each(ids: Iterable[T], read: Callable[[T], object]) -> list[dict]:
    """``read(id)`` for each id, leaving out any record that cannot be parsed.

    A listing is a *view*: one record it cannot parse costs that row, never the rest.
    ``api/active.py::_safe`` makes the same trade one level up and says why — a view that
    vanishes because one of its sources is sick tells the reader less than a view with a
    row missing. Before this seam each listing read its records bare, so a single
    zero-filled ``distill.yaml`` (left by a power loss mid-write) made the Models and
    Datasets pages impossible to open, with no way back to the record from inside the app.

    The catch holds the read and nothing else, and names only what a record that *is there*
    but is broken raises: a torn or zero-filled file is a ``yaml.YAMLError`` or a
    ``ValueError`` (``JSONDecodeError`` and ``UnicodeDecodeError`` are both), and a record
    deleted between the scan and the read is ``FileNotFoundError``. An empty file parses to
    ``None`` rather than raising, so a record that is not a mapping is left out too. A
    permission error still raises. Each skip is logged with the id, so the missing row can
    be traced to its file.

    A lookup of one record by id is not a listing and keeps raising: the caller asked for
    that record, and an absent answer would be a wrong one.
    """
    records = []
    for record_id in ids:
        try:
            record = read(record_id)
        except UNREADABLE_RECORD_ERRORS as exc:
            _log.warning("skipping unreadable record %s: %s", record_id, exc)
            continue
        if not isinstance(record, dict):
            _log.warning("skipping record %s: not a mapping", record_id)
            continue
        records.append(record)
    return records
