"""Never-reused ids: the one allocator behind every ``<stem>_<n>`` id the tool mints.

An id minted as "one past the highest that still exists" is handed to the next thing as soon as
the newest one is deleted, while finished runs, distill records, comparisons and export folders
still name it — and nothing reads wrong at the line that mints it. Recordings solved this first
(``experiment.next_recording_id``: a counter in ``config.yaml`` that only goes up); this module is
the same idea for every other family (DESIGN_PLAN §3 *Ids are never reused*).

**The mark.** Each family keeps a high-water mark in ``.ids.yaml`` beside it (``models/.ids.yaml``,
a recording's own ``.ids.yaml`` for its runs and regions), keyed by stem so families sharing a
home count separately. The next id is the larger of the mark and one past the highest id on disk,
so a family made before the mark existed continues from its highest id and **no stored id
changes**; ids reused before the mark existed cannot be detected.

**The claim.** Under an exclusive lock on ``.ids.lock`` (a sibling, never the mark itself, which is
replaced atomically), so two processes cannot take one id. :func:`claim_dir` also creates the id's
directory with ``exist_ok=False``; :func:`claim` is for a family whose members are entries in a
record the caller holds its own lock on (a recording's runs, its regions).

A mark that cannot be read is treated as absent: the ids on disk are still never reused, only a
freed id above them could be.
"""

import logging
import os
import re
import uuid
from collections.abc import Iterable
from pathlib import Path

import yaml

from detail.core import atomic
from detail.storage import dirscan

_log = logging.getLogger(__name__)

MARK_FILENAME = ".ids.yaml"
LOCK_FILENAME = ".ids.lock"


def _format(stem: str, number: int, digits: int) -> str:
    return f"{stem}_{number:0{digits}d}"


def _numbers(stem: str, names: Iterable[str]) -> list[int]:
    pattern = re.compile(rf"{re.escape(stem)}_(\d+)")
    return [int(m.group(1)) for name in names if (m := pattern.fullmatch(str(name)))]


def _read_marks(home: Path) -> dict:
    path = home / MARK_FILENAME
    try:
        data = yaml.safe_load(atomic.read_text(path))
    except FileNotFoundError:
        return {}
    except (ValueError, yaml.YAMLError) as exc:
        _log.warning("ignoring unreadable id mark %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _write_marks(home: Path, marks: dict) -> None:
    path = home / MARK_FILENAME
    tmp = home / f"{MARK_FILENAME}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            yaml.dump(marks, handle, sort_keys=True)
        atomic.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _first(marks: dict, stem: str, names: Iterable[str], start: int) -> int:
    try:
        marked = int(marks.get(stem, start))
    except (TypeError, ValueError):
        marked = start
    return max([marked, start, *(n + 1 for n in _numbers(stem, names))])


def claim(
    home: Path | str,
    stem: str,
    *,
    digits: int,
    taken: Iterable[str],
    start: int = 1,
) -> str:
    """The next ``<stem>_NNN`` id, spent: never returned again from ``home``.

    ``taken`` is every id of the family that exists now (the entries of the record that lists
    them). The caller writes the member; a member never written leaves a gap, never a reuse.
    """
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    with atomic.file_lock(home / LOCK_FILENAME):
        marks = _read_marks(home)
        number = _first(marks, stem, taken, start)
        marks[stem] = number + 1
        _write_marks(home, marks)
    return _format(stem, number, digits)


def claim_dir(
    root: Path | str,
    stem: str,
    *,
    digits: int,
    home: Path | str | None = None,
    taken: Iterable[str] = (),
    start: int = 1,
) -> str:
    """Claim the next ``<stem>_NNN`` id by creating ``root/<id>/``, and return it.

    ``home`` holds the mark (default ``root``); a family spread over several roots (the tool runs,
    one folder per tool, ids unique across all of them) names one home and passes the other roots'
    ids as ``taken``. Every child directory of ``root`` counts as taken, record or not: a
    directory claimed a moment ago has no record yet.
    """
    root = Path(root)
    home = Path(home) if home is not None else root
    root.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    with atomic.file_lock(home / LOCK_FILENAME):
        marks = _read_marks(home)
        names = [p.name for p in dirscan.child_dirs(root)] + list(taken)
        number = _first(marks, stem, names, start)
        while True:
            candidate = _format(stem, number, digits)
            try:
                (root / candidate).mkdir(exist_ok=False)
            except FileExistsError:
                number += 1
                continue
            break
        marks[stem] = number + 1
        _write_marks(home, marks)
    return candidate


def peek(
    root: Path | str,
    stem: str,
    *,
    digits: int,
    taken: Iterable[str] | None = None,
    start: int = 1,
) -> str:
    """The id :func:`claim_dir` would hand out next, without spending it — a prediction only:
    another process may claim it first."""
    root = Path(root)
    names = [p.name for p in dirscan.child_dirs(root)] if taken is None else list(taken)
    return _format(stem, _first(_read_marks(root), stem, names, start), digits)
