"""A crash-safe, append-only CSV of human decisions.

Human judgment is primary data (Tenet 9). It is not recomputable, so it is written with a
different discipline from everything else in ``storage/``: **appended** with ``flush()`` +
``fsync()`` rather than rewritten atomically. The difference is the point — a crash during a
whole-file rewrite can cost every decision in the file, while a crash during an append costs
at most the row being written, and even that row is tolerated on read.

Three things a naive append-only CSV gets wrong, all learned the hard way and all handled
here:

- **A torn row is the file's tail until something is appended after it.** A crash mid-row
  leaves no newline, so the next ``open(path, "a")`` glues the next row onto the half row, and
  a reader that stops at the first bad row then hides every later decision — for good, once a
  ``migrate`` rewrites the file from that read. So an append first cuts the torn tail back to
  the end of the last complete row (:func:`complete_length`), and a read skips a bad row in the
  middle (a file torn before this rule existed) instead of stopping, and ignores an
  unterminated last row however many fields it parses to: a row cut inside an open quote
  parses as a whole row.

- **The header is written once, at creation**, so a log started under an older column set
  keeps its narrow header forever while later appends carry the full width. ``csv.DictReader``
  maps by *position* against one header, so on a mixed-width file every field past the missing
  column shifts by one. Reading each row against the columns it was actually written under
  undoes that; ``migrate`` normalizes the file so it stops happening. A real annotation run on
  disk had 12 decisions unreadable this way.
- **Migration must not turn every append into a rewrite.** It is a no-op once the header is
  current, or the append-only guarantee is gone and each new decision risks the whole file.

Lifted from ``storage/annotation_runs.py``, which is where all of the above was first paid
for. A module that hand-rolls this is the bug (§21).
"""

import csv
import io
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

from detail.core import atomic


class AppendLog:
    """An append-only CSV at ``path`` whose current schema is ``columns``.

    Binds the two together because every operation needs both: the width identifies a row's
    shape, and the column list is what a migrated file is normalized to.
    """

    def __init__(self, path: Path | str, columns: Sequence[str]):
        self.path = Path(path)
        self.columns = list(columns)

    def append(self, values: Sequence) -> None:
        """Append one row, creating the file with a header if it is new.

        Durable per row (``flush`` + ``fsync``), so a crash loses at most this row and never
        an earlier one. A file whose header predates a column is normalized first.

        "New" is decided on **content, not existence**. ``open(path, "a")`` creates the file
        before the header is written, so a crash in that window leaves a 0-byte file that
        exists — and keying on existence then skipped the header, wrote a bare data row, and
        left ``read`` to consume that row *as* the header. The next append called ``migrate``,
        which rewrote the file under the real columns and dropped the row-turned-header for
        good. That is a human decision lost to the exact crash this class advertises surviving.
        """
        self.append_many([values])

    def append_many(self, rows: Sequence[Sequence]) -> None:
        """Append ``rows`` in one open and one ``fsync``: a batch that belongs together.

        A crash can still tear only the last row on disk, as with :meth:`append`, because the
        rows are written in order; what it may lose is the tail of this batch, never an earlier
        row. Nothing is written for an empty ``rows``.
        """
        rows = [list(r) for r in rows]
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        drop_torn_tail(self.path)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        if not is_new:
            self.migrate()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(self.columns)
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())

    def read(self) -> tuple[list[str], list[dict]]:
        """``(header, rows)``, each row resolved against the columns it was written under.

        A row matching neither the current width nor the file's own header width is skipped,
        and reading **goes on**: such a row is a torn write that something was later appended
        after (only a file torn before :meth:`append_many` cut torn tails can hold one), and
        stopping there would hide every decision written after it. An unterminated last row is
        the write a crash interrupted, so it is ignored even when its fields happen to parse to
        a full row. A missing file reads as empty, which is what a run with no decisions yet is.
        """
        if not self.path.is_file():
            return [], []
        data = self.path.read_bytes()
        data = data[: complete_length(data)]
        reader = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""))
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        rows = []
        for raw in reader:
            if not raw:
                continue
            columns = self._row_columns(len(raw), header)
            if columns is None:
                continue
            rows.append(dict(zip(columns, raw)))
        return header, rows

    def rows(self) -> list[dict]:
        """Just the rows — the common case."""
        return self.read()[1]

    def migrate(self) -> None:
        """Rewrite the file under the current ``columns`` if its header predates them.

        **A no-op when the header already matches**, which is what keeps an append an append:
        migrating on every write would rewrite the whole file each time and put every stored
        decision at risk of the next crash.

        Safe against the append-only guarantee because the rewrite is atomic (tmp +
        ``os.replace``): a reader sees the whole old file or the whole new one.
        """
        header, rows = self.read()
        if not header or header == self.columns:
            return
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
            for row in rows:
                writer.writerow([row.get(name, "") or "" for name in self.columns])
            f.flush()
            os.fsync(f.fileno())
        atomic.replace(tmp, self.path)

    def _row_columns(self, n_fields: int, header: Sequence[str]) -> list[str] | None:
        """Which column set a data row of ``n_fields`` was written under, or ``None``.

        The width identifies the shape, which makes rows written under an older header
        recoverable rather than merely tolerated. ``None`` means the row matches neither and
        is treated as torn.
        """
        if n_fields == len(self.columns):
            return list(self.columns)
        if n_fields == len(header):
            return list(header)
        return None


def complete_length(data: bytes) -> int:
    """How many leading bytes of the CSV ``data`` are whole, newline-terminated rows.

    Every row the writer finishes ends in a newline, so data ending in one is all whole. Else
    the tail from the start of the last record is a torn write. Where that record starts is
    found by parsing, not by the last newline: a quoted field may hold newlines of its own, so a
    row cut inside one has newlines in its torn part.
    """
    if not data or data.endswith(b"\n"):
        return len(data)
    ends = [0]

    def lines() -> Iterator[str]:
        pos = 0
        for line in data.splitlines(keepends=True):
            pos += len(line)
            ends.append(pos)
            yield line.decode("utf-8", errors="replace")

    reader = csv.reader(lines())
    record_ends = [0]
    try:
        for _ in reader:
            record_ends.append(ends[-1])
    except csv.Error:
        pass
    # The last record parsed is the unterminated one: everything before it is whole.
    whole = [end for end in record_ends if end < len(data)]
    return whole[-1]


def drop_torn_tail(path: Path) -> None:
    """Cut ``path`` back to its last complete row, so the next append starts a fresh line.

    A truncate, not a rewrite: the bytes before the torn row are never touched, so this keeps
    the append-only guarantee. The torn row was already lost — it was never readable.
    """
    if not path.is_file():
        return
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        data = f.read()
    keep = complete_length(data)
    with open(path, "r+b") as f:
        f.truncate(keep)
        f.flush()
        os.fsync(f.fileno())


def append_line(path: Path | str, line: str) -> None:
    """Append one newline-terminated ``line`` to a line-per-record text file (JSONL).

    The same torn-tail rule as :class:`AppendLog`: a line a crash cut short is dropped first,
    so the new line is never glued onto it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size:
        with open(path, "rb") as f:
            data = f.read()
        if not data.endswith(b"\n"):
            with open(path, "r+b") as f:
                f.truncate(data.rfind(b"\n") + 1)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()
        os.fsync(f.fileno())
