"""A crash-safe, append-only CSV of human decisions.

Human judgment is primary data (Tenet 9). It is not recomputable, so it is written with a
different discipline from everything else in ``storage/``: **appended** with ``flush()`` +
``fsync()`` rather than rewritten atomically. The difference is the point — a crash during a
whole-file rewrite can cost every decision in the file, while a crash during an append costs
at most the row being written, and even that row is tolerated on read.

Two things a naive append-only CSV gets wrong, both learned the hard way and both handled
here:

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
import os
from collections.abc import Sequence
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

        Parsing stops at the first row matching neither the current width nor the file's own
        header width. Only the final row can be torn — the file is append-only — so stopping
        there discards exactly the incomplete write and nothing else. A missing file reads as
        empty, which is what a run with no decisions yet is.
        """
        if not self.path.is_file():
            return [], []
        with open(self.path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
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
                    break
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
