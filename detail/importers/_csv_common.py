"""Shared parsing helpers for the generic CSV stream importers (Section 3).

Both ``timeseries_csv`` and ``events_csv`` recognize a single **named time column**
(``seconds`` | ``frame`` | ``t_ns``, case-insensitively) that fixes the stream's
clock, and share the strict, actionable validation the two importers wrap. Input
files are read BOM-tolerantly (``utf-8-sig``); the time column decides how native
time reaches canonical video ``t_ns`` (Section 3 "Clocks").

:func:`read_user_csv` is **the one opener for a CSV a user hands in** — a stream, a spatial
setup, a clip list — wherever the reader lives. It keeps ``utf-8-sig`` (never a silent
locale fallback) and turns a file that is not UTF-8 text into a
:class:`~detail.core.errors.UserInputError` that says to save it as "CSV UTF-8". Each row
remembers its file line and how many cells it had, so :func:`int_cell` / :func:`float_cell`
can say "row 7: expected 12 cells, found 6" instead of failing on a missing value.
``tests/storage/test_user_file_readers.py`` lists every other ``csv`` reader with why.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path

from detail.core import utils
from detail.core.errors import UserInputError

# The recognized time-column names (case-insensitive) and the clock each implies.
# ``seconds``/``t_ns`` are external-device time (aligned by a read-time offset);
# ``frame`` is the video frame index (resolved against the recording's map).
TIME_CLOCKS: dict[str, str] = {
    "seconds": "stream_seconds",
    "frame": "video_frames",
    "t_ns": "stream_seconds",
}
TIME_COLUMN_NAMES: tuple[str, ...] = tuple(TIME_CLOCKS)

LABEL_COLUMN = "label"
CODE_COLUMN = "code"

_NEEDS_TIME_COLUMN = "CSV needs a time column named one of: seconds, frame, t_ns"


class CsvRow(dict):
    """One data row keyed by header, as ``csv.DictReader`` builds it — a cell the row is
    short of is ``None``, extra cells sit under the ``None`` key — plus where it came from.
    """

    def __init__(
        self, values: dict, *, source: str, line: int, width: int, expected: int
    ) -> None:
        super().__init__(values)
        #: The file the row was read from, as messages name it.
        self.source = source
        #: The 1-based file line the row ends on (its only line, unless a quoted cell
        #: spans lines).
        self.line = line
        #: How many cells the row actually had, and how many the header names.
        self.width = width
        self.expected = expected


#: Separators a spreadsheet uses instead of a comma, as a message names them.
_OTHER_SEPARATORS = {";": "semicolon", "\t": "tab"}


def read_user_csv(path: Path | str) -> tuple[list[str], list[CsvRow]]:
    """Read a CSV a user handed in into ``(fieldnames, rows)``, BOM-tolerantly.

    Blank lines are skipped, as ``csv.DictReader`` skips them, and so is a row whose every cell
    is empty (``,,,,``, which a spreadsheet leaves where a row was cleared). A header cell left
    empty at the end of the header, over a column empty in every row, is dropped: a spreadsheet
    writes one for a stray formatted cell, and refusing the import for it named nothing the
    user could see. An empty file gives ``([], [])``; what that means is the caller's to say.

    A file that is one column whose header holds a semicolon or a tab is a spreadsheet saved
    with another separator (European Excel uses ``;``). It is refused, saying so, rather than
    read as one column, which every reader then reported as a missing column; never a silent
    switch of separator.

    Raises:
        FileNotFoundError: No file at ``path``.
        UserInputError: The file is not UTF-8 text (or not text at all), or it is separated
            by semicolons or tabs.
    """
    source = str(path)
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            records = [
                (reader.line_num, cells)
                for cells in reader
                if any(cell.strip() for cell in cells)
            ]
    except UnicodeDecodeError:
        raise UserInputError(
            f"{source} is not UTF-8 text. If it is a spreadsheet, save it from Excel with "
            "Save As → CSV UTF-8 (Comma delimited)"
        ) from None
    except csv.Error as exc:
        raise UserInputError(f"{source} is not a CSV text file ({exc})") from None
    if len(header) == 1:
        for separator, name in _OTHER_SEPARATORS.items():
            if separator in header[0]:
                raise UserInputError(
                    f"{source} looks {name}-separated (its header is the one column "
                    f"{header[0]!r}); save it with commas — in Excel, Save As → CSV UTF-8 "
                    "(Comma delimited)"
                )
    while (
        header
        and not header[-1].strip()
        and all(not "".join(cells[len(header) - 1 :]).strip() for _, cells in records)
    ):
        header = header[:-1]
        records = [(line, cells[: len(header)]) for line, cells in records]
    rows = []
    for line, cells in records:
        values: dict = dict(zip(header, cells))
        for name in header[len(cells) :]:
            values.setdefault(name, None)
        if len(cells) > len(header):
            values[None] = cells[len(header) :]
        rows.append(
            CsvRow(
                values,
                source=source,
                line=line,
                width=len(cells),
                expected=len(header),
            )
        )
    return list(header), rows


def _require_cell(row: CsvRow, column: str) -> str:
    value = row.get(column)
    if value is None:
        raise UserInputError(
            f"{row.source} row {row.line}: expected {row.expected} cells, found "
            f"{row.width} (no {column!r})"
        )
    return value


def int_cell(row: CsvRow, column: str) -> int:
    """``row[column]`` as an integer, naming the file, row and column on a mistake.

    Raises:
        UserInputError: The row is short of that cell, or it is not an integer.
    """
    value = _require_cell(row, column)
    try:
        return parse_int(value, column, row.line)
    except UserInputError as exc:
        raise UserInputError(f"{row.source} {exc}") from None


def float_cell(row: CsvRow, column: str) -> float:
    """``row[column]`` as a number, naming the file, row and column on a mistake.

    Raises:
        UserInputError: The row is short of that cell, or it is not a number.
    """
    value = _require_cell(row, column)
    try:
        return parse_float(value, column, row.line)
    except UserInputError as exc:
        raise UserInputError(f"{row.source} {exc}") from None


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a stream CSV into ``(fieldnames, rows)`` through :func:`read_user_csv`.

    Raises:
        UserInputError: If the file is not UTF-8 text, or is empty (no header row).
    """
    fieldnames, rows = read_user_csv(path)
    if not fieldnames:
        raise UserInputError("CSV is empty — it has no header row")
    return fieldnames, rows


def sniff_header_and_first(path: Path) -> tuple[list[str], list[str] | None]:
    """Read the header row and the first data row (as lists) for ``can_import``.

    Returns ``([], None)`` on an unreadable/empty file. The first data row is
    ``None`` when the file is header-only.
    """
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            first = next(reader, None)
    except OSError:
        return [], None
    return header, first


def time_column_matches(fieldnames: list[str]) -> list[tuple[str, str]]:
    """Every ``(source_column, time_key)`` in ``fieldnames`` naming a time column."""
    matches: list[tuple[str, str]] = []
    for name in fieldnames:
        key = name.strip().lower()
        if key in TIME_CLOCKS:
            matches.append((name, key))
    return matches


def resolve_time_column(fieldnames: list[str]) -> tuple[str, str]:
    """The single recognized time column as ``(source_column, time_key)``.

    Raises:
        ValueError: If the header has no recognized time column, or more than one.
    """
    matches = time_column_matches(fieldnames)
    if not matches:
        raise ValueError(_NEEDS_TIME_COLUMN)
    if len(matches) > 1:
        names = [source for source, _ in matches]
        raise ValueError(
            f"CSV has more than one time column {names}; include exactly one of "
            "seconds, frame, t_ns"
        )
    return matches[0]


def has_column(fieldnames: list[str], name: str) -> bool:
    """Whether ``fieldnames`` contains ``name`` (case-insensitive)."""
    return any(field.strip().lower() == name for field in fieldnames)


def find_column(fieldnames: list[str], name: str) -> str | None:
    """The source column whose name matches ``name`` (case-insensitive), or None."""
    for field in fieldnames:
        if field.strip().lower() == name:
            return field
    return None


def reject_duplicate_columns(fieldnames: list[str]) -> None:
    """Raise if the header names any column twice.

    ``csv.DictReader`` maps a row into a dict keyed by header name, so two columns sharing a
    name **silently collapse to whichever came last** and the first one's data is simply
    gone. Nothing downstream can detect that: the parse succeeds, the row count is right,
    every value is a plausible number. Duplicate headers are ordinary in the wild (a
    spreadsheet export, and every DeepLabCut CSV, whose top row repeats the scorer name once
    per body part), so this is a refusal rather than a best guess.

    Raises:
        ValueError: If any header name appears more than once.
    """
    seen: set[str] = set()
    repeated: list[str] = []
    for field in fieldnames:
        if field in seen and field not in repeated:
            repeated.append(field)
        seen.add(field)
    if repeated:
        raise ValueError(
            f"CSV header names {repeated} more than once; column names must be unique "
            "(a multi-row header, e.g. a DeepLabCut export, needs its header flattened "
            "to one row first)"
        )


def parse_utc_epoch_seconds(value: str | None, column: str, row: int) -> float:
    """Parse a UTC cell (ISO 8601 or Unix epoch seconds) to epoch seconds.

    A naive ISO 8601 timestamp (no offset, no ``Z``) is read as already UTC: the column was
    declared UTC, so there is nothing else it could mean.

    Raises:
        ValueError: If ``value`` is empty or matches neither format.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"row {row}, column {column!r}: empty UTC timestamp")
    try:
        epoch = float(text)
    except ValueError:
        epoch = None
    if epoch is not None:
        return _checked_epoch(epoch, text, column, row)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"row {row}, column {column!r}: {value!r} is not a UTC timestamp "
            "(expected ISO 8601, e.g. 2026-09-04T14:00:38Z, or Unix epoch seconds)"
        ) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _checked_epoch(epoch: float, text: str, column: str, row: int) -> float:
    """``epoch`` (Unix seconds), refused when it is no instant a clock could hold.

    Epoch *milliseconds* in a UTC column, the common logger mistake, used to fail on the
    datetime conversion as "year 58629 is out of range", which names nothing the user typed.

    Raises:
        UserInputError: Not finite, or beyond what a date can hold.
    """
    where = f"row {row}, column {column!r}"
    if not math.isfinite(epoch):
        raise UserInputError(f"{where}: {text!r} is not a time")
    try:
        datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        raise UserInputError(
            f"{where}: {text} is not Unix time in seconds: it falls outside the years 1 to "
            "9999. If it counts milliseconds, divide the column by 1000"
        ) from None
    return epoch


def require_data_rows(rows: list[dict[str, str]]) -> None:
    """Raise if there are no data rows (a header-only file is not a stream).

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("CSV has a header but no data rows")


def parse_int(value: str | None, column: str, row: int) -> int:
    """Parse an integer cell, naming the column and 1-based file row on failure.

    Raises:
        UserInputError: If ``value`` is empty or not an integer.
    """
    text = "" if value is None else str(value).strip()
    try:
        return int(text)
    except (TypeError, ValueError):
        raise UserInputError(
            f"row {row}, column {column!r}: {value!r} is not an integer"
        ) from None


def parse_float(value: str | None, column: str, row: int) -> float:
    """Parse a numeric cell, naming the column and 1-based file row on failure.

    Raises:
        UserInputError: If ``value`` is empty or not a number.
    """
    text = "" if value is None else str(value).strip()
    try:
        return float(text)
    except (TypeError, ValueError):
        raise UserInputError(
            f"row {row}, column {column!r}: {value!r} is not a number"
        ) from None


def cell_to_t_ns(value: str | None, time_key: str, column: str, row: int) -> int:
    """A ``seconds`` or ``t_ns`` cell as int64 ``t_ns`` (Tenet 8), refused if it cannot be one.

    Both kinds go through ``core.utils``' checked conversion, so a value past int64 (epoch
    milliseconds in a ``seconds`` column) is refused here, naming the row, instead of stored.

    Raises:
        UserInputError: If the cell is not a number, or is too large to be a time.
    """
    what = f"row {row}, column {column!r}"
    if time_key == "t_ns":
        return utils.check_t_ns(parse_int(value, column, row), what=what)
    return utils.seconds_to_t_ns(parse_float(value, column, row), what=what)
