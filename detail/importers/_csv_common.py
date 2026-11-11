"""Shared parsing helpers for the generic CSV stream importers (Section 3).

Both ``timeseries_csv`` and ``events_csv`` recognize a single **named time column**
(``seconds`` | ``frame`` | ``t_ns``, case-insensitively) that fixes the stream's
clock, and share the strict, actionable validation the two importers wrap. Input
files are read BOM-tolerantly (``utf-8-sig``); the time column decides how native
time reaches canonical video ``t_ns`` (Section 3 "Clocks").
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

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


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV into ``(fieldnames, rows)`` BOM-tolerantly.

    Raises:
        ValueError: If the file is empty (no header row).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("CSV is empty — it has no header row")
        rows = list(reader)
    return list(fieldnames), rows


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
        return float(text)
    except ValueError:
        pass
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
        ValueError: If ``value`` is empty or not an integer.
    """
    text = "" if value is None else str(value).strip()
    try:
        return int(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"row {row}, column {column!r}: {value!r} is not an integer"
        ) from None


def parse_float(value: str | None, column: str, row: int) -> float:
    """Parse a numeric cell, naming the column and 1-based file row on failure.

    Raises:
        ValueError: If ``value`` is empty or not a number.
    """
    text = "" if value is None else str(value).strip()
    try:
        return float(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"row {row}, column {column!r}: {value!r} is not a number"
        ) from None
