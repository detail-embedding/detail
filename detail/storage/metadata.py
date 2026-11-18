"""Grouping-variable storage (Section 11).

Recording-level metadata for cohort comparison (animal id, treatment, condition, …)
lives in a single experiment-root ``metadata.csv`` — one row per recording, ``recording_id``
the key (always the first column), all other columns user-defined and addable at any
time. The file is the **single source of truth** (no generated/user split, no
``schema_version`` — it is plain user-owned CSV). Rows preserve insertion order;
columns preserve header order with new ones appended; missing cells are empty strings.
"""

import csv
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path

METADATA_FILENAME = "metadata.csv"
RECORDING_KEY = "recording_id"
# Accepted as an import *key* and written into an export for legibility, but never stored
# as a column: a recording's name lives in its own recording.yaml (§11).
NAME_KEY = "name"
# The same for a recording's video file name — derived from its video parts, never stored.
# It is the key a sheet prepared *before* import can carry (#330): the id is minted by the
# tool and the name is only right once someone has renamed every recording.
FILE_KEY = "file"
#: Headers an import recognises as a video-file-name key, compared ignoring case, spaces
#: and hyphens. ``FILE_KEY`` first: it is what an export writes and what the table shows.
VIDEO_KEY_HEADERS = (
    FILE_KEY,
    "file_name",
    "filename",
    "video",
    "video_file",
    "video_name",
)


class MissingKeyColumnError(ValueError):
    """A CSV with no column recognised as a key. Carries its ``header`` so a caller can
    ask which column holds the video file name instead of giving up."""

    def __init__(self, header: list[str]):
        self.header = header
        super().__init__(
            f"CSV has no column to match rows to recordings — expected "
            f"{RECORDING_KEY!r}, {NAME_KEY!r}, or a video file name column "
            f"({', '.join(VIDEO_KEY_HEADERS)}). Its columns are: {', '.join(header)}. "
            "Name the one holding each recording's video file name as the key column."
        )


def _normalized_header(header: str) -> str:
    return header.strip().lower().replace(" ", "_").replace("-", "_")


def is_recording_key_header(cell: str) -> bool:
    """Whether a header cell names the recording key — ``recording``, ``Recording ID``,
    ``name`` … — in any case, spacing or hyphenation, the way every header is matched here.

    A header read as data becomes a recording called "Recording ID" that matches nothing.
    """
    return _normalized_header(cell) in ("recording", RECORDING_KEY, NAME_KEY)


def is_video_key_header(cell: str) -> bool:
    """Whether a header cell names a video-file-name key (:data:`VIDEO_KEY_HEADERS`)."""
    return _normalized_header(cell) in VIDEO_KEY_HEADERS


def key_kind(column: str) -> str:
    """What a key column's values are matched against: ``RECORDING_KEY``, ``NAME_KEY``, or
    ``FILE_KEY``. Any column that is not the id or the name is a video file name — the one
    key a person can name a column of their own for (#330)."""
    normalized = _normalized_header(column)
    if normalized in (RECORDING_KEY, NAME_KEY):
        return normalized
    return FILE_KEY


def detect_key_column(header: list[str]) -> str | None:
    """The header cell to key an import on, or ``None`` when none is recognised.

    ``recording_id`` wins (it is the identity), then ``name``, then the first
    :data:`VIDEO_KEY_HEADERS` match in file order. An export carries all three, and keying
    it back on the id is what makes it round-trip exactly.
    """
    by_normalized = {}
    for cell in header:
        by_normalized.setdefault(_normalized_header(cell), cell)
    for key in (RECORDING_KEY, NAME_KEY):
        if key in by_normalized:
            return by_normalized[key]
    return next((cell for cell in header if is_video_key_header(cell)), None)


def metadata_path(experiment_dir: Path | str) -> Path:
    return Path(experiment_dir) / METADATA_FILENAME


def _read_raw(
    experiment_dir: Path | str,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """(columns excluding the key, ordered ``{recording_id: {col: value}}``).

    Returns empties when the file is absent.
    """
    path = metadata_path(experiment_dir)
    if not path.is_file():
        return [], {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = [c for c in (reader.fieldnames or []) if c != RECORDING_KEY]
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            recording_id = (row.get(RECORDING_KEY) or "").strip()
            if recording_id:
                rows[recording_id] = {c: (row.get(c) or "") for c in cols}
    return cols, rows


def _write_csv(path: Path, cols: list[str], rows: dict[str, dict[str, str]]) -> None:
    """Write ``recording_id`` first, then ``cols``; one row per recording, blanks empty."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([RECORDING_KEY, *cols])
        for recording_id, values in rows.items():
            writer.writerow([recording_id, *(values.get(c, "") for c in cols)])


def _write_raw(
    experiment_dir: Path | str, cols: list[str], rows: dict[str, dict[str, str]]
) -> None:
    _write_csv(metadata_path(experiment_dir), cols, rows)


def export_metadata(
    experiment_dir: Path | str,
    out_path: Path | str,
    names: dict[str, str] | None = None,
    files: dict[str, str] | None = None,
) -> None:
    """Write the experiment's metadata to an arbitrary CSV path — a fill-in **template**
    (``recording_id``, then ``name`` and ``file`` when given, then every column, one row
    per recording). An undergrad fills the blanks offline, then ``import_metadata_csv``
    merges it back, keyed on any of the three.

    ``names`` and ``files`` are passed in rather than read here: a recording's name and
    video live in its own ``recording.yaml``, and this module owns ``metadata.csv`` only.
    """
    cols, rows = _read_raw(experiment_dir)
    derived = [
        (key, column)
        for key, column in ((NAME_KEY, names), (FILE_KEY, files))
        if column is not None
    ]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([RECORDING_KEY, *(key for key, _ in derived), *cols])
        for recording_id, values in rows.items():
            writer.writerow(
                [
                    recording_id,
                    *(column.get(recording_id, "") for _, column in derived),
                    *(values.get(c, "") for c in cols),
                ]
            )


def read_metadata(experiment_dir: Path | str) -> dict[str, dict[str, str]]:
    """All rows keyed by recording id: ``{recording_id: {column: value}}`` (empty if absent)."""
    return _read_raw(experiment_dir)[1]


#: What a metadata column holds, as ``infer_column_types`` reads it.
COLUMN_TYPES = ("number", "date", "time", "category")
#: ``HH:MM`` or ``HH:MM:SS`` (fractions of a second allowed): a clock time, no day.
_CLOCK = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?")


def parse_number(value: str) -> float | None:
    """``value`` as a finite number, or ``None`` (``"nan"`` and ``"inf"`` are not numbers here)."""
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_date(value: str) -> datetime | None:
    """``value`` as an ISO date or date-time (``2026-07-02``, ``2026-07-02T14:00``), or ``None``.

    A bare time (``14:00:21``) is not a date: it names no day. A value with a UTC offset
    (``2026-07-02T10:00+02:00``) is converted to UTC and returned **naive**, like every other,
    so a column mixing the two forms still compares: an aware and a naive datetime raise
    ``TypeError`` against each other.
    """
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
    if parsed is not None:
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    try:
        day = date.fromisoformat(text)
    except ValueError:
        return None
    return datetime(day.year, day.month, day.day)


def parse_time(value: str) -> float | None:
    """``value`` as a time of day (``14:00``, ``14:00:21``, ``14:00:21.5``) in seconds after
    midnight, or ``None``. A value with a day in it is a date, not a time."""
    match = _CLOCK.fullmatch(str(value).strip())
    if match is None:
        return None
    hours, minutes = int(match.group(1)), int(match.group(2))
    seconds = float(match.group(3) or 0.0)
    if hours > 23 or minutes > 59 or seconds >= 60:
        return None
    return hours * 3600.0 + minutes * 60.0 + seconds


def infer_column_types(metadata: dict[str, dict[str, str]]) -> dict[str, str]:
    """``{column: "number" | "date" | "time" | "category"}`` for every column ``metadata`` holds.

    A column is a **number** if every non-empty value parses as one, else a **date** if every
    non-empty value parses as an ISO date, else a **time** if every non-empty value is a clock
    time (``14:00:21``, a recording's start time of day; ``parse_time``), else a **category**.
    A time sorts and ranges like a number. Blank cells are ignored, so a
    number column with a missing value stays a number; a column with no values at all is a
    category, since there is nothing to range over. Number wins over date, so a column of years
    (``2026``) is a number. ``metadata`` is ``read_metadata``'s shape.
    """
    values: dict[str, list[str]] = {}
    for row in metadata.values():
        for column, value in row.items():
            if column == RECORDING_KEY:
                continue
            bucket = values.setdefault(column, [])
            text = "" if value is None else str(value).strip()
            if text:
                bucket.append(text)
    out: dict[str, str] = {}
    for column, filled in values.items():
        if filled and all(parse_number(v) is not None for v in filled):
            out[column] = "number"
        elif filled and all(parse_date(v) is not None for v in filled):
            out[column] = "date"
        elif filled and all(parse_time(v) is not None for v in filled):
            out[column] = "time"
        else:
            out[column] = "category"
    return out


def columns(experiment_dir: Path | str) -> list[str]:
    """User-defined columns (excluding ``recording_id``), in file order."""
    return _read_raw(experiment_dir)[0]


def upsert_row(
    experiment_dir: Path | str, recording_id: str, values: dict[str, str] | None = None
) -> None:
    """Create or update one recording's row; new columns are appended to the header.

    Called with no ``values`` it ensures an (otherwise empty) row exists — how a new
    recording gets its key row (§11: ``recording_id`` auto-populated on creation).
    """
    if not recording_id:
        raise ValueError("recording_id must be non-empty")
    cols, rows = _read_raw(experiment_dir)
    values = values or {}
    for key in values:
        if key != RECORDING_KEY and key not in cols:
            cols.append(key)
    row = rows.get(recording_id, {})
    row.update(
        {
            k: ("" if v is None else str(v))
            for k, v in values.items()
            if k != RECORDING_KEY
        }
    )
    rows[recording_id] = row
    _write_raw(experiment_dir, cols, rows)


def remove_row(experiment_dir: Path | str, recording_id: str) -> None:
    """Drop a recording's row. Idempotent — absent recording or file is a no-op."""
    if not metadata_path(experiment_dir).is_file():
        return
    cols, rows = _read_raw(experiment_dir)
    if recording_id in rows:
        del rows[recording_id]
        _write_raw(experiment_dir, cols, rows)


def add_column(experiment_dir: Path | str, name: str) -> None:
    """Add an (empty) user column so the cohort table can show it before any value."""
    if not name or name == RECORDING_KEY:
        raise ValueError(f"invalid metadata column name {name!r}")
    cols, rows = _read_raw(experiment_dir)
    if name not in cols:
        cols.append(name)
        _write_raw(experiment_dir, cols, rows)


def delete_column(experiment_dir: Path | str, name: str) -> None:
    """Remove a user column and its value from every row. Idempotent (absent → no-op);
    the reserved ``recording_id`` cannot be deleted."""
    if name == RECORDING_KEY:
        raise ValueError(f"cannot delete the reserved column {name!r}")
    cols, rows = _read_raw(experiment_dir)
    if name in cols:
        cols.remove(name)
        for values in rows.values():
            values.pop(name, None)
        _write_raw(experiment_dir, cols, rows)


def write_metadata(
    experiment_dir: Path | str,
    columns: list[str],
    rows: dict[str, dict[str, str]],
) -> None:
    """Replace the whole cohort table in one write: header ``recording_id`` + ``columns``,
    one row per ``rows`` entry. The bulk counterpart to :func:`upsert_row` — commits a fully
    edited table (added/deleted columns and cell edits together) atomically-in-intent.
    """
    _write_raw(experiment_dir, list(columns), dict(rows))


# --- bulk CSV import (§11) --------------------------------------------------- #
def read_import_header(path: Path | str) -> list[str]:
    """The stripped header row of an import CSV.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: An empty file, or an empty or duplicate column header.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    with open(p, newline="", encoding="utf-8-sig") as f:  # utf-8-sig tolerates a BOM
        header = next(csv.reader(f), None)
    if header is None:
        raise ValueError("CSV is empty")
    header = [h.strip() for h in header]
    if any(h == "" for h in header):
        raise ValueError("CSV has an empty column header")
    if len(header) != len(set(header)):
        raise ValueError("CSV has duplicate column headers")
    return header


def load_import_csv(
    path: Path | str, key_column: str | None = None
) -> tuple[str, list[str], list[tuple[str, dict[str, str]]]]:
    """Parse an external metadata CSV into ``(key_kind, data_columns, rows)`` where
    each row is ``(key_value, {column: value})`` in file order (duplicates preserved for
    validation). Cells are stripped; a fully blank line is skipped. ``key_kind`` is
    :func:`key_kind` of the key column.

    The key is whatever the person filling the sheet thinks of a recording as: its
    ``recording_id``, its ``name``, or — the one a sheet made before import can carry —
    its video file name (#330). ``key_column`` names it; left ``None`` it is
    :func:`detect_key_column`'s choice. Resolving a name or a file name to an id is the
    caller's job: both live in ``recording.yaml``, not in this file.

    The key column is dropped from the data, and so are the derived columns an export
    writes (``name``, ``file``) whichever column keys the import: they are never values to
    store. A ``name`` cell cannot rename a recording — that is ``Recording.set_name``.

    Raises:
        FileNotFoundError: If the file is absent.
        MissingKeyColumnError: No ``key_column`` given and none recognised.
        ValueError: Structural problems — empty file, empty or duplicate column headers,
            or a ``key_column`` the CSV does not have.
    """
    header = read_import_header(path)
    if key_column is None:
        key_column = detect_key_column(header)
        if key_column is None:
            raise MissingKeyColumnError(header)
    elif key_column not in header:
        raise ValueError(
            f"CSV has no column {key_column!r} — its columns are: {', '.join(header)}"
        )
    key_idx = header.index(key_column)
    derived = {key_column, RECORDING_KEY, NAME_KEY, FILE_KEY}
    columns = [h for h in header if h not in derived]
    rows: list[tuple[str, dict[str, str]]] = []
    with open(Path(path), newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # the header, validated above
        for raw in reader:
            if not any(c.strip() for c in raw):
                continue  # skip blank lines
            cells = (raw + [""] * len(header))[: len(header)]  # pad short rows
            values = {
                h: cells[i].strip() for i, h in enumerate(header) if h not in derived
            }
            rows.append((cells[key_idx].strip(), values))
    return key_kind(key_column), columns, rows


def validate_import(
    columns: list[str],
    rows: list[tuple[str, dict[str, str]]],
    existing_ids: list[str],
) -> tuple[list[str], list[str]]:
    """Validate parsed rows against the experiment's recordings. Returns
    ``(errors, warnings)`` — empty ``errors`` means safe to apply. Unknown
    ``recording_id`` is a hard error (the CSV must match existing recordings)."""
    errors: list[str] = []
    warnings: list[str] = []
    if not columns:
        warnings.append("no data columns besides recording_id — nothing to import")
    existing = set(existing_ids)
    seen: set[str] = set()
    for recording_id, _ in rows:
        if not recording_id:
            errors.append("a row has an empty recording_id")
        elif recording_id in seen:
            errors.append(f"duplicate recording_id in CSV: {recording_id!r}")
        else:
            seen.add(recording_id)
            if recording_id not in existing:
                errors.append(
                    f"unknown recording {recording_id!r} — not in this experiment"
                )
    return errors, warnings


def apply_import(
    experiment_dir: Path | str,
    columns: list[str],
    rows: list[tuple[str, dict[str, str]]],
) -> int:
    """Merge validated rows into ``metadata.csv``: add the CSV's columns, then for
    each row set its **non-blank** cells (blanks keep the existing value). Returns
    the number of recordings updated. Assumes :func:`validate_import` passed."""
    for column in columns:
        add_column(experiment_dir, column)
    for recording_id, values in rows:
        upsert_row(
            experiment_dir, recording_id, {k: v for k, v in values.items() if v != ""}
        )
    return len(rows)
