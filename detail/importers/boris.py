"""BORIS behavior-observation importer (Section 3).

Reads a BORIS **tabular events** export (the "Export events" table, CSV or TSV) into the
canonical ``behavior``-role interval stream. Each row is one *event marker* — a Subject +
Behavior at a ``Time`` in **seconds**, with a ``Behavior type`` of ``START`` / ``STOP``
(a state behavior, spanning START→STOP) or ``POINT`` (an instantaneous behavior). The
importer pairs each START with its following STOP into a ``(start_t_ns, end_t_ns)`` interval
and emits a POINT as a zero-length interval, on the ``video_seconds`` clock (seconds → ns is
arithmetic; the import layer fills the convenience frame columns via the recording's inverse
timestamp map).

BORIS subjects are user-named strings, not track ids. The per-track ``subject`` framework is
int and unused for now, so each row's ``subject`` stays null and the subject imported is named
in the stream metadata. ``covered_behaviors`` is the distinct behavior set and
``behavior_types`` records each behavior's point/state kind (from its markers), both consumed
by the §16 eval. Matching is tolerant (case-insensitive, whitespace-stripped headers).

**One observation, one subject.** An export can hold several observations (``Observation id``)
and several subjects; a recording is one animal in one session, so the importer takes exactly
one of each. A file with more than one is refused until a
:class:`~detail.configs.stream_select.BorisSelectionConfig` names which, and the refusal lists
what the file holds. Merging them put one animal's bouts and another's on a single track, and
the evaluation then scored one against both.

**One kind per behavior.** ``behavior_types`` holds one kind per label, so a label scored both
as START/STOP bouts and as POINT rows cannot be stored truthfully and is refused, naming the
rows of each kind. Marking it all ``point`` (as this importer once did) had evaluation score a
one-second bout as an instant.
"""

import csv
import io
from pathlib import Path

from detail.configs.stream_select import BorisSelectionConfig
from detail.core.errors import UserInputError
from detail.core.dataclasses import StreamImportResult
from detail.core.protocols import StreamImporter
from detail.core.registries import register_stream_importer
from detail.storage import clocks

_COLUMNS = ["t_ns", "end_t_ns", "label", "subject", "start_frame", "end_frame"]
_DTYPES = {
    "t_ns": "int64",
    "end_t_ns": "int64",
    "label": "str",
    "subject": "int32",
    "start_frame": "int32",
    "end_frame": "int32",
}
_SUBJECT = "subject"
_OBSERVATION = "observation id"
_BEHAVIOR = "behavior"
_TYPE = "behavior type"
_TIME = "time"
_AGGREGATED = "start (s)"  # a column unique to the *aggregated* export we do not read


@register_stream_importer("boris")
class BorisImporter(StreamImporter):
    """Import behavior intervals from a BORIS tabular-events export."""

    importer_id = "boris"
    display_name = "Behavior labels (BORIS tabular events export)"
    stream_type = "events"
    role = "behavior"
    clock = "video_seconds"
    aux_inputs = []
    config_model = BorisSelectionConfig

    def can_import(self, path: Path) -> bool:
        try:
            names = _header_names(path)
        except (OSError, UserInputError):
            return False
        return {_BEHAVIOR, _TYPE, _TIME} <= names and _AGGREGATED not in names

    def import_stream(
        self,
        data_path: Path,
        out_dir: Path,
        aux_paths: dict[str, Path],
        config: dict | None = None,
    ) -> StreamImportResult:
        cfg = BorisSelectionConfig.model_validate(config or {})
        events, observations = _read_events(data_path)
        events, observation = _choose_observation(events, observations, cfg.observation)
        events, _subject = _choose_subject(events, cfg.subject)
        rows, subjects, labels, types = _pair_events(events)
        metadata = {
            "source": "boris",
            "subjects": sorted(subjects),
            "covered_behaviors": sorted(labels),
            "behavior_types": types,
        }
        if observation is not None:
            metadata["observation"] = observation
        if len(observations) > 1:
            metadata["observations_in_file"] = len(observations)
        return StreamImportResult(
            type="events",
            shape="interval",
            clock="video_seconds",
            columns=_COLUMNS,
            rows=rows,
            time_columns={"start_frame": "t_ns", "end_frame": "end_t_ns"},
            dtypes=_DTYPES,
            metadata=metadata,
        )


def _read_events(data_path: Path) -> tuple[list[tuple], list[str]]:
    """Every event row as ``(time_s, subject, label, marker, row_no, observation)``, and the
    distinct observation ids in file order (empty when the export has no such column).

    Raises:
        UserInputError: A required column is missing, or a row is malformed.
    """
    try:
        return _parse_events(data_path)
    except UnicodeDecodeError:
        raise _not_text(data_path) from None


def _parse_events(data_path: Path) -> tuple[list[tuple], list[str]]:
    """:func:`_read_events` without the decode check, streamed row by row."""
    delimiter = _delimiter_of(_first_line(data_path))
    observations: list[str] = []
    events = []
    with open(data_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fields = {n.strip().lower(): n for n in (reader.fieldnames or [])}
        _check_columns(data_path, reader.fieldnames or [], fields)
        for i, raw in enumerate(reader, start=2):
            label = (raw[fields[_BEHAVIOR]] or "").strip()
            if not label:
                raise UserInputError(f"BORIS row {i}: empty Behavior")
            marker = (raw[fields[_TYPE]] or "").strip().upper()
            if marker not in ("START", "STOP", "POINT"):
                raise UserInputError(
                    f"BORIS row {i}: unexpected Behavior type {marker!r} "
                    "(expected START, STOP, or POINT)"
                )
            subject = (
                (raw[fields[_SUBJECT]] or "").strip() if _SUBJECT in fields else ""
            )
            observation = None
            if _OBSERVATION in fields:
                observation = (raw[fields[_OBSERVATION]] or "").strip()
                if observation not in observations:
                    observations.append(observation)
            time_s = _float(raw[fields[_TIME]], "Time", i)
            events.append((time_s, subject, label, marker, i, observation))
    return events, observations


def list_selection(path: Path, observation: str | None = None) -> dict[str, list[str]]:
    """What a BORIS export holds: its observation ids, and the subjects of ``observation``
    (of the whole file when ``None``), each in file order. ``""`` is "no subject".

    What a picker lists, read the same way the import reads the file.
    """
    events, observations = _read_events(Path(path))
    subjects: list[str] = []
    for event in events:
        if observation is not None and event[5] != observation:
            continue
        if event[1] not in subjects:
            subjects.append(event[1])
    return {"observations": observations, "subjects": subjects}


def _subject_name(subject: str) -> str:
    return repr(subject) if subject else "(no subject)"


def _choose_observation(
    events: list[tuple], observations: list[str], wanted: str | None
) -> tuple[list[tuple], str | None]:
    """The events of the one observation imported, and its id (``None`` with no column).

    Raises:
        UserInputError: Several observations and none named, or a named one the file lacks.
    """
    if wanted is None:
        if len(observations) > 1:
            raise UserInputError(
                f"this BORIS export holds {len(observations)} observations "
                f"({', '.join(map(repr, observations))}). A recording is one session, so "
                "choose one observation; they are never merged"
            )
        return events, (observations[0] if observations else None)
    if wanted not in observations:
        held = ", ".join(map(repr, observations)) or "no 'Observation id' column"
        raise UserInputError(
            f"observation {wanted!r} is not in this BORIS export; it holds {held}"
        )
    return [e for e in events if e[5] == wanted], wanted


def _choose_subject(events: list[tuple], wanted: str | None) -> tuple[list[tuple], str]:
    """The events of the one subject imported, and that subject (``""`` for none).

    Raises:
        UserInputError: Several subjects and none named, or a named one with no rows.
    """
    subjects: list[str] = []
    for event in events:
        if event[1] not in subjects:
            subjects.append(event[1])
    if wanted is None:
        if len(subjects) > 1:
            raise UserInputError(
                f"this BORIS export scores {len(subjects)} subjects "
                f"({', '.join(map(_subject_name, subjects))}). A behavior stream is one "
                "animal's labels, so choose one subject; their bouts are never pooled"
            )
        return events, (subjects[0] if subjects else "")
    if wanted not in subjects:
        raise UserInputError(
            f"subject {_subject_name(wanted)} has no rows in this BORIS export; it scores "
            f"{', '.join(map(_subject_name, subjects)) or 'no rows'}"
        )
    return [e for e in events if e[1] == wanted], wanted


def _pair_events(events: list[tuple]):
    """Pair START/STOP markers (and emit POINTs) into canonical interval rows.

    Events are matched per ``(subject, behavior)`` in time order: a START opens the
    interval, the next STOP closes it (BORIS state events for one subject+behavior do not
    overlap). Returns ``(rows, subjects, labels, behavior_types)``.

    Raises:
        UserInputError: On a STOP without an open START, a second START before a STOP, or a
            START left unpaired at the end (a truncated / malformed export).
    """
    _refuse_mixed_kinds(events)
    ordered = sorted(events, key=lambda e: (e[0], e[4]))  # time, then file order
    rows: list[tuple] = []
    subjects: set[str] = set()
    labels: list[str] = []
    types: dict[str, str] = {}
    open_start: dict[tuple, tuple] = {}  # (subject, label) -> (time_s, row_no)
    for time_s, subject, label, marker, row_no, *_rest in ordered:
        if label not in types:
            labels.append(label)
            types[label] = "state"
        if subject:
            subjects.add(subject)
        key = (subject, label)
        if marker == "POINT":
            types[label] = "point"
            t = clocks.seconds_to_t_ns(time_s, what=f"BORIS row {row_no}, Time")
            rows.append((t, t, label, None, None, None))
        elif marker == "START":
            if key in open_start:
                raise UserInputError(
                    f"BORIS row {row_no}: START for {label!r} before the previous "
                    f"START (row {open_start[key][1]}) was STOPped"
                )
            open_start[key] = (time_s, row_no)
        else:  # STOP
            if key not in open_start:
                raise UserInputError(
                    f"BORIS row {row_no}: STOP for {label!r} with no open START"
                )
            start_s, start_row = open_start.pop(key)
            rows.append(
                (
                    clocks.seconds_to_t_ns(
                        start_s, what=f"BORIS row {start_row}, Time"
                    ),
                    clocks.seconds_to_t_ns(time_s, what=f"BORIS row {row_no}, Time"),
                    label,
                    None,
                    None,
                    None,
                )
            )
    if open_start:
        label, row = next(iter((k[1], v[1]) for k, v in open_start.items()))
        raise UserInputError(
            f"BORIS: START for {label!r} (row {row}) has no matching STOP "
            "(truncated export?)"
        )
    return rows, subjects, labels, types


def _refuse_mixed_kinds(events: list[tuple]) -> None:
    """Refuse a label scored both as START/STOP bouts and as POINT rows.

    ``behavior_types`` holds one kind per label, and evaluation scores a ``state`` label by
    interval overlap and a ``point`` label by a time window, so neither choice is true of both
    halves of such a label.

    Raises:
        UserInputError: Naming the label and the rows of each kind.
    """
    rows_by_kind: dict[str, dict[str, list[int]]] = {}
    for _time, _subject, label, marker, row_no, *_rest in events:
        kind = "point" if marker == "POINT" else "state"
        rows_by_kind.setdefault(label, {}).setdefault(kind, []).append(row_no)
    for label, kinds in rows_by_kind.items():
        if len(kinds) > 1:
            raise UserInputError(
                f"BORIS behavior {label!r} is scored both as START/STOP bouts (rows "
                f"{_rows(kinds['state'])}) and as POINT events (rows {_rows(kinds['point'])}). "
                "A behavior is one kind or the other; rename one of them in BORIS, or fix "
                "its behavior type, and export again"
            )


def _rows(numbers: list[int]) -> str:
    shown = ", ".join(str(n) for n in numbers[:5])
    return shown + (f" and {len(numbers) - 5} more" if len(numbers) > 5 else "")


def _header_names(path: Path) -> set[str]:
    """The header's column names, lower-cased; only the first line is read.

    Raises:
        UserInputError: The file is not UTF-8 text.
    """
    first = _first_line(path)
    header = next(csv.reader(io.StringIO(first), delimiter=_delimiter_of(first)), [])
    return {h.strip().lower() for h in header}


def _first_line(path: Path) -> str:
    """The export's first line, BOM-tolerantly.

    Raises:
        UserInputError: The file is not UTF-8 text.
    """
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            return f.readline()
    except UnicodeDecodeError:
        raise _not_text(path) from None


def _not_text(path: Path) -> UserInputError:
    """A file that is not UTF-8 text: a video or another file picked by mistake, or a sheet
    re-saved in the Windows encoding."""
    return UserInputError(
        f"{path} is not UTF-8 text, so it is not a BORIS export as BORIS writes it; "
        "export it again from BORIS with Export events > Tabular events"
    )


def _delimiter_of(first_line: str) -> str:
    """BORIS writes tab-separated (``.tsv``) or comma-separated exports: read either."""
    return "\t" if "\t" in first_line else ","


def _check_columns(path: Path, header: list[str], fields: dict) -> None:
    """Refuse an export this importer does not read, saying what it is and what to export.

    Raises:
        UserInputError: The *aggregated* export (one row per bout, ``Start (s)`` and
            ``Stop (s)``), or a file without the tabular events export's columns; the
            message lists the columns the file has.
    """
    if _AGGREGATED in fields:
        raise UserInputError(
            f"{path.name} is a BORIS aggregated export (one row per bout, with 'Start (s)' "
            "and 'Stop (s)' columns). This importer reads the tabular events export, one "
            "row per START, STOP or POINT with a 'Time' column: export the observation "
            "from BORIS again with Export events > Tabular events"
        )
    missing = [name for name in (_BEHAVIOR, _TYPE, _TIME) if name not in fields]
    if missing:
        has = ", ".join(repr(h) for h in header) or "no header row"
        raise UserInputError(
            f"{path.name} is not a BORIS tabular events export: it needs 'Behavior', "
            f"'Behavior type' and 'Time' columns, and has {has} (missing "
            f"{', '.join(repr(m) for m in missing)})"
        )


def _float(value: str, field: str, row: int) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        raise UserInputError(
            f"BORIS row {row}: {field} {value!r} is not a number"
        ) from None
