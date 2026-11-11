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
int and unused for now, so subjects seen are recorded in the stream metadata and each row's
``subject`` stays null. ``covered_behaviors`` is the distinct behavior set and
``behavior_types`` records each behavior's point/state kind (from its markers), both consumed
by the §16 eval. Matching is tolerant (case-insensitive, whitespace-stripped headers).
"""

import csv
from pathlib import Path

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

    def can_import(self, path: Path) -> bool:
        try:
            names = _header_names(path)
        except OSError:
            return False
        return {_BEHAVIOR, _TYPE, _TIME} <= names and _AGGREGATED not in names

    def import_stream(
        self,
        data_path: Path,
        out_dir: Path,
        aux_paths: dict[str, Path],
        config: dict | None = None,
    ) -> StreamImportResult:
        delimiter = _sniff_delimiter(data_path)
        with open(data_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fields = {n.strip().lower(): n for n in (reader.fieldnames or [])}
            for required in (_BEHAVIOR, _TYPE, _TIME):
                if required not in fields:
                    raise ValueError(
                        "BORIS events export needs 'Behavior', 'Behavior type', and "
                        f"'Time' columns (missing {required!r})"
                    )
            events = []
            for i, raw in enumerate(reader, start=2):
                label = (raw[fields[_BEHAVIOR]] or "").strip()
                if not label:
                    raise ValueError(f"BORIS row {i}: empty Behavior")
                marker = (raw[fields[_TYPE]] or "").strip().upper()
                if marker not in ("START", "STOP", "POINT"):
                    raise ValueError(
                        f"BORIS row {i}: unexpected Behavior type {marker!r} "
                        "(expected START, STOP, or POINT)"
                    )
                subject = (
                    (raw[fields[_SUBJECT]] or "").strip() if _SUBJECT in fields else ""
                )
                time_s = _float(raw[fields[_TIME]], "Time", i)
                events.append((time_s, subject, label, marker, i))

        rows, subjects, labels, types = _pair_events(events)
        metadata = {
            "source": "boris",
            "subjects": sorted(subjects),
            "covered_behaviors": sorted(labels),
            "behavior_types": types,
        }
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


def _pair_events(events: list[tuple]):
    """Pair START/STOP markers (and emit POINTs) into canonical interval rows.

    Events are matched per ``(subject, behavior)`` in time order: a START opens the
    interval, the next STOP closes it (BORIS state events for one subject+behavior do not
    overlap). Returns ``(rows, subjects, labels, behavior_types)``.

    Raises:
        ValueError: On a STOP without an open START, a second START before a STOP, or a
            START left unpaired at the end (a truncated / malformed export).
    """
    ordered = sorted(events, key=lambda e: (e[0], e[4]))  # time, then file order
    rows: list[tuple] = []
    subjects: set[str] = set()
    labels: list[str] = []
    types: dict[str, str] = {}
    open_start: dict[tuple, tuple] = {}  # (subject, label) -> (time_s, row_no)
    for time_s, subject, label, marker, row_no in ordered:
        if label not in types:
            labels.append(label)
            types[label] = "state"
        if subject:
            subjects.add(subject)
        key = (subject, label)
        if marker == "POINT":
            types[label] = "point"
            t = clocks.seconds_to_t_ns(time_s)
            rows.append((t, t, label, None, None, None))
        elif marker == "START":
            if key in open_start:
                raise ValueError(
                    f"BORIS row {row_no}: START for {label!r} before the previous "
                    f"START (row {open_start[key][1]}) was STOPped"
                )
            open_start[key] = (time_s, row_no)
        else:  # STOP
            if key not in open_start:
                raise ValueError(
                    f"BORIS row {row_no}: STOP for {label!r} with no open START"
                )
            start_s, _ = open_start.pop(key)
            rows.append(
                (
                    clocks.seconds_to_t_ns(start_s),
                    clocks.seconds_to_t_ns(time_s),
                    label,
                    None,
                    None,
                    None,
                )
            )
    if open_start:
        label, row = next(iter((k[1], v[1]) for k, v in open_start.items()))
        raise ValueError(
            f"BORIS: START for {label!r} (row {row}) has no matching STOP "
            "(truncated export?)"
        )
    return rows, subjects, labels, types


def _header_names(path: Path) -> set[str]:
    delimiter = _sniff_delimiter(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f, delimiter=delimiter), [])
    return {h.strip().lower() for h in header}


def _sniff_delimiter(path: Path) -> str:
    with open(path, newline="", encoding="utf-8-sig") as f:
        first = f.readline()
    return "\t" if "\t" in first else ","


def _float(value: str, field: str, row: int) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"BORIS row {row}: {field} {value!r} is not a number"
        ) from None
