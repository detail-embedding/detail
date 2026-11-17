"""Generic timestamped-events CSV importer (Section 3).

Brings a user's own sparse event log in as an ``events`` / ``event`` stream: a CSV
with one named **time column** (``seconds`` | ``frame`` | ``t_ns``), a required
**``label``** text column, and an optional integer **``code``** column. The canonical
columns match MED-PC (``t_ns, code, label``) so an events consumer reads every source
uniformly; a missing ``code`` is null. The time column fixes the clock exactly as in
``timeseries_csv``.

Continuous sensor traces (numeric value channels, no ``label``) belong to the
``timeseries_csv`` importer — the user picks the shape by choosing the importer.
"""

from __future__ import annotations

from pathlib import Path

from detail.core.dataclasses import StreamImportResult
from detail.core.protocols import StreamImporter
from detail.core.registries import register_stream_importer
from detail.importers import _csv_common

_NS_PER_SECOND = 1_000_000_000


@register_stream_importer("events_csv")
class EventsCsvImporter(StreamImporter):
    """Import timestamped events from a generic named-column CSV."""

    importer_id = "events_csv"
    display_name = "Events CSV"
    stream_type = "events"
    role = None
    clock = "stream_seconds"
    aux_inputs = []

    def can_import(self, path: Path) -> bool:
        header, _first = _csv_common.sniff_header_and_first(Path(path))
        matches = _csv_common.time_column_matches(header)
        if len(matches) != 1:
            return False
        return _csv_common.has_column(header, _csv_common.LABEL_COLUMN)

    def import_stream(
        self,
        data_path: Path,
        out_dir: Path,
        aux_paths: dict[str, Path],
        config: dict | None = None,
    ) -> StreamImportResult:
        fieldnames, raw_rows = _csv_common.read_rows(Path(data_path))
        # Folded: every column here is matched case-insensitively (`find_column`).
        _csv_common.reject_duplicate_columns(fieldnames, fold=True)
        _csv_common.require_data_rows(raw_rows)
        time_source, time_key = _csv_common.resolve_time_column(fieldnames)
        label_source = _csv_common.find_column(fieldnames, _csv_common.LABEL_COLUMN)
        if label_source is None:
            raise ValueError(
                "events CSV has no 'label' column; add one, or use the Time-series "
                "CSV importer for numeric sensor data"
            )
        code_source = _csv_common.find_column(fieldnames, _csv_common.CODE_COLUMN)

        is_frame = time_key == "frame"
        columns = ["t_ns", "code", "label"] + (["frame"] if is_frame else [])
        rows: list[tuple] = []
        for i, raw in enumerate(raw_rows, start=2):  # row 1 is the header
            label = (raw.get(label_source) or "").strip()
            if not label:
                raise ValueError(f"events CSV row {i}: empty label")
            code = _parse_code(raw.get(code_source), i) if code_source else None
            if is_frame:
                frame = _csv_common.parse_int(raw.get(time_source), time_source, i)
                rows.append((None, code, label, frame))
            else:
                t_ns = _time_to_t_ns(raw.get(time_source), time_key, time_source, i)
                rows.append((t_ns, code, label))

        dtypes = {"t_ns": "int64", "code": "int64", "label": "str"}
        if is_frame:
            dtypes["frame"] = "int32"
        return StreamImportResult(
            type="events",
            shape="event",
            clock=_csv_common.TIME_CLOCKS[time_key],
            columns=columns,
            rows=rows,
            frame_columns={"t_ns": "frame"} if is_frame else None,
            dtypes=dtypes,
            metadata={"source": "events_csv"},
        )


def _parse_code(value: str | None, row: int) -> int | None:
    """Parse an optional integer event code; an empty cell is ``None``."""
    if value is None or str(value).strip() == "":
        return None
    return _csv_common.parse_int(value, _csv_common.CODE_COLUMN, row)


def _time_to_t_ns(value: str | None, time_key: str, column: str, row: int) -> int:
    """Convert a ``seconds`` or ``t_ns`` cell to int64 ``t_ns`` (Tenet 8)."""
    if time_key == "t_ns":
        return _csv_common.parse_int(value, column, row)
    return round(_csv_common.parse_float(value, column, row) * _NS_PER_SECOND)
