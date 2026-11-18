"""Continuous-signal CSV importer (Section 3).

Brings a user's own sensor traces in as a ``sampled`` / ``continuous`` stream: a CSV with a
**time column** and one or more **numeric channels**, each kept under its own name. The time
column fixes the clock (``seconds`` / ``t_ns`` / ``utc`` -> ``stream_seconds``, aligned later
by a read-time offset; ``frame`` -> ``video_frames``, resolved against the recording's
timestamp map). ``sample_rate_hz`` is the median inter-sample rate when time is resolvable at
import; ``None`` for the frame clock.

Two ways in, one importer. With **no config** it sniffs, as it always has: the time column
must be named ``seconds`` / ``frame`` / ``t_ns``, and every other column becomes a channel.
That is what ``can_import`` recognizes and what a file already shaped for DETAIL needs. With
a :class:`~detail.configs.stream_csv.CsvStreamConfig` the caller says which column is time,
what it holds, and which columns to bring in, so an arbitrary export imports under whatever
names its source tool used. The two paths meet immediately: both resolve to one
``(time_column, time_kind, channels)`` triple and share the rest.

One CSV produces **one** stream carrying the chosen channels. Splitting it per column would
multiply the per-stream alignment offset (channels off one device share a clock) and store a
verbatim copy of the source file per column.

Event-shaped data (a ``label`` column) belongs to the ``events_csv`` importer. That is
enforced on the sniffing path, where it is the only signal of intent; a caller naming
channels explicitly has already stated theirs.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from pathlib import Path

from detail.core import utils
from detail.configs.stream_csv import CsvChannel, CsvStreamConfig
from detail.core.dataclasses import StreamImportResult
from detail.core.protocols import StreamImporter
from detail.core.registries import register_stream_importer
from detail.importers import _csv_common

_NS_PER_SECOND = 1_000_000_000

# Column names the canonical schema owns. A channel may not take one: `t_ns` is the stream's
# int64 time anchor (Tenet 8) and `frame` is the convenience index beside it, so a channel of
# either name writes a second column of that name into one CSV and overwrites the reserved
# column's dtype with `float`. Nothing downstream can tell the two apart afterwards.
_RESERVED_COLUMNS = ("t_ns", "frame")

# The clock each *time kind* implies. ``_csv_common.TIME_CLOCKS`` is the same table keyed by
# the sniffable column *names*; this one adds ``utc``, which no column name spells and which
# rides the ``stream_seconds`` path once parsed to stream-relative time.
_KIND_CLOCKS: dict[str, str] = {**_csv_common.TIME_CLOCKS, "utc": "stream_seconds"}


@register_stream_importer("timeseries_csv")
class TimeseriesCsvImporter(StreamImporter):
    """Import a continuous sensor trace from a named-column CSV."""

    importer_id = "timeseries_csv"
    display_name = "Time-series CSV (sensor data)"
    stream_type = "sampled"
    role = None
    clock = "stream_seconds"
    aux_inputs = []
    config_model = CsvStreamConfig
    maps_unrecognized_files = True

    def can_import(self, path: Path) -> bool:
        header, first = _csv_common.sniff_header_and_first(Path(path))
        matches = _csv_common.time_column_matches(header)
        if len(matches) != 1:
            return False
        if _csv_common.has_column(header, _csv_common.LABEL_COLUMN):
            return False  # a label column is event data, not a continuous trace
        time_source, _key = matches[0]
        value_cols = [c for c in header if c != time_source]
        if not value_cols:
            return False
        if first is None:
            return (
                True  # header-only: accept on shape; import will reject the emptiness
            )
        idx = {c: i for i, c in enumerate(header)}
        for col in value_cols:
            i = idx[col]
            cell = first[i] if i < len(first) else ""
            try:
                float(str(cell).strip())
            except (TypeError, ValueError):
                return False
        return True

    def resolve_clock(self, path: Path, config: dict | None = None) -> str:
        """The clock the chosen (or sniffed) time column implies: ``frame`` is the video's."""
        cfg = CsvStreamConfig.model_validate(config or {})
        kind = cfg.time_kind
        if kind == "auto":
            name = cfg.time_column
            if name is None:
                header, _first = _csv_common.sniff_header_and_first(Path(path))
                matches = _csv_common.time_column_matches(header)
                if len(matches) != 1:
                    return self.clock
                name = matches[0][0]
            kind = name.strip().lower()
        return _KIND_CLOCKS.get(kind, self.clock)

    def import_stream(
        self,
        data_path: Path,
        out_dir: Path,
        aux_paths: dict[str, Path],
        config: dict | None = None,
    ) -> StreamImportResult:
        cfg = CsvStreamConfig.model_validate(config or {})
        fieldnames, raw_rows = _csv_common.read_rows(Path(data_path))
        _csv_common.reject_duplicate_columns(fieldnames)
        _csv_common.require_data_rows(raw_rows)

        time_source, time_kind = _resolve_time(fieldnames, cfg)
        channels = _resolve_channels(fieldnames, time_source, cfg)

        is_frame = time_kind == "frame"
        columns = (
            ["t_ns"] + (["frame"] if is_frame else []) + [c.name for c in channels]
        )

        # UTC needs both passes up front: every row's absolute instant, then the first row's
        # instant as the zero point the rest are relative to.
        utc_t_ns: list[int] | None = None
        anchor_iso: str | None = None
        if time_kind == "utc":
            utc_t_ns, anchor_iso = _utc_series(raw_rows, time_source)

        rows: list[tuple] = []
        t_ns_values: list[int] = []
        for i, raw in enumerate(raw_rows, start=2):  # row 1 is the header
            values = [
                _csv_common.parse_float(raw.get(c.column), c.column, i)
                for c in channels
            ]
            if is_frame:
                frame = _csv_common.parse_int(raw.get(time_source), time_source, i)
                rows.append((None, frame, *values))
            elif utc_t_ns is not None:
                t_ns_values.append(utc_t_ns[i - 2])
                rows.append((utc_t_ns[i - 2], *values))
            else:
                t_ns = _time_to_t_ns(raw.get(time_source), time_kind, time_source, i)
                t_ns_values.append(t_ns)
                rows.append((t_ns, *values))

        dtypes = {"t_ns": "int64"}
        if is_frame:
            dtypes["frame"] = "int32"
        for channel in channels:
            dtypes[channel.name] = "float"
        # The mapping actually used, not the one passed: `auto` is resolved here, and a run
        # whose record says `auto` cannot be reproduced once the file is gone.
        metadata = {
            "source": "timeseries_csv",
            "channels": [c.name for c in channels],
            "source_columns": [c.column for c in channels],
            "time_column": time_source,
            "time_kind": time_kind,
        }
        if anchor_iso is not None:
            # What wall-clock instant t_ns=0 is. The only record of it once alignment takes
            # over from here (Sync to video, or a manual offset).
            metadata["utc_anchor_iso"] = anchor_iso
        return StreamImportResult(
            type="sampled",
            shape="continuous",
            clock=_KIND_CLOCKS[time_kind],
            columns=columns,
            rows=rows,
            frame_columns={"t_ns": "frame"} if is_frame else None,
            dtypes=dtypes,
            sample_rate_hz=_sample_rate_hz(t_ns_values),
            metadata=metadata,
        )


def _resolve_time(fieldnames: list[str], cfg: CsvStreamConfig) -> tuple[str, str]:
    """The ``(time_column, time_kind)`` this import will use, both resolved.

    Raises:
        ValueError: If the named column is absent, no time column can be sniffed, or the
            kind is ``auto`` and the column's name does not say what it holds.
    """
    if cfg.time_column is None:
        time_source, sniffed = _csv_common.resolve_time_column(fieldnames)
    else:
        if cfg.time_column not in fieldnames:
            raise ValueError(
                f"time column {cfg.time_column!r} is not in the CSV header {fieldnames}"
            )
        time_source = cfg.time_column
        key = time_source.strip().lower()
        sniffed = key if key in _csv_common.TIME_CLOCKS else None
    if cfg.time_kind != "auto":
        return time_source, cfg.time_kind
    if sniffed is None:
        raise ValueError(
            f"cannot tell what column {time_source!r} holds from its name; set time_kind "
            f"to one of frame, seconds, t_ns, utc"
        )
    return time_source, sniffed


def _resolve_channels(
    fieldnames: list[str], time_source: str, cfg: CsvStreamConfig
) -> list[CsvChannel]:
    """The channels this import will write, defaulting to every non-time column.

    Raises:
        ValueError: If a named column is absent, is the time column, or a channel name
            collides with a reserved canonical column.
    """
    if not cfg.channels:
        if _csv_common.has_column(fieldnames, _csv_common.LABEL_COLUMN):
            raise ValueError(
                "time-series CSV has a 'label' column, which looks like event data; use "
                "the Events CSV importer, or name the channels you want explicitly"
            )
        channels = [
            CsvChannel(column=c, name=c) for c in fieldnames if c != time_source
        ]
        if not channels:
            raise ValueError(
                "time-series CSV has no value columns; add at least one numeric channel "
                f"beside the {time_source!r} time column"
            )
        return channels

    channels = [
        CsvChannel(column=c.column, name=c.name or c.column) for c in cfg.channels
    ]
    for channel in channels:
        if channel.column not in fieldnames:
            raise ValueError(
                f"channel column {channel.column!r} is not in the CSV header {fieldnames}"
            )
        if channel.column == time_source:
            raise ValueError(
                f"column {channel.column!r} is the time column; it cannot also be a channel"
            )
        if channel.name in _RESERVED_COLUMNS:
            raise ValueError(
                f"channel name {channel.name!r} is reserved by the canonical stream "
                f"schema {list(_RESERVED_COLUMNS)}; give the channel another name"
            )
    return channels


def _utc_series(
    raw_rows: list[dict[str, str]], time_source: str
) -> tuple[list[int], str]:
    """Every row's ``t_ns`` relative to the first row, and that row's instant as ISO 8601."""
    epoch_s = [
        _csv_common.parse_utc_epoch_seconds(raw.get(time_source), time_source, i)
        for i, raw in enumerate(raw_rows, start=2)
    ]
    anchor = epoch_s[0]
    t_ns = [
        utils.seconds_to_t_ns(s - anchor, what=f"row {i}, column {time_source!r}")
        for i, s in enumerate(epoch_s, start=2)
    ]
    return t_ns, datetime.fromtimestamp(anchor, tz=timezone.utc).isoformat()


def _time_to_t_ns(value: str | None, time_kind: str, column: str, row: int) -> int:
    """Convert a ``seconds`` or ``t_ns`` cell to int64 ``t_ns`` (Tenet 8)."""
    return _csv_common.cell_to_t_ns(value, time_kind, column, row)


def _sample_rate_hz(t_ns_values: list[int]) -> float | None:
    """Median inter-sample rate in Hz, or ``None`` when it can't be measured."""
    if len(t_ns_values) < 2:
        return None
    diffs = [b - a for a, b in zip(t_ns_values, t_ns_values[1:])]
    median_dt = statistics.median(diffs)
    if median_dt <= 0:
        return None
    return _NS_PER_SECOND / median_dt
