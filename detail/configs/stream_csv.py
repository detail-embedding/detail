"""CSV stream-importer config model (Section 3) — how to read an arbitrary CSV as a stream.

``timeseries_csv`` recognizes a file by *sniffing* a fixed time-column name
(``seconds`` / ``frame`` / ``t_ns``) and takes every remaining column as a channel. That
works for a file already shaped for DETAIL and not for an arbitrary export, where the time
column is called whatever the source tool called it (``t_sec``, ``Timestamp``) and only some
of the columns are wanted.

This model is what a caller hands the importer to say so. **Every field defaults to the
sniffing behavior**, so ``config=None`` and an all-default config are the same import: a
stream written before this model existed still reads, and a bundle manifest that names no
config still works.

One CSV makes **one** stream carrying the chosen channels, not one stream per column. The
alignment offset is per stream and channels off one device share a clock, so splitting a
40-column pose export into 40 streams would mean 40 copies of one offset to keep in step by
hand, and 40 verbatim copies of the source file in ``raw/``.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

#: What a time column can hold. ``utc`` has no clock of its own: it is parsed down to
#: stream-relative seconds anchored at the first data row, and rides the ``stream_seconds``
#: path like ``seconds`` and ``t_ns``.
#:
#: ``auto`` is a member rather than a ``None`` default. It was chosen when a
#: ``Literal[...] | None`` lost its ``enum`` inside the schema's ``anyOf``; ``describe_config``
#: now reads it from there (and ``choice_combo`` adds an "auto" row for the None), so either
#: shape works — this one stays because stored configs carry ``"auto"``.
TimeKind = Literal["auto", "frame", "seconds", "t_ns", "utc"]


class CsvChannel(BaseModel):
    """One CSV column brought in as a named channel of the stream."""

    column: str = Field(
        description="Header name of the column in the source CSV.",
        json_schema_extra={"label": "Column"},
    )
    name: str = Field(
        description="Name this channel carries in the canonical stream. Defaults to the "
        "column's own name when not given.",
        json_schema_extra={"label": "Channel name"},
    )


class CsvStreamConfig(BaseModel):
    """Explicit column mapping for a CSV stream import (Section 3)."""

    time_column: str | None = Field(
        None,
        description="Header name of the column carrying time. Leave unset to auto-detect a "
        "column named seconds, frame or t_ns, which is what this importer did before a "
        "mapping could be given.",
        json_schema_extra={"label": "Time column"},
    )
    time_kind: TimeKind = Field(
        "auto",
        description="What the time column holds: a video frame index, seconds since the "
        "stream's own start, integer nanoseconds since that start, or a UTC timestamp "
        "(ISO 8601 or Unix epoch seconds, with the first data row taken as t=0). 'auto' "
        "reads it from the time column's name.",
        json_schema_extra={"label": "Time is"},
    )
    channels: list[CsvChannel] = Field(
        default_factory=list,
        description="Which columns to bring in, each as a named channel of the one stream. "
        "Leave empty to take every column except the time column.",
        json_schema_extra={"label": "Channels"},
    )

    @field_validator("channels")
    @classmethod
    def _distinct(cls, value: list[CsvChannel]) -> list[CsvChannel]:
        """Reject a duplicate source column or channel name up front (Tenet 10).

        Both are silent corruptions rather than errors otherwise: two channels named the
        same write two identically-named columns into one canonical CSV, and the reader
        keeps whichever it saw last.
        """
        for key, label in (("column", "source column"), ("name", "channel name")):
            seen: set[str] = set()
            for channel in value:
                got = getattr(channel, key)
                if got in seen:
                    raise ValueError(f"{label} {got!r} is listed twice")
                seen.add(got)
        return value
