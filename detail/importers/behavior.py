"""Behavior ground-truth label importer — a plain interval CSV (Section 3).

Reads human-scored behavior intervals as ``start_frame,end_frame,label[,subject]``
(frame-indexed, the ``video_frames`` clock) and emits the canonical ``behavior``-role
interval stream. ``subject`` is optional — the per-track framework, written but unused
for now. Frame → ``t_ns`` resolution happens in the import layer (Approach Y), so this
importer never touches the recording's timestamp map.
"""

import csv
from pathlib import Path

from detail.core.dataclasses import StreamImportResult
from detail.core.protocols import StreamImporter
from detail.core.registries import register_stream_importer
from detail.storage import streams as streams_store

_COLUMNS = ["t_ns", "end_t_ns", "label", "subject", "start_frame", "end_frame"]
_DTYPES = {
    "t_ns": "int64",
    "end_t_ns": "int64",
    "label": "str",
    "subject": "int32",
    "start_frame": "int32",
    "end_frame": "int32",
}
_REQUIRED = ("start_frame", "end_frame", "label")


@register_stream_importer("behavior_csv")
class BehaviorCsvImporter(StreamImporter):
    """Import frame-indexed behavior intervals from a plain CSV."""

    importer_id = "behavior_csv"
    display_name = "Behavior labels (interval CSV)"
    stream_type = "events"
    role = "behavior"
    clock = "video_frames"
    aux_inputs = []

    def can_import(self, path: Path) -> bool:
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                header = next(csv.reader(f), [])
        except OSError:
            return False
        names = {h.strip().lower() for h in header}
        return all(c in names for c in _REQUIRED)

    def import_stream(
        self,
        data_path: Path,
        out_dir: Path,
        aux_paths: dict[str, Path],
        config: dict | None = None,
    ) -> StreamImportResult:
        with open(data_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = {name.strip().lower(): name for name in (reader.fieldnames or [])}
            missing = [c for c in _REQUIRED if c not in fields]
            if missing:
                raise ValueError(
                    f"behavior CSV is missing required column(s) {missing}; needs "
                    "start_frame,end_frame,label[,subject]"
                )
            has_subject = "subject" in fields
            has_origin = "origin" in fields
            origins: set[str] = set()
            rows: list[tuple] = []
            for i, raw in enumerate(reader, start=2):  # row 1 is the header
                start_frame = _int(raw[fields["start_frame"]], "start_frame", i)
                end_frame = _int(raw[fields["end_frame"]], "end_frame", i)
                label = (raw[fields["label"]] or "").strip()
                if not label:
                    raise ValueError(f"behavior CSV row {i}: empty label")
                if end_frame < start_frame:
                    raise ValueError(
                        f"behavior CSV row {i}: end_frame {end_frame} is before "
                        f"start_frame {start_frame}"
                    )
                subject = None
                if has_subject:
                    cell = (raw[fields["subject"]] or "").strip()
                    subject = _int(cell, "subject", i) if cell else None
                if has_origin:
                    cell = (raw[fields["origin"]] or "").strip()
                    if cell:
                        origins.add(cell)
                rows.append((None, None, label, subject, start_frame, end_frame))
        metadata = _behavior_metadata(rows)
        origin = _resolve_origin(origins)
        if origin is not None:
            metadata["origin"] = origin
        return StreamImportResult(
            type="events",
            shape="interval",
            clock="video_frames",
            columns=_COLUMNS,
            rows=rows,
            frame_columns={"t_ns": "start_frame", "end_t_ns": "end_frame"},
            dtypes=_DTYPES,
            metadata=metadata,
        )


def _resolve_origin(origins: set[str]) -> str | None:
    """The file's declared origin (§19), or ``None`` when it does not declare one.

    The tool's own exporters write this column, so a re-ingested prediction identifies itself
    rather than depending on the user remembering to say so. A hand-written CSV has no column
    and the import layer defaults it to ground truth.

    Raises:
        ValueError: The file declares more than one origin, or one that is not a known value.
            A file that is half prediction is not something to guess about.
    """
    if not origins:
        return None
    if len(origins) > 1:
        raise ValueError(
            f"behavior CSV declares more than one origin ({sorted(origins)}); a file's "
            "labels are either a human's scoring or a method's output, not both"
        )
    origin = origins.pop()
    if origin not in streams_store.BEHAVIOR_ORIGINS:
        raise ValueError(
            f"behavior CSV origin {origin!r} is not one of "
            f"{sorted(streams_store.BEHAVIOR_ORIGINS)}"
        )
    return origin


def _behavior_metadata(rows: list[tuple]) -> dict:
    """Covered vocabulary + per-behavior point/state type inferred from the rows.

    A label is a ``state`` behavior if any of its intervals has duration
    (``end_frame > start_frame``); a label all of whose intervals are zero-length
    is a ``point`` behavior.
    """
    labels: list[str] = []
    has_duration: dict[str, bool] = {}
    for _t, _e, label, _s, start_frame, end_frame in rows:
        if label not in has_duration:
            has_duration[label] = False
            labels.append(label)
        if end_frame > start_frame:
            has_duration[label] = True
    types = {label: ("state" if has_duration[label] else "point") for label in labels}
    return {"covered_behaviors": sorted(labels), "behavior_types": types}


def _int(value: str, field: str, row: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"behavior CSV row {row}: {field} {value!r} is not an integer"
        ) from None
