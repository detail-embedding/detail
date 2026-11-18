"""Stream storage: canonical CSVs, the recording.yaml manifest, and alignment (Section 3).

Each stream lives under ``streams/<name>/`` with ``raw/`` (the imported source
files, verbatim), ``canonical/`` (the converted CSV the pipeline reads), and
``alignment/`` (reserved for the §3 correction table). The stream's manifest is a
top-level ``streams:`` entry in ``recording.yaml``. Canonical ``t_ns`` is the
stream's **own clock**; the manifest ``alignment.offset_ns`` maps it to video time
(``video_t_ns = stream_t_ns + offset_ns``), applied at read time so re-aligning
is a one-number edit that never rewrites data. All paths are recording-relative.
"""

import csv
import re
import shutil
from pathlib import Path

from detail.core import atomic
from detail.core.errors import UserInputError
from detail.storage import clocks
from detail.storage import recording as recording_store

# Legacy ``type`` (events/sampled) → the ``shape`` axis (Section 3). Streams written
# before ``shape`` existed carry only ``type``; readers default through this.
_TYPE_TO_SHAPE = {"events": "event", "sampled": "continuous"}

# Where a ``behavior``-role stream's labels came from (§19). ``ground_truth`` is a human
# scoring the video; ``predicted`` is a method's output re-ingested as a stream.
#
# The default is ``ground_truth``, so every stream written before this field existed keeps
# meaning exactly what it meant. That is safe in the direction that matters: the streams on
# disk today were imported from annotation tools, and a *new* prediction has to travel through
# a writer that marks it.
BEHAVIOR_ORIGINS = frozenset({"ground_truth", "predicted"})
DEFAULT_BEHAVIOR_ORIGIN = "ground_truth"

STREAMS_DIRNAME = "streams"
CANONICAL_DIRNAME = "canonical"
RAW_DIRNAME = "raw"
ALIGNMENT_DIRNAME = "alignment"
CANONICAL_FILENAME = "part_001.csv"  # one continuous canonical file per stream

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_stream_name(name: str) -> None:
    """Validate a stream name is a safe single-segment directory component.

    Raises:
        ValueError: If ``name`` is empty or contains anything outside
            ``[A-Za-z0-9_-]`` (it is used as a directory name and a manifest key).
    """
    if not name:
        raise ValueError("stream name must be non-empty")
    if not _NAME_RE.match(name):
        raise ValueError(
            f"stream name {name!r} must contain only letters, digits, '-' and '_'"
        )


def stream_dir(recording_dir: Path | str, name: str) -> Path:
    return Path(recording_dir) / STREAMS_DIRNAME / name


def write_canonical_csv(
    out_dir: Path | str, columns: list[str], rows: list[tuple]
) -> str:
    """Write ``canonical/part_001.csv`` atomically; return its filename.

    Uses ``csv.writer`` so label fields containing commas are quoted correctly.
    The first column is ``t_ns`` (int64 video time, Section 3). Written via a
    sibling ``*.tmp`` + ``os.replace`` so a concurrent reader (a batch import runs
    under a worker) never sees a torn file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (CANONICAL_FILENAME + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    atomic.replace(tmp, out_dir / CANONICAL_FILENAME)
    return CANONICAL_FILENAME


def default_dtypes(columns: list[str]) -> dict[str, str]:
    """The dtype fallback for a manifest without an explicit ``dtypes`` map.

    Preserves the historical behavior — ``t_ns``/``end_t_ns``/``code`` and any
    ``*_frame`` column parse as ``int``, everything else stays ``str`` — so a
    MED-PC stream (and the manifest-less test callers) read unchanged.
    """
    ints = {"t_ns", "end_t_ns", "code"}
    return {c: "int64" if c in ints or c.endswith("_frame") else "str" for c in columns}


def _parse_cell(value: str, dtype: str):
    """Parse one CSV cell by its declared dtype; an empty cell is ``None``."""
    if value == "":
        return None
    if dtype in ("int64", "int32", "int"):
        return int(value)
    if dtype in ("float", "float64"):
        return float(value)
    return value


def read_canonical(
    recording_dir: Path | str, name: str, dtypes: dict[str, str] | None = None
) -> tuple[list[str], list[tuple]]:
    """Read a stream's canonical CSV into ``(columns, rows)``.

    Each cell is parsed by its column's dtype: manifest-aware callers pass
    ``dtypes`` (from the stream manifest); when omitted, ``default_dtypes`` gives
    the historical name-based fallback (``t_ns``/``code``/``*_frame`` → ``int``).
    An empty cell is ``None`` (a nullable column, e.g. an absent ``subject``).

    Raises:
        FileNotFoundError: If the canonical file is absent.
    """
    path = stream_dir(recording_dir, name) / CANONICAL_DIRNAME / CANONICAL_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no canonical stream file at {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        columns = next(reader)
        col_dtypes = dtypes if dtypes is not None else default_dtypes(columns)
        rows = [
            tuple(
                _parse_cell(v, col_dtypes.get(columns[i], "str"))
                for i, v in enumerate(row)
            )
            for row in reader
        ]
    return columns, rows


def stream_shape(manifest: dict) -> str:
    """The stream's ``shape`` (event/interval/continuous), Section 3.

    Reads the ``shape`` key; falls back to mapping a legacy ``type`` for streams
    written before ``shape`` existed.
    """
    shape = manifest.get("shape")
    if shape:
        return shape
    return _TYPE_TO_SHAPE.get(manifest.get("type", ""), "event")


def stream_origin(manifest: dict) -> str:
    """Where a stream's labels came from — ``ground_truth`` or ``predicted`` (§19).

    Falls back to ``ground_truth`` for a stream written before the field existed, which is
    what every stream on disk today is.
    """
    return str(manifest.get("origin") or DEFAULT_BEHAVIOR_ORIGIN)


def is_predicted(manifest: dict) -> bool:
    """Whether this stream holds a method's output rather than a human's scoring."""
    return stream_origin(manifest) == "predicted"


def streams_by_role(recording_dir: Path | str, role: str) -> list[dict]:
    """Stream manifest entries with the given ``role`` (Section 3).

    The discovery surface a role-consuming module reads through — e.g. the
    behavior eval selects recordings with a ``behavior``-role stream.
    """
    return [s for s in list_streams(recording_dir) if s.get("role") == role]


def append_stream_to_recording_yaml(recording_dir: Path | str, manifest: dict) -> None:
    """Append a stream manifest to top-level ``streams:`` in recording.yaml.

    Raises:
        ValueError: If a stream with the same ``name`` already exists.
    """
    name = manifest["name"]
    data = recording_store.read_recording(recording_dir)
    existing = list(data.get("streams", []))
    if any(entry.get("name") == name for entry in existing):
        raise ValueError(f"stream name {name!r} already exists in recording")
    existing.append(manifest)
    data["streams"] = existing
    recording_store.write_recording(recording_dir, data)


def list_streams(recording_dir: Path | str) -> list[dict]:
    """Return the ``streams:`` list from recording.yaml (empty list if none)."""
    return list(recording_store.read_recording(recording_dir).get("streams", []))


def delete_stream(recording_dir: Path | str, name: str) -> None:
    """Remove a stream's directory + its recording.yaml entry (idempotent)."""
    sdir = stream_dir(recording_dir, name)
    if sdir.is_dir():
        shutil.rmtree(sdir)
    data = recording_store.read_recording(recording_dir)
    data["streams"] = [s for s in data.get("streams", []) if s.get("name") != name]
    recording_store.write_recording(recording_dir, data)


def stream_offset_ns(manifest: dict) -> int:
    """A stream's stored alignment offset in ns, 0 when it has none (Section 3).

    **The one reader of the offset.** It lives at ``alignment.offset_ns``; a reader of a
    top-level ``offset_ns`` finds nothing and reports 0 for every stream, which is how the
    ``streams`` export once listed every offset as 0.
    """
    return int((manifest.get("alignment") or {}).get("offset_ns", 0) or 0)


def check_offset_allowed(name: str, clock: str | None, offset_ns: int) -> None:
    """Refuse an alignment offset for a stream already on the video's clock (Section 3).

    The one gate every alignment door calls (import, ``set_stream_offset``). A
    ``video_frames``/``video_seconds`` stream is anchored at import; an offset stored on one
    is ignored by scoring but drawn by the preview, so the two disagree with nothing to say
    so. A legacy manifest with no ``clock`` is an external-clock stream.

    Raises:
        UserInputError: If ``offset_ns`` is non-zero and ``clock`` is video-anchored.
    """
    clock = clock or "stream_seconds"
    if offset_ns and not clocks.needs_alignment(clock):
        raise UserInputError(
            f"stream {name!r} is on the {clock!r} clock, which is already the video's own, "
            "so it has no alignment offset to set (it would be ignored when scoring but "
            "drawn by the preview)"
        )


def set_stream_offset(recording_dir: Path | str, name: str, offset_ns: int) -> None:
    """Set a stream's constant alignment offset (signed ns; Section 3).

    **The one writer of an existing stream's alignment.** It refuses a video-anchored stream
    (``check_offset_allowed``), so every surface reaching it (API, CLI, GUI) is gated by the
    stored clock rather than by each remembering to check.

    Raises:
        ValueError: If no stream with this name exists, or it is on the video's clock.
    """
    data = recording_store.read_recording(recording_dir)
    streams = data.get("streams", [])
    match = next((s for s in streams if s.get("name") == name), None)
    if match is None:
        raise ValueError(f"unknown stream {name!r}")
    check_offset_allowed(name, match.get("clock"), int(offset_ns))
    match.setdefault("alignment", {})["offset_ns"] = int(offset_ns)
    match["alignment"]["active"] = "offset"
    recording_store.write_recording(recording_dir, data)


def apply_alignment(stream_t_ns: int, manifest: dict) -> int:
    """Map a stream-local ``t_ns`` to video ``t_ns`` via the manifest alignment.

    The simple case is a constant offset (``video_t_ns = stream_t_ns + offset_ns``);
    the §3 non-linear correction table is not built yet.

    Raises:
        NotImplementedError: If the manifest selects the (unbuilt) correction table.
    """
    alignment = manifest.get("alignment") or {}
    if alignment.get("active") not in (None, "offset"):
        raise NotImplementedError(
            "stream correction-table alignment is not implemented yet (Section 3)"
        )
    return int(stream_t_ns) + stream_offset_ns(manifest)
