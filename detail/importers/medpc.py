"""MED-PC (Med Associates) stream importer (Section 3).

MED-PC operant-chamber recordings are exported as two files:

* a **data** ``.txt`` — a ``Key: value`` header, single-letter scalar variables
  (``F:``–``Z:``), and array variables (``A:``–``E:``, ``T:``) dumped as a letter
  on its own line followed by index-prefixed rows of values; and
* a **program** ``.MPC`` (MedState Notation) whose comment block
  ``Event stamps ( array E )`` maps each integer event code to a label.

The recorded events live in two parallel arrays written in lockstep
(``SET E(G)=code, T(G)=H; ADD G``): ``E[i]`` is an integer event code and
``T[i]`` its time in **seconds** (final value == ``H``, the recording length).
Empty array slots use the sentinel ``-987.987``. This importer pairs E with T,
drops sentinels, maps each code to its label, and emits the canonical
``events`` rows ``(t_ns, code, label)`` in stream-local time.
"""

import re
from pathlib import Path

from detail.core.dataclasses import AuxInput, StreamImportResult
from detail.core.protocols import StreamImporter
from detail.core.registries import register_stream_importer

SENTINEL = -987.987  # MED-PC's "empty array slot" marker
_NS_PER_SECOND = 1_000_000_000

_ARRAY_HEADER_RE = re.compile(r"^([A-Z]):\s*$")
_SCALAR_RE = re.compile(r"^([A-Z]):\s+(-?\d+(?:\.\d+)?)\s*$")
_ARRAY_ROW_RE = re.compile(r"^\s+(\d+):\s+(.*)$")
_HEADER_RE = re.compile(r"^([^:]+):\s?(.*)$")
_EVENT_MAP_LINE_RE = re.compile(r"^(\d+)\s*=\s*(.+?)\s*$")

# Header keys (data .txt) carried into the manifest metadata for provenance.
_META_KEYS = (
    ("Subject", "subject"),
    ("Experiment", "experiment"),
    ("MSN", "msn"),
    ("Start Date", "start_date"),
    ("Start Time", "start_time"),
    ("End Time", "end_time"),
    ("Box", "box"),
    ("Group", "group"),
)


def parse_medpc_data(
    text: str,
) -> tuple[dict[str, str], dict[str, float], dict[str, list[float]]]:
    """Parse a MED-PC data ``.txt`` into ``(header, scalars, arrays)``.

    ``header`` maps the leading ``Key: value`` lines; ``scalars`` maps each
    single-letter variable (``F:``–``Z:``) to its float; ``arrays`` maps each
    array variable (``A:``, …, ``T:``) to its flat list of values in index order.
    """
    header: dict[str, str] = {}
    scalars: dict[str, float] = {}
    arrays: dict[str, list[float]] = {}
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        array_header = _ARRAY_HEADER_RE.match(line)
        if array_header:
            name = array_header.group(1)
            i += 1
            values: list[float] = []
            while i < n:
                row = _ARRAY_ROW_RE.match(lines[i])
                if row is None:
                    break
                values.extend(float(tok) for tok in row.group(2).split())
                i += 1
            arrays[name] = values
            continue
        scalar = _SCALAR_RE.match(line)
        if scalar:
            scalars[scalar.group(1)] = float(scalar.group(2))
            i += 1
            continue
        kv = _HEADER_RE.match(line)
        if kv:
            header[kv.group(1).strip()] = kv.group(2).strip()
        i += 1
    return header, scalars, arrays


def parse_event_map(mpc_text: str) -> dict[int, str]:
    """Extract the event-code → label map from a MED-PC ``.MPC`` program.

    The map lives in the comment block introduced by ``Event stamps ( array E )``
    as ``<int> = <label>`` lines (each comment-prefixed by ``\\``). Reading stops
    at the next section (a non-blank comment line that is not a ``code = label``).

    Raises:
        ValueError: If no ``Event stamps`` block is present, or it has no entries.
    """
    lines = mpc_text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if re.search(r"event stamps", line, re.IGNORECASE):
            start = idx + 1
            break
    if start is None:
        raise ValueError(
            "MED-PC program has no 'Event stamps ( array E )' block — cannot map "
            "event codes to labels"
        )
    event_map: dict[int, str] = {}
    for line in lines[start:]:
        content = line.lstrip("\\").strip()
        if not content:
            continue
        entry = _EVENT_MAP_LINE_RE.match(content)
        if entry is None:
            break  # reached the next section (e.g. "Variables")
        event_map[int(entry.group(1))] = entry.group(2)
    if not event_map:
        raise ValueError(
            "MED-PC program 'Event stamps' block has no '<code> = <label>' entries"
        )
    return event_map


@register_stream_importer("medpc")
class MedPCImporter(StreamImporter):
    """Import MED-PC operant-chamber events (data ``.txt`` + program ``.MPC``)."""

    importer_id = "medpc"
    display_name = "MED-PC (Med Associates)"
    stream_type = "events"
    aux_inputs = [AuxInput("program", "MED-PC program (.MPC)", [".mpc"])]

    def can_import(self, path: Path) -> bool:
        try:
            head = Path(path).read_text(encoding="utf-8-sig", errors="replace")[:2048]
        except OSError:
            return False
        return "MSN:" in head and "Start Date:" in head

    def import_stream(
        self,
        data_path: Path,
        out_dir: Path,
        aux_paths: dict[str, Path],
        config: dict | None = None,
    ) -> StreamImportResult:
        program_path = aux_paths.get("program")
        if program_path is None:
            raise ValueError(
                "MED-PC import needs the 'program' (.MPC) file for event labels"
            )
        header, _scalars, arrays = parse_medpc_data(
            Path(data_path).read_text(encoding="utf-8-sig", errors="replace")
        )
        event_map = parse_event_map(
            Path(program_path).read_text(encoding="utf-8-sig", errors="replace")
        )
        codes = arrays.get("E")
        times = arrays.get("T")
        if codes is None or times is None:
            raise ValueError(
                "MED-PC data file has no E (events) and T (times) arrays — got "
                f"arrays {sorted(arrays)}"
            )
        if len(codes) != len(times):
            raise ValueError(
                f"MED-PC E and T arrays differ in length ({len(codes)} vs "
                f"{len(times)}) — file may be truncated"
            )
        rows: list[tuple] = []
        for code, time in zip(codes, times):
            if _is_sentinel(code) or _is_sentinel(time):
                continue
            code_int = int(round(code))
            t_ns = round(time * _NS_PER_SECOND)
            label = event_map.get(code_int, f"event_{code_int}")
            rows.append((t_ns, code_int, label))
        metadata = {meta: header[key] for key, meta in _META_KEYS if key in header}
        metadata["event_map"] = {str(code): label for code, label in event_map.items()}
        return StreamImportResult(
            type="events",
            columns=["t_ns", "code", "label"],
            rows=rows,
            metadata=metadata,
        )


def _is_sentinel(value: float) -> bool:
    return abs(value - SENTINEL) < 1e-3
