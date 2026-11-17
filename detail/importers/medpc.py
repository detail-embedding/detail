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

**The one layout it reads** (and refuses others rather than guess): events in array ``E``,
their times in array ``T`` in **seconds**, and an ``Event stamps ( array E )`` comment block
in the program. A program logging into other arrays is refused by name; one logging ``T`` in
ticks (10 ms is common) is caught when a time exceeds the session length ``H``.

**Several sessions in one file.** MED-PC can save every box of a day into one data file, one
``Start Date:`` block each. A recording is one animal, so such a file is refused until a
:class:`~detail.configs.stream_select.MedpcSessionConfig` names one session, and the refusal
lists every session found. Keeping the last block, as this importer once did, silently gave a
recording another animal's events.

**Unused zero slots.** A program that dimensions ``E``/``T`` larger than it fills, and does not
seal them with the sentinel, prints the unused slots as ``0.000``. A trailing run of
``(code 0, time 0)`` pairs is dropped and counted in the stream's ``metadata.notes``; a code-0
event anywhere else is refused unless the program's event map names code 0.
"""

import re
from pathlib import Path

from detail.configs.stream_select import MedpcSessionConfig
from detail.core import utils
from detail.core.dataclasses import AuxInput, StreamImportResult
from detail.core.protocols import StreamImporter
from detail.core.registries import register_stream_importer

SENTINEL = -987.987  # MED-PC's "empty array slot" marker
_SESSION_START_RE = re.compile(r"^Start Date:")
# A time this far past H is not rounding; MED-PC prints three decimals.
_H_TOLERANCE_S = 1e-3

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


def split_sessions(text: str) -> list[str]:
    """A MED-PC data file's text split into one string per session (``Start Date:`` block).

    Anything before the first ``Start Date:`` (MED-PC's ``File:`` line) belongs to no session
    and is dropped. A text with no ``Start Date:`` at all is one session.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _SESSION_START_RE.match(line)]
    if not starts:
        return [text]
    bounds = [*starts, len(lines)]
    return ["\n".join(lines[a:b]) for a, b in zip(bounds, bounds[1:])]


def describe_session(number: int, header: dict[str, str]) -> str:
    """One session as a user tells it apart: ``2: Subject RAT2, box 2, 02/16/26 14:00:38``."""
    parts = [f"Subject {header.get('Subject') or '?'}"]
    if header.get("Box"):
        parts.append(f"box {header['Box']}")
    when = " ".join(
        v for v in (header.get("Start Date"), header.get("Start Time")) if v
    ).strip()
    if when:
        parts.append(when)
    return f"{number}: " + ", ".join(parts)


def list_sessions(path: Path) -> list[dict]:
    """Every session in a MED-PC data file, as ``{"session", "subject", "box", "label"}``.

    ``session`` counts from 1, in file order; ``label`` is :func:`describe_session`'s. What a
    picker lists and what a refusal names, so both describe a session the same way.
    """
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    out = []
    for number, block in enumerate(split_sessions(text), start=1):
        header, _scalars, _arrays = parse_medpc_data(block)
        out.append(
            {
                "session": number,
                "subject": header.get("Subject"),
                "box": header.get("Box"),
                "label": describe_session(number, header),
            }
        )
    return out


def select_session(headers: list[dict[str, str]], config: MedpcSessionConfig) -> int:
    """The 0-based index of the one session ``config`` picks out of ``headers``.

    Raises:
        ValueError: If no session matches, or several do (including a several-session file
            with no selector at all). The message lists every session in the file.
    """
    candidates = list(range(len(headers)))
    if config.session is not None:
        candidates = [i for i in candidates if i == config.session - 1]
    if config.subject is not None:
        candidates = [
            i for i in candidates if headers[i].get("Subject") == config.subject
        ]
    if config.box is not None:
        candidates = [i for i in candidates if headers[i].get("Box") == config.box]
    if len(candidates) == 1:
        return candidates[0]
    listing = "; ".join(describe_session(i + 1, h) for i, h in enumerate(headers))
    asked = {k: v for k, v in config.model_dump().items() if v is not None}
    if not candidates:
        raise ValueError(
            f"no session in this MED-PC file matches {asked}; the file holds: {listing}"
        )
    if not asked:
        raise ValueError(
            f"this MED-PC file holds {len(headers)} sessions ({listing}). A recording is one "
            "session, so choose one by subject, box or session number; they are never merged"
        )
    raise ValueError(
        f"{len(candidates)} sessions in this MED-PC file match {asked}; choose one by session "
        f"number too. The file holds: {listing}"
    )


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
    config_model = MedpcSessionConfig

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
        cfg = MedpcSessionConfig.model_validate(config or {})
        blocks = split_sessions(
            Path(data_path).read_text(encoding="utf-8-sig", errors="replace")
        )
        parsed = [parse_medpc_data(block) for block in blocks]
        chosen = select_session([header for header, _s, _a in parsed], cfg)
        header, scalars, arrays = parsed[chosen]
        event_map = parse_event_map(
            Path(program_path).read_text(encoding="utf-8-sig", errors="replace")
        )
        codes = arrays.get("E")
        times = arrays.get("T")
        if codes is None or times is None:
            raise ValueError(
                "MED-PC data file has no E (events) and T (times) arrays — got "
                f"arrays {sorted(arrays)}. This importer reads one program layout: event "
                "codes in E, their times in seconds in T"
            )
        if len(codes) != len(times):
            raise ValueError(
                f"MED-PC E and T arrays differ in length ({len(codes)} vs "
                f"{len(times)}) — file may be truncated"
            )
        pairs = [
            (index, code, time)
            for index, (code, time) in enumerate(zip(codes, times))
            if not (_is_sentinel(code) or _is_sentinel(time))
        ]
        pairs, dropped = _drop_unused_tail(pairs, event_map)
        _check_times_within_session(pairs, scalars.get("H"))
        rows: list[tuple] = []
        for index, code, time in pairs:
            code_int = int(round(code))
            if code_int == 0 and 0 not in event_map:
                raise ValueError(
                    f"MED-PC E({index}) is event code 0 at {time:g}s, between recorded "
                    "events, and the program's event map has no code 0. An unused slot "
                    "would come after the last event; refusing rather than inventing an "
                    "'event_0'"
                )
            t_ns = utils.seconds_to_t_ns(time, what=f"MED-PC T({index})")
            label = event_map.get(code_int, f"event_{code_int}")
            rows.append((t_ns, code_int, label))
        metadata = {meta: header[key] for key, meta in _META_KEYS if key in header}
        metadata["event_map"] = {str(code): label for code, label in event_map.items()}
        if len(blocks) > 1:
            metadata["session"] = chosen + 1
            metadata["sessions_in_file"] = len(blocks)
        if dropped:
            metadata["notes"] = [
                f"dropped {dropped} unused (code 0, time 0) slot(s) after the last event"
            ]
        return StreamImportResult(
            type="events",
            columns=["t_ns", "code", "label"],
            rows=rows,
            metadata=metadata,
        )


def _drop_unused_tail(
    pairs: list[tuple[int, float, float]], event_map: dict[int, str]
) -> tuple[list[tuple[int, float, float]], int]:
    """``pairs`` without its trailing run of ``(code 0, time 0)`` slots, and how many went.

    Such a run is a program's unused, unsealed array space: time cannot go back to 0 after a
    real event. It is kept only when nothing before it moved time forward **and** the program
    names code 0, where a lone ``(0, 0)`` could be a real event at the session's start.
    """
    end = len(pairs)
    while end and pairs[end - 1][1] == 0 and pairs[end - 1][2] == 0:
        end -= 1
    dropped = len(pairs) - end
    if not dropped:
        return pairs, 0
    time_moved = any(time > 0 for _i, _c, time in pairs[:end])
    if not time_moved and 0 in event_map:
        return pairs, 0
    return pairs[:end], dropped


def _check_times_within_session(
    pairs: list[tuple[int, float, float]], session_s: float | None
) -> None:
    """Refuse a ``T`` past the session length ``H`` — the sign of ``T`` logged in ticks.

    This importer reads ``T`` as seconds. A program logging 10 ms ticks imports with every
    time 100x too large and nothing else looks wrong, so the one cross-check the file offers
    (the last ``T`` equals ``H`` in the supported layout) is applied. No ``H``, no check.

    Raises:
        ValueError: If any time exceeds ``H``.
    """
    if session_s is None or not pairs:
        return
    index, _code, latest = max(pairs, key=lambda p: p[2])
    if latest > session_s + _H_TOLERANCE_S:
        raise ValueError(
            f"MED-PC T({index}) is {latest:g}, past the session length H = {session_s:g}s. "
            "This importer reads T as seconds; a program that logs time in ticks (e.g. "
            "10 ms) is not a layout it supports"
        )


def _is_sentinel(value: float) -> bool:
    return abs(value - SENTINEL) < 1e-3
