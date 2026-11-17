"""A run's activity log — what happened, when, kept rather than overwritten.

**What a bar cannot say.** Every surface in this package renders progress into a label that the
next tick replaces, so after two hours the screen shows the same sentence whether the run is
working or wedged. A user watching a long detection had no way to tell, and ran a second
tool beside the app purely to get timestamped lines out of it (#341).

A log answers it for free: the last row carries a clock, and there is nothing after it. That is
the whole property. The columns below are the same numbers the bar already had.

**Why it lives here and not in the GUI.** Three reasons, and each one rules the GUI out on its own:

- the run outlives the window — a detached coordinator keeps going with the app closed, and the
  log has to be there when it is reopened;
- the GUI never sees the event at all (`RunProgressWorker` emits a rendered *string*, so an
  accumulator there would be accumulating the wrong artifact);
- a hung run is a bug report, and `docs/bug_reports.md` asks for exactly this — *"a missing log is
  its own bug"*. #279 was diagnosed from a process table because nobody had one.

**Why CSV rather than prose.** The timing summary (#341, slice 3) is computed from these rows:
a per-recording wall-clock is `ts` grouped by `unit`, which nothing in the tree records today. One
artifact at two resolutions, so the rows need columns. `AppendLog` gives durability per row and
survives a torn tail.

**Its cadence is its own, and deliberately not the bar's.** ``PROGRESS_INTERVAL_S`` is 0.5s — a bar
frame. Logging at that rate is 72,000 fsynced rows for a ten-hour run, for information nobody reads
at that resolution. :data:`ACTIVITY_INTERVAL_S` is the log's own, with a forced row at every phase
and unit boundary so the *interesting* lines are never the ones the throttle ate.

**It can never fail a run.** `append` is `best_effort`, like the two progress writers it sits
beside: publishing is a statement *about* the work, and a raised write reaches a runner's isolation
boundary where it is recorded as the work itself failing (#289). A lost line costs a line.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from detail.core import progress as core_progress
from detail.storage.append_log import AppendLog

#: How often a row lands, in seconds. **Not** ``PROGRESS_INTERVAL_S`` (0.5s, a bar frame): at that
#: rate a ten-hour run writes 72,000 fsynced rows to say what a few hundred would. At 5s it is
#: ~7,200 rows and a few hundred KB, and a phase or unit boundary is written whatever the clock
#: says — so the rows that mark a change are never the ones the throttle dropped.
ACTIVITY_INTERVAL_S = 5.0

ACTIVITY_FILENAME = "activity.csv"

#: One row per sample. Every column is a number the bar already had, plus the clock it threw away.
#:
#: ``ts`` is a wall clock, not an elapsed: "nothing since 14:02" is the question being asked, and
#: an elapsed cannot answer it without knowing when you are reading. ``unit`` is the recording a
#: processing run is on, empty for a single-job run, which has one. ``eta_unit_s`` is that unit's
#: own remaining and ``eta_run_s`` the whole run's — two denominators, never conflated: recordings
#: differ by hours, so the run-wide rate says nothing about whether *this* one is stuck.
COLUMNS = (
    "ts",
    "elapsed_s",
    "unit",
    "unit_index",
    "unit_total",
    "phase",
    "done",
    "total",
    "eta_unit_s",
    "eta_run_s",
)


def activity_path(run_dir: Path | str) -> Path:
    """Where a run's log lives, beside its ``job.json`` / ``progress.json``."""
    return Path(run_dir) / ACTIVITY_FILENAME


def _now_iso() -> str:
    # Milliseconds: a unit is timed from its rows' clocks, and at whole seconds a 4 s
    # recording read anything from 3 to 5 s (#32).
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class ActivityLog:
    """The append-only log for one run directory.

    Construct one per run and call :meth:`sample` with each progress emission; it decides whether
    the row is due. A writer is shared by the recording threads a run orchestrator fans out over
    (``api/experiment.py`` runs them in a ``ThreadPoolExecutor``), so appends are serialised here:
    `AppendLog.append` opens the file per row and takes no lock of its own, and two threads writing
    at once is a torn row rather than two good ones.
    """

    def __init__(self, run_dir: Path | str, *, interval_s: float | None = None) -> None:
        self._log = AppendLog(activity_path(run_dir), list(COLUMNS))
        self._lock = threading.Lock()
        #: Serialises `sample` / `finish`, which read and replace the held row below.
        self._sample_lock = threading.Lock()
        self._throttle = core_progress.ThrottledProgress(
            self._forward,
            interval_s=ACTIVITY_INTERVAL_S if interval_s is None else interval_s,
        )
        #: The last (unit, phase) offered. A change in either forces a row past the throttle — a
        #: boundary is the one sample whose *timing* is the information, so dropping it to a
        #: cadence would lose the thing the log is for.
        self._last_mark: tuple[str, str] | None = None
        #: The last sample offered and **not** written: the throttle drops, it does not hold
        #: back, so without this the old unit's last sample was gone by the time the boundary
        #: was seen, and the unit was timed to whichever earlier row the clock happened to keep
        #: (#32). Written before the next boundary's row, and by `finish`.
        self._held: dict | None = None
        #: The last sample offered at all, written or not; `finish` stamps the run's end from it.
        self._last_row: dict | None = None

    def sample(self, row: dict, *, force: bool = False) -> None:
        """Offer one progress sample. Written if it is due, if a boundary moved, or if forced.

        At a boundary the previous sample is written first, if the throttle dropped it, so the
        old unit or phase ends at its own last sample rather than at an earlier row.
        """
        mark = (str(row.get("unit") or ""), str(row.get("phase") or ""))
        with self._sample_lock:
            # Read before the assignment. Written the other way round, `self._last_mark is None`
            # is tested against the value just stored and is dead — the first row still lands,
            # because `ThrottledProgress` forces its own first call, which is exactly what would
            # have made the dead branch look like it worked.
            boundary = self._last_mark is not None and mark != self._last_mark
            self._last_mark = mark
            if boundary and self._held is not None:
                self._throttle(self._held, force=True)
            self._held, self._last_row = row, row
            self._throttle(row, force=force or boundary)

    def finish(self, *, elapsed_s: float | None = None) -> None:
        """The run's end: the held last sample, then a row stamped now in the last unit and phase.

        Every writer calls this at its terminal status. Nothing did, so the log ended at the
        last row the throttle kept — seconds before the run did — and the report's last unit and
        the run's wall clock came out short (#32). The end row repeats the last sample's unit,
        phase and counts, so it extends that unit to the real end rather than adding a unit of
        its own; its ETAs are blank, since nothing remains to estimate. A log that was never
        sampled writes nothing.

        ``elapsed_s`` is the writer's own clock, the one its rows were stamped with. A writer
        without one passes nothing, and the end is the last sample's ``elapsed_s`` plus the
        wall time since it.
        """
        with self._sample_lock:
            if self._held is not None:
                self._throttle(self._held, force=True)
            if self._last_row is None:
                return
            now = _now_iso()
            if elapsed_s is None:
                elapsed_s = _elapsed_since(self._last_row, now)
            end = {
                **self._last_row,
                "ts": now,
                "elapsed_s": round(float(elapsed_s), 1),
                "eta_unit_s": "",
                "eta_run_s": "",
            }
            self._held = None
            self._throttle(end, force=True)

    def _forward(self, row: dict) -> None:
        """What the throttle passes on: written, and no longer held."""
        if row is self._held:
            self._held = None
        self._write(row)

    @core_progress.best_effort("run activity")
    def _write(self, row: dict) -> None:
        with self._lock:
            self._log.append([row.get(name, "") for name in COLUMNS])

    def rows(self) -> list[dict]:
        """Every row written so far. Missing file reads as empty, a torn tail is dropped."""
        return self._log.rows()


def _elapsed_since(row: dict, now_iso: str) -> float:
    """``row``'s ``elapsed_s`` carried forward to ``now_iso`` by the wall clock."""
    try:
        base = float(row.get("elapsed_s") or 0.0)
    except (TypeError, ValueError):
        base = 0.0
    then, now = _parse_ts(row.get("ts", "")), _parse_ts(now_iso)
    if then is None or now is None:
        return base
    return base + max((now - then).total_seconds(), 0.0)


def read(run_dir: Path | str) -> list[dict]:
    """The rows of a run's log, for a reader that has a path and no writer."""
    return AppendLog(activity_path(run_dir), list(COLUMNS)).rows()


def make_row(
    *,
    elapsed_s: float,
    unit: str = "",
    unit_index: int = 0,
    unit_total: int = 0,
    phase: str = "",
    done: int = 0,
    total: int = 0,
    eta_unit_s: float | None = None,
    eta_run_s: float | None = None,
) -> dict:
    """One row, with the clock stamped here so every caller stamps it the same way."""
    return {
        "ts": _now_iso(),
        "elapsed_s": round(float(elapsed_s), 1),
        "unit": unit,
        "unit_index": unit_index,
        "unit_total": unit_total,
        "phase": phase,
        "done": done,
        "total": total,
        "eta_unit_s": "" if eta_unit_s is None else round(float(eta_unit_s), 1),
        "eta_run_s": "" if eta_run_s is None else round(float(eta_run_s), 1),
    }


def _parse_ts(value: str) -> "datetime | None":
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def timings(rows: list[dict]) -> list[dict]:
    """Per-unit wall-clock, frames and rate, from the rows a run wrote.

    **Derived rather than recorded**, and that is the point: nothing in the tree stores when a
    recording started or finished (a run record has `created_at` and no finish time), so a
    performance report either invents a second bookkeeping path or reads the one the log already
    keeps. `ts` grouped by `unit` is the answer, and it costs nothing extra at run time.

    A unit's wall-clock is first row to last *within that unit*, so a run that interleaves
    recordings across threads still attributes each one its own span. The gap between one unit's
    last row and the next unit's first is not charged to either — it is setup, and silently
    folding it into a recording would make whichever ran first look slow.

    ``frames`` is the unit's own ``total`` where it reported one, which is the denominator its
    rate should use; ``fps`` is blank when either half is missing rather than zero, because a
    zero here reads as "ran and produced nothing".
    """
    by_unit: dict[str, list[dict]] = {}
    for row in rows:
        by_unit.setdefault(str(row.get("unit") or ""), []).append(row)

    out = []
    for unit, unit_rows in by_unit.items():
        stamps = [t for t in (_parse_ts(r.get("ts", "")) for r in unit_rows) if t]
        seconds = (stamps[-1] - stamps[0]).total_seconds() if len(stamps) > 1 else 0.0
        frames = 0
        for row in unit_rows:
            try:
                frames = max(frames, int(row.get("total") or 0))
            except (TypeError, ValueError):
                pass
        out.append(
            {
                "unit": unit,
                "started": stamps[0].isoformat(timespec="seconds") if stamps else "",
                "seconds": round(seconds, 1),
                "frames": frames,
                "fps": round(frames / seconds, 1) if frames and seconds > 0 else "",
            }
        )
    return out


#: An epoch counter, which training writes as its phase message (``epoch 3/100``). Every epoch is
#: a different string, so left alone a hundred-epoch run would report a hundred one-row phases.
_EPOCH = re.compile(r"^epoch \d+/\d+")


def phase_name(message: str) -> str:
    """The phase a logged message belongs to, with the per-row detail stripped.

    An analysis writes ``"clustering · 3 of 7"``, training writes ``"epoch 3/100"``, and detection
    and embedding write nothing at all once their model is loaded — so blank is the work itself.
    """
    name = str(message or "").split(" · ")[0].strip()
    if not name:
        return "processing"
    if _EPOCH.match(name):
        return "training epochs"
    return name


def phase_timings(rows: list[dict]) -> list[dict]:
    """Wall-clock per ``(unit, phase)``, in the order each phase first appeared.

    **A phase runs until the next phase starts**, not until its own last row. The log is throttled
    to one row per :data:`ACTIVITY_INTERVAL_S` but writes a row at every phase change, so a phase's
    own last row can sit up to five seconds before it really ended; measuring to the next phase's
    first row charges that time to the phase that was running. The last phase of a unit ends at
    the unit's last row. Units are kept apart, because a run that works on several recordings at
    once interleaves their rows.

    A phase that recurs within a unit (``loading`` → ``detecting`` → ``loading``) is summed into
    one entry, since the question asked of this table is "how long did loading take in total".

    **Blank rows before a unit's first named phase are ``starting``**, not the work: a processing
    run's coordinator writes a row as each recording begins, before the worker has said anything,
    and calling that ``processing`` would put the work before ``loading model``.
    """
    by_unit: dict[str, list[tuple[datetime, str]]] = {}
    for row in rows:
        stamp = _parse_ts(row.get("ts", ""))
        if stamp is None:
            continue
        unit = str(row.get("unit") or "")
        by_unit.setdefault(unit, []).append((stamp, str(row.get("phase") or "")))

    out: dict[tuple[str, str], float] = {}
    for unit, raw in by_unit.items():
        first_named = next((i for i, (_, m) in enumerate(raw) if m.strip()), 0)
        samples = [
            (stamp, "starting" if i < first_named else phase_name(message))
            for i, (stamp, message) in enumerate(raw)
        ]
        for i, (stamp, phase) in enumerate(samples):
            end = samples[i + 1][0] if i + 1 < len(samples) else stamp
            key = (unit, phase)
            out[key] = out.get(key, 0.0) + (end - stamp).total_seconds()
    return [
        {"unit": unit, "phase": phase, "seconds": round(seconds, 1)}
        for (unit, phase), seconds in out.items()
    ]


def run_span(rows: list[dict]) -> tuple[str, str, float]:
    """``(started, finished, seconds)`` for the whole run, from its first and last row.

    The run's own elapsed is not usable for this: `_finalize` writes a terminal status and emits
    no progress, so the last `elapsed_s` ever persisted is from the final recording boundary, not
    the end of the run. The log's last row is the last thing that actually happened.
    """
    stamps = [t for t in (_parse_ts(r.get("ts", "")) for r in rows) if t]
    if not stamps:
        return ("", "", 0.0)
    return (
        stamps[0].isoformat(timespec="seconds"),
        stamps[-1].isoformat(timespec="seconds"),
        round((stamps[-1] - stamps[0]).total_seconds(), 1),
    )
