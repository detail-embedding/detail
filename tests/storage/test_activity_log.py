"""A run's activity log: kept, timestamped, cheap, and unable to fail the run.

**The class.** Every progress surface in this package renders into a label the next tick replaces,
so a run that is wedged and a run that is working look identical after two hours. The information
to tell them apart was being produced and discarded; what a log adds is a clock on the last row and
nothing after it.

The failures a log of this kind has are not about content, and none of them is visible to a test
that only checks a row was written:

- **logging at the bar's cadence.** `PROGRESS_INTERVAL_S` is 0.5s. A ten-hour run at that rate is
  72,000 fsynced rows, and the code reads fine — the rows are all correct. Only counting them says
  otherwise.
- **losing the boundary.** A throttle drops whichever samples fall between its ticks, and the
  sample where the phase or the recording *changed* is the one whose timing is the information.
- **failing the run.** A raised write inside a runner's isolation boundary is recorded as the
  *work* failing (#289) — a log that cannot be written must cost a line, never a recording.
- **tearing.** The orchestrator fans recordings out over a thread pool and they share one writer.

**A span measured from throttled samples whose endpoints were dropped (#32).** A unit is timed
from its first logged row to its last, and the throttle drops rather than holds, so the old
unit's last sample was gone by the time the boundary forced the new unit's first row, and the
run's end was never logged at all: a 4 s recording read 0.0 s. The rule: the log holds the last
dropped sample and writes it at the next boundary, every writer calls ``finish`` at its
terminal status (a source scan below), and ``ts`` carries milliseconds.
"""

from __future__ import annotations

import logging
import threading

import pytest

from detail.core import progress as core_progress
from detail.storage import activity


def _row(**kw):
    kw.setdefault("elapsed_s", 1.0)
    return activity.make_row(**kw)


def test_a_row_carries_a_wall_clock(tmp_path):
    """An elapsed cannot answer "has anything happened since I looked away"; a clock can."""
    log = activity.ActivityLog(tmp_path)
    log.sample(_row(unit="recording_001", phase="detecting"))
    (row,) = log.rows()
    assert row["ts"], "no timestamp — the one column a bar does not already have"
    assert row["unit"] == "recording_001"
    assert row["phase"] == "detecting"


def test_the_log_is_cheaper_than_the_bar(tmp_path, monkeypatch):
    """The OQ-RUN-3 correction, as a number.

    The notes recommended logging at `PROGRESS_INTERVAL_S`, which is 0.5s — a bar frame. Every row
    written that way is correct, so nothing about the code looks wrong; the only symptom is 72,000
    fsynced rows for a ten-hour run. Wire this to the bar's cadence and this test fails.
    """
    # Simulated time, because a tight loop takes milliseconds and *any* interval collapses it to
    # one row — which is how the first version of this test looked right and proved nothing.
    clock = {"t": 0.0}
    monkeypatch.setattr(core_progress.time, "monotonic", lambda: clock["t"])

    log = activity.ActivityLog(tmp_path, interval_s=activity.ACTIVITY_INTERVAL_S)
    ticks = int(600 / core_progress.PROGRESS_INTERVAL_S)  # ten minutes of bar frames
    for i in range(ticks):
        clock["t"] += core_progress.PROGRESS_INTERVAL_S
        log.sample(_row(unit="recording_001", phase="detecting", done=i, total=ticks))

    written = len(log.rows())
    assert written < ticks / 5, (
        f"{written} rows where the bar would have written {ticks} — the log is running at the "
        "bar's cadence rather than its own"
    )


def test_a_phase_boundary_is_never_throttled_away(tmp_path):
    """The sample where the phase changed is the one whose timing is the information.

    Asserted as the *sequence*, not as a count: a log that wrote every row would also pass a
    "the phases are present" check, and a log that wrote one row per phase but in the wrong order
    would pass a set comparison.
    """
    log = activity.ActivityLog(
        tmp_path, interval_s=1000.0
    )  # nothing lands on the clock
    for phase in ("loading", "detecting", "writing"):
        for i in range(50):
            log.sample(_row(unit="recording_001", phase=phase, done=i))
    rows = log.rows()
    assert _runs([r["phase"] for r in rows]) == ["loading", "detecting", "writing"]
    # Each finished phase is written at its first sample and at its last (#32), and nothing
    # between: the clock wrote none of the other 48.
    assert [(r["phase"], r["done"]) for r in rows] == [
        ("loading", "0"), ("loading", "49"), ("detecting", "0"), ("detecting", "49"),
        ("writing", "0"),
    ]  # fmt: skip


def test_a_unit_boundary_is_never_throttled_away(tmp_path):
    """The other boundary, and the one a long run is actually watched for."""
    log = activity.ActivityLog(tmp_path, interval_s=1000.0)
    for i in range(1, 4):
        for _ in range(50):
            log.sample(_row(unit=f"recording_{i:03d}", phase="detecting", unit_index=i))
    assert _runs([r["unit"] for r in log.rows()]) == [
        "recording_001",
        "recording_002",
        "recording_003",
    ]
    assert len(log.rows()) == 5, "each finished unit's first and last sample, no more"


def _runs(values: list) -> list:
    """``values`` with each run of equal neighbours kept once."""
    return [v for i, v in enumerate(values) if i == 0 or v != values[i - 1]]


def test_the_same_phase_on_a_new_unit_still_writes(tmp_path):
    """The pair. Keying the boundary on the phase alone loses every recording change in a
    single-phase stage — which is every processing run, where the phase is the stage."""
    log = activity.ActivityLog(tmp_path, interval_s=1000.0)
    log.sample(_row(unit="recording_001", phase="detecting"))
    log.sample(_row(unit="recording_002", phase="detecting"))
    assert [r["unit"] for r in log.rows()] == ["recording_001", "recording_002"]


def test_a_log_that_cannot_be_written_does_not_fail_the_run(tmp_path, caplog):
    """#289's rule, one writer over. Publishing is a statement *about* the work.

    Forced through the real failure rather than a stubbed method: `best_effort` is what makes this
    true, so the test has to reach the thing it wraps.
    """

    def boom(values):
        raise PermissionError("locked")

    log = activity.ActivityLog(tmp_path)
    log._log.append = boom
    with caplog.at_level(logging.WARNING):
        log.sample(_row(unit="recording_001", phase="detecting"))  # must not raise
    assert "run activity" in caplog.text, (
        "the failure was swallowed silently — a log that can never be written leaves a user "
        "watching an empty band with nothing saying why"
    )


def test_best_effort_is_actually_applied(tmp_path):
    """Guard the guard above: with the decorator removed, the write raises and the test that
    asserts it does not would be the only thing standing between a lost line and a failed run.
    """
    assert hasattr(
        activity.ActivityLog._write, "unpublished"
    ), "the writer is not wrapped in best_effort, so a log error can fail a run"


def test_concurrent_threads_write_whole_rows(tmp_path):
    """The orchestrator fans recordings out over a `ThreadPoolExecutor` and they share one writer.

    `AppendLog.append` opens the file per row and takes no lock of its own, so two threads writing
    at once is one torn row rather than two good ones — and a torn tail is silently dropped on
    read, which is how this would go unnoticed.
    """
    log = activity.ActivityLog(tmp_path, interval_s=0.0)
    n = 40

    def write(i: int) -> None:
        log.sample(_row(unit=f"recording_{i:03d}", phase="detecting", done=i, total=n))

    threads = [threading.Thread(target=write, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = log.rows()
    assert len(rows) == n, f"{len(rows)} of {n} rows survived — some interleaved"
    assert {r["unit"] for r in rows} == {f"recording_{i:03d}" for i in range(n)}


def test_a_reader_needs_only_the_path(tmp_path):
    """The GUI reads a run it did not launch, and the CLI reads one from another process."""
    log = activity.ActivityLog(tmp_path)
    log.sample(_row(unit="recording_001", phase="detecting"))
    assert [r["unit"] for r in activity.read(tmp_path)] == ["recording_001"]
    assert activity.read(tmp_path / "nonexistent") == []


def test_two_etas_are_kept_apart(tmp_path):
    """Recordings differ by hours, so the run-wide rate says nothing about whether *this* one is
    stuck — which is the question being asked. Two columns, never one."""
    row = _row(
        unit="recording_013", phase="detecting", eta_unit_s=840.0, eta_run_s=9060.0
    )
    assert row["eta_unit_s"] == 840.0
    assert row["eta_run_s"] == 9060.0
    assert row["eta_unit_s"] != row["eta_run_s"]


def test_an_absent_eta_is_blank_not_zero(tmp_path):
    """Before a run has made progress there is no ETA, and `remaining_seconds` returns `None`.
    Writing that as `0` reads as "no time left" — the opposite of what it means."""
    row = _row(unit="recording_001", phase="loading")
    assert row["eta_unit_s"] == ""
    assert row["eta_run_s"] == ""


def test_the_log_cadence_is_not_the_bar_cadence():
    """Stated as a relationship rather than a number, so tuning one does not silently make them
    equal — which is the state this whole file exists to prevent."""
    assert activity.ACTIVITY_INTERVAL_S > core_progress.PROGRESS_INTERVAL_S


# --------------------------------------------------------------------------- #
# per-phase wall clock
# --------------------------------------------------------------------------- #
def _rows(samples):
    """Rows as the log writes them, from ``(seconds, unit, phase)``."""
    import datetime as dt

    base = dt.datetime(2026, 10, 20, 9, 0, tzinfo=dt.timezone.utc)
    return [
        {
            "ts": (base + dt.timedelta(seconds=s)).isoformat(timespec="seconds"),
            "unit": unit,
            "phase": phase,
        }
        for s, unit, phase in samples
    ]


def _by_phase(rows):
    return {(t["unit"], t["phase"]): t["seconds"] for t in activity.phase_timings(rows)}


def test_a_phase_runs_until_the_next_one_starts():
    """The throttle can leave a phase's own last row seconds before it ended; charging only
    to that row would undercount every phase by up to the log's interval."""
    got = _by_phase(
        _rows(
            [
                (0, "", "loading model"),
                (4, "", "loading model"),  # last throttled row, 26s before detection
                (30, "", ""),
                (90, "", ""),
            ]
        )
    )
    assert got == {("", "loading model"): 30.0, ("", "processing"): 60.0}


def test_blank_rows_before_the_first_phase_are_starting_not_the_work():
    """A processing coordinator writes a row as a recording begins, before the worker says
    anything; seen on a real MPS detection, where it put "processing" ahead of "loading model".
    A stage that never names a phase is all work, though, so blank alone is not "starting".
    """
    got = _by_phase(
        _rows([(0, "r", ""), (2, "r", "loading model"), (6, "r", ""), (16, "r", "")])
    )
    assert got == {
        ("r", "starting"): 2.0,
        ("r", "loading model"): 4.0,
        ("r", "processing"): 10.0,
    }
    assert _by_phase(_rows([(0, "", ""), (9, "", "")])) == {("", "processing"): 9.0}


def test_interleaved_units_are_timed_apart():
    """Two recordings at once write interleaved rows; reading them as one stream would charge
    each the other's time."""
    got = _by_phase(
        _rows(
            [
                (0, "a", "detecting"),
                (1, "b", "detecting"),
                (10, "a", "detecting"),
                (20, "b", "detecting"),
            ]
        )
    )
    assert got == {("a", "detecting"): 10.0, ("b", "detecting"): 19.0}


def test_epoch_counters_and_notes_collapse_into_one_phase():
    """``epoch 3/100`` changes every epoch and ``clustering · 3 of 7`` every step; left alone a
    hundred-epoch run reads as a hundred one-row phases."""
    assert activity.phase_name("epoch 3/100") == "training epochs"
    assert activity.phase_name("epoch 100/100 · loss 0.31") == "training epochs"
    assert activity.phase_name("clustering · 3 of 7") == "clustering"
    assert activity.phase_name("") == "processing"
    assert activity.phase_name("loading model") == "loading model"


# --------------------------------------------------------------------------- #
# a unit's span is first sample to *last* sample, and the run ends at its end (#32)
# --------------------------------------------------------------------------- #
def _scripted(monkeypatch):
    """A log on a simulated clock: ``at(seconds)`` sets both clocks a row is stamped from (the
    wall ``ts`` and the throttle's monotonic), so a test controls which rows the throttle drops.
    """
    import datetime as dt

    base = dt.datetime(2026, 10, 20, 9, 0, tzinfo=dt.timezone.utc)
    clock = {"t": 0.0}
    monkeypatch.setattr(core_progress.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(
        activity,
        "_now_iso",
        lambda: (base + dt.timedelta(seconds=clock["t"])).isoformat(
            timespec="milliseconds"
        ),
    )

    def at(seconds: float) -> None:
        clock["t"] = seconds

    return at


def _unit(log, at, unit, start, end, step=0.5, frames=25):
    t, i = start, 0
    while t <= end + 1e-9:
        at(t)
        log.sample(_row(elapsed_s=t, unit=unit, phase="", done=i, total=frames))
        t, i = round(t + step, 3), i + 1


def test_each_unit_is_timed_to_its_last_sample_not_its_last_written_row(
    tmp_path, monkeypatch
):
    """The throttle drops samples; a unit change forced the *new* unit's first row through,
    but the old unit's last sample had already been dropped. So a recording that took 4 s
    read 0.0 s (one row) and fps came out blank or several times too high (#32, finding 2).
    The held last sample is written before the next unit's first."""
    at = _scripted(monkeypatch)
    log = activity.ActivityLog(tmp_path, interval_s=activity.ACTIVITY_INTERVAL_S)
    _unit(log, at, "recording_000", 0.0, 4.0)
    _unit(log, at, "recording_001", 4.5, 8.5)
    at(9.0)
    log.finish(elapsed_s=9.0)
    got = {u["unit"]: u for u in activity.timings(log.rows())}
    assert got["recording_000"]["seconds"] == pytest.approx(4.0)
    assert got["recording_001"]["seconds"] == pytest.approx(4.5)
    assert got["recording_000"]["fps"] == pytest.approx(25 / 4.0, abs=0.05)


def test_the_last_row_is_the_end_of_the_run(tmp_path, monkeypatch):
    """Nothing was written at the run's terminal status, so the report's run ended at the
    last row the throttle happened to keep: 10.2 s for a run ``progress.json`` said took
    13.5 s. ``finish`` writes the end, stamped when the run ended, in the last unit."""
    at = _scripted(monkeypatch)
    log = activity.ActivityLog(tmp_path, interval_s=activity.ACTIVITY_INTERVAL_S)
    _unit(log, at, "recording_000", 0.0, 3.0)
    at(13.5)
    log.finish(elapsed_s=13.5)
    rows = log.rows()
    assert rows[-1]["elapsed_s"] == "13.5"
    assert rows[-1]["unit"] == "recording_000"
    started, finished, seconds = activity.run_span(rows)
    assert seconds == pytest.approx(13.5)
    assert [u["unit"] for u in activity.timings(rows)] == ["recording_000"]


def test_finish_on_a_log_that_never_sampled_writes_nothing(tmp_path):
    """No sample means no unit to end; an invented row would read as a unit of its own."""
    log = activity.ActivityLog(tmp_path)
    log.finish(elapsed_s=3.0)
    assert log.rows() == []


def test_the_clock_is_finer_than_a_second():
    """A unit is timed from its rows' ``ts``; at whole seconds a 4 s recording read anything
    from 3 to 5 s, a quarter of its rate either way."""
    import datetime as dt

    stamp = dt.datetime.fromisoformat(activity._now_iso())
    assert "." in activity._now_iso() and stamp.tzinfo is not None


def _writers_without_finish(sources: dict[str, str]) -> list[str]:
    """Every source that builds an ``ActivityLog`` and never calls ``finish`` on anything."""
    return sorted(
        name
        for name, text in sources.items()
        if "ActivityLog(" in text and ".finish(" not in text
    )


def test_every_writer_ends_its_log():
    """A writer that never calls ``finish`` ends its log at the last row the throttle kept,
    seconds short of the run (#32). Swept over the package, so a fourth writer is covered the
    day it is written."""
    from pathlib import Path

    root = Path("detail")
    sources = {
        p.as_posix(): p.read_text(encoding="utf-8-sig")
        for p in root.rglob("*.py")
        if p.as_posix() != "detail/storage/activity.py"
    }
    writers = [n for n, t in sources.items() if "ActivityLog(" in t]
    assert (
        len(writers) >= 3
    ), f"the scan found too few writers to mean anything: {writers}"
    assert _writers_without_finish(sources) == []


def test_the_finish_scan_would_catch_a_writer_that_never_ends():
    assert _writers_without_finish(
        {"a.py": "log = activity.ActivityLog(d)\nlog.sample(r)\n", "b.py": "x = 1\n"}
    ) == ["a.py"]
