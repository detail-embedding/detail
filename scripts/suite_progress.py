"""Live progress for a pytest run, where a person watching Claude Code can see it — app or terminal.

A foreground command's output reaches Claude Code only when the command ends, so a four-minute
suite looks the same as a hang. ``suite_progress_plugin`` (on every run through ``addopts``) keeps
one small JSON file per run up to date; this module draws those files for both places a person
watches from:

- ``watch`` — for the **Claude app** (and the terminal). It follows one run and prints a
  pytest-style line at every 10% (at most one per 15 seconds, so a short run posts only its
  result) and each failure as it happens. Run under the Monitor tool, each
  line reaches Claude as it is printed, and Claude posts it as a message: the only surface the
  app shows while a command runs. It exits when the run finishes, so the Monitor ends with it::

      detail-progress  pytest [███████████░░░░░░░░░]  56% 4,310/7,696  ✗1  ETA 1m48s

- ``statusline`` — for the **terminal**. The same line for each active run on the machine (other
  sessions' gates included), for Claude Code's ``statusLine`` setting, which re-runs it every
  second (``refreshInterval``, ``.claude/settings.json``). The app does not draw a status line.

The ``run-tests`` skill is the procedure.

**Why the directory is fixed, not under ``TMPDIR``.** Sessions here give each run a private
``TMPDIR`` and ``HOME``, the same reason ``with_suite_lock.py`` gives: a progress file under either
would be one neither reader finds. ``DETAIL_SUITE_PROGRESS`` overrides the directory. It sits in
the same ``machine_dir`` as the lock, for the lock's reasons, Windows and WSL included.

Standard library only, and ``statusline`` never raises: it runs every second in every terminal
session and must cost nothing and show nothing when there is nothing to show.

Usage:
    python scripts/suite_progress.py watch [--since EPOCH] [--step 10]
    python scripts/suite_progress.py statusline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

#: How long a finished run is kept, so its result can still be read on the status line.
KEEP_FINISHED_S = 120
#: An unfinished run older than this is dropped whatever its process says. The only way out for a
#: killed run on Windows, where ``_pid_alive`` cannot ask (see there).
MAX_RUN_S = 2 * 60 * 60
#: The most runs the status line shows at once (other sessions' gates included).
MAX_LINES = 3
BAR_WIDTH = 20


def machine_dir(os_name: str = os.name, environ: Mapping[str, str] = os.environ) -> str:
    """``with_suite_lock.machine_dir``, copied: that module imports ``detail``, and this one may not.

    ``tests/scripts/test_machine_dir.py`` holds the two copies equal.
    """
    if os_name == "nt":
        return environ.get("ProgramData") or r"C:\ProgramData"
    return "/tmp"


def progress_dir() -> Path:
    """One directory per machine, deliberately not under ``TMPDIR`` (see the module docstring)."""
    override = os.environ.get("DETAIL_SUITE_PROGRESS")
    if override:
        return Path(override)
    return Path(machine_dir()) / "detail-suite-progress"


def write_state(state: dict) -> None:
    """Replace a run's file atomically, so a reader never parses a half-written one."""
    directory = progress_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{state['pid']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)


def _pid_alive(pid: int) -> bool:
    """Whether a run's process is still there, POSIX only.

    Not ``executor.local.pid_alive``: the status line runs under whatever ``python3`` is first on
    the path, where ``detail`` may not be installed, so this module imports only the standard
    library. On Windows ``os.kill(pid, 0)`` sends Ctrl-C rather than asking, so it is not used
    there at all; a dead run is dropped by ``MAX_RUN_S`` instead.
    """
    if os.name == "nt":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_runs(now: float | None = None) -> list[dict]:
    """Every run worth showing, oldest first. Files of dead or long-finished runs are removed."""
    now = time.time() if now is None else now
    runs = []
    for path in sorted(progress_dir().glob("*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        finished_at = state.get("finished_at")
        finished_long_ago = (
            finished_at is not None and now - finished_at > KEEP_FINISHED_S
        )
        dead = finished_at is None and (
            now - state.get("started", now) > MAX_RUN_S
            or not _pid_alive(int(state.get("pid", 0)))
        )
        stale = finished_long_ago or dead
        if stale:
            path.unlink(missing_ok=True)
            continue
        runs.append(state)
    return sorted(runs, key=lambda s: s.get("started", 0))


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def eta_seconds(state: dict, now: float) -> float | None:
    """Remaining time at the rate so far. ``None`` until a test has finished."""
    done, total = state.get("done", 0), state.get("total")
    if not done or not total:
        return None
    return (total - done) * (now - state["started"]) / done


def render(state: dict, now: float) -> str:
    """One line for a run: a pytest-style bar while it runs, the result once it has finished."""
    label = state.get("label", "")
    prefix = f"{label}  pytest" if label else "pytest"
    failed = state.get("failed", 0)
    if state.get("finished_at") is not None:
        took = _duration(state["finished_at"] - state["started"])
        passed = f"{state.get('passed', 0):,} passed"
        if failed:
            return f"{prefix} ✗ {failed:,} failed, {passed} in {took}"
        return f"{prefix} ✓ {passed} in {took}"
    total = state.get("total")
    if not total:
        return f"{prefix} collecting… {_duration(now - state['started'])}"
    done = state.get("done", 0)
    fraction = min(1.0, done / total)
    full = int(fraction * BAR_WIDTH)
    bar = "█" * full + "░" * (BAR_WIDTH - full)
    eta = eta_seconds(state, now)
    tail = f"  ETA {_duration(eta)}" if eta is not None else ""
    fails = f"  ✗{failed:,}" if failed else ""
    return f"{prefix} [{bar}] {fraction:4.0%} {done:,}/{total:,}{fails}{tail}"


def statusline() -> None:
    """Print one line per active run; print nothing, and never raise, when there is none."""
    try:
        now = time.time()
        for state in read_runs(now)[-MAX_LINES:]:
            print(render(state, now))
    except (
        Exception
    ):  # noqa: BLE001 — an isolation boundary: a status line must never error
        pass


def milestones_crossed(before: int, after: int, total: int, step: int) -> list[int]:
    """The ``step``-percent marks passed between two ``done`` counts."""
    if not total:
        return []
    return [
        mark
        for mark in range(step, 101, step)
        if before * 100 < mark * total <= after * 100
    ]


def watch(
    root: Path, since: float, step: int, poll_s: float = 1.0, every_s: float = 15.0
) -> int:
    """Print the run's progress at each ``step``% and each new failure; exit when it ends.

    Follows the first run started in ``root`` at or after ``since`` that had not already finished
    when the watch began — a run that ended a moment ago is the last one, not the one being
    started. Exits 0 when it finishes green, 1 when it finishes red or its process dies unfinished.
    """
    root = root.resolve()
    began = time.time()
    waiting_said = False
    printed_done, seen_failures, pid = 0, set(), None
    # Milestones wait at least `every_s` after the watch began and after each other, so a run of
    # a few seconds posts its failures and its result and nothing else. Failures and the result
    # are never held back.
    last_print = began
    while True:
        now = time.time()
        runs = [
            s
            for s in read_runs(now)
            if s.get("started", 0) >= since
            and Path(s.get("root", "")).resolve() == root
        ]
        run = next((s for s in runs if s.get("pid") == pid), None) if pid else None
        fresh = [s for s in runs if (s.get("finished_at") or began + 1) > began]
        if run is None and pid is None and fresh:
            run, pid = fresh[0], fresh[0].get("pid")
        if run is None:
            if pid is not None:
                print(
                    "pytest stopped before it finished (killed, or it crashed)",
                    flush=True,
                )
                return 1
            if not waiting_said:
                print("waiting for pytest to start", flush=True)
                waiting_said = True
            time.sleep(poll_s)
            continue
        for node in run.get("failed_ids", []):
            if node not in seen_failures:
                seen_failures.add(node)
                print(f"  ✗ {node}", flush=True)
        done, total = run.get("done", 0), run.get("total") or 0
        if run.get("finished_at") is not None:
            print(render(run, now), flush=True)
            return 1 if run.get("failed") or run.get("exitstatus") else 0
        crossed = milestones_crossed(printed_done, done, total, step)
        if crossed and now - last_print >= every_s:
            print(render(run, now), flush=True)
            printed_done, last_print = done, now
        time.sleep(poll_s)


def _utf8_stdout() -> None:
    """Print the bar's block characters on any console.

    A Windows console's default encoding (cp1252 and the like) cannot encode ``█`` or ``✓``:
    ``statusline`` would swallow the error and show nothing, and ``watch`` would crash on its
    first line. Both write UTF-8 whatever the console's default.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("statusline", help="lines for Claude Code's status line (terminal)")
    w = sub.add_parser("watch", help="milestone lines for the Monitor tool")
    w.add_argument("--since", type=float, default=None, help="epoch; default: 10 s ago")
    w.add_argument("--step", type=int, default=10, help="percent between lines")
    w.add_argument(
        "--root", type=Path, default=Path.cwd(), help="the checkout pytest runs in"
    )
    w.add_argument("--poll", type=float, default=1.0, help="seconds between reads")
    w.add_argument(
        "--every",
        type=float,
        default=15.0,
        help="fewest seconds between milestone lines",
    )
    args = parser.parse_args(argv)
    if args.command == "statusline":
        statusline()
        return 0
    # A little slack, because the skill starts the run first and the watch right after it.
    since = time.time() - 10 if args.since is None else args.since
    return watch(args.root, since, args.step, args.poll, args.every)


if __name__ == "__main__":
    sys.exit(main())
