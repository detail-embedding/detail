"""A log of every pytest run, and a report of which tests fail in it and how often.

``suite_progress_plugin`` (on every run through ``addopts``) appends one JSON line per finished
run: the host, branch and commit, the paths it selected, every failed test id, and the machine's
state when each one failed — the load average and how many other pytest runs were going. This
module owns the log's location and format, and reads one or more logs back::

    python scripts/flaky_report.py                    # this clone's log
    python scripts/flaky_report.py a.jsonl b.jsonl    # several machines' logs, combined

**Why the log lives in the clone's git directory.** Not ``TMPDIR`` or ``HOME``: sessions here give
each run a private pair (``with_suite_lock.py`` records why), so a log under either would be split
across sessions. Not ``/tmp``: macOS clears it on reboot, and the log is meant to last a week.
``git rev-parse --git-common-dir`` is shared by every worktree of a clone, survives a reboot, and
git never tracks it. A second clone on the same machine keeps its own log.
``DETAIL_TEST_RUN_LOG`` overrides the path.

**A test's rate is failures over the runs that selected it.** A run selected a test when one of
its paths is a prefix of the test's id, or it named no paths. That over-counts runs that stopped
early (``--maxfail``) or filtered with ``-k`` / ``-m``, so a rate is a lower bound. Runs that were
interrupted are left out of both counts.

Standard library only: the plugin imports this in every run, before anything else loads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path

SCHEMA_VERSION = 1
LOG_NAME = "detail-test-runs.jsonl"
#: pytest's exit codes for a run that did not finish on its own terms.
_INTERRUPTED = {2, 3, 4}


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def log_path(root: Path) -> Path | None:
    """The log for the clone containing ``root``; ``None`` outside a git checkout."""
    override = os.environ.get("DETAIL_TEST_RUN_LOG")
    if override:
        return Path(override)
    common = _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common) / LOG_NAME if common else None


def git_head(root: Path) -> tuple[str | None, str | None]:
    """The branch and short commit ``root`` is on.

    Two calls: in one, ``--abbrev-ref`` carries over to the second ``HEAD`` and names the branch
    twice.
    """
    return (
        _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        _git(root, "rev-parse", "--short", "HEAD"),
    )


def append(path: Path, record: dict) -> None:
    """Add one run as one line, in a single write, so concurrent runs never interleave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps({"schema_version": SCHEMA_VERSION, **record}) + "\n").encode()
    fd = os.open(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def read(paths: list[Path]) -> list[dict]:
    """Every finished run in the given logs. A line that does not parse is skipped."""
    runs = []
    for path in paths:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            try:
                run = json.loads(line)
            except ValueError:
                continue
            if run.get("exitstatus") not in _INTERRUPTED:
                runs.append(run)
    return runs


def selects(run: dict, node_id: str) -> bool:
    """Whether ``run`` selected the test ``node_id`` (see the module docstring)."""
    paths = run.get("paths") or []
    return not paths or any(node_id.startswith(p.rstrip("/")) for p in paths)


def _commonest_error(seen: list[dict]) -> str | None:
    """The crash line this test fails with most often; runs logged before it was kept have none."""
    errors = Counter(f["error"] for f in seen if f.get("error"))
    return errors.most_common(1)[0][0] if errors else None


def report(runs: list[dict]) -> list[dict]:
    """One row per test that failed at least once, most failures first."""
    failures: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        for failure in run.get("failures", []):
            failures[failure["id"]].append(
                {**failure, "host": run.get("host"), "branch": run.get("branch")}
            )
    rows = []
    for node_id, seen in failures.items():
        loads = [f["load"] for f in seen if f.get("load") is not None]
        rows.append(
            {
                "id": node_id,
                "failed": len(seen),
                "runs": max(len(seen), sum(selects(run, node_id) for run in runs)),
                "hosts": sorted({f["host"] for f in seen if f.get("host")}),
                "load": (min(loads), max(loads)) if loads else None,
                "others": max((f.get("other_runs") or 0) for f in seen),
                "branches": len({f["branch"] for f in seen if f.get("branch")}),
                "error": _commonest_error(seen),
            }
        )
    return sorted(rows, key=lambda r: (-r["failed"], -r["failed"] / r["runs"], r["id"]))


def render(runs: list[dict], rows: list[dict]) -> str:
    hosts = sorted({r.get("host") for r in runs if r.get("host")})
    lines = [f"{len(runs)} runs from {len(hosts)} host(s): {', '.join(hosts)}"]
    loads = [r["load_start"] for r in runs if r.get("load_start") is not None]
    if loads:
        lines.append(
            f"load at run start: {min(loads):.1f}–{max(loads):.1f}, "
            f"median {sorted(loads)[len(loads) // 2]:.1f}"
        )
    if not rows:
        lines.append("no test failed")
        return "\n".join(lines)
    lines.append("")
    lines.append(
        f"{'failed / runs':>13}  {'load':>9}  {'others':>6}  {'branches':>8}  test"
    )
    for row in rows:
        load = f"{row['load'][0]:.1f}–{row['load'][1]:.1f}" if row["load"] else "?"
        lines.append(
            f"{row['failed']:>6} / {row['runs']:<4}  {load:>9}  {row['others']:>6}  "
            f"{row['branches']:>8}  {row['id']}"
        )
        if row["error"]:
            lines.append(f"{'':>45}{row['error'][:100]}")
    lines.append("")
    lines.append(
        "load: the 1-minute load average when it failed; "
        "others: the most other pytest runs going at the time"
    )
    lines.append(
        "branches: how many branches it failed on — a test failing on one branch is usually "
        "that branch's work in progress; on many, a flake"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "logs", nargs="*", type=Path, help="log files; default: this clone's"
    )
    args = parser.parse_args(argv)
    logs = args.logs or [log_path(Path.cwd())]
    missing = [p for p in logs if p is None or not p.is_file()]
    if missing:
        print(f"no log at {missing[0]}; it appears after the first finished run")
        return 1
    runs = read(logs)
    print(render(runs, report(runs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
