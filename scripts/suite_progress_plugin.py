"""A pytest plugin that keeps a run's progress in a file ``suite_progress.py`` can draw.

Loaded on every run by ``addopts`` in ``pyproject.toml``. It changes nothing about what runs or how:
it counts reports and rewrites one small JSON file at most twice a second.

- **Under xdist it runs in the controller only.** Every worker's reports reach the controller, and
  a worker registering too would write a file per worker, each counting a fraction of the run.
- **The total comes from wherever collection happened**: the session in a plain run, the first
  worker's collection under xdist, where the controller's own session collects nothing.
- **A run started inside a run does not register.** ``DETAIL_SUITE_PROGRESS_ACTIVE`` is set while a
  run is being tracked, and a pytest a test starts (in-process or as a subprocess, which inherits
  the environment) sees it and stays quiet, rather than showing on the status line as a second run.
- **A write that fails is dropped.** The file is a display; losing one update loses nothing.
- **Each finished run is also appended to a log** (``flaky_report.py`` owns it and reads it back),
  with every failure, the phase it failed in and its crash line, and the machine's state when it
  happened, so a test that fails now and then can be counted across a week of runs and every
  agent's sessions — and its failures told apart by cause. An interrupted run is logged
  with its exit status and left out of the report; a killed one never reaches the hook.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

import pytest

from scripts import flaky_report
from scripts import suite_progress

_ACTIVE = "DETAIL_SUITE_PROGRESS_ACTIVE"
#: Minimum seconds between writes; the status line redraws once a second.
_WRITE_EVERY_S = 0.5
#: Failed test ids kept in the file, for the watcher to print as they happen.
_MAX_FAILED_IDS = 50
#: A logged failure keeps the head of its crash line: enough to tell one cause from another.
_MAX_ERROR_CHARS = 300


def _load() -> float | None:
    """The 1-minute load average; ``None`` on Windows, which has none."""
    if not hasattr(os, "getloadavg"):
        return None
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


def _crash_line(report: pytest.TestReport) -> str | None:
    """The one line pytest prints after ``E``, or the report's last line when it has none."""
    crash = getattr(report.longrepr, "reprcrash", None)
    line = getattr(crash, "message", None)
    if not line:
        lines = str(report.longrepr or "").strip().splitlines()
        line = lines[-1] if lines else None
    return line[:_MAX_ERROR_CHARS] if line else None


def _selected_paths(config: pytest.Config) -> list[str]:
    """The run's path arguments, relative to the rootdir and in ``/`` form, as node ids are."""
    root = config.rootpath.resolve()
    base = config.invocation_params.dir
    paths = []
    for arg in config.args:
        path, sep, rest = arg.partition("::")
        try:
            relative = (base / path).resolve().relative_to(root).as_posix()
        except ValueError:
            relative = path
        paths.append("" if relative == "." else relative + sep + rest)
    return [] if "" in paths else paths


class _Progress:
    def __init__(self, config: pytest.Config) -> None:
        self._config = config
        self._failures: list[dict] = []
        self._load_start = _load()
        self._last_write = 0.0
        self._failed: set[str] = set()
        self._skipped: set[str] = set()
        self.state = {
            "pid": os.getpid(),
            "root": str(config.rootpath),
            "label": config.rootpath.name,
            "started": time.time(),
            "total": None,
            "done": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "failed_ids": [],
            "finished_at": None,
            "exitstatus": None,
        }
        self._write(force=True)
        self._other_runs_start = self._other_runs()

    def _write(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < _WRITE_EVERY_S:
            return
        self._last_write = now
        try:
            suite_progress.write_state(self.state)
        except OSError:
            pass

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        if session.items:
            self.state["total"] = len(session.items)
            self._write(force=True)

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids) -> None:
        if self.state["total"] is None:
            self.state["total"] = len(ids)
            self._write(force=True)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        node = report.nodeid
        if report.failed and node not in self._failed:
            self._failed.add(node)
            self.state["failed"] += 1
            self._failures.append(
                {
                    "id": node,
                    "when": report.when,
                    "error": _crash_line(report),
                    "load": _load(),
                    "other_runs": self._other_runs(),
                }
            )
            if len(self.state["failed_ids"]) < _MAX_FAILED_IDS:
                self.state["failed_ids"].append(node)
            self._write(force=True)
        elif report.skipped:
            self._skipped.add(node)
        if report.when == "teardown":
            self.state["done"] += 1
            if node in self._failed:
                pass
            elif node in self._skipped:
                self.state["skipped"] += 1
            else:
                self.state["passed"] += 1
            self._write()

    def _other_runs(self) -> int:
        """How many other pytest runs on this machine are still going."""
        try:
            runs = suite_progress.read_runs()
        except OSError:
            return 0
        return sum(
            1
            for run in runs
            if run.get("pid") != self.state["pid"] and run.get("finished_at") is None
        )

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self.state["finished_at"] = time.time()
        self.state["exitstatus"] = int(exitstatus)
        self._write(force=True)
        self._log()

    def _log(self) -> None:
        root = self._config.rootpath
        path = flaky_report.log_path(root)
        if path is None:
            return
        branch, commit = flaky_report.git_head(root)
        state = self.state
        record = {
            "host": socket.gethostname(),
            "platform": sys.platform,
            "checkout": Path(root).name,
            "branch": branch,
            "commit": commit,
            "started": round(state["started"], 1),
            "duration_s": round(state["finished_at"] - state["started"], 1),
            "exitstatus": state["exitstatus"],
            "paths": _selected_paths(self._config),
            "workers": getattr(self._config.option, "numprocesses", None),
            "total": state["total"],
            "passed": state["passed"],
            "failed": state["failed"],
            "skipped": state["skipped"],
            "load_start": self._load_start,
            "load_end": _load(),
            "other_runs_start": self._other_runs_start,
            "failures": self._failures,
        }
        try:
            flaky_report.append(path, record)
        except OSError:
            pass


def pytest_configure(config: pytest.Config) -> None:
    if hasattr(config, "workerinput") or os.environ.get(_ACTIVE):
        return
    os.environ[_ACTIVE] = "1"
    config.pluginmanager.register(_Progress(config), "suite-progress-writer")


def pytest_unconfigure(config: pytest.Config) -> None:
    if config.pluginmanager.get_plugin("suite-progress-writer") is not None:
        os.environ.pop(_ACTIVE, None)
