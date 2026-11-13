"""A stuck or crashed test ends the run on its own, naming what stuck — never a run left waiting.

**The class: a run that stops and does not end.** Nothing fails, nothing prints, every process
sits at 0% CPU, and the only detector is a person noticing the bar has not moved. Runs hung
near the end that way, each after one worker segfaulted in a GUI teardown. The mechanism,
reproduced here in a scratch suite of 40 one-test files: under
``--dist loadfile`` pytest-xdist 3.8 re-queues every file a dead worker had ever been given,
finished ones included, and a finished file goes to a worker as an empty list — that worker
never reports again and is never given more, so the controller waits forever.

Three settings in ``pyproject.toml`` close it, and each is pinned by running a real pytest in a
fresh interpreter with the repo's own settings (read from ``pyproject.toml``, so deleting one
fails here, not on the next hang):

- ``--max-worker-restart=0``, defaulted by the watchdog plugin — a dead worker fails the run at
  once with the test id.
  ``test_a_dead_worker_fails_the_run_naming_its_test``.
- ``faulthandler_timeout`` + ``faulthandler_exit_on_timeout`` — a test stuck *inside* a worker is
  dumped and its worker ended, which the line above turns into a failed run naming it.
  ``test_a_stuck_test_under_xdist_is_named_within_the_bound``.
- ``scripts/suite_watchdog.py`` (``suite_stall_timeout``) — the net for a stall no test owns: it is
  shown the xdist hang itself, with restarts switched back on, and must end it and say what the
  controller was waiting on. ``test_the_watchdog_ends_the_xdist_requeue_hang``; and a process a
  leaked thread keeps alive after the summary, ``test_a_process_that_will_not_exit_is_ended``.

The worker "crash" is ``os._exit``, not a segfault: same node-down to xdist, and no macOS crash
report written per run.
"""

from __future__ import annotations

import os
import sys
import time
import tomllib
from pathlib import Path

from tests.subprocesses import run_tree

REPO = Path(__file__).resolve().parents[2]

_INI_KEYS = (
    "addopts",
    "faulthandler_timeout",
    "faulthandler_exit_on_timeout",
    "suite_stall_timeout",
)

_ONE_TEST = "import time\ndef test_1(): time.sleep(0.05)\n"

#: Dies the first time only — a flaky crash, the shape that recovered-then-hung in the suite.
_DIES_ONCE = """
import os, pathlib
FLAG = pathlib.Path(__file__).with_name("died")
def test_dies_once():
    if not FLAG.exists():
        FLAG.write_text("x")
        os._exit(1)
"""


def _repo_ini() -> dict[str, object]:
    with open(REPO / "pyproject.toml", "rb") as f:
        options = tomllib.load(f)["tool"]["pytest"]["ini_options"]
    return {key: options[key] for key in _INI_KEYS if key in options}


def _project(tmp_path: Path, files: dict[str, str]) -> Path:
    """A scratch suite carrying the repo's own run settings."""
    lines = ["[pytest]"]
    for key, value in _repo_ini().items():
        if isinstance(value, bool):
            value = str(value).lower()
        lines.append(f"{key} = {value}")
    (tmp_path / "pytest.ini").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, body in files.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def _late_crash_suite(tmp_path: Path) -> Path:
    """Forty one-test files, the thirtieth killing its worker once: the layout that hangs."""
    files = {f"test_f{i:02d}.py": _ONE_TEST for i in range(1, 41)}
    files["test_f30.py"] = _DIES_ONCE
    return _project(tmp_path, files)


def _run(project: Path, *args: str, budget_s: float = 150.0):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = str(REPO)
    env["DETAIL_SUITE_PROGRESS_ACTIVE"] = (
        "1"  # the scratch run stays off the status line
    )
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args]
    started = time.monotonic()
    # `run_tree`, not `subprocess.run`: the scratch run starts xdist workers of its own, and
    # on Windows a timeout on the parent alone leaves the wait on their pipes unbounded.
    proc = run_tree(
        cmd,
        timeout=budget_s,
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc, time.monotonic() - started


def test_the_repo_settings_are_the_ones_under_test():
    ini = _repo_ini()
    assert "scripts.suite_watchdog" in str(ini["addopts"])
    assert ini["faulthandler_exit_on_timeout"] is True
    # The watchdog sits above the per-test bound, so a stuck test is named with its own stack.
    assert float(ini["suite_stall_timeout"]) > float(ini["faulthandler_timeout"])


def test_a_dead_worker_fails_the_run_naming_its_test(tmp_path):
    proc, took = _run(_late_crash_suite(tmp_path), "-n", "2", "--dist", "loadfile")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out[-3000:]
    assert "crashed while running 'test_f30.py::test_dies_once'" in out, out[-3000:]
    assert "worker restarting disabled" in out, out[-3000:]
    assert took < 90, f"took {took:.0f}s"


def test_a_stuck_test_under_xdist_is_named_within_the_bound(tmp_path):
    project = _project(
        tmp_path,
        {
            "test_stuck.py": "import time\ndef test_never_returns(): time.sleep(600)\n",
            "test_fine.py": _ONE_TEST,
        },
    )
    proc, took = _run(
        project, "-n", "2", "--dist", "loadfile", "-o", "faulthandler_timeout=3"
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out[-3000:]
    assert "Timeout" in out and "test_never_returns" in out, out[-3000:]
    assert "crashed while running 'test_stuck.py::test_never_returns'" in out, out[
        -3000:
    ]
    assert took < 90, f"took {took:.0f}s"


def test_the_watchdog_ends_the_xdist_requeue_hang(tmp_path):
    """Restarts back on — the setting that hung three runs — and only the watchdog to end it."""
    proc, took = _run(
        _late_crash_suite(tmp_path),
        "-n",
        "2",
        "--dist",
        "loadfile",
        "--max-worker-restart=4",
        "-o",
        "suite_stall_timeout=20",
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 3, out[-3000:]
    assert "suite_watchdog: no progress for" in out, out[-3000:]
    assert "never taken by a worker" in out, out[-3000:]
    assert took < 90, f"took {took:.0f}s"


def test_a_process_that_will_not_exit_is_ended(tmp_path):
    """A non-daemon thread a test leaked keeps the interpreter up after the summary."""
    project = _project(
        tmp_path,
        {
            "test_leak.py": (
                "import threading, time\n"
                "def test_leaks_a_thread():\n"
                "    threading.Thread(target=time.sleep, args=(600,)).start()\n"
            )
        },
    )
    # The grace after the session is the stall bound when that is shorter.
    proc, took = _run(project, "-o", "suite_stall_timeout=20")
    out = proc.stdout + proc.stderr
    assert "1 passed" in out, out[-3000:]
    assert proc.returncode == 3, out[-3000:]
    assert "exiting after the session finished" in out, out[-3000:]
    assert took < 90, f"took {took:.0f}s"
