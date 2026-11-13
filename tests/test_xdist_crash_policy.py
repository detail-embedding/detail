"""A crashed xdist worker ends the run; it is never replaced.

**The class: a parallel run that cannot end after one of its workers dies.** pytest-xdist
replaces a crashed worker by default, up to four times the worker count, and under the
``--dist loadfile`` the gate uses, the replacement is handed the crashed file again, the test
that killed its predecessor included. So one crash becomes a loop of crash, respawn, re-collect
the whole suite, crash; and a replacement that dies while bootstrapping is never noticed at all.
On 2026-10-19 two gate runs on the Windows box crashed a worker at 8,885/8,898 and never
finished, one for over an hour while holding the machine suite lock that every other session
queues on; on 2026-11-06 gates lost GUI workers again and some stalled holding the lock. The
crashes were Qt fail-fasts (``Qt6Core.dll`` ``0xC0000409``): a running QThread collected, which
``detail/gui/workers.py::_IN_FLIGHT`` now prevents.

Invisible to every test by construction: the suite under test is the thing that hangs, so no
test in it can observe the hang. Measured instead (#771), on a 16-test suite with one
``os.abort()`` (Windows, ``-n 3 --dist loadfile``): by default 3 of 4 runs ended only after 13
crashes each (78-130s) and the fourth sat until killed at 150s, its replacement having died
bootstrapping (``OSError: [Errno 22]``); with ``--max-worker-restart=0``, 3 of 3 ended in ~12s,
reporting ``worker 'gw0' crashed while running 'test_crash.py::test_aborts'``. Not re-measured
here: a nested ``-n`` run started by a test is a hang class of its own on Windows
(``tests/test_nested_runs_bounded.py``).

**The rule:** ``addopts`` carries ``--max-worker-restart=0``. A crash is a red run naming the
test it crashed on, which is what it was always going to be, instead of a hung one. It is also
what makes ``faulthandler_exit_on_timeout`` stop a run: a hung test exits its worker, and a
replaced worker would re-run the hung file. Inert without ``-n``.

The machine lock deliberately has **no** wall-clock cap of its own. The kernel releases it when
the holder exits for any reason, every hang route known here now ends the process, and a cap is
a guess about the machine's speed that would also cut off a legitimately long timing run;
``scripts/with_suite_lock.py`` serialises benchmarks too.
"""

import shlex
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def worker_restart_limit(addopts: str) -> int | None:
    """The ``--max-worker-restart`` value ``addopts`` sets, or ``None`` if it sets none.

    Both spellings argparse accepts; the last one wins, as it does on a command line.
    """
    tokens = shlex.split(addopts)
    limit = None
    for i, token in enumerate(tokens):
        if token.startswith("--max-worker-restart="):
            limit = int(token.split("=", 1)[1])
        elif token == "--max-worker-restart" and i + 1 < len(tokens):
            limit = int(tokens[i + 1])
    return limit


def _addopts() -> str:
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8-sig"))
    return config["tool"]["pytest"]["ini_options"]["addopts"]


def test_a_crashed_worker_ends_the_run():
    limit = worker_restart_limit(_addopts())
    assert limit == 0, (
        f"addopts sets --max-worker-restart to {limit!r}. A replaced worker under "
        "--dist loadfile re-runs the file that crashed its predecessor, and one crash becomes "
        "a run that never ends while holding the machine lock. Keep it at 0."
    )


def test_the_pin_would_catch_a_missing_or_raised_limit():
    """Guard the guard: the spellings someone would actually write, each read correctly."""
    assert worker_restart_limit("--strict-markers --maxfail=10") is None
    assert worker_restart_limit("--max-worker-restart=0 -p x") == 0
    assert worker_restart_limit("--max-worker-restart 4") == 4
    assert worker_restart_limit("--max-worker-restart=0 --max-worker-restart=8") == 8
