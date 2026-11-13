"""``tests/subprocesses.py`` bounds a hung process tree and kills all of it.

The class and why it matters are in ``tests/test_nested_runs_bounded.py``. This pins the two
properties the seam exists for, on a tree shaped like a nested pytest: a parent whose child holds
the parent's output pipes. The wait ends near the bound, and the grandchild is dead afterwards, so
it cannot keep a pipe open or a file locked into whatever runs next.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from detail.executor import local
from tests.subprocesses import run_tree

_TIMEOUT_S = 2.0

_PARENT = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
open(sys.argv[1], "w").write(str(child.pid))
print("parent got this far", flush=True)
time.sleep(120)
"""


def test_a_hung_tree_is_reported_within_the_bound_and_none_of_it_survives(
    tmp_path, capsys
):
    pid_file = tmp_path / "grandchild.pid"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run_tree(
            [sys.executable, "-c", _PARENT, str(pid_file)],
            timeout=_TIMEOUT_S,
            capture_output=True,
            text=True,
        )
    assert time.monotonic() - started < _TIMEOUT_S + 15
    # What the tree printed before it stalled is the only record of where it stopped.
    assert "parent got this far" in raised.value.output
    assert "parent got this far" in capsys.readouterr().err
    grandchild = int(pid_file.read_text(encoding="utf-8-sig"))
    deadline = time.monotonic() + 10
    while local.pid_alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not local.pid_alive(grandchild), "the grandchild outlived the tree kill"
