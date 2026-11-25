"""A test fakes one module's clock, never the process's.

**The class: a patch through a module's ``time`` attribute is a patch of ``time`` itself.**
``monkeypatch.setattr(core_progress.time, "monotonic", fake)`` looks local and is global, because
``core_progress.time`` *is* the one ``time`` module. Eight tests did it. On CI the unit job runs in a
single process with the stall watchdog's thread, which read the simulated clock (set to about
1,000 s on a runner up for 286 s) as 714 s of silence and ended a run 8 minutes old. On a laptop up
for days the same subtraction comes out negative, and tests run in xdist workers, so it never fired
locally. The five ``sleep`` patches made the watchdog's loop spin instead of wait.

Three parts, all here: :func:`tests.module_clock.patch_module_time` gives one module a private
``time`` and leaves the real one alone; no test file patches the shared module; and the watchdog
keeps the clock and sleep it loaded with, so a patch that slips past the scan still cannot end a
run. This file is the one place that patches the shared module, to show that last part.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from detail.core import atomic
from scripts import suite_watchdog
from tests.module_clock import patch_module_time

TESTS = Path(__file__).resolve().parent

#: ``setattr(x.time, "…")``, ``setattr(time, "…")`` and ``patch("time.…")``: each replaces a
#: function of the one shared ``time`` module.
_SHARED_TIME_PATCH = re.compile(
    r"""setattr\(\s*(?:[\w.]+\.)?time\s*,\s*["']\w+["']"""
    r"""|setattr\(\s*["']time\.\w+["']"""
    r"""|patch(?:\.object)?\(\s*(?:["']time\.\w+["']|(?:[\w.]+\.)?time\s*,)"""
)


def _offending_lines(text: str) -> list[int]:
    return [
        n
        for n, line in enumerate(text.splitlines(), 1)
        if _SHARED_TIME_PATCH.search(line)
    ]


def test_the_scan_catches_each_form_and_not_the_helper() -> None:
    assert _offending_lines(
        'monkeypatch.setattr(core_progress.time, "monotonic", f)'
    ) == [1]
    assert _offending_lines('monkeypatch.setattr(time, "sleep", f)') == [1]
    assert _offending_lines('monkeypatch.setattr("time.time", f)') == [1]
    assert _offending_lines('with mock.patch("time.sleep"):') == [1]
    assert _offending_lines("with mock.patch.object(time, 'monotonic'):") == [1]
    assert _offending_lines("patch_module_time(monkeypatch, atomic, sleep=f)") == []
    assert _offending_lines('monkeypatch.setattr(runtime, "x", f)') == []
    assert _offending_lines('monkeypatch.setattr(item, "start_time", f)') == []


def test_no_test_patches_the_shared_time_module() -> None:
    found = {}
    for path in sorted(TESTS.rglob("*.py")):
        rel = path.relative_to(TESTS.parent).as_posix()
        if path == Path(__file__).resolve():
            continue
        if lines := _offending_lines(path.read_text(encoding="utf-8-sig")):
            found[rel] = lines
    assert not found, (
        "these patch Python's one `time` module for the whole process, including the stall "
        f"watchdog's thread; use tests.module_clock.patch_module_time instead: {found}"
    )


def test_the_helper_fakes_one_module_and_leaves_time_alone(monkeypatch) -> None:
    real_sleep = time.sleep
    calls: list[float] = []
    patch_module_time(monkeypatch, atomic, sleep=calls.append)
    atomic.time.sleep(5)
    assert calls == [5]
    assert time.sleep is real_sleep
    assert atomic.time is not time
    assert atomic.time.monotonic is time.monotonic


def test_the_helper_refuses_a_name_time_does_not_have(monkeypatch) -> None:
    with pytest.raises(AttributeError, match="no function 'monotonik'"):
        patch_module_time(monkeypatch, atomic, monotonik=lambda: 0.0)


def test_the_watchdog_keeps_the_clock_it_loaded_with(monkeypatch) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 1e9)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    assert suite_watchdog._monotonic() < 1e9
    assert suite_watchdog._sleep is not time.sleep


def test_the_watchdog_never_reads_the_clock_through_time() -> None:
    source = Path(suite_watchdog.__file__).read_text(encoding="utf-8-sig")
    assert "time.monotonic()" not in source
    assert "time.sleep(" not in source
