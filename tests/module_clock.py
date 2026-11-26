"""Fake the clock one module sees, and only that module.

Patching ``monotonic`` on ``some_module.time`` reads as "patch this module's clock" but patches
Python's one ``time`` module, so every thread in the process gets the fake: the stall
watchdog read a simulated clock set ahead as 714 s with no progress and ended a CI run that was 8
minutes old. :func:`patch_module_time` gives the module a private copy of ``time`` instead.
``tests/test_module_clock.py`` holds both halves: the helper leaves the real clock alone, and no
test patches the shared module.
"""

from __future__ import annotations

import time
import types

import pytest


def patch_module_time(
    monkeypatch: pytest.MonkeyPatch, module: types.ModuleType, **fakes
) -> None:
    """Give ``module`` its own ``time``, with each named function in ``fakes`` replaced.

    The module must use ``import time`` and call ``time.<name>``; every other module, and the
    real ``time``, keep the real clock. Restored at teardown by ``monkeypatch``.

    Raises:
        AttributeError: a name in ``fakes`` is not a function of ``time``.
    """
    private = types.ModuleType("time")
    private.__dict__.update(vars(time))
    for name, fake in fakes.items():
        if not callable(getattr(time, name, None)):
            raise AttributeError(f"time has no function {name!r} to fake")
        setattr(private, name, fake)
    monkeypatch.setattr(module, "time", private)
