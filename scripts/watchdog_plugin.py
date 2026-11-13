"""Keep the hang watchdog armed for the rest of a test that has already failed.

Loaded on every run by ``addopts`` in ``pyproject.toml``, beside ``suite_progress_plugin``.

pytest's faulthandler plugin cancels the ``faulthandler_timeout`` timer in
``pytest_exception_interact``, which fires on every failed setup or call, not only under ``--pdb``.
So a failed test's teardown, and the module and session teardowns that follow a worker's last
test, ran with no watchdog: a hang there stopped the whole run with no dump and no failure line.
This arms the timer again. The class, how it hid and the guard are in
``tests/scripts/test_watchdog_plugin.py``.
"""

from __future__ import annotations

import faulthandler

import pytest
from _pytest import faulthandler as pytest_faulthandler


@pytest.hookimpl(trylast=True)
def pytest_exception_interact(node: pytest.Item | pytest.Collector) -> None:
    """Re-arm what ``_pytest.faulthandler``'s ``tryfirst`` impl of this hook just cancelled.

    Only for a test: a collector's failure happens outside any test's protocol, where nothing
    would cancel the timer again and it would fire into whatever ran next. The protocol wrapper
    cancels it once the test ends, as it does for a test that passes.
    """
    config = node.config
    if not isinstance(node, pytest.Item) or config.getoption("usepdb", False):
        return
    timeout = pytest_faulthandler.get_timeout_config_value(config)
    if timeout <= 0:
        return
    faulthandler.dump_traceback_later(
        timeout,
        file=config.stash[pytest_faulthandler.fault_handler_stderr_fd_key],
        exit=pytest_faulthandler.get_exit_on_timeout_config_value(config),
    )
