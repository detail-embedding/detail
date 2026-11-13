"""Checks (§1, the live checker framework) and their registration entrypoint.

A *check* answers one readiness / data-integrity question and returns a ``CheckResult``.
Each concrete ``Check`` lives in a module here and self-registers on import (Tenet 1).
``import_checks`` imports them so ``CHECKS`` is populated before any lookup — mirrors
``tools.import_tools`` / ``executor.runners.import_runner_modules``. Import-sensitive
checks defer their backend import into ``run`` (Tenet 13), so listing a module here pulls
in no torch/sam3.
"""

import importlib

CHECK_MODULES: tuple[str, ...] = (
    "detail.checks.environment",
    "detail.checks.setup",
    "detail.checks.recording",
    "detail.checks.analysis",
)


def import_checks() -> None:
    """Import every check module so its ``register_check`` decorator runs and populates
    ``CHECKS``."""
    for name in CHECK_MODULES:
        importlib.import_module(name)
