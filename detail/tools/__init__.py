"""Tools (Section 27) and their registration entrypoint.

A *tool* is a one-off, experiment-free operation (e.g. Quick segment: run a
segmentation model on a short clip). Each concrete ``Tool`` lives in its own module
and self-registers on import (Tenet 1). ``import_tools`` imports them so ``TOOLS``
is populated before any lookup — mirrors ``importers.import_stream_importers`` /
``executor.runners.import_runner_modules``. Listing a module here is safe even when
its backend deps are heavy; the import stays deferred to the first tool operation.
"""

import importlib

TOOL_MODULES: tuple[str, ...] = ("detail.tools.quick_segment",)


def import_tools() -> None:
    """Import every tool module so its ``register_tool`` decorator runs and
    populates ``TOOLS``."""
    for name in TOOL_MODULES:
        importlib.import_module(name)
