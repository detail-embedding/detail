"""Stream importers (Section 3) and their registration entrypoint.

Each concrete ``StreamImporter`` lives in its own module and self-registers on
import (Tenet 1). ``import_stream_importers`` imports them so ``STREAM_IMPORTERS``
is populated before any read — mirrors ``executor.runners.import_runner_modules``.
Listing a module here is safe even when its parsing deps are heavy; the import is
deferred to the first stream operation.
"""

import importlib

STREAM_IMPORTER_MODULES: tuple[str, ...] = (
    "detail.importers.medpc",
    "detail.importers.behavior",
    "detail.importers.boris",
    "detail.importers.timeseries_csv",
    "detail.importers.events_csv",
)


def import_stream_importers() -> None:
    """Import every stream-importer module so its ``register_stream_importer``
    decorator runs and populates ``STREAM_IMPORTERS``."""
    for name in STREAM_IMPORTER_MODULES:
        importlib.import_module(name)
