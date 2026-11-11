"""Behavior-query search methods (§20) and their registration entrypoint.

Each concrete ``QueryMethod`` lives in its own module and self-registers on import
(Tenet 1). ``import_query_methods`` imports them so ``QUERY_METHODS`` is populated before
any read — mirrors ``importers.import_stream_importers``.

**``soft_knn`` is the default (``configs/query.DEFAULT_METHOD``) and, since `dtw` was removed
(see ``docs/removed_capabilities.md``), the only method.** It was chosen on cost rather than
accuracy: a 30-minute recording took `soft_knn` about a second and `dtw` half a minute per
example clip, and which retrieves better was never measured.
"""

import importlib

QUERY_METHOD_MODULES: tuple[str, ...] = ("detail.query_methods.soft_knn",)


def import_query_methods() -> None:
    """Import every query-method module so its ``register_query_method`` decorator runs
    and populates ``QUERY_METHODS``."""
    for name in QUERY_METHOD_MODULES:
        importlib.import_module(name)
