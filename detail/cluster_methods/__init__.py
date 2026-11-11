"""Clustering methods for §22's two discovery modules, and their registration entrypoints.

Each concrete method lives in its own module and self-registers on import (Tenet 1). The two
``import_*`` functions run those imports so the registries are populated before any read — mirrors
``query_methods.import_query_methods``.

**Two registries, not one, and they are the point of this package's shape.** §22 keeps the pose and
temporal protocols apart so a bout method cannot be selected where a pose method belongs: they have
the same shape and answer different questions, and a shared interface would make that mistake
expressible. Sharing the *package* is fine — what must not be shared is the registry a module reads
and the base class a method satisfies.

A **cluster rule is a method here, not a parameter of one**. FINCH offers a nested cluster count and
labels every bout; HDBSCAN offers a `min_cluster_size` and leaves some unassigned. Both are declared
at class level, so one method taking `cluster_rule` as a parameter would have to declare a
`granularity` and a `has_noise` that are false for some of its own settings.
"""

import importlib

#: Empty since pose clustering and its `finch` / `two_step` methods were removed
#: (`docs/removed_capabilities.md`, section 12). The registry, its protocol and this entry point
#: stay, because pose clustering is expected back and restoring it is then two module paths here.
POSE_CLUSTER_METHOD_MODULES: tuple[str, ...] = ()

TEMPORAL_CLUSTER_METHOD_MODULES: tuple[str, ...] = (
    "detail.cluster_methods.dtw_finch",
    "detail.cluster_methods.codebook_viterbi",
)


def import_pose_cluster_methods() -> None:
    """Import every pose-cluster-method module so its ``register_pose_cluster_method`` decorator
    runs and populates ``POSE_CLUSTER_METHODS``."""
    for name in POSE_CLUSTER_METHOD_MODULES:
        importlib.import_module(name)


def import_temporal_cluster_methods() -> None:
    """Import every temporal-cluster-method module so its ``register_temporal_cluster_method``
    decorator runs and populates ``TEMPORAL_CLUSTER_METHODS``."""
    for name in TEMPORAL_CLUSTER_METHOD_MODULES:
        importlib.import_module(name)
