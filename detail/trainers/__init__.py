"""Detection-model trainers (§12) and their registration entrypoint.

Each concrete ``Trainer`` lives in its own module and self-registers on import (Tenet 1).
``import_trainers`` imports them so ``TRAINERS`` is populated before any read — the same shape
as ``tools.import_tools`` and ``query_methods.import_query_methods``.

This entrypoint arrived late. ``TRAINERS`` was populated by a direct
``import detail.trainers.ultralytics_yolo`` at each use site, so there was no one place to ask
"what trainers exist", and the parameter reference walked the registries that had an
entrypoint and silently skipped this one — leaving ``UltralyticsYOLOConfig`` out of the docs,
the CLI help map, and every config guard. A registry with no entrypoint is a registry nothing
can enumerate.
"""

import importlib

TRAINER_MODULES: tuple[str, ...] = ("detail.trainers.ultralytics_yolo",)


def import_trainers() -> None:
    """Import every trainer module so its ``register_trainer`` decorator runs and populates
    ``TRAINERS``."""
    for name in TRAINER_MODULES:
        importlib.import_module(name)
