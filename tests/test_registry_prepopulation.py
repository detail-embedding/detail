"""The registry-isolation fixture pre-populates the production registries at every test's
start, so a run behaves the same whether or not it collected the pipeline test files (a
subset like ``pytest tests/gui`` used to start with empty registries — see
``conftest._prepopulate_registries``).

This module imports **no** production backend at module top on purpose: if the fixture did
not pre-populate, these registries would be empty when the file is run on its own, and every
assertion below would fail. Run it alone to confirm the fixture is doing its job:
``pytest tests/test_registry_prepopulation.py``.
"""

from detail.core.registries import (
    EMBEDDING_MODELS,
    MODELS,
    QUERY_METHODS,
    STREAM_IMPORTERS,
    TOOLS,
    TRACKERS,
    TRAINED_LOADERS,
    TRAINERS,
    TRANSFORMS,
)


def test_every_isolated_registry_holds_its_production_entries():
    assert "sam3" in MODELS
    assert "yolo" in TRAINED_LOADERS
    assert "ultralytics_yolo" in TRAINERS
    assert "anchor_tracker" in TRACKERS
    assert "dinov3_convnext_base" in EMBEDDING_MODELS
    assert "resize" in TRANSFORMS
    assert "medpc" in STREAM_IMPORTERS
    assert "quick_segment" in TOOLS
    assert "soft_knn" in QUERY_METHODS
