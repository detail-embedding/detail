"""The bout-segmentation helpers have one home, and temporal clustering reaches them there.

``segment_recording``, ``pool_recordings``, ``bout_medoid_distance``, ``detector_params`` and
``detector_params_rows`` used to be defined across three pipeline modules that were hidden from a
default build (``cluster_fit``, ``cluster_apply``, ``syllable_discovery``), and ``LazyTrajectories``
in a fourth (``pipeline/clip_query``). Temporal clustering and query eval (itself since
removed), which were visible, imported them from there — lazily, so an import-only test stayed
green while the visible feature depended on hidden code at run time. All four modules have
since been removed
(``docs/removed_capabilities.md``); the import scan below keeps the visible callers off their
names, so a restored copy cannot become a dependency again by accident.

The class: a visible module importing from a hidden capability's module.
"""

import ast
from pathlib import Path

import numpy as np

import detail
from detail.analysis import bout_segmentation

_PACKAGE_ROOT = Path(detail.__file__).parent
_HIDDEN_HOMES = (
    "detail.pipeline.cluster_fit",
    "detail.pipeline.cluster_apply",
    "detail.pipeline.syllable_discovery",
    "detail.pipeline.clip_query",
)
_VISIBLE_CALLERS = (
    "analysis/bout_segmentation.py",
    "cluster_methods/dtw_finch.py",
)


def _imported_modules(source: str) -> set[str]:
    """Every module a source file imports, at module level or inside a function."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_visible_callers_import_nothing_from_the_hidden_modules():
    for relative in _VISIBLE_CALLERS:
        imported = _imported_modules(
            (_PACKAGE_ROOT / relative).read_text(encoding="utf-8-sig")
        )
        borrowed = sorted(imported & set(_HIDDEN_HOMES))
        assert not borrowed, f"{relative} imports {borrowed}"


def test_the_import_scan_sees_a_lazy_import():
    """The counterexample: the scan must catch the shape the bug had, an import in a function."""
    source = "def fit():\n    from detail.pipeline import cluster_fit\n"
    assert "detail.pipeline.cluster_fit" in _imported_modules(source)


# Moved from tests/pipeline/test_fit_provenance.py when the clusterer fit was removed: they test
# ``segment_recording``'s resolved gap, which temporal clustering's `dtw_finch` relies on.
_SEGMENT_CONFIG = {
    "window_ns": 500_000_000,
    "min_len_ns": 200_000_000,
    "band_ns": 200_000_000,
    "pool": "mean",
    "feat_stride": 1,
    "detector": "novelty",
    "detector_params": {"percentile": 90.0, "min_size": 4},
    "max_gap_ns": None,  # derive from this recording's cadence
}


def test_the_resolved_gap_is_recorded_per_recording():
    """The request may be ``None``; the answer never is, and it can differ per recording."""
    t_ns = np.arange(200, dtype=np.int64) * 20_000_000  # 50 Hz
    emb = np.random.default_rng(0).random((200, 8)).astype(np.float32)
    frame_idx = np.arange(200, dtype=np.int64)

    _bouts, _band, resolved, _trace = bout_segmentation.segment_recording(
        emb, frame_idx, t_ns, dict(_SEGMENT_CONFIG)
    )

    assert resolved["max_gap_ns"] is not None
    assert resolved["max_gap_ns"] > 0, "a derived gap must be a real duration"


def test_a_faster_recording_derives_a_shorter_gap():
    """The same config means different gaps on recordings of different cadence."""
    emb = np.random.default_rng(0).random((200, 8)).astype(np.float32)
    frame_idx = np.arange(200, dtype=np.int64)
    slow = bout_segmentation.segment_recording(
        emb,
        frame_idx,
        np.arange(200, dtype=np.int64) * 40_000_000,
        dict(_SEGMENT_CONFIG),
    )[2]
    fast = bout_segmentation.segment_recording(
        emb,
        frame_idx,
        np.arange(200, dtype=np.int64) * 10_000_000,
        dict(_SEGMENT_CONFIG),
    )[2]
    assert fast["max_gap_ns"] < slow["max_gap_ns"], (
        "the gap is derived from cadence, so two recordings under one config get different "
        "gaps — which is why a record must persist the answer, not the request"
    )
