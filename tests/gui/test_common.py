"""Run-label + common-run helpers for upstream selectors."""

import pytest

pytest.importorskip("PyQt6")

from detail.gui.modules._common import (  # noqa: E402
    common_complete_run_items,
    derived_descriptor,
    run_label,
)


def test_run_label_prefers_processing_run_id():
    rec = {
        "id": "track_01",
        "processing_run_id": "run_003",
        "tracker_id": "anchor_tracker",
        "detection_run_id": "det_01",
        "created_at": "2026-06-09T14:32:05Z",
    }
    label = run_label("tracking", rec)
    assert label == "run_003 · anchor_tracker · det_01 · 2026-06-09 14:32"


def test_run_label_falls_back_to_id_without_processing_run():
    rec = {"id": "det_02", "model_id": "fake_yolo", "model_variant": "11s"}
    label = run_label("detection", rec)
    assert label.startswith("det_02 · fake_yolo · 11s")  # no processing run yet


def test_run_label_shows_a_derived_embedding_transform():
    # A derived run (§14) is otherwise indistinguishable from its base masked run in the
    # dropdown; the transform must show so the two are legible.
    base = {
        "id": "emb_01",
        "model_id": "dinov3_convnext_base",
        "model_variant": "convnext_base",
        "constructor": "masked_color_centered",
    }
    derived = {
        **base,
        "id": "emb_02",
        "model_variant": "convnext_base|derived:drop_top(k=1)(src=emb_01)",
    }
    assert (
        run_label("embedding", base)
        == "emb_01 · dinov3_convnext_base · masked_color_centered"
    )
    assert (
        run_label("embedding", derived)
        == "emb_02 · dinov3_convnext_base · masked_color_centered · drop_top(k=1)"
    )


def test_derived_descriptor_parses_transform_chains():
    assert derived_descriptor(None) is None
    assert derived_descriptor("convnext_base") is None  # a model run, not derived
    assert derived_descriptor("base|derived:center()(src=emb_01)") == "center"
    assert (
        derived_descriptor("base|derived:drop_top(k=1)(src=emb_01)") == "drop_top(k=1)"
    )
    # a derive of a derive chains the transforms
    chained = "base|derived:center()(src=emb_01)|derived:drop_top(k=1)(src=emb_02)"
    assert derived_descriptor(chained) == "center → drop_top(k=1)"


class _Recording:
    def __init__(self, tracking_runs):
        self.tracking_runs = tracking_runs


def test_common_complete_run_items_labels_common_ids():
    rec = {
        "id": "track_01",
        "status": "complete",
        "processing_run_id": "run_003",
        "tracker_id": "anchor_tracker",
    }
    recordings = [_Recording([dict(rec)]), _Recording([dict(rec)])]
    items = common_complete_run_items(recordings, "tracking")
    assert len(items) == 1
    label, value = items[0]
    assert value == "track_01"  # the per-recording id stays the combo's data
    assert "run_003" in label and "anchor_tracker" in label


# --- spec-based upstream selection (§6.5) ----------------------------------- #
def test_spec_items_annotates_coverage():
    from detail.gui.modules._common import spec_items

    groups = [
        {
            "spec": {"module": "detection"},
            "label": "sam3 @ 0.25",
            "recordings": ["s0", "s1"],
        }
    ]
    items = spec_items(groups, total=3)
    assert items == [("sam3 @ 0.25  ·  2/3 recordings", {"module": "detection"})]


def test_covered_ids_matches_by_spec():
    from detail.gui.modules._common import covered_ids

    spec = {"module": "detection", "identity": {"model_id": "sam3"}, "parent": {}}
    groups = [{"spec": spec, "label": "x", "recordings": ["s0", "s1"]}]
    assert covered_ids(groups, spec) == {"s0", "s1"}
    other = {"module": "detection", "identity": {"model_id": "model_001"}, "parent": {}}
    assert covered_ids(groups, other) == set()
    assert covered_ids(groups, None) is None  # a root/"original" choice covers all
