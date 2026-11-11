"""A run recorded by a capability this version no longer has still lists and opens.

**The class.** A reader that resolves a recorded id against a registry of what the tool can
*run*. Run outputs are never invalidated (CLAUDE.md, *Expensive derived data survives an
update*), so a record on disk outlives the code that wrote it — and a reader that requires the
writer to still be registered turns "this version cannot make another of these" into "this
experiment will not open". Each test here writes the record a removed capability left behind
and asserts the reading path survives it. Launching is a different question with its own
tests: an id that is not registered is refused by name at ``validate()``.

The capabilities are listed, with what each was and how to restore it, in
``docs/removed_capabilities.md``.
"""

import numpy as np
import pytest

from detail.api.experiment import Experiment


def _recording_with_embedding_run(tmp_path, make_embedding_run, **run_fields):
    experiment = Experiment.create("p", tmp_path)
    recording = experiment.create_recording("recording_001")
    n = 12
    make_embedding_run(
        recording.dir,
        np.random.default_rng(0).normal(size=(n, 8)).astype(np.float32),
        np.arange(n),
        np.arange(n) * 1_000_000,
        **run_fields,
    )
    return experiment, recording


def test_an_embedding_run_made_with_a_removed_model_lists_and_opens(
    tmp_path, make_embedding_run
):
    """The PE-Core models are gone; a run that names one is still an embedding run."""
    _experiment, recording = _recording_with_embedding_run(
        tmp_path,
        make_embedding_run,
        model_id="pe_core_l14_336",
        model_variant="PE-Core-L-14-336",
    )

    assert [r["id"] for r in recording.embedding_runs] == ["emb_01"]
    preview = recording.embedding_preview("emb_01")
    assert preview.metadata()["model_id"] == "pe_core_l14_336"
    assert preview.track_ids() == [0]


def test_a_tracking_run_made_with_the_removed_tracker_lists_and_opens(
    processed_recording,
):
    """``iou_tracker`` is gone; every tracking run it wrote still names it."""
    import yaml

    from detail.core.registries import TRACKERS
    from detail.storage import recording as recording_store

    recording = processed_recording.recording
    data = recording_store.read_recording(recording.dir)
    data["tracking_runs"][0].update(
        tracker_id="iou_tracker", hyperparams={"iou_threshold": 0.3}
    )
    recording_store.write_recording(recording.dir, data)
    track_yaml = recording.dir / "tracking" / "track_01" / "track.yaml"
    record = yaml.safe_load(track_yaml.read_text(encoding="utf-8-sig"))
    record.update(tracker_id="iou_tracker", hyperparams={"iou_threshold": 0.3})
    track_yaml.write_text(yaml.dump(record), encoding="utf-8")
    assert "iou_tracker" not in TRACKERS

    [listed] = recording.tracking_runs
    assert listed["tracker_id"] == "iou_tracker"
    ids, presence = recording.tracking_preview("track_01").track_presence()
    assert ids == [0] and presence[0].any()
    assert processed_recording.tracking_run_id in {
        run["id"] for run in processed_recording.experiment.processing_runs
    }


#: The step lists the six removed input-image recipes resolved to, as an embedding run records
#: them in ``renderer_spec``. Written out, not built: the builders are gone and the record is
#: what is left.
_REMOVED_RECIPE_SPECS = {
    "silhouette": [{"type": "init_binary_mask"}, {"type": "pad_to_square"}],
    "masked_color_centered_blue": [
        {"type": "crop_to_bbox"},
        {"type": "apply_mask", "background": [0, 0, 255]},
        {"type": "center_in_square", "size": 48, "fill": [0, 0, 255]},
    ],
    "masked_color_centered_noise": [
        {"type": "crop_to_bbox"},
        {"type": "apply_mask", "background": 0},
        {"type": "center_in_square", "size": 48},
        {"type": "resize", "size": 32},
        {"type": "fill_background_noise", "seed": 1234},
    ],
    "water_drop": [
        {"type": "crop_to_bbox"},
        {"type": "apply_mask", "background": 0},
        {"type": "center_in_square", "size": 48},
        {"type": "resize", "size": 32},
        {
            "type": "fill_background_bands",
            "ring": [0, 0, 255],
            "alt": [0, 0, 0],
            "thickness": 2,
        },
    ],
    "context_crop": [
        {"type": "init_frame"},
        {"type": "crop_to_bbox", "padding": 1.0},
        {"type": "pad_to_square"},
    ],
    "bbox_crop_color": [
        {"type": "init_frame"},
        {"type": "crop_to_bbox"},
        {"type": "center_in_square", "size": 48},
    ],
}


@pytest.mark.parametrize("recipe", sorted(_REMOVED_RECIPE_SPECS))
def test_an_embedding_run_made_with_a_removed_input_image_opens_and_renders(
    tmp_path, make_embedding_run, recipe
):
    """A removed recipe's run is read from its recorded steps, and every step is still here.

    Removing a recipe is safe only because a run records the steps it resolved to. The render
    is what would break if a step type were ever removed along with the recipe that used it.
    """
    from detail.core.dataclasses import RenderContext
    from detail.transforms import basic  # noqa: F401 — registers the steps
    from detail.transforms.constructors import CONSTRUCTORS
    from detail.transforms.renderer import MaskRenderer

    assert recipe not in CONSTRUCTORS
    spec = _REMOVED_RECIPE_SPECS[recipe]
    _experiment, recording = _recording_with_embedding_run(
        tmp_path, make_embedding_run, constructor=recipe, renderer_spec=spec
    )

    assert [r["id"] for r in recording.embedding_runs] == ["emb_01"]
    recorded = recording.embedding_preview("emb_01").metadata()
    assert recorded["constructor"] == recipe
    assert recorded["renderer_spec"] == spec

    mask = np.zeros((40, 60), bool)
    mask[10:30, 20:40] = True
    image = MaskRenderer.from_spec(recorded["renderer_spec"]).render(
        RenderContext(
            frame=np.full((40, 60, 3), 200, np.uint8),
            mask=mask,
            bbox=(0.2, 0.1, 0.8, 0.9),
            image=None,
            frame_idx=0,
            t_ns=0,
            class_name="mouse",
            confidence=0.9,
            object_id=0,
            aux={},
        )
    )
    assert image.ndim == 3 and image.dtype == np.uint8
