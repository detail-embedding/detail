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


#: The header ``metrics.csv`` carried in the last version that wrote one. Written out, not
#: imported: the writer went with the module, and the file on a user's disk is what is left.
_QUERY_EVAL_METRICS_HEADER = (
    "embed_run_id,behavior,behavior_type,n_recordings,fusion,band_s,iou_threshold,"
    "point_window_s,n_exemplars,seed,trajectory_split_gap_s,max_results,n_gt,n_pred,tp,fp,fn,"
    "ap,precision,recall,best_f1,best_f1_threshold,n_folds,ap_std,precision_std,recall_std,"
    "best_f1_std,method,recall_null,recall_lift,ap_null,ap_lift,n_holdout,negative_pool"
)
_QUERY_EVAL_ARTIFACTS = [
    {"path": "metrics.csv", "kind": "query_eval_metrics", "label": "metrics"},
    {"path": "folds.csv", "kind": "query_eval_folds", "label": "per-fold metrics"},
    {
        "path": "overlap.csv",
        "kind": "query_eval_overlap",
        "label": "scores vs nulls by overlap",
    },
    {
        "path": "predictions.csv",
        "kind": "query_eval_predictions",
        "label": "predictions",
    },
    {
        "path": "ground_truth.csv",
        "kind": "query_eval_ground_truth",
        "label": "ground truth",
    },
    {"path": "pr/groom/", "kind": "query_eval_pr", "label": "PR · groom"},
]


def a_query_eval_run_on_disk(tmp_path) -> Experiment:
    """An experiment holding one completed Behavior-query eval run, ``an_01``, as that module
    left it: the record it wrote (one behavior scored, one skipped) and its tables as text.

    Shared with ``tests/gui/test_run_detail.py``, which opens the same run's page.
    """
    from detail.storage import analysis_runs as analysis_store

    experiment = Experiment.create("p", tmp_path)
    run_dir = analysis_store.analysis_dir(experiment.dir, "an_01")
    (run_dir / "pr" / "groom").mkdir(parents=True)
    (run_dir / "metrics.csv").write_text(
        _QUERY_EVAL_METRICS_HEADER
        + "\nemb_01,groom,state,1,min,0.5,0.5,0.5,1,0,,,3,4,3,1,0,0.8,0.75,1.0,0.85,0.1,"
        "2,0.05,0.0,0.0,0.0,dtw,0.2,0.8,0.1,0.7,1,unlabeled\n",
        encoding="utf-8",
    )
    (run_dir / "folds.csv").write_text(
        "behavior,fold,n_gt,n_pred,tp,fp,fn,ap,precision,recall,best_f1,best_f1_threshold\n"
        "groom,0,3,4,3,1,0,0.8,0.75,1.0,0.85,0.1\n"
        "groom,1,3,4,2,2,1,0.7,0.5,0.66,0.6,0.2\n",
        encoding="utf-8",
    )
    (run_dir / "pr" / "groom" / "fold_00.csv").write_text(
        "recall,precision\n0.5,1.0\n1.0,0.75\n", encoding="utf-8"
    )
    analysis_store.write_analysis_yaml(
        run_dir,
        {
            "id": "an_01",
            "module_id": "query_eval",
            "recording_ids": ["recording_001"],
            "status": "complete",
            "embed_run_id": "emb_01",
            "track_id": 0,
            "params": {
                "method": "dtw",
                "fusion": "min",
                "band_s": 0.5,
                "iou_threshold": 0.5,
                "n_exemplars": 1,
                "seed": 0,
                "folds": 2,
            },
            "behaviors": [
                {
                    "behavior": "groom",
                    "behavior_type": "state",
                    "n_recordings": 1,
                    "n_folds": 2,
                    "skipped": None,
                },
                {
                    "behavior": "rear",
                    "behavior_type": "state",
                    "n_recordings": 0,
                    "n_folds": 0,
                    "skipped": "no eligible recordings",
                },
            ],
            "columns": _QUERY_EVAL_METRICS_HEADER.split(","),
            "artifacts": _QUERY_EVAL_ARTIFACTS,
        },
    )
    return experiment


def test_an_analysis_run_made_by_a_removed_module_lists_reads_and_deletes(tmp_path):
    """Behavior-query eval (``query_eval``) is gone; a run it wrote is still an analysis run.

    The analysis half of the class: the record names a module no registry holds, with a
    per-behavior list and artifact kinds nothing reads any more. Listing, reading and deleting
    it need none of that; launching another is refused by name.
    """
    from detail.executor.runners import ANALYSIS_MODULES

    experiment = a_query_eval_run_on_disk(tmp_path)
    run_dir = experiment.analysis_run_dir("an_01")

    assert [(r.id, r.module_id, r.status) for r in experiment.analysis_runs] == [
        ("an_01", "query_eval", "complete")
    ]
    record = experiment.analysis_record("an_01")
    assert record["module_id"] == "query_eval"
    assert [b["behavior"] for b in record["behaviors"]] == ["groom", "rear"]
    assert experiment.authored_run("an_01") is None

    with pytest.raises(ValueError, match="unknown analysis module: 'query_eval'"):
        experiment.run_analysis("query_eval", record["params"])
    assert "query_eval" not in ANALYSIS_MODULES

    experiment.delete_analysis("an_01")
    assert experiment.analysis_runs == []
    assert not run_dir.exists()
