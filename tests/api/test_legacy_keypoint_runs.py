"""A run made by the pre-§26 ``keypoints`` stage still opens (§26, the 2026-10-16 catch-up).

**The class: a run whose module the build no longer has.** ``remote-detail`` placed points with
one ``keypoints`` stage (``recordings/<id>/keypoints/kp_NN``, an experiment-level record with
``module_id: keypoints``); main's §26 rebuild replaced it with ``keypoint_detect`` /
``keypoint_track``, and the catch-up merge adopted the rebuild. An experiment made before it
still holds those runs, and every place that looks a run's module up in the registry meets an id
it does not know. Hiding governs launching, not reading (CLAUDE.md): the run must still list,
open and export, even though nothing can make a new one.

The fixture writes the old run the way the old stage did — its zarr writer
(``KeypointsPartWriter``), its ``keypoints.yaml``, its ``recording.yaml`` entry and its
experiment-level record — so each test reads exactly what is on a user's disk.
"""

import numpy as np
import pytest
import yaml

from detail.storage import recording as recording_store
from detail.storage import runs as runs_store
from detail.storage.zarr_io import KeypointsPartWriter

#: rd's skeleton shape (keypoints with sides, a body axis), as its keypoints.yaml recorded it.
OLD_SKELETON = {
    "keypoints": [
        {"name": "nose", "side": "midline", "pair": None},
        {"name": "tail_base", "side": "midline", "pair": None},
    ],
    "edges": [["nose", "tail_base"]],
    "front": "nose",
    "back": "tail_base",
}


@pytest.fixture
def legacy(processed_recording):
    """``processed_recording`` plus an old ``kp_01`` on its tracking run, recorded as the
    experiment-level run ``run_kp`` with ``module_id: keypoints``."""
    it = processed_recording
    recording = it.recording
    run_dir = recording.dir / "keypoints" / "kp_01"
    with KeypointsPartWriter(
        run_dir / "track_0" / "part_001.zarr", n_keypoints=2
    ) as writer:
        writer.add_row(
            0, 0, np.array([[0.3, 0.2], [0.1, 0.2]], np.float32), np.ones(2, np.float32)
        )
    (run_dir / "keypoints.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": "kp_01",
                "tracking_run_id": "track_01",
                "track_ids": [0],
                "model_id": "dinov3_match",
                "skeleton": OLD_SKELETON,
                "visibility": {"min_score": 0.4},
                "status": "complete",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    data = recording_store.read_recording(recording.dir)
    data["keypoint_runs"] = [
        {
            "id": "kp_01",
            "status": "complete",
            "output": "keypoints/kp_01/",
            "tracking_run_id": "track_01",
            "model_id": "dinov3_match",
            "track_ids": [0],
        }
    ]
    recording_store.write_recording(recording.dir, data)
    runs_store.write_run(
        it.experiment.dir,
        "run_kp",
        {
            "id": "run_kp",
            "module_id": "keypoints",
            "recording_ids": [recording.id],
            "per_recording": {recording.id: {"run_id": "kp_01", "status": "complete"}},
            "config": {"model_id": "dinov3_match", "tracking_run_id": "track_01"},
            "status": "complete",
        },
    )
    return it


def test_the_experiment_lists_an_old_keypoints_run(legacy):
    runs = {r["id"]: r for r in legacy.experiment.processing_runs}
    assert runs["run_kp"]["module_id"] == "keypoints"


def _rows(path):
    import csv

    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_an_old_run_offers_the_keypoint_targets(legacy):
    from detail.api import exporters
    from detail.api.export_targets import KEYPOINT_TARGETS
    from detail.api.exporters import ExportSubject

    subject = ExportSubject("run", legacy.experiment.dir, "run_kp")
    reasons = {
        t.id: exporters.unavailable_reason(t, subject)
        for t in exporters.targets_for(subject)
    }
    for target_id in KEYPOINT_TARGETS:
        assert not (reasons[target_id] or "").startswith("this is a"), reasons[
            target_id
        ]
    assert reasons["keypoints"] is None  # the long table needs nothing more
    assert reasons["masks"].startswith("this is a keypoints run")  # not every target


def test_an_old_run_exports_its_points_with_the_cut_off_it_recorded(legacy, tmp_path):
    from detail.api import exporters
    from detail.api.exporters import ExportSubject

    result = exporters.write_export(
        ExportSubject("run", legacy.experiment.dir, "run_kp"),
        [exporters.target("run", "keypoints")],
        tmp_path,
    )
    (table,) = [f for f in result.files if f.endswith("_keypoints.csv")]
    rows = _rows(result.dest / table)
    assert [r["keypoint"] for r in rows] == ["nose", "tail_base"]
    assert float(rows[0]["x_norm"]) == pytest.approx(0.3)
    assert all(
        r["visible"] == "true" for r in rows
    )  # scored 1.0, above the 0.4 recorded


def test_an_old_skeleton_reads_in_the_rebuilds_shape():
    from detail.core.skeleton import Skeleton

    old = {
        "keypoints": [
            {"name": "nose", "side": "midline", "pair": None},
            {"name": "left_ear", "side": "left", "pair": "right_ear"},
            {"name": "right_ear", "side": "right", "pair": "left_ear"},
        ],
        "edges": [["nose", "left_ear"]],
        "front": "nose",
        "back": "nose",
    }
    skeleton = Skeleton.from_dict(old)
    assert skeleton.names == ("nose", "left_ear", "right_ear")
    assert skeleton.pairs == (("left_ear", "right_ear"),)
    assert skeleton.edges == (("nose", "left_ear"),)
    assert Skeleton.from_dict(skeleton.to_dict()) == skeleton  # already the new shape


@pytest.fixture
def legacy_aligned(legacy):
    """``legacy`` plus ``emb_old``: an embedding run the stage before §26 aligned to ``kp_01`` —
    its part carries rd's ``align_deg`` / ``align_mirrored`` columns, its spec rd's
    ``align_rotate`` step, and its record no heading definition."""
    from detail.storage.zarr_io import PartZarrWriter

    it = legacy
    recording = it.recording
    run_dir = recording.dir / "embeddings" / "emb_old"
    columns = (
        ("frame_idx", "int32", (10000,), ()),
        ("t_ns", "int64", (10000,), ()),
        ("det_row", "int32", (10000,), ()),
        ("embedding", "float32", (1000, 4), (4,)),
        ("align_deg", "float32", (10000,), ()),
        ("align_mirrored", "bool", (10000,), ()),
    )
    with PartZarrWriter(run_dir / "track_0" / "part_001.zarr", columns) as writer:
        writer._add_row(
            {
                "frame_idx": 0,
                "t_ns": 0,
                "det_row": 0,
                "embedding": np.ones(4, np.float32),
                "align_deg": 90.0,
                "align_mirrored": False,
            }
        )
    spec = [
        {"type": "crop_to_bbox"},
        {"type": "init_binary_mask"},
        {"type": "align_rotate"},
        {"type": "center_in_square", "size": 32},
    ]
    meta = {
        "schema_version": 1,
        "id": "emb_old",
        "tracking_run_id": "track_01",
        "model_id": "dinov3_vit",
        "output_dim": 4,
        "track_ids": [0],
        "renderer_spec": spec,
        "align": "rotate",
        "keypoint_run_id": "kp_01",
        "align_min_score": 0.4,
        "align_hold_s": 1.0,
        "status": "complete",
    }
    (run_dir / "embedding.yaml").write_text(
        yaml.dump(meta, sort_keys=False), encoding="utf-8"
    )
    data = recording_store.read_recording(recording.dir)
    data.setdefault("embedding_runs", []).append(
        {
            "id": "emb_old",
            "status": "complete",
            "output": "embeddings/emb_old/",
            "tracking_run_id": "track_01",
            "model_id": "dinov3_vit",
            "align": "rotate",
            "keypoint_run_id": "kp_01",
            "track_ids": [0],
        }
    )
    recording_store.write_recording(recording.dir, data)
    runs_store.write_run(
        it.experiment.dir,
        "run_emb",
        {
            "id": "run_emb",
            "module_id": "embedding",
            "recording_ids": [recording.id],
            "per_recording": {
                recording.id: {"run_id": "emb_old", "status": "complete"}
            },
            "status": "complete",
        },
    )
    it.embedding_meta = meta
    return it


def test_an_old_aligned_runs_preview_replays_the_turn_it_recorded(legacy_aligned):
    """§26 has no heading for a ``kp_NN`` run, and recomputing one would show a picture the
    run never embedded; the run recorded its turn on every row, so the preview replays it.
    """
    from detail.api.run_preview import InputImagePreview
    from tests import experiments

    it = legacy_aligned
    experiments.write_video(it.recording.dir / "video" / "part_001.mp4")
    preview = InputImagePreview.for_embedding_run(it.recording.dir, it.embedding_meta)
    assert preview.heading_for(0, "part_001", 0) == (90.0, False)
    assert preview.heading_for(0, "part_001", 3) is None  # no row: not embedded
    _frame, _rgb, image = preview.input_at(0, 0, 0)
    assert image is not None
