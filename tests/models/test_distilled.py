"""Tests for promoting a trained model to the machine-global catalog (§12)."""

import pytest

from detail.api import models as api_models
from detail.api.experiment import Experiment
from detail.models import distilled
from detail.models import imported as imported_store
from detail.storage import datasets as datasets_store
from detail.storage import experiment as experiment_store
from detail.storage import models as models_store


def _trained_model(
    experiment_dir, model_id="model_001", status="complete", name="mouse_seg"
):
    """Write a completed experiment-trained ``model.yaml`` + a best-weights file."""
    mdir = models_store.model_dir(experiment_dir, model_id)
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "best.pt").write_bytes(b"weights-bytes")
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "name": name,
            "trainer_id": "ultralytics_yolo",
            "dataset_id": "dataset_003",
            "base_model": "yolo11s-seg.pt",
            "model_family": "yolo",
            "task": "segment",
            "classes": [{"id": 0, "name": "mouse"}],
            "weights": {"best": "run/weights/best.pt", "last": "run/weights/last.pt"},
            "metrics": {"box_map50": 0.92, "mask_map50": 0.89},
            "status": status,
        },
    )
    return mdir


def test_promote_writes_provenance_and_copies_weights(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "rotation_pilot")
    _trained_model(proj)
    gid = distilled.register_distilled_model(
        proj,
        "model_001",
        source_experiment_name="rotation_pilot",
        source_experiment_id="rotation_pilot",
    )
    rec = imported_store.read(gid)
    assert rec["source"] == "distilled"
    assert rec["model_family"] == "yolo"
    assert rec["classes"] == [{"id": 0, "name": "mouse"}]
    assert rec["weights"]["best"] == imported_store.WEIGHTS_RELPATH
    # self-contained: the weights were copied into the global catalog dir
    assert (
        imported_store.model_dir(gid) / imported_store.WEIGHTS_RELPATH
    ).read_bytes() == b"weights-bytes"
    assert rec["created_at"]  # stamped on promote
    prov = rec["provenance"]
    # a list so one model can name several sources later; distill writes one entry today
    assert prov["source_experiments"] == [
        {"name": "rotation_pilot", "id": "rotation_pilot"}
    ]
    assert prov["source_dataset_id"] == "dataset_003"
    assert prov["base_model"] == "yolo11s-seg.pt"
    assert prov["trainer_id"] == "ultralytics_yolo"
    assert prov["metrics"]["box_map50"] == 0.92
    assert "distilled_at" in prov


def test_promote_records_no_absolute_source_path(tmp_path):
    # Tenet 6: provenance is metadata only — no path back into the source experiment.
    proj = experiment_store.create_experiment(tmp_path, "rp")
    _trained_model(proj)
    gid = distilled.register_distilled_model(
        proj, "model_001", source_experiment_name="rp", source_experiment_id="rp"
    )
    text = (imported_store.model_dir(gid) / models_store.MODEL_FILENAME).read_text(
        encoding="utf-8-sig"
    )
    assert str(tmp_path) not in text


def test_promote_rejects_incomplete_model(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    _trained_model(proj, status="partial")
    with pytest.raises(ValueError, match="only a complete model"):
        distilled.register_distilled_model(
            proj, "model_001", source_experiment_name="p", source_experiment_id="p"
        )


def test_promote_missing_model_raises(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    with pytest.raises(FileNotFoundError):
        distilled.register_distilled_model(
            proj, "model_999", source_experiment_name="p", source_experiment_id="p"
        )


def test_promote_rejects_id_collision(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    _trained_model(proj, name="dup")
    distilled.register_distilled_model(
        proj,
        "model_001",
        source_experiment_name="p",
        source_experiment_id="p",
        name="dup",
    )
    _trained_model(proj, model_id="model_002", name="dup")
    with pytest.raises(ValueError, match="already exists"):
        distilled.register_distilled_model(
            proj,
            "model_002",
            source_experiment_name="p",
            source_experiment_id="p",
            name="dup",
        )


def test_promote_rejects_reserved_id_shape(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    _trained_model(proj)
    with pytest.raises(ValueError, match="reserved"):
        distilled.register_distilled_model(
            proj,
            "model_001",
            source_experiment_name="p",
            source_experiment_id="p",
            name="model_050",
        )


def test_catalog_marks_distilled_with_provenance(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "rp")
    _trained_model(proj)
    gid = distilled.register_distilled_model(
        proj, "model_001", source_experiment_name="rp", source_experiment_id="rp"
    )
    rows = {r.id: r for r in api_models.list_models()}
    assert rows[gid].source == "distilled"
    assert rows[gid].provenance["source_dataset_id"] == "dataset_003"
    refs = {r.id: r for r in api_models.imported_models(role=None)}
    assert refs[gid].source == "distilled"
    assert refs[gid].provenance["base_model"] == "yolo11s-seg.pt"


def test_promote_carries_the_preview(tmp_path):
    # An auto-captured overlay preview (WS2 writes it into the trained model dir) travels
    # into the promoted global model so the catalog shows the same thumbnail.
    proj = experiment_store.create_experiment(tmp_path, "rp")
    mdir = _trained_model(proj)
    (mdir / "preview.png").write_bytes(b"\x89PNG shot")
    rec = models_store.read_model_yaml(mdir)
    rec["preview"] = "preview.png"
    models_store.write_model_yaml(mdir, rec)
    gid = distilled.register_distilled_model(
        proj, "model_001", source_experiment_name="rp", source_experiment_id="rp"
    )
    grec = imported_store.read(gid)
    assert grec["preview"] == "preview.png"
    assert (
        imported_store.model_dir(gid) / "preview.png"
    ).read_bytes() == b"\x89PNG shot"


def test_experiment_promote_model_wraps_register(tmp_path):
    proj = Experiment.create("rotation_pilot", tmp_path)
    _trained_model(proj.dir)
    gid = proj.promote_model("model_001")
    rec = imported_store.read(gid)
    assert rec["source"] == "distilled"
    assert rec["provenance"]["source_experiments"][0]["id"] == proj.dir.name


def _dataset(experiment_dir, dataset_id="dataset_003"):
    """Write a completed ``dataset.yaml`` with the labeling composition (§12)."""
    ddir = datasets_store.dataset_dir(experiment_dir, dataset_id)
    ddir.mkdir(parents=True)
    datasets_store.write_dataset_yaml(
        ddir,
        {
            "id": dataset_id,
            "name": "mouse_ds",
            "created_at": "2026-08-07T18:00:00Z",
            "source": {
                "kind": "labeling",
                "model": "sam3",
                "model_variant": None,
                "recording_ids": ["recording_001"],
                "n_frames": 500,
                "logit_threshold": 0.5,
            },
            "classes": [{"id": 0, "name": "mouse"}],
            "val_split": 0.15,
            "n_train": 425,
            "n_val": 75,
            "per_class": {"0": 500},
            "status": "complete",
        },
    )
    return ddir


def test_promote_captures_dataset_composition(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "rp")
    _trained_model(proj)
    _dataset(proj)
    gid = distilled.register_distilled_model(
        proj, "model_001", source_experiment_name="rp", source_experiment_id="rp"
    )
    ds = imported_store.read(gid)["provenance"]["dataset"]
    assert ds["model"] == "sam3"
    assert ds["n_frames"] == 500
    assert ds["n_train"] == 425 and ds["n_val"] == 75
    assert ds["per_class"] == {"0": 500}


def test_promote_omits_dataset_block_when_source_dataset_is_gone(tmp_path):
    # The dataset can be deleted independently of the model it produced — provenance is
    # best-effort, so this must not block promoting weights that already exist.
    proj = experiment_store.create_experiment(tmp_path, "rp")
    _trained_model(proj)  # dataset_id="dataset_003", but no dataset.yaml written
    gid = distilled.register_distilled_model(
        proj, "model_001", source_experiment_name="rp", source_experiment_id="rp"
    )
    prov = imported_store.read(gid)["provenance"]
    assert "dataset" not in prov
    assert (
        prov["source_dataset_id"] == "dataset_003"
    )  # the id itself still carries over


def test_promote_captures_training_history(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "rp")
    _trained_model(proj)
    history = [
        {"epoch": 1, "box_map50": 0.3, "mask_map50": 0.25},
        {"epoch": 60, "box_map50": 0.92, "mask_map50": 0.89},
    ]
    gid = distilled.register_distilled_model(
        proj,
        "model_001",
        source_experiment_name="rp",
        source_experiment_id="rp",
        training_history=history,
    )
    assert imported_store.read(gid)["provenance"]["training_history"] == history


def test_promote_omits_training_history_when_none_given(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "rp")
    _trained_model(proj)
    gid = distilled.register_distilled_model(
        proj, "model_001", source_experiment_name="rp", source_experiment_id="rp"
    )
    assert "training_history" not in imported_store.read(gid)["provenance"]


def test_experiment_promote_model_passes_metric_history(tmp_path, monkeypatch):
    proj = Experiment.create("rotation_pilot", tmp_path)
    _trained_model(proj.dir)
    history = [{"epoch": 1, "box_map50": 0.5}]
    monkeypatch.setattr(
        Experiment, "model_metric_history", lambda self, model_id: history
    )
    gid = proj.promote_model("model_001")
    assert imported_store.read(gid)["provenance"]["training_history"] == history


# --- a keypoint student (§12 *Keypoint models in the catalog*) --------------------- #
POSE_SKELETON = {
    "names": ["nose", "left_ear", "right_ear", "tail_base"],
    "edges": [["nose", "left_ear"], ["nose", "right_ear"], ["nose", "tail_base"]],
    "pairs": [["left_ear", "right_ear"]],
}


def _pose_student(experiment_dir, model_id="model_002", **overrides):
    """A registered keypoint student's ``model.yaml``, as ``register_student`` leaves it."""
    mdir = _trained_model(
        experiment_dir, model_id=model_id, name="rat pose distill_001"
    )
    record = models_store.read_model_yaml(mdir)
    record.update(
        role="keypoints",
        task="pose",
        model_family="yolo_pose",
        classes=[{"id": 0, "name": "animal"}],
        skeleton=POSE_SKELETON,
        skeleton_ref={"id": "sk_07", "name": "rat overhead", "digest": "x"},
        weights_sha256="stale-on-purpose",
        teacher={"kind": "keypoint_detect", "distill_run_id": "distill_001"},
    )
    record.update(overrides)
    models_store.write_model_yaml(
        mdir, {k: v for k, v in record.items() if v is not None}
    )
    return mdir


def test_promoting_a_keypoint_student_carries_what_rebuilds_it(tmp_path):
    import hashlib

    from detail.core.skeleton import Skeleton
    from detail.storage import skeleton_library

    proj = experiment_store.create_experiment(tmp_path, "rotation_pilot")
    _pose_student(proj)
    gid = distilled.register_distilled_model(
        proj,
        "model_002",
        source_experiment_name="rotation_pilot",
        source_experiment_id="rotation_pilot",
    )
    rec = imported_store.read(gid)
    assert imported_store.is_pose_record(rec)
    assert rec["role"] == "keypoints" and rec["model_family"] == "yolo_pose"
    assert rec["skeleton"] == POSE_SKELETON
    digest = Skeleton.from_dict(POSE_SKELETON).digest()
    assert rec["skeleton_digest"] == digest
    # the library entry it was added to on this machine, under the student's ref name
    assert rec["skeleton_ref"]["digest"] == digest
    assert rec["skeleton_ref"]["name"] == "rat overhead"
    assert skeleton_library.read(rec["skeleton_ref"]["id"])["skeleton"] == POSE_SKELETON
    # the sha256 of the bytes the catalog holds, not whatever the student recorded
    assert rec["weights_sha256"] == hashlib.sha256(b"weights-bytes").hexdigest()
    assert rec["provenance"]["teacher"] == {
        "kind": "keypoint_detect",
        "distill_run_id": "distill_001",
    }
    # and the catalog row says keypoint, not detection
    row = {r.id: r for r in api_models.list_models()}[gid]
    assert row.role == "keypoint" and row.capabilities == ["keypoint"]


def test_promoting_the_same_skeleton_twice_adds_one_library_entry(tmp_path):
    from detail.storage import skeleton_library

    proj = experiment_store.create_experiment(tmp_path, "rotation_pilot")
    _pose_student(proj)
    kwargs = dict(source_experiment_name="rp", source_experiment_id="rp")
    first = distilled.register_distilled_model(proj, "model_002", name="a", **kwargs)
    second = distilled.register_distilled_model(proj, "model_002", name="b", **kwargs)
    assert len(skeleton_library.discover()) == 1
    assert (
        imported_store.read(first)["skeleton_ref"]
        == imported_store.read(second)["skeleton_ref"]
    )


def test_a_keypoint_student_without_a_skeleton_is_refused(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "rotation_pilot")
    _pose_student(proj, skeleton=None)
    with pytest.raises(ValueError, match="has no skeleton in its model.yaml"):
        distilled.register_distilled_model(
            proj, "model_002", source_experiment_name="rp", source_experiment_id="rp"
        )
    assert (
        imported_store.discover(role=None) == []
    )  # refused before anything was copied


def test_a_detection_promote_adds_no_pose_fields(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "rotation_pilot")
    _trained_model(proj)
    gid = distilled.register_distilled_model(
        proj, "model_001", source_experiment_name="rp", source_experiment_id="rp"
    )
    rec = imported_store.read(gid)
    assert not imported_store.is_pose_record(rec)
    assert "skeleton" not in rec and "role" not in rec
