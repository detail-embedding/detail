"""Resolving a distilled keypoint student (§12 *Distill keypoints*, PR 8): the resolver's
catalog branch, beside ``KEYPOINT_MODELS``."""

import pytest

from detail.core.skeleton import Skeleton
from detail.models.resolve import (
    keypoint_students,
    resolve_keypoint_model,
)
from detail.models.yolo_pose import TrainedYoloPose
from detail.storage import models as models_store

SK = Skeleton(names=("nose", "tail_base"), edges=(("nose", "tail_base"),))


def _student_record(experiment_dir, model_id):
    """The pose record the keypoint lookup finds for ``model_id``, or None."""
    from detail.models.resolve import _find_keypoint_student

    hit = _find_keypoint_student(experiment_dir, model_id)
    return None if hit is None else hit[0]


def _pose_student(root, model_id="model_001", status="complete", task="pose"):
    (root / "models").mkdir(parents=True, exist_ok=True)
    mdir = models_store.model_dir(root, model_id)
    (mdir / "run" / "weights").mkdir(parents=True, exist_ok=True)
    (mdir / "run" / "weights" / "best.pt").write_bytes(b"")
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "name": "student",
            "status": status,
            "task": task,
            "model_family": "yolo_pose",
            "weights": {"best": "run/weights/best.pt"},
            "skeleton": SK.to_dict(),
        },
    )
    return root


def test_a_pose_student_resolves_from_its_record(tmp_path):
    exp = _pose_student(tmp_path / "exp")

    model = resolve_keypoint_model("model_001", None, experiment_dir=exp)

    assert isinstance(model, TrainedYoloPose)
    assert model.skeleton() == SK
    assert model._weights_path == (
        models_store.model_dir(exp, "model_001") / "run/weights/best.pt"
    )
    assert [s["id"] for s in keypoint_students(exp)] == ["model_001"]
    assert _student_record(exp, "model_001")["name"] == "student"
    assert _student_record(exp, "model_009") is None


def test_the_matchers_settings_are_ignored_and_an_example_set_refused(tmp_path):
    exp = _pose_student(tmp_path / "exp")
    model = resolve_keypoint_model(
        "model_001",
        None,
        experiment_dir=exp,
        decode="skeleton",
        side_rule="auto",
        crop_px=448,
        examples=None,
        example_images=None,
    )
    assert isinstance(model, TrainedYoloPose)
    with pytest.raises(ValueError, match="takes no example set"):
        resolve_keypoint_model("model_001", None, experiment_dir=exp, examples=object())


def test_an_unknown_id_names_the_students_and_only_complete_pose_ones_resolve(tmp_path):
    exp = _pose_student(tmp_path / "exp")
    with pytest.raises(ValueError, match=r"unknown keypoint method 'nope'.*model_001"):
        resolve_keypoint_model("nope", None, experiment_dir=exp)
    with pytest.raises(ValueError, match="unknown keypoint method 'nope'"):
        resolve_keypoint_model("nope", None)

    running = _pose_student(tmp_path / "b", status="running")
    assert keypoint_students(running) == []
    with pytest.raises(ValueError, match="not complete"):
        resolve_keypoint_model("model_001", None, experiment_dir=running)

    segment = _pose_student(tmp_path / "c", task="segment")
    assert _student_record(segment, "model_001") is None
    with pytest.raises(ValueError, match="unknown keypoint method"):
        resolve_keypoint_model("model_001", None, experiment_dir=segment)


# --- the machine catalog (§12 *Keypoint models in the catalog*) ----------------------- #
def _catalog_pose(model_id="rat_pose", status="complete", name="catalog student"):
    """A pose record in the (per-test isolated) machine catalog, the shape a promote,
    a restored zip or an imported pose ``.pt`` leaves."""
    from detail.models import imported as imported_store

    mdir = imported_store.model_dir(model_id)
    (mdir / "weights").mkdir(parents=True, exist_ok=True)
    (mdir / "weights" / "best.pt").write_bytes(b"")
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "name": name,
            "status": status,
            "role": "keypoints",
            "task": "pose",
            "model_family": "yolo_pose",
            "weights": {"best": "weights/best.pt"},
            "skeleton": SK.to_dict(),
        },
    )
    return mdir


def test_a_catalog_pose_model_resolves_from_any_experiment(tmp_path):
    from detail.models.resolve import (
        keypoint_method_class,
        resolve_keypoint_weights_source,
    )

    mdir = _catalog_pose()
    exp = tmp_path / "exp"
    exp.mkdir()

    model = resolve_keypoint_model("rat_pose", None, experiment_dir=exp)

    assert isinstance(model, TrainedYoloPose)
    assert model._weights_path == mdir / "weights/best.pt"
    assert model.skeleton() == SK
    assert keypoint_method_class(exp, "rat_pose") is TrainedYoloPose
    assert keypoint_method_class(None, "rat_pose") is TrainedYoloPose
    assert _student_record(exp, "rat_pose")["name"] == "catalog student"
    assert resolve_keypoint_weights_source(exp, "rat_pose") == "imported"
    assert [(s["id"], s["source"]) for s in keypoint_students(exp)] == [
        ("rat_pose", "imported")
    ]


def test_students_rows_carry_their_source_experiment_first(tmp_path):
    exp = _pose_student(tmp_path / "exp")
    _catalog_pose()
    _catalog_pose("half_done", status="running")

    rows = keypoint_students(exp)

    assert [(s["id"], s["source"]) for s in rows] == [
        ("model_001", "trained"),
        ("rat_pose", "imported"),
    ]


def test_the_experiment_shadows_the_catalog_on_an_id_clash_unless_a_source_is_recorded(
    tmp_path,
):
    """An experiment model and a catalog model can share an id (§12): the lookup takes the
    experiment's, the picker offers the one the id reaches, and a run that recorded
    ``weights_source`` rebuilds exactly that one."""
    from detail.models.resolve import resolve_keypoint_weights_source

    exp = _pose_student(tmp_path / "exp")
    catalog = _catalog_pose("model_001", name="catalog twin")
    experiment_weights = (
        models_store.model_dir(exp, "model_001") / "run/weights/best.pt"
    )

    assert resolve_keypoint_weights_source(exp, "model_001") == "trained"
    assert _student_record(exp, "model_001")["name"] == "student"
    assert (
        resolve_keypoint_model("model_001", None, experiment_dir=exp)._weights_path
        == experiment_weights
    )
    assert [(s["id"], s["source"]) for s in keypoint_students(exp)] == [
        ("model_001", "trained")
    ]
    recorded = resolve_keypoint_model(
        "model_001", None, experiment_dir=exp, weights_source="imported"
    )
    assert recorded._weights_path == catalog / "weights/best.pt"
    assert (
        resolve_keypoint_model(
            "model_001", None, experiment_dir=exp, weights_source="trained"
        )._weights_path
        == experiment_weights
    )


def test_a_recorded_source_that_no_longer_holds_the_model_is_named(tmp_path):
    from detail.models.resolve import resolve_keypoint_weights_source

    exp = _pose_student(tmp_path / "exp")
    with pytest.raises(ValueError, match="'model_001'.*not in the model catalog"):
        resolve_keypoint_model(
            "model_001", None, experiment_dir=exp, weights_source="imported"
        )
    with pytest.raises(ValueError, match="unknown keypoint weights_source 'elsewhere'"):
        resolve_keypoint_model(
            "model_001", None, experiment_dir=exp, weights_source="elsewhere"
        )
    with pytest.raises(ValueError, match="unknown keypoint method 'nope'"):
        resolve_keypoint_weights_source(exp, "nope")
    from detail.core.registries import KEYPOINT_MODELS

    assert resolve_keypoint_weights_source(exp, next(iter(KEYPOINT_MODELS))) is None
