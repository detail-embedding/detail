"""The detect form offers a trained keypoint student as a method (§12 *Distill keypoints*,
PR 8): it joins the picker once the experiment is set, needs no example set, and is launched
by its model id. The model catalog's keypoint models join it too (§12 *Keypoint models in the
catalog*), each labelled by where it comes from."""

from detail.api.experiment import Experiment
from detail.core.skeleton import Skeleton
from detail.gui.modules.registry import GUI_MODULES
from detail.storage import models as models_store


def _student(proj, model_id="model_001", name="rat student"):
    mdir = models_store.model_dir(proj.dir, model_id)
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "best.pt").write_bytes(b"")
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "name": name,
            "status": "complete",
            "task": "pose",
            "model_family": "yolo_pose",
            "weights": {"best": "run/weights/best.pt"},
            "skeleton": Skeleton(names=("nose", "tail_base")).to_dict(),
        },
    )


def test_a_trained_keypoint_student_is_offered_as_a_method(qapp, tmp_path):
    form = GUI_MODULES["keypoint_detect"].config_widget()
    proj = Experiment.create("p", tmp_path)
    _student(proj)
    assert form._method.findData("model_001") < 0

    form.on_experiment_set(proj)

    index = form._method.findData("model_001")
    assert index >= 0 and "rat student" in form._method.itemText(index)
    form._method.setCurrentIndex(index)
    assert form._example_picker.isHidden(), "a student needs no example set"
    assert form._examples_label.isHidden()
    assert form._variant.count() == 0
    assert form.config()["model_id"] == "model_001"
    assert form.config()["model_variant"] is None
    # Selectable is not enough: the form must also let a student run. It once answered
    # "No keypoint method is registered in this build" for every student.
    form._picker.spec = lambda: {"module_id": "detection"}
    assert form.validation_error() is None
    # A student reports a pose confidence (`TrainedYoloPose`, found through the one
    # lookup — it is not in KEYPOINT_MODELS), so its threshold row shows.
    assert not form._min_conf.isHidden() and not form._min_conf_label.isHidden()

    form.on_experiment_set(proj)  # re-attaching does not duplicate the entry
    offered = [form._method.itemData(i) for i in range(form._method.count())]
    assert offered.count("model_001") == 1
    assert form._method.currentData() == "model_001", "the selection survives a refresh"


def test_an_incomplete_or_detection_model_is_not_offered(qapp, tmp_path):
    form = GUI_MODULES["keypoint_detect"].config_widget()
    proj = Experiment.create("p", tmp_path)
    _student(proj, "model_001")
    mdir = models_store.model_dir(proj.dir, "model_002")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {"id": "model_002", "name": "seg", "status": "complete", "task": "segment"},
    )
    mdir = models_store.model_dir(proj.dir, "model_003")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir, {"id": "model_003", "name": "half", "status": "running", "task": "pose"}
    )

    form.on_experiment_set(proj)

    offered = {form._method.itemData(i) for i in range(form._method.count())}
    assert "model_001" in offered
    assert "model_002" not in offered and "model_003" not in offered


def _catalog_pose_model(model_id="rat_pose", name="rat catalog"):
    from detail.models import imported as imported_store

    mdir = imported_store.model_dir(model_id)
    (mdir / "weights").mkdir(parents=True)
    (mdir / "weights" / "best.pt").write_bytes(b"")
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "name": name,
            "source": "imported",
            "status": "complete",
            "role": "keypoints",
            "task": "pose",
            "model_family": "yolo_pose",
            "weights": {"best": "weights/best.pt"},
            "skeleton": Skeleton(names=("nose", "tail_base")).to_dict(),
        },
    )


def test_catalog_keypoint_models_are_offered_and_say_where_they_come_from(
    qapp, tmp_path
):
    form = GUI_MODULES["keypoint_detect"].config_widget()
    proj = Experiment.create("p", tmp_path)
    _student(proj)
    _catalog_pose_model()

    form.on_experiment_set(proj)

    trained = form._method.itemText(form._method.findData("model_001"))
    catalog = form._method.itemText(form._method.findData("rat_pose"))
    assert trained.endswith("· trained in this experiment")
    assert "rat catalog" in catalog and catalog.endswith("· from the model catalog")

    form._method.setCurrentIndex(form._method.findData("rat_pose"))
    # Through the one lookup (`keypoint_method_class`), not KEYPOINT_MODELS alone: a catalog
    # model is a student, so it needs no example set and reports a pose confidence.
    assert form._example_picker.isHidden()
    assert not form._min_conf.isHidden()
    assert form.config()["model_id"] == "rat_pose"
    form._picker.spec = lambda: {"module_id": "detection"}
    assert form.validation_error() is None


def test_a_pose_model_runs_where_no_registered_method_can(qapp, tmp_path, monkeypatch):
    """A catalog model is often restored on a machine without DINOv3 weights. The form said
    "No keypoint method can run here" and refused every launch, though a pose model needs no
    foundation model."""
    monkeypatch.setattr(
        "detail.gui.model_availability.unavailable_reason",
        lambda model_id: "needs dinov3_vit vit_large installed",
    )
    form = GUI_MODULES["keypoint_detect"].config_widget()
    form._picker.spec = lambda: {"module_id": "detection"}
    assert form.validation_error() is not None  # nothing to run yet
    proj = Experiment.create("p", tmp_path)
    _catalog_pose_model()

    form.on_experiment_set(proj)

    assert form._method.currentData() == "rat_pose"
    assert form.validation_error() is None
    assert form._no_methods.isHidden()
    assert all(form._method.itemData(i) for i in range(form._method.count()))
