"""resolve_detection_model: pretrained registry lookup + trained model.yaml load."""

import pytest

from detail.core.protocols import DetectionModel
from detail.core.registries import MODELS, TRAINED_LOADERS
from detail.models import imported as imported_store
from detail.models.resolve import (
    resolve_detection_model,
    resolve_detection_model_auto,
)
from detail.storage import models as models_store
from detail.storage import experiment as experiment_store


class _StubPretrained(DetectionModel):
    model_id = "stub_pretrained"
    supported_cue_types = []
    supported_backends = ["cpu"]

    def load(self, device): ...

    def predict(self, frame, prompt):
        raise NotImplementedError

    def predict_batch(self, frames, prompts):
        raise NotImplementedError


class _StubTrainedLoader(DetectionModel):
    model_id = "stub_trained_loader"
    supported_cue_types = []
    supported_backends = ["cpu"]

    def __init__(self, weights_path, classes, *, negative_class_ids=()):
        super().__init__(None)
        self.weights_path = weights_path
        self.classes = classes

    def load(self, device): ...

    def predict(self, frame, prompt):
        raise NotImplementedError

    def predict_batch(self, frames, prompts):
        raise NotImplementedError


def _write_trained_model(
    proj, model_id="model_001", family="stubfam", status="complete", best="run/best.pt"
):
    mdir = models_store.model_dir(proj, model_id)
    (mdir / "run").mkdir(parents=True)
    if best is not None:
        (mdir / best).write_bytes(b"weights")
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "model_family": family,
            "status": status,
            "classes": [{"id": 0, "name": "mouse"}, {"id": 1, "name": "pellet"}],
            "weights": {"best": best, "last": "run/last.pt"},
        },
    )
    return mdir


# --- pretrained ------------------------------------------------------------- #
def test_pretrained_resolves_from_registry(tmp_path, monkeypatch):
    monkeypatch.setitem(MODELS, "stub_pretrained", _StubPretrained)
    model = resolve_detection_model(tmp_path, "stub_pretrained", "pretrained")
    assert isinstance(model, _StubPretrained)


def test_pretrained_is_the_default_source(tmp_path, monkeypatch):
    monkeypatch.setitem(MODELS, "stub_pretrained", _StubPretrained)
    assert isinstance(
        resolve_detection_model(tmp_path, "stub_pretrained"), _StubPretrained
    )


def test_pretrained_unknown_id_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown model_id"):
        resolve_detection_model(tmp_path, "nope", "pretrained")


def test_unknown_weights_source_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown weights_source"):
        resolve_detection_model(tmp_path, "x", "borrowed")


# --- trained ---------------------------------------------------------------- #
def test_trained_resolves_with_weights_path_and_classes(tmp_path, monkeypatch):
    proj = experiment_store.create_experiment(tmp_path, "p")
    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubTrainedLoader)
    mdir = _write_trained_model(proj)
    model = resolve_detection_model(proj, "model_001", "trained")
    assert isinstance(model, _StubTrainedLoader)
    assert model.weights_path == mdir / "run/best.pt"
    assert model.classes == ["mouse", "pellet"]  # names, in id order


def test_trained_missing_model_yaml_raises(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    with pytest.raises(FileNotFoundError):
        resolve_detection_model(proj, "model_404", "trained")


def test_trained_incomplete_model_raises(tmp_path, monkeypatch):
    proj = experiment_store.create_experiment(tmp_path, "p")
    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubTrainedLoader)
    _write_trained_model(proj, status="running")
    with pytest.raises(ValueError, match="not complete"):
        resolve_detection_model(proj, "model_001", "trained")


def test_trained_unknown_family_raises(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    _write_trained_model(proj, family="unregistered_family")
    with pytest.raises(ValueError, match="no trained-model loader"):
        resolve_detection_model(proj, "model_001", "trained")


def test_trained_without_best_weights_raises(tmp_path, monkeypatch):
    proj = experiment_store.create_experiment(tmp_path, "p")
    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubTrainedLoader)
    _write_trained_model(proj, best=None)
    with pytest.raises(ValueError, match="no best-weights"):
        resolve_detection_model(proj, "model_001", "trained")


# --- auto source resolution (--model <id>, source unknown) ------------------ #
def _write_imported_model(model_id="ext_model", family="stubfam"):
    """Write a model into the (per-test isolated) machine-global catalog."""
    mdir = imported_store.model_dir(model_id)
    (mdir / "weights").mkdir(parents=True)
    (mdir / "weights/best.pt").write_bytes(b"weights")
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "model_family": family,
            "status": "complete",
            "classes": [{"id": 0, "name": "mouse"}],
            "weights": {"best": "weights/best.pt"},
        },
    )


def test_auto_resolves_pretrained(tmp_path, monkeypatch):
    monkeypatch.setitem(MODELS, "stub_pretrained", _StubPretrained)
    model, src = resolve_detection_model_auto(tmp_path, "stub_pretrained")
    assert isinstance(model, _StubPretrained) and src == "pretrained"


def test_auto_resolves_experiment_trained(tmp_path, monkeypatch):
    proj = experiment_store.create_experiment(tmp_path, "p")
    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubTrainedLoader)
    _write_trained_model(proj)
    model, src = resolve_detection_model_auto(proj, "model_001")
    assert isinstance(model, _StubTrainedLoader) and src == "trained"


def test_auto_resolves_imported(tmp_path, monkeypatch):
    proj = experiment_store.create_experiment(tmp_path, "p")
    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubTrainedLoader)
    _write_imported_model("ext_model")
    model, src = resolve_detection_model_auto(proj, "ext_model")
    assert isinstance(model, _StubTrainedLoader) and src == "imported"
    assert model.classes == ["mouse"]


def test_auto_unknown_model_raises(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    with pytest.raises(ValueError, match="unknown model"):
        resolve_detection_model_auto(proj, "nope")


# --- a trained keypoint student is a method too (§12 *Distill keypoints*) ------------- #
def _pose_student(experiment_dir, model_id="model_001"):
    from detail.storage import models as models_store

    mdir = models_store.model_dir(experiment_dir, model_id)
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir, {"id": model_id, "name": "s", "status": "complete", "task": "pose"}
    )


def test_keypoint_method_class_knows_a_student_as_well_as_a_registered_method(tmp_path):
    """Every site that needs a keypoint method's class-level facts (its score, its default
    track method) asks this one function; a student is not in KEYPOINT_MODELS, and the sites
    that asked the registry alone read a student's runs as having no cut-off at all."""
    from types import SimpleNamespace

    from detail.api.keypoint_export import default_cutoff
    from detail.core.registries import KEYPOINT_MODELS
    from detail.models.resolve import keypoint_method_class
    from detail.models.yolo_pose import TrainedYoloPose
    from detail.pipeline.embedding import _heading_cutoff

    experiment_dir = tmp_path / "exp"
    _pose_student(experiment_dir)
    registered = next(iter(KEYPOINT_MODELS))
    assert (
        keypoint_method_class(experiment_dir, registered) is KEYPOINT_MODELS[registered]
    )
    assert keypoint_method_class(experiment_dir, "model_001") is TrainedYoloPose
    assert keypoint_method_class(experiment_dir, "model_404") is None
    assert keypoint_method_class(None, "model_001") is None
    assert keypoint_method_class(experiment_dir, None) is None

    # A detect run lives at <experiment>/recordings/<rec>/keypoint_detect/<kd>.
    detect = SimpleNamespace(
        recording_dir=experiment_dir / "recordings" / "recording_001",
        meta={"model_id": "model_001"},
    )
    student_cutoff = TrainedYoloPose.score_spec.default_cutoff
    assert default_cutoff(detect) == student_cutoff, "the export's display cut-off"
    track = SimpleNamespace(meta={"config": {}})
    assert _heading_cutoff(track, detect) == student_cutoff, "the heading's cut-off"


# --- pose models are kept out of detection (§12 *Keypoint models in the catalog*) ----- #
def _pose_record(mdir, model_id, family):
    (mdir / "weights").mkdir(parents=True)
    (mdir / "weights/best.pt").write_bytes(b"weights")
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "model_family": family,
            "task": "pose",
            "status": "complete",
            "classes": [{"id": 0, "name": "rat"}],
            "weights": {"best": "weights/best.pt"},
        },
    )


@pytest.mark.parametrize(
    "family",
    [
        "yolo_pose",  # a promoted student: no detection loader for its family at all
        "stubfam",  # an imported pose .pt read as family yolo: a loader that would build it
    ],
)
def test_detection_refuses_a_pose_record_by_name(tmp_path, monkeypatch, family):
    """A pose model built as a detector either failed with "no trained-model loader" or —
    for an imported pose .pt, whose family is the detector's — was built as a detector and
    failed at predict. Every detection entry point refuses it and names keypoint detection.
    """
    from detail.models.resolve import resolve_pretrained_or_imported

    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubTrainedLoader)
    proj = experiment_store.create_experiment(tmp_path, "p")
    _pose_record(models_store.model_dir(proj, "model_001"), "model_001", family)
    _pose_record(imported_store.model_dir("rat_pose"), "rat_pose", family)

    message = "keypoint .pose. model.*keypoint detection"
    with pytest.raises(ValueError, match=message):
        resolve_detection_model_auto(proj, "model_001")
    with pytest.raises(ValueError, match=message):
        resolve_detection_model_auto(proj, "rat_pose")
    with pytest.raises(ValueError, match=message):
        resolve_detection_model(proj, "model_001", "trained")
    with pytest.raises(ValueError, match=message):
        resolve_detection_model(proj, "rat_pose", "imported")
    with pytest.raises(ValueError, match=message):
        resolve_pretrained_or_imported("rat_pose")
