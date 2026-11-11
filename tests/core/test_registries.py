import pytest

from detail.core import registries

REGISTER_CASES = [
    ("register_model", "MODELS"),
    ("register_trainer", "TRAINERS"),
    ("register_tracker", "TRACKERS"),
    ("register_embedding_model", "EMBEDDING_MODELS"),
    ("register_keypoint_model", "KEYPOINT_MODELS"),
    ("register_keypoint_track_method", "KEYPOINT_TRACK_METHODS"),
    ("register_transform", "TRANSFORMS"),
]


@pytest.mark.parametrize("register_name,registry_name", REGISTER_CASES)
def test_decorator_registers_and_returns_class(register_name, registry_name):
    register = getattr(registries, register_name)
    registry = getattr(registries, registry_name)

    @register("fake_key")
    class Fake:
        pass

    try:
        assert registry["fake_key"] is Fake
    finally:
        del registry["fake_key"]


@pytest.mark.parametrize("_,registry_name", REGISTER_CASES)
def test_registry_is_dict(_, registry_name):
    assert isinstance(getattr(registries, registry_name), dict)


def test_importing_registries_registers_nothing():
    # Fresh interpreter: importing the registries module must not pull in any
    # concrete backend (those self-register only when their own module is
    # imported). Checked in a subprocess so other test modules' imports — e.g.
    # the IoUTracker — don't pollute the global registry under test.
    import subprocess
    import sys

    code = (
        "from detail.core import registries as r; "
        "assert not any([r.MODELS, r.TRAINERS, r.TRACKERS, "
        "r.EMBEDDING_MODELS, r.KEYPOINT_MODELS, r.KEYPOINT_TRACK_METHODS, r.TRANSFORMS])"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_history_reads_in_a_process_that_never_trained(tmp_path):
    """The GUI process's case, which this session cannot see: `conftest` imports every registry
    before any test runs, so `TRAINERS` is never empty here.

    A GUI that opens an existing model, or watches a distill whose training runs in its detached
    coordinator, never calls `train_model` — the one path that imported the trainers — so it read
    an empty registry, got ``[]``, and drew no training curve at all. A fresh interpreter is the
    only place that state exists.
    """
    import subprocess
    import sys

    from detail.api.experiment import Experiment
    from detail.storage import models as models_store

    exp = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(exp.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {"id": "model_001", "status": "running", "trainer_id": "ultralytics_yolo"},
    )
    (mdir / "run").mkdir()
    rows = ["epoch,metrics/mAP50(B),metrics/mAP50(M)", "1,0.40,0.35", "2,0.61,0.55"]
    (mdir / "run" / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    code = (
        "import sys; from pathlib import Path; "
        "from detail.api.experiment import Experiment; "
        "h = Experiment.open(Path(sys.argv[1])).model_metric_history('model_001'); "
        "assert [r['epoch'] for r in h] == [1, 2], h"
    )
    subprocess.run([sys.executable, "-c", code, str(exp.dir)], check=True, timeout=120)


# --- a model id is one namespace across the model registries (§5) ------------ #
def test_a_model_id_already_held_by_another_role_is_refused():
    """``MODELS.get(id) or EMBEDDING_MODELS.get(id)`` answered a duplicate with the detection
    model silently; the decorators now refuse at import, naming the role that holds the id.
    """

    @registries.register_model("one_namespace_stub")
    class Det:
        pass

    try:
        with pytest.raises(ValueError, match="already registered as a detection model"):

            @registries.register_embedding_model("one_namespace_stub")
            class Emb:
                pass

        assert "one_namespace_stub" not in registries.EMBEDDING_MODELS

        # re-registering under the *same* role replaces, as every registry always has
        @registries.register_model("one_namespace_stub")
        class Det2:
            pass

        assert registries.MODELS["one_namespace_stub"] is Det2
    finally:
        del registries.MODELS["one_namespace_stub"]


def test_model_registries_lists_every_model_role():
    roles = [role for role, _ in registries.MODEL_REGISTRIES]
    assert roles == ["detection", "embedding", "keypoint"]
    assert dict(registries.MODEL_REGISTRIES)["detection"] is registries.MODELS
    assert dict(registries.MODEL_REGISTRIES)["embedding"] is registries.EMBEDDING_MODELS
    assert dict(registries.MODEL_REGISTRIES)["keypoint"] is registries.KEYPOINT_MODELS
