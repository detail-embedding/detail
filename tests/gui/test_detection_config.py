"""Detection config form: one model list, each row saying whether it can run here."""

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.core.registries import MODELS  # noqa: E402
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui.modules.registry import GUI_MODULES  # noqa: E402
from detail.storage import models as models_store  # noqa: E402


def _trained_model(experiment, model_id="model_001", classes=None):
    """A complete trained model with a real class list — unlike ``closed_set_model``
    (a built-in registry stub, whose ``entry["classes"]`` is always ``[]``), this is
    the shape that actually populates the per-class threshold rows."""
    mdir = models_store.model_dir(experiment.dir, model_id)
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "name": "m",
            "status": "complete",
            "model_family": "yolo",
            "classes": classes
            or [{"id": 0, "name": "black mouse"}, {"id": 1, "name": "white mouse"}],
            "weights": {"best": "run/best.pt", "last": None},
        },
    )


@pytest.fixture
def closed_set_model(monkeypatch):
    """A registered detection model that takes no text prompt, like a trained YOLO.

    Scoped to the test via ``monkeypatch`` so it never leaks into the real catalog.
    """
    from detail.core.dataclasses import FrameResult
    from detail.core.protocols import DetectionModel

    class _ClosedSet(DetectionModel):
        model_id = "closed_set_stub"
        architecture = "yolo"
        supported_cue_types = []  # closed-set: it detects its trained classes
        supported_backends = ["cpu", "mps", "cuda"]
        supports_batching = True

        def load(self, device):
            pass

        def predict(self, frame, prompt):
            return FrameResult.from_detections(frame_idx=0, t_ns=0, detections=[])

    monkeypatch.setitem(MODELS, "closed_set_stub", _ClosedSet)
    return "closed_set_stub"


def _select(widget, model_id: str) -> None:
    """Pick a model by id. A label can carry its reason as a suffix, so it is not the id."""
    for index in range(widget._model.count()):
        data = widget._model.itemData(index)
        if data and data["id"] == model_id:
            widget._model.setCurrentIndex(index)
            return
    raise AssertionError(f"{model_id} not offered; rows: {_rows(widget)}")


def _rows(widget) -> list[str]:
    return [widget._model.itemText(i) for i in range(widget._model.count())]


def _backends(widget) -> list[str]:
    return [
        widget._compute.backend.itemText(i)
        for i in range(widget._compute.backend.count())
    ]


def test_the_test_button_follows_the_model_not_a_mode_switch(
    qapp, tmp_path, closed_set_model
):
    """It used to key off a *Weights source* toggle. A prompted model gets the prompt
    tester, a closed-set one gets the detection preview — the model's own property."""
    widget = GUI_MODULES["detection"].config_widget()
    widget.on_experiment_set(Experiment.create("p", tmp_path))

    _select(widget, "sam3")  # open-vocab
    assert widget._test_btn.isEnabled()
    assert widget._test_btn.text() == "Test prompts"

    _select(widget, closed_set_model)  # no prompt — previews its own classes instead
    assert widget._test_btn.isEnabled()
    assert widget._test_btn.text() == "Test detection"


def test_prompt_rows_show_only_for_a_prompted_model(qapp, tmp_path, closed_set_model):
    widget = GUI_MODULES["detection"].config_widget()
    widget.on_experiment_set(Experiment.create("p", tmp_path))
    _select(widget, "sam3")
    assert widget._prompt.toolTip() and widget._class.toolTip()
    assert "what the model searches for" in widget._prompt_note.text()
    assert not widget._prompt.isHidden()

    _select(widget, closed_set_model)  # closed-set: no prompt, its classes instead
    assert widget._prompt.isHidden()


def test_load_config_restores_which_model_was_used(qapp, tmp_path):
    """Refine/the settings picker must reopen on the *run's own* model, not whatever
    _rebuild_models defaults to. Each combo row's data is the whole entry dict (kind,
    classes, variants alongside the id), so a bare ``self._model.findData(model_id)``
    against a plain id string never matches and silently no-ops -- caught here because
    a fresh widget's default (the first built-in, alphabetically) is never
    "model_001"."""
    experiment = Experiment.create("p", tmp_path)
    _trained_model(experiment)
    widget = GUI_MODULES["detection"].config_widget()
    widget.on_experiment_set(experiment)

    widget.load_config({"model_id": "model_001"})

    assert widget.config()["model_id"] == "model_001"


def test_per_class_conf_rows_appear_only_for_a_multi_class_trained_model(
    qapp, tmp_path, closed_set_model
):
    experiment = Experiment.create("p", tmp_path)
    _trained_model(experiment)  # black mouse, white mouse
    widget = GUI_MODULES["detection"].config_widget()
    widget.on_experiment_set(experiment)

    _select(widget, "sam3")  # prompt-capable: nothing to override per-class
    assert widget._class_conf_container.isHidden()

    _select(widget, closed_set_model)  # closed-set, but this stub has no classes
    assert widget._class_conf_container.isHidden()
    assert widget._class_conf_rows == {}

    _select(widget, "model_001")  # closed-set, two known classes
    assert not widget._class_conf_container.isHidden()
    assert set(widget._class_conf_rows) == {"black mouse", "white mouse"}


def test_per_class_conf_rows_seed_to_the_global_default(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    _trained_model(experiment)
    widget = GUI_MODULES["detection"].config_widget()
    widget.on_experiment_set(experiment)
    widget._conf.setValue(0.42)

    _select(widget, "model_001")

    assert widget._class_conf_rows["black mouse"].value() == pytest.approx(0.42)
    assert widget._class_conf_rows["white mouse"].value() == pytest.approx(0.42)


def test_config_round_trips_class_conf_thresholds_through_load_config(qapp, tmp_path):
    experiment = Experiment.create("p", tmp_path)
    _trained_model(experiment)
    widget = GUI_MODULES["detection"].config_widget()
    widget.on_experiment_set(experiment)
    _select(widget, "model_001")
    widget._class_conf_rows["black mouse"].setValue(0.7)

    emitted = widget.config()
    assert emitted["class_conf_thresholds"] == {
        "black mouse": pytest.approx(0.7),
        "white mouse": pytest.approx(0.5),
    }

    restored = GUI_MODULES["detection"].config_widget()
    restored.on_experiment_set(experiment)
    restored.load_config(emitted)
    assert restored._class_conf_rows["black mouse"].value() == pytest.approx(0.7)
    assert restored._class_conf_rows["white mouse"].value() == pytest.approx(0.5)


def test_the_backend_combo_never_offers_a_device_this_machine_lacks(
    qapp, monkeypatch, real_machine
):
    """A Mac (mps, no cuda). SAM 3 is cuda-only.

    This used to assert the opposite — that the combo showed ONLY cuda, mirroring the
    model's own support list. That is the bug: `setCurrentText` on a combo holding one
    unavailable entry is a no-op, so the form sat on `cuda` with its device box live and
    the run died on `requested cuda device(s) [0] not available`.
    """
    monkeypatch.setattr("detail.api.models.is_installed", lambda *a, **k: True)
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    widget = GUI_MODULES["detection"].config_widget()

    assert "cuda" not in _backends(widget)
    sam3_row = next(r for r in _rows(widget) if r.startswith("sam3"))
    assert sam3_row.endswith("needs CUDA")  # visible, greyed, with the reason


def test_a_cuda_box_still_opens_on_sam3(qapp, monkeypatch, real_machine):
    """The fix must not move the default where the default works."""
    monkeypatch.setattr("detail.api.models.is_installed", lambda *a, **k: True)
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: False)
    widget = GUI_MODULES["detection"].config_widget()

    assert widget.config()["model_id"] == "sam3"
    assert widget._compute.backend.currentText() == "cuda"
    assert widget._compute.device.isEnabled()
    assert not widget._prompt.isHidden()  # a prompted model keeps its prompt rows
    assert widget.validation_error() is None


def test_refine_restores_the_prompt_in_either_of_its_two_forms(qapp):
    """The prompt is the setting a user is most likely to be iterating on, and the one the
    round-trip guard cannot reach: it is a nested object, not a number or a choice, so
    `tests/gui/test_config_round_trip.py` will not invent a different value for it.

    Two forms arrive. `config()` emits a live `Prompt`; a run record on disk holds
    `Prompt.to_dict()`. Both are asserted here because the first attempt at this read the
    serialized dict key by key and looked for `name` — the dataclass field is `class_name`, so
    a Refine from a *stored* run silently kept the form's default class while the live form
    worked, which is exactly the shape of failure that makes this whole class hard to see.
    """
    from detail.core.dataclasses import Prompt, PromptTarget, TextCue
    from detail.gui.modules.registry import GUI_MODULES

    live = Prompt(targets=[PromptTarget("rat", TextCue("a white rat"))])

    for prompt in (live, live.to_dict()):
        form = GUI_MODULES["detection"].config_widget()
        form.load_config({"prompt": prompt})
        target = form.config()["prompt"].targets[0]
        assert target.class_name == "rat"
        assert target.cue.text == "a white rat"


def test_an_imported_keypoint_model_is_not_offered_as_a_detector(qapp, tmp_path):
    """§26 Phase 4: the machine catalog holds keypoint models too; the detection form lists
    the detection ones only (a pose model handed to a detection loader cannot run)."""
    from pathlib import Path

    pytest.importorskip("onnxruntime")
    from detail.api import models as api_models

    data = (
        Path(__file__).parents[1]
        / "models"
        / "data"
        / "sleap_onnx"
        / "centered_instance"
    )
    pose = api_models.import_keypoint_model(data, name="mouse_pose")
    widget = GUI_MODULES["detection"].config_widget()
    widget.on_experiment_set(Experiment.create("p", tmp_path))
    offered = [widget._model.itemData(i) for i in range(widget._model.count())]
    assert pose not in [d["id"] for d in offered if d]


def _catalog_record(model_id: str, task: str) -> None:
    """A complete model in the machine-wide catalog (isolated per test), written as an
    import or a restore writes one."""
    from detail.core.skeleton import PRESETS
    from detail.models import imported as imported_store
    from detail.storage import models as models_store

    mdir = imported_store.model_dir(model_id)
    (mdir / "weights").mkdir(parents=True)
    (mdir / "weights" / "best.pt").write_bytes(b"w")
    record = {
        "id": model_id,
        "name": model_id,
        "source": "imported",
        "status": "complete",
        "task": task,
        "model_family": "yolo_pose" if task == "pose" else "yolo",
        "classes": [{"id": 0, "name": "animal"}],
        "weights": {"best": "weights/best.pt"},
    }
    if task == "pose":
        record["role"] = "keypoints"
        record["skeleton"] = PRESETS["midline_3"][1].to_dict()
    models_store.write_model_yaml(mdir, record)


def test_a_catalog_keypoint_model_is_not_offered_as_a_detector(qapp, tmp_path):
    """§12 *Keypoint models in the catalog*: a pose model in the catalog is a keypoint
    method. Offered here it would build as a detector and fail at run time."""
    _catalog_record("mouse_seg", "segment")
    _catalog_record("rat_pose", "pose")
    widget = GUI_MODULES["detection"].config_widget()
    widget.on_experiment_set(Experiment.create("p", tmp_path))
    offered = {
        widget._model.itemData(i)["id"]
        for i in range(widget._model.count())
        if widget._model.itemData(i)
    }
    assert "mouse_seg" in offered
    assert "rat_pose" not in offered
