"""Detection config form: one model list, each row saying whether it can run here."""

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.core.registries import MODELS  # noqa: E402
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui.modules.registry import GUI_MODULES  # noqa: E402


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
