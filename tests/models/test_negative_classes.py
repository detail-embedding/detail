"""A detector never reports its negative prompt's class, on any path that builds one (§12).

**The class.** A student distilled with a negative prompt (a tether's ``cable``) was trained on
one more class than its ``model.yaml`` ``classes`` lists. The loader drops that class's rows,
and that is the whole feature: nothing downstream of the model (detection runs, tracking,
keypoints, exports, the tuning sweep, the Quick tools) knows the class exists. So the bug is a
path that builds a trained detector without the record's ``negative_class_ids``: it works,
reports plausible detections, and puts the cable back into every run made through it, with
nothing failing at the line that forgot.

**Why it is guarded on every path, not on ``TrainedYOLO``.** The drop is one line in one
class, but the ids reach it through each construction site — the resolver for an experiment
model, for a catalog model, for a source found by lookup, for a Quick tool, and the training
preview, which builds the model itself before the record is complete. Each is driven here with
a real ``TrainedYOLO`` and an Ultralytics output that has a row of each class, through both
predict paths. ``negative_class_ids`` is a required keyword, so a new site that omits it is a
``TypeError``, not a silent default (``test_a_construction_site_that_forgets_the_ids_is_refused``).
"""

from pathlib import Path

import numpy as np
import pytest

from detail.core.dataclasses import Prompt
from detail.models import imported as imported_store
from detail.models import resolve
from detail.models import trained_yolo
from detail.models.trained_yolo import TrainedYOLO
from detail.pipeline import training
from detail.storage import experiment as experiment_store
from detail.storage import models as models_store

W, H = 32, 24
#: The output a negative-trained student gives: the animal (class 0) and the cable (class 1).
ROWS = [0, 1]


def _record(model_id: str) -> dict:
    return {
        "id": model_id,
        "model_family": "yolo",
        "status": "complete",
        "classes": [{"id": 0, "name": "mouse"}],
        "negative_classes": [{"id": 1, "name": "cable"}],
        "weights": {"best": "run/weights/best.pt"},
    }


class _Tensor:
    """What ``_host_arrays`` reads off an Ultralytics tensor: ``.cpu().numpy()``."""

    def __init__(self, values) -> None:
        self._values = np.asarray(values)

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _Boxes:
    def __init__(self, classes: list[int]) -> None:
        self.xyxy = _Tensor([[2.0, 2.0, 12.0, 10.0]] * len(classes))
        self.conf = _Tensor([0.9] * len(classes))
        self.cls = _Tensor([float(c) for c in classes])

    def __len__(self) -> int:
        return len(self.cls.numpy())


class _Result:
    def __init__(self, classes: list[int], h: int, w: int) -> None:
        self.boxes = _Boxes(classes)
        self.masks = type("Masks", (), {"data": _Tensor(np.ones((len(classes), h, w)))})


class _Ultralytics:
    """The external inference call, the one legitimate mock: a row of every class."""

    def predict(self, frame, **_kwargs):
        return [_Result(ROWS, *frame.shape[:2])]


def _reported(model) -> list[list[str]]:
    """The class names the model reports through ``predict`` and through the batched path."""
    model._model = _Ultralytics()
    frame = np.zeros((H, W, 3), np.uint8)
    single = model.predict(frame, Prompt(targets=[]))
    arrays = trained_yolo._host_arrays(_Result(ROWS, H, W))
    (batched,) = model.finish_batch([(arrays, W, H)])
    return [
        [d.class_name for d in single.detections()],
        [d.class_name for d in batched.detections()],
    ]


def _experiment_model(tmp_path: Path) -> Path:
    proj = experiment_store.create_experiment(tmp_path, "p")
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(mdir, _record("model_001"))
    return proj


def _catalog_model() -> str:
    mdir = imported_store.model_dir("tethered_mouse")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(mdir, _record("tethered_mouse"))
    return "tethered_mouse"


def _built(path: str, tmp_path: Path):
    if path == "experiment":
        return resolve.resolve_detection_model(
            _experiment_model(tmp_path), "model_001", "trained"
        )
    if path == "experiment_by_lookup":
        model, source = resolve.resolve_detection_model_auto(
            _experiment_model(tmp_path), "model_001"
        )
        assert source == "trained"
        return model
    if path == "catalog":
        return resolve.resolve_detection_model(tmp_path, _catalog_model(), "imported")
    if path == "catalog_by_lookup":
        model, source = resolve.resolve_detection_model_auto(tmp_path, _catalog_model())
        assert source == "imported"
        return model
    if path == "quick_tool":
        return resolve.resolve_pretrained_or_imported(_catalog_model())
    raise AssertionError(path)


@pytest.mark.parametrize(
    "path",
    [
        "experiment",
        "experiment_by_lookup",
        "catalog",
        "catalog_by_lookup",
        "quick_tool",
    ],
)
def test_every_resolver_path_drops_the_negative_class(path, tmp_path):
    model = _built(path, tmp_path)
    assert isinstance(model, TrainedYOLO)
    assert model.classes == ["mouse"]
    assert _reported(model) == [["mouse"], ["mouse"]]


def test_the_training_preview_drops_the_negative_class(tmp_path, monkeypatch):
    """The preview is the first thing a user sees of the model; it must not draw the cable."""
    import cv2

    from detail.core import visualize

    val = tmp_path / "ds" / "images" / "val"
    val.mkdir(parents=True)
    cv2.imwrite(str(val / "frame.jpg"), np.zeros((H, W, 3), np.uint8))

    class _Preview(TrainedYOLO):
        def load(self, device: str) -> None:
            self._model = _Ultralytics()

    drawn = []
    real_draw = visualize.draw_detections

    def draw(frame, detections, *args, **kwargs):
        drawn.extend(d.class_name for d in detections)
        return real_draw(frame, detections, *args, **kwargs)

    monkeypatch.setattr("detail.pipeline.training.TrainedYOLO", _Preview)
    monkeypatch.setattr(visualize, "draw_detections", draw)
    record = _record("model_001")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rel = training._capture_preview(
        tmp_path / "ds",
        tmp_path / "best.pt",
        record["classes"],
        run_dir,
        "cpu",
        None,
        negative_class_ids=resolve.negative_class_ids(record),
    )
    assert rel == "preview.png"
    assert drawn == ["mouse"]


def test_a_model_without_a_negative_prompt_reports_every_class(tmp_path):
    """The drop is the record's, not a rule about ids past ``classes``: an older record, an
    import, or a distill without a negative prompt reports what its weights find."""
    proj = experiment_store.create_experiment(tmp_path, "p")
    mdir = models_store.model_dir(proj, "model_001")
    mdir.mkdir(parents=True)
    record = _record("model_001")
    record["classes"].append({"id": 1, "name": "cable"})
    del record["negative_classes"]
    models_store.write_model_yaml(mdir, record)
    model = resolve.resolve_detection_model(proj, "model_001", "trained")
    assert _reported(model) == [["mouse", "cable"], ["mouse", "cable"]]


def test_a_construction_site_that_forgets_the_ids_is_refused(tmp_path):
    """The mechanism: a site that builds the loader without the ids fails at once instead of
    reporting the negative class."""
    with pytest.raises(TypeError, match="negative_class_ids"):
        TrainedYOLO(tmp_path / "best.pt", ["mouse"])
