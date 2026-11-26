"""The GUI reaches the negative prompt and shows it back (§12 *Negative prompt*).

Both labelling dialogs (Distill new model, Auto-label) carry the field under Advanced, test
it with the user's prompts, refuse a clash at Test prompts, and pass it to the API. The
dataset page, the experiment model page and the catalog model page show it when there is one
and add no row when there is not; the model pages also say beside the metrics that mAP50
averages the negative class in.
"""

import pytest
from PyQt6.QtWidgets import QFormLayout

from detail.api.experiment import Experiment
from detail.configs.labeling import LabelingConfig
from detail.gui.forms import AdvancedSection
from detail.models import distilled
from detail.storage import datasets as datasets_store
from detail.storage import models as models_store

NEGATIVE_ROW = "cable (learned apart, never reported)"
METRICS_NOTE = "\n(mAP50 averages in cable, which this model learned but never reports)"


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001")
    return proj


def _dialog(kind: str, experiment):
    from detail.gui import fit

    if kind == "distill":
        return fit.DistillDialog(experiment)
    return fit.GenerateLabelingDialog(experiment)


def _fill(dialog) -> None:
    dialog._name.setText("m")
    dialog._class_rows[0][0].setText("mouse")
    dialog._class_rows[0][1].setText("a mouse")


def _in_advanced(widget) -> bool:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, AdvancedSection):
            return True
        parent = parent.parentWidget()
    return False


@pytest.mark.parametrize("kind", ["distill", "auto_label"])
def test_the_field_is_empty_under_advanced_with_the_configs_help(
    qapp, experiment, kind
):
    dialog = _dialog(kind, experiment)
    assert dialog._negative.text() == ""
    assert _in_advanced(dialog._negative)
    assert (
        dialog._negative.toolTip()
        == LabelingConfig.model_fields["negative_prompt"].description
    )


@pytest.mark.parametrize("kind", ["distill", "auto_label"])
def test_test_prompts_tries_the_negative_prompt_beside_the_classes(
    qapp, experiment, monkeypatch, kind
):
    from detail.gui import fit

    tested = {}

    class _PromptTest:
        def __init__(self, recordings, **kwargs):
            tested.update(kwargs)

        def exec(self):
            return 0

    monkeypatch.setattr(fit, "PromptTestDialog", _PromptTest)
    dialog = _dialog(kind, experiment)
    _fill(dialog)
    dialog._negative.setText(" cable ")
    dialog._open_prompt_test()
    assert tested["prompt_text"].splitlines() == ["a mouse", "cable"]


@pytest.mark.parametrize("kind", ["distill", "auto_label"])
def test_a_negative_prompt_that_is_a_class_is_said_at_test_prompts(
    qapp, experiment, monkeypatch, kind
):
    from detail.gui import fit

    said = []
    monkeypatch.setattr(
        fit.QMessageBox, "information", lambda _p, _t, text: said.append(text)
    )
    monkeypatch.setattr(
        fit,
        "PromptTestDialog",
        lambda *_a, **_k: pytest.fail("Test prompts opened on a clash"),
    )
    dialog = _dialog(kind, experiment)
    _fill(dialog)
    dialog._negative.setText("A mouse")
    dialog._open_prompt_test()
    assert said == [
        "the negative prompt 'A mouse' is also class 'mouse'; a class is labelled as "
        "what you want detected, and the negative prompt as what you do not, so change "
        "one of them"
    ]


@pytest.mark.parametrize("typed, expected", [("  cable ", "cable"), ("   ", None)])
@pytest.mark.parametrize(
    "kind, method", [("distill", "distill"), ("auto_label", "label_dataset")]
)
def test_the_dialogs_pass_the_negative_prompt(
    qapp, experiment, monkeypatch, kind, method, typed, expected
):
    captured = {}

    def launch(name, classes=None, **kwargs):
        captured.update(kwargs, classes=classes or kwargs.get("classes"))
        return object()

    monkeypatch.setattr(experiment, method, launch)
    dialog = _dialog(kind, experiment)
    _fill(dialog)
    dialog._negative.setText(typed)
    dialog._run()
    assert captured["negative_prompt"] == expected
    assert [c.name for c in captured["classes"]] == ["mouse"]  # the API adds it


# --- the pages that show it back -------------------------------------------- #
def _row_text(widget, label: str) -> str | None:
    for form in widget.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            if item is not None and item.widget() and item.widget().text() == label:
                return form.itemAt(row, QFormLayout.ItemRole.FieldRole).widget().text()
    return None


def _write_dataset(experiment, negatives) -> None:
    ddir = datasets_store.dataset_dir(experiment.dir, "dataset_001")
    ddir.mkdir(parents=True)
    record = {
        "id": "dataset_001",
        "name": "d",
        "status": "complete",
        "source": {"kind": "labeling", "model": "sam3"},
        "classes": [{"id": 0, "name": "mouse"}],
    }
    if negatives is not None:
        record["negative_classes"] = negatives
    datasets_store.write_dataset_yaml(ddir, record)


def _write_model(experiment, negatives) -> None:
    mdir = models_store.model_dir(experiment.dir, "model_001")
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "best.pt").write_bytes(b"weights")
    record = {
        "id": "model_001",
        "name": "tethered",
        "trainer_id": "ultralytics_yolo",
        "dataset_id": "dataset_001",
        "model_family": "yolo",
        "task": "segment",
        "classes": [{"id": 0, "name": "mouse"}],
        "weights": {"best": "run/weights/best.pt"},
        "metrics": {"best_box_map50": 0.8},
        "status": "complete",
    }
    if negatives is not None:
        record["negative_classes"] = negatives
    models_store.write_model_yaml(mdir, record)


CABLE = [{"id": 1, "name": "cable"}]


@pytest.mark.parametrize("negatives, expected", [(CABLE, NEGATIVE_ROW), (None, None)])
def test_the_dataset_page_shows_the_negative_prompt(
    qapp, experiment, negatives, expected
):
    from detail.gui.pages import DatasetDetailPage

    _write_dataset(experiment, negatives)
    page = DatasetDetailPage(experiment, "dataset_001")
    assert _row_text(page, "Classes:") == "mouse"
    assert _row_text(page, "Negative prompt:") == expected


@pytest.mark.parametrize(
    "negatives, expected, note",
    [(CABLE, NEGATIVE_ROW, METRICS_NOTE), (None, None, "")],
)
def test_the_model_pages_show_the_negative_prompt(
    qapp, experiment, negatives, expected, note
):
    from detail.gui.pages import GlobalModelDetailPage, ModelDetailPage

    _write_model(experiment, negatives)
    page = ModelDetailPage(experiment, "model_001")
    assert _row_text(page, "Classes:") == "mouse"
    assert _row_text(page, "Negative prompt:") == expected
    assert _row_text(page, "Metrics:") == "best_box_map50=0.8" + note
    gid = distilled.register_distilled_model(
        experiment.dir,
        "model_001",
        source_experiment_name="p",
        source_experiment_id="p",
    )
    catalog = GlobalModelDetailPage(gid)
    assert _row_text(catalog, "Classes") == "mouse"
    assert _row_text(catalog, "Negative prompt") == expected
    assert _row_text(catalog, "Metrics") == "{'best_box_map50': 0.8}" + note
