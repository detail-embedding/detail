"""The GUI reaches the tether option and tracking's *Leave out classes* (§12, §13).

The two labeling dialogs (Distill new model, Auto-label) pass the checkbox to the API and test
the cable prompt with the user's; the tracking form says what the default leaves out for the
detection run picked, and its config carries the choice.
"""

import pytest

from detail.api.experiment import Experiment
from detail.storage import models as models_store


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    return Experiment.create("p", tmp_path)


def _fill(dialog):
    dialog._name.setText("m")
    dialog._class_rows[0][0].setText("mouse")
    dialog._class_rows[0][1].setText("a mouse")


def test_the_distill_dialog_passes_tether_and_tests_the_cable(
    qapp, experiment, monkeypatch
):
    from detail.gui import fit as fitmod

    captured = {}
    monkeypatch.setattr(
        experiment, "distill", lambda name, **kw: captured.update(kw) or object()
    )
    tested = {}

    class _PromptTest:
        def __init__(self, recordings, **kwargs):
            tested.update(kwargs)

        def exec(self):
            return 0

    monkeypatch.setattr(fitmod, "PromptTestDialog", _PromptTest)
    experiment.create_recording("recording_001")
    dialog = fitmod.DistillDialog(experiment)
    assert not dialog._tether.isChecked()  # off unless asked for
    _fill(dialog)
    dialog._tether.setChecked(True)
    dialog._open_prompt_test()
    assert tested["prompt_text"].splitlines() == ["a mouse", "cable"]
    dialog._run()
    assert captured["tether"] is True
    assert [c.name for c in captured["classes"]] == ["mouse"]  # the API adds the cable


def test_the_auto_label_dialog_passes_tether(qapp, experiment, monkeypatch):
    from detail.gui import fit as fitmod

    captured = {}
    monkeypatch.setattr(
        experiment,
        "label_dataset",
        lambda name, classes, **kw: captured.update(kw) or object(),
    )
    dialog = fitmod.GenerateLabelingDialog(experiment)
    _fill(dialog)
    dialog._tether.setChecked(True)
    dialog._run()
    assert captured["tether"] is True


def test_a_cable_class_with_the_option_on_is_said_at_test_prompts(
    qapp, experiment, monkeypatch
):
    from detail.gui import fit as fitmod

    said = []
    monkeypatch.setattr(
        fitmod.QMessageBox, "information", lambda _p, _t, text: said.append(text)
    )
    dialog = fitmod.DistillDialog(experiment)
    _fill(dialog)
    dialog._class_rows[0][0].setText("cable")
    dialog._tether.setChecked(True)
    dialog._open_prompt_test()
    assert said and "already the cable" in said[0]


SPEC = {"module": "detection", "identity": {"model_id": "model_001"}, "parent": None}


def _form_with(auxiliary):
    from detail.gui.modules.tracking import _TrackingConfig

    form = _TrackingConfig()
    form.apply_recordings(
        [],
        {
            "recordings": [],
            "groups": [{"spec": SPEC, "label": "det", "recordings": ["r"]}],
            "arena_groups": [],
            "rates": None,
            "auxiliary": [(SPEC, auxiliary)],
        },
    )
    return form


def test_the_tracking_form_leaves_out_the_models_auxiliary_classes_by_default(qapp):
    form = _form_with(["cable"])
    assert form.config()["exclude_classes"] is None  # resolved by validate()
    assert "Leaves out cable" in form._exclude_note.text()
    form._exclude.setCurrentIndex(1)
    assert form.config()["exclude_classes"] == []
    assert form._exclude_note.text() == "Every class is tracked, cable included."


def test_the_tracking_form_says_when_there_is_nothing_to_leave_out(qapp):
    form = _form_with([])
    assert "no auxiliary classes" in form._exclude_note.text()


def test_the_auxiliary_classes_are_read_from_the_model(tmp_path):
    from detail.gui.modules.tracking import _auxiliary_by_spec

    mdir = models_store.model_dir(tmp_path, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir, {"id": "model_001", "auxiliary_classes": ["cable"]}
    )
    spec = {"identity": {"model_id": "model_001", "weights_source": "trained"}}
    groups = [{"spec": spec, "label": "det", "recordings": ["recording_001"]}]
    assert _auxiliary_by_spec(tmp_path, groups) == [(spec, ["cable"])]
