"""The heading picker (§14) and the embedding form's alignment rows: the picker greys a
definition the skeleton cannot satisfy, Define… saves through the experiment, and the form
carries the alignment settings and refuses an incomplete one before the run would."""

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui.modules._heading import (
    HeadingDefinitionDialog,
    HeadingPicker,
)  # noqa: E402


@pytest.fixture
def experiment(tmp_path):
    proj = Experiment.create("p", tmp_path)
    proj.save_heading_definition("nose_to_tail", ["nose"], ["tail_base"])
    proj.save_heading_definition("ears_to_hip", ["left_ear", "right_ear"], ["hip"])
    return proj


def test_the_picker_lists_definitions_and_greys_the_ones_the_skeleton_lacks(
    qapp, experiment
):
    picker = HeadingPicker()
    assert picker.name() is None and not picker.define.isEnabled()
    picker.set_experiment(experiment)
    assert picker.define.isEnabled()
    assert [picker.combo.itemData(i) for i in range(picker.combo.count())] == [
        None,
        "nose_to_tail",
        "ears_to_hip",
    ]
    picker.set_name("ears_to_hip")
    assert picker.name() == "ears_to_hip" and picker.reason() is None
    picker.set_skeleton_names(("nose", "left_ear", "right_ear", "tail_base"))
    row = picker.combo.model().item(picker.combo.findData("ears_to_hip"))
    assert not row.isEnabled()
    assert "skeleton lacks: ['hip']" in picker.combo.itemText(2)
    assert picker.name() is None  # a greyed choice is cleared, not kept silently
    picker.set_name("nose_to_tail")
    assert picker.name() == "nose_to_tail" and picker.reason() is None
    picker.set_skeleton_names(None)
    assert picker.combo.model().item(2).isEnabled()


def test_define_saves_through_the_experiment_and_refuses_a_bad_set(qapp, experiment):
    dialog = HeadingDefinitionDialog(experiment, ("nose", "tail_base", "hip"))
    dialog._name.setText("new_one")
    for widget, picks in ((dialog._front_list, {"nose"}), (dialog._back_list, {"hip"})):
        for i in range(widget.count()):
            item = widget.item(i)
            from PyQt6.QtCore import Qt

            item.setCheckState(
                Qt.CheckState.Checked
                if item.text() in picks
                else Qt.CheckState.Unchecked
            )
    dialog._save()
    assert dialog.saved == {"name": "new_one", "front": ["nose"], "back": ["hip"]}
    assert {d["name"] for d in experiment.heading_definitions()} >= {"new_one"}
    typed = HeadingDefinitionDialog(experiment, None)
    typed._name.setText("typed")
    typed._front_text.setText("nose, left_ear")
    typed._back_text.setText("nose")
    typed._save()  # the same point on both sides: refused, nothing saved
    assert typed.saved is None
    assert "typed" not in {d["name"] for d in experiment.heading_definitions()}


class _Recording:
    """A stand-in with a tracking run and a keypoint track run, as the launch dialog's
    recordings expose them (record lists named by ``runs_key``)."""

    def __init__(self, rid="s1"):
        self.id = rid
        self.tracking_runs = [
            {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
        ]
        self.keypoint_detect_runs = [
            {"id": "kd_01", "status": "complete", "model_id": "dinov3_match"}
        ]
        self.keypoint_track_runs = [
            {
                "id": "kt_01",
                "status": "complete",
                "detect_run_id": "kd_01",
                "method": "temporal_decode",
                "config": {"despike": True},
            }
        ]

    def track_ids(self, run_id):
        return [0]

    def keypoint_track_run(self, run_id):
        raise FileNotFoundError("no run on disk for this stand-in")


def test_the_embedding_form_carries_alignment_and_refuses_an_incomplete_one(
    qapp, experiment
):
    from detail.gui.modules.embedding import _EmbeddingConfig

    form = _EmbeddingConfig()
    form.on_experiment_set(experiment)
    config = form.config()
    assert config["align"] == "none" and config["keypoint_track_spec"] is None
    assert config["heading"] is None and config["max_gap_s"] == 0.5
    assert not form._keypoint.isEnabled() and not form._heading.isEnabled()
    form.on_recordings_changed([_Recording(), _Recording("s2")])
    assert form.validation_error() is None
    form._align.setCurrentIndex(form._align.findData("rotate"))
    assert form._keypoint.isEnabled() and form._heading.isEnabled()
    assert not form._mirror_hold.isEnabled()  # a mirror-only setting
    assert form.config()["keypoint_track_spec"]["module"] == "keypoint_track"
    assert "Choose a heading definition" in form.validation_error()
    form._heading.set_name("nose_to_tail")
    assert form.validation_error() is None
    assert form.config()["heading"] == "nose_to_tail"
    form._constructor.setCurrentIndex(form._constructor.findData("full_frame"))
    assert "crops to the subject" in form.validation_error()
    form._constructor.setCurrentIndex(
        form._constructor.findData("masked_color_centered_contrast")
    )
    form._align.setCurrentIndex(form._align.findData("mirror"))
    assert form._mirror_hold.isEnabled()
    form.load_config({"align": "rotate", "heading": "nose_to_tail", "max_gap_s": 0.25})
    config = form.config()
    assert config["align"] == "rotate" and config["max_gap_s"] == pytest.approx(0.25)
    form.load_config({"align": "none"})
    assert form.config()["align"] == "none" and form.config()["heading"] is None
    assert form.validation_error() is None
