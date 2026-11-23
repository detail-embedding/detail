"""The ``keypoint_detect`` launch form and run viewer (§26 "Surfaces").

The form is built through the module-UI registry (what the launch dialog does); the viewer
opens a run the real stage wrote with the fake method from the pipeline tests, so what it
draws and summarises is a run's actual rows, not a stub.
"""

import pytest

from detail.core.registries import KEYPOINT_MODELS
from detail.gui.modules.registry import GUI_MODULES
from detail.gui.qt import Qt
from detail.storage import recording as recording_store
from tests.gui.conftest import drain_until
from tests.pipeline.test_keypoint_detect import (
    tracked,  # noqa: F401  (the fixture)
    _FakeKeypointModel,
    _FakePoseModel,
    _config,
    _run,
)


@pytest.fixture
def form(qapp):
    return GUI_MODULES["keypoint_detect"].config_widget()


def test_the_form_opens_on_auto_and_asks_for_the_detection_run_first(form):
    """§26 revised 2026-10-21: a detect run reads a detection run, so the form asks for one
    and has no tracking run, no track ids and no *All tracks*."""
    config = form.config()
    assert config["decode"] is None and config["side_rule"] is None  # "auto" rows
    assert config["crop_px"] is None and config["batch_size"] == 16
    assert config["example_set_id"] is None and config["detection_spec"] is None
    assert config["model_id"] == "dinov3_match"
    assert config["min_pose_confidence"] is None
    assert not {"tracking_spec", "track_ids", "all_tracks"} & set(config)
    assert form.validation_error() == (
        "No detection run in the selected recordings — run detection first."
    )


def test_min_pose_confidence_shows_only_for_a_method_that_reports_one(
    qapp, monkeypatch
):
    """`validate` refuses the threshold for a method that reports no pose confidence, so
    the row is hidden there and the config carries None; for one that does (the student,
    here its whole-frame fake) the row shows, opens on the method default and round-trips.
    """
    monkeypatch.setitem(KEYPOINT_MODELS, "fake_pose", _FakePoseModel)
    form = GUI_MODULES["keypoint_detect"].config_widget()
    form.show()
    form._method.setCurrentIndex(form._method.findData("dinov3_match"))
    assert not form._min_conf.isVisible() and not form._min_conf_label.isVisible()
    form._method.setCurrentIndex(form._method.findData("fake_pose"))
    assert form._min_conf.isVisible() and form._min_conf_label.isVisible()
    assert form.config()["min_pose_confidence"] is None  # the method's 0.25
    assert "0.25" in form._min_conf.specialValueText()
    form.load_config({"model_id": "fake_pose", "min_pose_confidence": 0.4})
    assert form.config()["min_pose_confidence"] == pytest.approx(0.4)
    form.load_config({"model_id": "fake_pose", "min_pose_confidence": 0.0})
    assert form.config()["min_pose_confidence"] == 0.0  # a legal value, not "unset"
    form._method.setCurrentIndex(form._method.findData("dinov3_match"))
    assert form.config()["min_pose_confidence"] is None  # hidden: never sent


def test_settings_round_trip_and_selections_are_re_asked(form):
    """``load_config`` restores the method and every knob; the detection run and the example
    set name artifacts this experiment mints, so they are not carried (the stale-reference
    failure every tool that restored them has met)."""
    form.load_config(
        {
            "model_id": "dinov3_match",
            "model_variant": "vit_base",
            "decode": "per_point",
            "side_rule": "hard",
            "crop_px": 672,
            "batch_size": 4,
            "every_n_frames": 2,
            "example_set_id": "ex_07",
            "detection_spec": {"module": "detection", "identity": {}},
        }
    )
    config = form.config()
    assert (
        config["model_id"] == "dinov3_match" and config["model_variant"] == "vit_base"
    )
    assert (config["decode"], config["side_rule"], config["crop_px"]) == (
        "per_point",
        "hard",
        672,
    )
    assert config["batch_size"] == 4 and config["every_n_frames"] == 2
    assert config["example_set_id"] is None  # a selection: re-asked, never restored
    assert config["detection_spec"] is None


def test_the_example_set_comes_from_the_shared_picker_once_an_experiment_is_set(
    qapp, form, tmp_path
):
    """The detect form carries PR 2's `ExampleSetPicker` (§26 *Example sets follow the model
    that needs them*); its chosen set is the config's `example_set_id`, and the form refuses
    to run a method that learns from examples until one is chosen."""
    from detail.api.experiment import Experiment

    proj = Experiment.create("p", tmp_path)
    created = proj.create_example_set("dots", "quadruped_side_5")
    form.show()
    form.on_experiment_set(proj)
    picker = form._example_picker
    assert drain_until(qapp, lambda: not picker._reading), "the sets never loaded"
    qapp.processEvents()
    assert picker.set_id() == created.id
    assert form.config()["example_set_id"] == created.id
    assert picker._new.isEnabled() and picker._mark.isEnabled()


def test_the_picker_shows_only_for_a_method_that_learns_from_examples(
    qapp, monkeypatch
):
    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp", _FakeKeypointModel)
    form = GUI_MODULES["keypoint_detect"].config_widget()
    form.show()
    assert form._method.findData("fake_kp") >= 0
    form._method.setCurrentIndex(form._method.findData("dinov3_match"))
    assert form._example_picker.isVisible() and form._examples_label.isVisible()
    form._method.setCurrentIndex(form._method.findData("fake_kp"))
    assert not form._example_picker.isVisible()  # hidden, not greyed
    assert not form._examples_label.isVisible()
    assert form.config()["example_set_id"] is None
    assert form.config()["model_variant"] is None  # the fake has no sizes


@pytest.fixture
def detect_run(tracked):  # noqa: F811 — the imported fixture
    """``tracked``'s recording with a finished ``kd_01`` written by the real stage over the
    per-detection fake: one instance per detection, two on frames 1 to 4."""
    it, recording = tracked
    _run(it, recording, _config())
    data = recording_store.read_recording(recording.dir)
    data.setdefault("keypoint_detect_runs", []).append(
        {
            "id": "kd_01",
            "status": "complete",
            "detection_run_id": "det_01",
            "model_id": "fake_kp",
            "output": "keypoint_detect/kd_01/",
        }
    )
    recording_store.write_recording(recording.dir, data)
    return recording


def test_the_viewer_draws_every_instance_on_the_detection_frames(qapp, detect_run):
    """§26 revised: the detect viewer shows the detection run's frames and draws every pose
    on a frame — no subject, so two detections on a frame are two poses — with a coverage
    line per part (frames walked, frames with a pose, poses)."""
    from detail.gui.keypoint_run_preview import KeypointDetectRunPreview
    from detail.gui.run_preview import DetectionRunPreview, TrackingRunPreview

    view = GUI_MODULES["keypoint_detect"].preview_widget(detect_run, "kd_01")
    assert isinstance(view, KeypointDetectRunPreview)
    assert isinstance(view, DetectionRunPreview)
    assert not isinstance(view, TrackingRunPreview)
    assert drain_until(qapp, lambda: hasattr(view, "_scrubber")), "the run never loaded"
    summary = view._summary.text()
    assert "keypoints: fake_kp · 2 points" in summary
    assert "on detection run det_01" in summary
    assert "part 001 — 5/5 frames with a pose · 9 poses" in summary
    kind, with_points = view._compute(1)  # frame 1: two detections, two poses
    assert kind == "img"
    _actual, rgb, detections = view._preview.frame(0, 1, with_masks=True)
    detection_only = view._overlay(rgb, detections)
    assert with_points.shape == detection_only.shape
    assert (with_points != detection_only).any()  # the poses were drawn on top
    # every instance, not one: drawing only the first leaves the second pose's pixels bare
    from detail.core.visualize import draw_keypoints

    run = view._run
    found = run.at("part_001", 1)
    assert len(found["row"]) == 2
    first_only = draw_keypoints(
        detection_only,
        found["points"][0],
        run.skeleton,
        scores=found["point_scores"][0],
    )
    assert (with_points != first_only).any()


def test_a_launch_with_nothing_checked_greys_no_recording(qapp, tracked):
    """Reported from the app: every recording greyed "(not in the selected run)" before
    one could be checked. Recordings start unchecked, the upstream groups come from the
    checked ones, and the picker answered "none covered" when nothing was chosen, so the
    dialog greyed every row and the form could never be filled. The class is every form
    whose `covered_recording_ids` reads `UpstreamSpecPicker.covered()`."""
    from detail.gui.launch import LaunchRunDialog

    it, _recording = tracked
    form = GUI_MODULES["keypoint_detect"].config_widget()
    form.on_recordings_changed([])
    assert (
        form.covered_recording_ids() is None
    )  # nothing narrowed yet, not nothing covered
    dialog = LaunchRunDialog(it.experiment)
    dialog._module_combo.setCurrentIndex(
        dialog._module_combo.findData("keypoint_detect")
    )
    rows = [dialog._recordings.item(i) for i in range(dialog._recordings.count())]
    eligible = [r for r in rows if "(needs" not in r.text()]
    assert eligible, "the fixture has recordings with a complete detection run"
    for row in eligible:
        assert row.flags() & Qt.ItemFlag.ItemIsEnabled, row.text()
        assert "not in the selected run" not in row.text()


def test_a_method_with_only_a_smaller_size_installed_is_offered_on_that_size(
    qapp, monkeypatch
):
    """Found from the app report: `dinov3_match` shares DINOv3's weights and has no catalog
    row, so availability found no sizes and checked only the default — `vit_base` installed
    without `vit_large` read as no keypoint method available."""
    from detail.api import models as api_models
    from detail.gui import model_availability

    installed = {("dinov3_match", "vit_base")}
    monkeypatch.setattr(
        api_models,
        "is_installed",
        lambda model_id, variant=None: (model_id, variant) in installed,
    )
    assert model_availability._variants("dinov3_match") == ["vit_large", "vit_base"]
    assert model_availability.unavailable_reason("dinov3_match") is None
    form = GUI_MODULES["keypoint_detect"].config_widget()
    assert form._method.currentData() == "dinov3_match"
    assert form._variant.currentData() == "vit_base"  # opened on what is installed
    large = form._variant.model().item(form._variant.findData("vit_large"))
    assert not large.isEnabled() and "not installed" in form._variant.itemText(
        form._variant.findData("vit_large")
    )


def test_with_no_supported_size_installed_the_form_names_what_to_install(
    qapp, monkeypatch
):
    """Reported from the app: only DINOv3 `vit_small` installed, which `dinov3_match` does
    not use (§26: large default, base), and the form said only "Install models via Model
    manager" — where `dinov3_match` has no row. It names the row and the sizes."""
    from detail.api import models as api_models

    monkeypatch.setattr(
        api_models, "is_installed", lambda model_id, variant=None: False
    )
    form = GUI_MODULES["keypoint_detect"].config_widget()
    message = form.validation_error()
    assert message == (
        "No keypoint method can run here: dinov3_match needs dinov3_vit vit_large or "
        "vit_base installed. Install it on the Model manager."
    )
    assert form._no_methods.text() == message


# -- Test keypoints (PR 4b): Test prompts' counterpart on the detect form --------------- #
from tests.pipeline.test_keypoint_dataset import patch_teacher  # noqa: E402,F401


def _ready_form(qapp, patch_teacher):
    """A detect form with the patch fake chosen, its example set loaded and the recording
    selected, whose detection run has one tracking run: everything Test keypoints needs.
    """
    it, recording, set_id, _seen = patch_teacher
    form = GUI_MODULES["keypoint_detect"].config_widget()
    form.show()
    form._method.setCurrentIndex(form._method.findData("fake_learner"))
    form.on_experiment_set(it.experiment)
    picker = form._example_picker
    assert drain_until(qapp, lambda: not picker._reading), "the sets never loaded"
    picker.select(set_id)
    qapp.processEvents()
    form.on_recordings_changed([recording])
    return it, recording, set_id, form


def test_test_keypoints_is_the_last_row_and_says_why_until_it_can_run(qapp, tracked):
    from detail.gui.qt import QFormLayout

    form = GUI_MODULES["keypoint_detect"].config_widget()
    rows = [
        (layout, layout.getWidgetPosition(form._test)[0])
        for layout in form.findChildren(QFormLayout)
        if layout.getWidgetPosition(form._test)[0] >= 0
    ]
    assert len(rows) == 1, "the button is one row of the form"
    layout, row = rows[0]
    assert (
        row == layout.rowCount() - 1
    ), "the last row, where detection has Test prompts"
    assert form._test.text() == "Test keypoints"
    assert not form._test.isEnabled()
    assert form._test.toolTip() == "Open an experiment first"


def test_test_keypoints_greys_with_the_reason_for_each_missing_input(
    qapp, patch_teacher, monkeypatch
):
    it, recording, set_id, form = _ready_form(qapp, patch_teacher)
    assert form._test.isEnabled(), form._test.toolTip()

    form._example_picker._set.setCurrentIndex(-1)
    qapp.processEvents()
    assert not form._test.isEnabled()
    assert "Choose an example set" in form._test.toolTip()
    form._example_picker.select(set_id)
    qapp.processEvents()

    form.on_recordings_changed([])
    assert not form._test.isEnabled()
    assert form._test.toolTip() == "Choose a detection run first"
    form.on_recordings_changed([recording])

    # the frames come from the tracking on the chosen detection run: none, then two
    data = recording_store.read_recording(recording.dir)
    tracking = list(data["tracking_runs"])
    data["tracking_runs"] = []
    recording_store.write_recording(recording.dir, data)
    form.on_recordings_changed([recording])
    assert not form._test.isEnabled()
    assert "run tracking on it first" in form._test.toolTip()
    data["tracking_runs"] = tracking + [
        {"id": "track_02", "status": "complete", "detection_run_id": "det_01"}
    ]
    recording_store.write_recording(recording.dir, data)
    form.on_recordings_changed([recording])
    assert not form._test.isEnabled()
    assert "more than one tracking run on this detection run" in form._test.toolTip()
    data["tracking_runs"] = tracking
    recording_store.write_recording(recording.dir, data)
    form.on_recordings_changed([recording])
    assert form._test.isEnabled(), form._test.toolTip()

    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp", _FakeKeypointModel)
    form._method.addItem("fake_kp", "fake_kp")
    form._method.setCurrentIndex(form._method.findData("fake_kp"))
    assert not form._test.isEnabled()
    assert "learns from an example set" in form._test.toolTip()


def test_test_keypoints_hands_the_window_the_forms_choices(
    qapp, patch_teacher, monkeypatch
):
    """The test places points the way the run will: the form's method, example set, the
    tracking on the chosen detection run, compute and Advanced settings all reach
    `test_keypoints`."""
    from detail.gui import keypoint_test

    it, recording, set_id, form = _ready_form(qapp, patch_teacher)
    form._crop_px.setValue(96)
    form._decode.setCurrentIndex(form._decode.findData("per_point"))
    teacher = form.test_teacher()
    assert teacher["example_set"] == set_id and teacher["model_id"] == "fake_learner"
    assert teacher["tracking"] == {recording.id: "track_01"}
    assert teacher["settings"] == {"crop_px": 96, "decode": "per_point"}
    assert teacher["label_compute"] == form._compute.compute_target()

    opened = []

    class Window:
        def __init__(self, experiment, teacher, names, *, cutoff=None, parent=None):
            opened.append((experiment, teacher, names, cutoff))

        def exec(self):
            return 0

    monkeypatch.setattr(keypoint_test, "KeypointDetectTestDialog", Window)
    form._test.click()
    experiment, handed, names, cutoff = opened[0]
    assert experiment is it.experiment and handed == teacher
    assert names == {recording.id: recording.name}
    assert cutoff == KEYPOINT_MODELS["fake_learner"].score_spec.default_cutoff
