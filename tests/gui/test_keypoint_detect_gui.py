"""The ``keypoint_detect`` launch form and run viewer (§26 "Surfaces").

The form is built through the module-UI registry (what the launch dialog does); the viewer
opens a run the real stage wrote with the fake method from the pipeline tests, so what it
draws and summarises is a run's actual rows, not a stub.
"""

import pytest
from PyQt6.QtCore import Qt

from detail.core.registries import KEYPOINT_MODELS
from detail.gui.modules.registry import GUI_MODULES
from detail.storage import recording as recording_store
from tests.gui.conftest import drain_until
from tests.pipeline.test_keypoint_detect import (
    tracked,  # noqa: F401  (the fixture)
    two_classes,  # noqa: F401  (the fixture)
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


def test_the_class_row_offers_the_detection_runs_classes_and_writes_the_field(
    qapp, two_classes  # noqa: F811 — the imported fixture
):
    """The class row offers what the chosen detection run actually detected (read off its
    parts on a thread, as tracking's row is), opens on every class, writes the chosen one,
    narrows the recordings to those that have it, and is hidden — and sent as None — for a
    whole-frame method, whose `validate` refuses it."""
    it, recording = two_classes
    form = GUI_MODULES["keypoint_detect"].config_widget()
    form.show()
    form._method.setCurrentIndex(form._method.findData("fake_kp"))
    assert form._class.isVisible() and form._class_label.isVisible()
    form.on_recordings_changed([recording])
    assert drain_until(qapp, lambda: form._class.count() == 3), "classes never read"
    assert [form._class.itemData(i) for i in range(3)] == [None, "mouse", "plate"]
    assert form._class.itemText(0) == "(every class)"
    assert form.config()["class_name"] is None
    form._class.setCurrentIndex(form._class.findData("mouse"))
    assert form.config()["class_name"] == "mouse"
    assert form.covered_recording_ids() == {recording.id}
    form._method.setCurrentIndex(form._method.findData("fake_pose"))
    assert not form._class.isVisible() and not form._class_label.isVisible()
    assert form.config()["class_name"] is None


def test_a_restored_class_waits_for_the_detection_runs_classes(
    qapp, two_classes  # noqa: F811 — the imported fixture
):
    """A class is a setting (the same animal on the next recording), so *Start from*
    restores it — even when the config arrives before the classes have been read."""
    _it, recording = two_classes
    form = GUI_MODULES["keypoint_detect"].config_widget()
    form.show()
    form.load_config({"model_id": "fake_kp", "class_name": "plate"})
    form.on_recordings_changed([recording])
    assert drain_until(qapp, lambda: form._class.count() == 3), "classes never read"
    assert form.config()["class_name"] == "plate"
    form.load_config({"model_id": "fake_kp", "class_name": None})
    assert form.config()["class_name"] is None


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

    view = GUI_MODULES["keypoint_detect"].preview_widget(
        detect_run, "kd_01", experiment_dir=None, parent_run_id=None
    )
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
        "vit_base installed; superanimal_quadruped not installed; "
        "superanimal_topviewmouse not installed. Install one on the Model manager."
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
    from PyQt6.QtWidgets import QFormLayout

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


def test_several_tracking_runs_ask_which_one_test_keypoints_draws_from(
    qapp, patch_teacher
):
    """The social flow always has several tracking runs on one detection run (each class's
    track and their merge). *Test on tracking* appears then — never chosen for the person —
    offers only the tracking runs on the chosen detection run, and the one chosen is what
    the test draws its frames from; with one tracking run it stays hidden."""
    it, recording, _set_id, form = _ready_form(qapp, patch_teacher)
    assert not form._test_tracking.combo.isVisible()  # one tracking run: nothing to ask
    assert form._test.isEnabled(), form._test.toolTip()

    data = recording_store.read_recording(recording.dir)
    first = dict(data["tracking_runs"][0])
    data["tracking_runs"].append(
        {**first, "id": "track_02", "tracker_id": "track_merge", "class_name": None}
    )
    recording_store.write_recording(recording.dir, data)
    form.on_recordings_changed([recording])
    combo = form._test_tracking.combo
    assert combo.isVisible() and form._test_tracking_label.isVisible()
    assert combo.currentIndex() == 0 and form._test_tracking.spec() is None
    assert not form._test.isEnabled()
    assert "choose which one Test keypoints draws" in form._test.toolTip()
    assert combo.count() == 3  # the choose row, and the two tracking runs

    for index in range(1, combo.count()):
        combo.setCurrentIndex(index)
        teacher = form.test_teacher()
        if teacher["tracking"] == {recording.id: "track_02"}:
            break
    else:
        raise AssertionError("track_02 was never offered")
    assert form._test.isEnabled(), form._test.toolTip()


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


def test_an_imported_keypoint_model_is_offered_as_a_method(qapp):
    """A catalog ONNX import is machine-global: the form offers it in any experiment, with
    no example set to choose, and launches it by its id."""
    from tests.pipeline.test_keypoint_detect import _import_onnx_fixture

    model_id = _import_onnx_fixture()
    form = GUI_MODULES["keypoint_detect"].config_widget()
    form.show()
    index = form._method.findData(model_id)
    assert index >= 0 and "from the catalog" in form._method.itemText(index)
    form._method.setCurrentIndex(index)
    assert not form._example_picker.isVisible()
    assert form.config()["model_id"] == model_id


def _class_learner(monkeypatch):
    class _Learner(_FakeKeypointModel):
        model_id = "fake_learner"
        needs_examples = True

        def __init__(self, variant=None, **_kwargs):
            super().__init__(variant)

    monkeypatch.setitem(KEYPOINT_MODELS, "fake_learner", _Learner)


def test_one_example_set_per_class_replaces_class_and_examples(
    qapp, two_classes, monkeypatch  # noqa: F811 — the imported fixture
):
    """§26 *One example set per detection class*: greyed until the detection run's classes
    are read and number two; ticked, Class and Examples give way to one row per class, and
    the config and Test keypoints carry the class-to-set map, never a class or one set.
    """
    from tests.pipeline.test_keypoint_detect import SKELETON

    _class_learner(monkeypatch)
    it, recording = two_classes
    white = it.experiment.create_example_set("white mouse", SKELETON)
    black = it.experiment.create_example_set("black mouse", SKELETON)
    form = GUI_MODULES["keypoint_detect"].config_widget()
    form.show()
    form._method.setCurrentIndex(form._method.findData("fake_learner"))
    form.on_experiment_set(it.experiment)
    assert form._per_class.isVisible() and not form._per_class.isEnabled()
    assert "fewer than two classes" in form._per_class.toolTip()
    form.on_recordings_changed([recording])
    assert drain_until(qapp, lambda: form._per_class.isEnabled()), "classes never read"
    assert form._class_sets.classes() == ["mouse", "plate"]
    assert not form._class_sets.isVisible()  # unticked: the single form as before

    form._per_class.setChecked(True)
    assert form._class_sets.isVisible()
    assert not form._class.isVisible() and not form._example_picker.isVisible()
    rows = form._class_sets._rows
    for _tick, picker in rows.values():
        assert drain_until(qapp, lambda p=picker: not p._reading), "sets never loaded"
    rows["mouse"][1].select(white.id)
    rows["plate"][1].select(black.id)
    qapp.processEvents()
    assert form.validation_error() is None
    config = form.config()
    assert config["class_example_sets"] == {"mouse": white.id, "plate": black.id}
    assert config["class_name"] is None and config["example_set_id"] is None
    teacher = form.test_teacher()
    assert teacher["class_example_sets"] == config["class_example_sets"]
    assert "example_set" not in teacher

    rows["plate"][0].setChecked(False)
    assert not rows["plate"][1].isEnabled()
    assert "two classes or more" in form.validation_error()

    form._per_class.setChecked(False)
    assert form.config()["class_example_sets"] is None
    assert form._example_picker.isVisible() and form._class.isVisible()


def test_start_from_ticks_the_classes_and_asks_again_for_their_sets(
    qapp, two_classes, monkeypatch  # noqa: F811 — the imported fixture
):
    """Which classes had a set is a setting; the sets are selections (re-asked), as the
    single Examples picker's set is — even when the config arrives before the classes.
    """
    _class_learner(monkeypatch)
    _it, recording = two_classes
    form = GUI_MODULES["keypoint_detect"].config_widget()
    form.show()
    form.load_config(
        {"model_id": "fake_learner", "class_example_sets": {"plate": "ex_07"}}
    )
    form.on_recordings_changed([recording])
    assert drain_until(qapp, lambda: form._class_sets.classes() == ["mouse", "plate"])
    assert form._per_class.isChecked()
    ticks = {
        name: tick.isChecked() for name, (tick, _p) in form._class_sets._rows.items()
    }
    assert ticks == {"mouse": False, "plate": True}
    assert form._class_sets.value() == {}  # the set itself is chosen again
