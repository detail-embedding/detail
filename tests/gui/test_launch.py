"""Tests for launch-dialog logic: upstream eligibility + a config widget."""

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.core.registries import TRACKERS  # noqa: E402
from detail.gui.launch import _eligible  # noqa: E402
from detail.gui.workers import BackgroundCall  # noqa: E402
from detail.storage import recording as recording_store  # noqa: E402


class _SyncBackgroundCall(BackgroundCall):
    """Run the launch inline so done/failed fire deterministically in the test
    thread (real QObject signals; only the OS thread is skipped)."""

    def start(self) -> None:  # type: ignore[override]
        self.run()


def _select(widget, recordings: list) -> None:
    """Drive a config widget's recording selection the way the real launch dialog
    does: every eligible recording through ``resolve_eligible_recordings`` /
    ``apply_eligible_recordings`` (§6.5's spec/coverage pickers — a module pick, not a
    checkbox click), then the checked subset through the older ``on_recordings_changed``
    pair. A test that only drives the second half sees an empty spec picker: nothing
    populates it any more except the first."""
    widget.apply_eligible_recordings(
        recordings, widget.resolve_eligible_recordings(recordings)
    )
    widget.on_recordings_changed(recordings)


class _FakeRecording:
    def __init__(self, *, video=False, detection=False, annotated=False):
        self.video_parts = [{}] if video else []
        self.detection_runs = [{"status": "complete"}] if detection else []
        self.tracking_runs = []
        self.embedding_runs = []
        self.is_annotated = annotated


def test_eligible_video_upstream():
    assert _eligible(_FakeRecording(video=True), "video") is True
    assert _eligible(_FakeRecording(video=False), "video") is False


def test_eligible_module_upstream():
    assert _eligible(_FakeRecording(detection=True), "detection") is True
    assert _eligible(_FakeRecording(detection=False), "detection") is False


def test_eligible_behavior_is_a_stream_not_a_run():
    # A "behavior" requirement (the query eval's ground truth) is a stream role, not a
    # {kind}_runs record — it must resolve against is_annotated, else every recording is
    # wrongly ineligible (no behavior_runs attribute exists).
    assert _eligible(_FakeRecording(annotated=True), "behavior") is True
    assert _eligible(_FakeRecording(annotated=False), "behavior") is False


def test_common_complete_run_ids_intersection():
    from detail.gui.modules._common import common_complete_run_ids

    class _S:
        def __init__(self, runs):
            self.detection_runs = runs

    s1 = _S(
        [
            {"id": "det_01", "status": "complete"},
            {"id": "det_02", "status": "complete"},
        ]
    )
    s2 = _S([{"id": "det_01", "status": "complete"}])
    s3 = _S([{"id": "det_01", "status": "running"}])  # not complete → excluded
    assert common_complete_run_ids([s1, s2], "detection") == ["det_01"]
    assert common_complete_run_ids([s1, s3], "detection") == []
    assert common_complete_run_ids([], "detection") == []


def test_detection_config_widget(qapp, monkeypatch, stub_yolo_model):
    from detail.core.dataclasses import ComputeTarget, Prompt
    from detail.gui.modules.detection import _DetectionConfig

    # pin a GPU so the backend default is deterministic across machines/CI
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    widget = _DetectionConfig()
    assert widget._model.findText("sam3") >= 0  # sam3 is offered
    widget._model.setCurrentText("sam3")  # default depends on registry order
    cfg = widget.config()
    assert cfg["model_id"] == "sam3"
    assert isinstance(cfg["prompt"], Prompt)
    assert cfg["prompt"].targets[0].class_name == "mouse"
    assert cfg["prompt"].targets[0].cue.text == "mouse"
    assert isinstance(cfg["compute"], ComputeTarget)
    assert cfg["compute"].backend == "cuda" and cfg["compute"].device_indices == [0]
    assert cfg["conf_threshold"] == 0.5 and cfg["every_n_frames"] == 1
    assert cfg["batch_size"] == 1  # sam3 does not batch → fixed at 1
    # A cpu backend clears the CUDA device index. SAM3 is cuda-only (its combo offers no
    # cpu), so exercise the non-cuda branch with a cpu-capable model — the backend combo
    # tracks the selected model.
    widget._model.setCurrentText(stub_yolo_model)
    widget._compute.backend.setCurrentText("cpu")
    assert widget.config()["compute"].device_indices is None


def test_detection_config_batch_default_by_model(qapp, stub_yolo_model):
    from detail.gui.modules.detection import _DetectionConfig

    widget = _DetectionConfig()
    widget._model.setCurrentText(
        stub_yolo_model
    )  # batch-capable → tunable, defaults to 4
    assert widget._batch.isEnabled()
    assert widget._batch.value() == 4
    widget._model.setCurrentText("sam3")  # not batch-capable → fixed 1, disabled
    assert not widget._batch.isEnabled()
    assert widget._batch.value() == 1
    assert widget.config()["batch_size"] == 1


def test_detection_config_variant_combo(qapp, stub_yolo_model):
    from detail.gui.modules.detection import _DetectionConfig

    widget = _DetectionConfig()
    widget._model.setCurrentText(stub_yolo_model)
    variants = [widget._variant.itemData(i) for i in range(widget._variant.count())]
    assert variants == ["11s", "11m", "11l"]
    assert widget._variant.isEnabled()
    widget._variant.setCurrentText("11m")
    assert widget.config()["model_variant"] == "11m"
    # a model with no size variants → disabled combo, variant None
    widget._model.setCurrentText("sam3")
    assert not widget._variant.isEnabled()
    assert widget.config()["model_variant"] is None


def test_tracking_config_widget(qapp):
    from detail.gui.modules.tracking import _TrackingConfig

    widget = _TrackingConfig()
    # A tracker the form offers: a form test has to drive the form through a choice a user
    # can actually make.
    assert widget._tracker.findText("iou_single_tracker") >= 0
    widget._tracker.setCurrentText("iou_single_tracker")
    # no recordings selected → no detection spec → invalid
    assert widget.validation_error() is not None
    assert widget.config()["detection_spec"] is None

    class _S:
        id = "s1"
        detection_runs = [{"id": "det_01", "status": "complete"}]
        arenas = [{"id": "arena_001", "name": "rig A camera"}]

    _select(widget, [_S(), _S()])
    cfg = widget.config()
    # the picker carries a detection spec (not a per-recording run id); it resolves to
    # each recording's own detection run at launch
    assert cfg["detection_spec"] is not None
    assert cfg["detection_spec"]["module"] == "detection"
    assert widget.covered_recording_ids() == {"s1"}
    assert cfg["tracker_id"] == "iou_single_tracker"
    assert cfg["hyperparams"] == {
        name: default
        for name, (_type, default) in TRACKERS[
            "iou_single_tracker"
        ].HYPERPARAMS_SCHEMA.items()
    }
    assert cfg["arena_name"] is None
    assert widget.validation_error() is None
    widget._arena.setCurrentIndex(widget._arena.findData("rig A camera"))
    assert widget.config()["arena_name"] == "rig A camera"


def test_the_tracking_form_opens_on_the_models_default_tracker(qapp):
    """The combo's opening selection is `configs/tracking.DEFAULT_TRACKER`, not sort order.

    It used to be `addItems(sorted(TRACKERS))` and whatever landed at index 0 — which happens
    to be the same tracker, and that is the whole hazard: registering one whose id sorts
    earlier would silently move the default with no edit to the GUI, and the CLI (which read
    its own hardcoded literal) would go on disagreeing. So this asserts the combo **resolved
    the id**, not merely that it sits on the tracker that sorts first.
    """
    from detail.configs.tracking import DEFAULT_TRACKER
    from detail.gui.modules.tracking import _TrackingConfig

    widget = _TrackingConfig()
    assert widget._tracker.currentText() == DEFAULT_TRACKER
    assert widget.config()["tracker_id"] == DEFAULT_TRACKER

    # The knobs on screen are that tracker's own, so the form a user sees and the run they
    # launch describe the same algorithm.
    schema = TRACKERS[DEFAULT_TRACKER].HYPERPARAMS_SCHEMA
    assert set(widget._hp_widgets) == set(schema)
    assert widget.config()["hyperparams"] == {
        name: default for name, (_type, default) in schema.items()
    }


def test_the_forms_default_is_not_merely_the_first_tracker_alphabetically(qapp):
    """Guard the guard: the assertion above must be about the constant, not about sorting.

    With a tracker registered whose id sorts before the default, index 0 and the default are
    different trackers — and only a form that reads the constant still opens on the right one.
    """
    from detail.configs.tracking import DEFAULT_TRACKER
    from detail.core.protocols import Tracker
    from detail.gui.modules.tracking import _TrackingConfig

    class _Aardvark(Tracker):
        tracker_id = "aaa_probe_tracker"
        supported_backends = ["cpu"]
        HYPERPARAMS_SCHEMA = {"conf_threshold": (float, 0.5)}

        def validate_hyperparams(self, hyperparams: dict) -> dict:
            return dict(hyperparams)

        def track(self, *args, **kwargs) -> None:  # pragma: no cover - never run
            raise AssertionError("probe tracker must not run")

    TRACKERS["aaa_probe_tracker"] = _Aardvark
    try:
        assert sorted(TRACKERS)[0] == "aaa_probe_tracker" != DEFAULT_TRACKER
        widget = _TrackingConfig()
        assert widget._tracker.currentText() == DEFAULT_TRACKER
    finally:
        TRACKERS.pop("aaa_probe_tracker", None)


def test_the_tracking_form_offers_only_surfaced_trackers(qapp):
    from detail import surfaces
    from detail.gui.modules.tracking import _TrackingConfig

    widget = _TrackingConfig()
    offered = [widget._tracker.itemText(i) for i in range(widget._tracker.count())]
    assert offered, "the picker offers nothing at all"
    hidden = [t for t in offered if not surfaces.is_surfaced(t)]
    assert not hidden, f"the form offers hidden trackers: {hidden}"
    # Every offered id is registered. This was a strict subset while `iou_tracker` was
    # registered and withheld; no tracker is withheld now that it has been deleted.
    assert set(offered) <= set(TRACKERS)


def test_a_run_that_used_a_hidden_tracker_says_so_rather_than_silently_defaulting(qapp):
    """Seeding from a run whose tracker this build does not offer must name the substitution.

    Leaving the form on its default under a row reading "seeded from this run" is the
    seeded-form failure `test_start_from.py` exists for — and it is quieter here than usual,
    because the knob loop in `load_config` then drops every key that tracker had (the widgets
    on screen are the default's), so the form ends up neither the run's settings nor visibly
    different from them.
    """
    from detail import surfaces
    from detail.gui.modules.tracking import _TrackingConfig

    # The multi-subject tracker was removed from the tree; tracking runs on disk still name it.
    hidden = "iou_tracker"
    assert hidden not in TRACKERS
    widget = _TrackingConfig()
    assert widget._tracker_note.isHidden(), "the note shows before anything is loaded"

    widget.load_config({"tracker_id": hidden, "hyperparams": {"iou_threshold": 0.9}})
    assert not widget._tracker_note.isHidden()
    text = widget._tracker_note.text()
    assert hidden in text, f"the note does not name the run's tracker: {text!r}"
    assert (
        widget._tracker.currentText() in text
    ), "the note does not say what is selected"
    # The form really is on something it can run, not on a stale id.
    selected = widget.config()["tracker_id"]
    assert surfaces.is_surfaced(selected)

    # And the note tells the truth: the substituted tracker is on **its own** defaults, not
    # carrying the run's value for a knob that merely shares a name. `iou_threshold` exists
    # on both and means frame-to-frame association on one, fragment overlap on the other, so
    # 0.9 travelling across leaves the form part one tracker and part another — with the note
    # above claiming otherwise. Caught by rendering the form and reading it, not by a test.
    schema = TRACKERS[selected].HYPERPARAMS_SCHEMA
    assert widget.config()["hyperparams"] == {
        name: default for name, (_type, default) in schema.items()
    }, "a substituted tracker inherited the previous tracker's knob values"


def test_seeding_from_an_offered_tracker_leaves_no_note(qapp):
    """Guard the guard: a note that is always visible says nothing."""
    from detail.configs.tracking import DEFAULT_TRACKER
    from detail.gui.modules.tracking import _TrackingConfig

    widget = _TrackingConfig()
    widget.load_config({"tracker_id": DEFAULT_TRACKER, "hyperparams": {}})
    assert widget._tracker_note.isHidden()


def test_tracking_anchor_tracker_hyperparams(qapp):
    """Selecting anchor_tracker emits its own keys (not the IoU tracker's), and the
    emitted dict is accepted by AnchorTracker.validate_hyperparams (regression: the
    form used to always send `max_missed_frames`, which anchor_tracker rejects)."""
    from detail.gui.modules.tracking import _TrackingConfig
    from detail.trackers.anchor_tracker import AnchorTracker

    widget = _TrackingConfig()
    assert widget._tracker.findText("anchor_tracker") >= 0
    widget._tracker.setCurrentText("anchor_tracker")
    hyperparams = widget.config()["hyperparams"]
    assert set(hyperparams) == set(AnchorTracker.HYPERPARAMS_SCHEMA)
    assert "max_missed_frames" not in hyperparams
    # The emitted dict must pass validation — the bug was that it did not.
    AnchorTracker().validate_hyperparams(hyperparams)


def test_tracking_conf_below_detection_warns(qapp):
    from detail.gui.modules.tracking import _TrackingConfig

    class _S:
        id = "s1"
        detection_runs = [{"id": "det_01", "status": "complete", "conf_threshold": 0.5}]
        arenas = []

    widget = _TrackingConfig()
    conf = widget._hp_widgets["conf_threshold"]  # built from the tracker's schema
    _select(widget, [_S()])
    conf.setValue(0.2)  # below the detection run's 0.5
    warning = widget._conf_warnings["conf_threshold"]
    assert not warning.isHidden()  # isVisible() is False until shown
    conf.setValue(0.6)  # at/above → no warning
    assert warning.isHidden()


def test_tracking_secondary_conf_below_detection_warns(qapp):
    """A single-subject tracker's secondary_conf floor is also flagged when below the
    detection run's confidence — not just conf_threshold."""
    from detail.gui.modules.tracking import _TrackingConfig

    class _S:
        id = "s1"
        detection_runs = [{"id": "det_01", "status": "complete", "conf_threshold": 0.5}]
        arenas = []

    widget = _TrackingConfig()
    widget._tracker.setCurrentText("iou_single_tracker")  # exposes secondary_conf
    secondary = widget._hp_widgets["secondary_conf"]
    warning = widget._conf_warnings["secondary_conf"]
    _select(widget, [_S()])
    secondary.setValue(0.05)  # below the detection run's 0.5
    assert not warning.isHidden()
    secondary.setValue(0.5)  # at/above → no warning
    assert warning.isHidden()


def test_embedding_config_widget(qapp):
    from detail.gui.modules.embedding import _EmbeddingConfig

    widget = _EmbeddingConfig()
    # no recordings → no tracking spec → invalid
    assert widget.validation_error() is not None
    assert widget.config()["tracking_spec"] is None

    class _S:
        id = "s1"
        tracking_runs = [{"id": "track_01", "status": "complete"}]

        def track_ids(self, run_id):
            return [0, 1, 2]

    _select(widget, [_S(), _S()])
    cfg = widget.config()
    assert cfg["tracking_spec"] is not None
    assert cfg["tracking_spec"]["module"] == "tracking"
    assert cfg["track_ids"] == [0]
    assert cfg["all_tracks"] is False
    # the GUI sends a named constructor; the spec is resolved server-side per
    # recording (so centered recipes size their canvas to each recording's masks)
    assert cfg["constructor"] == "masked_color_centered_contrast"  # the default
    assert "renderer_spec" not in cfg
    assert widget.validation_error() is None
    widget._tracks.setText("")  # no track ids → invalid
    assert widget.validation_error() is not None
    widget._tracks.setText("0, 2, 5")
    assert widget.config()["track_ids"] == [0, 2, 5]
    # All tracks: track ids field is ignored + disabled; config flags it
    widget._all_tracks.setChecked(True)
    assert widget.validation_error() is None
    assert not widget._tracks.isEnabled()
    assert widget.config()["all_tracks"] is True
    assert "Available tracks" in widget._available.text()  # ids shown to the user


def _experiment_with_mixed_recordings(tmp_path):
    """Two fully-eligible recordings (video + complete detection/tracking/embedding runs) plus a
    bare recording that satisfies no module's upstream."""
    proj = Experiment.create("p", tmp_path)
    for sid in ("recording_001", "recording_002"):
        recording = proj.create_recording(sid)
        data = recording_store.read_recording(recording.dir)
        data["video"] = {
            "files": [
                {
                    "index": 0,
                    "file": "video/part_001.mp4",
                    "timestamps": {
                        "generated": None,
                        "user": None,
                        "active": "generated",
                    },
                }
            ]
        }
        for key in ("detection_runs", "tracking_runs", "embedding_runs"):
            data[key] = [{"id": f"{key[:-5]}_01", "status": "complete"}]
        recording_store.write_recording(recording.dir, data)
    proj.create_recording("recording_003")  # bare — eligible for nothing
    return proj


def test_launch_dialog_orders_and_numbers_modules(qapp, tmp_path):
    from detail.gui.launch import LaunchRunDialog

    dialog = LaunchRunDialog(Experiment.create("p", tmp_path))
    combo = dialog._module_combo
    ids = [combo.itemData(i) for i in range(combo.count())]
    labels = {combo.itemData(i): combo.itemText(i) for i in range(combo.count())}
    # pipeline order; the three core stages are numbered, pre/post steps are not
    assert ids.index("detection") < ids.index("tracking") < ids.index("embedding")
    assert labels["detection"].startswith("1. ")
    assert labels["tracking"].startswith("2. ")
    assert labels["embedding"].startswith("3. ")


def test_launch_dialog_seed_preselects_module_and_recordings(qapp, tmp_path):
    from detail.gui.launch import LaunchRunDialog

    proj = _experiment_with_mixed_recordings(tmp_path)  # 001/002 have video, 003 bare
    dialog = LaunchRunDialog(
        proj, seed={"module_id": "detection", "recording_ids": ["recording_001"]}
    )
    assert dialog._module_combo.currentData() == "detection"
    assert dialog._selected_recordings() == ["recording_001"]


def test_refine_carries_the_previous_runs_settings_into_the_form(qapp, tmp_path):
    """The *real* path a user takes, not `load_config` called directly.

    ``tests/gui/test_config_round_trip.py`` proves each form is its own inverse; this proves
    the dialog hands it something it can invert. A run record nests its knobs under ``params``
    and *also* promotes a few keys to the top level, so a form reading either level alone sees
    a config with holes — every implementation used to carry its own two-level fallback, and no
    two agreed on which keys survived. ``_seed_config`` flattens it once.

    The two existing seed tests assert the module and the recordings, which is what Refine got
    right the whole time; the settings, which is what it was for, went untested.
    """
    from detail.gui.launch import LaunchRunDialog

    proj = _experiment_with_mixed_recordings(tmp_path)
    dialog = LaunchRunDialog(
        proj,
        seed={
            "id": "run_007",
            "module_id": "detection",
            "status": "completed",
            "recording_ids": ["recording_001"],
            "params": {"conf_threshold": 0.42, "every_n_frames": 3},
        },
    )
    config = dialog._config.config()
    assert config["conf_threshold"] == pytest.approx(0.42)
    assert config["every_n_frames"] == 3


def test_seed_config_prefers_the_runs_own_params_over_a_promoted_copy(qapp):
    """A promoted top-level key is a *copy* of what ``params`` holds, so ``params`` wins.

    They agree on every run written today. They stop agreeing the moment a runner promotes a
    key it also enriches — and then the form would open on the pre-enrichment value while the
    record shows the resolved one, which is the silent half of this whole class.
    """
    from detail.gui.launch import seed_config

    flat = seed_config(
        {
            "id": "run_007",
            "module_id": "detection",
            "status": "completed",
            "conf_threshold": 0.1,
            "queries": [{"name": "rearing"}],
            "params": {"conf_threshold": 0.42},
        }
    )
    assert flat["conf_threshold"] == pytest.approx(0.42)
    assert flat["queries"] == [{"name": "rearing"}]  # promoted, not in params — kept
    # Record bookkeeping is not config and must not reach a form as if it were.
    assert "id" not in flat and "status" not in flat and "module_id" not in flat


def test_a_next_stage_launch_does_not_inherit_the_upstream_runs_settings(
    qapp, tmp_path
):
    """ "Run tracking →" seeds this dialog with the **detection** run it came from.

    The seed's ``module_id`` is the *next* stage, so the form on screen belongs to a different
    module than the record does. Pushing that record's parameters into it would be worse than
    not seeding at all — a next-step seed carries only recordings, and that absence is the
    signal ``_apply_seed`` keys on.
    """
    from detail.gui.launch import LaunchRunDialog

    proj = _experiment_with_mixed_recordings(tmp_path)
    fresh = LaunchRunDialog(proj)
    fresh._module_combo.setCurrentIndex(fresh._module_combo.findData("tracking"))
    defaults = fresh._config.config()

    seeded = LaunchRunDialog(
        proj,
        seed={"module_id": "tracking", "recording_ids": ["recording_001"]},
    )
    assert seeded._config.config()["every_n_frames"] == defaults["every_n_frames"]


def _config_forms(dialog):
    from detail.gui.modules.registry import ConfigWidget

    holder = dialog._config_holder
    widgets = [holder.itemAt(i).widget() for i in range(holder.count())]
    return [w for w in widgets if isinstance(w, ConfigWidget)]


def test_switching_module_replaces_the_config_form(qapp, tmp_path):
    """It used to *stack* them: tracking's launcher showed detection's arguments above
    its own.

    ``deleteLater`` only schedules the delete, and Qt runs a DeferredDelete solely in an
    event loop at the same or an outer level as the one that posted it. The dialog is
    built inside a slot (level 1) and then opened with ``exec()`` (level 2), so the old
    form survived — visible and laid out — for as long as the dialog was open. Hence the
    check for a *synchronous* removal: after the switch the holder must already be down
    to one form, with the old one unparented."""
    from detail.gui.launch import LaunchRunDialog

    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    detection_form = dialog._config
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("tracking"))

    assert _config_forms(dialog) == [dialog._config]
    assert detection_form.parent() is None


def test_seeding_a_downstream_module_replaces_the_config_form(qapp, tmp_path):
    """The path it was reported from: 'Run tracking →' off a detection run's page seeds
    the dialog, which builds detection's form and then switches to tracking."""
    from detail.gui.launch import LaunchRunDialog

    proj = _experiment_with_mixed_recordings(tmp_path)
    dialog = LaunchRunDialog(
        proj, seed={"module_id": "tracking", "recording_ids": ["recording_001"]}
    )
    assert _config_forms(dialog) == [dialog._config]


def test_switching_analysis_module_replaces_the_config_form(qapp, tmp_path):
    """Same swap, same bug, in the analysis launcher."""
    from detail.gui.launch import LaunchAnalysisDialog

    dialog = LaunchAnalysisDialog(_experiment_with_mixed_recordings(tmp_path))
    if dialog._module_combo.count() < 2:
        pytest.skip("needs two analysis modules to switch between")
    first_form = dialog._config
    dialog._module_combo.setCurrentIndex(1 - dialog._module_combo.currentIndex())

    assert _config_forms(dialog) == [dialog._config]
    assert first_form.parent() is None


def test_launch_run_select_all_checks_only_eligible(qapp, tmp_path):
    from detail.gui.launch import LaunchRunDialog

    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    dialog._select_all.click()  # exercise the real button → slot wiring
    # both video recordings are checked; the bare one (needs video) is left unchecked
    assert set(dialog._selected_recordings()) == {"recording_001", "recording_002"}
    # the config form was told about the new selection (single update, signals blocked)
    assert dialog._config is not None


def test_launch_run_clear_unchecks_all(qapp, tmp_path):
    from detail.gui.launch import LaunchRunDialog

    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    dialog._select_all.click()
    assert dialog._selected_recordings()  # something is checked
    dialog._clear.click()  # exercise the real button → slot wiring
    assert dialog._selected_recordings() == []


def test_launch_analysis_select_all_checks_only_eligible(qapp, tmp_path):
    # Name the module rather than taking whichever sorts first: the fixture's bare recording is
    # ineligible for `query` (it needs an embedding run), which is the property under test. Left
    # implicit, this asserted whatever the default module happened to require that week.
    from detail.gui.launch import LaunchAnalysisDialog

    dialog = LaunchAnalysisDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("query"))
    dialog._select_all_recordings()
    assert set(dialog._selected_recordings()) == {"recording_001", "recording_002"}


def test_launch_analysis_clear_unchecks_all(qapp, tmp_path):
    from detail.gui.launch import LaunchAnalysisDialog

    dialog = LaunchAnalysisDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._select_all.click()
    assert dialog._selected_recordings()  # something is checked
    dialog._clear.click()
    assert dialog._selected_recordings() == []


def test_launch_narrows_to_covered_recordings(qapp, tmp_path, monkeypatch):
    """A spec that only some selected recordings have narrows the run to those, after
    confirming the reduced set; declining aborts; full coverage runs untouched (§6.5).
    """
    from PyQt6.QtWidgets import QMessageBox

    from detail.gui.launch import LaunchRunDialog

    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    selected = ["recording_001", "recording_002"]

    dialog._config.covered_recording_ids = lambda: {"recording_001"}
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    assert dialog._narrow_to_covered(selected) == ["recording_001"]

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )
    assert dialog._narrow_to_covered(selected) is None  # declined → abort

    dialog._config.covered_recording_ids = lambda: None  # all covered → no prompt
    assert dialog._narrow_to_covered(selected) == selected

    dialog._config.covered_recording_ids = lambda: set()  # none covered → warn + abort
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    assert dialog._narrow_to_covered(selected) is None


def test_launch_analysis_narrows_source_recording_ids(qapp, tmp_path, monkeypatch):
    """LaunchAnalysisDialog runs over only the recordings covered by the chosen spec —
    the narrowed set flows into `source_recording_ids` (§6.5)."""
    from PyQt6.QtWidgets import QMessageBox

    from detail.gui import launch
    from detail.gui.launch import LaunchAnalysisDialog

    monkeypatch.setattr(launch, "BackgroundCall", _SyncBackgroundCall)
    dialog = LaunchAnalysisDialog(_experiment_with_mixed_recordings(tmp_path))
    monkeypatch.setattr(
        dialog, "_selected_recordings", lambda: ["recording_001", "recording_002"]
    )
    dialog._config.covered_recording_ids = lambda: {"recording_001"}
    dialog._config.validation_error = lambda: None
    dialog._config.config = lambda: {"tracking_spec": {"module": "tracking"}}
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )

    captured = {}

    class _Run:
        id = "an_01"

    monkeypatch.setattr(
        type(dialog._experiment),
        "run_analysis",
        lambda self, module_id, config: (captured.update(config), _Run())[1],
    )
    dialog._run()
    assert captured["source_recording_ids"] == ["recording_001"]  # narrowed to covered


def test_launch_names_each_recording_as_it_validates(qapp, tmp_path, monkeypatch):
    """The status line is replaced by each progress report, so a validation that takes
    minutes says which recording it is on instead of one static line that reads as hung.
    Drives the real dialog through `_run` with a `run_processing` that reports the way the
    API does, and records the line after every report."""
    from detail.gui import launch
    from detail.gui.launch import LaunchRunDialog

    monkeypatch.setattr(launch, "BackgroundCall", _SyncBackgroundCall)
    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    dialog._select_all.click()
    dialog._config.validation_error = lambda: None
    dialog._config.config = lambda: {}
    dialog._config.covered_recording_ids = lambda: None
    seen, order = [], []

    class _Run:
        id = "run_001"

    def _reporting(self, mid, recordings, config, progress=None):
        for i, sid in enumerate(recordings):
            order.append(sid)
            progress("Validating", i / len(recordings), f"{sid} ({i + 1}/2)")
            seen.append(dialog._status.text())
        return _Run()

    monkeypatch.setattr(type(dialog._experiment), "run_processing", _reporting)
    monkeypatch.setattr(dialog, "accept", lambda: None)
    dialog._run()
    assert len(order) == 2
    assert seen == [f"Validating — {sid} ({i + 1}/2)" for i, sid in enumerate(order)]


def test_launch_run_spawns_off_thread(qapp, tmp_path, monkeypatch):
    """`_run` validates + spawns via a worker (off the UI thread), then on
    success stores the handle and accepts — no synchronous run_processing on the UI
    thread (the multi-second freeze this fixes)."""
    from detail.gui import launch
    from detail.gui.launch import LaunchRunDialog

    monkeypatch.setattr(launch, "BackgroundCall", _SyncBackgroundCall)
    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    dialog._select_all.click()
    dialog._config.validation_error = lambda: None
    dialog._config.config = lambda: {"model_id": "sam3"}
    dialog._config.covered_recording_ids = lambda: None  # all covered → no narrowing

    class _Run:
        id = "det_run_01"

    captured = {}
    monkeypatch.setattr(
        type(dialog._experiment),
        "run_processing",
        lambda self, mid, recordings, config, progress=None: (
            captured.update(mid=mid, recordings=list(recordings), config=config),
            _Run(),
        )[1],
    )
    accepted = []
    monkeypatch.setattr(dialog, "accept", lambda: accepted.append(True))
    dialog._run()
    assert captured["mid"] == "detection"
    assert set(captured["recordings"]) == {"recording_001", "recording_002"}
    assert captured["config"] == {"model_id": "sam3"}
    assert dialog.run_handle.id == "det_run_01"
    assert accepted == [True]


def test_launch_run_error_surfaces_and_reenables(qapp, tmp_path, monkeypatch):
    """A validation error raised by run_processing (off-thread) surfaces via a critical
    dialog and re-enables the form — it must not accept or leave the form frozen."""
    from PyQt6.QtWidgets import QMessageBox

    from detail.gui import launch
    from detail.gui.launch import LaunchRunDialog

    monkeypatch.setattr(launch, "BackgroundCall", _SyncBackgroundCall)
    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    dialog._select_all.click()
    dialog._config.validation_error = lambda: None
    dialog._config.config = lambda: {}
    dialog._config.covered_recording_ids = lambda: None

    def _boom(self, mid, recordings, config, progress=None):
        raise ValueError("no weights")

    monkeypatch.setattr(type(dialog._experiment), "run_processing", _boom)
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.append(a[-1]))
    accepted = []
    monkeypatch.setattr(dialog, "accept", lambda: accepted.append(True))
    dialog._run()
    assert shown == ["no weights"]  # the ValueError message, surfaced verbatim
    assert accepted == []  # not accepted on failure
    assert dialog._ok_button.isEnabled()  # form re-enabled after the failure
    assert dialog._status.text() == ""  # busy status cleared


def test_the_launch_dialog_says_which_recording_validation_is_on(
    qapp, tmp_path, monkeypatch
):
    """Validating a large selection is minutes of reading; a status that read "Validating N
    recording(s)…" throughout is indistinguishable from a hang (the 71-recording embedding launch
    that looked stuck). The status follows the callback the launch hands to `run_processing`.
    """
    from detail.gui import launch
    from detail.gui.launch import LaunchRunDialog

    monkeypatch.setattr(launch, "BackgroundCall", _SyncBackgroundCall)
    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    dialog._select_all.click()
    dialog._config.validation_error = lambda: None
    dialog._config.config = lambda: {}
    dialog._config.covered_recording_ids = lambda: None

    class _Run:
        id = "det_run_01"

    shown = []

    def _validating(self, mid, recordings, config, progress=None):
        shown.append(dialog._status.text())  # before the first report: the opening line
        progress("Validating recordings", 0.0, "1 of 2")
        shown.append(dialog._status.text())
        progress("Validating recordings", 1.0, "2 of 2")
        shown.append(dialog._status.text())
        progress("Estimating disk space", 0.0, "1 of 2")
        shown.append(dialog._status.text())
        return _Run()

    monkeypatch.setattr(type(dialog._experiment), "run_processing", _validating)
    monkeypatch.setattr(dialog, "accept", lambda: None)
    dialog._run()

    assert shown == [
        "Validating 2 recording(s)…",
        "Validating recordings — 1 of 2",
        "Validating recordings — 2 of 2",
        "Estimating disk space — 1 of 2",
    ]


def test_the_launch_anyway_retry_reports_progress_too(qapp, tmp_path, monkeypatch):
    """A launch refused for space and then launched anyway runs the same two passes again;
    `work_anyway` is a second lambda, which is where a callback goes missing."""
    from PyQt6.QtWidgets import QMessageBox

    from detail.core.disk_space import InsufficientSpaceError, SpaceEstimate
    from detail.gui import launch
    from detail.gui.launch import LaunchRunDialog

    monkeypatch.setattr(launch, "BackgroundCall", _SyncBackgroundCall)
    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    dialog._select_all.click()
    dialog._config.validation_error = lambda: None
    dialog._config.config = lambda: {}
    dialog._config.covered_recording_ids = lambda: None
    refusal = InsufficientSpaceError(
        SpaceEstimate(bytes=2, free_bytes=1, where="C:", per_recording={})
    )

    class _Run:
        id = "det_run_01"

    reported = []

    def _run_processing(
        self, mid, recordings, config, *, ignore_free_space=False, progress=None
    ):
        progress("Validating recordings", 0.0, "1 of 2")
        reported.append(ignore_free_space)
        if not ignore_free_space:
            raise refusal
        return _Run()

    monkeypatch.setattr(type(dialog._experiment), "run_processing", _run_processing)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(dialog, "accept", lambda: None)
    dialog._run()

    assert reported == [False, True], "both launches reported through the callback"
    assert dialog._status.text() == "Validating recordings — 1 of 2"


def test_covered_recording_ids_across_pickers(qapp):
    """Each picker's `covered_recording_ids` returns the set of selected recordings that
    have the chosen upstream spec — and `None` (all) for the root/"Original" choice."""
    from detail.gui.modules.analysis_location import _LocationExportConfig
    from detail.gui.modules.detection import _DetectionConfig
    from detail.gui.modules.embedding import _EmbeddingConfig

    trk = {
        "id": "track_01", "status": "complete",
        "tracker_id": "anchor_tracker", "arena_ids": [], "hyperparams": {},
    }  # fmt: skip

    class _S:
        video_fps = 30.0  # the detection form reads it for its stride note

        def __init__(self, sid, **runs):
            self.id = sid
            for key, value in runs.items():
                setattr(self, key, value)

        def track_ids(self, _run_id):
            return [0]

    # analysis (tracking spec): only s1 has a tracking run → covers {s1}
    loc = _LocationExportConfig()
    _select(loc, [_S("s1", tracking_runs=[trk]), _S("s2", tracking_runs=[])])
    assert loc.covered_recording_ids() == {"s1"}

    # embedding (tracking spec): same
    emb = _EmbeddingConfig()
    _select(emb, [_S("s1", tracking_runs=[trk]), _S("s2", tracking_runs=[])])
    assert emb.covered_recording_ids() == {"s1"}

    # detection reads the recording's video directly, so every selected recording is
    # covered whatever stride it runs at.
    det = _DetectionConfig()
    det.on_recordings_changed([_S("s1"), _S("s2")])
    assert det.covered_recording_ids() is None


# --- Stage 4: off-thread recording-change resolution -------------------------- #
def test_embedding_resolve_apply_split(qapp):
    """resolve_eligible_recordings gathers the spec groups + per-recording track ids
    WITHOUT touching widgets (it runs off the UI thread); apply_eligible_recordings
    installs them. Over every *eligible* recording, not just the checked ones — the
    §6.5 spec/coverage pickers' own pair (checked-selection reactions still go through
    on_recordings_changed, unchanged)."""
    from detail.gui.modules.embedding import _EmbeddingConfig

    class _S:
        id = "s1"
        tracking_runs = [{"id": "track_01", "status": "complete"}]

        def track_ids(self, run_id):
            return [0, 1, 2]

    widget = _EmbeddingConfig()
    payload = widget.resolve_eligible_recordings([_S()])
    assert widget._tracking.count() == 0  # resolve did not mutate the UI
    assert payload["tracks"] == {("s1", "track_01"): [0, 1, 2]}

    widget.apply_eligible_recordings([_S()], payload)
    assert widget._tracking.count() >= 1  # combo populated on apply
    widget.on_recordings_changed([_S()])  # the checked-selection reaction
    assert "Available tracks" in widget._available.text()


def test_launch_run_light_config_applies_inline(qapp, tmp_path, monkeypatch):
    """A config that does not override resolve_recordings (detection) applies on the UI
    thread — no BackgroundCall — since its per-selection work is cheap."""
    from detail.gui import launch
    from detail.gui.launch import LaunchRunDialog

    spawned = []

    class _RecordingCall(_SyncBackgroundCall):
        def start(self):  # type: ignore[override]
            spawned.append(True)
            super().start()

    monkeypatch.setattr(launch, "BackgroundCall", _RecordingCall)
    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    seen = []
    dialog._config.on_recordings_changed = lambda recordings: seen.append(
        {s.id for s in recordings}
    )
    dialog._select_all.click()
    assert spawned == []  # light config → inline, no background thread
    assert seen and seen[-1] == {"recording_001", "recording_002"}


def test_launch_run_heavy_config_resolves_off_thread(qapp, tmp_path, monkeypatch):
    """A config that overrides resolve_recordings (like embedding) is resolved through
    BackgroundCall — resolve runs first (off-thread), then apply on the UI thread."""
    from detail.gui import launch
    from detail.gui.launch import LaunchRunDialog
    from detail.gui.modules.registry import ConfigWidget

    monkeypatch.setattr(launch, "BackgroundCall", _SyncBackgroundCall)
    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))

    events = []

    class _Heavy(ConfigWidget):
        def resolve_recordings(self, recordings):
            events.append(("resolve", tuple(sorted(s.id for s in recordings))))
            return tuple(sorted(s.id for s in recordings))

        def apply_recordings(self, recordings, resolved):
            events.append(("apply", resolved))

    dialog._config = _Heavy()
    dialog._select_all.click()
    ids = ("recording_001", "recording_002")
    assert ("resolve", ids) in events and ("apply", ids) in events
    assert events.index(("resolve", ids)) < events.index(("apply", ids))


def test_stale_recording_resolution_is_dropped(qapp, tmp_path, monkeypatch):
    """When the selection changes again before an off-thread resolution finishes, the
    stale result is dropped (its apply is skipped) — the generation guard prevents an
    out-of-order combo."""
    from detail.gui import launch
    from detail.gui.launch import LaunchRunDialog
    from detail.gui.modules.registry import ConfigWidget

    pending = []

    class _Deferred(launch.BackgroundCall):
        def start(self):  # type: ignore[override]
            pending.append(self)  # capture; fire manually below

    monkeypatch.setattr(launch, "BackgroundCall", _Deferred)

    applied = []

    class _Heavy(ConfigWidget):
        def resolve_recordings(self, recordings):
            return tuple(sorted(s.id for s in recordings))

        def apply_recordings(self, recordings, resolved):
            applied.append(resolved)

    dialog = LaunchRunDialog(_experiment_with_mixed_recordings(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    dialog._config = _Heavy()

    dialog._select_all.click()  # generation 1 → pending[0]
    dialog._update_config_recordings()  # generation 2 → pending[1]
    assert len(pending) == 2

    pending[0].run()  # stale (gen 1); current is gen 2 → apply skipped
    assert applied == []
    pending[1].run()  # current (gen 2) → applied
    assert applied == [("recording_001", "recording_002")]


class _DetectedAt4:
    """A recording whose only detection run used every 4th frame of a 30 fps source."""

    id = "s1"
    video_fps = 30.0
    arenas: list = []
    detection_runs = [
        {
            "id": "det_01",
            "status": "complete",
            "model_id": "sam3",
            "conf_threshold": 0.5,
            "every_n_frames": 4,
        }
    ]


def test_tracking_form_shows_the_missed_tolerance_in_seconds(qapp):
    """A missed-frame tolerance counts *processed* frames, so 30 is 1 s at every frame and
    4 s at every 4th. The form says which, rather than leaving one number to mean
    different things on different runs.

    Driven through `iou_single_tracker`, whose spelling of the knob is `max_missed`; the
    other spelling, `max_missed_frames`, belongs to a tracker this build does not offer, and
    a form test has to use a tracker the form offers."""
    from detail.gui.modules.tracking import _TrackingConfig

    widget = _TrackingConfig()
    widget._tracker.setCurrentText("iou_single_tracker")
    _select(widget, [_DetectedAt4()])

    assert widget._every_n.value() == 4  # inherited from the detection run
    widget._hp_widgets["max_missed"].setValue(30)
    assert widget._missed_notes["max_missed"].text() == "4.0 s at this rate"

    widget._every_n._spin.setValue(8)
    assert widget._missed_notes["max_missed"].text() == "8.0 s at this rate"


def test_tracking_form_refuses_a_cadence_its_detection_run_cannot_produce(qapp):
    from detail.gui.modules.tracking import _TrackingConfig

    widget = _TrackingConfig()
    _select(widget, [_DetectedAt4()])
    assert widget.validation_error() is None
    assert widget.config()["every_n_frames"] == 4

    widget._every_n._spin.setValue(6)  # typed; the arrows step 4 -> 8 -> 12
    assert "use 4, 8, 12" in widget.validation_error()


def test_every_model_form_offers_its_declared_default_variant(qapp, monkeypatch):
    """The size picker must land on the size the model says is its default.

    `dinov3_convnext_base` lists tiny, small, base, large and declares *base*. Filling the
    combo in dict order and leaving it on index 0 meant a form nobody touched submitted
    `convnext_tiny` — a size the documented install never fetches — and the run failed
    validate with "is not downloaded". That is the walkthrough's "model not installed" bug.
    """
    from detail.core.registries import EMBEDDING_MODELS, MODELS
    from detail.gui.modules.detection import _DetectionConfig
    from detail.gui.modules.embedding import _EmbeddingConfig

    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)

    for widget, registry in (
        (_EmbeddingConfig(), EMBEDDING_MODELS),
        (_DetectionConfig(), MODELS),
    ):
        for model_id, model_cls in sorted(registry.items()):
            if not model_cls.variants():
                continue
            widget._model.setCurrentText(model_id)
            assert widget._variant.currentData() == model_cls.default_variant(), (
                f"{model_id}: form offers {widget._variant.currentData()!r}, "
                f"model declares {model_cls.default_variant()!r}"
            )


def test_tracking_form_calls_the_polygon_a_field_of_view(qapp):
    """The concept is the field of view everywhere else — concepts, the glossary, the
    editor. This form was the last place still saying "Arena"."""
    from PyQt6.QtWidgets import QFormLayout, QLabel

    from detail.gui.modules.tracking import _TrackingConfig

    widget = _TrackingConfig()
    layout = widget.findChild(QFormLayout)
    labels = []
    for row in range(layout.rowCount()):
        item = layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
        if item is not None and isinstance(item.widget(), QLabel):
            labels.append(item.widget().text())
        elif item is not None and item.widget() is not None:
            labels += [c.text() for c in item.widget().findChildren(QLabel)]
    assert any("Field of view" in text for text in labels), labels
    assert not any(text.startswith("Arena") for text in labels), labels


def test_multi_subject_is_advanced_but_still_recorded(qapp):
    """``num_subjects`` is a tracker knob like any other: reachable under the collapsed
    *Tracker parameters* section, not removed from the form.

    It used to have no widget at all, on the grounds that the trackers reject anything but
    1 — which left a real knob with no way to reach it and no way to see what it was set
    to. Its default still has to reach the config either way, because ``hyperparams`` is a
    run identity field and a config that omits it would not match one that states it.
    """
    from detail.gui.forms import AdvancedSection
    from detail.gui.modules.tracking import _KNOBS, _TrackingConfig
    from detail.trackers.anchor_tracker import AnchorTracker

    widget = _TrackingConfig()
    widget._tracker.setCurrentText("anchor_tracker")

    assert "num_subjects" in AnchorTracker.HYPERPARAMS_SCHEMA  # a real knob...
    assert "num_subjects" in widget._hp_widgets  # ...that now has a widget
    section = widget.findChild(AdvancedSection)
    assert _KNOBS in section.groups()  # ...under the Tracker parameters heading
    assert not section.is_expanded()  # ...inside the collapsed Advanced Settings
    hyperparams = widget.config()["hyperparams"]
    assert hyperparams["num_subjects"] == 1  # ...at its default in the config
    AnchorTracker().validate_hyperparams(hyperparams)


# --------------------------------------------------------------------------- #
# Upstream coverage is visible while you choose, not only at submit
# --------------------------------------------------------------------------- #
def _experiment_with_divergent_detections(tmp_path):
    """Two recordings whose detection runs differ: only one has the 0.10 run.

    The shape that produced "no complete detection run matching the selected spec in
    recording 'poopie2'" — both recordings are *eligible* (each has a complete detection
    run), but the run you pick exists in only one of them.
    """
    proj = Experiment.create("p", tmp_path)
    for sid, runs in (
        ("has_both", [("det_01", 0.1), ("det_02", 0.5)]),
        ("has_one", [("det_01", 0.5)]),
    ):
        recording = proj.create_recording(sid)
        data = recording_store.read_recording(recording.dir)
        data["video"] = {
            "files": [
                {
                    "index": 0,
                    "file": "video/part_001.mp4",
                    "timestamps": {
                        "generated": None,
                        "user": None,
                        "active": "generated",
                    },
                }
            ]
        }
        data["detection_runs"] = [
            {
                "id": run_id,
                "status": "complete",
                "model_id": "sam3",
                "model_variant": None,
                "weights_source": "pretrained",
                "conf_threshold": conf,
                "class_conf_thresholds_fingerprint": "ccf",
                "prompt_fingerprint": "fp",
                "prompt_classes": ["mouse"],
                "every_n_frames": 1,
                "created_at": f"2026-01-0{i + 1}T00:00:00Z",
            }
            for i, (run_id, conf) in enumerate(runs)
        ]
        recording_store.write_recording(recording.dir, data)
    return proj


def _row(dialog, recording_id):
    from PyQt6.QtCore import Qt

    for i in range(dialog._recordings.count()):
        if dialog._recording_ids[i] == recording_id:
            item = dialog._recordings.item(i)
            return item, bool(item.flags() & Qt.ItemFlag.ItemIsEnabled)
    raise AssertionError(f"{recording_id} not in the list")


def test_a_recording_you_have_not_checked_is_never_greyed(qapp, tmp_path, monkeypatch):
    """The form's coverage is an answer about the recordings it was given — the checked
    ones — so a row outside that set has no answer yet, not a "no". Read as "no", checking
    one recording greyed every other as "not in the selected run" (they had the same run),
    and unchecking one after *Select all* greyed it for good, since a greyed row cannot be
    checked again. Both recordings here share the 0.5 detection run."""
    from PyQt6.QtCore import Qt

    from detail.gui import launch
    from detail.gui.launch import LaunchRunDialog

    monkeypatch.setattr(launch, "BackgroundCall", _SyncBackgroundCall)
    dialog = LaunchRunDialog(_experiment_with_divergent_detections(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("tracking"))

    item, _ = _row(dialog, "has_one")
    item.setCheckState(Qt.CheckState.Checked)
    item, enabled = _row(dialog, "has_both")
    assert enabled and item.text() == "has_both"

    dialog._select_all.click()
    item.setCheckState(Qt.CheckState.Unchecked)
    item, enabled = _row(dialog, "has_both")
    assert enabled and item.text() == "has_both"
    item.setCheckState(Qt.CheckState.Checked)  # and it can come back
    assert set(dialog._selected_recordings()) == {"has_one", "has_both"}


def test_a_recording_without_the_chosen_upstream_run_is_greyed_out(
    qapp, tmp_path, monkeypatch
):
    """It used to stay checked and enabled, and you found out at submit — as a prompt, or
    (launching another way) as a ValueError from run_processing after pressing Run."""
    from PyQt6.QtCore import Qt

    from detail.gui import launch
    from detail.gui.launch import LaunchRunDialog

    # The tracking form resolves its selection off the UI thread now — it reads disk per
    # recording — so the detection combo is filled by a worker rather than inline. Run that
    # worker synchronously, which is what `_SyncBackgroundCall` is for.
    monkeypatch.setattr(launch, "BackgroundCall", _SyncBackgroundCall)

    proj = _experiment_with_divergent_detections(tmp_path)
    dialog = LaunchRunDialog(proj)
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("tracking"))
    for i in range(dialog._recordings.count()):
        dialog._recordings.item(i).setCheckState(Qt.CheckState.Checked)

    combo = dialog._config._detection
    covering_one = next(i for i in range(combo.count()) if "1/2" in combo.itemText(i))
    combo.setCurrentIndex(covering_one)

    item, enabled = _row(dialog, "has_one")
    assert not enabled
    assert "not in the selected run" in item.text()
    assert "has_one" not in dialog._selected_recordings()  # cannot reach the run
    assert "has_both" in dialog._selected_recordings()


def test_switching_back_restores_the_recording(qapp, tmp_path, monkeypatch):
    """The first cut unchecked uncovered rows — which shrank the set the spec groups are
    computed from, so a row could never come back. It must recover."""
    from PyQt6.QtCore import Qt

    from detail.gui import launch
    from detail.gui.launch import LaunchRunDialog

    # As above: tracking's selection reaction is off-thread, so drive it inline here.
    monkeypatch.setattr(launch, "BackgroundCall", _SyncBackgroundCall)

    proj = _experiment_with_divergent_detections(tmp_path)
    dialog = LaunchRunDialog(proj)
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("tracking"))
    for i in range(dialog._recordings.count()):
        dialog._recordings.item(i).setCheckState(Qt.CheckState.Checked)

    combo = dialog._config._detection
    combo.setCurrentIndex(
        next(i for i in range(combo.count()) if "1/2" in combo.itemText(i))
    )
    assert not _row(dialog, "has_one")[1]

    combo.setCurrentIndex(
        next(i for i in range(combo.count()) if "2/2" in combo.itemText(i))
    )
    item, enabled = _row(dialog, "has_one")
    assert enabled
    assert item.text() == "has_one"  # the marker is gone, not stacked
    assert set(dialog._selected_recordings()) == {"has_both", "has_one"}


def test_tracking_class_combo_populates_and_round_trips(qapp, tmp_path):
    """The class filter (§13): populates from the *selected* detection spec's real
    classes, rebuilds when that selection changes, narrows coverage, and round-trips
    through config()/load_config() — the GUI half of the class-filtered tracking
    feature (the pipeline half is tests/pipeline/test_tracking.py)."""
    from detail.gui.modules.tracking import _TrackingConfig
    from detail.storage.zarr_io import DetectionPartWriter
    from tests import waits

    proj = Experiment.create("p", tmp_path)
    sess = proj.create_recording("recording_001")
    sdir = sess.dir
    det_dir = sdir / "detections" / "det_01"
    with DetectionPartWriter(det_dir / "part_001.zarr") as w:
        w.add_row(0, 0, 0, "black mouse", 0.9, (0.1, 0.1, 0.2, 0.2), "")
        w.add_row(0, 0, 1, "white mouse", 0.8, (0.6, 0.6, 0.7, 0.7), "")
    data = recording_store.read_recording(sdir)
    data.setdefault("detection_runs", []).append(
        {
            "id": "det_01",
            "status": "complete",
            "output": "detections/det_01/",
            "model_id": "sam3",
        }
    )
    recording_store.write_recording(sdir, data)

    widget = _TrackingConfig()
    _select(widget, [sess])
    waits.until(qapp, lambda: widget._class.count() == 3, "class combo populating")
    names = [widget._class.itemData(i) for i in range(widget._class.count())]
    assert names == [None, "black mouse", "white mouse"]

    widget._class.setCurrentIndex(widget._class.findData("white mouse"))
    assert widget.config()["class_name"] == "white mouse"
    assert widget.covered_recording_ids() == {"recording_001"}

    # A second, single-class detection run: selecting it rebuilds the class list —
    # unlike the arena list, this is not a fixed property of the recording.
    det2_dir = sdir / "detections" / "det_02"
    with DetectionPartWriter(det2_dir / "part_001.zarr") as w:
        w.add_row(0, 0, 0, "mouse", 0.9, (0.1, 0.1, 0.2, 0.2), "")
    data = recording_store.read_recording(sdir)
    data["detection_runs"].append(
        {
            "id": "det_02",
            "status": "complete",
            "output": "detections/det_02/",
            "model_id": "sam3",
        }
    )
    recording_store.write_recording(sdir, data)
    _select(widget, [sess])
    other_index = next(
        i
        for i in range(widget._detection.count())
        if widget._detection.itemData(i) != widget._detection.currentData()
    )
    widget._detection.setCurrentIndex(other_index)
    waits.until(qapp, lambda: widget._class.count() == 2, "class combo rebuilding")
    assert [widget._class.itemData(i) for i in range(widget._class.count())] == [
        None,
        "mouse",
    ]

    # Round-trips through config()/load_config() like arena_name — a fresh recording
    # with only the two-class run, so the default detection selection is unambiguous.
    sess3 = proj.create_recording("recording_003")
    with DetectionPartWriter(
        sess3.dir / "detections" / "det_01" / "part_001.zarr"
    ) as w:
        w.add_row(0, 0, 0, "black mouse", 0.9, (0.1, 0.1, 0.2, 0.2), "")
        w.add_row(0, 0, 1, "white mouse", 0.8, (0.6, 0.6, 0.7, 0.7), "")
    data = recording_store.read_recording(sess3.dir)
    data.setdefault("detection_runs", []).append(
        {
            "id": "det_01",
            "status": "complete",
            "output": "detections/det_01/",
            "model_id": "sam3",
        }
    )
    recording_store.write_recording(sess3.dir, data)

    restored = _TrackingConfig()
    _select(restored, [sess3])
    waits.until(qapp, lambda: restored._class.count() == 3, "class combo populating")
    restored.load_config({"class_name": "black mouse"})
    assert restored.config()["class_name"] == "black mouse"


def test_coverage_marking_leaves_ineligible_rows_alone(qapp, tmp_path):
    """A recording with no detection run at all is disabled for *eligibility*. Coverage
    must not resurrect it — two different reasons, one flag."""
    from PyQt6.QtCore import Qt

    from detail.gui.launch import LaunchRunDialog

    proj = _experiment_with_divergent_detections(tmp_path)
    proj.create_recording("bare")  # no runs at all
    dialog = LaunchRunDialog(proj)
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("tracking"))
    for i in range(dialog._recordings.count()):
        item = dialog._recordings.item(i)
        if item.flags() & Qt.ItemFlag.ItemIsEnabled:
            item.setCheckState(Qt.CheckState.Checked)

    item, enabled = _row(dialog, "bare")
    assert not enabled
    assert "needs detection" in item.text()
    assert "bare" not in dialog._selected_recordings()


def test_the_analysis_picker_leads_with_the_modules_people_reach_for():
    """Order, not alphabet: the picker opened on `Clustering eval` because `c` sorts first.

    Location export, then pose clustering, then behavior scan — the order a study actually
    goes in. Everything else follows alphabetically so a new module has a defined place
    without anyone editing a list.
    """
    from detail.gui.launch import _ordered_analysis_ids

    ordered = _ordered_analysis_ids(
        ["query_eval", "cluster_eval", "query", "location_export"]
    )
    assert ordered[:2] == ["location_export", "query"]
    assert ordered[2:] == ["cluster_eval", "query_eval"]  # the rest, alphabetically


def test_the_analysis_picker_loses_nothing_to_the_ordering():
    """Every surfaced module still appears exactly once — an ordering that drops one is the
    bug a hand-written list invites."""
    from detail.gui.modules import registry
    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES
    from detail.gui.launch import _ordered_analysis_ids

    surfaced = registry.surfaced(ANALYSIS_GUI_MODULES)
    ordered = _ordered_analysis_ids(surfaced)
    assert sorted(ordered) == sorted(surfaced)
    assert len(ordered) == len(set(ordered))


def test_a_pending_module_in_the_order_is_skipped_not_crashed():
    """`pose_clustering` (§22) is listed while it is not registered (removed, expected back), so it
    takes its slot the day it registers again. An id that is not there must simply not appear.
    """
    from detail.gui.launch import _ANALYSIS_ORDER, _ordered_analysis_ids

    assert "pose_clustering" in _ANALYSIS_ORDER
    assert _ordered_analysis_ids(["query"]) == ["query"]


def test_the_field_of_view_picker_offers_names_not_ids(qapp):
    """The dropdown showed `arena_001`, which is both unreadable and wrong.

    Unreadable because nobody named their field of view `arena_001`; wrong because the id is
    minted per recording, so one entry meant a different polygon on each of them. Both go
    away together — the label *is* the identity now.
    """
    from detail.gui.modules.tracking import _TrackingConfig

    class _R:
        def __init__(self, rid, arenas):
            self.id = rid
            self.detection_runs = [{"id": "det_01", "status": "complete"}]
            self.arenas = arenas

    # the cohort whose ids do not line up: one recording kept an old arena, so its copy of
    # the shared field of view is arena_002 while the other's is arena_001
    widget = _TrackingConfig()
    _select(
        widget,
        [
            _R(
                "tracked",
                [
                    {"id": "arena_001", "name": "old rig"},
                    {"id": "arena_002", "name": "rig A camera"},
                ],
            ),
            _R("fresh", [{"id": "arena_001", "name": "rig A camera"}]),
        ],
    )
    labels = [widget._arena.itemText(i) for i in range(widget._arena.count())]
    assert not any("arena_00" in label for label in labels), labels
    assert any(label.startswith("rig A camera") for label in labels)
    # coverage is surfaced per option, so a name only one recording has says so
    assert any("rig A camera  ·  2/2 recordings" == label for label in labels)
    assert any("old rig  ·  1/2 recordings" == label for label in labels)


def test_a_recording_without_the_chosen_field_of_view_is_narrowed_out(qapp):
    """Coverage intersects with the detection spec's — two independent narrowings.

    A recording with the detection run but no arena of that name would otherwise be
    launched and tracked whole-frame, which is a different run from the one asked for.
    """
    from detail.gui.modules.tracking import _TrackingConfig

    class _R:
        def __init__(self, rid, arenas):
            self.id = rid
            self.detection_runs = [{"id": "det_01", "status": "complete"}]
            self.arenas = arenas

    widget = _TrackingConfig()
    _select(
        widget,
        [
            _R("has_it", [{"id": "arena_001", "name": "rig A camera"}]),
            _R("lacks_it", [{"id": "arena_001", "name": "something else"}]),
        ],
    )
    # whole frame: both recordings are covered
    assert widget.covered_recording_ids() == {"has_it", "lacks_it"}

    widget._arena.setCurrentIndex(widget._arena.findData("rig A camera"))
    assert widget.covered_recording_ids() == {"has_it"}


def test_the_field_of_view_selection_survives_start_from_a_previous_run(qapp):
    """It used to be refused outright, and the docstring said why: the record stored
    `arena_001`, which is a different polygon on the next recording, so restoring it would
    bind to the wrong one without anything failing. A name is the same thing wherever it
    appears, so the selection can come back."""
    from detail.gui.modules.tracking import _TrackingConfig
    from detail.pipeline.tracking import TrackingModule

    class _R:
        id = "s1"
        detection_runs = [{"id": "det_01", "status": "complete"}]
        arenas = [{"id": "arena_002", "name": "rig A camera"}]

    widget = _TrackingConfig()
    _select(widget, [_R()])
    widget._arena.setCurrentIndex(widget._arena.findData("rig A camera"))

    # through the real writer, not a hand-built dict: config() -> the run record
    record = TrackingModule().extra_record_fields(
        {**widget.config(), "detection_run_id": "det_01", "arena_ids": ["arena_002"]}
    )
    assert record["arena_name"] == "rig A camera"

    fresh = _TrackingConfig()
    _select(fresh, [_R()])
    fresh.load_config(record)
    assert fresh.config()["arena_name"] == "rig A camera"
