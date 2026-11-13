"""The ``keypoint_track`` launch form and run viewer (§26 "Surfaces").

The form is built through the module-UI registry (what the launch dialog does); the viewer
opens runs the real stage wrote over PR 4's fake detect method, so the coverage rows and the
alert come from a run's actual rows.
"""

import numpy as np
import pytest

from detail.configs.keypoint_track import (
    DespikeSmoothConfig,
    KeypointTrackConfig,
    TemporalDecodeConfig,
)
from detail.gui.modules.registry import GUI_MODULES
from detail.pipeline import keypoint_track
from detail.storage import recording as recording_store
from tests.gui.conftest import drain_until
from tests.pipeline.test_keypoint_detect import _config, _run
from tests.pipeline.test_keypoint_detect import tracked  # noqa: F401 — the fixture
from tests.pipeline.test_keypoint_track import _track_run


@pytest.fixture
def form(qapp):
    return GUI_MODULES["keypoint_track"].config_widget()


def test_the_form_asks_for_the_detect_run_first_and_marks_no_default_without_one(form):
    """In-app review: the default is the detect method's own method, selected and marked
    "(default)" in the list, not a separate row. With no detect run chosen there is no
    detect method to ask, so nothing is marked — no fallback id stands in (§26 *Track
    methods*: "a surface that keeps its own table or a fallback id is the bug")."""
    config = form.config()
    assert config["keypoint_detect_spec"] is None
    assert config["method"] in {"temporal_decode", "despike_smooth"}
    assert form.validation_error() == (
        "No keypoint detect run in the selected recordings — run keypoint detect first."
    )
    items = {
        form._method.itemData(i): form._method.itemText(i)
        for i in range(form._method.count())
    }
    assert set(items) == {"temporal_decode", "despike_smooth"}  # no third "default" row
    assert not any(text.endswith("(default)") for text in items.values())


def _choose_detect_method(form, monkeypatch, model_id):
    monkeypatch.setattr(
        form._picker,
        "spec",
        lambda: {"module": "keypoint_detect", "identity": {"model_id": model_id}},
    )
    form._sync_default()


@pytest.mark.parametrize(
    "model_id, expected",
    [("dinov3_match", "temporal_decode"), ("model_001", "despike_smooth")],
)
def test_each_detect_method_opens_on_its_own_track_method(
    qapp, tmp_path, monkeypatch, model_id, expected
):
    """§26: each detect method defaults to its own track method in every surface, through
    `keypoint_method_class` — `dinov3_match` declares `temporal_decode`, a trained student
    (not in `KEYPOINT_MODELS`, found through the experiment's models) `despike_smooth`. The
    form once had a `despike_smooth` fallback, which gave a student the right answer by
    accident and would have given any other unregistered method a guess."""
    import detail.models.patch_match  # noqa: F401 — registers dinov3_match
    from detail.api.experiment import Experiment
    from detail.storage import models as models_store

    proj = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(proj.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir, {"id": "model_001", "name": "s", "status": "complete", "task": "pose"}
    )
    form = GUI_MODULES["keypoint_track"].config_widget()
    form.on_experiment_set(proj)
    _choose_detect_method(form, monkeypatch, model_id)
    assert form.config()["method"] == expected
    assert form._method.itemText(form._method.findData(expected)).endswith("(default)")
    others = [
        form._method.itemText(i)
        for i in range(form._method.count())
        if form._method.itemData(i) != expected
    ]
    assert not any(text.endswith("(default)") for text in others)


def test_an_unknown_detect_method_marks_no_default(qapp, monkeypatch, form):
    """A method the lookup does not know gets no guessed default; the stage refuses it."""
    _choose_detect_method(form, monkeypatch, "model_404")
    assert form._default_method() is None


def test_the_tracking_run_opens_on_the_automatic_row_and_the_match_round_trips(form):
    """§26 *Assignment*: the tracking run's first row lets `validate` resolve the one on the
    detect run's detection run (and refuse several), so the config sends no
    `tracking_spec`; the two match settings are read off the model and round-trip."""
    assert form._tracking.itemData(0) is None
    assert form._tracking.currentIndex() == 0
    config = form.config()
    assert "tracking_spec" not in config
    assert config["match_min_inside"] == pytest.approx(
        KeypointTrackConfig.model_fields["match_min_inside"].default
    )
    assert config["match_mask_grow"] == pytest.approx(
        KeypointTrackConfig.model_fields["match_mask_grow"].default
    )
    form.load_config({"match_min_inside": 0.7, "match_mask_grow": 0.1})
    config = form.config()
    assert config["match_min_inside"] == pytest.approx(0.7)
    assert config["match_mask_grow"] == pytest.approx(0.1)


def test_the_default_follows_the_detect_run_until_the_user_picks(form, monkeypatch):
    """A detect run whose method settles with `despike_smooth` moves the selection and the
    "(default)" mark there; once the user picks a method, a change of detect run leaves it.
    """
    from detail.core.registries import KEYPOINT_MODELS
    from tests.pipeline.test_keypoint_detect import _FakeKeypointModel

    monkeypatch.setitem(
        KEYPOINT_MODELS, "fake_kp", _FakeKeypointModel
    )  # despike_smooth
    _choose_detect_method(form, monkeypatch, "fake_kp")
    assert form._method.currentData() == "despike_smooth"
    assert form._method.itemText(form._method.findData("despike_smooth")).endswith(
        "(default)"
    )
    form._method.setCurrentIndex(form._method.findData("temporal_decode"))
    form._method.activated.emit(form._method.currentIndex())  # the user's own pick
    form._sync_default()
    assert form._method.currentData() == "temporal_decode"


def test_the_advanced_section_a_method_fills_is_reachable(qapp, form):
    """Reported from the app: choosing a method showed no Advanced section at all. Only
    `ConfigForm.finish()` shows or hides it, which the rebuild never called. Asserts what the
    user sees, not that the widgets were built: the earlier test checked `_knobs` and passed
    over the bug."""
    from detail.gui.forms import AdvancedSection

    form.show()
    qapp.processEvents()
    [section] = form.findChildren(AdvancedSection)
    assert section.isVisible(), "the default method's settings are unreachable"
    form._method.setCurrentIndex(form._method.findData("despike_smooth"))
    qapp.processEvents()
    assert section.isVisible(), "the method's settings are unreachable"
    assert set(form._knobs) == set(DespikeSmoothConfig.model_fields)


def test_choosing_a_method_shows_its_every_setting_and_they_round_trip(
    form, monkeypatch
):
    import detail.models.patch_match  # noqa: F401 — registers dinov3_match

    form._method.setCurrentIndex(form._method.findData("temporal_decode"))
    assert set(form._knobs) == set(TemporalDecodeConfig.model_fields)
    assert form.config()["config"] == TemporalDecodeConfig().model_dump()
    form.load_config(
        {
            "method": "temporal_decode",
            "config": {
                "smooth_window_s": 0.2,
                "despike": False,
                "decode_cutoff": 0.3,
                "mirror_weight": 2.5,
                "smooth_kernel": "local_quadratic",
            },
            "keypoint_detect_spec": {"module": "keypoint_detect", "identity": {}},
        }
    )
    config = form.config()
    assert config["method"] == "temporal_decode"
    assert config["config"]["smooth_window_s"] == pytest.approx(0.2)
    assert config["config"]["despike"] is False
    assert config["config"]["decode_cutoff"] == pytest.approx(0.3)
    assert config["config"]["mirror_weight"] == pytest.approx(2.5)
    assert config["config"]["smooth_kernel"] == "local_quadratic"
    assert (
        config["keypoint_detect_spec"] is None
    )  # a selection: re-asked, never restored
    form._method.setCurrentIndex(form._method.findData("despike_smooth"))
    assert set(form._knobs) == set(DespikeSmoothConfig.model_fields)
    monkeypatch.setattr(
        form._picker,
        "spec",
        lambda: {"module": "keypoint_detect", "identity": {"model_id": "dinov3_match"}},
    )
    form.load_config(
        {"method": None}
    )  # a run that recorded none: the detect run's default
    assert form.config()["method"] == "temporal_decode"


def test_the_points_row_lists_the_chosen_detect_runs_skeleton_and_round_trips(
    form, monkeypatch
):
    """`track_points` is rendered from its `ui: skeleton_points` hint, not by name: the
    picker is the knob for it under either method, on the main form."""
    from detail.api import run_spec

    spec = {"module": "keypoint_detect", "identity": {"model_id": "dinov3_match"}}
    form._skeletons = {
        run_spec.freeze_spec(spec): (("snout", "l_fin", "r_fin"), (("l_fin", "r_fin"),))
    }
    monkeypatch.setattr(form._picker, "spec", lambda: spec)
    form._sync_points()
    assert form._knobs["track_points"] is form._points
    assert form.config()["config"]["track_points"] is None  # every point
    form._points._boxes["r_fin"].setChecked(False)
    assert form.config()["config"]["track_points"] == ["snout", "l_fin"]
    form.load_config(
        {"method": "despike_smooth", "config": {"track_points": ["snout"]}}
    )
    assert form._knobs["track_points"] is form._points
    assert form.config()["config"]["track_points"] == ["snout"]
    form._points.set_value([])
    assert "Tick at least one point" in form.validation_error()


@pytest.fixture
def track_runs(tracked):  # noqa: F811 — the imported fixture
    """``tracked``'s recording with ``kd_01`` (one pose per detection) and two track runs
    over it on ``track_01``: ``kt_01`` places every point, ``kt_02`` places none (a cut-off
    above every candidate score)."""
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
    _track_run(it, recording, {"detect_run_id": "kd_01", "method": None, "config": {}})
    _track_run(
        it,
        recording,
        {
            "detect_run_id": "kd_01",
            "method": "temporal_decode",
            "config": {"decode_cutoff": 0.9},
        },
        run_id="kt_02",
    )
    return recording


def test_the_form_reads_the_detect_runs_skeleton_off_the_ui_thread(form, track_runs):
    """`resolve_recordings` (the launch dialog's off-thread half) reads each detect-run
    group's skeleton from one of its runs; applying it fills the points row."""
    from tests.pipeline.test_keypoint_detect import SKELETON

    resolved = form.resolve_recordings([track_runs])
    assert list(resolved["skeletons"].values()) == [(SKELETON.names, SKELETON.pairs)]
    form.apply_recordings([track_runs], resolved)
    assert form._points.names() == list(SKELETON.names)


def _open(qapp, recording, run_id):
    from detail.gui.keypoint_run_preview import KeypointTrackRunPreview

    view = GUI_MODULES["keypoint_track"].preview_widget(
        recording, run_id, experiment_dir=None, parent_run_id=None
    )
    assert isinstance(view, KeypointTrackRunPreview)
    assert drain_until(qapp, lambda: hasattr(view, "_scrubber")), "the run never loaded"
    return view


def test_the_viewer_draws_the_final_points_and_reports_full_coverage(qapp, track_runs):
    view = _open(qapp, track_runs, "kt_01")
    summary = view._summary.text()
    assert "keypoints: despike_smooth over detect run kd_01 · 2 points" in summary
    assert "subjects from tracking run track_01" in summary
    # the assignment (§26): nine poses, seven given to a subject — frame 2's second pose
    # of track 0 and frame 4's untracked detection are not
    assert "part 001 — 7/9 poses given to a subject · 2 not" in summary
    assert "track 0 — 5/5 subject frames given a pose" in summary
    assert "track 1 — 2/2 subject frames given a pose" in summary
    assert not view._alert.isVisible() and view._alert.text() == ""
    assert view._kp_timeline is not None
    kind, with_points = view._compute(1)  # frame 1: both tracks have a subject
    assert kind == "img"
    _actual, rgb, detections = view._preview.frame(0, 1, with_masks=True)
    tracking_only = view._overlay(rgb, detections)
    assert with_points.shape == tracking_only.shape
    assert (with_points != tracking_only).any()  # the points were drawn on top


def test_a_point_under_the_coverage_threshold_raises_the_alert(qapp, track_runs):
    view = _open(qapp, track_runs, "kt_02")
    summary = view._summary.text()
    assert "keypoints: temporal_decode over detect run kd_01" in summary
    # every subject frame was given a pose; the decode cut-off then placed no point,
    # which is what the per-point coverage and the alert say
    assert "track 0 — 5/5 subject frames given a pose" in summary
    alert = view._alert.text()
    assert alert.startswith("4 point(s) under 80% coverage: nose on track 0 (0%)")
    assert "Mark more example frames" in alert and "lower the decode cut-off" in alert
    kind, bare = view._compute(1)  # nothing placed: the tracking picture alone
    assert kind == "img"
    _actual, rgb, detections = view._preview.frame(0, 1, with_masks=True)
    np.testing.assert_array_equal(bare, view._overlay(rgb, detections))


def test_a_finished_track_run_offers_embedding_seeded_with_this_run(
    qapp, track_runs, monkeypatch
):
    """Reported from the app: a keypoint track run's page had no next step, though every
    other stage offers one. Embedding reads a track run as its *second* input (`aux_refs`,
    §14), which `_next_stages` did not count. The button opens the embedding form on this
    run's recordings with its keypoint picker on this run's settings; Align is left alone
    (rotate or mirror depends on the camera, Tenet 9)."""
    from PyQt6.QtWidgets import QDialog, QPushButton

    from detail.api import run_spec
    from detail.gui import launch
    from detail.gui.pages import RunDetailPage, _next_stages
    from detail.storage import runs as runs_store

    recording = track_runs
    proj_dir = recording.dir.parent.parent
    from detail.api.experiment import Experiment

    proj = Experiment.open(proj_dir)
    _record_track_run(recording)
    runs_store.write_run(
        proj.dir,
        "run_021",
        {
            "id": "run_021",
            "module_id": "keypoint_track",
            "recording_ids": [recording.id],
            "per_recording": {recording.id: {"run_id": "kt_01", "status": "complete"}},
            "status": "complete",
        },
    )
    assert _next_stages("keypoint_track") == ["embedding"]
    page = RunDetailPage(proj, "run_021")
    [button] = [
        b for b in page.findChildren(QPushButton) if b.text() == "Run embedding →"
    ]

    opened = {}

    def _capture(dialog):
        opened["dialog"] = dialog
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(launch.LaunchRunDialog, "exec", _capture)
    button.click()
    dialog = opened["dialog"]
    assert dialog._module_combo.currentData() == "embedding"
    form = dialog._config
    want = run_spec.spec_from_run(
        proj.get_recording(recording.id), "keypoint_track", "kt_01"
    )
    assert drain_until(
        qapp, lambda: form._keypoint_picker.spec() is not None
    ), "the keypoint picker never filled"
    assert run_spec.freeze_spec(form._keypoint_picker.spec()) == run_spec.freeze_spec(
        want
    )
    assert form.config()["align"] == "none"  # left for the user to choose


def _record_track_run(recording) -> None:
    """The recording's own `kt_01` entry, as the executor writes it after a run (the
    fixture runs the stage directly, which writes the run's files but not this entry).
    """
    module = keypoint_track.KeypointTrackModule()
    data = recording_store.read_recording(recording.dir)
    config = module.validate(
        recording.dir, data, {"detect_run_id": "kd_01", "method": None, "config": {}}
    )
    data.setdefault("keypoint_track_runs", []).append(
        module.run_record("kt_01", config, "complete")
    )
    recording_store.write_recording(recording.dir, data)


def test_an_aligned_embedding_runs_page_loads_and_shows_its_heading_loss(
    qapp, track_runs, monkeypatch
):
    """Reported from the app: an aligned run's page read "(failed to load: name
    'PartZarrReader' is not defined)". Only `align != "none"` reaches the heading-loss read,
    and no test built that page on an aligned run, so the branch shipped unrun. Built here on
    a real aligned run over the real keypoint track run."""
    from PyQt6.QtWidgets import QLabel

    from detail.gui.run_preview import EmbeddingRunPreview
    from tests.pipeline.test_embedding import _registered_embedder

    recording = _aligned_run(track_runs, monkeypatch)
    with _registered_embedder():
        view = EmbeddingRunPreview(recording, "emb_01")
        assert drain_until(
            qapp,
            lambda: view._preview is not None or "failed" in view._status.text(),
        ), "the page never loaded"
    assert "failed to load" not in view._status.text(), view._status.text()
    texts = [label.text() for label in view.findChildren(QLabel)]
    assert any("no heading within the gap" in t for t in texts), texts
    assert any("track 0: heading on" in t for t in texts), texts

    loss = recording.embedding_preview("emb_01").heading_loss(
        recording.tracking_preview("track_01")
    )
    assert loss["track_ids"] == [0, 1]
    assert set(loss["coverage"]) == {"track_0", "track_1"}
    for track_id, kept in loss["kept"].items():
        assert kept.dtype == bool and kept.shape == (loss["total"],), track_id


def test_the_other_alignment_only_paths_run_on_a_real_aligned_run(
    qapp, track_runs, monkeypatch
):
    """The sweep after the page bug: every branch only `align != "none"` reaches, run on a
    real aligned run. The clicked frame's model input is turned (`EmbeddingPreview.frame_view`
    through an aligned `InputImagePreview`), and the launch form's input preview resolves an
    aligned spec (`InputImageScrubber._resolve_preview`)."""
    from detail.gui.run_preview import InputImageScrubber

    recording = _aligned_run(track_runs, monkeypatch)
    preview = recording.embedding_preview("emb_01")
    view = preview.frame_view(0, "part_001", 1)
    assert view is not None
    _actual, rgb, model_input = view
    assert (
        model_input is not None and model_input.ndim == 3
    )  # rendered, through the turn

    scrubber = InputImageScrubber(
        [recording],
        {recording.id: "track_01"},
        [0],
        "masked_color_centered",
        alignment={
            "align": "rotate",
            "heading_definition": {"front": ["nose"], "back": ["tail"]},
            "max_gap_s": 0.5,
            "mirror_hold_s": 1.0,
            "facing_camera_ratio": 0.5,
        },
        keypoint_run_ids={recording.id: "kt_01"},
    )
    aligned = scrubber._resolve_preview(recording, "track_01")
    assert aligned._alignment is not None
    assert "align_rotate" in [s["type"] for s in aligned._renderer.to_spec()]


def _aligned_run(recording, monkeypatch):
    """A real aligned embedding run (`emb_01`, rotate, nose to tail) over the fixture's
    keypoint track run, with every recording entry the executor would have written."""
    from detail.api.experiment import Experiment
    from detail.core.dataclasses import ComputeTarget
    from detail.pipeline import embedding
    from tests.pipeline.test_embedding import _registered_embedder

    _record_track_run(recording)
    proj = Experiment.open(recording.dir.parent.parent)
    proj.save_heading_definition("nose_to_tail", ["nose"], ["tail"])
    monkeypatch.setattr(
        "detail.storage.video_integrity.validate_source_videos", lambda *a, **k: []
    )
    data = recording_store.read_recording(recording.dir)
    if not any(r.get("id") == "det_01" for r in data.get("detection_runs", [])):
        # the fixture writes the detection files but not this entry; embedding reads it
        data.setdefault("detection_runs", []).append(
            {"id": "det_01", "status": "complete", "downsampling_run_id": None}
        )
        recording_store.write_recording(recording.dir, data)
    config = {
        "tracking_run_id": "track_01",
        "track_ids": [0, 1],
        "all_tracks": False,
        "model_id": "fake_embedder",
        "model_variant": None,
        "constructor": "masked_color_centered",
        "batch_size": 8,
        "every_n_frames": None,
        "compute": ComputeTarget("cpu", None, None, None),
        "align": "rotate",
        "keypoint_run_id": "kt_01",
        "heading": "nose_to_tail",
        "max_gap_s": 0.5,
        "mirror_hold_s": 1.0,
        "facing_camera_ratio": 0.5,
    }
    with _registered_embedder():
        module = embedding.EmbeddingModule()
        config = module.validate(recording.dir, data, config)
        job = module.build_job(recording.id, "emb_01", config)
        run_dir = recording.dir / "embeddings" / "emb_01"
        run_dir.mkdir(parents=True)
        embedding.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
        data = recording_store.read_recording(recording.dir)
        data.setdefault("embedding_runs", []).append(
            module.run_record("emb_01", config, "complete")
        )
        recording_store.write_recording(recording.dir, data)
    return recording


def test_a_next_step_from_a_tracking_run_opens_the_tracking_picker_on_it(
    qapp, track_runs
):
    """*Run keypoint_track →* on a tracking run's page hands that run's settings to the
    tracking picker (it is this stage's second input), in place of the automatic row; and
    the chosen spec reaches the config, for `validate` to check against the detect run's
    detection run."""
    from detail.api import run_spec

    recording = track_runs
    want = run_spec.spec_from_run(recording, "tracking", "track_01")
    form = GUI_MODULES["keypoint_track"].config_widget()
    form.prefer_upstream_spec("tracking", want)
    form.on_recordings_changed([recording])
    assert form._tracking.currentIndex() > 0
    assert run_spec.freeze_spec(form.config()["tracking_spec"]) == run_spec.freeze_spec(
        want
    )
    assert form.covered_recording_ids() == {recording.id}
    form._tracking.setCurrentIndex(0)  # back to automatic: nothing sent
    assert "tracking_spec" not in form.config()
