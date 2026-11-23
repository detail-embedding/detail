"""*Test cut-off…* (§12 *Distill keypoints*): the window that tests a keypoint teacher on frames
drawn the way the distill draws them, redraws at any cut-off without testing again, reports the
held-out check in pixels, and hands a chosen cut-off back to the distill form. The recording,
runs and video are real (the `taught` fixture); the teacher is the fake keypoint method the
detect tests use."""

import dataclasses

import pytest

from detail.gui import fit as fit_gui
from detail.gui import keypoint_distill as kd_gui
from detail.gui import keypoint_test as test_gui
from detail.gui.qt import QDialog, QLabel, QPushButton, QTest, Qt
from tests.pipeline.test_keypoint_dataset import (  # noqa: F401 — fixtures
    patch_teacher,
    taught,
)
from tests.pipeline.test_keypoint_detect import (
    tracked,
)  # noqa: F401 — `taught` needs it

DETECT = {"detect_runs": {"recording_002": "kd_01"}}


@pytest.fixture
def inline(monkeypatch):
    """Run the window's off-thread test inline, so the cards are drawn when the test looks."""
    monkeypatch.setattr(
        test_gui,
        "run_progress",
        lambda target, work, on_done, **k: on_done(work(lambda *a: None)),
    )
    monkeypatch.setattr(
        kd_gui, "run_busy", lambda target, fn, on_done, **k: on_done(fn())
    )


def _counting(experiment, monkeypatch):
    calls = []
    real = experiment.test_keypoints

    def counted(*args, **kwargs):
        calls.append(kwargs.get("seed"))
        return real(*args, **kwargs)

    monkeypatch.setattr(experiment, "test_keypoints", counted)
    return calls


def _cards(dialog) -> list[QLabel]:
    grid = dialog._grid
    return [grid.itemAt(i).widget() for i in range(grid.count())]


def test_the_window_draws_a_card_per_frame_and_redraws_without_testing_again(
    qapp, taught, inline, monkeypatch
):
    it, _recording = taught
    calls = _counting(it.experiment, monkeypatch)
    dialog = test_gui.KeypointTestDialog(it.experiment, dict(DETECT, n_frames=2))

    assert calls == [0]
    assert len(_cards(dialog)) == 2 and all(c.pixmap() for c in _cards(dialog))
    assert "fake_kp on 2 of 2 frames" in dialog._summary.text()
    assert "heatmap peak, 0 to 1" in dialog._summary.text()
    # The fake scores every point 0.5: at 0.4 all are taught visible, at 0.6 none are.
    placed = dialog._result.taught_visible(None)[1]
    assert placed > 0
    dialog._cutoff.set_value(0.4)
    dialog._redraw()
    assert (
        dialog._cutoff._caption.text() == f"taught visible: {placed} of {placed} points"
    )
    dialog._cutoff.set_value(0.6)
    dialog._redraw()
    assert dialog._cutoff._caption.text() == f"taught visible: 0 of {placed} points"
    assert calls == [0], "moving the cut-off redraws; it never runs the teacher again"
    assert "not available (this teacher has no example set)" in dialog._held.text()

    dialog._another.click()
    assert calls == [0, 1], "Another sample draws again with the next seed"
    dialog._use.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.chosen_cutoff() == pytest.approx(0.6)


def test_a_card_draws_points_only_no_box():
    """The user's call (2026-10-09): the cards show the teacher's points and nothing else. A
    frame whose points are all unplaced draws as the plain frame, card and full size alike.
    """
    import numpy as np

    from detail.core.skeleton import PRESETS
    from detail.pipeline.keypoint_preview import TestedFrame

    skeleton = PRESETS["midline_3"][1]
    image = np.full((48, 64, 3), 90, np.uint8)
    frame = TestedFrame(
        recording_id="r",
        part_id="p",
        frame_idx=0,
        t_ns=0,
        image=image,
        tracks={0: ((0.2, 0.2, 0.8, 0.8), np.full((3, 2), np.nan), np.full(3, np.nan))},
    )
    for full in (False, True):
        np.testing.assert_array_equal(
            test_gui._card_image(frame, skeleton, 0.5, full=full), image
        )


def test_a_card_opens_its_frame_full_size_at_the_grid_cut_off(
    qapp, taught, inline, monkeypatch
):
    """Reported from the app: a 240px card is too small to judge a point by. Each card is the
    shared enlargeable cell; its render is the full frame, drawn at the cut-off shown.
    """
    from detail.gui import full_frame

    it, _recording = taught
    dialog = test_gui.KeypointTestDialog(it.experiment, dict(DETECT, n_frames=2))
    dialog._cutoff.set_value(0.6)
    dialog._redraw()
    card = _cards(dialog)[0]
    assert isinstance(card, full_frame.EnlargeableThumb)
    assert card.toolTip() == "Double-click to see this frame full size"
    opened = {}
    monkeypatch.setattr(
        full_frame,
        "open_full_frame",
        lambda parent, render, *, title: opened.update(image=render(), title=title),
    )
    QTest.mouseDClick(card, Qt.MouseButton.LeftButton)
    frame = dialog.teacher_test().frames[0]
    assert opened["image"].shape == frame.image.shape, "full size, not the card"
    assert opened["title"].endswith("cut-off 0.6")


def test_the_held_out_check_is_in_pixels_with_its_counts(qapp, taught, inline):
    it, _recording = taught
    dialog = test_gui.KeypointTestDialog(it.experiment, dict(DETECT, n_frames=2))
    rows = [
        {"frame": f, "point": "nose", "score": 0.9, "error_px": 3.0, "body_px": 60.0}
        for f in range(8)
    ] + [{"frame": 9, "point": "nose", "score": 0.1, "error_px": 40.0, "body_px": 60.0}]
    dialog._show(dataclasses.replace(dialog._result, held_out=rows))
    dialog._cutoff.set_value(0.5)
    dialog._redraw()
    text = dialog._held.text()
    assert "At or above the cut-off: median 3.0 px (5.0% of body size)" in text
    assert "n = 8" in text
    assert "Below the cut-off: 1 placement(s), too few for a median" in text

    dialog._show(dataclasses.replace(dialog._result, held_out=[]))
    assert "needs at least two example frames" in dialog._held.text()


def test_a_caller_with_no_cutoff_field_gets_no_use_button(qapp, taught, inline):
    """The detect form's *Test keypoints…* (PR 4b) has nowhere to write a cut-off yet."""
    it, _recording = taught
    dialog = test_gui.KeypointTestDialog(
        it.experiment,
        dict(DETECT, n_frames=2),
        title="Test keypoints",
        choose_cutoff=False,
    )
    assert dialog.windowTitle() == "Test keypoints"
    assert dialog._use is None and dialog.chosen_cutoff() is None
    labels = [b.text() for b in dialog.findChildren(QPushButton)]
    assert "Use this cut-off" not in labels and "Another sample" in labels
    assert len(_cards(dialog)) == 2, "the slider and cards still work for looking"


def test_a_test_that_cannot_run_says_why_and_offers_no_cut_off(
    qapp, taught, inline, monkeypatch
):
    it, _recording = taught
    monkeypatch.setattr(
        test_gui,
        "run_progress",
        lambda target, work, on_done, *, on_error, **k: on_error(
            ValueError("no kd_99")
        ),
    )
    dialog = test_gui.KeypointTestDialog(
        it.experiment, {"detect_runs": {"recording_002": "kd_99"}}
    )
    assert "The test could not run: no kd_99" in dialog._summary.text()
    assert not dialog._use.isEnabled() and dialog._use.toolTip()
    assert dialog._another.isEnabled(), "trying again is allowed"


def test_test_cutoff_follows_the_teacher_and_writes_the_choice_back(
    qapp, taught, inline, monkeypatch
):
    it, _recording = taught
    proj = it.experiment
    dialog = fit_gui.DistillDialog(proj, model_type="keypoints")
    form = dialog.keypoint_form()
    form._kind.setCurrentIndex(form._kind.findData("keypoint_detect"))
    assert form._test.isEnabled(), "the one finished run is pre-ticked"
    assert form._score_cutoff.specialValueText() == "required: set a value"
    assert dialog.why_text() == "Set a score cut-off: fake_kp has no default"

    opened = {}

    def use(self):
        opened.update(teacher=self._teacher, cutoff=self.cutoff())
        self._cutoff.set_value(0.3)
        self._use_cutoff()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(test_gui.KeypointTestDialog, "exec", use)
    form._test.click()
    assert opened["teacher"]["detect_runs"] == {"recording_002": "kd_01"}
    assert opened["teacher"]["every_n_frames"] == 1, "drawn the way the distill would"
    assert opened["cutoff"] == pytest.approx(0.5), "no default: opens mid-scale"
    assert kd_gui.optional_value(form._score_cutoff) == pytest.approx(0.3)
    assert dialog.why_text() == "" and dialog._ok.isEnabled()

    form._runs.item(0).setCheckState(form._runs.item(0).checkState().Unchecked)
    assert not form._test.isEnabled()
    assert form._test.toolTip() == "Tick at least one keypoint detect run"
    assert dialog.why_text() == "Tick at least one keypoint detect run"
    assert not dialog._ok.isEnabled()


def test_a_method_with_no_default_says_the_cutoff_is_required(
    qapp, patch_teacher, inline
):
    from tests.gui.test_keypoint_distill import _settled

    it, _recording, _set_id, _seen = patch_teacher
    dialog = fit_gui.DistillDialog(it.experiment, model_type="keypoints")
    form = dialog.keypoint_form()
    dialog.show()
    _settled(qapp, form)
    form._method.setCurrentIndex(form._method.findData("fake_learner"))
    index = next(
        i
        for i, group in enumerate(form._tracking._groups)
        if "recording_002" in group["recordings"]
    )
    form._tracking.combo.setCurrentIndex(index)
    assert form._score_cutoff.specialValueText() == "required: set a value"
    assert form._test.isEnabled(), "Test cut-off… is how a cut-off gets chosen"
    assert dialog.why_text() == "Set a score cut-off: fake_learner has no default"
    form._score_cutoff.setValue(0.4)
    assert dialog.why_text() == "" and dialog._ok.isEnabled()


# -- Test keypoints on the keypoint detect form (PR 4b): Test prompts' layout ----------- #
def _detect_window(experiment, teacher, **kwargs):
    return test_gui.KeypointDetectTestDialog(
        experiment, teacher, {"recording_002": "rat cage"}, **kwargs
    )


def test_the_detect_window_waits_for_run_and_samples_what_is_chosen(
    qapp, taught, inline, monkeypatch
):
    """Laid out as detection's Test prompts: nothing runs until Run keypoint test, and the
    Recordings and Sample frames rows decide what it runs on."""
    it, _recording = taught
    seen = []
    real = it.experiment.test_keypoints

    def recorded(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(it.experiment, "test_keypoints", recorded)
    dialog = _detect_window(it.experiment, dict(DETECT, tracking={}))
    assert seen == [] and _cards(dialog) == []
    assert not dialog._another.isEnabled() and not dialog._per_point.isEnabled()
    assert dialog._another.toolTip() == "Run keypoint test first"
    assert dialog.windowTitle() == "Keypoint test"
    assert dialog._summary.text() == "Choose the frames, then Run keypoint test."
    assert [dialog._recordings.itemText(i) for i in range(2)] == [
        "All selected (1)",
        "rat cage",
    ]
    assert dialog._n.value() == 12

    dialog._n.setValue(2)
    dialog._run_btn.click()
    assert seen[-1]["n_frames"] == 2 and len(_cards(dialog)) == 2
    assert dialog._another.isEnabled(), "a result to draw again from"
    assert "fake_kp on 2 of 2 frames, spread across the recordings" in (
        dialog._summary.text()
    )


def test_the_detect_window_has_no_use_button_and_reads_as_viewer_wording(
    qapp, taught, inline
):
    """A detect run stores raw points: the slider is for looking, its caption counts points
    shown, and the legend speaks of viewers and exports, not of teaching."""
    it, _recording = taught
    dialog = _detect_window(it.experiment, dict(DETECT, tracking={}))
    dialog._n.setValue(2)
    labels = [b.text() for b in dialog.findChildren(QPushButton)]
    assert "Use this cut-off" not in labels and dialog.chosen_cutoff() is None
    assert "Run keypoint test" in labels and "Another sample" in labels
    assert "not visible in a viewer or export" in dialog._legend.text()
    assert "taught" not in dialog._legend.text()
    body = dialog.body()
    order = [body.itemAt(i).widget() for i in range(body.count())]
    assert order.index(dialog._cutoff) < order.index(
        dialog._legend
    ), "the slider sits above the frames, as on Test prompts"
    dialog._run_btn.click()
    placed = dialog._result.taught_visible(None)[1]
    dialog._cutoff.set_value(0.4)
    dialog._redraw()
    assert dialog._cutoff._caption.text() == f"{placed} of {placed} points shown"


def test_one_recording_narrows_the_tracking_the_test_is_given(
    qapp, taught, monkeypatch
):
    it, _recording = taught
    seen = []
    monkeypatch.setattr(it.experiment, "test_keypoints", lambda **kw: seen.append(kw))
    monkeypatch.setattr(
        test_gui,
        "run_progress",
        lambda target, work, on_done, *, on_error, **k: (
            work(lambda *a: None),
            on_error(ValueError("stand-in: no result")),
        ),
    )
    teacher = {
        "example_set": "ex_01",
        "tracking": {"recording_002": "track_01", "recording_003": "track_02"},
        "model_id": "fake_learner",
    }
    dialog = test_gui.KeypointDetectTestDialog(
        it.experiment,
        teacher,
        {"recording_002": "rat cage", "recording_003": "mouse cage"},
    )
    dialog._run_btn.click()
    assert seen[-1]["tracking"] == teacher["tracking"], "All selected: every recording"
    dialog._recordings.setCurrentIndex(dialog._recordings.findData("recording_003"))
    dialog._run_btn.click()
    assert seen[-1]["tracking"] == {"recording_003": "track_02"}
    assert seen[-1]["example_set"] == "ex_01" and seen[-1]["n_frames"] == 12


def test_the_example_set_check_is_one_line_until_per_point(qapp, taught, inline):
    it, _recording = taught
    dialog = _detect_window(it.experiment, dict(DETECT, tracking={}))
    dialog._n.setValue(2)
    dialog._run_btn.click()
    rows = [
        {"frame": f, "point": "nose", "score": 0.9, "error_px": 3.0, "body_px": 60.0}
        for f in range(8)
    ] + [{"frame": 9, "point": "nose", "score": 0.1, "error_px": 40.0, "body_px": 60.0}]
    dialog._show(dataclasses.replace(dialog._result, held_out=rows))
    dialog._cutoff.set_value(0.5)
    dialog._redraw()
    text = dialog._held.text()
    assert text.startswith("Example set check: median 3.0 px (5.0% of body size)")
    assert "\n" not in text and "below it, n = 1 (too few for a median)" in text
    assert dialog._per_point.isEnabled()
    dialog._per_point.setChecked(True)
    assert "nose, at or above" in dialog._held.text()
