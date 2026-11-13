"""*Tune tracking…* (§26 *Tuning a track method*): the track form's button, and the window that
scores settings and hands one back.

The window's work is ``Experiment.tune_keypoint_tracking`` (scored in
``tests/pipeline/test_keypoint_track_tune.py``); here a stand-in experiment returns a result at
once, so what is tested is the window: when it can run, what it shows, and what it gives the
form.
"""

import pytest

from detail.gui.keypoint_track_tune import KeypointTrackTuneDialog
from detail.gui.modules.registry import GUI_MODULES
from detail.pipeline import keypoint_track_tune as tune
from tests.gui.conftest import drain_until

SPEC = {"identity": {"model_id": "dinov3_match"}}


def _setting(name, overrides, pick, check):
    setting = tune.TuneSetting(name, overrides)
    setting.scores = {
        half: {
            "group:all": {"pck10": {"value": value}},
            "group:paired": {"pck10": {"value": value - 0.1}},
            "group:unpaired": {"pck10": {"value": value + 0.05}},
        }
        for half, value in (("pick", pick), ("check", check))
    }
    return setting


def _result(base=None):
    settings = [
        _setting(tune.DETECTION_ONLY, None, 0.74, 0.75),
        _setting(tune.CURRENT, {}, 0.59, 0.66),
        _setting("transition 1, smooth off, despike off", {"transition_sigma_body": 1.0, "smooth": False}, 0.70, 0.74),  # fmt: skip
        _setting("transition 3, smooth off, despike off", {"transition_sigma_body": 3.0, "smooth": False}, 0.79, 0.80),  # fmt: skip
    ]
    return tune.TuneResult(
        detect_runs={"recording_001": "kd_01", "recording_002": "kd_01"},
        example_set_id="ex_02",
        method="temporal_decode",
        base_config=base or {"transition_sigma_body": 0.15, "smooth": True},
        pick_recordings=["recording_001"],
        check_recordings=["recording_002"],
        coverage={"labelled": 20, "excluded": 0, "uncovered": 0, "scored": 20},
        settings=settings,
        best="transition 3, smooth off, despike off",
    )


def _choice(set_id, labelled, reason=None):
    return {
        "id": set_id,
        "name": f"set {set_id}",
        "labelled": labelled,
        "usable": reason is None,
        "reason": reason,
    }


class _Experiment:
    dir = (
        None  # the form looks a method up through it; the built-in ones need no folder
    )

    def __init__(self, choices, result=None):
        self._choices, self._result, self.calls = choices, result, []
        self.asked_for = None

    def tune_example_sets(self, spec):
        self.asked_for = spec
        return self._choices

    def tune_keypoint_tracking(self, set_id, **options):
        self.calls.append((set_id, options))
        options["progress"]("scoring", 1.0, "done")
        return self._result


@pytest.fixture
def form(qapp):
    return GUI_MODULES["keypoint_track"].config_widget()


def test_the_button_says_why_until_it_can_tune(form, monkeypatch):
    assert not form._tune.isEnabled()
    assert form._tune.toolTip() == "Open an experiment first"
    form.on_experiment_set(_Experiment([]))
    assert form._tune.toolTip() == "Choose a keypoint detect run first"
    monkeypatch.setattr(form._picker, "spec", lambda: SPEC)
    form._sync_tune()  # what choosing a detect run in the picker does
    form._method.setCurrentIndex(form._method.findData("despike_smooth"))
    assert not form._tune.isEnabled() and "temporal_decode" in form._tune.toolTip()
    form._method.setCurrentIndex(form._method.findData("temporal_decode"))
    assert form._tune.isEnabled()


def test_the_settings_the_window_hands_back_fill_the_form(form, monkeypatch):
    import detail.gui.keypoint_track_tune as window

    opened = {}

    class _Dialog:
        def __init__(self, experiment, spec, base, parent, **kwargs):
            opened.update(spec=spec, base=base, **kwargs)
            self.chosen_config = {**base, "transition_sigma_body": 3.0, "smooth": False}

        def exec(self):
            return 1

    monkeypatch.setattr(window, "KeypointTrackTuneDialog", _Dialog)
    monkeypatch.setattr(form._picker, "spec", lambda: SPEC)
    form.on_experiment_set(_Experiment([]))
    form._open_tune()
    assert opened["spec"] == SPEC
    assert opened["base"]["transition_sigma_body"] == 0.15  # the form's own settings
    # the poses reach the labelled subjects the way the run would give them (§26)
    assert opened["tracking_spec"] is None  # the automatic row: the stage's own default
    assert opened["match"] == {"min_inside": 0.5, "mask_grow": 0.05}
    config = form.config()
    assert config["method"] == "temporal_decode"
    assert config["config"]["transition_sigma_body"] == 3.0
    assert config["config"]["smooth"] is False


def test_the_window_offers_only_sets_that_can_score_and_names_the_rest(qapp):
    """Reported from the app: the window opened on the detect run's own example set, which
    scoring then refused. The choice is read off the UI thread for this detect run; the sets
    it cannot score with are left out of the list and named, with why."""
    experiment = _Experiment(
        [
            _choice("ex_01", 50, "the detect run learned from it"),
            _choice("ex_02", 45),
            _choice("ex_03", 12, "labelled on another skeleton than the detect run's"),
        ]
    )
    dialog = KeypointTrackTuneDialog(experiment, SPEC, {})
    assert not dialog._run.isEnabled() and "Reading" in dialog._run.toolTip()
    assert drain_until(qapp, lambda: dialog._sets_loaded)
    assert experiment.asked_for == SPEC
    assert [dialog._sets.itemData(i) for i in range(dialog._sets.count())] == ["ex_02"]
    assert dialog._sets.currentData() == "ex_02" and dialog._run.isEnabled()
    note = dialog._left_out.text()
    assert (
        "ex_01), the detect run learned from it" in note and "another skeleton" in note
    )


def test_with_no_set_that_can_score_the_window_says_what_to_do(qapp):
    experiment = _Experiment([_choice("ex_01", 50, "the detect run learned from it")])
    dialog = KeypointTrackTuneDialog(experiment, SPEC, {})
    assert drain_until(qapp, lambda: dialog._sets_loaded)
    assert dialog._sets.count() == 0
    assert not dialog._run.isEnabled() and "Label a second" in dialog._run.toolTip()
    assert not dialog._use.isEnabled()


def test_scoring_fills_the_table_selects_the_recommendation_and_hands_it_back(qapp):
    experiment = _Experiment([_choice("ex_02", 20)], _result())
    dialog = KeypointTrackTuneDialog(experiment, SPEC, {"transition_sigma_body": 0.15})
    assert drain_until(qapp, lambda: dialog._sets_loaded)
    assert dialog._sets.currentData() == "ex_02"
    dialog.show()
    dialog.score()
    assert drain_until(qapp, lambda: dialog._model.rowCount() == 4)
    set_id, options = experiment.calls[0]
    assert set_id == "ex_02" and options["keypoint_detect_spec"] == SPEC
    assert options["base_config"] == {"transition_sigma_body": 0.15}
    assert dialog._selected().name == "transition 3, smooth off, despike off"
    assert dialog._model.item(3, 2).text() == "80.0%"  # the check half, all points
    summary = dialog._summary.text()
    assert "Recommended: transition 3" in summary and "75.0%" in summary
    assert dialog._use.isEnabled()
    dialog._table.selectRow(0)  # detection alone is not a tracking setting
    assert not dialog._use.isEnabled() and "Detection alone" in dialog._use.toolTip()
    dialog._table.selectRow(3)
    dialog._take()
    assert dialog.chosen_config == {
        "transition_sigma_body": 3.0,
        "smooth": False,
    }  # the result's base with the row's overrides
    dialog.close()
