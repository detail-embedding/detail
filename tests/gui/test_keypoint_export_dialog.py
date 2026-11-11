"""The Export dialog on a keypoint run (§26 "Surfaces"): the five rows, and the knobs every
keypoint format shares — source, cut-off, blanking, the video's overlay."""

import pytest

pytest.importorskip("PyQt6")

from detail.api import exporters  # noqa: E402
from detail.api.exporters import ExportSubject  # noqa: E402
from tests.api.test_keypoint_export import (  # noqa: E402,F401  (the fixture)
    DETECT_RUN,
    FIVE,
    TRACK_RUN,
    keypoint_runs,
)


def _dialog(it, run_id):
    from detail.gui.export import ExportDialog

    subject = ExportSubject("run", it.experiment.dir, run_id)
    return ExportDialog(subject, exporters.targets_for(subject), None)


def test_a_track_run_lists_the_five_rows_and_their_shared_knobs(qapp, keypoint_runs):
    it, _recording = keypoint_runs
    dialog = _dialog(it, TRACK_RUN)
    for target_id in FIVE:
        check = dialog._include[target_id]
        assert check.isEnabled() and check.isChecked(), target_id
    assert dialog._kp_source is not None and dialog._kp_cutoff is not None
    assert dialog._kp_blank is not None and dialog._kp_overlay is not None
    assert [dialog._kp_source.itemData(i) for i in range(2)] == ["track", "detect"]
    # The wording chosen in review: the source by its stage, the two boxes by their effect.
    assert [dialog._kp_source.itemText(i) for i in range(2)] == [
        "Tracking",
        "Detection, before tracking and smoothing",
    ]
    assert dialog._kp_blank.text() == "Omit points below score threshold"
    assert dialog._kp_overlay.text() == "Video: show masks and boxes"
    options = dialog._options()
    assert "cutoff" not in options  # the method's default, until moved
    assert options["blank_below_cutoff"] is False and options["overlay"] is False
    assert options["source"] == "track"
    dialog._kp_cutoff.setValue(0.6)
    dialog._kp_blank.setChecked(True)
    dialog._kp_source.setCurrentIndex(1)
    dialog._kp_overlay.setChecked(True)
    options = dialog._options()
    assert options["cutoff"] == pytest.approx(0.6)
    assert options["blank_below_cutoff"] is True and options["overlay"] is True
    assert options["source"] == "detect"
    for widget in (
        dialog._kp_source,
        dialog._kp_cutoff,
        dialog._kp_blank,
        dialog._kp_overlay,
    ):
        assert widget.toolTip(), "every knob says what it does"


def test_a_detect_run_has_no_source_to_choose(qapp, keypoint_runs):
    it, _recording = keypoint_runs
    dialog = _dialog(it, DETECT_RUN)
    assert dialog._kp_source is None and dialog._kp_cutoff is not None
    assert "source" not in dialog._options()


def test_a_tracking_run_greys_the_keypoint_rows_with_the_reason(qapp, keypoint_runs):
    it, _recording = keypoint_runs
    dialog = _dialog(it, it.tracking_run_id)
    check = dialog._include["keypoints"]
    assert not check.isEnabled()
    assert (
        "this is a tracking run; these come from a keypoint track run" in check.text()
    )
    assert dialog._kp_cutoff is None  # no knob for a row that cannot run
    assert dialog._include["tracks"].isEnabled()
