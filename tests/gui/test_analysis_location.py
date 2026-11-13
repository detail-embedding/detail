"""The Location export launch form's keypoint and pair inputs (§16 *Keypoints and pairs in the
export*): an optional keypoint track picker tied to the ticked tracking run, and the edge tick.
"""

from PyQt6.QtCore import Qt

from detail.api.experiment import Experiment
from detail.storage import recording as recording_store


def _form_over(qapp, recordings):
    from detail.gui.modules.analysis_location import _LocationExportConfig

    form = _LocationExportConfig()
    form.apply_eligible_recordings(
        recordings, form.resolve_eligible_recordings(recordings)
    )
    item = form._tracking.widget.item(0)
    item.setCheckState(Qt.CheckState.Checked)
    return form


def _recording_with_keypoints(proj, sid, tracking_run_id="track_01"):
    from tests.experiments import write_keypoint_track_run
    from tests.pipeline.test_location_analysis import _add_tracked_recording

    recording = _add_tracked_recording(proj, sid)
    write_keypoint_track_run(recording.dir, n_frames=1, tracking_run_id=tracking_run_id)
    return proj.resolve_recording(sid)


def test_no_keypoints_is_the_default_and_a_choice_reaches_the_config(qapp, tmp_path):
    from detail.gui.modules.analysis_location import NO_KEYPOINTS

    proj = Experiment.create("p", tmp_path)
    form = _form_over(qapp, [_recording_with_keypoints(proj, "recording_001")])
    combo = form._keypoints.combo
    assert combo.currentText() == NO_KEYPOINTS
    assert "keypoint_track_spec" not in form.config()
    assert form.config()["pair_edge_distance"] is False
    combo.setCurrentIndex(1)
    form._fields["pair_edge_distance"].setChecked(True)
    config = form.config()
    assert config["keypoint_track_spec"]["module"] == "keypoint_track"
    assert config["pair_edge_distance"] is True


def test_a_keypoint_run_of_another_tracking_run_is_greyed(qapp, tmp_path):
    """Two tracking settings; the keypoint track run went to track_02, the one not ticked."""
    from detail.gui.modules.analysis_location import _LocationExportConfig

    proj = Experiment.create("p", tmp_path)
    recording = _recording_with_keypoints(
        proj, "recording_001", tracking_run_id="track_02"
    )
    data = recording_store.read_recording(recording.dir)
    original = data["tracking_runs"][0]
    data["tracking_runs"] = [
        {**original, "id": "track_01", "hyperparams": {}},
        {**original, "id": "track_02", "hyperparams": {"max_missed_frames": 9}},
    ]
    recording_store.write_recording(recording.dir, data)
    recordings = [proj.resolve_recording("recording_001")]
    form = _LocationExportConfig()
    form.apply_eligible_recordings(
        recordings, form.resolve_eligible_recordings(recordings)
    )
    listing = form._tracking.widget
    for i in range(listing.count()):
        spec = listing.item(i).data(Qt.ItemDataRole.UserRole)
        if spec["identity"]["hyperparams"] == {}:  # track_01's setting
            listing.item(i).setCheckState(Qt.CheckState.Checked)
    item = form._keypoints.combo.model().item(1)
    assert item is not None and not item.isEnabled()
    assert "another tracking run" in form._keypoints.combo.itemText(1)


def test_the_keypoint_choice_per_recording_follows_its_tracking_run(qapp, tmp_path):
    """§6.5 per-recording overrides: the keypoint row offers a recording only the runs whose
    poses went to the tracking run read there — and naming that tracking run per recording
    is what brings a keypoint run made on it into reach."""
    from detail.gui.modules.analysis_location import _LocationExportConfig

    proj = Experiment.create("p", tmp_path)
    recording = _recording_with_keypoints(
        proj, "recording_001", tracking_run_id="track_02"
    )
    data = recording_store.read_recording(recording.dir)
    original = data["tracking_runs"][0]
    data["tracking_runs"] = [
        {**original, "id": "track_01", "hyperparams": {}},
        {**original, "id": "track_02", "hyperparams": {"max_missed_frames": 9}},
    ]
    recording_store.write_recording(recording.dir, data)
    recordings = [proj.resolve_recording("recording_001")]
    form = _LocationExportConfig()
    form.apply_eligible_recordings(
        recordings, form.resolve_eligible_recordings(recordings)
    )
    listing = form._tracking.widget
    for i in range(listing.count()):
        spec = listing.item(i).data(Qt.ItemDataRole.UserRole)
        on = spec["identity"]["hyperparams"] == {}  # track_01's setting only
        listing.item(i).setCheckState(
            Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        )
    assert not form._keypoint_overrides.button.isEnabled()

    form._tracking.overrides.set_chosen({"recording_001": "track_02"})

    assert form._keypoint_overrides.button.isEnabled()
    assert form._keypoint_overrides._options["recording_001"][0][0] == "kt_01"
    form._keypoint_overrides.set_chosen({"recording_001": "kt_01"})
    config = form.config()
    assert config["tracking_overrides"] == {"recording_001": "track_02"}
    assert config["keypoint_track_overrides"] == {"recording_001": "kt_01"}
