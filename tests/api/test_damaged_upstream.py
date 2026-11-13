"""A launch over a run the OS lost is refused up front, naming the recording and the file.

The class and the incident are written up in ``tests/core/test_damaged_files.py``; this is the
launch half. ``validate`` reads ``recording.yaml``, where a detection run whose chunks and
``zarr.json`` were zero-filled by a shutdown still says ``complete``. So a 5-recording tracking
launch was accepted, four recordings ran, and the fifth failed in its worker a second later
with a decoder error that named no file (run_004, Oct 2026). Tenet 10 says a launch validates
every selected recording **before** anything starts, and this is a validation it was missing.

The check is `ProcessingModule.check_upstream_intact`, metadata only and over the whole lineage
(embedding -> tracking -> detection), asked next to ``validate`` on the three paths that launch a
recording's run: ``Experiment.run_processing``, the retry's re-validation, and ``Recording.run``.
These drive every one of them, because a check wired into two of three is the shape this repo's
``wire-every-surface`` skill exists for.

Where the fix stops: only what a worker is about to *read* is checked, and only its metadata.
A zero-filled chunk in an otherwise intact run still reaches the worker, where the codec names it
(``tests/storage/test_zarr_damaged.py``). Analysis modules (clustering, query) read their own
upstream through their own validators and are not covered here.
"""

import pytest

from detail.api.experiment import Experiment
from detail.core import atomic
from detail.executor.runners import PROCESSING_MODULES
from detail.pipeline import tracking  # noqa: F401  (registers the module)
from detail.storage import recording as recording_store
from tests.api.test_experiment import TRACKING_CONFIG, _detection_run, _video_recording


@pytest.fixture
def experiment(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    for sid in ("recording_001", "recording_002"):
        _video_recording(proj, sid)
        _detection_run(proj.get_recording(sid).dir)
    return proj


def _lose(experiment, sid):
    """The report: the group's ``zarr.json`` zero-filled at its real length."""
    path = (
        experiment.get_recording(sid).dir
        / "detections"
        / "det_01"
        / "part_001.zarr"
        / "zarr.json"
    )
    path.write_bytes(b"\x00" * path.stat().st_size)
    return path


def test_a_healthy_upstream_passes(experiment):
    sdir = experiment.get_recording("recording_001").dir
    data = recording_store.read_recording(sdir)
    PROCESSING_MODULES["tracking"].check_upstream_intact(
        sdir, data, dict(TRACKING_CONFIG)
    )


def test_run_processing_refuses_the_launch_and_names_the_recording(experiment):
    lost = _lose(experiment, "recording_002")

    with pytest.raises(atomic.DamagedFileError) as caught:
        experiment.run_processing(
            "tracking", ["recording_001", "recording_002"], dict(TRACKING_CONFIG)
        )

    assert str(lost) in str(caught.value)
    assert "recording_002" in str(caught.value)
    # Tenet 10: nothing was recorded or started for the recordings that were fine
    assert experiment.processing_runs == []


def test_run_processing_over_only_the_healthy_recording_is_not_refused(experiment):
    """The check is per recording: one lost run does not poison a launch that leaves it out."""
    _lose(experiment, "recording_002")
    sdir = experiment.get_recording("recording_001").dir
    PROCESSING_MODULES["tracking"].check_upstream_intact(
        sdir, recording_store.read_recording(sdir), dict(TRACKING_CONFIG)
    )


def test_a_single_recording_run_is_refused_too(experiment):
    lost = _lose(experiment, "recording_001")

    with pytest.raises(atomic.DamagedFileError, match="zarr.json") as caught:
        experiment.get_recording("recording_001").run("tracking", dict(TRACKING_CONFIG))

    assert str(lost) in str(caught.value)
    assert (
        recording_store.read_recording(
            experiment.get_recording("recording_001").dir
        ).get("tracking_runs", [])
        == []
    )  # no run record was reserved for it


def test_a_retry_names_every_recording_whose_upstream_was_lost(experiment):
    for sid in ("recording_001", "recording_002"):
        _lose(experiment, sid)

    errors, _validated = experiment._validate_retry_recordings(
        PROCESSING_MODULES["tracking"],
        ["recording_001", "recording_002"],
        dict(TRACKING_CONFIG),
    )

    assert set(errors) == {"recording_001", "recording_002"}
    assert all("DamagedFileError" in reason for reason in errors.values())


def test_the_whole_lineage_is_checked_not_just_the_parent(experiment):
    """An embedding reads its tracking run, and tracking read the detection run beneath it.
    Losing the detection run is invisible from the tracking run's own folder."""
    sdir = experiment.get_recording("recording_001").dir
    data = recording_store.read_recording(sdir)
    data.setdefault("tracking_runs", []).append(
        {"id": "track_01", "output": "tracking/track_01/", "detection_run_id": "det_01"}
    )
    embedding = PROCESSING_MODULES["embedding"]
    config = {"tracking_run_id": "track_01"}

    embedding.check_upstream_intact(sdir, data, config)  # healthy: no raise

    lost = _lose(experiment, "recording_001")
    with pytest.raises(atomic.DamagedFileError) as caught:
        embedding.check_upstream_intact(sdir, data, config)
    assert str(lost) in str(caught.value)


def test_a_module_rooted_at_the_video_has_no_upstream_run_to_check(experiment):
    detection = PROCESSING_MODULES["detection"]
    sdir = experiment.get_recording("recording_001").dir
    _lose(experiment, "recording_001")  # nothing detection reads
    detection.check_upstream_intact(sdir, recording_store.read_recording(sdir), {})
