"""Per-recording Check framework (§1): the video-integrity check + the recording runner."""

from detail.api import checks as api_checks
from detail.checks.recording import VideoIntegrityCheck
from detail.core.dataclasses import CheckStatus, RecordingTarget


def _target(tmp_path, recording_id, files):
    return RecordingTarget(
        "exp", recording_id, str(tmp_path), {"video": {"files": files}}
    )


def test_video_integrity_check_ok_when_present(tmp_path):
    (tmp_path / "video").mkdir()
    (tmp_path / "video" / "part_001.mp4").write_bytes(b"x")
    result = VideoIntegrityCheck().run(
        _target(
            tmp_path,
            "recording_001",
            [{"id": "part_001", "file": "video/part_001.mp4"}],
        )
    )
    assert result.status == CheckStatus.OK
    assert result.target_id == "recording_001"


def test_video_integrity_check_fails_on_missing(tmp_path):
    result = VideoIntegrityCheck().run(
        _target(
            tmp_path, "recording_002", [{"id": "part_001", "file": "video/gone.mp4"}]
        )
    )
    assert result.status == CheckStatus.FAIL
    assert result.target_id == "recording_002"
    assert "missing_file" in result.detail


def test_video_integrity_check_of_a_recording_with_no_video_yet(tmp_path):
    """A recording made before its video was added has no ``video`` block (#33)."""
    target = RecordingTarget("exp", "recording_003", str(tmp_path), {"name": "m1"})
    result = VideoIntegrityCheck().run(target)
    assert result.status == CheckStatus.SKIPPED
    assert result.detail == "no video yet"
    assert result.target_id == "recording_003"


def test_video_integrity_check_no_target_is_skipped():
    # A machine-context call (run() with no target) is a no-op, not a crash.
    assert VideoIntegrityCheck().run().status == CheckStatus.SKIPPED


def test_machine_run_checks_excludes_recording_checks():
    # The per-recording check must not pollute the machine-global run/list.
    ids = {i.check_id for i in api_checks.list_checks()}
    assert "video_integrity" not in ids
    assert "torch" in ids


def test_run_recording_checks_enumerates_and_tags(tmp_path, monkeypatch):
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    monkeypatch.setattr(
        experiment_store,
        "discover_recordings",
        lambda d: ["recording_001", "recording_002"],
    )
    monkeypatch.setattr(
        experiment_store, "recording_path", lambda d, sid: tmp_path / sid
    )
    monkeypatch.setattr(
        recording_store,
        "read_recording",
        lambda d: {"video": {"files": [{"id": "part_001", "file": "video/gone.mp4"}]}},
    )
    results = api_checks.run_recording_checks(tmp_path)
    assert {r.target_id for r in results} == {"recording_001", "recording_002"}
    assert all(r.status == CheckStatus.FAIL for r in results)  # both videos missing
