"""Referenced-video integrity checks (Section 2) — the shared pre-run-gate truth."""

import pytest

from detail.storage import integrity, locations, video_integrity, video_roots


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(locations, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")


def _kinds(issues):
    return [i.kind for i in issues]


def test_copied_part_present_is_clean(tmp_path):
    (tmp_path / "video").mkdir()
    (tmp_path / "video" / "part_001.mp4").write_bytes(b"data")
    data = {"video": {"files": [{"id": "part_001", "file": "video/part_001.mp4"}]}}
    assert video_integrity.validate_source_videos(tmp_path, data) == []


def test_copied_part_missing_reports_missing_file(tmp_path):
    data = {"video": {"files": [{"id": "part_001", "file": "video/gone.mp4"}]}}
    issues = video_integrity.validate_source_videos(tmp_path, data)
    assert _kinds(issues) == ["missing_file"]


def _reference_recording(tmp_path, content=b"a-referenced-video" * 100):
    root = tmp_path / "nas"
    root.mkdir(exist_ok=True)
    src = root / "m.mp4"
    src.write_bytes(content)
    video_roots.register_video_root("nas", root)
    part = {
        "id": "part_001",
        "source": {"root": "nas", "rel": "m.mp4"},
        "fingerprint": integrity.file_fingerprint(src),
    }
    return {"video": {"storage": "reference", "files": [part]}}, src


def test_reference_present_and_unchanged_is_clean(tmp_path, isolated_storage):
    data, _ = _reference_recording(tmp_path)
    assert video_integrity.validate_source_videos(tmp_path, data) == []


def test_reference_moved_reports_missing_file(tmp_path, isolated_storage):
    data, src = _reference_recording(tmp_path)
    src.unlink()  # user moved/renamed the external file
    assert _kinds(video_integrity.validate_source_videos(tmp_path, data)) == [
        "missing_file"
    ]


def test_reference_content_change_reports_changed_fingerprint(
    tmp_path, isolated_storage
):
    data, src = _reference_recording(tmp_path)
    src.write_bytes(b"a-DIFFERENT-video" * 100)  # same path, new content
    issues = video_integrity.validate_source_videos(tmp_path, data)
    assert _kinds(issues) == ["changed_fingerprint"]
    assert "remove and re-add" in issues[0].detail


def test_reference_unregistered_root_reports_missing_root(tmp_path, isolated_storage):
    part = {
        "id": "part_001",
        "source": {"root": "ghost", "rel": "m.mp4"},
        "fingerprint": {"size": 1, "sha256_ends": "x"},
    }
    data = {"video": {"storage": "reference", "files": [part]}}
    assert _kinds(video_integrity.validate_source_videos(tmp_path, data)) == [
        "missing_root"
    ]


def test_raise_on_video_issues(tmp_path):
    data = {"video": {"files": [{"id": "part_001", "file": "video/gone.mp4"}]}}
    issues = video_integrity.validate_source_videos(tmp_path, data)
    with pytest.raises(ValueError, match="not usable"):
        video_integrity.raise_on_video_issues(issues)
    video_integrity.raise_on_video_issues([])  # empty is a no-op


def _probe(fps, resolution=(64, 48)):
    return {"resolution": list(resolution), "fps": fps}


@pytest.mark.parametrize("fps", [29.99, 30.0, 29.97 * 1.009])
def test_a_variable_rate_camera_s_next_file_is_another_part(fps):
    """A container's rate is its frames over its duration, so two files from one
    variable-rate camera (a phone, a webcam, a screen recorder) read as 29.97 and 29.99. At
    three decimals that refused them; frame times come from each part's own timestamps, so
    a rate within ``PART_FPS_TOLERANCE`` misplaces nothing."""
    assert video_integrity.another_parts_shape(_probe(29.97), _probe(fps)) is None


@pytest.mark.parametrize("fps", [25.0, 29.97 * 1.011, 59.94])
def test_another_rate_is_not_another_part(fps):
    reason = video_integrity.another_parts_shape(_probe(29.97), _probe(fps))
    assert reason is not None and "frames a second" in reason


def test_another_size_is_not_another_part_whatever_its_rate():
    reason = video_integrity.another_parts_shape(_probe(30.0), _probe(30.0, (128, 96)))
    assert reason == "it is 128×96, and the recording's video is 64×48"
