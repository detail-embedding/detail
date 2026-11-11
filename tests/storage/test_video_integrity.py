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
