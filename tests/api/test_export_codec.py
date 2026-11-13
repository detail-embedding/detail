"""The codec an export asks for is the codec in the file, and the record says which (Tenet 15).

Over the export catalogue, end to end through ``exporters.write_export`` on a real run: the
``codec`` option reaches the writer (read back off the file with PyAV), ``export.yaml`` records
``video_codec``, and a machine that cannot encode H.264 writes mp4v and **says so in the record**
— the silent ``avc1`` → ``mp4v`` fallback this replaced is the failure being guarded. The
registry half: a target that writes an mp4 declares ``writes_video``, so every surface that
offers the codec (the Export dialog, ``export run --codec``) offers it for that target.
"""

import av
import pytest

from detail.api import export_targets  # noqa: F401  (registers the catalogue)
from detail.api import exporters
from detail.api.exporters import ExportScope, ExportSubject
from detail.api.montage_export import ExportFormat
from detail.storage import video as video_store
from tests.api.test_export_preview_targets import _experiment_dir, _export, _write_run
from tests.api.test_run_preview import _recording_with_detection

_STREAM_NAME = {"h264": "h264", "mp4v": "mpeg4"}


@pytest.fixture
def detection_run(tmp_path):
    recording = _recording_with_detection(tmp_path, n_frames=5)
    experiment_dir = _experiment_dir(recording)
    _write_run(experiment_dir, "det_run_01", "detection", "s1", "det_01")
    return ExportSubject("run", experiment_dir, "det_run_01")


def _stream(path) -> tuple[str, str]:
    with av.open(str(path)) as container:
        ctx = container.streams.video[0].codec_context
        return ctx.name, ctx.pix_fmt


@pytest.mark.parametrize("codec", ["h264", "mp4v"])
def test_the_codec_asked_for_is_in_the_file_and_the_record(
    tmp_path, detection_run, codec
):
    result = _export(
        detection_run, "preview_video", tmp_path / "out", recording_id="s1", codec=codec
    )
    assert result.status == "complete"
    written = video_store.written_codec(codec)
    name, pix_fmt = _stream(result.dest / "s1_preview.mp4")
    assert name == _STREAM_NAME[written] and pix_fmt == "yuv420p"
    manifest = exporters.read_manifest(result.dest)
    assert manifest["video_codec"]["asked"] == codec
    assert manifest["video_codec"]["written"] == written


def test_h264_is_the_default(tmp_path, detection_run):
    if not video_store.h264_available():
        pytest.skip("this PyAV build has no libx264; the fallback test covers this box")
    result = _export(
        detection_run, "preview_video", tmp_path / "out", recording_id="s1"
    )
    assert _stream(result.dest / "s1_preview.mp4")[0] == "h264"
    assert exporters.read_manifest(result.dest)["video_codec"] == {
        "asked": "h264",
        "written": "h264",
    }


def test_a_fallback_is_written_into_the_record(tmp_path, detection_run, monkeypatch):
    monkeypatch.setattr(video_store, "h264_available", lambda: False)
    result = _export(
        detection_run, "preview_video", tmp_path / "out", recording_id="s1"
    )
    assert _stream(result.dest / "s1_preview.mp4")[0] == "mpeg4"
    record = exporters.read_manifest(result.dest)["video_codec"]
    assert record["asked"] == "h264" and record["written"] == "mp4v"
    assert "PowerPoint" in record["note"]


def test_an_export_without_video_records_no_codec(tmp_path, detection_run):
    result = _export(
        detection_run,
        "preview_frame",
        tmp_path / "out",
        scope=ExportScope(frames=(2, 2)),
        recording_id="s1",
    )
    assert "video_codec" not in exporters.read_manifest(result.dest)


def test_every_target_that_writes_an_mp4_declares_it():
    """Data-driven over the catalogue, so a video target added tomorrow is held to it: its
    artifact says mp4 exactly when it declares ``writes_video``."""
    exporters._load()
    wrong = [
        (t.subject_kind, t.id)
        for t in exporters.all_targets()
        if t.writes_video != ("mp4" in t.artifact.lower())
    ]
    assert not wrong, f"writes_video disagrees with the artifact for {wrong}"
    assert any(t.writes_video for t in exporters.all_targets())


def test_a_montage_format_refuses_an_unknown_codec():
    assert ExportFormat("mp4").codec == "h264"
    with pytest.raises(ValueError, match="unknown video codec 'avc1'"):
        ExportFormat("mp4", codec="avc1")
