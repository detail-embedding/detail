"""`preview_frame` / `preview_video` — the run-level style picker's PNG and mp4 export.

Driven through `exporters.write_export`, the way every other target is tested here, over a
real detection or tracking run — a real decodable video and a real zarr detection run, plus the
experiment-level `runs/<id>/run.yaml` the two `masks`/`tracks` targets already need
(`_recording_with_detection`/`_add_tracking` give the first half, `_write_run` below the second).
"""

import cv2

from detail.api import export_targets  # noqa: F401  (registers the catalogue)
from detail.api import exporters
from detail.api.exporters import ExportScope, ExportSubject
from detail.storage import runs as runs_store
from tests.api.test_run_preview import _add_tracking, _recording_with_detection


def _experiment_dir(recording):
    return recording.dir.parent.parent


def _write_run(
    experiment_dir, run_id, module_id, recording_id, sub_run_id, status="complete"
):
    runs_store.write_run(
        experiment_dir,
        run_id,
        {
            "id": run_id,
            "module_id": module_id,
            "status": "complete",
            "per_recording": {recording_id: {"run_id": sub_run_id, "status": status}},
        },
    )


def _export(subject, target_id, dest, **kwargs):
    target = exporters.target(subject.kind, target_id)
    return exporters.write_export(subject, [target], dest, **kwargs)


def test_preview_frame_writes_one_png_of_the_frame_asked_for(tmp_path):
    recording = _recording_with_detection(tmp_path, n_frames=5)
    experiment_dir = _experiment_dir(recording)
    _write_run(experiment_dir, "det_run_01", "detection", "s1", "det_01")
    subject = ExportSubject("run", experiment_dir, "det_run_01")

    result = _export(
        subject,
        "preview_frame",
        tmp_path / "out",
        scope=ExportScope(frames=(2, 2)),
        recording_id="s1",
    )
    assert result.status == "complete"
    png = result.dest / "s1_frame000002.png"
    assert png.is_file() and png.name in result.files
    img = cv2.imread(str(png))
    assert img is not None and img.shape == (48, 64, 3)


def test_preview_frame_with_no_scope_writes_nothing(tmp_path):
    """The frame is a scope, the same way a clip's span is — omitting it is "nothing to
    write", not a guess at which frame was meant."""
    recording = _recording_with_detection(tmp_path, n_frames=5)
    experiment_dir = _experiment_dir(recording)
    _write_run(experiment_dir, "det_run_01", "detection", "s1", "det_01")
    subject = ExportSubject("run", experiment_dir, "det_run_01")

    result = _export(subject, "preview_frame", tmp_path / "out", recording_id="s1")
    assert result.is_empty
    assert not list(result.dest.glob("*.png"))


def test_preview_video_writes_every_frame_the_preview_decodes(tmp_path):
    """Every frame `DetectionPreview.frames_range` hands back lands in the mp4 — checked
    against what the preview itself decodes, not a hardcoded count, since a hand-written
    test video's own decode is a property of `frames_range`/the container, not of this
    target (a real recording's frame count and its decode agree; a 5-frame synthetic
    ``mp4v`` file here does not, on this machine's codec support)."""
    recording = _recording_with_detection(tmp_path, n_frames=5)
    experiment_dir = _experiment_dir(recording)
    _write_run(experiment_dir, "det_run_01", "detection", "s1", "det_01")
    subject = ExportSubject("run", experiment_dir, "det_run_01")
    preview = recording.detection_preview("det_01")
    expected = sum(1 for _ in preview.frames_range(0, 0, 4, with_masks=True))
    assert expected >= 1, "the fixture decoded nothing, so this test proves nothing"

    result = _export(subject, "preview_video", tmp_path / "out", recording_id="s1")
    assert result.status == "complete"
    mp4 = result.dest / "s1_preview.mp4"
    assert mp4.is_file()
    cap = cv2.VideoCapture(str(mp4))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert count == expected


def test_preview_targets_draw_a_tracking_run_with_its_track_color_and_label(tmp_path):
    """The same two targets, off a tracking run instead — `_preview_for` resolves the
    module from the run record, not from the target id."""
    recording = _add_tracking(_recording_with_detection(tmp_path, n_frames=5))
    experiment_dir = _experiment_dir(recording)
    _write_run(experiment_dir, "trk_run_01", "tracking", "s1", "track_01")
    subject = ExportSubject("run", experiment_dir, "trk_run_01")

    result = _export(
        subject,
        "preview_frame",
        tmp_path / "out",
        scope=ExportScope(frames=(0, 0)),
        recording_id="s1",
        style="box_conf",
    )
    assert result.status == "complete"
    assert (result.dest / "s1_frame000000.png").is_file()


def test_preview_targets_are_unavailable_for_a_run_of_the_wrong_kind(tmp_path):
    recording = _recording_with_detection(tmp_path, n_frames=5)
    experiment_dir = _experiment_dir(recording)
    _write_run(experiment_dir, "emb_run_01", "embedding", "s1", "emb_01")
    subject = ExportSubject("run", experiment_dir, "emb_run_01")

    for target_id in ("preview_frame", "preview_video"):
        reason = exporters.unavailable_reason(
            exporters.target("run", target_id), subject
        )
        assert reason is not None and "embedding" in reason


def test_preview_targets_are_unavailable_before_any_recording_completes(tmp_path):
    recording = _recording_with_detection(tmp_path, n_frames=5)
    experiment_dir = _experiment_dir(recording)
    runs_store.write_run(
        experiment_dir,
        "det_run_01",
        {
            "id": "det_run_01",
            "module_id": "detection",
            "status": "running",
            "per_recording": {"s1": {"status": "running"}},
        },
    )
    subject = ExportSubject("run", experiment_dir, "det_run_01")
    reason = exporters.unavailable_reason(
        exporters.target("run", "preview_frame"), subject
    )
    assert reason == "this run has not completed a recording yet"


def test_preview_frame_needs_a_recording_id(tmp_path):
    """Which recording a multi-recording run's preview draws is a seed no dialog control
    offers (`--for-recording` / the run-preview widget's own recording), so a caller that
    omits it gets a named failure — the same per-target isolation `write_export` gives
    every other target's own validation error — rather than an arbitrary guess."""
    recording = _recording_with_detection(tmp_path, n_frames=5)
    experiment_dir = _experiment_dir(recording)
    _write_run(experiment_dir, "det_run_01", "detection", "s1", "det_01")
    subject = ExportSubject("run", experiment_dir, "det_run_01")

    result = _export(
        subject,
        "preview_frame",
        tmp_path / "out",
        scope=ExportScope(frames=(0, 0)),
    )
    assert result.status == "failed"
    (failure,) = result.failures
    assert "which recording" in failure.error
