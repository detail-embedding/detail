"""GUI config widget for track_merge (§13): two tracking-spec pickers."""

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.core.dataclasses import TrackingJob  # noqa: E402
from detail.pipeline import tracking  # noqa: E402
from detail.storage import recording as recording_store  # noqa: E402
from detail.storage.zarr_io import DetectionPartWriter  # noqa: E402


def _two_class_recording(tmp_path):
    proj = Experiment.create("p", tmp_path)
    sess = proj.create_recording("recording_001")
    sdir = sess.dir
    det_dir = sdir / "detections" / "det_01"
    det_dir.mkdir(parents=True)
    (det_dir / "job.json").write_text(
        '{"schema_version": 1, "mask_resolution": [24, 32]}', encoding="utf-8"
    )
    with DetectionPartWriter(det_dir / "part_001.zarr") as w:
        w.add_row(0, 0, 0, "black mouse", 0.9, (0.1, 0.1, 0.2, 0.2), "")
        w.add_row(0, 0, 1, "white mouse", 0.8, (0.6, 0.6, 0.7, 0.7), "")
    data = recording_store.read_recording(sdir)
    data.setdefault("detection_runs", []).append(
        {"id": "det_01", "status": "complete", "model_id": "sam3"}
    )
    recording_store.write_recording(sdir, data)

    for track_id, class_name in (
        ("track_black", "black mouse"),
        ("track_white", "white mouse"),
    ):
        job = TrackingJob(
            job_id=track_id,
            recording_id=sess.id,
            track_id=track_id,
            detection_run_id="det_01",
            tracker_id="anchor_tracker",
            hyperparams={"conf_threshold": 0.05},
            arena_ids=[],
            created_at="2026-01-01T00:00:00Z",
            class_name=class_name,
        )
        run_dir = sdir / "tracking" / track_id
        run_dir.mkdir(parents=True)
        tracking.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
        data = recording_store.read_recording(sdir)
        data.setdefault("tracking_runs", []).append(
            {
                "id": track_id,
                "status": "complete",
                "detection_run_id": "det_01",
                "tracker_id": "anchor_tracker",
                "arena_ids": [],
                "class_name": class_name,
                "hyperparams": {"conf_threshold": 0.05},
                "every_n_frames": 1,
                "created_at": "2026-01-01T00:00:00Z",
            }
        )
        recording_store.write_recording(sdir, data)
    return sess


def test_track_merge_config_populates_and_covers(qapp, tmp_path):
    from detail.gui.modules.track_merge import _TrackMergeConfig

    sess = _two_class_recording(tmp_path)
    widget = _TrackMergeConfig()
    widget.apply_eligible_recordings([sess], widget.resolve_eligible_recordings([sess]))

    assert widget._rows[0].count() == 2  # black mouse's spec + white mouse's spec
    labels = {widget._rows[0].itemText(i) for i in range(widget._rows[0].count())}
    assert any("anchor_tracker" in label for label in labels)

    assert widget.validation_error() is not None  # nothing picked yet

    black_index = next(
        i
        for i in range(widget._rows[0].count())
        if widget._rows[0].itemData(i) is not None
        and widget._rows[0].itemData(i)["identity"]["class_name"] == "black mouse"
    )
    white_index = next(
        i
        for i in range(widget._rows[1].count())
        if widget._rows[1].itemData(i) is not None
        and widget._rows[1].itemData(i)["identity"]["class_name"] == "white mouse"
    )
    widget._rows[0].setCurrentIndex(black_index)
    widget._rows[1].setCurrentIndex(white_index)

    assert widget.validation_error() is None
    assert widget.covered_recording_ids() == {"recording_001"}
    cfg = widget.config()
    assert len(cfg["source_specs"]) == 2
    assert {s["identity"]["class_name"] for s in cfg["source_specs"]} == {
        "black mouse",
        "white mouse",
    }


def test_track_merge_config_rejects_the_same_source_twice(qapp, tmp_path):
    from detail.gui.modules.track_merge import _TrackMergeConfig

    sess = _two_class_recording(tmp_path)
    widget = _TrackMergeConfig()
    widget.apply_eligible_recordings([sess], widget.resolve_eligible_recordings([sess]))

    black_index = next(
        i
        for i in range(widget._rows[0].count())
        if widget._rows[0].itemData(i) is not None
        and widget._rows[0].itemData(i)["identity"]["class_name"] == "black mouse"
    )
    widget._rows[0].setCurrentIndex(black_index)
    widget._rows[1].setCurrentIndex(black_index)
    assert "more than once" in widget.validation_error()
