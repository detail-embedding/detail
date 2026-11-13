"""Embedding's by-subject track selector (§13): a merge run's object_id is a
position, not an animal, so the config widget offers subject-name checkboxes
instead of a raw integer field when the picked tracking run is a track_merge
output."""

import json

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.core.dataclasses import TrackingJob, TrackMergeJob  # noqa: E402
from detail.pipeline import track_merge, tracking  # noqa: E402
from detail.storage import recording as recording_store  # noqa: E402
from detail.storage.zarr_io import DetectionPartWriter  # noqa: E402


def _merged_recording(tmp_path):
    """A real detection run, two class-filtered tracking runs, and a real
    track_merge run over them — the actual artifact shape embedding reads."""
    proj = Experiment.create("p", tmp_path)
    sess = proj.create_recording("recording_001")
    sdir = sess.dir
    det_dir = sdir / "detections" / "det_01"
    det_dir.mkdir(parents=True)
    (det_dir / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [24, 32]}),
        encoding="utf-8",
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

    merge_job = TrackMergeJob(
        job_id="track_merged",
        recording_id=sess.id,
        track_id="track_merged",
        detection_run_id="det_01",
        source_run_ids=["track_black", "track_white"],
        created_at="2026-01-01T00:00:00Z",
    )
    merge_dir = sdir / "tracking" / "track_merged"
    merge_dir.mkdir(parents=True)
    track_merge.run(merge_job, merge_dir, progress_cb=lambda e: None, stop_event=None)
    data = recording_store.read_recording(sdir)
    data.setdefault("tracking_runs", []).append(
        {
            "id": "track_merged",
            "status": "complete",
            "detection_run_id": "det_01",
            "source_run_ids": ["track_black", "track_white"],
            "tracker_id": "track_merge",
            "arena_ids": [],
            "class_name": "black mouse+white mouse",
            "hyperparams": {},
            "every_n_frames": 1,
            "created_at": "2026-01-01T00:00:01Z",
        }
    )
    recording_store.write_recording(sdir, data)
    return sess


def _select(widget, recordings):
    widget.apply_eligible_recordings(
        recordings, widget.resolve_eligible_recordings(recordings)
    )
    widget.on_recordings_changed(recordings)


def test_embedding_offers_subject_checkboxes_for_a_merged_run(qapp, tmp_path):
    from detail.gui.modules.embedding import _EmbeddingConfig

    sess = _merged_recording(tmp_path)
    widget = _EmbeddingConfig()
    _select(widget, [sess])

    merged_index = next(
        i
        for i in range(widget._tracking.count())
        if isinstance(widget._tracking.itemData(i), dict)
        and widget._tracking.itemData(i)["identity"].get("tracker_id") == "track_merge"
    )
    widget._tracking.setCurrentIndex(merged_index)

    assert not widget._subjects_row.isHidden()
    assert widget._tracks_row.isHidden()
    names = {c.text() for c in widget._subject_checks}
    assert names == {"black mouse", "white mouse"}
    assert all(c.isChecked() for c in widget._subject_checks)  # default: every subject

    cfg = widget.config()
    assert cfg["all_tracks"] is False
    assert set(cfg["track_ids"]) == {"black mouse", "white mouse"}

    # Uncheck one: only the remaining subject's name is sent.
    next(c for c in widget._subject_checks if c.text() == "white mouse").setChecked(
        False
    )
    cfg = widget.config()
    assert cfg["track_ids"] == ["black mouse"]
    assert widget.validation_error() is None

    # Uncheck every subject: a clear validation error, not a silent empty submit.
    for c in widget._subject_checks:
        c.setChecked(False)
    assert "Pick at least one subject" in widget.validation_error()


def test_embedding_uses_the_raw_field_for_an_ordinary_tracking_run(qapp, tmp_path):
    """No regression: a plain (non-merge) tracking run keeps the integer field."""
    from detail.gui.modules.embedding import _EmbeddingConfig

    sess = _merged_recording(tmp_path)
    widget = _EmbeddingConfig()
    _select(widget, [sess])

    plain_index = next(
        i
        for i in range(widget._tracking.count())
        if isinstance(widget._tracking.itemData(i), dict)
        and widget._tracking.itemData(i)["identity"].get("tracker_id")
        == "anchor_tracker"
    )
    widget._tracking.setCurrentIndex(plain_index)

    assert widget._subjects_row.isHidden()
    assert not widget._tracks_row.isHidden()
    cfg = widget.config()
    assert isinstance(cfg["track_ids"], list)
    assert all(isinstance(t, int) for t in cfg["track_ids"])
