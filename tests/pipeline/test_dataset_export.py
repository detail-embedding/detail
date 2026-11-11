"""Dataset export runner: build a YOLO dataset from a detection run (§12)."""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from pycocotools import mask as mask_utils

from detail.core.dataclasses import DatasetExportJob
from detail.pipeline import dataset_export
from detail.storage import experiment as experiment_store
from detail.storage import runs as runs_store
from detail.storage import recording as recording_store
from detail.storage.zarr_io import DetectionPartWriter

H, W = 24, 32  # mask_resolution (height, width); the video is (W, H)


def _mask_counts(f):
    m = np.zeros((H, W), np.uint8)
    top = f % (H - 6)
    m[top : top + 6, 4:14] = 1  # a 6x10 rectangle
    return mask_utils.encode(np.asfortranarray(m))["counts"].decode("ascii")


def _write_video(path: Path, n_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    assert vw.isOpened()
    for f in range(n_frames):
        vw.write(np.full((H, W, 3), (f * 7) % 250 + 1, np.uint8))
    vw.release()


def _setup(tmp_path, n_frames=4):
    """A experiment + recording + completed detection run (det_01) over n_frames, and a
    experiment run record (run_001) indexing it."""
    proj = experiment_store.create_experiment(tmp_path, "p")
    recording_store.create_recording(proj, "recording_001")
    recording_dir = experiment_store.recording_path(proj, "recording_001")
    _write_video(recording_dir / "video" / "part_001.mp4", n_frames)

    data = recording_store.read_recording(recording_dir)
    data["video"] = {"files": [{"index": 0, "file": "video/part_001.mp4"}]}
    data["detection_runs"] = [
        {"id": "det_01", "status": "complete", "downsampling_run_id": None}
    ]
    recording_store.write_recording(recording_dir, data)

    det_dir = recording_dir / "detections" / "det_01"
    det_dir.mkdir(parents=True)
    with DetectionPartWriter(det_dir / "part_001.zarr") as writer:
        for f in range(n_frames):  # one "mouse" detection per frame
            writer.add_row(
                f, f * 1000, 0, "mouse", 0.9, (0.1, 0.1, 0.5, 0.5), _mask_counts(f)
            )
    (det_dir / "job.json").write_text(
        json.dumps(
            {"schema_version": 1, "status": "complete", "mask_resolution": [H, W]}
        ),
        encoding="utf-8",
    )

    runs_store.write_run(
        proj,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["recording_001"],
            "config": {},
            "status": "complete",
            "per_recording": {
                "recording_001": {"run_id": "det_01", "status": "complete"}
            },
            "created_at": "2026-06-09T00:00:00Z",
        },
    )
    return proj


def _job(**kw):
    base = dict(
        job_id="export_01",
        dataset_id="dataset_001",
        name="mouse_set",
        source_run_id="run_001",
        frame_selection={"mode": "all"},
        val_split=0.0,
        min_contour_area=1,
        created_at="2026-06-09T00:00:00Z",
    )
    base.update(kw)
    return DatasetExportJob(**base)


def _run(proj, job):
    run_dir = dataset_export.run_dir_for(proj, job)
    run_dir.mkdir(parents=True)
    dataset_export.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    return run_dir


def test_export_writes_images_labels_and_yaml(tmp_path):
    proj = _setup(tmp_path, n_frames=4)
    run_dir = _run(proj, _job())

    images = sorted((run_dir / "images" / "train").glob("*.jpg"))
    labels = sorted((run_dir / "labels" / "train").glob("*.txt"))
    assert len(images) == 4 and len(labels) == 4  # val_split 0 -> all train
    # each label is a YOLO polygon line: "<class_id> x y x y ..." normalized
    line = labels[0].read_text(encoding="utf-8-sig").strip()
    parts = line.split()
    assert parts[0] == "0"  # class id for the only class
    coords = [float(v) for v in parts[1:]]
    assert len(coords) >= 6 and all(0.0 <= v <= 1.0 for v in coords)
    # the extracted image is the detection-resolution frame
    img = cv2.imread(str(images[0]))
    assert img.shape == (H, W, 3)

    meta = yaml.safe_load((run_dir / "dataset.yaml").read_text(encoding="utf-8-sig"))
    assert meta["schema_version"] == 1
    assert meta["status"] == "complete"
    assert meta["source"] == {
        "kind": "run_export",
        "run_id": "run_001",
        "module": "detection",
        "frame_selection": {"mode": "all"},
    }
    assert meta["classes"] == [{"id": 0, "name": "mouse"}]
    assert meta["n_train"] == 4 and meta["n_val"] == 0
    data = yaml.safe_load((run_dir / "data.yaml").read_text(encoding="utf-8-sig"))
    assert data["names"] == {0: "mouse"} and data["path"] == "."


def test_export_every_n_selection(tmp_path):
    proj = _setup(tmp_path, n_frames=6)
    run_dir = _run(proj, _job(frame_selection={"mode": "every_n", "n": 2}))
    # frames 0,2,4 selected -> 3 images
    assert len(list((run_dir / "images" / "train").glob("*.jpg"))) == 3


def test_export_val_split_is_held_out(tmp_path):
    proj = _setup(tmp_path, n_frames=10)
    run_dir = _run(proj, _job(val_split=0.5))
    meta = yaml.safe_load((run_dir / "dataset.yaml").read_text(encoding="utf-8-sig"))
    assert meta["n_val"] == 5 and meta["n_train"] == 5  # stratified 50/50


def test_generate_dataset_api_end_to_end(tmp_path):
    from detail.api.experiment import Experiment

    proj_dir = _setup(tmp_path, n_frames=10)
    experiment = Experiment(proj_dir)
    # min_contour_area=1 so the tiny synthetic masks aren't filtered out
    run = experiment.generate_dataset(
        "mouse_set", "run_001", val_split=0.2, min_contour_area=1
    ).wait()
    assert run.status == "complete"
    refs = experiment.datasets
    assert len(refs) == 1
    assert refs[0].id == "dataset_001"
    assert refs[0].status == "complete"
    # 10 labeled frames, val_split 0.2 -> 8 train / 2 val (stratified)
    assert refs[0].n_train == 8 and refs[0].n_val == 2


def test_generate_dataset_missing_run_raises(tmp_path):
    import pytest

    from detail.api.experiment import Experiment

    proj_dir = _setup(tmp_path)
    with pytest.raises(FileNotFoundError):
        Experiment(proj_dir).generate_dataset("d", "run_999")


def test_generate_dataset_rejects_incomplete_run(tmp_path):
    import pytest

    from detail.api.experiment import Experiment

    proj_dir = _setup(tmp_path)
    rec = runs_store.read_run(proj_dir, "run_001")
    rec["status"] = "running"
    runs_store.write_run(proj_dir, "run_001", rec)
    with pytest.raises(ValueError, match="not complete"):
        Experiment(proj_dir).generate_dataset("d", "run_001")


def test_export_rejects_non_run_source(tmp_path):
    proj = _setup(tmp_path)
    runs_store.write_run(
        proj,
        "run_002",
        {
            "id": "run_002",
            "module_id": "embedding",
            "per_recording": {},
            "status": "complete",
        },
    )
    import pytest

    with pytest.raises(ValueError, match="detection or tracking run"):
        _run(proj, _job(dataset_id="dataset_002", source_run_id="run_002"))


def test_export_failure_marks_dataset_failed(tmp_path):
    """A crash after the initial dataset.yaml exists flips status to failed —
    not left stuck at 'running' (which would hide it from delete, §1.1)."""
    import pytest

    from detail.storage import datasets as datasets_store

    proj = _setup(tmp_path)
    runs_store.write_run(
        proj,
        "run_002",
        {
            "id": "run_002",
            "module_id": "embedding",
            "per_recording": {},
            "status": "complete",
        },
    )
    job = _job(dataset_id="dataset_002", source_run_id="run_002")
    run_dir = dataset_export.run_dir_for(proj, job)
    run_dir.mkdir(parents=True)
    # the initial record generate_dataset writes before submit (status: running)
    datasets_store.write_dataset_yaml(
        run_dir, {"id": "dataset_002", "name": "x", "status": "running"}
    )
    with pytest.raises(ValueError, match="detection or tracking run"):
        dataset_export.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)
    record = datasets_store.read_dataset_yaml(run_dir)
    assert record["status"] == "failed"
    assert "detection or tracking run" in record["error"]  # recorded for the GUI


def test_export_streams_intermediate_progress(tmp_path, monkeypatch):
    # 0.0 interval -> the shared throttle forwards every in-loop emit (the live-bar
    # contract: a single-part export shows intermediate progress, not one event).
    monkeypatch.setattr("detail.core.progress.PROGRESS_INTERVAL_S", 0.0)
    proj = _setup(tmp_path, n_frames=6)
    job = _job()  # min_contour_area=1, val_split=0.0 -> all 6 frames labeled
    run_dir = dataset_export.run_dir_for(proj, job)
    run_dir.mkdir(parents=True)
    events = []
    dataset_export.run(job, run_dir, progress_cb=events.append, stop_event=None)
    assert events, "no progress events emitted"
    total = events[-1].frames_total
    assert any(0 < e.frames_done < total for e in events), [
        e.frames_done for e in events
    ]
    assert events[-1].frames_done == total
    # #18: the runner populates eta_s (not None) on a mid-run event.
    assert any(e.eta_s is not None for e in events if 0 < e.frames_done < total)
