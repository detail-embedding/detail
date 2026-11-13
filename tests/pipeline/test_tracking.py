import json
import time

import numpy as np
import pytest
import zarr

from detail.core.dataclasses import JobStatus, TrackingJob
from detail.executor.local import LocalExecutor
from detail.executor.runners import JOB_RUNNERS
from detail.pipeline import tracking
from detail.storage import recording as recording_store
from detail.storage import spatial as spatial_store
from detail.storage.zarr_io import (
    DetectionPartReader,
    DetectionPartWriter,
    PartZarrReader,
)


def _write_part(path, frames):
    with DetectionPartWriter(path) as w:
        for frame_idx, dets in frames:
            for d_idx, (bbox, conf, cname) in enumerate(dets):
                w.add_row(
                    frame_idx, frame_idx * 1000, d_idx, cname, float(conf), bbox, ""
                )


def _drift(base, f, step=0.01):
    x1, y1, x2, y2 = base
    return (x1 + step * f, y1, x2 + step * f, y2)


def _tracking_job(tracker_id="anchor_tracker"):
    return TrackingJob(
        job_id="track_01",
        recording_id="recording_001",
        track_id="track_01",
        detection_run_id="det_01",
        tracker_id=tracker_id,
        hyperparams={"iou_threshold": 0.3},
        arena_ids=[],
        created_at="2026-01-01T00:00:00Z",
    )


def _build_detection_run(experiment_root):
    det = experiment_root / "recordings" / "recording_001" / "detections" / "det_01"
    det.mkdir(parents=True)
    _write_part(
        det / "part_001.zarr",
        [(f, [(_drift((0.1, 0.1, 0.2, 0.2), f), 0.9, "mouse")]) for f in range(4)],
    )
    # The visible trackers read the mask grid from the detection run's envelope.
    (det / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [24, 32]}), encoding="utf-8"
    )


def _build_two_class_detection_run(experiment_root):
    """Both a black mouse and a white mouse detected in every frame — the shape a
    2-class detection run has when two animals are told apart by class rather than
    by tracking (§13's class filter)."""
    det = experiment_root / "recordings" / "recording_001" / "detections" / "det_01"
    det.mkdir(parents=True)
    (det / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [24, 32]}), encoding="utf-8"
    )
    _write_part(
        det / "part_001.zarr",
        [
            (
                f,
                [
                    (_drift((0.1, 0.1, 0.2, 0.2), f), 0.9, "black mouse"),
                    (_drift((0.6, 0.6, 0.7, 0.7), f), 0.8, "white mouse"),
                ],
            )
            for f in range(4)
        ],
    )


def test_runner_registered_and_resolves_run_dir(tmp_path):
    assert "tracking" in JOB_RUNNERS
    rd = tracking.tracking_run_dir(tmp_path, _tracking_job())
    assert rd == tmp_path / "recordings" / "recording_001" / "tracking" / "track_01"


def test_runner_dispatches_to_tracker(tmp_path):
    _build_detection_run(tmp_path)
    run_dir = tmp_path / "recordings" / "recording_001" / "tracking" / "track_01"
    run_dir.mkdir(parents=True)
    tracking.run(_tracking_job(), run_dir, progress_cb=lambda e: None, stop_event=None)
    assert (run_dir / "track.yaml").exists()
    assign = PartZarrReader(run_dir / "assignments" / "part_001.zarr").read()[
        "object_id"
    ]
    assert assign.tolist() == [0, 0, 0, 0]


def test_runner_unknown_tracker_raises(tmp_path):
    run_dir = tmp_path / "recordings" / "recording_001" / "tracking" / "track_01"
    run_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="unknown tracker_id"):
        tracking.run(
            _tracking_job(tracker_id="nope"),
            run_dir,
            progress_cb=lambda e: None,
            stop_event=None,
        )


def _wait_terminal(ex, job_id, timeout=20):
    deadline = time.time() + timeout
    terminal = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.PARTIAL}
    while time.time() < deadline:
        if ex.status(job_id) in terminal:
            return ex.status(job_id)
        time.sleep(0.05)
    raise AssertionError(f"job never reached terminal (last={ex.status(job_id)})")


def test_runner_loads_arena_polygons_and_passes_through(
    tmp_path, monkeypatch, recording_dir
):
    # Build a detection run under the conftest recording_dir, then point the
    # runner at it. Capture what the tracker received.
    experiment_root = recording_dir.parent.parent
    det = recording_dir / "detections" / "det_01"
    det.mkdir(parents=True)
    _write_part(
        det / "part_001.zarr",
        [(f, [(_drift((0.1, 0.1, 0.2, 0.2), f), 0.9, "mouse")]) for f in range(4)],
    )
    # write timestamp CSV for part offset computation
    recording_store.write_timestamps_csv(
        recording_dir / "video" / "timestamps" / "generated" / "part_001.csv",
        np.arange(4, dtype=np.int64) * 1000,
    )
    arena_id = spatial_store.next_arena_id(recording_dir)
    spatial_store.write_arena_keyframes(
        recording_dir, arena_id, [(0, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])]
    )
    spatial_store.append_arena_to_recording_yaml(recording_dir, arena_id, "main", 4)

    captured = {}

    class _RecordingTracker:
        tracker_id = "anchor_tracker"

        def track(self, job, det_dir, out_dir, **kwargs):
            captured["arena_polygons"] = kwargs["arena_polygons"]
            captured["part_frame_offsets"] = kwargs["part_frame_offsets"]
            captured["arena_ids"] = list(job.arena_ids)

    monkeypatch.setitem(
        tracking.TRACKERS,
        "anchor_tracker",
        lambda: _RecordingTracker(),  # noqa: PLW0108
    )

    run_dir = recording_dir / "tracking" / "track_01"
    run_dir.mkdir(parents=True)
    job = TrackingJob(
        job_id="track_01",
        recording_id="recording_001",
        track_id="track_01",
        detection_run_id="det_01",
        tracker_id="anchor_tracker",
        hyperparams={"iou_threshold": 0.3},
        arena_ids=[arena_id],
        created_at="2026-01-01T00:00:00Z",
    )
    tracking.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)

    assert captured["arena_ids"] == [arena_id]
    assert captured["arena_polygons"] is not None
    assert arena_id in captured["arena_polygons"]
    assert captured["arena_polygons"][arena_id].n_vertices == 4
    # single part -> offsets just [0]
    assert captured["part_frame_offsets"] == [0]


def test_tracker_class_filter_produces_two_independent_clean_tracks(tmp_path):
    """The actual end-to-end proof: two classes, two single-subject tracking runs on
    the *same* detection run, each producing a track with only its own animal's
    detections — no cross-contamination, and no genuine multi-object association
    layer needed since detection already told the two animals apart.

    Without the filter, anchor_tracker's own selection (highest confidence wins) would
    *always* pick the black mouse (0.9 > 0.8) — proof the white-mouse run genuinely
    filtered the black mouse's rows out of its candidate pool before selection ever ran,
    rather than merely losing a tie-break.
    """
    _build_two_class_detection_run(tmp_path)

    for class_name in ("black mouse", "white mouse"):
        job = _tracking_job(tracker_id="anchor_tracker")
        job.track_id = f"track_{class_name.split()[0]}"
        job.hyperparams = {"conf_threshold": 0.05}
        job.class_name = class_name
        run_dir = tmp_path / "recordings" / "recording_001" / "tracking" / job.track_id
        run_dir.mkdir(parents=True)
        tracking.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)

        assigned = PartZarrReader(run_dir / "assignments" / "part_001.zarr").read()
        assert (assigned["object_id"] >= 0).all(), f"{class_name}: subject not found"
        tracks = PartZarrReader(run_dir / "tracks.zarr").read()
        assert tracks["class_name"].tolist() == [class_name]


def test_class_filter_survives_a_detection_run_torn_by_a_crash(tmp_path):
    """The user-facing bug this class of fix closes: a detection run interrupted
    mid-flush (a crash on a remote/multi-GPU worker) leaves one part's columns at
    unequal length until a *writer* reopens it (`storage/zarr_io.py`'s own
    documented recovery) — but tracking only ever opens it with a read-only
    `DetectionPartReader`. Before the reader learned the same "shortest column
    wins" policy, `filter_by_class`'s mask (built from `class_name`) crashed with
    `IndexError: boolean index did not match indexed array along axis 0` the
    moment it was applied to a column of a different length — with no traceback
    reaching the GUI, just that message, since the failure happens inside the
    tracking worker subprocess.
    """
    _build_two_class_detection_run(tmp_path)
    det = tmp_path / "recordings" / "recording_001" / "detections" / "det_01"
    root = zarr.open_group(str(det / "part_001.zarr"), mode="a")
    root["bbox"].resize((root["bbox"].shape[0] + 2, 4))  # torn: one column ahead

    job = _tracking_job(tracker_id="anchor_tracker")
    job.hyperparams = {"conf_threshold": 0.05}
    job.class_name = "white mouse"
    run_dir = tmp_path / "recordings" / "recording_001" / "tracking" / "track_01"
    run_dir.mkdir(parents=True)
    tracking.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)

    tracks = PartZarrReader(run_dir / "tracks.zarr").read()
    assert tracks["class_name"].tolist() == ["white mouse"]


def test_resolve_class_name_accepts_a_class_the_detection_run_has(tmp_path):
    _build_two_class_detection_run(tmp_path)
    recording_dir = tmp_path / "recordings" / "recording_001"
    resolved = tracking.TrackingModule._resolve_class_name(
        {"class_name": "black mouse"}, recording_dir, "det_01", "recording_001"
    )
    assert resolved == "black mouse"


def test_resolve_class_name_none_is_a_true_passthrough(tmp_path):
    _build_two_class_detection_run(tmp_path)
    recording_dir = tmp_path / "recordings" / "recording_001"
    assert (
        tracking.TrackingModule._resolve_class_name(
            {}, recording_dir, "det_01", "recording_001"
        )
        is None
    )


def test_resolve_class_name_rejects_an_unknown_class(tmp_path):
    _build_two_class_detection_run(tmp_path)
    recording_dir = tmp_path / "recordings" / "recording_001"
    with pytest.raises(ValueError, match="no class named 'orange mouse'"):
        tracking.TrackingModule._resolve_class_name(
            {"class_name": "orange mouse"}, recording_dir, "det_01", "recording_001"
        )


def test_tracking_end_to_end_via_local_executor(tmp_path):
    _build_detection_run(tmp_path)
    ex = LocalExecutor(
        tmp_path,
        runner_modules=("detail.pipeline.tracking",),
        poll_interval=0.02,
    )
    job_id = ex.submit(_tracking_job())
    assert _wait_terminal(ex, job_id) == JobStatus.COMPLETE

    track_dir = tmp_path / "recordings" / "recording_001" / "tracking" / "track_01"
    assert (track_dir / "track.yaml").exists()
    assign = PartZarrReader(track_dir / "assignments" / "part_001.zarr").read()[
        "object_id"
    ]
    assert assign.tolist() == [0, 0, 0, 0]


# --- no-arena multi-part global frame numbering (review finding #3) ---------- #
def _two_part_recording_yaml(root, part_lengths, fps=30.0):
    """Minimal recording.yaml with one part per entry in ``part_lengths`` (timestamp
    CSVs only — tracking reads the manifest + timestamps, never the .mp4)."""
    sdir = root / "recordings" / "recording_001"
    files = []
    for i, n in enumerate(part_lengths):
        rel = f"video/timestamps/generated/part_{i + 1:03d}.csv"
        (sdir / rel).parent.mkdir(parents=True, exist_ok=True)
        recording_store.write_timestamps_csv(
            sdir / rel, np.arange(n, dtype=np.int64) * int(1e9 / fps)
        )
        files.append(
            {
                "index": i,
                "file": f"video/part_{i + 1:03d}.mp4",
                "timestamps": {"generated": rel, "user": None, "active": "generated"},
            }
        )
    recording_store.write_recording(
        sdir,
        {
            "id": "recording_001",
            "name": "s",
            "created_at": "x",
            "video": {"fps_actual": fps, "files": files},
        },
    )
    return sdir


@pytest.mark.parametrize(
    "tracker_id,hyperparams",
    [
        ("anchor_tracker", {"conf_threshold": 0.1, "require_contiguous": False}),
    ],
)
def test_no_arena_multipart_uses_true_part_length_for_global_frames(
    tmp_path, tracker_id, hyperparams
):
    """Global frame numbering must offset each part by its **true length**, not the
    last *detected* frame. Part 1 is 5 frames but only 0-2 carry detections; part
    2's detections (local 0-2) are global 5-7 — so the last global frame is 7, not
    3+2=5 (the old detection-derived fallback)."""
    _two_part_recording_yaml(tmp_path, part_lengths=[5, 3])
    det = tmp_path / "recordings" / "recording_001" / "detections" / "det_01"
    det.mkdir(parents=True)
    (det / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [24, 32]}), encoding="utf-8"
    )
    box = (0.1, 0.1, 0.3, 0.3)
    _write_part(det / "part_001.zarr", [(f, [(box, 0.9, "mouse")]) for f in (0, 1, 2)])
    _write_part(det / "part_002.zarr", [(f, [(box, 0.9, "mouse")]) for f in (0, 1, 2)])

    run_dir = tmp_path / "recordings" / "recording_001" / "tracking" / "track_01"
    run_dir.mkdir(parents=True)
    job = TrackingJob(
        job_id="track_01",
        recording_id="recording_001",
        track_id="track_01",
        detection_run_id="det_01",
        tracker_id=tracker_id,
        hyperparams=hyperparams,
        arena_ids=[],
        created_at="2026-01-01T00:00:00Z",
    )
    tracking.run(job, run_dir, progress_cb=lambda e: None, stop_event=None)

    tracks = PartZarrReader(run_dir / "tracks.zarr").read()
    assert int(max(tracks["last_frame"])) == 7
    assert int(min(tracks["first_frame"])) == 0


def test_part_offsets_accumulate_source_part_lengths(tmp_path):
    """Offsets are the source video's per-part lengths. Every stage's rows index the
    source frames — a stride thins which are processed, it does not renumber them — so
    there is one offset table however a run was thinned."""
    sdir = _two_part_recording_yaml(tmp_path, part_lengths=[100, 80])
    assert recording_store.part_frame_offsets(sdir) == [0, 100]


def _stride_detection_run(root, frames):
    """A detection run whose rows sit on ``frames`` — one steady box, never absent."""
    det = root / "recordings" / "recording_001" / "detections" / "det_01"
    det.mkdir(parents=True)
    _write_part(
        det / "part_001.zarr",
        [(f, [((0.4, 0.4, 0.6, 0.6), 0.9, "mouse")]) for f in frames],
    )
    # The visible trackers read the mask grid from the detection run's envelope.
    (det / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [24, 32]}), encoding="utf-8"
    )
    return det


def test_single_subject_tracker_ages_at_the_run_cadence(tmp_path):
    """The same regression on a tracker a user can pick, where losing the subject shows.

    ``iou_single_tracker`` only ever writes one id, so "one track" cannot fail there. What
    ageing wrongly costs it is the *lock*: past ``max_missed`` it re-acquires on confidence
    alone. So a second, more confident animal stands elsewhere from the second processed
    frame on. Walking at the cadence, the subject never misses a frame and continuity keeps
    it; walking every source frame, three empty frames expire the lock and the tracker
    jumps to the distractor.

    Revert ``frame_walk``'s step to 1 and this fails.
    """
    from detail.trackers.iou_single_tracker import IoUSingleTracker

    frames = [0, 4, 8, 12, 16, 20]
    subject, distractor = (0.4, 0.4, 0.6, 0.6), (0.05, 0.05, 0.2, 0.2)
    det_dir = tmp_path / "recordings" / "recording_001" / "detections" / "det_01"
    det_dir.mkdir(parents=True)
    _write_part(
        det_dir / "part_001.zarr",
        [
            (f, [(subject, 0.8, "mouse")] + ([(distractor, 0.9, "mouse")] if f else []))
            for f in frames
        ],
    )
    (det_dir / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [24, 32]}), encoding="utf-8"
    )
    out_dir = tmp_path / "recordings" / "recording_001" / "tracking" / "track_01"
    out_dir.mkdir(parents=True)
    job = _tracking_job(tracker_id="iou_single_tracker")
    job.every_n_frames = 4
    job.hyperparams = {"max_missed": 2}

    IoUSingleTracker().track(job, det_dir, out_dir, progress_cb=None, stop_event=None)

    detections = DetectionPartReader(det_dir / "part_001.zarr").read()
    assigned = PartZarrReader(out_dir / "assignments" / "part_001.zarr").read()
    followed = {
        tuple(round(float(v), 2) for v in box)
        for box, oid in zip(detections["bbox"], assigned["object_id"])
        if int(oid) >= 0
    }
    assert followed == {subject}, (
        f"expected the subject followed across {frames}, got {sorted(followed)} — the "
        "tracker aged on frames the run never processed and re-acquired the distractor"
    )


def test_tracker_stride_ignores_rows_off_its_cadence(tmp_path):
    """Tracking at a coarser stride than detection links only its own frames; the
    detection rows in between are left unassigned rather than silently linked."""
    from detail.trackers.anchor_tracker import AnchorTracker

    det_dir = _stride_detection_run(tmp_path, [0, 2, 4, 6, 8])
    out_dir = tmp_path / "recordings" / "recording_001" / "tracking" / "track_02"
    out_dir.mkdir(parents=True)
    job = _tracking_job()
    job.every_n_frames = 4  # detection ran at 2; link every 4th source frame

    AnchorTracker().track(job, det_dir, out_dir, progress_cb=None, stop_event=None)

    frame_idx = DetectionPartReader(det_dir / "part_001.zarr").column("frame_idx")[:]
    assigned = PartZarrReader(out_dir / "assignments" / "part_001.zarr").read()
    linked = {
        int(f) for f, oid in zip(frame_idx, assigned["object_id"]) if int(oid) >= 0
    }
    assert linked == {0, 4, 8}


def test_tracking_record_never_gains_a_compute_field():
    """Tracking has no ComputeTarget (it always runs local) — locked down so a
    future "helpful" change doesn't add one it can't act on (§ run
    location/duration/cost scope decision)."""
    module = tracking.TrackingModule()
    fields = module.extra_record_fields(
        {"detection_run_id": "det_01", "tracker_id": "anchor_tracker"}
    )
    assert "compute" not in fields
