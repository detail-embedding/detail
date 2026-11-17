"""The builders behind the shared experiment fixtures, callable at any fixture scope.

``tests/conftest.py``'s ``experiment_dir``, ``recording_dir`` and ``make_embedding_run`` are
function-scoped, as ``DESIGN_PLAN.md`` Tenet 16 specifies, because almost every test wants a
private experiment it can write to. A handful of tests want something else: one **expensive**
artifact built once and copied per test. A module- or session-scoped fixture cannot request a
function-scoped one, so without a seam the only way to get there is to write the setup out again
-- which Tenet 16 forbids in the same paragraph that specifies the fixtures ("never define
equivalent setup inline").

So the bodies live here and the fixtures are the interface to them. One definition, two callers:
the fixture for the common case, a direct call for a wider-scoped builder. Nothing about the
fixtures' names, scope or behaviour changed when this moved.

The first caller of the second kind was ``tests/pipeline/test_pose_clustering_module.py`` (removed
with pose clustering), where 28 tests each paid ~22s to recompute one identical deterministic
fit (#534).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import yaml


def build_experiment_dir(path: Path) -> Path:
    """Minimal valid experiment directory at ``path``. The body of ``experiment_dir``."""
    for d in ("recordings", "analysis", "models", "datasets"):
        (path / d).mkdir()
    (path / "config.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "name": "test_experiment",
                "created_at": "2026-01-01T00:00:00Z",
            },
            sort_keys=False,  # schema_version stays first (Tenet 7), like the real writers
        ),
        encoding="utf-8",
    )
    return path


def build_recording_dir(experiment_dir: Path) -> Path:
    """Minimal valid recording directory with recording.yaml and all stage subdirs.

    The body of ``recording_dir``.
    """
    s = experiment_dir / "recordings" / "recording_001"
    for d in ("video/timestamps/generated", "detections", "tracking", "embeddings"):
        (s / d).mkdir(parents=True)
    (s / "recording.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": "recording_001",
                "name": "recording_001",
                "created_at": "2026-01-01T00:00:00Z",
                "video": {
                    "storage": "default",
                    "fps_container": 30.0,
                    "fps_actual": 30.0,
                    "fps_source": "container",
                    "resolution": [640, 480],
                    "total_duration_s": 10.0,
                    "files": [
                        {
                            "index": 0,
                            "file": "video/part_001.mp4",
                            "timestamps": {
                                "generated": "video/timestamps/generated/part_001.csv",
                                "user": None,
                                "active": "generated",
                            },
                        }
                    ],
                },
            },
            sort_keys=False,  # schema_version stays first (Tenet 7)
        ),
        encoding="utf-8",
    )
    return s


def add_video_parts(
    recording_dir: Path, part_lengths: list[int], period_ns: int = 33_333_333
) -> np.ndarray:
    """Give a recording several video parts with real timestamp maps; return the whole map.

    Each part's map is written **recording-wide**, as ``Recording.add_video_part`` writes it
    (``local_t_ns + offset_ns``): part 2's first frame continues part 1's clock. Returns the
    concatenated map, one ``t_ns`` per recording frame (``Recording.frame_timestamps``).
    No video files are written; nothing that reads a map needs one.
    """
    from detail.storage import recording as recording_store

    data = recording_store.read_recording(recording_dir)
    files, maps, frame = [], [], 0
    for index, length in enumerate(part_lengths):
        stem = f"part_{index + 1:03d}"
        rel = f"video/timestamps/generated/{stem}.csv"
        stamps = (np.arange(length, dtype=np.int64) + frame) * period_ns
        frame += length
        (recording_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        recording_store.write_timestamps_csv(recording_dir / rel, stamps)
        maps.append(stamps)
        files.append(
            {
                "index": index,
                "id": stem,
                "file": f"video/{stem}.mp4",
                "timestamps": {"generated": rel, "user": None, "active": "generated"},
            }
        )
    data["video"] = {**(data.get("video") or {}), "files": files}
    recording_store.write_recording(recording_dir, data)
    return np.concatenate(maps)


def add_tracking_parts(
    recording_dir: Path, whole_map: np.ndarray, part_lengths: list[int]
) -> None:
    """A complete ``det_01`` / ``track_01`` over every part of `add_video_parts`' recording,
    subject 0 detected and tracked on every frame.

    Each part's stores hold **part-local** ``frame_idx`` (§5) and recording-wide ``t_ns`` from
    the map, as the stages write them.
    """
    import json

    from detail.storage import recording as recording_store
    from detail.storage.zarr_io import AssignmentPartWriter, DetectionPartWriter

    data = recording_store.read_recording(recording_dir)
    data["detection_runs"] = [{"id": "det_01", "status": "complete"}]
    data["tracking_runs"] = [
        {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
    ]
    recording_store.write_recording(recording_dir, data)
    det_dir = recording_dir / "detections" / "det_01"
    trk_dir = recording_dir / "tracking" / "track_01"
    det_dir.mkdir(parents=True, exist_ok=True)
    (trk_dir / "assignments").mkdir(parents=True, exist_ok=True)
    (det_dir / "job.json").write_text(
        json.dumps({"schema_version": 1, "status": "complete"}), encoding="utf-8"
    )
    (trk_dir / "track.yaml").write_text(
        yaml.dump({"schema_version": 1, "detection_run_id": "det_01"}),
        encoding="utf-8",
    )
    first = 0
    for number, length in enumerate(part_lengths, start=1):
        stem = f"part_{number:03d}"
        with DetectionPartWriter(det_dir / f"{stem}.zarr") as writer:
            for local in range(length):
                writer.add_row(
                    local,
                    int(whole_map[first + local]),
                    0,
                    "mouse",
                    0.9,
                    (0.1, 0.1, 0.2, 0.2),
                    "",
                )
        with AssignmentPartWriter(trk_dir / "assignments" / f"{stem}.zarr") as writer:
            for _ in range(length):
                writer.add_object_id(0)
        first += length


def write_video(path: Path, n_frames: int = 5, size: int = 40) -> Path:
    """A tiny constant-rate mp4 whose frame ``i`` is the flat grey ``20 + 45 i`` (clipped to 255
    from the sixth frame on), so a decoded picture says which frame it is. The video behind a fixture recording that only wrote its
    timestamps (``processed_recording``), for the stages that decode frames."""
    import cv2
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (size, size)
    )
    for i in range(n_frames):
        writer.write(np.full((size, size, 3), min(20 + i * 45, 255), np.uint8))
    writer.release()
    return path


def write_embedding_run(
    recording_dir,
    emb,
    frame_idx,
    t_ns,
    *,
    embed_run_id="emb_01",
    track_id=0,
    renderer_spec=None,
    model_id="dinov3_vit",
    model_variant="vit_large",
    tracking_run_id="track_01",
    status="complete",
    constructor=None,
    readout=None,
    input_size=None,
    canvas_scope="per_recording",
    bg_color_scope="shared",
    channel_order="rgb",
    cx=None,
    cy=None,
    parts=None,
) -> Path:
    """Write a per-track embedding run (store + embedding.yaml) into a recording.

    What ``make_embedding_run`` returns. Uses the real ``EmbeddingPartWriter`` (no storage
    mocks) so the schema matches the pipeline. ``emb``/``frame_idx``/``t_ns`` are the track's
    trajectory; the metadata fields drive clip-query's comparability check (Tenet 9 lineage).

    ``parts`` splits the rows over video parts, in order: ``[6, 6]`` writes the first six rows to
    ``part_001.zarr`` and the next six to ``part_002.zarr``. ``frame_idx`` is then what each part
    stores, **part-local** (§5), as the stage writes it. Default: every row in ``part_001``.
    """
    from detail.storage.zarr_io import EmbeddingPartWriter

    run_dir = Path(recording_dir) / "embeddings" / embed_run_id
    lengths = [len(emb)] if parts is None else [int(n) for n in parts]
    assert sum(lengths) == len(emb), f"parts {lengths} do not add up to {len(emb)} rows"
    first = 0
    for number, length in enumerate(lengths, start=1):
        rows = range(first, first + length)
        first += length
        store = run_dir / f"track_{track_id}" / f"part_{number:03d}.zarr"
        store.parent.mkdir(parents=True, exist_ok=True)
        with EmbeddingPartWriter(store, output_dim=int(emb.shape[1])) as writer:
            for i in rows:
                writer.add_row(int(frame_idx[i]), int(t_ns[i]), 0, emb[i])
        # §14's kinematics sibling, only when a test asks for it: a run made before
        # kinematics existed has none, and that is the state most fixtures should be in.
        if cx is not None:
            from detail.storage.zarr_io import (
                KinematicsPartWriter,
                kinematics_part_path,
            )

            with KinematicsPartWriter(kinematics_part_path(store)) as kin:
                for i in rows:
                    kin.add_row(
                        int(frame_idx[i]),
                        int(t_ns[i]),
                        float(cx[i]),
                        float(cy[i] if cy is not None else cx[i]),
                        0.05,
                        [0.0, 0.0, 1.0, 1.0],
                    )
    meta = {
        "schema_version": 1,
        "id": embed_run_id,
        "created_at": "2026-01-01T00:00:00Z",
        "tracking_run_id": tracking_run_id,
        "model_id": model_id,
        "model_variant": model_variant,
        "constructor": constructor,
        "readout": readout,
        "input_size": input_size,
        "output_dim": int(emb.shape[1]),
        "track_ids": [track_id],
        "canvas_scope": canvas_scope,
        "bg_color_scope": bg_color_scope,
        # A run made today; pass None for one from before the channel-order fix (read "bgr").
        **({} if channel_order is None else {"channel_order": channel_order}),
        "renderer_spec": renderer_spec or {"steps": ["silhouette"]},
        "status": status,
    }
    (run_dir / "embedding.yaml").write_text(
        yaml.dump(meta, sort_keys=False), encoding="utf-8"
    )
    _index_in_recording(Path(recording_dir), embed_run_id, meta, track_id)
    return run_dir


def _index_in_recording(recording_dir, embed_run_id, meta, track_id) -> None:
    """Index the run in ``recording.yaml`` when the recording has one, as the stage does.

    Built by the module's own ``run_record``, so an analysis that resolves its upstream from
    ``recording.yaml`` (§6.5) sees what a real run leaves. A second call for the same run id
    (another track) indexes it once, with every track.
    """
    from detail.executor.runners import PROCESSING_MODULES, import_runner_modules
    from detail.storage import recording as recording_store

    if not (recording_dir / "recording.yaml").is_file():
        return
    if "embedding" not in PROCESSING_MODULES:
        from detail.api._runner_modules import PIPELINE_RUNNERS

        import_runner_modules(PIPELINE_RUNNERS)
    data = recording_store.read_recording(recording_dir)
    runs = data.setdefault("embedding_runs", [])
    existing = next((r for r in runs if r.get("id") == embed_run_id), None)
    if existing is not None:
        existing["track_ids"] = sorted(
            set(existing.get("track_ids") or []) | {track_id}
        )
    else:
        record = PROCESSING_MODULES["embedding"].run_record(
            embed_run_id, {**meta, "track_ids": [track_id]}, meta["status"]
        )
        record["created_at"] = meta["created_at"]
        runs.append(record)
    recording_store.write_recording(recording_dir, data)
