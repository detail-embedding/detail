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
    extra=None,
) -> Path:
    """Write a per-track embedding run (store + embedding.yaml) into a recording.

    What ``make_embedding_run`` returns. Uses the real ``EmbeddingPartWriter`` (no storage
    mocks) so the schema matches the pipeline. ``emb``/``frame_idx``/``t_ns`` are the track's
    trajectory; the metadata fields drive clip-query's comparability check (Tenet 9 lineage).
    ``extra`` adds metadata fields a test needs that have no parameter here (alignment, §26).
    """
    from detail.storage.zarr_io import EmbeddingPartWriter

    run_dir = Path(recording_dir) / "embeddings" / embed_run_id
    store = run_dir / f"track_{track_id}" / "part_001.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    with EmbeddingPartWriter(store, output_dim=int(emb.shape[1])) as writer:
        for i in range(len(emb)):
            writer.add_row(int(frame_idx[i]), int(t_ns[i]), 0, emb[i])
    # §14's kinematics sibling, only when a test asks for it: a run made before
    # kinematics existed has none, and that is the state most fixtures should be in.
    if cx is not None:
        from detail.storage.zarr_io import KinematicsPartWriter, kinematics_part_path

        with KinematicsPartWriter(kinematics_part_path(store)) as kin:
            for i in range(len(emb)):
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
        **(extra or {}),
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


def write_keypoints_run(
    recording_dir,
    rows,
    *,
    skeleton,
    run_id="kp_01",
    tracking_run_id="track_01",
    track_id=0,
    min_score=None,
    status="complete",
    model_id="dinov3_match",
    despike_window_s=0.1,
    despike_half_window=3,
) -> Path:
    """Write a keypoints run (§26) into a recording as the stage leaves one.

    What ``make_keypoints_run`` returns. ``rows`` is ``[(frame_idx, xy (K, 2), score (K,))]``
    for one track, ``t_ns = frame_idx * 1000`` as in ``processed_recording``; ``skeleton`` a
    ``Skeleton``. Uses the real ``KeypointsPartWriter`` and indexes the run in
    ``recording.yaml`` through the module's own ``run_record``, so a consumer resolving it by
    id or by settings sees what a real run leaves.
    """
    import numpy as np

    from detail.executor.runners import PROCESSING_MODULES, import_runner_modules
    from detail.storage import recording as recording_store
    from detail.storage.zarr_io import KeypointsPartWriter

    run_dir = Path(recording_dir) / "keypoints" / run_id
    with KeypointsPartWriter(
        run_dir / f"track_{track_id}" / "part_001.zarr", n_keypoints=len(skeleton)
    ) as writer:
        for frame, xy, score in rows:
            writer.add_row(
                int(frame),
                int(frame) * 1000,
                np.asarray(xy, np.float32),
                np.asarray(score, np.float32),
            )
    (run_dir / "keypoints.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": run_id,
                "tracking_run_id": tracking_run_id,
                "track_ids": [track_id],
                "model_id": model_id,
                "skeleton": skeleton.to_dict(),
                "visibility": {"min_score": min_score},
                "status": status,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if "keypoints" not in PROCESSING_MODULES:
        from detail.api._runner_modules import PIPELINE_RUNNERS

        import_runner_modules(PIPELINE_RUNNERS)
    record = PROCESSING_MODULES["keypoints"].run_record(
        run_id,
        {
            "tracking_run_id": tracking_run_id,
            "model_id": model_id,
            "model_variant": None,
            "every_n_frames": 1,
            "despike": True,
            "despike_window_s": despike_window_s,
            "despike_half_window": despike_half_window,
            "despike_k": 3.0,
            "min_score": min_score,
            "track_ids": [track_id],
        },
        status,
    )
    data = recording_store.read_recording(recording_dir)
    data.setdefault("keypoint_runs", []).append(record)
    recording_store.write_recording(recording_dir, data)
    return run_dir


#: The labeled positions of the audit builder's four keypoints, normalized full-frame.
AUDIT_LABELS = {
    "nose": (0.5, 0.3),
    "left_ear": (0.45, 0.35),
    "right_ear": (0.55, 0.35),
    "tail_base": (0.5, 0.7),
}
AUDIT_BBOX = (0.2, 0.2, 0.8, 0.8)


def write_keypoint_track_run(
    recording_dir,
    *,
    names=tuple(AUDIT_LABELS),
    n_frames=10,
    offsets=None,
    scores=None,
    example_marks=None,
    model_id="fake_kp",
    detect_run_id="kd_01",
    track_run_id="kt_01",
    tracking_run_id="track_01",
    processing_run_id=None,
    aliases=None,
) -> None:
    """A complete ``keypoint_detect`` + ``keypoint_track`` pair in one recording, its rows one
    per frame ``i`` at ``t_ns = i * 1000`` for track 0 in ``part_001``, each point at its
    ``AUDIT_LABELS`` position moved by ``offsets[name]`` (normalized).

    The recording's detection, tracking and keypoint records carry **every identity field**,
    so two recordings built alike share one run spec (§6.5) — the unit an audit scores.
    ``example_marks`` is the frame records the detect run copied from its example set.
    ``scores`` maps a name to its point score (default 0.9); ``aliases`` places a run point
    named differently from the labels (``{"snout": "nose"}``) at that label's position.
    """
    import shutil

    import numpy as np

    from detail.api import run_spec
    from detail.core.skeleton import Skeleton
    from detail.storage import keypoint_runs
    from detail.storage import recording as recording_store
    from detail.storage.keypoint_zarr import KeypointTrackPartWriter

    recording_dir = Path(recording_dir)
    offsets = offsets or {}
    scores = scores or {}
    aliases = aliases or {}
    skeleton = Skeleton(names=tuple(names))
    keypoint_runs.write_detect_yaml(
        keypoint_runs.detect_run_dir(recording_dir, detect_run_id),
        {
            "schema_version": 1,
            "id": detect_run_id,
            "created_at": "2026-01-01T00:00:00Z",
            "detection_run_id": "det_01",
            "model_id": model_id,
            "example_set": (
                None
                if example_marks is None
                else {"id": "ex_09", "name": "examples", "marks": example_marks}
            ),
            "skeleton": skeleton.to_dict(),
            "skeleton_digest": skeleton.digest(),
            "n_candidates": 1,
            "every_n_frames": 1,
        },
    )
    track_dir = keypoint_runs.track_run_dir(recording_dir, track_run_id)
    keypoint_runs.write_track_yaml(
        track_dir,
        {
            "schema_version": 1,
            "id": track_run_id,
            "created_at": "2026-01-01T00:00:00Z",
            "detect_run_id": detect_run_id,
            "tracking_run_id": tracking_run_id,
            "match": {"min_inside": 0.5, "mask_grow": 0.05},
            "method": "despike_smooth",
            "config": {},
            "skeleton": skeleton.to_dict(),
            "skeleton_digest": skeleton.digest(),
        },
    )
    points = np.zeros((n_frames, len(names), 2), np.float32)
    point_scores = np.zeros((n_frames, len(names)), np.float32)
    for j, name in enumerate(names):
        x, y = AUDIT_LABELS[aliases.get(name, name)]
        dx, dy = offsets.get(name, (0.0, 0.0))
        points[:, j] = (x + dx, y + dy)
        point_scores[:, j] = scores.get(name, 0.9)
    part = keypoint_runs.track_part_path(track_dir, 0, "part_001")
    part.parent.mkdir(parents=True, exist_ok=True)
    if part.exists():  # the writer resumes an existing part; a second call rewrites it
        shutil.rmtree(part)
    with KeypointTrackPartWriter(part, tuple(names)) as writer:
        writer.add_rows(
            np.arange(n_frames),
            np.arange(n_frames, dtype=np.int64) * 1000,
            np.arange(n_frames, dtype=np.int32),
            points,
            point_scores,
            np.zeros(points.shape[:2], np.int8),
        )
    data = recording_store.read_recording(recording_dir)
    data["detection_runs"] = [
        {
            "id": "det_01",
            "status": "complete",
            "created_at": "2026-01-01T00:00:00Z",
            "model_id": "fake_detector",
            "model_variant": None,
            "weights_source": None,
            "conf_threshold": 0.5,
            "every_n_frames": 1,
            "teacher_window_s": None,
            "prompt_fingerprint": "p",
            "prompt_classes": ["mouse"],
            "class_conf_thresholds_fingerprint": None,
        }
    ]
    data["tracking_runs"] = [
        {
            "id": tracking_run_id,
            "status": "complete",
            "created_at": "2026-01-01T00:00:00Z",
            "detection_run_id": "det_01",
            "tracker_id": "iou_tracker",
            "arena_ids": [],
            "class_name": None,
            "hyperparams": {},
            "every_n_frames": 1,
        }
    ]
    data["keypoint_detect_runs"] = [
        {
            "id": detect_run_id,
            "status": "complete",
            "created_at": "2026-01-01T00:00:00Z",
            "detection_run_id": "det_01",
            "min_pose_confidence": None,
            "model_id": model_id,
            "model_variant": None,
            "weights_source": None,
            "backbone_id": None,
            "backbone_variant": None,
            "skeleton_digest": skeleton.digest(),
            "example_set_digest": None,
            "n_candidates": 1,
            "every_n_frames": 1,
            "crop_px": None,
            "decode": None,
            "side_rule": "off",
            "matching": None,
        }
    ]
    track_record = {
        "id": track_run_id,
        "status": "complete",
        "created_at": "2026-01-01T00:00:00Z",
        "detect_run_id": detect_run_id,
        "tracking_run_id": tracking_run_id,
        "tracking_identity": run_spec.spec_of(
            data, "tracking", data["tracking_runs"][0], recording_dir
        ),
        "match": {"min_inside": 0.5, "mask_grow": 0.05},
        "method": "despike_smooth",
        "config": {},
    }
    if processing_run_id is not None:
        track_record["processing_run_id"] = processing_run_id
    data["keypoint_track_runs"] = [track_record]
    recording_store.write_recording(recording_dir, data)


def audit_frame(recording_id, i, *, marks=None, **kw):
    """Audit frame ``i`` of a recording built by :func:`write_keypoint_track_run`, labeled
    at ``AUDIT_LABELS`` unless ``marks`` says otherwise."""
    from detail.core.examples import ExampleFrame, ExampleMark

    if marks is None:
        marks = [ExampleMark(n, x=x, y=y) for n, (x, y) in AUDIT_LABELS.items()]
    return ExampleFrame(
        image=f"frames/{recording_id}_{i:03d}.png",
        recording_id=recording_id,
        tracking_run_id="track_01",
        track_id=0,
        part_id="part_001",
        frame_idx=i,
        t_ns=i * 1000,
        bbox=AUDIT_BBOX,
        marks=marks,
        round=1,
        **kw,
    )


def add_audit_frames(experiment, set_id, frames) -> None:
    """Append ``frames`` to an audit set as if drawn and labeled (no images on disk)."""
    from detail.storage import example_sets as example_store

    current = example_store.read_set(experiment.dir, set_id)
    current.frames.extend(frames)
    example_store.write_set(experiment.dir, current)
