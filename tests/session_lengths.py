"""An experiment of twelve recordings of very different lengths, for Time in session (§22).

The builder behind ``tests/conftest.py``'s ``session_lengths`` fixture: one real temporal-clustering
fit (``dtw_finch``) over recordings from 8 s to 120 min, so a range that is a *fraction of each
recording* is exercised where the fraction and the absolute time disagree by three orders of
magnitude. Every recording's timestamp map goes through the real writers, and the multi-part ones
through the real importer (``Recording.add_video_part``), so a later part's ``t_ns`` is whatever
the import path really writes rather than what a fixture assumes it writes.

The rows a recording is clustered on sit in **bursts** of ``BURST_ROWS`` consecutive frames placed
at chosen fractions of the recording (``0`` its first frame, ``1`` its last), so its clips land at
the very start, the very end, the middle, inside a later part, or only at one end. Between bursts
the video runs on unobserved, as a run that clustered part of a video does.

``Layout`` is what the tests compute their expected answers from: each recording's first and last
``t_ns`` read straight off the maps written here, never through the code under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The embed cadence the burst rows are written at: every frame of a 30 fps video.
FPS = 30.0
STEP_NS = int(round(1e9 / FPS))
#: Rows per burst: three bouts of thirty rows, ``_bouty_trajectory``'s shape, so segmentation has
#: transitions to find inside every burst.
BURST_ROWS = 90
BOUT_ROWS = 30


@dataclass(frozen=True)
class Spec:
    """One recording: its name, the seconds of each part, where its bursts sit, and its quirks."""

    name: str
    part_s: tuple[float, ...]
    bursts: tuple[float, ...]
    #: Imported through ``Recording.add_video_part`` from real (tiny) videos.
    imported: bool = False
    #: Frames dropped and a stall: the map's times are irregular.
    irregular: bool = False
    #: A user-supplied active map whose clock does not start at 0.
    user_start_ns: int | None = None


SPECS: tuple[Spec, ...] = (
    Spec("eight_seconds", (8,), (0.0, 1.0)),
    Spec("twenty_seconds", (20,), (0.0, 1.0)),
    Spec("forty_five_seconds", (45,), (0.0, 0.5, 1.0)),
    Spec("two_minutes_user_clock", (120,), (0.1, 0.45, 0.9), user_start_ns=7 * 10**9),
    Spec("three_minutes_dropped", (180,), (0.2, 0.6, 1.0), irregular=True),
    Spec("five_minutes_two_parts", (180, 120), (0.0, 0.3, 0.7, 1.0), imported=True),
    Spec("ten_minutes_three_parts", (240, 120, 240), (0.05, 0.5, 0.95), imported=True),
    Spec("fifteen_minutes_start_only", (900,), (0.0,)),
    Spec("thirty_minutes", (1800,), (0.0, 0.25, 0.5, 0.75, 1.0)),
    Spec("sixty_minutes", (3600,), (0.01, 0.5, 0.99)),
    Spec("ninety_minutes_end_only", (5400,), (1.0,)),
    Spec("two_hours", (7200,), (0.0, 0.495, 1.0)),
)


@dataclass
class Layout:
    """What the fixture built: ``{recording_id: (first_ns, last_ns)}`` read off the maps written,
    each recording's name, and the per-part maps for the tests that look inside them."""

    spans: dict[str, tuple[int, int]] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    part_maps: dict[str, list[np.ndarray]] = field(default_factory=dict)
    specs: dict[str, Spec] = field(default_factory=dict)


def _write_video(path: Path, n_frames: int) -> None:
    """A tiny constant-rate mp4 at ``FPS``: the only thing the importer needs is its packets."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (16, 16))
    frame = np.zeros((16, 16, 3), np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def _irregular_map(n_frames: int, rng) -> np.ndarray:
    """A 30 fps clock with ~8% of frames dropped, a jitter, and a 2 s stall in the middle."""
    t = np.arange(n_frames, dtype=np.int64) * STEP_NS
    t += rng.integers(-2_000_000, 2_000_000, size=n_frames)
    t[0] = 0
    t[n_frames // 2 :] += 2 * 10**9
    keep = rng.random(n_frames) > 0.08
    keep[0] = keep[-1] = True
    return np.sort(t[keep])


def _single_part(recording, spec: Spec, rng) -> np.ndarray:
    """Write a one-part recording's map and video manifest through the storage writers."""
    from detail.storage import recording as recording_store

    n = int(round(spec.part_s[0] * FPS))
    if spec.irregular:
        t_ns = _irregular_map(n, rng)
    else:
        t_ns = np.arange(n, dtype=np.int64) * STEP_NS
    kind = "generated"
    if spec.user_start_ns is not None:
        t_ns = t_ns + spec.user_start_ns
        kind = "user"
    rel = f"video/timestamps/{kind}/part_001.csv"
    (recording.dir / rel).parent.mkdir(parents=True, exist_ok=True)
    recording_store.write_timestamps_csv(recording.dir / rel, t_ns)
    data = recording_store.read_recording(recording.dir)
    data["video"] = {
        "storage": "default",
        "fps_container": FPS,
        "fps_actual": FPS,
        "fps_source": "container",
        "resolution": [16, 16],
        "total_duration_s": float(spec.part_s[0]),
        "files": [
            {
                "index": 0,
                "id": "part_001",
                "file": "video/part_001.mp4",
                "timestamps": {
                    "generated": rel if kind == "generated" else None,
                    "user": rel if kind == "user" else None,
                    "active": kind,
                },
            }
        ],
    }
    recording_store.write_recording(recording.dir, data)
    return t_ns


def _imported_parts(recording, spec: Spec, video_dir: Path) -> None:
    for i, seconds in enumerate(spec.part_s):
        path = video_dir / f"{spec.name}_part{i + 1}.mp4"
        _write_video(path, int(round(seconds * FPS)))
        recording.add_video_part(path)


def _bursts_in(maps: list[np.ndarray], bursts) -> list[tuple[int, int]]:
    """``(part, first frame)`` of each burst: at its fraction of the recording's frames, clamped
    so the whole burst fits inside one part (a burst at ``1.0`` ends on the last frame).
    """
    sizes = [len(m) for m in maps]
    total = sum(sizes)
    out = []
    for frac in bursts:
        g = int(round(frac * (total - BURST_ROWS)))
        part, offset = 0, 0
        while g >= offset + sizes[part]:
            offset += sizes[part]
            part += 1
        local = min(g - offset, sizes[part] - BURST_ROWS)
        out.append((part, local))
    return out


def _bouty(n_bouts: int, rng, anchors) -> np.ndarray:
    rows = []
    for i in range(n_bouts):
        kind = int(rng.integers(len(anchors)))
        ramp = np.linspace(0.0, 1.0, BOUT_ROWS)[:, None]
        path = anchors[kind, 0] * (1 - ramp) + anchors[kind, 1] * ramp
        rows.append(path + 0.05 * rng.normal(size=path.shape))
    return np.vstack(rows).astype(np.float32)


def _write_embeddings(recording_dir: Path, maps, bursts, rng, anchors, index) -> None:
    """Each part's embedding (and kinematics) store, rows only inside the bursts, frame_idx local
    to the part and ``t_ns`` off that part's active map — what the embedding stage writes.
    """
    import yaml

    from detail.storage.zarr_io import (
        EmbeddingPartWriter,
        KinematicsPartWriter,
        kinematics_part_path,
    )

    per_part: dict[int, list[int]] = {}
    for part, local in bursts:
        per_part.setdefault(part, []).extend(range(local, local + BURST_ROWS))
    run_dir = recording_dir / "embeddings" / "emb_01"
    dim = anchors.shape[2]
    for part in sorted(per_part):
        frames = np.array(sorted(set(per_part[part])), dtype=np.int64)
        emb = _bouty(len(frames) // BOUT_ROWS + 1, rng, anchors)[: len(frames)]
        store = run_dir / "track_0" / f"part_{part + 1:03d}.zarr"
        store.parent.mkdir(parents=True, exist_ok=True)
        t_ns = maps[part][frames]
        with EmbeddingPartWriter(store, output_dim=dim) as writer:
            for i, f in enumerate(frames):
                writer.add_row(int(f), int(t_ns[i]), 0, emb[i])
        with KinematicsPartWriter(kinematics_part_path(store)) as kin:
            for i, f in enumerate(frames):
                x = float(f) / len(maps[part])
                kin.add_row(int(f), int(t_ns[i]), x, x, 0.05, [0.0, 0.0, 1.0, 1.0])
    meta = {
        "schema_version": 1,
        "id": "emb_01",
        "created_at": "2026-01-01T00:00:00Z",
        "tracking_run_id": "track_01",
        "model_id": "dinov3_vit",
        "model_variant": "vit_large",
        "constructor": None,
        "readout": None,
        "input_size": None,
        "output_dim": dim,
        "track_ids": [0],
        "canvas_scope": "per_recording",
        "bg_color_scope": "shared",
        "channel_order": "rgb",
        "renderer_spec": {"steps": ["silhouette"]},
        "status": "complete",
    }
    (run_dir / "embedding.yaml").write_text(
        yaml.dump(meta, sort_keys=False), encoding="utf-8"
    )
    index(recording_dir, "emb_01", meta, 0)


def build(experiment_dir: Path, video_dir: Path, experiments) -> Layout:
    """Fill ``experiment_dir`` (an empty experiment) with :data:`SPECS`; return the layout.

    ``experiments`` is ``tests/experiments.py`` as the conftest loaded it (by path), whose
    ``_index_in_recording`` indexes each embedding run in its recording as the stage does.
    """
    from detail.api.experiment import Experiment
    from detail.storage import recording as recording_store

    rng = np.random.default_rng(7)
    anchors = rng.normal(size=(3, 2, 8))
    experiment = Experiment(experiment_dir)
    layout = Layout()
    for spec in SPECS:
        recording = experiment.add_recording(spec.name)
        if spec.imported:
            _imported_parts(recording, spec, video_dir)
        else:
            _single_part(recording, spec, rng)
        data = recording_store.read_recording(recording.dir)
        maps = [
            recording_store.TimestampMap.from_csv(
                recording_store.active_timestamp_csv(recording.dir, data, i)
            ).values
            for i in range(len(data["video"]["files"]))
        ]
        layout.part_maps[recording.id] = maps
        layout.spans[recording.id] = (int(maps[0][0]), int(maps[-1][-1]))
        layout.names[recording.id] = spec.name
        layout.specs[recording.id] = spec
        _write_embeddings(
            recording.dir,
            maps,
            _bursts_in(maps, spec.bursts),
            rng,
            anchors,
            experiments._index_in_recording,
        )
    return layout
