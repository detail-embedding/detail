"""Motion inside the mask, per embedded frame — the ``motion/`` sibling of an embedding part (§14).

An outline can hold still while what is inside it moves: a paw stroking the face, a tongue on a
paw, a scratch. Every outline measure reads those frames as still. This measures the movement
**inside** the mask: the mean absolute change in brightness between a frame and the one before it,
over the mask's pixels — overall, at the body's two **ends** (the outer thirds along its long axis)
and in its **middle**. Grooming the face lights up an end; scratching a flank, the middle.

**One sequential pass per video part, every track at once.** The change needs the frame before each
embedded frame, so every frame is decoded, forward only (Tenet 11): the cost is a full decode of the
video, which is why this is a backfill a person asks for, with progress, and not part of opening a
run. Masks come from the detection store, decoded per needed frame, at the detection's mask
resolution; the frame is reduced to that size rather than the mask enlarged.

**Its own sibling, beside ``kinematics/``.** The kinematics backfill rewrites its sidecar whole; a
pass that costs a video decode must not be thrown away by one that does not. Same rows, same order
as the embedding part, NaN where a row has no mask or no previous frame.

``cv2`` for the decode and the resize (``cv2.VideoCapture``, never FFmpeg — Tenet 15).
"""

from pathlib import Path

import numpy as np

from detail.core import atomic
from detail.core import yaml_load
from detail.core.utils import decode_mask
from detail.executor import rundir
from detail.storage import recording as recording_store
from detail.storage.zarr_io import (
    DetectionPartReader,
    PartZarrReader,
    PartZarrWriter,
    resolve_assignment_rows,
)

MOTION_DIRNAME = "motion"
#: The sidecar's columns: the rows of the embedding part, then three changes in brightness, each
#: a fraction of full scale (0-1).
MOTION_COLUMNS = (
    ("frame_idx", "int32", (10000,), ()),
    ("t_ns", "int64", (10000,), ()),
    ("motion_inside", "float32", (10000,), ()),
    ("motion_ends", "float32", (10000,), ()),
    ("motion_middle", "float32", (10000,), ()),
)


class MotionPartWriter(PartZarrWriter):
    """One track's motion rows for one part, parallel to its embedding part."""

    def __init__(self, path: Path | str, buffer_size: int = 10000) -> None:
        super().__init__(path, MOTION_COLUMNS, buffer_size)

    def add_row(
        self, frame_idx: int, t_ns: int, inside: float, ends: float, middle: float
    ) -> None:
        self._add_row(
            {
                "frame_idx": frame_idx,
                "t_ns": t_ns,
                "motion_inside": inside,
                "motion_ends": ends,
                "motion_middle": middle,
            }
        )


def motion_part_path(embedding_part: Path | str) -> Path:
    """``track_0/part_001.zarr`` → ``track_0/motion/part_001.zarr``."""
    path = Path(embedding_part)
    return path.parent / MOTION_DIRNAME / path.name


def mask_motion(diff: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """``(inside, ends, middle)``: the mean of ``diff`` (a frame's change, 0-1, at the mask's
    size) over the mask, over its outer thirds along the long axis, and over its middle third.
    NaN for an empty mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        nan = float("nan")
        return nan, nan, nan
    values = diff[ys, xs].astype(np.float64)
    coords = np.stack([xs, ys]).astype(np.float64)
    centred = coords - coords.mean(axis=1, keepdims=True)
    _w, vectors = np.linalg.eigh(np.cov(coords))
    along = vectors[:, 1] @ centred
    lo, hi = np.quantile(along, [1 / 3, 2 / 3])
    ends = (along < lo) | (along > hi)
    middle = ~ends
    return (
        float(values.mean()),
        float(values[ends].mean()) if ends.any() else float("nan"),
        float(values[middle].mean()) if middle.any() else float("nan"),
    )


def missing_parts(recording_dir: Path | str, embed_run_id: str) -> list[Path]:
    """Embedding parts with no motion sibling, in walk order."""
    run_dir = Path(recording_dir) / "embeddings" / embed_run_id
    return [
        part
        for track_dir in sorted(run_dir.glob("track_*"))
        for part in sorted(track_dir.glob("part_*.zarr"))
        if not motion_part_path(part).exists()
    ]


def backfill_run(
    recording_dir: Path | str,
    embed_run_id: str,
    *,
    progress=None,
    overwrite: bool = False,
) -> int:
    """Write the motion sibling for every track of one embedding run; returns parts written.

    ``progress(stage, fraction, detail)``: a full decode of every video part, so minutes to an
    hour on a long recording — a silent long op is indistinguishable from a hang.

    Raises:
        ValueError: An unknown embedding run.
    """
    import cv2

    recording_dir = Path(recording_dir)
    run_dir = recording_dir / "embeddings" / embed_run_id
    if not (run_dir / "embedding.yaml").exists():
        raise ValueError(
            f"no embedding run {embed_run_id!r} in recording {recording_dir.name!r}"
        )
    meta = yaml_load.safe_load(atomic.read_text(run_dir / "embedding.yaml"))
    tracking_dir = recording_dir / "tracking" / meta["tracking_run_id"]
    track_meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    detection_dir = recording_dir / "detections" / track_meta["detection_run_id"]
    class_name = track_meta.get("class_name")
    mask_hw = tuple(rundir.read_envelope(detection_dir)["mask_resolution"])

    data = recording_store.read_recording(recording_dir)
    files = recording_store.video_files(data)
    by_stem = {recording_store.part_stem(f): f for f in files}
    parts_by_stem: dict[str, list[Path]] = {}
    for track_dir in sorted(run_dir.glob("track_*")):
        for part in sorted(track_dir.glob("part_*.zarr")):
            if overwrite or not motion_part_path(part).exists():
                parts_by_stem.setdefault(part.stem, []).append(part)
    stems = [s for s in by_stem if s in parts_by_stem]
    written = 0
    for index, stem in enumerate(stems):
        video = recording_store.resolve_part_video_path(recording_dir, by_stem[stem])

        def report(fraction: float, index=index, stem=stem) -> None:
            if progress is not None:
                progress("motion", (index + fraction) / max(len(stems), 1), f"{stem}")

        report(0.0)
        written += _backfill_part(
            parts_by_stem[stem],
            video,
            detection_dir,
            tracking_dir,
            class_name,
            mask_hw,
            cv2,
            progress=report,
        )
    if progress is not None:
        progress("motion", 1.0, f"{written} part(s)")
    return written


def _backfill_part(
    embedding_parts: list[Path],
    video: Path,
    detection_dir: Path,
    tracking_dir: Path,
    class_name,
    mask_hw,
    cv2,
    *,
    progress=None,
) -> int:
    """Every track's part of one video part, from one forward decode."""
    stem = embedding_parts[0].name
    det_reader = DetectionPartReader(detection_dir / stem)
    assign_reader = PartZarrReader(tracking_dir / "assignments" / stem)
    raw_rows = resolve_assignment_rows(det_reader, assign_reader, class_name)
    n = len(raw_rows)
    det = {k: np.asarray(v)[raw_rows] for k, v in det_reader.read().items()}
    object_id = np.asarray(assign_reader.read()["object_id"][:n]).astype(np.int64)
    frame_col = np.asarray(det["frame_idx"]).astype(np.int64)

    # (track, frame) -> the detection rows that make its mask; and each part's rows.
    wanted: dict[int, dict[int, list[int]]] = {}
    rows_of: dict[Path, dict] = {}
    for part in embedding_parts:
        track_id = int(part.parent.name.split("_")[-1])
        embedded = PartZarrReader(part).read()
        rows_of[part] = embedded
        mine = np.flatnonzero(object_id == track_id)
        by_frame: dict[int, list[int]] = {}
        for row in mine.tolist():
            by_frame.setdefault(int(frame_col[row]), []).append(row)
        for frame in np.asarray(embedded["frame_idx"]).tolist():
            if int(frame) in by_frame:
                wanted.setdefault(int(frame), {})[track_id] = by_frame[int(frame)]
    results: dict[tuple[int, int], tuple[float, float, float]] = {}
    last = max(wanted) if wanted else -1
    cap = cv2.VideoCapture(str(video))
    height, width = int(mask_hw[0]), int(mask_hw[1])
    try:
        previous = None
        frame_index = 0
        while frame_index <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index + 1 in wanted or frame_index in wanted:
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grey = cv2.resize(grey, (width, height), interpolation=cv2.INTER_AREA)
                grey = grey.astype(np.float32) / 255.0
                if frame_index in wanted and previous is not None:
                    diff = np.abs(grey - previous)
                    for track_id, rows in wanted[frame_index].items():
                        mask = None
                        for row in rows:
                            decoded = decode_mask(
                                str(det["mask_rle"][row]), (height, width)
                            )
                            mask = decoded if mask is None else (mask | decoded)
                        results[(track_id, frame_index)] = mask_motion(diff, mask)
                previous = grey
            else:
                previous = None
            if progress is not None and frame_index % 2000 == 0 and last > 0:
                progress(frame_index / last)
            frame_index += 1
    finally:
        cap.release()

    nan = (float("nan"),) * 3
    for part, embedded in rows_of.items():
        track_id = int(part.parent.name.split("_")[-1])
        out = motion_part_path(part)
        if out.exists():
            import shutil

            shutil.rmtree(out)
        with MotionPartWriter(out) as writer:
            for frame, t_ns in zip(
                np.asarray(embedded["frame_idx"]).tolist(),
                np.asarray(embedded["t_ns"]).tolist(),
            ):
                writer.add_row(
                    int(frame), int(t_ns), *results.get((track_id, int(frame)), nan)
                )
    return len(rows_of)
