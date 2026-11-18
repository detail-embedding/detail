"""Keypoint run records (§26): the run directories, ``detect.yaml`` / ``track.yaml`` and the
readers.

``detect.yaml`` is written when the run **starts** (so a stopped or crashed run still says what
it was), ``resolved`` is added once the model has loaded, and ``coverage`` + ``timing`` are
appended when it ends. It carries no ``status``: the executor's is the only status (``job.json``,
``recording.yaml``). Every write is tmp + ``atomic.replace`` — the GUI polls the file while the
executor subprocess rewrites it.

Layout (§26 "Run directory structure")::

    recordings/<recording>/keypoint_detect/<kd_NN>/
        job.json · detect.yaml · part_<id>.zarr          ← one row per pose instance, no subject
    recordings/<recording>/keypoint_track/<kt_NN>/
        job.json · track.yaml · assignments/part_<id>.zarr · track_<id>/part_<id>.zarr

A run in the per-track detect layout from before 2026-10-21 (``track_<id>/`` under the detect
run, a ``tracking_run_id`` in its yaml) is refused with ``LegacyKeypointLayoutError`` — "re-run
keypoint detection" — and never read through a second path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from detail.core import atomic
from detail.core.skeleton import Skeleton
from detail.storage.keypoint_zarr import (
    KeypointAssignmentPartReader,
    KeypointDetectPartReader,
    KeypointTrackPartReader,
    LegacyKeypointLayoutError,
    legacy_message,
)

__all__ = ["LegacyKeypointLayoutError"]

SCHEMA_VERSION = 1
DETECT_DIRNAME = "keypoint_detect"
DETECT_YAML = "detect.yaml"


def detect_run_dir(recording_dir: Path | str, run_id: str) -> Path:
    return Path(recording_dir) / DETECT_DIRNAME / run_id


def part_path(run_dir: Path | str, part_id: str) -> Path:
    """A detect run's store for one video part: ``<run>/<part_id>.zarr``."""
    return Path(run_dir) / f"{part_id}.zarr"


def track_dir(run_dir: Path | str, track_id: int) -> Path:
    """A track run's folder for one subject: ``<run>/track_<id>/``."""
    return Path(run_dir) / f"track_{int(track_id)}"


def track_part_path(run_dir: Path | str, track_id: int, part_id: str) -> Path:
    """A track run's store for one (subject, part)."""
    return track_dir(run_dir, track_id) / f"{part_id}.zarr"


def assignment_path(run_dir: Path | str, part_id: str) -> Path:
    """A track run's assignment store for one part, parallel to the detect part's rows."""
    return Path(run_dir) / "assignments" / f"{part_id}.zarr"


def write_yaml(path: Path | str, data: dict) -> None:
    """``data`` as YAML, ``schema_version`` first (Tenet 7), tmp + atomic replace."""
    if next(iter(data)) != "schema_version":
        raise ValueError("schema_version must be the first key of a run yaml")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    atomic.replace(tmp, path)


def detect_record(job, skeleton_digest: str) -> dict:
    """The ``detect.yaml`` body a run writes at start (§26), in the order the spec shows."""
    return {
        "schema_version": SCHEMA_VERSION,
        "id": job.detect_run_id,
        "created_at": job.created_at,
        "detection_run_id": job.detection_run_id,
        "min_pose_confidence": (
            None if job.min_pose_confidence is None else float(job.min_pose_confidence)
        ),
        "model_id": job.model_id,
        "model_variant": job.model_variant,
        # where a student's weights live: "trained" | "imported"; None for a registered method
        "weights_source": getattr(job, "weights_source", None),
        "backbone": job.backbone,
        "example_set": job.example_set,
        "skeleton": dict(job.skeleton),
        "skeleton_digest": skeleton_digest,
        # the library entry the skeleton came from; the dependents search reads its id
        "skeleton_ref": (
            None
            if getattr(job, "skeleton_ref", None) is None
            else dict(job.skeleton_ref)
        ),
        "n_candidates": int(job.n_candidates),
        "every_n_frames": int(job.every_n_frames),
        "crop_px": job.crop_px,
        "decode": job.decode,
        "side_rule": job.side_rule,
        "matching": job.matching,
        "resolved": {},
        "batch_size": int(job.batch_size),
        "compute": job.compute.to_dict(),
    }


def write_detect_yaml(run_dir: Path | str, data: dict) -> None:
    write_yaml(Path(run_dir) / DETECT_YAML, data)


def read_detect_yaml(run_dir: Path | str) -> dict:
    path = Path(run_dir) / DETECT_YAML
    if not path.is_file():
        raise FileNotFoundError(f"no {DETECT_YAML} in {run_dir}")
    return yaml.safe_load(atomic.read_text(path))


def update_detect_yaml(run_dir: Path | str, **fields) -> dict:
    """Read, set ``fields``, write back atomically; returns the new record."""
    data = read_detect_yaml(run_dir)
    data.update(fields)
    write_detect_yaml(run_dir, data)
    return data


def _instance_columns(k: int, n_candidates: int) -> dict[str, np.ndarray]:
    """The core columns with zero rows: what ``at`` returns for a frame with no instance."""
    out = {
        "row": np.zeros(0, np.int64),
        "frame_idx": np.zeros(0, np.int32),
        "t_ns": np.zeros(0, np.int64),
        "bbox": np.zeros((0, 4), np.float32),
        "confidence": np.zeros(0, np.float32),
        "detection_row": np.zeros(0, np.int32),
        "points": np.zeros((0, k, 2), np.float32),
        "point_scores": np.zeros((0, k), np.float32),
    }
    if n_candidates > 1:
        out["candidates"] = np.zeros((0, k, n_candidates, 2), np.float32)
        out["candidate_scores"] = np.zeros((0, k, n_candidates), np.float32)
    return out


class KeypointDetectRun:
    """Read access to one detect run: its record, the skeleton the rows follow, each part's
    instances, and the instances on a frame (what the viewer draws).

    Raises:
        LegacyKeypointLayoutError: The run is in the per-track layout (before 2026-10-21).
    """

    def __init__(self, recording_dir: Path | str, run_id: str) -> None:
        self.recording_dir = Path(recording_dir)
        self.run_id = run_id
        self.dir = detect_run_dir(recording_dir, run_id)
        self.meta = read_detect_yaml(self.dir)
        if "detection_run_id" not in self.meta or any(self.dir.glob("track_*")):
            raise LegacyKeypointLayoutError(
                legacy_message(f"keypoint detect run {run_id!r}")
            )
        self._frame_idx: dict[str, np.ndarray] = {}

    @property
    def skeleton(self) -> Skeleton:
        return Skeleton.from_dict(self.meta["skeleton"])

    @property
    def detection_run_id(self) -> str:
        return str(self.meta["detection_run_id"])

    @property
    def n_candidates(self) -> int:
        return int(self.meta.get("n_candidates", 1))

    def part_ids(self) -> list[str]:
        """The parts with a store, by part id (``part_001``), sorted. A part where nothing was
        placed has no store; its coverage still records it."""
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("part_*.zarr"))

    def reader(self, part_id: str) -> KeypointDetectPartReader:
        reader = KeypointDetectPartReader(part_path(self.dir, part_id))
        reader.check_skeleton(tuple(self.meta["skeleton"]["names"]))
        return reader

    def rows(self, part_id: str) -> dict[str, np.ndarray]:
        """Every column of one part, the extras keyed ``extras/<name>``; a row's index is
        the ``instance_row`` a track run refers to it by."""
        return self.reader(part_id).read_all()

    def coverage(self) -> dict:
        """``{part_id: {frames, frames_with_instance, instances}}`` as the run recorded it;
        empty while it is still running."""
        return dict(self.meta.get("coverage") or {})

    def at(self, part_id: str, frame_idx: int) -> dict[str, np.ndarray]:
        """Every instance on one frame of a part — the core columns plus ``row`` (each
        instance's row index in the part) — with zero rows when there is none."""
        k = len(self.meta["skeleton"]["names"])
        path = part_path(self.dir, part_id)
        if not path.is_dir():
            return _instance_columns(k, self.n_candidates)
        reader = self.reader(part_id)
        if part_id not in self._frame_idx:
            self._frame_idx[part_id] = np.asarray(reader.column("frame_idx")[:])
        frames = self._frame_idx[part_id]
        lo = int(np.searchsorted(frames, frame_idx, side="left"))
        hi = int(np.searchsorted(frames, frame_idx, side="right"))
        out = _instance_columns(k, reader.n_candidates)
        if hi == lo:
            return out
        out["row"] = np.arange(lo, hi, dtype=np.int64)
        for name in out:
            if name != "row":
                out[name] = np.asarray(reader.column(name)[lo:hi])
        return out


# --------------------------------------------------------------------------- #
# Track runs (§26): keypoint_track/<kt_NN>/track.yaml + track_<id>/<part>.zarr
# --------------------------------------------------------------------------- #
TRACK_DIRNAME = "keypoint_track"
TRACK_YAML = "track.yaml"


def track_run_dir(recording_dir: Path | str, run_id: str) -> Path:
    return Path(recording_dir) / TRACK_DIRNAME / run_id


def track_record(job, skeleton_digest: str) -> dict:
    """The ``track.yaml`` body written at run start (§26), ``schema_version`` first."""
    return {
        "schema_version": SCHEMA_VERSION,
        "id": job.track_run_id,
        "created_at": job.created_at,
        "detect_run_id": job.detect_run_id,
        "tracking_run_id": job.tracking_run_id,
        "match": dict(job.match),
        "method": job.method,
        "config": dict(job.config),
        "skeleton": dict(job.skeleton),
        "skeleton_digest": skeleton_digest,
        "skeleton_ref": (
            None
            if getattr(job, "skeleton_ref", None) is None
            else dict(job.skeleton_ref)
        ),
    }


def write_track_yaml(run_dir: Path | str, data: dict) -> None:
    write_yaml(Path(run_dir) / TRACK_YAML, data)


def read_track_yaml(run_dir: Path | str) -> dict:
    path = Path(run_dir) / TRACK_YAML
    if not path.is_file():
        raise FileNotFoundError(f"no {TRACK_YAML} in {run_dir}")
    return yaml.safe_load(atomic.read_text(path))


def update_track_yaml(run_dir: Path | str, **fields) -> dict:
    data = read_track_yaml(run_dir)
    data.update(fields)
    write_track_yaml(run_dir, data)
    return data


class KeypointTrackRun:
    """Read access to one track run: the assignment per part, and per subject the final
    points (``instance_row`` joins a row back to the detect run's instance).

    Raises:
        LegacyKeypointLayoutError: The run was made from a per-track detect run.
    """

    def __init__(self, recording_dir: Path | str, run_id: str) -> None:
        self.recording_dir = Path(recording_dir)
        self.run_id = run_id
        self.dir = track_run_dir(recording_dir, run_id)
        self.meta = read_track_yaml(self.dir)
        if "tracking_run_id" not in self.meta:
            raise LegacyKeypointLayoutError(
                legacy_message(
                    f"keypoint track run {run_id!r} (from detect run "
                    f"{self.meta.get('detect_run_id')!r})"
                )
                + ", then keypoint track"
            )
        self._frame_idx: dict[tuple[int, str], np.ndarray] = {}

    @property
    def skeleton(self) -> Skeleton:
        return Skeleton.from_dict(self.meta["skeleton"])

    @property
    def track_ids(self) -> list[int]:
        folder = self.dir
        if not folder.is_dir():
            return []
        return sorted(
            int(p.name.split("_", 1)[1]) for p in folder.glob("track_*") if p.is_dir()
        )

    def part_ids(self, track_id: int) -> list[str]:
        folder = track_dir(self.dir, track_id)
        if not folder.is_dir():
            return []
        return sorted(p.stem for p in folder.glob("part_*.zarr"))

    def reader(self, track_id: int, part_id: str) -> KeypointTrackPartReader:
        reader = KeypointTrackPartReader(track_part_path(self.dir, track_id, part_id))
        reader.check_skeleton(tuple(self.meta["skeleton"]["names"]))
        return reader

    def rows(self, track_id: int, part_id: str) -> dict[str, np.ndarray]:
        return self.reader(track_id, part_id).read()

    def assignments(self, part_id: str) -> dict[str, np.ndarray]:
        """``{subject_id, inside}`` parallel to the detect part's rows (§26 *Assignment*)."""
        return KeypointAssignmentPartReader(
            assignment_path(self.dir, part_id)
        ).read_all()

    def coverage(self) -> dict:
        return dict(self.meta.get("coverage") or {})

    def at(
        self, track_id: int, part_id: str, frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        key = (int(track_id), part_id)
        path = track_part_path(self.dir, track_id, part_id)
        if not path.is_dir():
            return None
        reader = KeypointTrackPartReader(path)
        if key not in self._frame_idx:
            self._frame_idx[key] = np.asarray(reader.column("frame_idx")[:])
        frames = self._frame_idx[key]
        i = int(np.searchsorted(frames, frame_idx))
        if i >= len(frames) or int(frames[i]) != int(frame_idx):
            return None
        return (
            np.asarray(reader.column("points")[i]),
            np.asarray(reader.column("point_scores")[i]),
        )
