"""Zarr stores for the keypoint stages (§26 "Run directory structure").

The **detect** part is instance-indexed with no subject, the shape a detection part has: one
row per pose instance, rows in frame order, none for a frame with no instance, any number for a
frame with several. Its writer flushes only on frame boundaries (``boundary_key=frame_idx``, as
``DetectionPartWriter.add_row``), so the store always ends on a whole frame, even after a stop
or a crash. Its attrs carry ``layout: "instances"``; a store without it is the per-track
layout from before 2026-10-21 and is refused (``LegacyKeypointLayoutError``), never read
through a second path.

The **track** run writes ``assignments/<part>.zarr`` parallel to the detect rows, then one
per-(subject, part) store in embedding's layout. Every writer follows the zarr v3 idiom (§9):
``zarr_store.open_group`` through ``PartZarrWriter``, ``schema_version`` first, buffer-flush,
resume from ``shape[0]``. Column names mirror ``FrameResult``. ``K`` and the column order are
the skeleton's, recorded in the run's yaml **and** as ``skeleton_names`` in each store's attrs,
so rows stay readable after a crash that never reached the yaml — a reader checks both and
fails on a mismatch. Method-only arrays (``FrameResult.extras``) live under ``extras/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from detail.core.dataclasses import FrameResult
from detail.storage.zarr_io import PartZarrReader, PartZarrWriter

ROW_CHUNK = 10000
EXTRAS_PREFIX = "extras/"
#: The detect part's ``layout`` attr since 2026-10-21; its absence marks the per-track layout.
INSTANCES_LAYOUT = "instances"


class LegacyKeypointLayoutError(ValueError):
    """A keypoint run written in the per-track layout (before 2026-10-21): refused, with the
    one thing to do about it."""


def legacy_message(what: str) -> str:
    return (
        f"{what} was written in the per-track layout used before 2026-10-21 and cannot be "
        "read; re-run keypoint detection on the detection run"
    )


def extras_spec(result: FrameResult) -> dict[str, tuple[str, tuple[int, ...]]]:
    """``{name: (dtype, trailing shape)}`` of a result's method-only arrays — what the writer
    needs to create their columns, read off the method's first result."""
    return {
        name: (str(arr.dtype), tuple(int(d) for d in np.shape(arr)[1:]))
        for name, arr in result.extras.items()
    }


def detect_columns(
    k: int, n_candidates: int, extras: dict[str, tuple[str, tuple[int, ...]]]
) -> tuple:
    """The detect part's columns (§26): ``candidates`` only when the method keeps more than
    one per point, and each extra under ``extras/``."""
    cols: list[tuple] = [
        ("frame_idx", "int32", (ROW_CHUNK,), ()),
        ("t_ns", "int64", (ROW_CHUNK,), ()),
        ("bbox", "float32", (ROW_CHUNK, 4), (4,)),
        ("confidence", "float32", (ROW_CHUNK,), ()),
        ("detection_row", "int32", (ROW_CHUNK,), ()),
        ("points", "float32", (ROW_CHUNK, k, 2), (k, 2)),
        ("point_scores", "float32", (ROW_CHUNK, k), (k,)),
    ]
    if n_candidates > 1:
        cols.append(
            (
                "candidates",
                "float32",
                (ROW_CHUNK, k, n_candidates, 2),
                (k, n_candidates, 2),
            )
        )
        cols.append(
            (
                "candidate_scores",
                "float32",
                (ROW_CHUNK, k, n_candidates),
                (k, n_candidates),
            )
        )
    for name, (dtype, trailing) in sorted(extras.items()):
        cols.append(
            (EXTRAS_PREFIX + name, dtype, (ROW_CHUNK, *trailing), tuple(trailing))
        )
    return tuple(cols)


def is_legacy_store(path: Path | str) -> bool:
    """A finished store written before the ``layout`` attr: skeleton recorded, no layout."""
    attrs = PartZarrReader(path)._root.attrs
    return "skeleton_names" in attrs and attrs.get("layout") != INSTANCES_LAYOUT


def _check_skeleton_attr(root, path, skeleton_names: tuple[str, ...]) -> None:
    if tuple(root.attrs["skeleton_names"]) != skeleton_names:
        raise ValueError(
            f"{path} holds rows for skeleton {list(root.attrs['skeleton_names'])}, not "
            f"{list(skeleton_names)}; a run never changes its skeleton"
        )


class KeypointDetectPartWriter(PartZarrWriter):
    """One part's pose instances: a row per instance, in frame order, no subject."""

    def __init__(
        self,
        path: Path | str,
        skeleton_names: tuple[str, ...],
        n_candidates: int,
        extras: dict[str, tuple[str, tuple[int, ...]]] | None = None,
        buffer_size: int = ROW_CHUNK,
    ) -> None:
        self.skeleton_names = tuple(str(n) for n in skeleton_names)
        self.n_candidates = int(n_candidates)
        self.extras = dict(extras or {})
        path = Path(path)
        if path.is_dir() and is_legacy_store(path):
            raise LegacyKeypointLayoutError(legacy_message(f"keypoint store {path}"))
        super().__init__(
            path,
            detect_columns(len(self.skeleton_names), self.n_candidates, self.extras),
            buffer_size,
        )
        attrs = self._root.attrs
        if "skeleton_names" not in attrs:
            attrs["layout"] = INSTANCES_LAYOUT
            attrs["skeleton_names"] = list(self.skeleton_names)
            attrs["n_candidates"] = self.n_candidates
        else:
            _check_skeleton_attr(self._root, path, self.skeleton_names)

    def add_instance(self, result: FrameResult, row: int) -> None:
        """Buffer ``result``'s ``row``-th instance. A flush lands only where a new frame
        begins, so the store never ends part-way through a frame (Tenet 12)."""
        values = {
            "frame_idx": int(result.frame_idx),
            "t_ns": int(result.t_ns),
            "bbox": (
                np.full(4, np.nan, np.float32)
                if result.bbox is None
                else np.asarray(result.bbox[row], dtype=np.float32)
            ),
            "confidence": (
                np.float32(np.nan)
                if result.confidence is None
                else np.float32(result.confidence[row])
            ),
            "detection_row": (
                -1 if result.detection_row is None else int(result.detection_row[row])
            ),
            "points": np.asarray(result.points[row], dtype=np.float32),
            "point_scores": np.asarray(result.point_scores[row], dtype=np.float32),
        }
        if self.n_candidates > 1:
            values["candidates"] = np.asarray(result.candidates[row], dtype=np.float32)
            values["candidate_scores"] = np.asarray(
                result.candidate_scores[row], dtype=np.float32
            )
        for name in self.extras:
            values[EXTRAS_PREFIX + name] = result.extras[name][row]
        self._add_row(values, boundary_key=int(result.frame_idx))


class KeypointDetectPartReader(PartZarrReader):
    """Reader for a detect part: the skeleton the rows follow, and every column including
    the method's ``extras/``. Refuses a store in the per-track layout."""

    def __init__(self, path: Path | str) -> None:
        super().__init__(path)
        if self._root.attrs.get("layout") != INSTANCES_LAYOUT:
            raise LegacyKeypointLayoutError(legacy_message(f"keypoint store {path}"))

    @property
    def skeleton_names(self) -> tuple[str, ...]:
        return tuple(self._root.attrs["skeleton_names"])

    @property
    def n_candidates(self) -> int:
        return int(self._root.attrs.get("n_candidates", 1))

    @property
    def extras_names(self) -> tuple[str, ...]:
        if "extras" not in self._root:
            return ()
        return tuple(sorted(self._root["extras"].array_keys()))

    def read_all(self) -> dict[str, np.ndarray]:
        """Every column, the extras keyed ``extras/<name>``."""
        out = self.read()
        for name in self.extras_names:
            out[EXTRAS_PREFIX + name] = self._root[EXTRAS_PREFIX + name][:]
        return out

    def check_skeleton(self, names: tuple[str, ...]) -> None:
        """A store whose attrs disagree with the yaml is unreadable, not quietly misread."""
        if self.skeleton_names != tuple(names):
            raise ValueError(
                f"detect rows follow skeleton {list(self.skeleton_names)} but the run's "
                f"yaml says {list(names)}; the run cannot be read"
            )


# --------------------------------------------------------------------------- #
# Track run: the assignment, parallel to the detect rows (§26 *Assignment*)
# --------------------------------------------------------------------------- #
ASSIGNMENT_COLUMNS = (
    ("subject_id", "int32", (ROW_CHUNK,), ()),
    ("inside", "float32", (ROW_CHUNK,), ()),
)


class KeypointAssignmentPartWriter(PartZarrWriter):
    """``assignments/<part>.zarr``: one row per detect instance, in the detect part's row
    order — the subject it was given (−1 = unassigned) and the inside share that won (NaN when
    unassigned)."""

    def __init__(self, path: Path | str, buffer_size: int = ROW_CHUNK) -> None:
        super().__init__(path, ASSIGNMENT_COLUMNS, buffer_size)

    def add(self, subject_id: int, inside: float) -> None:
        self._add_row({"subject_id": int(subject_id), "inside": np.float32(inside)})


class KeypointAssignmentPartReader(PartZarrReader):
    def read_all(self) -> dict[str, np.ndarray]:
        return self.read()


def track_columns(k: int) -> tuple:
    """The track part's columns (§26): the final point per keypoint and how it came to be."""
    return (
        ("frame_idx", "int32", (ROW_CHUNK,), ()),
        ("t_ns", "int64", (ROW_CHUNK,), ()),
        ("instance_row", "int32", (ROW_CHUNK,), ()),
        ("points", "float32", (ROW_CHUNK, k, 2), (k, 2)),
        ("point_scores", "float32", (ROW_CHUNK, k), (k,)),
        ("changed", "int8", (ROW_CHUNK, k), (k,)),
    )


class KeypointTrackPartWriter(PartZarrWriter):
    """One subject's final points in one part (§26): a row per frame the subject is tracked
    at the run's cadence, ``instance_row`` the detect row assigned there (−1 = none, NaN
    points), one point per keypoint, and how each came to be (``changed``: 0 as detected, 1
    another candidate chosen, 2 moved by a filter). Written whole, in one pass; no resume.
    """

    def __init__(
        self,
        path: Path | str,
        skeleton_names: tuple[str, ...],
        buffer_size: int = ROW_CHUNK,
    ) -> None:
        self.skeleton_names = tuple(str(n) for n in skeleton_names)
        super().__init__(path, track_columns(len(self.skeleton_names)), buffer_size)
        attrs = self._root.attrs
        if "skeleton_names" not in attrs:
            attrs["skeleton_names"] = list(self.skeleton_names)
        elif tuple(attrs["skeleton_names"]) != self.skeleton_names:
            raise ValueError(
                f"{path} holds rows for skeleton {list(attrs['skeleton_names'])}, not "
                f"{list(self.skeleton_names)}; a run never changes its skeleton"
            )

    def add_rows(
        self,
        frame_idx: np.ndarray,
        t_ns: np.ndarray,
        instance_row: np.ndarray,
        points: np.ndarray,
        point_scores: np.ndarray,
        changed: np.ndarray,
    ) -> None:
        """Buffer a part's rows at once — the method returns whole arrays."""
        for i in range(len(frame_idx)):
            self._add_row(
                {
                    "frame_idx": int(frame_idx[i]),
                    "t_ns": int(t_ns[i]),
                    "instance_row": int(instance_row[i]),
                    "points": np.asarray(points[i], dtype=np.float32),
                    "point_scores": np.asarray(point_scores[i], dtype=np.float32),
                    "changed": np.asarray(changed[i], dtype=np.int8),
                }
            )


class KeypointTrackPartReader(PartZarrReader):
    @property
    def skeleton_names(self) -> tuple[str, ...]:
        return tuple(self._root.attrs["skeleton_names"])

    def check_skeleton(self, names: tuple[str, ...]) -> None:
        if self.skeleton_names != tuple(names):
            raise ValueError(
                f"track rows follow skeleton {list(self.skeleton_names)} but the run's yaml "
                f"says {list(names)}; the run cannot be read"
            )
