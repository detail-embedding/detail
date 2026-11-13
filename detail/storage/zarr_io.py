"""Per-part zarr writers and readers for the pipeline stages (Sections 9, 13, 14).

All per-part stores share one buffer-flush mechanism (`PartZarrWriter`): rows
accumulate in memory and are appended via resize + tail-assignment (the zarr v3
idiom, Tenet 12). On reopen, every column is truncated to its common length,
repairing a write torn across columns by a crash — the standard recovery step
for any per-part writer.

**A reader applies the same policy, read-only.** A crash between one column's
resize and the next's leaves the *store itself* with columns of unequal length
until a writer reopens it — a read-only `PartZarrReader` has no write access to
repair that, but every consumer that cross-references two columns by row index
(a tracker masking by class, a frame-grouping walk, anything indexing `bbox`
alongside `class_name`) silently assumes they agree. `n_rows` and `read()` both
resolve to `_common_length` — the shorter column, the same number a reopened
writer would truncate to — so a torn store reads as a consistent, if slightly
shorter, one rather than one column's row N meaning a different detection than
another's.

The detection writer is the only **detection-indexed** one (many rows per frame),
so it overrides the flush trigger to fire only on frame boundaries — on-disk data
then always ends at a complete frame, the unit detection resume restarts from
(Section 7). The assignment, embedding, and tracks-summary writers are plain
row-indexed: flush whenever the buffer fills.

**A wrong `class_name` reads as an empty run, not an error.** `resolve_assignment_rows`
filters the detection reader's rows by `class_name` before pairing them with an
assignment; passing a `class_name` the tracker never actually filtered by (e.g. a
run record's own display label, rather than the value the tracker wrote to its
*own* `track.yaml`) can silently filter every row away, leaving a real, fully-tracked
run looking like it tracked nothing. The guard below catches this the moment
`assignment.n_rows` exceeds the filtered row count — a tracker can never have
assigned more rows than it saw, so that inequality means the `class_name` passed in
is not the filter this run actually used, not a crash-torn store (those make the
assignment *shorter*, never longer, which is the already-handled case this guard
does not touch).
"""

from pathlib import Path

import numpy as np

from detail.storage import zarr_store

SCHEMA_VERSION = 1

# Column spec: (name, dtype, chunks, extra_shape). extra_shape is the trailing
# per-row shape (e.g. (4,) for bbox); () for scalar columns.
_DETECTION_COLUMNS = (
    ("frame_idx", "int32", (10000,), ()),
    ("t_ns", "int64", (10000,), ()),
    ("detection_idx", "int32", (10000,), ()),
    ("class_name", "str", (10000,), ()),
    ("confidence", "float32", (10000,), ()),
    ("bbox", "float32", (10000, 4), (4,)),
    ("mask_rle", "str", (1000,), ()),  # ASCII COCO-RLE counts (Section 9)
)
_ASSIGNMENT_COLUMNS = (("object_id", "int32", (10000,), ()),)
_TRACKS_COLUMNS = (
    ("track_id", "int32", (10000,), ()),
    ("class_name", "str", (10000,), ()),
    ("first_frame", "int32", (10000,), ()),
    ("last_frame", "int32", (10000,), ()),
    ("n_detections", "int32", (10000,), ()),
    ("n_gaps", "int32", (10000,), ()),
    ("mean_conf", "float32", (10000,), ()),
)
# Per-frame, per-subject geometry, written beside a track's embedding parts (Section 14).
# Everything normalized, matching the detection zarr's `bbox`: a value is then comparable across
# recordings of different resolutions and across a downsampled copy of one. Absolute pixels are
# recoverable by multiplying by the frame size; the reverse is not.
#: The mask's outline (``analysis/mask_shape.FIELDS``), after the position columns. A sidecar
#: written before these existed simply lacks them, which readers treat as "not measured".
KINEMATICS_SHAPE_COLUMNS = (
    "mask_top_norm",
    "mask_bottom_norm",
    "floor_width_norm",
    "major_axis_norm",
    "minor_axis_norm",
    "orientation_deg",
    "solidity",
    "circularity",
    "protrusions",
    "protrusion_depth_norm",
)
_KINEMATICS_COLUMNS = (
    ("frame_idx", "int32", (10000,), ()),
    ("t_ns", "int64", (10000,), ()),
    ("cx_norm", "float32", (10000,), ()),
    ("cy_norm", "float32", (10000,), ()),
    ("mask_area_norm", "float32", (10000,), ()),
    ("bbox_norm", "float32", (10000, 4), (4,)),
    *((name, "float32", (10000,), ()) for name in KINEMATICS_SHAPE_COLUMNS),
)


def _to_buffer(dtype: str, values: list) -> np.ndarray:
    if dtype == "str":
        return np.array(values, dtype=object)
    return np.asarray(values, dtype=dtype)


def _append(arr, buf: np.ndarray) -> None:
    """Grow a zarr array along axis 0 and write buf into the new tail.

    The transient network-share replace zarr's internal writes are exposed to is ridden
    out by the store this array was opened on (``storage/zarr_store.py``), not here. This
    module used to wrap each of its own calls instead, and that habit is what left
    ``storage/frame_view.py`` — the next module to write a zarr — covered on one call in
    five.
    """
    n = arr.shape[0]
    arr.resize((n + len(buf), *arr.shape[1:]))
    arr[n:] = buf


def _common_length(root, names) -> int:
    """The row count every column in ``names`` agrees on: the shortest one.

    Equal when the store is intact; the shorter of two lengths when a crash tore a
    write between one column's resize and the next's (see module docstring). A
    writer reopening the store truncates every column to this; a reader with no
    write access reads only up to it.
    """
    return min((root[name].shape[0] for name in names), default=0)


class PartZarrWriter:
    """Buffer-flush writer over a fixed set of columns, with crash-safe resume.

    Subclasses pass their column specs and expose a typed ``add_*`` method that
    builds a values dict and calls ``_add_row``. Detection-indexed subclasses
    pass a ``boundary_key`` so flushes land on unit boundaries.
    """

    def __init__(
        self, path: Path | str, columns: tuple, buffer_size: int = 10000
    ) -> None:
        self._path = Path(path)
        self._columns = columns
        self._names = tuple(c[0] for c in columns)
        self._buffer_size = buffer_size
        self._buffers: dict[str, list] = {name: [] for name in self._names}
        self._n_buffered = 0
        self._last_boundary = None
        self._root = zarr_store.open_group(self._path, mode="a")
        if self._names[0] not in self._root:
            self._init_columns(columns)
        else:
            self._verify_existing_columns()
            self._truncate_to_consistent_length()

    @property
    def rows_written(self) -> int:
        """Rows persisted on disk (excludes the in-memory buffer)."""
        return self._root[self._names[0]].shape[0]

    def flush(self) -> None:
        """Persist the buffered rows to disk."""
        if self._n_buffered == 0:
            return
        for name, dtype, _chunks, _extra in self._columns:
            _append(self._root[name], _to_buffer(dtype, self._buffers[name]))
            self._buffers[name].clear()
        self._n_buffered = 0

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "PartZarrWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _add_row(self, values: dict, boundary_key=None) -> None:
        """Buffer one row. Flush when full; on a detection-indexed writer, only
        when the buffer is full *and* a new boundary (frame) begins."""
        if self._n_buffered >= self._buffer_size and (
            boundary_key is None or boundary_key != self._last_boundary
        ):
            self.flush()
        for name in self._names:
            self._buffers[name].append(values[name])
        self._n_buffered += 1
        self._last_boundary = boundary_key

    def _init_columns(self, columns: tuple) -> None:
        """Write the schema attr and create each column array on a fresh store."""
        self._root.attrs["schema_version"] = SCHEMA_VERSION
        for name, dtype, chunks, extra in columns:
            self._root.create_array(name, shape=(0, *extra), chunks=chunks, dtype=dtype)

    def _verify_existing_columns(self) -> None:
        for name, _dtype, _chunks, extra in self._columns:
            if name not in self._root:
                raise ValueError(
                    f"existing store {self._path} is missing column {name!r}"
                )
            trailing = tuple(self._root[name].shape[1:])
            if trailing != extra:
                raise ValueError(
                    f"column {name!r} in {self._path} has trailing shape {trailing}, "
                    f"expected {extra}"
                )

    def _truncate_to_consistent_length(self) -> None:
        n = _common_length(self._root, self._names)
        for name in self._names:
            arr = self._root[name]
            if arr.shape[0] != n:
                arr.resize((n, *arr.shape[1:]))


class DetectionPartWriter(PartZarrWriter):
    """Detection part zarr (Section 9). Detection-indexed: flushes on frame boundaries."""

    def __init__(self, path: Path | str, buffer_size: int = 10000) -> None:
        super().__init__(path, _DETECTION_COLUMNS, buffer_size)

    def add_row(
        self,
        frame_idx: int,
        t_ns: int,
        detection_idx: int,
        class_name: str,
        confidence: float,
        bbox: tuple[float, float, float, float],
        mask_rle: str,
    ) -> None:
        """Buffer one detection row. Rows must arrive in non-decreasing frame order."""
        self._add_row(
            {
                "frame_idx": frame_idx,
                "t_ns": t_ns,
                "detection_idx": detection_idx,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "mask_rle": mask_rle,
            },
            boundary_key=frame_idx,
        )


class AssignmentPartWriter(PartZarrWriter):
    """Tracking assignment zarr (Section 13): object_id parallel to the detection rows."""

    def __init__(self, path: Path | str, buffer_size: int = 10000) -> None:
        super().__init__(path, _ASSIGNMENT_COLUMNS, buffer_size)

    def add_object_id(self, object_id: int) -> None:
        """Append the track id for the next detection row (-1 = unassigned)."""
        self._add_row({"object_id": object_id})


#: §14 *Egocentric alignment*: present on an aligned run's parts only — the heading the
#: image was turned by (interpolated or not), where it came from (0 own row, 1
#: interpolated) and whether the image was mirrored.
ALIGNMENT_COLUMNS = (
    ("heading_deg", "float32", (10000,), ()),
    ("heading_source", "int8", (10000,), ()),
    ("align_mirrored", "bool", (10000,), ()),
)

#: §14 *Pose in the embedding*: present on a run with pose only — how many keypoints each
#: frame had to work with. 0 = the keypoint track run gave this frame nothing (it is embedded
#: as the mask alone, or with a zero pose block).
POSE_COLUMNS = (("pose_points", "uint8", (10000,), ()),)


class EmbeddingPartWriter(PartZarrWriter):
    """Embedding part zarr for one track (Section 14). One row per embedded frame.

    ``aligned`` adds the three alignment columns (§14), present only when the run turned
    its images — an unaligned run's part is byte-for-byte what it was before alignment
    existed, so the two never read as the same recipe. ``posed`` adds ``pose_points`` the
    same way, for a run with pose.
    """

    def __init__(
        self,
        path: Path | str,
        output_dim: int,
        buffer_size: int = 10000,
        *,
        aligned: bool = False,
        posed: bool = False,
    ) -> None:
        columns = (
            ("frame_idx", "int32", (10000,), ()),
            ("t_ns", "int64", (10000,), ()),
            ("det_row", "int32", (10000,), ()),
            ("embedding", "float32", (1000, output_dim), (output_dim,)),
        )
        if aligned:
            columns = columns + ALIGNMENT_COLUMNS
        if posed:
            columns = columns + POSE_COLUMNS
        self.aligned = bool(aligned)
        self.posed = bool(posed)
        super().__init__(path, columns, buffer_size)

    def add_row(
        self,
        frame_idx: int,
        t_ns: int,
        det_row: int,
        embedding: np.ndarray,
        *,
        heading_deg: float | None = None,
        heading_source: int | None = None,
        align_mirrored: bool | None = None,
        pose_points: int | None = None,
    ) -> None:
        """Buffer one (frame, embedding) row. embedding is a (D,) vector; the three
        alignment values are required on an aligned part and refused on an unaligned one.
        """
        row = {
            "frame_idx": frame_idx,
            "t_ns": t_ns,
            "det_row": det_row,
            "embedding": embedding,
        }
        given = heading_deg is not None or heading_source is not None
        if self.aligned:
            if heading_deg is None or heading_source is None or align_mirrored is None:
                raise ValueError(
                    "an aligned embedding part needs every alignment value"
                )
            row.update(
                {
                    "heading_deg": float(heading_deg),
                    "heading_source": int(heading_source),
                    "align_mirrored": bool(align_mirrored),
                }
            )
        elif given or align_mirrored:
            raise ValueError("an unaligned embedding part takes no alignment values")
        if self.posed:
            if pose_points is None:
                raise ValueError(
                    "an embedding part with pose needs pose_points per row"
                )
            row["pose_points"] = int(pose_points)
        elif pose_points is not None:
            raise ValueError("an embedding part without pose takes no pose_points")
        self._add_row(row)


def kinematics_part_path(embedding_part_path: Path | str) -> Path:
    """The kinematics sibling for an embedding part (§14).

    ``track_0/part_001.zarr`` → ``track_0/kinematics/part_001.zarr``. One function so the writer,
    the reader and the backfill cannot disagree about where it lives — two halves of a pair that
    must be found together, resolved in one place.
    """
    path = Path(embedding_part_path)
    return path.parent / "kinematics" / path.name


class KinematicsPartWriter(PartZarrWriter):
    """Kinematics part zarr for one track (Section 14) — one row per embedded frame.

    Parallel to :class:`EmbeddingPartWriter`: same rows, same order, same part. It is a sibling
    file rather than four more columns on the embedding part because asking where the animal was
    must not read the embedding column — 589 MB against 7 ms for the small columns beside it
    (`api/run_preview.py::load_track_timeline`).

    A row with no describable geometry writes NaN rather than zero. Zero is a *position* — the
    top-left corner — so it would be indistinguishable from a real one on the very screen this
    exists to make trustworthy.
    """

    def __init__(self, path: Path | str, buffer_size: int = 10000) -> None:
        super().__init__(path, _KINEMATICS_COLUMNS, buffer_size)

    def add_row(
        self,
        frame_idx: int,
        t_ns: int,
        cx_norm: float,
        cy_norm: float,
        mask_area_norm: float,
        bbox_norm: np.ndarray,
        shape=None,
    ) -> None:
        """Buffer one frame's geometry. ``bbox_norm`` is normalized xyxy, (4,); ``shape`` is
        the outline (``mask_shape.FrameShape``), NaN for every field when ``None``."""
        outline = (
            shape._asdict()
            if shape is not None
            else dict.fromkeys(KINEMATICS_SHAPE_COLUMNS, float("nan"))
        )
        self._add_row(
            {
                "frame_idx": frame_idx,
                "t_ns": t_ns,
                "cx_norm": cx_norm,
                "cy_norm": cy_norm,
                "mask_area_norm": mask_area_norm,
                "bbox_norm": bbox_norm,
                **{name: outline[name] for name in KINEMATICS_SHAPE_COLUMNS},
            }
        )


class KeypointsPartWriter(PartZarrWriter):
    """Keypoints part zarr for one track (§26) — one row per frame the track has a subject.

    ``xy`` is normalized full-frame x, y per keypoint, NaN where the method placed nothing —
    undefined, never zero, as in kinematics; ``score`` is the method's own confidence, NaN with
    it. ``despiked`` marks the positions the temporal filter replaced; rows are written with it
    False and :func:`write_despiked` rewrites both columns once the part is complete, because the
    filter needs the frames after a row as well as before it. ``carried`` marks a row the change
    gate did not run the method on: the last computed frame's points, moved with the subject's
    mask. The column order of ``K`` is the run's skeleton, recorded once in ``keypoints.yaml``.
    A part written before ``carried`` existed has no such column; every row in it was computed.
    """

    def __init__(
        self, path: Path | str, n_keypoints: int, buffer_size: int = 10000
    ) -> None:
        k = int(n_keypoints)
        columns = (
            ("frame_idx", "int32", (10000,), ()),
            ("t_ns", "int64", (10000,), ()),
            ("xy", "float32", (10000, k, 2), (k, 2)),
            ("score", "float32", (10000, k), (k,)),
            ("despiked", "bool", (10000, k), (k,)),
            ("carried", "bool", (10000,), ()),
        )
        super().__init__(path, columns, buffer_size)
        self._k = k

    def add_row(
        self,
        frame_idx: int,
        t_ns: int,
        xy: np.ndarray,
        score: np.ndarray,
        *,
        carried: bool = False,
    ) -> None:
        """Buffer one frame. ``xy`` is ``(K, 2)``, ``score`` ``(K,)``."""
        self._add_row(
            {
                "frame_idx": frame_idx,
                "t_ns": t_ns,
                "xy": xy,
                "score": score,
                "despiked": np.zeros(self._k, dtype=bool),
                "carried": bool(carried),
            }
        )


class TracksSummaryWriter(PartZarrWriter):
    """Track summary zarr (Section 13): one row per track, written once at run end."""

    def __init__(self, path: Path | str, buffer_size: int = 10000) -> None:
        super().__init__(path, _TRACKS_COLUMNS, buffer_size)

    def add_track(
        self,
        track_id: int,
        class_name: str,
        first_frame: int,
        last_frame: int,
        n_detections: int,
        n_gaps: int,
        mean_conf: float,
    ) -> None:
        self._add_row(
            {
                "track_id": track_id,
                "class_name": class_name,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "n_detections": n_detections,
                "n_gaps": n_gaps,
                "mean_conf": mean_conf,
            }
        )


class PartZarrReader:
    """Read access to any per-part zarr store."""

    def __init__(self, path: Path | str) -> None:
        self._root = zarr_store.open_group(path, mode="r")

    @property
    def schema_version(self) -> int:
        return self._root.attrs["schema_version"]

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self._root.array_keys())

    @property
    def n_rows(self) -> int:
        """Rows every column agrees on (``_common_length``) — the shorter length if
        a crash tore the store's columns apart mid-flush, the same number a writer
        reopening it would truncate to (module docstring)."""
        names = list(self._root.array_keys())
        return _common_length(self._root, names) if names else 0

    def column(self, name: str):
        """The underlying zarr array for a column (lazy) — reads exactly this
        column, so a store whose columns disagree in length (module docstring)
        is not repaired here; a caller cross-referencing this against another
        column should slice to ``n_rows`` first."""
        return self._root[name]

    def read(self) -> dict[str, np.ndarray]:
        """Load every column fully into memory, truncated to ``n_rows`` — every
        column agreeing on which row is which (module docstring), which every
        caller that indexes across columns (a tracker's class/arena filter, frame
        grouping) already assumes."""
        names = tuple(self._root.array_keys())
        n = _common_length(self._root, names)
        return {name: self._root[name][:n] for name in names}


def resolve_assignment_rows(
    reader: "DetectionPartReader",
    assignment: "PartZarrReader | None",
    class_name: str | None,
) -> np.ndarray:
    """The detection reader's own row index behind each assignment row, in order.

    A tracking run's ``object_id`` assignment is parallel to the detection rows the
    tracker actually kept when it ran (Section 13) — every row when the run had no
    class filter, only one class's rows otherwise (``anchor_tracker`` and its
    siblings build ``object_ids`` from ``trackers._common.filter_by_class``'d data,
    tests/pipeline/test_tracking.py). Assignment row ``i`` is then detection reader
    row ``resolve_assignment_rows(...)[i]`` — *not* row ``i`` of the reader — so
    pairing them by raw position is silently wrong the moment a class filter
    narrowed what the tracker saw (the class this function exists to close: a
    tracking preview reading "the first N rows" of the unfiltered detection reader
    reads as the tracker dropping a whole trailing span of an otherwise
    fully-detected video).

    Also bounded by the shorter of the reader's own row count and the assignment's
    — the same guard against a crash-torn store that ``PartZarrReader.n_rows``
    applies within one reader (module docstring), extended across the pair.
    """
    n_reader = reader.n_rows
    raw_rows = np.arange(n_reader)
    if class_name and n_reader:
        raw_class = np.asarray(reader.column("class_name")[:n_reader])
        raw_rows = np.flatnonzero(raw_class == class_name)
    if assignment is not None:
        if assignment.n_rows > len(raw_rows):
            raise ValueError(
                f"assignment has {assignment.n_rows} rows but only {len(raw_rows)} "
                f"detection rows matched class_name={class_name!r}; the tracker "
                "could not have produced more assigned rows than it saw, so this "
                "class_name is not the filter this run's tracker actually applied"
            )
        raw_rows = raw_rows[: min(len(raw_rows), assignment.n_rows)]
    return raw_rows


#: ``object_ids_on_detection_rows``' value for a detection row the tracker never saw: one its
#: class filter left out, or one past a crash-torn assignment's end. Below ``-1`` (seen and left
#: unassigned), so ``>= 0`` and ``== track_id`` read as before and only "unassigned" must ask.
UNSEEN = -2


def object_ids_on_detection_rows(
    reader: "DetectionPartReader",
    assignment: "PartZarrReader",
    class_name: str | None,
) -> np.ndarray:
    """The tracking assignment laid onto the detection part's **own** rows: element ``i`` is
    the track of detection row ``i`` — ``-1`` when the tracker saw it and assigned none,
    :data:`UNSEEN` when it never saw it.

    For a caller that indexes the whole detection part by position. Reading ``object_id`` raw
    and indexing the part with it is right only while the run had no class filter; with one
    the two arrays are not parallel (:func:`resolve_assignment_rows`) and every row after the
    first filtered-out one is paired with the wrong detection.
    """
    raw_rows = resolve_assignment_rows(reader, assignment, class_name)
    out = np.full(reader.n_rows, UNSEEN, dtype=np.int64)
    out[raw_rows] = np.asarray(
        assignment.column("object_id")[: len(raw_rows)], dtype=np.int64
    )
    return out


class DetectionPartReader(PartZarrReader):
    """Reader for the detection part zarr, with the frame-based resume primitive."""

    @property
    def last_frame_idx(self) -> int | None:
        """The highest frame_idx with a row, or None if empty.

        Rows are written in frame order, so the last row holds the max frame_idx.
        Frames with zero detections leave no rows; the authoritative
        frames-processed count lives in job.json (Section 9), not here.
        """
        if self.n_rows == 0:
            return None
        return int(self._root["frame_idx"][-1])
