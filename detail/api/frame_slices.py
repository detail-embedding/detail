"""Assemble the signals a descriptor reads — the bridge from stores to `analysis.descriptors`.

A descriptor is pure over a `FrameSlice`; something has to build one. That is here rather than in
`analysis/` because it opens zarr and resolves recordings, and `analysis/descriptors.py` is
deliberately numpy-only so the pipeline can import it.

**Two loaders, one shape.** The geometry a descriptor wants is captured twice in this codebase:
during the embed pass (a per-frame sidecar, cheap to read) and in the detection store (boxes, and
masks as RLE). `from_tracking` reads the detection store, which is what exists on `main` today.
When the kinematics sidecar lands, a second loader fills the same `FrameSlice` from it — the same
descriptors, the same values, less work. That is why the loader is a seam rather than a step
inside the descriptor code: a cache should be addable without anything above it changing.

**The sidecar is the first choice; boxes are the fallback.** `from_kinematics` fills the slice
from the embed pass's sidecar — mask centroid, mask area and the union box, at the embedding's
cadence — so the four mass-centre descriptors have their inputs. `from_tracking` reads the
detection store's boxes only (a mask centroid and area would need an RLE decode per frame, the
cost the sidecar exists to avoid), so there those descriptors report `NaN` and a surface shows
them as unavailable. `track_slice` picks: the sidecar when the embedding run wrote one, the
boxes otherwise — a run embedded before the sidecar existed still answers what a box can.
"""

import dataclasses
from pathlib import Path

import numpy as np

from detail.analysis.descriptors import FrameSlice
from detail.storage import recording as recording_store
from detail.storage.zarr_io import (
    KINEMATICS_SHAPE_COLUMNS,
    PartZarrReader,
    kinematics_part_path,
)


def from_tracking(
    recording_dir: Path | str,
    *,
    tracking_run_id: str,
    track_id: int,
    embed_run_id: str | None = None,
) -> FrameSlice:
    """One track's whole trajectory, as a `FrameSlice` in **pixel** units.

    Pixels rather than the store's normalized boxes, matching `dynamics.DynamicsInputs`: a
    normalized speed is anisotropic — x and y scale differently on a non-square frame — so it is
    neither comparable across recordings nor to a mask-pixel measure. The rescale happens here,
    once, so no descriptor has to know about it.

    ``embed_run_id`` adds the embeddings, which `pose_step` needs. Without it that descriptor
    reports `NaN` like any other missing input; the geometry is unaffected.

    Raises:
        ValueError: the tracking run is not in this recording's record.
    """
    from detail.api.run_preview import TrackingPreview

    recording_dir = Path(recording_dir)
    data = recording_store.read_recording(recording_dir)
    width, height = _resolution(data)
    preview = TrackingPreview(recording_dir, tracking_run_id)

    frames: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    times: list[np.ndarray] = []
    offset = 0
    for part in preview._parts:
        reader, assignment = part["reader"], part["assignment"]
        n_rows = part["n_rows"]
        stamps = np.asarray(part["timestamps"], dtype=np.int64)
        if reader is not None and assignment is not None and n_rows:
            # part["frame_idx"]/part["raw_rows"] are already resolved by _build_parts:
            # this run's own class filter (§13), if it had one, plus the shorter-wins
            # bound against a crash-torn store (storage/zarr_io.py). object_id is
            # bounded the same way before a mask built from it indexes bbox — reading
            # bbox as a raw [:n_rows] slice would silently pair the wrong rows the
            # moment a class filter made row i of the assignment something other than
            # the reader's own row i.
            object_id = np.asarray(assignment.column("object_id")[:n_rows])
            rows = np.flatnonzero(object_id == int(track_id))
            if len(rows):
                # Part-local frame indices are made global, so a multi-part recording is one
                # timeline: a step between the last row of one part and the first of the next
                # is otherwise a jump backwards, and reads as an enormous speed.
                local = np.asarray(part["frame_idx"])[rows]
                frames.append(local + offset)
                boxes.append(np.asarray(reader.column("bbox")[part["raw_rows"]])[rows])
                # The part's own PTS map is already recording-global (Tenet 8; the import
                # writes `local + offset_ns`), so it is used as stored: adding the earlier
                # parts' elapsed time again put every row after part 1 that much too late.
                times.append(stamps[np.clip(local, 0, len(stamps) - 1)])
        offset += part["n_frames"]

    if not frames:
        return FrameSlice(
            frame_idx=np.empty(0, dtype=np.int64), t_ns=np.empty(0, dtype=np.int64)
        )

    frame_idx = np.concatenate(frames)
    bbox = np.concatenate(boxes).astype(np.float64)
    t_ns = np.concatenate(times)
    order = np.argsort(frame_idx, kind="stable")
    frame_idx, bbox, t_ns = frame_idx[order], bbox[order], t_ns[order]
    bbox *= np.array([width, height, width, height], dtype=np.float64)

    # Every frame gets its real time from the recording's own map, so the slice is sorted in
    # time whether or not it was embedded. It used to take `t_ns` from the embedding join and
    # leave `0` where a frame had none — which put those rows *before* the recording started,
    # made `duration_s` wrong for any bout containing one, and quietly broke the assumption
    # `between` now relies on. The join below still supplies the embeddings; it no longer
    # supplies the clock.
    embeddings = _embeddings_for(recording_dir, embed_run_id, int(track_id), frame_idx)
    return FrameSlice(
        frame_idx=frame_idx,
        t_ns=t_ns,
        stride=int(preview._parts[0].get("every_n_frames") or 1),
        bbox=bbox,
        embeddings=embeddings,
    )


def from_kinematics(
    recording_dir: Path | str,
    *,
    embed_run_id: str,
    track_id: int,
    every_n_frames: int = 1,
) -> FrameSlice | None:
    """One animal's embedded frames as a `FrameSlice` from the §14 kinematics sidecar, in
    **pixel** units — or ``None`` when the embedding run wrote no sidecar for this track.

    Centroid, area and box come from the sidecar (fractions of the frame, rescaled here by ``(width, height)``), the embeddings
    from the same run, and ``stride`` is the **embedding** run's cadence, because the rows are
    the embedded frames. The sidecar's ``frame_idx`` restarts in every part, so it is put on the
    recording-wide scale; its ``t_ns`` already is. A NaN centroid (an empty mask, no detection)
    stays NaN — never zero, which would place the animal in the corner.
    """
    recording_dir = Path(recording_dir)
    data = recording_store.read_recording(recording_dir)
    width, height = _resolution(data)
    columns = ("frame_idx", "t_ns", "cx_norm", "cy_norm", "mask_area_norm", "bbox_norm")
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in columns}
    outline: dict[str, list[np.ndarray]] = {n: [] for n in KINEMATICS_SHAPE_COLUMNS}
    complete = True  # every part carries the outline; a mix would misalign rows
    for offset, part in _series_parts(recording_dir, embed_run_id, int(track_id)):
        sidecar = kinematics_part_path(part)
        if not sidecar.exists():
            continue
        reader = PartZarrReader(sidecar)
        if reader.n_rows == 0:
            continue
        for name in columns:
            values = np.asarray(reader.column(name)[: reader.n_rows])
            chunks[name].append(
                values.astype(np.int64) + offset if name == "frame_idx" else values
            )
        if KINEMATICS_SHAPE_COLUMNS[0] in reader.column_names:
            for name in KINEMATICS_SHAPE_COLUMNS:
                outline[name].append(np.asarray(reader.column(name)[: reader.n_rows]))
        else:
            complete = False
    if not chunks["frame_idx"]:
        return None
    joined = {name: np.concatenate(parts) for name, parts in chunks.items()}
    order = np.argsort(joined["frame_idx"], kind="stable")
    frame_idx = joined["frame_idx"][order]
    centroid = np.stack(
        [
            joined["cx_norm"][order].astype(np.float64) * width,
            joined["cy_norm"][order].astype(np.float64) * height,
        ],
        axis=1,
    )
    motion = _motion_rows(recording_dir, embed_run_id, int(track_id), order)
    shape = (
        _outline_px(
            {n: np.concatenate(v)[order] for n, v in outline.items()}, width, height
        )
        if complete and outline[KINEMATICS_SHAPE_COLUMNS[0]]
        else None
    )
    return FrameSlice(
        frame_idx=frame_idx,
        t_ns=joined["t_ns"][order].astype(np.int64),
        stride=max(1, int(every_n_frames or 1)),
        embeddings=_embeddings_for(
            recording_dir, embed_run_id, int(track_id), frame_idx
        ),
        centroid=centroid,
        area=joined["mask_area_norm"][order].astype(np.float64) * width * height,
        bbox=joined["bbox_norm"][order].astype(np.float64)
        * np.array([width, height, width, height]),
        shape=shape,
        body_length=_body_length(shape),
        motion=motion,
    )


def _motion_rows(recording_dir: Path, embed_run_id: str, track_id: int, order):
    """The motion sibling's columns on the kinematics rows (the same rows, the same order, so
    the same ``order``), or ``None`` unless every part has one."""
    from detail.pipeline.motion_backfill import motion_part_path

    columns = {"inside": [], "ends": [], "middle": []}
    for _offset, part in _series_parts(recording_dir, embed_run_id, track_id):
        if not kinematics_part_path(part).exists():
            continue
        path = motion_part_path(part)
        if not path.exists():
            return None
        reader = PartZarrReader(path)
        for key in columns:
            columns[key].append(
                np.asarray(reader.column(f"motion_{key}")[: reader.n_rows], np.float64)
            )
    if not columns["inside"]:
        return None
    joined = {k: np.concatenate(v) for k, v in columns.items()}
    if len(joined["inside"]) != len(order):
        return None  # a sibling from another embed of this part: not these rows
    return {k: v[order] for k, v in joined.items()}


def _outline_px(columns: dict, width: int, height: int) -> dict:
    """The sidecar's normalized outline in pixels, degrees and ratios (``mask_shape``'s rule:
    vertical extents are fractions of the height, lengths of ``sqrt(height x width)``).
    """
    scale = float(np.sqrt(width * height))
    as_float = {k: np.asarray(v, dtype=np.float64) for k, v in columns.items()}
    return {
        "top": as_float["mask_top_norm"] * height,
        "bottom": as_float["mask_bottom_norm"] * height,
        "floor_width": as_float["floor_width_norm"] * width,
        "major": as_float["major_axis_norm"] * scale,
        "minor": as_float["minor_axis_norm"] * scale,
        "orientation": as_float["orientation_deg"],
        "solidity": as_float["solidity"],
        "circularity": as_float["circularity"],
        "protrusions": as_float["protrusions"],
        "protrusion_depth": as_float["protrusion_depth_norm"] * scale,
    }


def _body_length(shape: dict | None) -> float | None:
    """The track's typical length: the median of its mask's long axis over the whole track."""
    if shape is None:
        return None
    major = shape["major"][np.isfinite(shape["major"])]
    return float(np.median(major)) if len(major) else None


def track_slice(recording, embed_record: dict, subject) -> FrameSlice | None:
    """One unit's whole trajectory from the best source its embedding run has — the one place
    a surface asks for it, so every descriptor surface measures the same rows.

    ``embed_record`` is the recording's embedding-run entry. The animal's sidecar is read
    (:func:`from_kinematics`), falling back to the tracking boxes (:func:`from_tracking`) for a run embedded before the sidecar existed.
    ``None`` when the run names no tracking run.
    """
    slice_ = _track_slice(recording, embed_record, subject)
    if slice_ is None:
        return None
    view = getattr(recording, "camera_view", None)
    slice_ = dataclasses.replace(
        slice_, camera_view=getattr(view, "view", None) if view is not None else None
    )
    run_id = keypoint_run_for(recording, embed_record)
    if run_id is None:
        return slice_
    try:
        points = keypoints_for(
            Path(recording.dir), run_id, int(subject), slice_.frame_idx
        )
    except (FileNotFoundError, KeyError, ValueError, OSError):
        points = None
    if points is None:
        return slice_
    length = _nose_to_tail(points)
    return dataclasses.replace(
        slice_,
        points=points,
        points_run=run_id,
        body_length=length if length is not None else slice_.body_length,
    )


def _nose_to_tail(points: dict) -> float | None:
    """The track's typical nose-to-tailbase distance, when both points exist: the body length a
    skeleton gives, preferred over the mask's long axis (which a curled body shortens).
    """
    from detail.analysis.descriptors import point_named

    nose, tail = point_named(points, "nose"), point_named(points, "tailbase")
    if nose is None or tail is None:
        return None
    lengths = np.hypot(*(nose - tail).T)
    lengths = lengths[np.isfinite(lengths)]
    return float(np.median(lengths)) if len(lengths) else None


def keypoint_run_for(recording, embed_record: dict) -> str | None:
    """Which keypoint track run describes this embedding run's animals, or ``None``.

    The one the embedding run drew or appended its pose from, when it used one; otherwise the
    newest complete track run on the same tracking run. Resolved here and **stated** in a
    slice (``points_run``) and in a profile's basis, so a different answer tomorrow is
    visible rather than silent.
    """
    named = embed_record.get("keypoint_run_id")
    if named:
        return str(named)
    tracking = embed_record.get("tracking_run_id")
    runs = [
        r
        for r in (getattr(recording, "keypoint_track_runs", None) or [])
        if r.get("status") == "complete" and r.get("tracking_run_id") == tracking
    ]
    return str(runs[-1]["id"]) if runs else None


def keypoints_for(
    recording_dir: Path, run_id: str, track_id: int, frame_idx: np.ndarray
) -> dict | None:
    """``{name: (N, 2) px}`` for ``track_id`` at the slice's recording-wide ``frame_idx``,
    NaN where a row has no point or its score is under the run's cut-off (the cut-off the
    embedding pass reads points with). ``None`` when the run holds nothing for this track.
    """
    from detail.pipeline.embedding import point_cutoff
    from detail.storage.keypoint_runs import KeypointDetectRun, KeypointTrackRun

    run = KeypointTrackRun(recording_dir, run_id)
    if int(track_id) not in run.track_ids:
        return None
    try:
        cutoff = point_cutoff(
            run, KeypointDetectRun(recording_dir, run.meta["detect_run_id"])
        )
    except (FileNotFoundError, KeyError, ValueError):
        cutoff = None
    data = recording_store.read_recording(recording_dir)
    width, height = _resolution(data)
    stems = [recording_store.part_stem(f) for f in recording_store.video_files(data)]
    offsets = recording_store.part_frame_offsets(recording_dir) or [0] * len(stems)
    names = tuple(run.skeleton.names)
    frames, points, scores = [], [], []
    for part_id in run.part_ids(int(track_id)):
        rows = run.rows(int(track_id), part_id)
        offset = offsets[stems.index(part_id)] if part_id in stems else 0
        frames.append(np.asarray(rows["frame_idx"], dtype=np.int64) + offset)
        points.append(np.asarray(rows["points"], dtype=np.float64))
        scores.append(np.asarray(rows["point_scores"], dtype=np.float64))
    if not frames:
        return None
    frames_all = np.concatenate(frames)
    points_all = np.concatenate(points) * np.array([width, height], dtype=np.float64)
    scores_all = np.concatenate(scores)
    if cutoff is not None:
        points_all[scores_all < cutoff] = np.nan
    order = np.argsort(frames_all, kind="stable")
    frames_all, points_all = frames_all[order], points_all[order]
    where = np.searchsorted(frames_all, frame_idx)
    where = np.clip(where, 0, max(len(frames_all) - 1, 0))
    hit = frames_all[where] == np.asarray(frame_idx, dtype=np.int64)
    out = {}
    for k, name in enumerate(names):
        column = np.full((len(frame_idx), 2), np.nan)
        column[hit] = points_all[where[hit], k]
        out[name] = column
    return out


def _track_slice(recording, embed_record: dict, subject) -> FrameSlice | None:
    tracking_run_id = embed_record.get("tracking_run_id")
    embed_run_id = embed_record.get("id")
    if tracking_run_id is None:
        return None
    cadence = embed_record.get("every_n_frames") or 1
    if embed_run_id is not None:
        slice_ = from_kinematics(
            recording.dir,
            embed_run_id=embed_run_id,
            track_id=int(subject),
            every_n_frames=cadence,
        )
        if slice_ is not None:
            return slice_
    return from_tracking(
        recording.dir,
        tracking_run_id=tracking_run_id,
        track_id=int(subject),
        embed_run_id=embed_run_id,
    )


def between(slice_: FrameSlice, start_t_ns: int, end_t_ns: int) -> FrameSlice:
    """The part of a slice inside a time span — one bout out of a whole trajectory.

    By **time**, not by row: a bout is recorded as a span of `t_ns`, and row indices belong to
    whichever store produced them. Inclusive at both ends, matching how a §19 instance is
    written.
    """
    t_ns = np.asarray(slice_.t_ns, dtype=np.int64)
    # `searchsorted`, not a boolean mask. A mask touches every row of the trajectory for every
    # bout, so describing a run is O(instances x rows) — a thousand bouts against a hundred
    # thousand rows is a hundred million comparisons per descriptor pass, and it is all to find
    # a contiguous span. The rows are in time order by construction, so the bounds are two
    # binary searches. Slicing also *views* rather than copies, which the mask could not.
    lo = int(np.searchsorted(t_ns, int(start_t_ns), side="left"))
    hi = int(np.searchsorted(t_ns, int(end_t_ns), side="right"))
    keep = slice(lo, hi)
    n = len(t_ns)

    def cut(value):
        # Every per-row array is cut and every other field carried, **by inspection** rather
        # than by a list: the list was complete until a field was added, and a new per-row
        # field then reached every bout as the whole trajectory, or not at all.
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == n:
            return value[keep]
        if isinstance(value, dict):
            return {k: cut(v) for k, v in value.items()}
        return value

    return FrameSlice(
        **{
            f.name: (t_ns[keep] if f.name == "t_ns" else cut(getattr(slice_, f.name)))
            for f in dataclasses.fields(FrameSlice)
        }
    )


def postures_from(run_dir: Path | str, slice_: FrameSlice, *, recording_id, subject):
    """``slice_`` with a posture label per frame, read from a pose run's §19 frame view.

    The pose→behavior bridge, and it needs no new computation: a saved pose-clustering run (the
    module is removed) already records which posture holds at every frame, and a behavior bout is a
    stretch of those frames. So what a behavior is *made of* is a join, not a measurement.

    Aligned by **time**, not by row. The two runs sample independently — a pose run may embed at
    a different cadence, and its frame view is indexed by its own ``t_ns`` axis — so each of this
    slice's frames takes the label of the pose frame at or before it. A positional zip of two
    stores that merely happen to have similar lengths is the join that looks right and is wrong
    everywhere except the middle.

    Frames the pose run did not label come back ``-1``: it may cover less of the recording, and a
    posture invented for an unlabelled frame would be indistinguishable from a real one.

    Returns the slice unchanged when the run has no view for this unit — the ordinary case, and
    a reason to show fewer numbers rather than to fail.
    """
    from detail.storage import frame_view as frame_view_store

    try:
        view = frame_view_store.read_frame_view(run_dir, recording_id, subject)
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return slice_
    source_labels = view.source_labels
    if not source_labels or not len(slice_):
        return slice_

    axis = np.asarray(view.t_ns[:], dtype=np.int64)
    if not len(axis):
        return slice_
    # `searchsorted` gives the pose frame at or before each of ours; clipped so a frame before
    # the pose run's first is attributed to nothing rather than wrapping to its last.
    rows = (
        np.searchsorted(axis, np.asarray(slice_.t_ns, dtype=np.int64), side="right") - 1
    )
    inside = rows >= 0

    labels = np.full(len(slice_), -1, dtype=np.int64)
    for ordinal, source_label in enumerate(source_labels):
        occupied = np.asarray(view.root[source_label][:]) > 0
        hit = inside & occupied[np.clip(rows, 0, len(occupied) - 1)]
        labels[hit] = ordinal
    slice_.labels = labels
    return slice_


def _resolution(data: dict) -> tuple[float, float]:
    """The recording's frame size, for turning normalized boxes into pixels."""
    resolution = (data.get("video") or {}).get("resolution") or [1.0, 1.0]
    return float(resolution[0]), float(resolution[1])


def _series_parts(
    recording_dir: Path, embed_run_id: str, track_id: int
) -> list[tuple[int, Path]]:
    """``(recording-wide frame offset, part store)`` for each part a series was embedded in.

    A store's ``frame_idx`` is the detection part's own and restarts at zero in every video part,
    so a row's recording-wide frame number is that plus the frames in the parts before it. A join
    between this and a slice (whose frame numbers are recording-wide) that skips the offset
    compares part-local numbers with recording-wide ones: for a later part it finds nothing, or
    finds another part's frame with the same number.
    """
    data = recording_store.read_recording(recording_dir)
    stems = [recording_store.part_stem(f) for f in recording_store.video_files(data)]
    offsets = recording_store.part_frame_offsets(recording_dir) or [0] * len(stems)
    run_dir = recording_dir / "embeddings" / embed_run_id / f"track_{track_id}"
    return [
        (offsets[stems.index(part.stem)] if part.stem in stems else 0, part)
        for part in sorted(run_dir.glob("part_*.zarr"))
    ]


def _embeddings_for(recording_dir, embed_run_id, track_id, frame_idx):
    """The embeddings aligned to ``frame_idx``, or ``None``.

    The embedding store holds its own rows at its own cadence, which is a *subset* of the
    detection rows — it embeds every Nth tracked frame. So the two are joined on ``frame_idx``
    rather than assumed parallel; a positional zip would pair a box with another frame's
    embedding and every appearance-change number would be quietly wrong.

    A frame with no embedding gets ``NaN``, which `cosine_step` already treats as undefined —
    the same answer it gives across a tracking gap, and for the same reason.
    """
    if not embed_run_id:
        return None
    vectors: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    for offset, part in _series_parts(Path(recording_dir), embed_run_id, track_id):
        reader = PartZarrReader(part)
        if reader.n_rows == 0:
            continue
        vectors.append(
            np.asarray(reader.column("embedding")[: reader.n_rows], dtype=np.float32)
        )
        frames.append(
            np.asarray(reader.column("frame_idx")[: reader.n_rows]).astype(np.int64)
            + offset
        )
    if not vectors:
        return None
    emb, emb_frames = np.concatenate(vectors), np.concatenate(frames)

    # `searchsorted` over the embedding's own sorted frame list, not a dict comprehension: the
    # dict was one Python-level pass per frame over a hundred thousand of them, per subject.
    order = np.argsort(emb_frames, kind="stable")
    sorted_frames = np.asarray(emb_frames)[order]
    at = np.searchsorted(sorted_frames, frame_idx)
    at_clipped = np.clip(at, 0, len(sorted_frames) - 1)
    found = sorted_frames[at_clipped] == frame_idx
    if not found.any():
        return None
    embeddings = np.full((len(frame_idx), emb.shape[1]), np.nan, dtype=np.float64)
    embeddings[found] = emb[order[at_clipped[found]]]
    return embeddings
