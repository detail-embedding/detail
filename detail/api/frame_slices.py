"""Assemble the signals a descriptor reads — the bridge from stores to `analysis.descriptors`.

A descriptor is pure over a `FrameSlice`; something has to build one. That is here rather than in
`analysis/` because it opens zarr and resolves recordings, and `analysis/descriptors.py` is
deliberately numpy-only so the pipeline can import it.

**Two loaders, one shape.** The geometry a descriptor wants is captured twice in this codebase:
during the embed pass (the §14 ``kinematics/`` sidecar, one row per embedded frame, cheap to
read) and in the detection store (boxes, and masks as RLE). `from_tracking` reads the detection
store; `from_kinematics` reads the sidecar. Both fill the same `FrameSlice`, so a descriptor
cannot tell which one fed it. That is why the loader is a seam rather than a step inside the
descriptor code: the second loader was added without anything above it changing.

**Which to use.** `from_tracking` covers every tracked frame but only with boxes (and the
embeddings, when asked), so the four mass-centre descriptors report `NaN` from it and a surface
shows them as unavailable rather than blank — an RLE decode per frame is exactly what the sidecar
exists to avoid repeating. `from_kinematics` carries the mask centroid and area too, at the
embedding's cadence, which is the cadence the clustering saw. A run from before the sidecar
existed has none: the loader says so with `None`, and `Recording.backfill_kinematics` writes it.

**Two clocks, deliberately not reconciled here.** `from_tracking` offsets each later part's
`t_ns` by the elapsed time of the parts before it, so a multi-part recording is one timeline;
the sidecar carries each part's own clock as the embedding rows do, which is what a clustering
run's `start_t_ns`/`end_t_ns` were taken from, so `between` agrees with those spans by
construction. A multi-part recording therefore reads different `t_ns` from the two loaders past
its first part. That disagreement predates this module and is recorded rather than settled.
"""

from pathlib import Path

import numpy as np

from detail.analysis.descriptors import FrameSlice
from detail.storage import recording as recording_store


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
    elapsed_ns = 0
    for part in preview._parts:
        reader, assignment = part["reader"], part["assignment"]
        stamps = np.asarray(part["timestamps"], dtype=np.int64)
        if reader is not None and assignment is not None and reader.n_rows:
            object_id = np.asarray(assignment.column("object_id")[:])
            rows = np.flatnonzero(object_id == int(track_id))
            if len(rows):
                # Part-local frame indices are made global, so a multi-part recording is one
                # timeline: a step between the last row of one part and the first of the next
                # is otherwise a jump backwards, and reads as an enormous speed.
                local = np.asarray(part["frame_idx"])[rows]
                frames.append(local + offset)
                boxes.append(np.asarray(reader.column("bbox")[:])[rows])
                # The part's own PTS map — the authoritative frame -> t_ns (Tenet 8), offset so
                # later parts continue the timeline rather than restarting it.
                times.append(stamps[np.clip(local, 0, len(stamps) - 1)] + elapsed_ns)
        offset += part["n_frames"]
        elapsed_ns += (
            int(stamps[-1]) + int(np.median(np.diff(stamps))) if len(stamps) > 1 else 0
        )

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
    recording_dir: Path | str, *, embed_run_id: str, track_id: int
) -> FrameSlice | None:
    """One track's embedded frames from the §14 ``kinematics/`` sidecar, as a `FrameSlice` in
    **pixel** units — centroid, mask area and box filled; ``None`` when the run has no sidecar.

    One row per *embedded* frame (the run's ``every_n_frames`` is the slice's ``stride``), with
    ``frame_idx`` made global across parts the way `from_tracking` does, using each part's frame
    count from its timestamp map — never from the sidecar's own highest row. ``t_ns`` is the
    sidecar's column (see the module docstring on the two clocks). ``embeddings`` is left
    ``None``: a caller that wants them has `from_tracking`.

    Raises:
        ValueError: ``recording.yaml`` records no ``video.resolution`` (the sidecar is
            normalized to the frame, so without a size there are no pixels to give — and
            normalized numbers labelled as pixels would be wrong everywhere without failing);
            or the sidecar names a video part the recording does not list.
    """
    from detail.api.run_preview import embedding_cadence, iter_track_kinematics

    recording_dir = Path(recording_dir)
    parts = iter_track_kinematics(recording_dir, embed_run_id, int(track_id))
    if not parts:
        return None
    data = recording_store.read_recording(recording_dir)
    if not (data.get("video") or {}).get("resolution"):
        raise ValueError(
            f"{recording_dir.name}: recording.yaml records no video.resolution, so the "
            "kinematics sidecar's normalized geometry cannot be given in pixels"
        )
    width, height = _resolution(data)
    offsets = _part_offsets(recording_dir, data)

    frames, times, cx, cy, areas, boxes = [], [], [], [], [], []
    for stem, columns in parts:
        if stem not in offsets:
            raise ValueError(
                f"{recording_dir.name}: embedding run {embed_run_id!r} has kinematics for "
                f"part {stem!r}, which recording.yaml does not list"
            )
        frames.append(np.asarray(columns["frame_idx"], dtype=np.int64) + offsets[stem])
        times.append(np.asarray(columns["t_ns"], dtype=np.int64))
        cx.append(np.asarray(columns["cx_norm"], dtype=np.float64))
        cy.append(np.asarray(columns["cy_norm"], dtype=np.float64))
        areas.append(np.asarray(columns["mask_area_norm"], dtype=np.float64))
        boxes.append(np.asarray(columns["bbox_norm"], dtype=np.float64).reshape(-1, 4))

    frame_idx = np.concatenate(frames)
    order = np.argsort(frame_idx, kind="stable")
    frame_idx = frame_idx[order]
    centroid = np.stack(
        [np.concatenate(cx)[order] * width, np.concatenate(cy)[order] * height], axis=1
    )
    return FrameSlice(
        frame_idx=frame_idx,
        t_ns=np.concatenate(times)[order],
        stride=embedding_cadence(recording_dir, data, embed_run_id),
        centroid=centroid,
        area=np.concatenate(areas)[order] * width * height,
        bbox=np.concatenate(boxes)[order]
        * np.array([width, height, width, height], dtype=np.float64),
    )


def _part_offsets(recording_dir: Path, data: dict) -> dict[str, int]:
    """``{part stem: the global frame its first frame is}``, from each part's timestamp map —
    the authoritative frame count (Tenet 8), as `from_tracking` and `_build_parts` count.
    """
    offsets: dict[str, int] = {}
    total = 0
    for part_index, file in enumerate(recording_store.video_files(data)):
        offsets[recording_store.part_stem(file)] = total
        total += len(
            recording_store.load_active_timestamps(
                recording_dir, data, part_index
            ).values
        )
    return offsets


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
    take = lambda a: None if a is None else np.asarray(a)[keep]  # noqa: E731
    return FrameSlice(
        frame_idx=np.asarray(slice_.frame_idx)[keep],
        t_ns=t_ns[keep],
        stride=slice_.stride,
        part=slice_.part,
        embeddings=take(slice_.embeddings),
        centroid=take(slice_.centroid),
        area=take(slice_.area),
        bbox=take(slice_.bbox),
        labels=take(slice_.labels),
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
    from detail.api.run_preview import load_track_trajectory

    try:
        emb, emb_frames, _emb_t_ns = load_track_trajectory(
            recording_dir, embed_run_id, int(track_id)
        )
    except (FileNotFoundError, ValueError):
        return None

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
