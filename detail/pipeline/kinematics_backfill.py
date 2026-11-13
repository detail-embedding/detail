"""Regenerate an embedding run's ``kinematics/`` sibling from the detection zarr (§14).

An embedding run made before kinematics existed carries none, and re-embedding to get it would
cost a GPU and hours for data that is already on disk: the detection zarr keeps every mask RLE
and every box, and the embedding parts say which frames were embedded. This walks the two
**pairwise, part by part** and writes the sibling the embed pass would have written.

Pairwise is the whole point. A detection is identified by ``(track_id, part_index, det_row)`` and
the embedding zarr stores no part, so flattening the parts and then matching by index resolves one
part's rows against another's — plausibly, and wrongly. Walking a part at a time never forms that
index, which is also why the inline writer collects geometry during the pass rather than
reconstructing it after.

CPU-only: no model, no torch, no video decode. The masks are re-decoded, which is the cost.
"""

from pathlib import Path

import numpy as np

from detail.analysis import geometry
from detail.core import atomic
from detail.core import cpu
from detail.core import lookahead
from detail.core import yaml_load
from detail.core.utils import decode_mask
from detail.executor import rundir
from detail.storage.zarr_io import (
    KINEMATICS_SHAPE_COLUMNS,
    DetectionPartReader,
    KinematicsPartWriter,
    PartZarrReader,
    kinematics_part_path,
    resolve_assignment_rows,
)


def _embedding_meta(run_dir: Path) -> dict:
    return yaml_load.safe_load(atomic.read_text(run_dir / "embedding.yaml"))


def is_current(embedding_part: Path) -> bool:
    """Whether a part's kinematics sibling exists **and** carries the mask's outline.

    A sidecar written before the outline columns (``mask_shape``) existed is complete for where
    the animal was and silent on what shape it was; a backfill brings it up to date rather than
    leaving the shape measures unavailable on every run made before them.
    """
    sidecar = kinematics_part_path(embedding_part)
    if not sidecar.exists():
        return False
    return KINEMATICS_SHAPE_COLUMNS[0] in PartZarrReader(sidecar).column_names


def missing_parts(recording_dir: Path | str, embed_run_id: str) -> list[Path]:
    """Embedding parts in this run with no current kinematics sibling, in walk order."""
    run_dir = Path(recording_dir) / "embeddings" / embed_run_id
    return [
        part
        for track_dir in sorted(run_dir.glob("track_*"))
        for part in sorted(track_dir.glob("part_*.zarr"))
        if not is_current(part)
    ]


def backfill_run(
    recording_dir: Path | str,
    embed_run_id: str,
    *,
    progress=None,
    overwrite: bool = False,
) -> int:
    """Write the kinematics sibling for every part of one embedding run; returns parts written.

    ``progress`` is the standard ``(stage, fraction, detail)`` callback — this is minutes on a
    long recording, and a silent long op is indistinguishable from a hang.

    Already-backfilled parts are skipped unless ``overwrite``, so a re-run after an interruption
    resumes rather than redoing the work. A part whose sidecar predates the outline columns is
    not "already backfilled" (:func:`is_current`): it is rewritten with them.
    """
    recording_dir = Path(recording_dir)
    run_dir = recording_dir / "embeddings" / embed_run_id
    if not (run_dir / "embedding.yaml").exists():
        raise ValueError(
            f"no embedding run {embed_run_id!r} in recording {recording_dir.name!r}"
        )
    meta = _embedding_meta(run_dir)
    tracking_dir = recording_dir / "tracking" / meta["tracking_run_id"]
    track_meta = yaml_load.safe_load(atomic.read_text(tracking_dir / "track.yaml"))
    detection_dir = recording_dir / "detections" / track_meta["detection_run_id"]
    class_name = track_meta.get("class_name")
    frame_hw = rundir.read_envelope(detection_dir)["mask_resolution"]

    parts = [
        part
        for track_dir in sorted(run_dir.glob("track_*"))
        for part in sorted(track_dir.glob("part_*.zarr"))
    ]
    if not overwrite:
        parts = [p for p in parts if not is_current(p)]
    written = 0
    for index, part in enumerate(parts):
        track_id = int(part.parent.name.split("_")[-1])

        def report(fraction: float, part=part, index=index) -> None:
            """Progress *within* the part as well as across parts.

            A recording is usually **one** part, so a bar that only moved at part boundaries sat
            at 0% for minutes and then jumped to 100% — which is the "looks hung" failure
            the live-progress rule exists to stop, and the reason that rule says to test the
            inner pair on a single-unit fixture.
            """
            if progress is not None:
                progress(
                    "kinematics",
                    (index + fraction) / max(len(parts), 1),
                    part.parent.name,
                )

        report(0.0)
        _backfill_part(
            part,
            detection_dir,
            tracking_dir,
            track_id,
            frame_hw,
            class_name,
            progress=report,
        )
        written += 1
    if progress is not None:
        progress("kinematics", 1.0, f"{written} part(s)")
    return written


def _decode_workers() -> int:
    """Threads used to decode masks. The same rule — and the same reason — as the embedding
    worker's render pool: the RLE decode releases the GIL, so threads help, and the
    1080p decode is memory-bound so returns flatten past ~8."""
    return cpu.thread_workers()


def _backfill_part(
    embedding_part: Path,
    detection_dir: Path,
    tracking_dir: Path,
    track_id: int,
    frame_hw,
    class_name: str | None = None,
    *,
    progress=None,
) -> None:
    """One part. Reads only this part's detection rows, so no cross-part index is ever formed."""
    det_reader = DetectionPartReader(detection_dir / embedding_part.name)
    assign_reader = PartZarrReader(tracking_dir / "assignments" / embedding_part.name)
    # object_id is parallel to the detection rows the tracker actually kept when it
    # ran (§13) — every row when this run had no class filter, only one class's
    # otherwise (anchor_tracker et al. build object_ids from filter_by_class'd
    # data). raw_rows recovers which reader row each assignment row really is.
    raw_rows = resolve_assignment_rows(det_reader, assign_reader, class_name)
    n = len(raw_rows)
    det = {k: np.asarray(v)[raw_rows] for k, v in det_reader.read().items()}
    object_id = np.asarray(assign_reader.read()["object_id"][:n])
    frame_col = np.asarray(det["frame_idx"])
    bbox_col = np.asarray(det["bbox"])
    mine = np.where(object_id.astype(np.int64) == int(track_id))[0]
    by_frame: dict[int, list[int]] = {}
    for row in mine.tolist():
        by_frame.setdefault(int(frame_col[row]), []).append(row)

    embedded = PartZarrReader(embedding_part).read()
    out = kinematics_part_path(embedding_part)
    if out.exists():
        # A partial store from an interrupted run would otherwise be appended to, doubling
        # its rows against an embedding part that has not changed.
        import shutil

        shutil.rmtree(out)
    frames = np.asarray(embedded["frame_idx"]).tolist()
    times = np.asarray(embedded["t_ns"]).tolist()
    size = tuple(frame_hw)

    def geometry_of(frame_idx: int):
        """One frame's geometry. Decoding dominates, and it is what the pool parallelizes."""
        rows = by_frame.get(int(frame_idx), [])
        mask = None
        for row in rows:
            decoded = decode_mask(str(det["mask_rle"][row]), size)
            mask = decoded if mask is None else (mask | decoded)
        return geometry.frame_geometry(
            mask, [tuple(float(v) for v in bbox_col[r]) for r in rows], frame_hw
        )

    # Decoded ahead on a thread pool, consumed in order, because serial decode took
    # minutes per recording. The rows must still be *written* in frame order, so the pool
    # runs a bounded window ahead of the writer rather than gathering everything — the seam
    # the embedding worker renders through too (`core/lookahead.py`).
    with KinematicsPartWriter(out) as writer:
        geoms = lookahead.ordered_map(
            geometry_of, frames, workers=_decode_workers(), window=512
        )
        try:
            for i, geom in enumerate(geoms):
                writer.add_row(
                    int(frames[i]),
                    int(times[i]),
                    geom.cx_norm,
                    geom.cy_norm,
                    geom.mask_area_norm,
                    geom.bbox_norm,
                    geom.shape,
                )
                if progress is not None and i % 2000 == 0:
                    progress(i / max(len(frames), 1))
        finally:
            geoms.close()
