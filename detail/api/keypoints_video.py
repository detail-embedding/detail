"""The ``keypoints_video`` export (§26 *Analysis and export*; #689): the footage with
the run's points drawn on every frame, the tracking overlay opt-in.

One sequential pass per video part (Tenet 11: ``cv2.VideoCapture``, read forward, never seek)
through one `PointSource`, so the cut-off and the ``source`` choice are the ones every other
keypoint target used. A frame the run placed nothing on is written as it was decoded, so the
output has the part's every frame and plays at the recording's own rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from detail.api.keypoint_export import PointSource
from detail.core.dataclasses import Detection
from detail.core.progress import ThrottledProgress
from detail.core.utils import decode_mask
from detail.core.visualize import (
    MASK_TINT,
    bgr_to_rgb,
    draw_detections,
    draw_keypoints,
    rgb_to_bgr,
)
from detail.executor import rundir
from detail.storage import recording as recording_store
from detail.storage import video as video_store
from detail.storage.zarr_io import DetectionPartReader, PartZarrReader


def _tracking_overlay(source: PointSource, part_id: str) -> dict[int, list[dict]]:
    """Per frame, the detections under the points, as the rows the overlay is drawn from
    (the mask decoded only when the frame is written): a track run's tracked detections, from
    the tracking run it assigned to; a detect run's instances belong to no subject, so its
    overlay is every detection of the detection run it read, with no track id."""
    detection_dir = source.recording_dir / "detections" / source.detect.detection_run_id
    det_path = detection_dir / f"{part_id}.zarr"
    if not det_path.is_dir():
        return {}
    det = DetectionPartReader(det_path).read()
    if source.tracking_run_id is None:
        object_id = np.full(len(det["frame_idx"]), -1, dtype=np.int64)
        keep = np.arange(len(object_id))
    else:
        tracking_dir = source.recording_dir / "tracking" / source.tracking_run_id
        assign_path = tracking_dir / "assignments" / f"{part_id}.zarr"
        if not assign_path.is_dir():
            return {}
        object_id = np.asarray(PartZarrReader(assign_path).read()["object_id"])
        keep = np.flatnonzero(object_id >= 0)
    by_frame: dict[int, list[dict]] = {}
    frames = np.asarray(det["frame_idx"])
    for row in keep.tolist():
        by_frame.setdefault(int(frames[row]), []).append(
            {
                "bbox": tuple(float(v) for v in det["bbox"][row]),
                "mask_rle": str(det["mask_rle"][row]),
                "class_name": str(det["class_name"][row]),
                "confidence": float(det["confidence"][row]),
                "object_id": int(object_id[row]) if object_id[row] >= 0 else None,
            }
        )
    return by_frame


def export_keypoints_video(
    source: PointSource,
    dest: Path | str,
    stem: str,
    *,
    overlay: bool = False,
    progress=None,
) -> list[Path]:
    """Write ``<stem>_<part>_keypoints.mp4`` for every part the run has rows in.

    Args:
        source: the run read for export (its cut-off decides which points are drawn).
        dest: the export folder; must exist.
        stem: the file-name stem (the recording's name, made path-safe).
        overlay: also draw the tracking run's masks and boxes under the points.
        progress: a ``(stage, fraction, detail)`` callback; throttled here.
    """
    import cv2

    dest = Path(dest)
    parts = source.part_ids()
    stems = source.part_stems()
    emit = ThrottledProgress(
        None if progress is None else (lambda event: progress(*event))
    )
    written: list[Path] = []
    skeleton = source.skeleton
    for index, part_id in enumerate(parts):
        if part_id not in stems:
            continue
        part = recording_store.video_files(source.recording)[stems.index(part_id)]
        video_path = recording_store.resolve_part_video_path(source.recording_dir, part)
        if not video_path.is_file():
            continue
        n_frames = source.frame_count(part_id)
        tracks = [t for t in source.series_ids() if part_id in source.series_parts(t)]
        overlay_rows = _tracking_overlay(source, part_id) if overlay else {}
        # The overlay's masks decode at the size their detection run encoded them at (§9).
        encoded = (
            rundir.mask_resolution(
                source.recording_dir / "detections" / source.detect.detection_run_id,
                part_id,
            )
            if overlay
            else None
        )
        out_path = dest / f"{stem}_{part_id}_keypoints.mp4"
        emit(("drawing", index / len(parts), out_path.name), force=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        writer = None
        try:
            frame_idx = 0
            while frame_idx < n_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = bgr_to_rgb(bgr)
                h, w = rgb.shape[:2]
                rows = overlay_rows.get(frame_idx)
                if rows:
                    rgb = draw_detections(
                        rgb,
                        [
                            Detection(
                                mask=decode_mask(
                                    r["mask_rle"], encoded or (h, w), out=(h, w)
                                ),
                                bbox=r["bbox"],
                                class_name=r["class_name"],
                                confidence=r["confidence"],
                                object_id=r["object_id"],
                            )
                            for r in rows
                        ],
                        MASK_TINT,
                    )
                for track_id in tracks:
                    found = source.at(track_id, part_id, frame_idx)
                    if found is None:
                        continue
                    points, scores, visible = found
                    rgb = draw_keypoints(
                        rgb, points, skeleton, visible=visible, scores=scores
                    )
                if writer is None:
                    writer, _codec = video_store.open_video_writer(
                        out_path, source.fps(part_id), w, h
                    )
                writer.write(rgb_to_bgr(rgb))
                frame_idx += 1
                emit(
                    (
                        "drawing",
                        (index + frame_idx / n_frames) / len(parts),
                        out_path.name,
                    ),
                    force=frame_idx >= n_frames,
                )
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        if writer is not None:
            written.append(out_path)
    return written
