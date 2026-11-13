"""The step a per-detection keypoint method shares (§26 "KeypointModel protocol").

An ``InstanceFrame`` holds every detection of one frame; a top-down method places one pose
instance per detection from that detection's box. This turns a batch of frames into one batched
call over all their detections, in frame order then detection-row order, and folds the answers
back into one ``FrameResult`` per frame — one row per detection, ``detection_row`` set and
``track_id`` unset, because identity is tracking's and ``keypoint_track`` assigns it (§26
*Assignment*). Each method writes only the placing itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from detail.core.dataclasses import FrameResult
from detail.core.keypoints import InstanceFrame

#: ``place`` takes ``[(frame RGB (H, W, 3), normalized xyxy box), ...]`` and returns, per
#: detection, ``(points (K, 2) normalized full-frame, NaN = not placed; scores (K,))``.
Placer = Callable[
    [list[tuple[np.ndarray, tuple[float, float, float, float]]]],
    Sequence[tuple[np.ndarray, np.ndarray]],
]


def place_subjects(
    frames: Sequence[InstanceFrame], n_keypoints: int, place: Placer
) -> list[FrameResult]:
    """One ``FrameResult`` per frame from one ``place`` call over every detection in
    ``frames``."""
    order = [(i, row) for i, f in enumerate(frames) for row in sorted(f.boxes)]
    placed = (
        place([(frames[i].frame, frames[i].boxes[r]) for i, r in order])
        if order
        else []
    )
    if len(placed) != len(order):
        raise ValueError(
            f"the method placed {len(placed)} instances for the {len(order)} detections "
            "it was given"
        )
    by_frame: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for (i, row), (xy, score) in zip(order, placed):
        by_frame.setdefault(i, []).append((row, xy, score))
    out = []
    for i, frame in enumerate(frames):
        rows = by_frame.get(i, [])
        n = len(rows)
        points = np.full((n, n_keypoints, 2), np.nan, np.float32)
        scores = np.full((n, n_keypoints), np.nan, np.float32)
        for j, (_row, xy, score) in enumerate(rows):
            points[j] = np.asarray(xy, np.float32).reshape(n_keypoints, 2)
            scores[j] = np.asarray(score, np.float32).reshape(n_keypoints)
        det_rows = [r for r, _xy, _s in rows]
        out.append(
            FrameResult(
                frame_idx=frame.frame_idx,
                t_ns=frame.t_ns,
                n=n,
                confidence=np.full(n, np.nan, np.float32),
                bbox=np.asarray([frame.boxes[r] for r in det_rows], np.float32).reshape(
                    n, 4
                ),
                detection_row=np.asarray(det_rows, np.int32),
                points=points,
                point_scores=scores,
            )
        )
    return out
