"""Single-subject selection + fragment merge, shared by the single-subject
trackers (Section 13).

One subject per recording (``object_id`` 0): the tracker picks one detection per
frame as the subject and merges overlapping lower-confidence fragments onto it
(the embedding stage OR-unions them into one silhouette, Section 14). The generic
bbox / frame / arena / contiguity / validation primitives live in ``_common`` and
are re-exported here; only the subject-selection and fragment-attachment logic —
specific to the single-subject case — lives in this module.

``anchor_tracker`` selects by confidence alone with the mask-IoU gate disabled;
``iou_single_tracker`` passes a confidence-plus-IoU-continuity ``score`` and turns
the mask-IoU gate on. Pure compute over the detection columns (Tenet 7); masks
decode only for the merge / contiguity gates.
"""

from collections.abc import Callable

import numpy as np

from detail.core.utils import decode_mask
from detail.trackers._common import (  # re-exported for the single-subject trackers
    SUBJECT_ID,
    Bbox,
    is_contiguous,
    iou,
)

__all__ = [
    "SUBJECT_ID",
    "Bbox",
    "is_contiguous",
    "iou",
    "mask_iou",
    "select_subject",
    "attach_secondaries",
]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks (intersection over union of True pixels); 0 if
    either is empty. Unlike bbox IoU this is ~0 for a reflection whose box
    overlaps the subject but whose pixels are a separate region."""
    inter = int(np.count_nonzero(a & b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return inter / union if union > 0 else 0.0


def select_subject(
    rows: list[int],
    conf: np.ndarray,
    mask_rle: np.ndarray,
    *,
    floor: float,
    require_contiguous: bool,
    resolution: tuple[int, int] | None,
    score: Callable[[int], float],
) -> int | None:
    """Highest-``score`` row at or above ``floor`` whose mask passes the
    contiguity gate; ``None`` if none qualifies. Masks decode lazily in score
    order, so a frame usually pays for only the winning candidate — ordered by an
    arbitrary ``score`` (confidence for ``anchor``, confidence + IoU continuity
    for ``iou_single``)."""
    candidates = sorted(
        (i for i in rows if float(conf[i]) >= floor), key=score, reverse=True
    )
    for i in candidates:
        if require_contiguous and not is_contiguous(
            decode_mask(str(mask_rle[i]), resolution)
        ):
            continue
        return i
    return None


def attach_secondaries(
    subject: int,
    rows: list[int],
    bbox: np.ndarray,
    conf: np.ndarray,
    mask_rle: np.ndarray,
    *,
    secondary_conf: float,
    merge_iou: float,
    merge_mask_iou: float,
    max_merge: int,
    require_contiguous: bool,
    resolution: tuple[int, int] | None,
    object_ids: np.ndarray,
) -> None:
    """Give ``SUBJECT_ID`` to overlapping lower-confidence fragments of the subject.

    Two-stage gate: bbox IoU (``merge_iou``) is the cheap pre-filter, so the many
    far-away spurious detections never pay a mask decode; survivors must then clear
    a **mask** IoU floor (``merge_mask_iou``) against the subject mask. The mask
    floor is what rejects a glass reflection — its box overlaps the subject but its
    pixels are a separate region (~0 mask IoU). The subject mask decodes once per
    frame and is reused, and each candidate mask decodes at most once (shared by the
    mask-IoU and contiguity checks). ``merge_mask_iou <= 0`` disables the mask gate;
    ``max_merge == 0`` disables merging entirely.
    """
    subject_box = tuple(float(x) for x in bbox[subject])
    candidates = sorted(
        (i for i in rows if i != subject and float(conf[i]) >= secondary_conf),
        key=lambda i: float(conf[i]),
        reverse=True,
    )
    subject_mask: np.ndarray | None = None
    attached = 0
    for i in candidates:
        if attached >= max_merge:
            break
        if iou(subject_box, tuple(float(x) for x in bbox[i])) < merge_iou:
            continue
        # Cheap bbox gate passed; the rest needs masks. Decode the candidate (and
        # the subject, once) only when a mask gate is active, and reuse the decoded
        # candidate mask for both the mask-IoU and contiguity checks.
        cand_mask = None
        if merge_mask_iou > 0.0:
            cand_mask = decode_mask(str(mask_rle[i]), resolution)
            if subject_mask is None:
                subject_mask = decode_mask(str(mask_rle[subject]), resolution)
            if mask_iou(subject_mask, cand_mask) < merge_mask_iou:
                continue
        if require_contiguous:
            if cand_mask is None:
                cand_mask = decode_mask(str(mask_rle[i]), resolution)
            if not is_contiguous(cand_mask):
                continue
        object_ids[i] = SUBJECT_ID
        attached += 1
