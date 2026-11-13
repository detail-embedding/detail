"""Closed-set keypoints with a locally trained YOLO-pose student (§12 *Distill keypoints*).

The apply side of a keypoint distill: load the ``best.pt`` a pose training run produced and
place the student's own skeleton on every animal it finds, no example set, one candidate per
point. It assigns no subject: identity is tracking's, and ``keypoint_track`` gives each pose to
a subject by mask (§26 *Assignment*). Before 2026-10-21 the student matched itself to tracks by
box IoU here; that is gone. It is run the way it was trained (the dataset's ``student_input``,
§12):

- **whole frame** (the default; ``per_detection = False``): one forward pass per frame, and
  **every pose whose box confidence is at least ``min_pose_confidence``** comes back (§26
  *Visibility*: 0.25 by default, Ultralytics' own), each with its box, its confidence and
  ``detection_row = −1``.
- **crop** (``per_detection = True`` on the instance): one square crop per **detection**
  (``patch_match.view``, the crop the dataset was cut with); the pose nearest the crop's centre
  is that detection's instance, ``detection_row`` its row, and its points go back to the frame
  through the same view. The way forward for animals that overlap.

Either way the image is handed over **BGR**: Ultralytics reads a numpy array as BGR and the
student was trained on ``cv2.imwrite``'d images (#749 on main, #757 on remote-detail).

The threshold reaches the model through ``set_min_pose_confidence`` after ``load()`` — the
detection models' ``set_conf_threshold`` shape — rather than the constructor, because the
resolver builds the student from its catalog record, which knows nothing of a run.

A confidence of 0 is NaN, never a guess: Ultralytics reports an unseen keypoint at ``(0, 0)``
with a near-zero confidence, and a point the student did not see is absent, not placed in the
corner. The point score is the detector's own confidence (``ScoreSpec``), so its cut-off is a
confidence, not a cosine.

Like ``TrainedYOLO`` this is constructed by the resolver (``models/resolve.py``) from a model
record's weights path and skeleton copy, not registered under an id in ``KEYPOINT_MODELS``:
there is one of these per trained model. ``from ultralytics import YOLO`` stays inside ``load``
and ``introspect`` through the one seam (Tenet 13).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from detail.configs.keypoint_detect import DEFAULT_MIN_POSE_CONFIDENCE
from detail.core.dataclasses import FrameResult, ModelReadiness
from detail.core.installed import is_importable
from detail.core.keypoints import InstanceFrame, ScoreSpec
from detail.core.protocols import KeypointModel
from detail.core.skeleton import Skeleton
from detail.core.visualize import rgb_to_bgr
from detail.models._ultralytics import SplitPredict
from detail.models._ultralytics import import_ultralytics


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


class TrainedYoloPose(KeypointModel):
    """Place a trained student's own skeleton on every animal in the frame (closed-set)."""

    model_id = "trained_yolo_pose"  # the family loader's id; not a KEYPOINT_MODELS key
    architecture = "yolo_pose"
    supported_backends = ["cpu", "mps", "cuda"]
    needs_examples: ClassVar[bool] = False
    per_detection: ClassVar[bool] = False
    needs_mask: ClassVar[bool] = False
    n_candidates: ClassVar[int] = 1
    reports_confidence: ClassVar[bool] = True
    score_spec: ClassVar[ScoreSpec] = ScoreSpec(
        meaning="detector confidence", low=0.0, high=1.0, default_cutoff=0.5
    )
    default_track_method: ClassVar[str] = "despike_smooth"

    def __init__(
        self,
        weights_path: Path | str,
        skeleton: Skeleton,
        *,
        student_input: str = "whole_frame",
        crop_px: int = 448,
    ) -> None:
        super().__init__(None)  # no size variants: the weights are the model
        skeleton.validate()
        if student_input not in ("whole_frame", "crop"):
            raise ValueError(
                f"student_input must be whole_frame or crop, got {student_input!r}"
            )
        self._weights_path = Path(weights_path)
        self._skeleton = skeleton
        self._student_input = student_input
        self._crop_px = int(crop_px)
        # A crop student reads one crop per detection, so the detect stage hands it the
        # detections' boxes; a whole-frame one reads the frame and finds every animal itself.
        self.per_detection = student_input == "crop"
        self._model = None
        self._device = "cpu"
        self._min_pose_confidence = DEFAULT_MIN_POSE_CONFIDENCE
        self._split = None

    def resolved(self) -> dict:
        """How the student was run, which is how it was trained (Tenet 9)."""
        out: dict = {"student_input": self._student_input}
        if self._student_input == "crop":
            out["student_crop_px"] = self._crop_px
        return out

    def skeleton(self) -> Skeleton:
        return self._skeleton

    @staticmethod
    def introspect(weights_path: Path | str) -> tuple[int, list[int] | None]:
        """``(K, flip_idx)`` from a pose checkpoint, for the register stage and the catalog
        import: how many points the student places and which indices its flip augmentation
        swapped. Backend stays lazy (Tenet 13).

        Raises:
            ValueError: The checkpoint is not a pose model.
        """
        import_ultralytics()
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        if str(getattr(model, "task", "")) != "pose":
            raise ValueError(
                f"{weights_path} is a {getattr(model, 'task', 'unknown')!s} model, not a "
                "pose model; a keypoint student is trained from a -pose base checkpoint"
            )
        spec = getattr(model.model, "yaml", None) or {}
        kpt_shape = getattr(model.model, "kpt_shape", None) or spec.get("kpt_shape")
        if not kpt_shape:
            raise ValueError(f"{weights_path} carries no kpt_shape")
        flip = spec.get("flip_idx")
        return int(kpt_shape[0]), (list(int(i) for i in flip) if flip else None)

    def readiness(self) -> ModelReadiness:
        if not is_importable("ultralytics"):
            return ModelReadiness(
                ready=False,
                reason="missing_dependency",
                message=(
                    "Ultralytics is required to run a trained YOLO-pose model — install "
                    "it with `pip install ultralytics`."
                ),
                action_url="https://docs.ultralytics.com/",
                repo_id=None,
            )
        if not self._weights_path.is_file():
            return ModelReadiness(
                ready=False,
                reason="missing_weights",
                message=f"trained weights not found at {self._weights_path}",
                action_url=None,
                repo_id=None,
            )
        return ModelReadiness(
            ready=True, reason="ok", message="", action_url=None, repo_id=None
        )

    def load(self, device: str) -> None:
        import_ultralytics()
        from ultralytics import YOLO

        self._device = device.split(",")[0] if "," in device else device
        self._model = YOLO(str(self._weights_path))
        self._split = None

    def set_min_pose_confidence(self, threshold: float) -> None:
        """Poses below ``threshold`` box confidence are not returned (§26 *Visibility*)."""
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"min_pose_confidence {threshold} is not in [0, 1]")
        self._min_pose_confidence = threshold
        self._split = None  # rebuilt with the new threshold on the next batch

    def predict_batch(self, frames: list[InstanceFrame]) -> list[FrameResult]:
        if not frames:
            return []
        return self.finish_batch(self.run_prepared(self.prepare_batch(frames)))

    def prepare_batch(self, frames: list[InstanceFrame]) -> object:
        """Ultralytics' letterbox, on the CPU (§26 *Execution*); a crop student's crops are
        cut here too, one per detection."""
        split = self._split_predict()
        # An InstanceFrame is RGB; Ultralytics reads a numpy array as BGR, and the student was
        # trained on BGR images (cv2.imwrite), so it must see BGR here too.
        if self._student_input == "crop":
            jobs, images = self._crops(frames)
        else:
            jobs, images = None, [rgb_to_bgr(f.frame) for f in frames]
        tensor = split.prepare(images) if images else None
        return split, tensor, images, frames, jobs

    def run_prepared(self, prepared: object) -> object:
        """One batched forward pass and NMS on the device, then each image's boxes, scores
        and keypoints copied to the host."""
        split, tensor, images, frames, jobs = prepared
        results = split.run(tensor, images) if images else []
        return frames, jobs, [_host_pose_arrays(r) for r in results]

    def finish_batch(self, raw: object) -> list[FrameResult]:
        frames, jobs, arrays = raw
        if jobs is not None:
            return self._from_crops(frames, jobs, arrays)
        return [self._poses(frame, a) for frame, a in zip(frames, arrays)]

    def _split_predict(self) -> SplitPredict:
        if self._model is None:
            raise RuntimeError(
                "TrainedYoloPose.load() must be called before predict_batch()"
            )
        if self._split is None:
            self._split = SplitPredict(
                self._model,
                conf=self._min_pose_confidence,
                device=self._device,
                verbose=False,
            )
        return self._split

    def _crops(self, frames: list[InstanceFrame]) -> tuple[list, list]:
        """One BGR crop per detection, and ``(frame index, detection row, view)`` for each."""
        from detail.models.patch_match.view import make_view, render

        jobs = []  # (frame index, detection row, view)
        crops = []
        for i, frame in enumerate(frames):
            h, w = frame.frame.shape[:2]
            bgr = rgb_to_bgr(frame.frame)
            for det_row in sorted(frame.boxes):
                view = make_view(tuple(frame.boxes[det_row]), (w, h), self._crop_px, 1)
                jobs.append((i, int(det_row), view))
                crops.append(render(bgr, view))
        return jobs, crops

    def _from_crops(self, frames, jobs, arrays) -> list[FrameResult]:
        """Each detection takes the pose nearest its crop's centre, mapped back through the
        crop's view."""
        k = len(self._skeleton)
        placed: dict[int, list] = {i: [] for i in range(len(frames))}
        for (i, det_row, view), found_arrays in zip(jobs, arrays):
            h, w = frames[i].frame.shape[:2]
            found = self._from_crop(found_arrays, view, (w, h), k)
            if found is not None and found[0] >= self._min_pose_confidence:
                placed[i].append((det_row, *found))
        out = []
        for i, frame in enumerate(frames):
            rows = placed[i]
            n = len(rows)
            out.append(
                FrameResult(
                    frame_idx=int(frame.frame_idx),
                    t_ns=int(frame.t_ns),
                    n=n,
                    confidence=np.array([r[1] for r in rows], np.float32).reshape(n),
                    bbox=np.array([r[2] for r in rows], np.float32).reshape(n, 4),
                    detection_row=np.array([r[0] for r in rows], np.int32).reshape(n),
                    points=np.array([r[3] for r in rows], np.float32).reshape(n, k, 2),
                    point_scores=np.array([r[4] for r in rows], np.float32).reshape(
                        n, k
                    ),
                )
            )
        return out

    def _from_crop(self, arrays: tuple | None, view, frame_wh, k: int):
        """``(confidence, bbox, points, scores)`` in normalized frame coordinates for the
        pose nearest the crop's centre — the detection the crop was cut around — or None when
        the student found nothing, or nothing covering the centre."""
        if arrays is None:
            return None
        xyxyn, conf, xyn, kp_conf = arrays
        det_boxes = xyxyn.astype(np.float64).reshape(-1, 4)
        det_conf = conf.astype(np.float32).reshape(-1)
        xy = xyn.astype(np.float64).reshape(len(det_boxes), -1, 2)
        kp_conf = (
            kp_conf.astype(np.float32).reshape(len(det_boxes), -1)
            if kp_conf is not None
            else np.ones(xy.shape[:2], dtype=np.float32)
        )
        if xy.shape[1] != k:
            raise ValueError(
                f"the student places {xy.shape[1]} points but its skeleton names {k}"
            )
        centre = np.array([0.5, 0.5])
        covers = (
            (det_boxes[:, 0] <= 0.5)
            & (det_boxes[:, 2] >= 0.5)
            & (det_boxes[:, 1] <= 0.5)
            & (det_boxes[:, 3] >= 0.5)
        )
        if not covers.any():
            return None  # another animal reaching into the crop is not this detection
        middles = (det_boxes[:, :2] + det_boxes[:, 2:]) / 2
        distance = np.where(covers, np.linalg.norm(middles - centre, axis=1), np.inf)
        best = int(np.argmin(distance))
        w, h = frame_wh
        size = float(view.size)

        def back(u: float, v: float) -> tuple[float, float]:
            x, y = view.to_source(u * size, v * size)
            return x / w, y / h

        x0, y0 = back(*det_boxes[best, :2])
        x1, y1 = back(*det_boxes[best, 2:])
        points = np.full((k, 2), np.nan, dtype=np.float32)
        scores = np.full(k, np.nan, dtype=np.float32)
        seen = kp_conf[best] > 0.0
        for j in np.flatnonzero(seen):
            points[j] = back(*xy[best, j])
            scores[j] = kp_conf[best, j]
        return (
            float(det_conf[best]),
            np.array([x0, y0, x1, y1], dtype=np.float32),
            points,
            scores,
        )

    def _poses(self, frame: InstanceFrame, arrays: tuple | None) -> FrameResult:
        """Every pose the student returned at or above the threshold, one row each."""
        k = len(self._skeleton)
        if arrays is None:
            det_boxes = np.zeros((0, 4), np.float32)
            det_conf = np.zeros(0, np.float32)
            xy = np.zeros((0, k, 2), np.float32)
            kp_conf = np.zeros((0, k), np.float32)
        else:
            xyxyn, conf, xyn, kp_conf = arrays
            det_boxes = xyxyn.astype(np.float32).reshape(-1, 4)
            det_conf = conf.astype(np.float32).reshape(-1)
            xy = xyn.astype(np.float32).reshape(len(det_boxes), -1, 2)
            if xy.shape[1] != k:
                raise ValueError(
                    f"the student places {xy.shape[1]} points but its skeleton names {k}"
                )
            kp_conf = (
                kp_conf.astype(np.float32).reshape(len(det_boxes), -1)
                if kp_conf is not None
                else np.ones(xy.shape[:2], dtype=np.float32)
            )
        # Ultralytics already filters at ``conf``; this keeps the rule true whatever it does.
        keep = det_conf >= self._min_pose_confidence
        det_boxes, det_conf, xy, kp_conf = (
            det_boxes[keep],
            det_conf[keep],
            xy[keep],
            kp_conf[keep],
        )
        n = len(det_conf)
        placed = kp_conf > 0.0
        points = np.where(placed[..., None], xy, np.nan).astype(np.float32)
        scores = np.where(placed, kp_conf, np.nan).astype(np.float32)
        return FrameResult(
            frame_idx=int(frame.frame_idx),
            t_ns=int(frame.t_ns),
            n=n,
            confidence=det_conf,
            bbox=det_boxes,
            detection_row=np.full(n, -1, np.int32),
            points=points,
            point_scores=scores,
        )


def _host_pose_arrays(result) -> tuple | None:
    """An Ultralytics pose ``Results``' normalised boxes, box scores, normalised keypoints and
    keypoint scores (None when it reports none) as host arrays, or None when it found
    nothing."""
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or not len(boxes):
        return None
    kp_conf = keypoints.conf
    return (
        _to_numpy(boxes.xyxyn),
        _to_numpy(boxes.conf),
        _to_numpy(keypoints.xyn),
        None if kp_conf is None else _to_numpy(kp_conf),
    )
