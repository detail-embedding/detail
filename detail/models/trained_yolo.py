"""Closed-set detection with a locally-trained YOLO model (Section 12).

The apply side of custom training: load a ``best.pt`` produced by
``UltralyticsYOLOTrainer`` and detect the model's *own* trained classes — no
prompt. ``supported_cue_types = []`` marks it closed-set, so detection takes an
empty prompt and ``predict`` ignores the prompt entirely. The class names come
from the recorded dataset classes (``model.yaml``), not from the run config.

Unlike the registry-keyed pretrained models, this is constructed by the resolver
(``models/resolve.py``) with an explicit local weights path + class list, and is
registered under its ``model_family`` (``"yolo"``) via ``register_trained_loader``.
``from ultralytics import YOLO`` stays inside ``load`` (Tenet 13).
"""

from pathlib import Path

import numpy as np

from detail.core.dataclasses import (
    Detection,
    FrameResult,
    ModelReadiness,
    Prompt,
)
from detail.core.installed import is_importable
from detail.core.protocols import DetectionModel
from detail.core.registries import register_trained_loader
from detail.models._ultralytics import SplitPredict
from detail.models._ultralytics import import_ultralytics
from detail.models._ultralytics import read_checkpoint


@register_trained_loader("yolo")
class TrainedYOLO(DetectionModel):
    """Detect a trained model's own classes (closed-set; no prompt cues)."""

    model_id = "trained_yolo"  # family loader id; not a MODELS registry key
    architecture = "yolo"
    supported_cue_types = []  # closed-set: prompt must be empty
    supported_backends = ["cpu", "mps", "cuda"]
    supports_batching = True  # Ultralytics runs a list of frames in one forward pass

    def __init__(self, weights_path: Path | str, classes: list[str]) -> None:
        super().__init__(None)  # no size variants
        self._weights_path = Path(weights_path)
        self._classes = list(classes)
        self._model = None
        self._device = "cpu"
        self._conf = 0.5
        self._split = None

    @property
    def classes(self) -> list[str]:
        """The model's baked-in class names (closed-set); used to offer a
        per-class preview. Open-vocab models have no fixed class list."""
        return list(self._classes)

    @staticmethod
    def introspect(weights_path: Path | str) -> tuple[list[str], str]:
        """Read class names + task from a YOLO checkpoint, for import (the
        per-family hook the imported-model catalog calls; backend stays lazy,
        Tenet 13). Class names are baked into Ultralytics weights, so an import
        needs no metadata beyond the ``.pt`` itself.

        Returns:
            ``(classes, task)`` — ordered class names and the model task
            (``"segment"`` / ``"detect"``).

        Raises:
            FileNotFoundError: ``weights_path`` does not exist.
            UserInputError: The file is not a YOLO ``.pt`` checkpoint (``read_checkpoint``).
            ValueError: The checkpoint carries no class names.
        """
        model = read_checkpoint(weights_path)
        names = model.names  # {id: name} (or a list)
        classes = (
            [names[k] for k in sorted(names)]
            if isinstance(names, dict)
            else list(names)
        )
        if not classes:
            raise ValueError(f"no class names found in {weights_path}")
        return classes, str(getattr(model, "task", "detect"))

    def readiness(self) -> ModelReadiness:
        if not is_importable("ultralytics"):
            return ModelReadiness(
                ready=False,
                reason="missing_dependency",
                message=(
                    "Ultralytics is required to run trained YOLO models — install "
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

        # resolve_device yields "0"/"0,1" for cuda; a single model uses the first.
        self._device = device.split(",")[0] if "," in device else device
        self._model = YOLO(str(self._weights_path))
        self._split = None

    def set_conf_threshold(self, threshold: float) -> None:
        self._conf = float(threshold)
        self._split = None  # rebuilt with the new threshold on the next batch

    def predict(self, frame: np.ndarray, prompt: Prompt) -> FrameResult:
        if self._model is None:
            raise RuntimeError("TrainedYOLO.load() must be called before predict()")
        h, w = frame.shape[:2]
        # Closed-set: the prompt is ignored; the model detects its trained classes.
        results = self._model.predict(
            frame,
            conf=self._conf,
            device=self._device,
            retina_masks=True,  # full-resolution masks (matching the frame)
            verbose=False,
        )
        return FrameResult.from_detections(
            frame_idx=0,
            t_ns=0,
            detections=_detections(_host_arrays(results[0]), self._classes, w, h),
        )

    def predict_batch(
        self, frames: list[np.ndarray], prompts: list[Prompt]
    ) -> list[FrameResult]:
        if not frames:
            return []
        return self.finish_batch(self.run_prepared(self.prepare_batch(frames, prompts)))

    def prepare_batch(self, frames: list[np.ndarray], prompts: list[Prompt]) -> object:
        """Ultralytics' letterbox, on the CPU (§7 *Execution*). Closed-set: prompts are
        ignored."""
        split = self._split_predict()
        return split, split.prepare(frames), frames

    def run_prepared(self, prepared: object) -> object:
        """One batched forward pass, NMS and full-resolution masks on the device, then each
        frame's boxes, scores, classes and masks copied to the host."""
        split, tensor, frames = prepared
        results = split.run(tensor, frames)
        return [
            (_host_arrays(r), f.shape[1], f.shape[0]) for r, f in zip(results, frames)
        ]

    def finish_batch(self, raw: object) -> list[FrameResult]:
        return [
            FrameResult.from_detections(
                frame_idx=0,
                t_ns=0,
                detections=_detections(arrays, self._classes, w, h),
            )
            for arrays, w, h in raw
        ]

    def _split_predict(self):
        if self._model is None:
            raise RuntimeError(
                "TrainedYOLO.load() must be called before predict_batch()"
            )
        if self._split is None:
            self._split = SplitPredict(
                self._model,
                conf=self._conf,
                device=self._device,
                retina_masks=True,  # full-resolution masks (matching the frame)
                verbose=False,
            )
        return self._split


def _host_arrays(result) -> tuple | None:
    """An Ultralytics ``Results``' boxes, scores, classes and masks as host arrays, or None
    when it found nothing."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    return xyxy, conf, cls, masks


def _detections(arrays, class_names: list[str], w: int, h: int) -> list[Detection]:
    """``_host_arrays``' output as normalized ``Detection``s."""
    if arrays is None:
        return []
    xyxy, conf, cls, masks = arrays
    detections = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        class_index = int(cls[i])
        name = (
            class_names[class_index]
            if class_index < len(class_names)
            else str(class_index)
        )
        detections.append(
            Detection(
                mask=(masks[i] > 0.5) if masks is not None else None,
                bbox=(x1 / w, y1 / h, x2 / w, y2 / h),
                class_name=name,
                confidence=float(conf[i]),
                object_id=None,
            )
        )
    return detections
