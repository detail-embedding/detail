"""Meta SAM 3 image model — text-prompt (concept) segmentation (Section 5).

Weights are gated on Hugging Face (``facebook/sam3``); readiness and install come
from ``HFHubModel``. The backend (``sam3`` + ``torch`` + ``PIL``) is imported
lazily inside ``load``/``predict`` (Tenet 13), so importing this module pulls in
no backend and registration works even when SAM3 is not installed.

SAM3 is installed from source (it is not on PyPI under Meta — the PyPI ``sam3`` is
a third party):
    git clone https://github.com/facebookresearch/sam3 && cd sam3 && pip install -e .

Requires Python ≥3.12 and PyTorch ≥2.7, and officially targets CUDA — MPS/CPU are
unverified by Meta.
"""

import contextlib
import os

import numpy as np

from detail.core.dataclasses import (
    Detection,
    FrameResult,
    ModelReadiness,
    Prompt,
    TextCue,
)
from detail.core.installed import is_importable
from detail.core.protocols import DetectionModel
from detail.core.registries import register_model
from detail.core.utils import torch_device
from detail.core.visualize import bgr_to_rgb
from detail.models.hf_access import HFHubModel


@register_model("sam3")
class SAM3(HFHubModel, DetectionModel):
    """Detect every instance of each prompt target's text concept in a frame."""

    model_id = "sam3"
    architecture = "sam"
    hf_repo = "facebook/sam3"  # newer checkpoint: "facebook/sam3.1"
    gated = True
    # The builder's own manifest (download_ckpt_from_hf fetches exactly these two):
    # what is_installed() verifies and what install() downloads. config.json is
    # unused by the builder — it fetches and discards it — but stays here so our
    # snapshot mirrors upstream's.
    hf_files = ["config.json", "sam3.pt"]
    supported_cue_types = [TextCue]
    supported_backends = ["cuda"]  # SAM3 needs an NVIDIA GPU; no CPU/MPS path

    def __init__(self, variant: str | None = None) -> None:
        super().__init__(
            variant
        )  # no size variants yet (build_sam3_image_model is fixed)
        self._processor = None
        self._device = "cpu"
        # Each prompt text's encoded features, per loaded model (see ``_ground``).
        self._text_features: dict[str, dict] = {}

    def readiness(self) -> ModelReadiness:
        """The sam3 backend package first, then HFHubModel's cache-first weight check.

        ``load`` imports ``sam3`` lazily; a machine with the weights cached but the ``sam3``
        package not installed would otherwise pass the pre-flight and fail deep in the pool
        worker with a bare ``ModuleNotFoundError``. Surface it here (``find_spec`` does not
        import the backend, so Tenet 13 holds) — the same pattern DINOv3 uses. On Windows
        the backend also needs Triton.
        """
        needed = ("sam3", "triton") if os.name == "nt" else ("sam3",)
        missing = [m for m in needed if not is_importable(m)]
        if missing:
            return ModelReadiness(
                ready=False,
                reason="missing_dependency",
                message=(
                    f"SAM 3 needs its backend package ({', '.join(missing)}): clone the "
                    "repo and `pip install -e ../sam3` (see the SAM 3 backend setup)."
                ),
                action_url="https://github.com/facebookresearch/sam3",
                repo_id=self.hf_repo,
            )
        return super().readiness()

    def load(self, device: str) -> None:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # Hand the builder an explicit checkpoint path from our own cache. Its
        # download_ckpt_from_hf() takes no cache_dir and would fetch a second copy
        # into the default HF cache, but it is guarded by `if load_from_HF and
        # checkpoint_path is None` — passing the path skips the fetch entirely, and
        # model_builder is the only network call in the sam3 package. An env var
        # cannot do this: huggingface_hub binds HF_HUB_CACHE at import, and
        # readiness() (and sam3's own module import) have already imported it.
        checkpoint_path = str(self.local_file("sam3.pt"))
        self._device = torch_device(device)
        # Set the current device BEFORE building: the decoder caches a coord grid
        # in __init__ with device="cuda" (the current device) as a plain attribute
        # that model.to() does not move — so on a non-zero GPU it would be stuck on
        # cuda:0 while the weights move to cuda:1.
        if self._device.startswith("cuda"):
            torch.cuda.set_device(torch.device(self._device))
        # Keyword: the positional order is (bpe_path, device, eval_mode, checkpoint_path).
        model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        model.to(self._device)
        model.eval()
        # Pin the processor to the same device as the model: Sam3Processor
        # defaults device="cuda" (i.e. cuda:0) and puts every input tensor there,
        # so on a non-zero device (e.g. --device 1, the inline path) inputs would
        # land on cuda:0 while the weights are on cuda:1 — a device mismatch.
        self._processor = Sam3Processor(model, device=self._device)
        self._text_features = {}

    def set_conf_threshold(self, threshold: float) -> None:
        """Forward the run's threshold to SAM3's internal score filter.

        SAM3 filters proposals against ``Sam3Processor.confidence_threshold``
        (default 0.5) before returning them. Without this call our caller-side
        threshold has no effect below 0.5 because SAM3 already dropped those.
        """
        if self._processor is None:
            raise RuntimeError("SAM3.load() must be called before set_conf_threshold()")
        self._processor.set_confidence_threshold(threshold)

    def predict(self, frame: np.ndarray, prompt: Prompt) -> FrameResult:
        import torch
        from PIL import Image

        if self._processor is None:
            raise RuntimeError("SAM3.load() must be called before predict()")
        # SAM3 allocates some tensors on the *current* CUDA device (e.g. the
        # decoder's coord grid uses device="cuda"). Inference runs in a worker
        # thread whose current device defaults to cuda:0, so pin it to the model's
        # device — otherwise those tensors mismatch the weights on a non-zero GPU.
        if self._device.startswith("cuda"):
            torch.cuda.set_device(torch.device(self._device))
        h, w = frame.shape[:2]
        # cv2 frames are BGR; SAM3 takes an RGB PIL image.
        pil = Image.fromarray(bgr_to_rgb(frame))
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._device.startswith("cuda")
            else contextlib.nullcontext()
        )
        detections: list[Detection] = []
        with torch.inference_mode(), autocast:
            state = self._processor.set_image(pil)
            for target in prompt.targets:
                if not isinstance(target.cue, TextCue):
                    raise TypeError(
                        f"SAM3 only accepts TextCue, got {type(target.cue).__name__}"
                    )
                out = self._ground(state, target.cue.text)
                detections.extend(_to_detections(out, target.class_name, w, h))
        # frame_idx / t_ns are assigned by the detection runner, not the model.
        return FrameResult.from_detections(frame_idx=0, t_ns=0, detections=detections)

    def predict_batch(
        self, frames: list[np.ndarray], prompts: list[Prompt]
    ) -> list[FrameResult]:
        return [self.predict(f, p) for f, p in zip(frames, prompts)]

    def _ground(self, state: dict, text: str) -> dict:
        """``Sam3Processor.set_text_prompt`` with the text encoded once per loaded model.

        The upstream method runs ``backbone.forward_text`` on every frame, and its output is
        the same every time; SAM3's own video model
        caches it the same way. Otherwise these are that method's steps, in order —
        ``tests/models/test_sam3.py`` fails if the upstream method changes. Called inside
        ``predict``'s ``inference_mode`` and autocast, the contexts its decorator gives it.
        """
        proc = self._processor
        features = self._text_features.get(text)
        if features is None:
            features = proc.model.backbone.forward_text([text], device=proc.device)
            self._text_features[text] = features
        state["backbone_out"].update(features)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = proc.model._get_dummy_prompt()
        return proc._forward_grounding(state)


def _to_detections(out: dict, class_name: str, w: int, h: int) -> list[Detection]:
    scores = _to_numpy(out["scores"])
    masks, boxes = out["masks"], out["boxes"]
    return [
        Detection(
            mask=_to_bool_mask(masks[i]),
            bbox=_to_xyxy_norm(boxes[i], w, h),
            class_name=class_name,
            confidence=float(scores[i]),
            object_id=None,
        )
        for i in range(len(scores))
    ]


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _to_bool_mask(m) -> np.ndarray:
    arr = np.squeeze(_to_numpy(m))
    return arr if arr.dtype == bool else arr > 0.5


def _to_xyxy_norm(box, w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in _to_numpy(box)[:4])
    return (x1 / w, y1 / h, x2 / w, y2 / h)
