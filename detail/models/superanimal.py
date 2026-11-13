"""DeepLabCut SuperAnimal pose models, run natively — no ``deeplabcut`` dependency (§26).

A SuperAnimal HRNet-W32 checkpoint is a stock ``timm`` ``hrnet_w32`` (``features_only``,
branches 1-4, ``feature_location=""``) plus one 1x1 transposed-convolution heatmap head;
everything else is pre- and post-processing. This module re-implements those few lines: crop
the subject's box to a 256x256 square **with context** (deeplabcut's ``top_down_crop``, margin
0), normalize with ImageNet statistics, take each heatmap's argmax at stride 4 and map it back
through the crop's scale and offset. That reproduces deeplabcut 3.0.2 exactly on the CPU
(``tests/models/test_superanimal.py`` holds it to a fixture of deeplabcut's own outputs).

The model is top-down and is handed the **tracked subject's box** — its detector is never used,
so it inherits DETAIL's tracking, the same input every keypoint method gets.

Weights come from DETAIL's own ``install()`` (``HFHubModel``) into ``model_cache_dir()`` and are
loaded by path with ``weights_only=True``; deeplabcut's downloader, which writes into its own
package directory, is never used. ``torch`` and ``timm`` are imported inside ``load`` /
``predict_batch`` only (Tenet 13).

The keypoint names are the models' own vocabularies (the checkpoints carry none; these come from
deeplabcut's project configs) and are never mapped onto a user's names. The left/right pairs are
declared here because nothing else can know them; a heading is the user's to define (§14).

Ported from the pre-§26 keypoints stage onto the ``InstanceFrame`` protocol: the model places
one subject per tracked box (``_per_subject.place_subjects``), one point per keypoint
(``n_candidates = 1``), its score the heatmap peak.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from detail.core.dataclasses import FrameResult, ModelLicense, ModelReadiness
from detail.core.installed import is_importable
from detail.core.keypoints import ScoreSpec, InstanceFrame
from detail.core.protocols import KeypointModel
from detail.core.registries import register_keypoint_model
from detail.core.skeleton import Skeleton
from detail.core.utils import torch_device
from detail.models._per_subject import place_subjects
from detail.models.hf_access import HFHubModel

INPUT_PX = 256  # deeplabcut's top-down crop size; the SuperAnimal configs set none
STRIDE = 4  # HRNet's high-resolution branch, kept by the 1x1 head
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_HEAD_PREFIX = "heads.bodypart.heatmap_head.deconv_layers.0."
_BACKBONE_PREFIX = "backbone.model."

QUADRUPED_REPO = "mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped"
TOPVIEWMOUSE_REPO = "mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse"


def _license(repo: str) -> ModelLicense:
    """The terms on each SuperAnimal model card (identical text, one card per repo)."""
    return ModelLicense(
        name="Modified MIT (academic, non-commercial)",
        summary=(
            "SuperAnimal weights are released for academic, non-commercial research only "
            "and may not be used to harm animals; commercial use needs a license from EPFL."
        ),
        url=f"https://huggingface.co/{repo}",
    )


def _pairs(*names: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    """The ``(left, right)`` pairs, as written."""
    return tuple(names)


def _skeleton(
    names: Sequence[str],
    pairs: tuple[tuple[str, str], ...],
    edges: Sequence[tuple[str, str]],
) -> Skeleton:
    return Skeleton(names=tuple(names), edges=tuple(edges), pairs=pairs)


# deeplabcut/modelzoo/project_configs/superanimal_quadruped.yaml, in order.
QUADRUPED_SKELETON = _skeleton(
    (
        "nose upper_jaw lower_jaw mouth_end_right mouth_end_left right_eye right_earbase "
        "right_earend right_antler_base right_antler_end left_eye left_earbase left_earend "
        "left_antler_base left_antler_end neck_base neck_end throat_base throat_end back_base "
        "back_end back_middle tail_base tail_end front_left_thai front_left_knee front_left_paw "
        "front_right_thai front_right_knee front_right_paw back_left_paw back_left_thai "
        "back_right_thai back_left_knee back_right_knee back_right_paw belly_bottom "
        "body_middle_right body_middle_left"
    ).split(),
    _pairs(
        ("mouth_end_left", "mouth_end_right"),
        ("left_eye", "right_eye"),
        ("left_earbase", "right_earbase"),
        ("left_earend", "right_earend"),
        ("left_antler_base", "right_antler_base"),
        ("left_antler_end", "right_antler_end"),
        ("front_left_thai", "front_right_thai"),
        ("front_left_knee", "front_right_knee"),
        ("front_left_paw", "front_right_paw"),
        ("back_left_thai", "back_right_thai"),
        ("back_left_knee", "back_right_knee"),
        ("back_left_paw", "back_right_paw"),
        ("body_middle_left", "body_middle_right"),
    ),
    (
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_earbase"),
        ("right_eye", "right_earbase"),
        ("left_earbase", "left_earend"),
        ("right_earbase", "right_earend"),
        ("nose", "neck_base"),
        ("neck_base", "back_middle"),
        ("back_middle", "tail_base"),
        ("tail_base", "tail_end"),
        ("neck_base", "front_left_thai"),
        ("neck_base", "front_right_thai"),
        ("front_left_thai", "front_left_knee"),
        ("front_left_knee", "front_left_paw"),
        ("front_right_thai", "front_right_knee"),
        ("front_right_knee", "front_right_paw"),
        ("tail_base", "back_left_thai"),
        ("tail_base", "back_right_thai"),
        ("back_left_thai", "back_left_knee"),
        ("back_left_knee", "back_left_paw"),
        ("back_right_thai", "back_right_knee"),
        ("back_right_knee", "back_right_paw"),
    ),
)

# deeplabcut/modelzoo/project_configs/superanimal_topviewmouse.yaml, in order.
TOPVIEWMOUSE_SKELETON = _skeleton(
    (
        "nose left_ear right_ear left_ear_tip right_ear_tip left_eye right_eye neck mid_back "
        "mouse_center mid_backend mid_backend2 mid_backend3 tail_base tail1 tail2 tail3 tail4 "
        "tail5 left_shoulder left_midside left_hip right_shoulder right_midside right_hip "
        "tail_end head_midpoint"
    ).split(),
    _pairs(
        ("left_ear", "right_ear"),
        ("left_ear_tip", "right_ear_tip"),
        ("left_eye", "right_eye"),
        ("left_shoulder", "right_shoulder"),
        ("left_midside", "right_midside"),
        ("left_hip", "right_hip"),
    ),
    (
        ("nose", "head_midpoint"),
        ("head_midpoint", "neck"),
        ("head_midpoint", "left_ear"),
        ("head_midpoint", "right_ear"),
        ("left_ear", "left_ear_tip"),
        ("right_ear", "right_ear_tip"),
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("neck", "mid_back"),
        ("mid_back", "mouse_center"),
        ("mouse_center", "mid_backend"),
        ("mid_backend", "mid_backend2"),
        ("mid_backend2", "mid_backend3"),
        ("mid_backend3", "tail_base"),
        ("tail_base", "tail1"),
        ("tail1", "tail2"),
        ("tail2", "tail3"),
        ("tail3", "tail4"),
        ("tail4", "tail5"),
        ("tail5", "tail_end"),
        ("neck", "left_shoulder"),
        ("neck", "right_shoulder"),
        ("mouse_center", "left_midside"),
        ("mouse_center", "right_midside"),
        ("tail_base", "left_hip"),
        ("tail_base", "right_hip"),
    ),
)


def crop_with_context(
    image: np.ndarray, box_xywh: tuple[float, float, float, float]
) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    """deeplabcut's ``top_down_crop`` (margin 0, with context) into a square ``INPUT_PX``.

    The box is widened to a square around its centre — context, not padding — and any part
    outside the frame is zero-filled. Returns the crop, the ``(x, y)`` offset and the ``(sx, sy)``
    scale that map a crop pixel back: ``x_frame = x_crop * sx + offset_x``.
    """
    h_img, w_img = image.shape[:2]
    x, y, w, h = box_xywh
    cx, cy = x + w / 2, y + h / 2
    w = h = max(w, h)
    x1, y1 = int(round(cx - w / 2)), int(round(cy - h / 2))
    x2, y2 = int(round(cx + w / 2)), int(round(cy + h / 2))
    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right, pad_bottom = max(0, x2 - w_img), max(0, y2 - h_img)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w_img), min(y2, h_img)
    crop_w, crop_h = x2 - x1, y2 - y1
    canvas = np.zeros(
        (crop_h + pad_top + pad_bottom, crop_w + pad_left + pad_right, image.shape[2]),
        dtype=image.dtype,
    )
    canvas[pad_top : pad_top + crop_h, pad_left : pad_left + crop_w] = image[
        y1:y2, x1:x2
    ]
    out = cv2.resize(canvas, (INPUT_PX, INPUT_PX), interpolation=cv2.INTER_LINEAR)
    scale = (canvas.shape[1] / INPUT_PX, canvas.shape[0] / INPUT_PX)
    return out, (x1 - pad_left, y1 - pad_top), scale


def decode_heatmaps(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(n, K, H, W)`` heatmaps -> crop-pixel ``(n, K, 2)`` positions and ``(n, K)`` scores.

    Each keypoint is its heatmap's argmax, placed at the centre of that stride-4 cell (no
    location refinement — the SuperAnimal HRNet configs disable it); the score is the peak
    value clipped to [0, 1].
    """
    n, k, hh, ww = heatmaps.shape
    flat = heatmaps.reshape(n, k, hh * ww)
    idx = flat.argmax(axis=2)
    score = np.clip(np.take_along_axis(flat, idx[..., None], axis=2)[..., 0], 0.0, 1.0)
    xs = (idx % ww).astype(np.float32) * STRIDE + STRIDE / 2
    ys = (idx // ww).astype(np.float32) * STRIDE + STRIDE / 2
    return np.stack([xs, ys], axis=2), score.astype(np.float32)


class _SuperAnimalHRNet(HFHubModel, KeypointModel):
    """Shared native implementation; a subclass names its repo, file stem and skeleton."""

    gated = False  # the mwmathis repos are public
    supported_backends = ["cuda", "mps", "cpu"]
    architecture = "hrnet"
    per_detection = True  # top-down: one instance per detection's box
    n_candidates = 1
    # deeplabcut's own default ``pcutoff`` (its project config template): below it a point is
    # drawn as not visible there, so it is the starting cut-off here too.
    score_spec = ScoreSpec("heatmap peak", 0.0, 1.0, 0.6)
    default_track_method = "despike_smooth"
    _skeleton_def: Skeleton

    def __init__(self, variant: str | None = None) -> None:
        super().__init__(
            variant
        )  # resolves self.variant + self.weight (the weights file)
        self.hf_files = [
            self.weight
        ]  # the manifest is_installed verifies and install fetches
        self._backbone = None
        self._head = None
        self._device = "cpu"

    def skeleton(self) -> Skeleton:
        return self._skeleton_def

    def readiness(self) -> ModelReadiness:
        for module in ("torch", "timm"):
            if not is_importable(module):
                return ModelReadiness(
                    ready=False,
                    reason="missing_dependency",
                    message=(
                        f"{self.model_id} needs PyTorch and timm — install them with "
                        "`pip install torch timm`."
                    ),
                    action_url=None,
                    repo_id=self.hf_repo,
                )
        return super().readiness()

    def load(self, device: str) -> None:
        import timm
        import torch

        # local_file raises rather than downloading: install() is the only downloader (§5).
        state = torch.load(
            self.local_file(self.weight), map_location="cpu", weights_only=True
        )["model"]
        # feature_location="" as deeplabcut passes it: timm's own default adds "incre" modules
        # the checkpoint does not have.
        backbone = timm.create_model(
            "hrnet_w32",
            pretrained=False,
            features_only=True,
            feature_location="",
            out_indices=(1, 2, 3, 4),
        )
        backbone.load_state_dict(
            {
                k.removeprefix(_BACKBONE_PREFIX): v
                for k, v in state.items()
                if k.startswith(_BACKBONE_PREFIX)
            },
            strict=True,
        )
        head = torch.nn.ConvTranspose2d(
            32, len(self._skeleton_def), kernel_size=1, stride=1
        )
        head.load_state_dict(
            {
                k.removeprefix(_HEAD_PREFIX): v
                for k, v in state.items()
                if k.startswith(_HEAD_PREFIX)
            },
            strict=True,
        )
        self._device = torch_device(device)
        self._backbone = backbone.to(self._device).eval()
        self._head = head.to(self._device).eval()

    def predict_batch(self, frames: list[InstanceFrame]) -> list[FrameResult]:
        if self._backbone is None or self._head is None:
            raise RuntimeError(f"{type(self).__name__}.load() must run before predict")
        return place_subjects(frames, len(self._skeleton_def), self._place)

    def _place(self, subjects) -> list[tuple[np.ndarray, np.ndarray]]:
        """Each ``(frame, box)``'s points, from one batched forward pass."""
        import torch

        images, offsets, scales, sizes = [], [], [], []
        for frame, bbox in subjects:
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = bbox
            box = (x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h)
            image, offset, scale = crop_with_context(frame, box)
            images.append(image)
            offsets.append(offset)
            scales.append(scale)
            sizes.append((w, h))
        x = (np.stack(images).astype(np.float32) / 255.0 - _MEAN) / _STD
        tensor = torch.from_numpy(x).permute(0, 3, 1, 2).to(self._device)
        with torch.inference_mode():
            heat = self._head(self._backbone(tensor)[0]).float().cpu().numpy()
        xy_crop, score = decode_heatmaps(heat)
        results = []
        for i in range(len(subjects)):
            (ox, oy), (sx, sy), (w, h) = offsets[i], scales[i], sizes[i]
            xy = np.empty_like(xy_crop[i])
            xy[:, 0] = (xy_crop[i][:, 0] * sx + ox) / w
            xy[:, 1] = (xy_crop[i][:, 1] * sy + oy) / h
            results.append((xy.astype(np.float32), score[i]))
        return results


@register_keypoint_model("superanimal_quadruped")
class SuperAnimalQuadruped(_SuperAnimalHRNet):
    """DeepLabCut SuperAnimal-Quadruped: 39 keypoints, side-view quadrupeds."""

    model_id = "superanimal_quadruped"
    hf_repo = QUADRUPED_REPO
    license = _license(QUADRUPED_REPO)
    VARIANTS = {"hrnet_w32": "superanimal_quadruped_hrnet_w32.pt"}
    DEFAULT_VARIANT = "hrnet_w32"
    _skeleton_def = QUADRUPED_SKELETON


@register_keypoint_model("superanimal_topviewmouse")
class SuperAnimalTopViewMouse(_SuperAnimalHRNet):
    """DeepLabCut SuperAnimal-TopViewMouse: 27 keypoints, mice filmed from above."""

    model_id = "superanimal_topviewmouse"
    hf_repo = TOPVIEWMOUSE_REPO
    license = _license(TOPVIEWMOUSE_REPO)
    VARIANTS = {"hrnet_w32": "superanimal_topviewmouse_hrnet_w32.pt"}
    DEFAULT_VARIANT = "hrnet_w32"
    _skeleton_def = TOPVIEWMOUSE_SKELETON
