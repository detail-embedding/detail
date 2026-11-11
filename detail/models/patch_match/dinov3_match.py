"""``dinov3_match``: ``PatchMatch`` on ``dinov3_vit`` (§26 "Patch matching").

The weights are DINOv3 ViT's (same repos, same cache, same Meta gate): this method installs
nothing of its own and declares ``shares_weights_with``, so the Model manager lists the
backbone once, as *embedding · patch features*, rather than a second row. The backbone and its
variant are identity, since the model id no longer says which weights ran.
"""

from __future__ import annotations

from detail.core.registries import register_keypoint_model
from detail.models.patch_match.engine import PatchMatch


@register_keypoint_model("dinov3_match")
class DINOv3Match(PatchMatch):
    """Zero-training keypoints: DINOv3 patch features matched to a few marked example frames."""

    model_id = "dinov3_match"
    gated = True
    hf_gate = "dinov3"
    shares_weights_with = "dinov3_vit"
    # Variant name -> the dinov3_vit variant it loads: the two sizes the research measured.
    # ViT-L is the default; the matching constants were set on ViT-B (§26 *Open questions* 5).
    VARIANTS = {"vit_large": "vit_large", "vit_base": "vit_base"}
    DEFAULT_VARIANT = "vit_large"

    @classmethod
    def backbone_class(cls) -> type:
        from detail.models.dinov3 import DINOv3ViT

        return DINOv3ViT

    @property
    def hf_repo(self) -> str:
        return self._make_backbone().hf_repo
