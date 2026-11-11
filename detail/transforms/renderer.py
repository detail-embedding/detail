"""MaskRenderer — compose MaskTransforms into a reproducible image pipeline (Section 14).

Fully serializable via ``to_spec`` / ``from_spec`` so every embedding run records
exactly what produced its inputs. Transforms are looked up by name in the
TRANSFORMS registry, so the transform modules must be imported first.
"""

import numpy as np

from detail.core.dataclasses import RenderContext
from detail.core.protocols import MaskTransform
from detail.core.registries import TRANSFORMS


class MaskRenderer:
    def __init__(self, transforms: list[MaskTransform]) -> None:
        self.transforms = list(transforms)

    def render(self, ctx: RenderContext) -> np.ndarray:
        for transform in self.transforms:
            ctx = transform.apply(ctx)
        return ctx.image

    @property
    def needs_frame(self) -> bool:
        """True if any transform requires ``ctx.frame`` (the decoded pixels).

        When False — the silhouette constructors, which build from ``ctx.mask``
        alone — the embedding worker skips video decode entirely and passes
        ``frame=None`` (§14). Only ``apply_mask`` sets ``reads_frame``.
        """
        return any(t.reads_frame for t in self.transforms)

    @classmethod
    def from_spec(cls, spec: list[dict]) -> "MaskRenderer":
        return cls(
            [
                TRANSFORMS[step["type"]](
                    **{k: v for k, v in step.items() if k != "type"}
                )
                for step in spec
            ]
        )

    def to_spec(self) -> list[dict]:
        return [transform.to_spec() for transform in self.transforms]
