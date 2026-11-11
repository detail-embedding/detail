"""Auto-derived documentation metadata for the input-image constructors (§14).

The **Input images** docs page is generated from this module, so the constructor registry is the
single source: each recipe's label comes from :data:`CONSTRUCTORS`, its description from the build
function's docstring, and its render steps from calling ``build()``. Add or change a constructor
and the docs update with no manual edit. Rendering the example images (which needs a sample frame)
is done separately in ``scripts/gen_docs_input_images.py``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from detail.transforms.constructors import CONSTRUCTORS, DEFAULT_CONSTRUCTOR

# A canvas large enough for the "centered, true-scale" recipes to place a subject mask when the
# docs render each example image. Docs-only — real runs size this from the data (§14).
DOC_CANVAS_PX = 774

# A placeholder max-contrast background for recipes that need one (real runs compute it from the
# animal's mean color; §14). Blue, the value it resolves to for a dark animal, in RGB — the order
# the renderer paints in and the docs images are saved from.
DOC_BG_COLOR = [0, 0, 255]

# Transform steps that read the subject mask (vs. keeping the full background).
_MASK_STEPS = frozenset({"init_binary_mask", "apply_mask"})


@dataclass(frozen=True)
class ConstructorDoc:
    """One input-image recipe, described from code for the docs."""

    key: str
    label: str
    description: str
    steps: tuple[str, ...]
    uses_mask: bool
    is_default: bool


def describe_constructors(resize: int = 256) -> list[ConstructorDoc]:
    """Doc metadata for every registered constructor, derived from ``CONSTRUCTORS`` + docstrings.

    ``resize`` is the final square size baked into each recipe's step list when it is materialized;
    it does not change which step *types* a recipe uses.
    """
    docs: list[ConstructorDoc] = []
    for key, recipe in CONSTRUCTORS.items():
        build_kwargs: dict = {
            "canvas_px": DOC_CANVAS_PX if recipe.needs_canvas else None,
            "resize": resize,
        }
        if recipe.needs_bg_color:
            build_kwargs["bg_color"] = DOC_BG_COLOR
        spec = recipe.build(**build_kwargs)
        steps = tuple(step["type"] for step in spec)
        docs.append(
            ConstructorDoc(
                key=key,
                label=recipe.label,
                description=" ".join((inspect.getdoc(recipe.build) or "").split()),
                steps=steps,
                uses_mask=any(step in _MASK_STEPS for step in steps),
                is_default=key == DEFAULT_CONSTRUCTOR,
            )
        )
    return docs
