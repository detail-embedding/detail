"""Named input-image constructors (Section 14).

A *constructor* is a named ``MaskRenderer`` pipeline that turns a detection's
(frame, mask, bbox) into the square image fed to the embedding model. The GUI
offers these by name with no further controls; the builder functions take
explicit parameters so the API can drive finer experiments.

Constructors flagged ``needs_canvas`` depend on a dataset statistic — the size
of the largest mask across the tracks being embedded — so every mask can be
centered, at true scale, in a shared canvas. The caller computes that size and
passes it as ``canvas_px``; resolution lives in the API layer (which reads the
detection zarr), not here, to keep this module storage-free.

**A saved run does not need its recipe to be here.** An embedding run records the resolved
step list (``renderer_spec``), not only the recipe's name, and it is rendered from those
steps. Six recipes were removed from this table (``docs/removed_capabilities.md``); the
steps they were built from all stay registered in ``transforms/basic.py`` — including
``fill_background_noise`` and ``fill_background_bands``, which no recipe here uses any
more — so a run made with one still previews its input image. Removing a step type is what
would break a saved run, and is a different change from removing a recipe.
"""

from collections.abc import Callable
from dataclasses import dataclass


def silhouette_centered(
    canvas_px: int | None = None, resize: int | None = None
) -> list[dict]:
    """Cropped white silhouette, centered at true scale in the shared canvas."""
    if canvas_px is None:
        raise ValueError("silhouette_centered requires canvas_px (largest mask size)")
    # crop-first: crop the mask to the subject before building the binary image, so
    # the render never materializes a full-resolution silhouette just to crop it (§14).
    spec: list[dict] = [
        {"type": "crop_to_bbox"},
        {"type": "init_binary_mask"},
        {"type": "center_in_square", "size": canvas_px},
    ]
    if resize is not None:
        spec.append({"type": "resize", "size": resize})
    return spec


def masked_color_centered(
    canvas_px: int | None = None, resize: int | None = None
) -> list[dict]:
    """Cropped full-color animal (pixels under the mask), centered at true scale."""
    if canvas_px is None:
        raise ValueError("masked_color_centered requires canvas_px (largest mask size)")
    # crop-first: crop the frame + mask to the subject before applying the mask, so
    # the render works on the subject region, not the whole decoded frame (§14).
    spec: list[dict] = [
        {"type": "crop_to_bbox"},
        {"type": "apply_mask", "background": 0},
        {"type": "center_in_square", "size": canvas_px},
    ]
    if resize is not None:
        spec.append({"type": "resize", "size": resize})
    return spec


def masked_color_centered_contrast(
    canvas_px: int | None = None,
    resize: int | None = None,
    bg_color: list[int] | None = None,
) -> list[dict]:
    """``masked_color_centered`` on the mathematically max-contrast background for this animal.

    ``bg_color`` (``[R, G, B]``) is the in-gamut color that maximizes CIELAB distance from the
    animal's mean masked-pixel color, computed once per run from the data. It fills both the
    masked-out region and the canvas padding, so the subject sits on the single most separable
    solid background its own coloring allows. For a dark animal this resolves to blue.
    """
    if canvas_px is None:
        raise ValueError(
            "masked_color_centered_contrast requires canvas_px (largest mask size)"
        )
    if bg_color is None:
        raise ValueError(
            "masked_color_centered_contrast requires bg_color (computed max-contrast color)"
        )
    spec: list[dict] = [
        {"type": "crop_to_bbox"},
        {"type": "apply_mask", "background": bg_color},
        {"type": "center_in_square", "size": canvas_px, "fill": bg_color},
    ]
    if resize is not None:
        spec.append({"type": "resize", "size": resize})
    return spec


def full_frame(canvas_px: int | None = None, resize: int | None = None) -> list[dict]:
    """The whole decoded frame, with no localization and no mask (the scene baseline)."""
    spec: list[dict] = [{"type": "init_frame"}, {"type": "pad_to_square"}]
    if resize is not None:
        spec.append({"type": "resize", "size": resize})
    return spec


@dataclass(frozen=True)
class Constructor:
    """A named, GUI-selectable input-image recipe."""

    key: str
    label: str
    needs_canvas: bool
    build: Callable[..., list[dict]]
    # True when ``build`` needs a ``bg_color`` computed from the data (the max-contrast
    # background), so the caller resolves that statistic before building the spec.
    needs_bg_color: bool = False


CONSTRUCTORS: dict[str, Constructor] = {
    "silhouette_centered": Constructor(
        "silhouette_centered",
        "Silhouette (cropped, true scale)",
        True,
        silhouette_centered,
    ),
    "masked_color_centered": Constructor(
        "masked_color_centered",
        "Color (cropped, true scale)",
        True,
        masked_color_centered,
    ),
    "masked_color_centered_contrast": Constructor(
        "masked_color_centered_contrast",
        "Color on max-contrast bg (cropped, true scale)",
        True,
        masked_color_centered_contrast,
        needs_bg_color=True,
    ),
    "full_frame": Constructor("full_frame", "Full frame (no mask)", False, full_frame),
}

# Max-contrast, the measured middle of a sweep over input images and the one that keeps the
# mask: full frame scored higher there, but on one experiment, and a whole-frame image can
# group recordings by rig as readily as by behavior.
DEFAULT_CONSTRUCTOR = "masked_color_centered_contrast"


def surfaced_constructors() -> list[str]:
    """The constructor keys this build offers, registry order kept (§25).

    For the places a user *picks* one — the GUI picker, the CLI's ``--constructor`` choices,
    the generated Input images page. Never for the lookup that resolves one.
    """
    from detail import surfaces

    return surfaces.surfaced(list(CONSTRUCTORS))


# --------------------------------------------------------------------------- #
# §14 Egocentric alignment
# --------------------------------------------------------------------------- #
#: The recipes alignment applies to: the subject is cropped, so a turn re-crops to the
#: turned mask and the centred canvas places it as it places every subject.
ALIGNABLE = (
    "masked_color_centered_contrast",
    "masked_color_centered",
    "silhouette_centered",
)

_ALIGN_STEPS = {"rotate": "align_rotate", "mirror": "align_mirror"}
_BEFORE = ("center_in_square", "pad_to_square", "resize")


def alignment_step(spec: list[dict]) -> str | None:
    """Which alignment a renderer spec carries: ``rotate``, ``mirror`` or ``None``."""
    steps = {step["type"] for step in spec}
    found = [mode for mode, step in _ALIGN_STEPS.items() if step in steps]
    return found[0] if found else None


def with_alignment(spec: list[dict], align: str) -> list[dict]:
    """Insert the alignment step after the subject image is built and cropped, before the
    first ``center_in_square`` / ``pad_to_square`` / ``resize`` (a resize to a square is
    anisotropic on a non-square crop and would bend the axis). ``none`` returns the spec
    unchanged. The turn's exposed corners take the recipe's canvas fill.

    Raises:
        ValueError: ``align`` is not a mode, or the spec never crops to the subject.
    """
    if align == "none":
        return [dict(step) for step in spec]
    if align not in _ALIGN_STEPS:
        raise ValueError(f"align must be one of none, rotate, mirror; got {align!r}")
    types = [step["type"] for step in spec]
    if "crop_to_bbox" not in types:
        raise ValueError(
            "alignment needs an input image that crops to the subject; use one of "
            + ", ".join(ALIGNABLE)
        )
    at = next((i for i, t in enumerate(types) if t in _BEFORE), len(types))
    step: dict = {"type": _ALIGN_STEPS[align]}
    if align == "rotate":
        canvas = next((s for s in spec if s["type"] == "center_in_square"), None)
        if canvas is not None and canvas.get("fill", 0) != 0:
            step["fill"] = canvas["fill"]
    out = [dict(s) for s in spec]
    out.insert(at, step)
    return out
