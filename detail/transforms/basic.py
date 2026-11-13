"""First MaskTransform implementations (Section 14).

Each transform takes a RenderContext, does one operation, updates ``ctx.image``,
and returns the context. They are dataclasses (picklable for DataLoader workers)
and self-register on import. cv2/numpy are core dependencies, not lazy backends.
"""

from dataclasses import dataclass
from typing import ClassVar

import cv2
import numpy as np

from detail.core.dataclasses import RenderContext
from detail.core.protocols import MaskTransform
from detail.core.registries import register_transform
from detail.core.visualize import draw_keypoints


@register_transform("init_binary_mask")
@dataclass
class InitBinaryMask(MaskTransform):
    """White mask on black background, sized to the mask (not the source frame).

    The detection mask is the rendered image here, so the output resolution
    must come from ``ctx.mask`` — when ``ctx.frame`` is the real video frame
    its size can differ from ``ctx.mask`` (the frame and the detection mask are
    source). Falls back to ``ctx.frame.shape`` only when there is no mask.
    """

    def apply(self, ctx: RenderContext) -> RenderContext:
        if ctx.mask is not None:
            h, w = ctx.mask.shape[:2]
        elif ctx.frame is not None:
            h, w = ctx.frame.shape[:2]
        else:
            raise ValueError("init_binary_mask needs a mask or a frame to size from")
        image = np.zeros((h, w, 3), dtype=np.uint8)
        if ctx.mask is not None:
            image[ctx.mask > 0] = 255
        ctx.image = image
        return ctx

    def to_spec(self) -> dict:
        return {"type": "init_binary_mask"}


def _crop_window(h: int, w: int, bbox, padding: float) -> tuple[int, int, int, int]:
    """``(py1, py2, px1, px2)``: the pixel window the padded normalized ``bbox`` crops an
    ``h``×``w`` array to."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - bw * padding, x2 + bw * padding
    y1, y2 = y1 - bh * padding, y2 + bh * padding
    px1, px2 = max(0, round(x1 * w)), min(w, round(x2 * w))
    py1, py2 = max(0, round(y1 * h)), min(h, round(y2 * h))
    px2, py2 = max(px2, px1 + 1), max(py2, py1 + 1)  # never empty
    return py1, py2, px1, px2


def _crop_to_bbox(arr: np.ndarray, bbox, padding: float) -> np.ndarray:
    """Crop ``arr`` (2-D mask or H×W×C image) to the padded normalized ``bbox``."""
    h, w = arr.shape[:2]
    py1, py2, px1, px2 = _crop_window(h, w, bbox, padding)
    return arr[py1:py2, px1:px2]


@register_transform("crop_to_bbox")
@dataclass
class CropToBbox(MaskTransform):
    padding: float = 0.0  # fractional padding around the bbox on each side

    def apply(self, ctx: RenderContext) -> RenderContext:
        if ctx.bbox is None:
            return ctx
        source = ctx.image if ctx.image is not None else ctx.mask
        if source is None:
            source = ctx.frame
        if source is not None and ctx.aux is not None:
            # Where the crop sits in the source grid, so a later step that holds full-frame
            # coordinates (``draw_pose``) can follow the crop without knowing the bbox.
            h, w = source.shape[:2]
            py1, py2, px1, px2 = _crop_window(h, w, ctx.bbox, self.padding)
            # Normalized, so it holds on whichever grid the image is built from; the source
            # size rides along for a step that sizes something in source pixels.
            ctx.aux["crop_window"] = (px1 / w, py1 / h, px2 / w, py2 / h, w, h)
        if ctx.image is not None:
            ctx.image = _crop_to_bbox(ctx.image, ctx.bbox, self.padding)
            return ctx
        # Crop-first (no image built yet): crop the source(s) so the downstream
        # init_binary_mask / apply_mask builds only the subject region — avoids
        # materializing a full-resolution image just to crop it away (§14). This is
        # output-identical to cropping after the build *when the mask and frame share
        # a resolution* — the production invariant, since detection emits
        # frame-resolution masks. If a mask and frame of different resolutions are
        # ever cropped here, crop and mask→image stop commuting exactly (each grid
        # rounds the bbox independently), so masked-color output would differ from the
        # crop-after order at the subject boundary; the mask-only silhouettes are
        # unaffected (single grid).
        if ctx.mask is not None:
            ctx.mask = _crop_to_bbox(ctx.mask, ctx.bbox, self.padding)
        if ctx.frame is not None:
            ctx.frame = _crop_to_bbox(ctx.frame, ctx.bbox, self.padding)
        return ctx

    def to_spec(self) -> dict:
        return {"type": "crop_to_bbox", "padding": self.padding}


@register_transform("pad_to_square")
@dataclass
class PadToSquare(MaskTransform):
    """Pad the shorter axis to match the longer — no cropping, no distortion."""

    def apply(self, ctx: RenderContext) -> RenderContext:
        h, w = ctx.image.shape[:2]
        size = max(h, w)
        top, left = (size - h) // 2, (size - w) // 2
        ctx.image = cv2.copyMakeBorder(
            ctx.image,
            top,
            size - h - top,
            left,
            size - w - left,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        return ctx

    def to_spec(self) -> dict:
        return {"type": "pad_to_square"}


@register_transform("resize")
@dataclass
class Resize(MaskTransform):
    size: int  # both dimensions — output is (size, size, C)

    def apply(self, ctx: RenderContext) -> RenderContext:
        h, w = ctx.image.shape[:2]
        # INTER_AREA is the correct (antialiased) kernel for shrinking; without it
        # a large mask silhouette downscaled to the model input aliases its edges,
        # measurably shifting the embedding vs. the HF processor's own resize (§14).
        interp = (
            cv2.INTER_AREA if (h > self.size or w > self.size) else cv2.INTER_LINEAR
        )
        ctx.image = cv2.resize(ctx.image, (self.size, self.size), interpolation=interp)
        return ctx

    def to_spec(self) -> dict:
        return {"type": "resize", "size": self.size}


@register_transform("apply_mask")
@dataclass
class ApplyMask(MaskTransform):
    """Keep the frame's pixels under the mask, flatten everything else.

    Leads a masked-RGB pipeline: when no image exists yet it initializes from
    ``ctx.frame`` (the real decoded frame, per Section 14), so the animal's true
    colors survive. The mask is resized to the working image when their
    resolutions differ (detection may have run on a downsampled source).
    """

    # The one transform that consumes the decoded pixels → a renderer containing it
    # needs the video frame (no decode-skip). ClassVar so it's not a dataclass field.
    reads_frame: ClassVar[bool] = True

    background: int | list[int] = (
        0  # scalar (grey) or [B, G, R] written outside the mask
    )

    def apply(self, ctx: RenderContext) -> RenderContext:
        base = ctx.image if ctx.image is not None else ctx.frame
        out = base.copy()
        if ctx.mask is not None:
            mask = ctx.mask
            if mask.shape[:2] != out.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (out.shape[1], out.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            out[~mask] = self.background
        ctx.image = out
        return ctx

    def to_spec(self) -> dict:
        return {"type": "apply_mask", "background": self.background}


@register_transform("init_frame")
@dataclass
class InitFrame(MaskTransform):
    """Use the raw decoded frame as the image, ignoring the mask (the no-mask baseline).

    Leads a full-frame / bbox-crop pipeline (§14 exploration) to measure how much the mask
    contributes. It consumes the decoded pixels, so it sets ``reads_frame`` — the same
    decode-forcing flag as ``apply_mask`` — and errors if handed no frame.
    """

    reads_frame: ClassVar[bool] = True

    def apply(self, ctx: RenderContext) -> RenderContext:
        if ctx.frame is None:
            raise ValueError("init_frame needs the decoded video frame")
        ctx.image = ctx.frame.copy()
        return ctx

    def to_spec(self) -> dict:
        return {"type": "init_frame"}


@register_transform("center_in_square")
@dataclass
class CenterInSquare(MaskTransform):
    """Place the image, unscaled, centered in a ``size``×``size`` black canvas.

    Preserves true pixel scale, so masks of different sizes keep their relative
    size (a small animal stays small, a large one fills more of the frame). An
    image larger than the canvas in either axis is center-cropped to fit.
    """

    size: int
    fill: int | list[int] = 0  # canvas padding: a scalar (black) or [B, G, R]

    def apply(self, ctx: RenderContext) -> RenderContext:
        image = ctx.image
        h, w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1
        canvas = np.full((self.size, self.size, channels), self.fill, dtype=image.dtype)
        ch, cw = min(h, self.size), min(w, self.size)
        sy, sx = (h - ch) // 2, (w - cw) // 2
        dy, dx = (self.size - ch) // 2, (self.size - cw) // 2
        canvas[dy : dy + ch, dx : dx + cw] = image[sy : sy + ch, sx : sx + cw]
        ctx.image = canvas
        # Carry the mask into canvas coordinates so a downstream background fill
        # (noise / radiating bands) knows which pixels are the subject. No-op for the
        # existing centered constructors, whose mask is unused after this step.
        if ctx.mask is not None:
            mask = ctx.mask
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            mcanvas = np.zeros((self.size, self.size), dtype=bool)
            mcanvas[dy : dy + ch, dx : dx + cw] = mask[sy : sy + ch, sx : sx + cw]
            ctx.mask = mcanvas
        return ctx

    def to_spec(self) -> dict:
        spec: dict = {"type": "center_in_square", "size": self.size}
        if self.fill != 0:
            spec["fill"] = self.fill
        return spec


@register_transform("draw_pose")
@dataclass
class DrawPose(MaskTransform):
    """Paint the subject's keypoints onto its image, so one embedding carries mask and pose
    (§14 "Pose in the embedding").

    Reads ``ctx.aux["pose"]`` (``{"xy": (K, 2) normalized full-frame, "score": (K,)}``), which the
    embedding worker fills from a keypoint track run joined on ``(part, frame_idx)``; a frame with
    no row passes through undrawn and the run records it (``pose_points`` 0). The points are
    mapped through the crop (``aux["crop_window"]``, which ``crop_to_bbox`` leaves) so they land on
    the animal, and drawn **before** any alignment step, so a rotation or flip carries them with
    the body. Each dot is coloured by the point's side (left, right, midline: the colours the run
    viewer draws, ``visualize.draw_keypoints``), and the skeleton's edges join them, so the model
    sees which end is which without being told.

    ``skeleton`` is the keypoint track run's own (``Skeleton.to_dict``), resolved into the spec so
    the run records what it drew; ``min_score`` is the floor below which a point is not drawn;
    ``size`` is a dot's radius as a fraction of the source frame's shorter side. A spec recorded
    before §26 carries ``sides`` / ``edges`` (by column) instead, and still renders.
    """

    skeleton: dict | None = None
    min_score: float | None = None
    size: float = 0.01
    sides: list | None = None  # pre-§26 specs only
    edges: list | None = None  # pre-§26 specs only

    def _skeleton(self, k: int):
        from detail.core.skeleton import Skeleton

        if self.skeleton is not None:
            return Skeleton.from_dict(self.skeleton)
        # A pre-§26 spec: columns named by position, the sides kept as pairs (the i-th left
        # with the i-th right), which is all the colours read.
        names = tuple(f"p{i}" for i in range(k))
        sides = list(self.sides or ["midline"] * k)
        lefts = [names[i] for i, side in enumerate(sides) if side == "left"]
        rights = [names[i] for i, side in enumerate(sides) if side == "right"]
        return Skeleton(
            names=names,
            edges=tuple((names[a], names[b]) for a, b in (self.edges or [])),
            pairs=tuple(zip(lefts, rights)),
        )

    def apply(self, ctx: RenderContext) -> RenderContext:
        pose = ctx.aux.get("pose") if ctx.aux else None
        if pose is None or ctx.image is None:
            return ctx
        xy = np.asarray(pose["xy"], dtype=np.float64)
        score = np.asarray(pose["score"], dtype=np.float64)
        shown = np.isfinite(xy).all(axis=-1)
        if self.min_score is not None:
            shown &= ~(score < float(self.min_score))
        if not shown.any():
            return ctx
        h, w = ctx.image.shape[:2]
        x0, y0, x1, y1, src_w, src_h = ctx.aux.get(
            "crop_window", (0.0, 0.0, 1.0, 1.0, w, h)
        )
        local = (xy - np.array([x0, y0])) / np.array([x1 - x0, y1 - y0])
        ctx.image = draw_keypoints(
            ctx.image,
            local,
            self._skeleton(len(xy)),
            visible=shown,
            radius=max(2, round(self.size * min(src_w, src_h))),
        )
        return ctx

    def to_spec(self) -> dict:
        spec: dict = {"type": "draw_pose"}
        if self.skeleton is not None:
            spec["skeleton"] = dict(self.skeleton)
        else:
            spec["sides"] = list(self.sides or [])
            spec["edges"] = [list(e) for e in self.edges or []]
        if self.min_score is not None:
            spec["min_score"] = self.min_score
        if self.size != 0.01:
            spec["size"] = self.size
        return spec


def _mask_for_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """The boolean subject mask resized (nearest) to ``image``'s H×W if they differ."""
    if mask.shape[:2] == image.shape[:2]:
        return mask
    return cv2.resize(
        mask.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


@register_transform("fill_background_noise")
@dataclass
class FillBackgroundNoise(MaskTransform):
    """Replace every non-subject pixel with one FIXED seeded random-noise field.

    The subject (pixels under the mask) is kept; the rest of the frame — both the
    masked-out region and the canvas padding — becomes uniform RGB noise seeded from
    ``seed``, so the *same* noise sits behind every frame of a run (an unnatural,
    constant background, like the solid-color constructors but textured). Runs after a
    ``center_in_square`` that carried the mask into canvas coordinates.
    """

    seed: int = 1234

    def apply(self, ctx: RenderContext) -> RenderContext:
        image = ctx.image
        rng = np.random.default_rng(self.seed)
        noise = rng.integers(0, 256, image.shape, dtype=np.uint8)
        out = image.copy()
        if ctx.mask is not None:
            mask = _mask_for_image(ctx.mask, image)
            out[~mask] = noise[~mask]
        else:
            out = noise
        ctx.image = out
        return ctx

    def to_spec(self) -> dict:
        return {"type": "fill_background_noise", "seed": self.seed}


@register_transform("fill_background_bands")
@dataclass
class FillBackgroundBands(MaskTransform):
    """Fill the non-subject region with concentric bands radiating from the silhouette.

    A distance transform from the mask boundary sorts every background pixel into a
    band (``floor(distance / thickness)``); even bands (nearest the subject) take
    ``ring``, odd bands take ``alt``, giving a "water drop" ripple that accentuates the
    outline. The subject's real pixels are kept. ``ring`` / ``alt`` are ``[B, G, R]``;
    ``thickness`` is the band width in pixels at the image's resolution.
    """

    ring: list[int]
    alt: list[int]
    thickness: int

    def apply(self, ctx: RenderContext) -> RenderContext:
        image = ctx.image
        if ctx.mask is None:
            return ctx
        mask = _mask_for_image(ctx.mask, image)
        dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        band = np.floor(dist / max(1, self.thickness)).astype(np.int32)
        even = (band % 2) == 0
        ring = np.array(self.ring, dtype=np.uint8)
        alt = np.array(self.alt, dtype=np.uint8)
        background = np.where(even[..., None], ring, alt)
        out = image.copy()
        out[~mask] = background[~mask]
        ctx.image = out
        return ctx

    def to_spec(self) -> dict:
        return {
            "type": "fill_background_bands",
            "ring": self.ring,
            "alt": self.alt,
            "thickness": self.thickness,
        }


# --------------------------------------------------------------------------- #
# §14 Egocentric alignment — inserted by `constructors.with_alignment` after the crop
# --------------------------------------------------------------------------- #
def _turned_box(mask: np.ndarray, deg: float) -> tuple[np.ndarray, int, int]:
    """Turn a bool mask so a heading of ``deg`` points along +x (cv2's ``angle=deg`` is
    counter-clockwise in image coordinates, which maps the heading vector onto +x), about its
    centre with nearest-neighbour edges on a canvas large enough to hold it; return the
    turned mask and the canvas size."""
    h, w = mask.shape[:2]
    side = int(np.ceil(np.hypot(h, w))) + 2
    centre = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, float(deg), 1.0)
    matrix[0, 2] += (side - w) / 2.0
    matrix[1, 2] += (side - h) / 2.0
    turned = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (side, side),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return turned, side, side


def turned_mask_extent(mask: np.ndarray, deg: float) -> tuple[int, int]:
    """``(width, height)`` in pixels of a mask's box after turning by the heading, without
    resampling: the mask's pixel coordinates rotated and measured — exact, and what the
    canvas pre-pass sizes the shared canvas from (§14 *Canvas*).

    A column of pixels is a vertical segment, so along any direction its extreme is its top
    or its bottom pixel: those two per column are all that is turned, not every pixel (7x
    faster, and the same answer).
    """
    occupied = np.flatnonzero(mask.any(axis=0))
    if occupied.size == 0:
        return 0, 0
    columns = mask[:, occupied]
    top = columns.argmax(axis=0)
    bottom = mask.shape[0] - 1 - columns[::-1].argmax(axis=0)
    xs = np.concatenate([occupied, occupied])
    ys = np.concatenate([top, bottom])
    theta = np.radians(-float(deg))
    c, s = np.cos(theta), np.sin(theta)
    rx = xs * c - ys * s
    ry = xs * s + ys * c
    # round before ceil: cos(90°) is 6e-17, not 0, and a 3.0000000000000004 span is 3 px
    return (
        int(np.ceil(round(float(rx.max() - rx.min()), 6))) + 1,
        int(np.ceil(round(float(ry.max() - ry.min()), 6))) + 1,
    )


@register_transform("align_rotate")
@dataclass
class AlignRotate(MaskTransform):
    """Turn the crop by the frame's heading so the animal faces right (+x, as Keypoint-MoSeq aligns), then re-crop to the
    **turned mask's** box (§14). Reads ``ctx.aux["heading_deg"]``; the exposed corners take
    ``fill`` (the recipe's background), and a frame with no heading is refused here rather
    than embedded unturned."""

    fill: int | list[int] = 0

    def apply(self, ctx: RenderContext) -> RenderContext:
        deg = ctx.aux.get("heading_deg") if ctx.aux else None
        if deg is None or not np.isfinite(float(deg)):
            raise ValueError(
                f"frame {ctx.frame_idx} has no heading; an aligned run never embeds an "
                "unturned frame"
            )
        if ctx.image is None or ctx.mask is None:
            raise ValueError(
                "align_rotate needs the cropped subject image and its mask"
            )
        mask = _mask_for_image(ctx.mask, ctx.image)
        turned_mask, side, _ = _turned_box(mask, float(deg))
        h, w = ctx.image.shape[:2]
        centre = ((w - 1) / 2.0, (h - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(centre, float(deg), 1.0)
        matrix[0, 2] += (side - w) / 2.0
        matrix[1, 2] += (side - h) / 2.0
        fill = self.fill
        value = (
            tuple(int(v) for v in fill) if isinstance(fill, list) else (int(fill),) * 3
        )
        turned = cv2.warpAffine(
            ctx.image,
            matrix,
            (side, side),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=value,
        )
        ys, xs = np.nonzero(turned_mask)
        if xs.size:
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            turned = turned[y0:y1, x0:x1]
            turned_mask = turned_mask[y0:y1, x0:x1]
        ctx.image = np.ascontiguousarray(turned)
        ctx.mask = turned_mask
        return ctx

    def to_spec(self) -> dict:
        spec: dict = {"type": "align_rotate"}
        if self.fill != 0:
            spec["fill"] = self.fill
        return spec


@register_transform("align_mirror")
@dataclass
class AlignMirror(MaskTransform):
    """Flip the crop left-right where the frame's ``align_mirrored`` flag is set (§14
    ``mirror``: the animal faced left for longer than the hold). No turn, no re-crop."""

    def apply(self, ctx: RenderContext) -> RenderContext:
        if ctx.aux and ctx.aux.get("align_mirrored"):
            ctx.image = np.ascontiguousarray(ctx.image[:, ::-1])
            if ctx.mask is not None:
                ctx.mask = np.ascontiguousarray(ctx.mask[:, ::-1])
        return ctx

    def to_spec(self) -> dict:
        return {"type": "align_mirror"}
