"""Shared mask / bbox / label drawing helpers (§15).

Pure functions over RGB uint8 frames — no Qt (Tenet 3), no model backends
(Tenet 13), no experiment state (Tenet 2). Callers compose by chaining: each
primitive returns a new ``np.ndarray`` and never mutates its input. The same
calls work from pipeline runners (prompt test, render preview, tracking preview)
and from a future GUI panel that wraps the resulting frame in a ``QImage``.

Scope is intentionally narrow — only the primitives an existing caller needs.
``draw_polygon`` (arenas / ROIs) and ``draw_point`` (PointCue handles, vertex
editing) land when their use cases are implemented, not speculatively.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from detail.core.chart_theme import LIGHT, rgb
from detail.core.dataclasses import Detection
from detail.core.skeleton import Skeleton

RGB = tuple[int, int, int]
BboxXYXYNorm = tuple[float, float, float, float]


# The chart theme's identity colours (Okabe-Ito minus black and yellow), as RGB and in the
# theme's order — one palette, so a track or a cluster is the same colour on a video frame as
# in the chart beside it. This was a second copy in a *different* order, which made the first
# cluster orange on screen and blue in its exported figure. Canonical (light-theme) steps: a
# frame is not a chart surface and does not change with the application's theme.
PALETTE_OKABE_ITO: tuple[RGB, ...] = tuple(rgb(c) for c in LIGHT.categorical)

#: The default tint for a subject's mask when nothing names a colour — Okabe-Ito amber, which
#: was this palette's first slot before it took the chart order, kept by name so rendered clips
#: did not change colour when the charts did.
MASK_TINT: RGB = rgb(LIGHT.hue("orange"))


#: The detection-run preview's one solid color — a detection has no identity across
#: frames for a color to track, unlike a tracking run's per-track colors below.
DETECTION_BOX_COLOR: RGB = (60, 220, 70)


def track_color(object_id: int | None) -> RGB:
    """The color a tracking preview draws one track in — stable across frames and
    parts, since the same track id always lands on the same palette entry.

    ``None`` (an untracked or unassigned detection) falls back to the plain detection
    color rather than a palette entry, so it never coincides with a real track's color.
    """
    if object_id is None:
        return DETECTION_BOX_COLOR
    return PALETTE_OKABE_ITO[object_id % len(PALETTE_OKABE_ITO)]


def assign_colors(
    keys: list[str],
    *,
    seed: int | None = None,
    palette: tuple[RGB, ...] = PALETTE_OKABE_ITO,
) -> dict[str, RGB]:
    """Assign one palette color per key.

    Deterministic given a seed (re-runs with the same seed produce the same
    mapping); random when ``seed`` is ``None``. Cycles the palette when
    ``len(keys) > len(palette)``.

    Raises:
        ValueError: If ``palette`` is empty.
    """
    if not palette:
        raise ValueError("palette must contain at least one color")
    rng = random.Random(seed)
    shuffled = list(palette)
    rng.shuffle(shuffled)
    return {key: shuffled[i % len(shuffled)] for i, key in enumerate(keys)}


def _blend_mask_into(
    out: np.ndarray, mask: np.ndarray | None, color: RGB, alpha: float
) -> None:
    """Blend a mask into ``out`` **in place** — the shared body of the two mask paths."""
    h, w = out.shape[:2]
    if mask is None or mask.shape != (h, w):
        return
    # Blend inside the mask's bounding box, not across the frame. A subject fills a small part
    # of an arena, and
    # `out[mask > 0]` builds a frame-sized boolean, scans it, gathers and scatters, all at full
    # resolution however small the animal is. The box is several times faster for
    # byte-identical output. It is the largest item in drawing a masked frame, and
    # at 40 frames a second it was the reason playback with masks could not reach 1x.
    rows = np.any(mask, axis=1)
    if not rows.any():
        return
    cols = np.any(mask, axis=0)
    y0 = int(np.argmax(rows))
    y1 = len(rows) - int(np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = len(cols) - int(np.argmax(cols[::-1]))
    window = out[y0:y1, x0:x1]
    pixels = mask[y0:y1, x0:x1] > 0
    tint = np.asarray(color, dtype=np.float32)
    window[pixels] = ((1.0 - alpha) * window[pixels] + alpha * tint).astype(np.uint8)


def _outline_bbox_into(
    out: np.ndarray,
    bbox_xyxy_norm: BboxXYXYNorm | None,
    color: RGB,
    label: str | None,
    thickness: int,
) -> None:
    """Outline a bbox on ``out`` **in place** (cv2's own draw calls already mutate)."""
    if bbox_xyxy_norm is None:
        return
    h, w = out.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy_norm
    p1 = (int(round(x1 * w)), int(round(y1 * h)))
    p2 = (int(round(x2 * w)), int(round(y2 * h)))
    cv2.rectangle(out, p1, p2, color, thickness)
    if label:
        draw_text_label(out, label, p1, color)


#: The two text inks a label may take — near-black and near-white, chosen per label by
#: :func:`readable_text_color` so the text always contrasts with the tag behind it.
LABEL_INK_DARK: RGB = (20, 20, 20)
LABEL_INK_LIGHT: RGB = (245, 245, 245)


def _relative_luminance(color: RGB) -> float:
    """WCAG 2 relative luminance of an sRGB colour, 0 (black) to 1 (white)."""

    def channel(value: int) -> float:
        c = value / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def readable_text_color(background: RGB) -> RGB:
    """Dark or light ink, whichever has the higher WCAG contrast ratio on ``background``."""
    lum = _relative_luminance(background)
    on_dark = (_relative_luminance(LABEL_INK_LIGHT) + 0.05) / (lum + 0.05)
    on_light = (lum + 0.05) / (_relative_luminance(LABEL_INK_DARK) + 0.05)
    return LABEL_INK_LIGHT if on_dark >= on_light else LABEL_INK_DARK


def label_scale(frame_height: int) -> float:
    """The font scale a frame label takes, growing with the frame.

    A fixed scale was the bug: 0.5 is ~11 px of text, which on a 1080p frame shown as a
    thumbnail shrinks to a few unreadable pixels.
    """
    return max(frame_height / 900.0, 0.4)


def draw_text_label(
    out: np.ndarray,
    text: str,
    anchor_xy: tuple[int, int],
    background: RGB,
    *,
    scale: float | None = None,
    above: bool = True,
) -> None:
    """Draw ``text`` **in place** on a filled tag of ``background``, in whichever ink reads
    on it — the one way text goes onto a frame.

    Text drawn straight onto footage in the overlay's own colour vanished against the
    translucent mask of that colour and against busy bedding; a solid tag behind it and a
    contrast-picked ink make it legible on any frame and any palette entry.

    ``above`` puts the tag's bottom-left at ``anchor_xy`` (a box label sitting on the box's
    top edge); when there is no room above, or ``above`` is False, the tag hangs below the
    anchor instead. The tag is kept inside the frame horizontally.
    """
    h, w = out.shape[:2]
    if scale is None:
        scale = label_scale(h)
    thickness = max(1, int(round(scale * 1.5)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = max(2, int(round(3 * scale)))
    box_w = tw + 2 * pad
    box_h = th + base + 2 * pad
    x0 = min(max(0, anchor_xy[0]), max(0, w - box_w))
    y0 = anchor_xy[1] - box_h if above else anchor_xy[1]
    if y0 < 0:
        y0 = max(0, anchor_xy[1])
    y0 = min(y0, max(0, h - box_h))
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), background, cv2.FILLED)
    cv2.putText(
        out,
        text,
        (x0 + pad, y0 + pad + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        readable_text_color(background),
        thickness,
        cv2.LINE_AA,
    )


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Swap a decoded BGR frame's channels, contiguously.

    ``frame[:, :, ::-1].copy()`` reads the same and costs 45x more: on a 1080p frame the
    numpy reverse-and-copy measures 5.36 ms against cv2's 0.12 ms, because cv2 has a SIMD
    path and a strided fancy-index copy does not. Per frame in a loop that is the whole
    difference between a clip that plays and one you wait for. The outputs are bit-identical.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """The inverse of :func:`bgr_to_rgb`. Same swap, same reason to use it."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def draw_mask(
    frame: np.ndarray,
    mask: np.ndarray | None,
    color: RGB,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a translucent mask fill onto an RGB frame.

    Returns the input unchanged (copy) if ``mask`` is ``None`` or shape-mismatched.
    """
    out = frame.copy()
    _blend_mask_into(out, mask, color, alpha)
    return out


def draw_bbox(
    frame: np.ndarray,
    bbox_xyxy_norm: BboxXYXYNorm | None,
    color: RGB,
    *,
    label: str | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Outline a normalized [0,1] bbox; optional label above the top edge."""
    out = frame.copy()
    _outline_bbox_into(out, bbox_xyxy_norm, color, label, thickness)
    return out


def _label_for(det: Detection) -> str:
    return f"{det.class_name} {det.confidence:.2f}"


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    color: RGB,
    *,
    min_conf: float = 0.0,
) -> np.ndarray:
    """Recurring pattern: draw every detection in a list with one color.

    Mask fill + bbox outline + ``class_name conf`` label per detection. ``min_conf``
    draws only detections scoring at or above it; the default draws everything.

    Copies the frame **once**, not once per primitive: the previous form chained
    ``draw_mask`` / ``draw_bbox``, each of which copies, so a frame carrying N
    detections was copied 2N times. That is invisible at a threshold of 0.5 and
    ruinous at 0.0, where a frame can carry every proposal the model found.
    """
    out = frame.copy()
    for det in detections:
        if det.confidence < min_conf:
            continue
        _blend_mask_into(out, det.mask, color, 0.5)
        if det.bbox is not None:
            _outline_bbox_into(out, det.bbox, color, _label_for(det), 2)
    return out


def draw_detections_by_class(
    frame: np.ndarray,
    detections: Iterable[Detection],
    colors: dict[str, RGB],
    *,
    default: RGB = (255, 255, 255),
    min_conf: float = 0.0,
) -> np.ndarray:
    """``draw_detections`` where each detection takes its class's own color.

    Exists so a multi-class overlay is one copy rather than one per detection — the
    per-detection ``draw_detections`` loop callers used to write was the 2N-copy shape
    described above.
    """
    out = frame.copy()
    for det in detections:
        if det.confidence < min_conf:
            continue
        color = colors.get(det.class_name, default)
        _blend_mask_into(out, det.mask, color, 0.5)
        if det.bbox is not None:
            _outline_bbox_into(out, det.bbox, color, _label_for(det), 2)
    return out


# --------------------------------------------------------------------------- #
# Preview style — the one drawing vocabulary behind a run's live preview,
# its frame/video export, and (eventually) the CLI.
# --------------------------------------------------------------------------- #
#: What a detection/tracking run preview can draw. Declaration order is dropdown/CLI
#: order — a run-level preview needs box, confidence and track-id combinations a bout
#: clip never asks for, which is why this is a second vocabulary rather than a reuse of
#: `clip_render.RENDERS` (raw/overlay/mask): that one is pinned to exactly three flavours
#: by an explicit ask, for a different surface (behavior bouts, not model QA).
MASK_BOX_CONF = "mask_box_conf"
MASK_BOX = "mask_box"
MASK_ONLY = "mask_only"
BOX_CONF = "box_conf"
BOX_ONLY = "box_only"
RAW_VIDEO = "raw_video"
RAW_MASK = "raw_mask"
PREVIEW_STYLES = (
    MASK_BOX_CONF,
    MASK_BOX,
    MASK_ONLY,
    BOX_CONF,
    BOX_ONLY,
    RAW_VIDEO,
    RAW_MASK,
)  # fmt: skip

#: What a surface shows for each style — a GUI dropdown and CLI `--style` help both read
#: this rather than retyping it, the same reason `clip_render`'s render labels exist.
PREVIEW_STYLE_LABELS: dict[str, str] = {
    MASK_BOX_CONF: "Mask, box, confidence",
    MASK_BOX: "Mask, box",
    MASK_ONLY: "Mask only",
    BOX_CONF: "Box, confidence",
    BOX_ONLY: "Box only",
    RAW_VIDEO: "Raw video",
    RAW_MASK: "Raw mask",
}


@dataclass(frozen=True)
class _StyleFlags:
    show_mask: bool
    show_box: bool
    show_confidence: bool
    #: `Raw mask` only: each detection's real, unrecolored pixels cut out onto a solid
    #: green background, for downstream keying/compositing. A genuinely different render
    #: path (a green canvas + a literal pixel cutout) rather than a draw *on* the frame,
    #: so it is its own branch in `draw_preview_overlay` instead of another flag combo.
    green_screen: bool


_STYLE_FLAGS: dict[str, _StyleFlags] = {
    MASK_BOX_CONF: _StyleFlags(True, True, True, False),
    MASK_BOX: _StyleFlags(True, True, False, False),
    MASK_ONLY: _StyleFlags(True, False, False, False),
    BOX_CONF: _StyleFlags(False, True, True, False),
    BOX_ONLY: _StyleFlags(False, True, False, False),
    RAW_VIDEO: _StyleFlags(False, False, False, False),
    RAW_MASK: _StyleFlags(False, False, False, True),
}

#: Green screen, plain and unambiguous against a subject's own colours.
_GREEN_SCREEN: RGB = (0, 255, 0)


def resolve_preview_style(style: str) -> str:
    """``style`` if it names a real style, or raise naming what is available."""
    if style not in _STYLE_FLAGS:
        raise ValueError(
            f"unknown preview style {style!r}; expected any of {list(PREVIEW_STYLES)}"
        )
    return style


def preview_needs_mask(style: str) -> bool:
    """Whether ``style`` draws from the decoded mask — the cost `with_masks` gates.

    Mask RLEs are decoded only when a style actually uses them: a box-only or raw-video
    preview has no reason to pay for it, which is the same saving `frame`'s own
    ``with_masks`` parameter already existed to offer — this just tells a caller which
    styles need it.
    """
    flags = _STYLE_FLAGS[resolve_preview_style(style)]
    return flags.show_mask or flags.green_screen


def draw_preview_overlay(
    frame: np.ndarray,
    detections: Iterable[Detection],
    style: str,
    *,
    color_fn: Callable[[Detection], RGB],
    label_fn: Callable[[Detection, bool], str | None] | None = None,
) -> np.ndarray:
    """Draw one frame in the chosen preview ``style`` — the one overlay renderer behind
    the live run preview and its frame/video export, so a run draws identically
    wherever it is asked for.

    ``color_fn`` picks a detection's color — a detection preview uses one solid color, a
    tracking preview colors by track id. Never called for `raw_video`/`raw_mask`, which
    draw nothing per-detection in that sense (a green-screen cutout keeps a detection's
    own pixels, not a flat color) — a caller may still pass one unconditionally, since
    those two styles simply never reach it.

    ``label_fn(det, show_confidence)`` returns the label to draw beside a box, or
    ``None`` for none — ``show_confidence`` is this style's own flag, not a suggestion:
    a tracking preview keeps its track-id label even when it says ``False`` (*"the
    track id is the reason a tracking preview exists"*), while a detection preview has
    no label at all when it does. Only called when the style draws a box; omitted
    entirely, no label is drawn for a style that does.
    """
    flags = _STYLE_FLAGS[resolve_preview_style(style)]
    if flags.green_screen:
        return _draw_raw_mask(frame, detections)
    if not flags.show_mask and not flags.show_box:
        return frame.copy()  # raw_video
    out = frame.copy()
    for det in detections:
        color = color_fn(det)
        if flags.show_mask:
            _blend_mask_into(out, det.mask, color, 0.5)
        if flags.show_box and det.bbox is not None:
            label = label_fn(det, flags.show_confidence) if label_fn else None
            _outline_bbox_into(out, det.bbox, color, label, 2)
    return out


def _draw_raw_mask(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """Every detection's real pixels, cut out onto a solid green background.

    No video underneath — that is the point (downstream keying/compositing) and the
    reason this is not "mask blended at alpha=1": a blend still starts from the source
    frame, and green screen means the frame is gone everywhere a mask does not cover.
    """
    out = np.empty_like(frame)
    out[:] = _GREEN_SCREEN
    for det in detections:
        if det.mask is None or det.mask.shape != frame.shape[:2]:
            continue
        out[det.mask] = frame[det.mask]
    return out


# --------------------------------------------------------------------------- #
# Keypoints (§26)
# --------------------------------------------------------------------------- #
#: A keypoint's colour by its place in the skeleton: the left member of a pair is the palette's
#: blue and the right its vermillion, the two Okabe-Ito keeps apart under every colour-vision
#: deficiency, so a left/right swap is visible to everyone; a midline point (in no pair) is
#: green. There is no ``side`` on a skeleton (§26) — the role is read from ``pairs``.
KEYPOINT_COLORS: dict[str, RGB] = {
    "left": rgb(LIGHT.hue("blue")),
    "right": rgb(LIGHT.hue("vermillion")),
    "midline": rgb(LIGHT.hue("green")),
}
_KEYPOINT_DIM: RGB = (150, 150, 150)


def keypoint_roles(skeleton: Skeleton) -> list[str]:
    """``"left"`` / ``"right"`` / ``"midline"`` per skeleton point, in column order."""
    roles = ["midline"] * len(skeleton.names)
    for left, right in skeleton.pair_indices():
        roles[left] = "left"
        roles[right] = "right"
    return roles


def draw_keypoints(
    frame: np.ndarray,
    points: np.ndarray,
    skeleton: Skeleton,
    *,
    visible: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    radius: int | None = None,
    hollow: np.ndarray | None = None,
    edge_color: RGB = (235, 235, 235),
    color: RGB | None = None,
) -> np.ndarray:
    """One subject's keypoints: the skeleton's edges as thin lines, then a dot per point
    coloured by its role (``KEYPOINT_COLORS``). Returns a new frame.

    ``points`` is ``(K, 2)`` normalized full-frame ``x, y`` in ``[0, 1]`` (what ``FrameResult``
    and the detect zarr hold); pixels are computed here, as ``draw_bbox`` does. A NaN point is
    not drawn, nor one ``visible`` marks False, and an edge is drawn only between two drawn
    points. ``scores`` in ``[0, 1]`` **dim** a point towards grey rather than hide it: every
    placed point stays on the picture, and low confidence is reported in the coverage rows.
    The dot scales with the frame so it reads on a 1080p frame and a 480p one alike.
    ``hollow`` (``(K,)`` bool) draws a point as a ring in its role colour instead of a dot:
    placed, but not taught as visible (a test's score cut-off). ``edge_color`` is the
    skeleton's lines: near-white over footage, darker on a white background. ``color``
    draws every point in one colour instead of by role — one animal among several in a
    montage overlay, told apart from the others by its colour.
    """
    out = frame.copy()
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] != len(skeleton.names):
        raise ValueError(
            f"{xy.shape[0]} points for a skeleton of {len(skeleton.names)} keypoints"
        )
    h, w = out.shape[:2]
    r = radius or max(2, round(min(h, w) / 160))
    ok = np.isfinite(xy).all(axis=1)
    if visible is not None:
        ok &= np.asarray(visible, dtype=bool).reshape(-1)
    px = np.zeros(xy.shape, dtype=np.int32)
    px[ok] = np.round(xy[ok] * np.array([w, h], dtype=np.float64)).astype(np.int32)
    for a, b in skeleton.edges:
        i, j = skeleton.index(a), skeleton.index(b)
        if ok[i] and ok[j]:
            cv2.line(
                out,
                (int(px[i, 0]), int(px[i, 1])),
                (int(px[j, 0]), int(px[j, 1])),
                tuple(int(c) for c in edge_color),
                max(1, r // 2),
                cv2.LINE_AA,
            )
    roles = keypoint_roles(skeleton)
    for k in np.flatnonzero(ok):
        ink = KEYPOINT_COLORS[roles[k]] if color is None else tuple(color)
        if scores is not None:
            s = float(np.clip(np.nan_to_num(float(scores[k]), nan=0.0), 0.0, 1.0))
            ink = tuple(
                int(round(c * s + d * (1.0 - s))) for c, d in zip(ink, _KEYPOINT_DIM)
            )
        centre = (int(px[k, 0]), int(px[k, 1]))
        if hollow is not None and bool(np.asarray(hollow).reshape(-1)[k]):
            cv2.circle(out, centre, r, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.circle(out, centre, r, ink, 1, cv2.LINE_8)
            continue
        cv2.circle(out, centre, r + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(out, centre, r, ink, -1, cv2.LINE_AA)
    return out


#: The onion skin's most opaque layer (the frame just before the one shown); older layers fade
#: linearly towards nothing. The montage's mask outline fades by the same rule.
ONION_ALPHA = 0.55


def draw_fading_paths(
    frame: np.ndarray, paths: np.ndarray, colours: Sequence[RGB]
) -> np.ndarray:
    """Lines through ``paths`` (``(n, K, 2)`` normalized, oldest first) on ``frame``, path
    ``k`` in ``colours[k]``, each fading from full at the newest point to nothing at the
    oldest; a NaN breaks a path. Returns a new frame.

    Blended one age at a time, and only over the box that age's segments cover: a segment is
    a few pixels long, and a whole-frame blend per segment (what this replaced, for one path)
    costs a frame copy for every point and every frame of the trail.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    paths = np.asarray(paths, dtype=np.float64)
    n = len(paths)
    if n < 2:
        return out
    paths = paths.reshape(n, -1, 2)
    thickness = max(1, round(min(h, w) / 300))
    pad = thickness + 2
    px = paths * np.array([w, h], dtype=np.float64)
    for i in range(1, n):
        ok = np.isfinite(px[i - 1]).all(axis=1) & np.isfinite(px[i]).all(axis=1)
        # A point that did not move leaves no trail (a zero-length line would draw a dot).
        ok[ok] = np.hypot(*(px[i][ok] - px[i - 1][ok]).T) >= 0.5
        if not ok.any():
            continue
        ends = np.concatenate([px[i - 1][ok], px[i][ok]])
        x0 = int(max(np.floor(ends[:, 0].min()) - pad, 0))
        y0 = int(max(np.floor(ends[:, 1].min()) - pad, 0))
        x1 = int(min(np.ceil(ends[:, 0].max()) + pad + 1, w))
        y1 = int(min(np.ceil(ends[:, 1].max()) + pad + 1, h))
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]
        layer = roi.copy()
        for k in np.flatnonzero(ok):
            a, b = px[i - 1, k], px[i, k]
            cv2.line(
                layer,
                (int(round(a[0])) - x0, int(round(a[1])) - y0),
                (int(round(b[0])) - x0, int(round(b[1])) - y0),
                tuple(int(c) for c in colours[k]),
                thickness,
                cv2.LINE_AA,
            )
        weight = i / (n - 1)
        out[y0:y1, x0:x1] = cv2.addWeighted(layer, weight, roi, 1.0 - weight, 0)
    return out


def fade_weights(
    layers: Sequence[tuple[int, int, np.ndarray] | None], *, alpha: float = ONION_ALPHA
) -> tuple[np.ndarray, int, int] | None:
    """How much of a trail colour each pixel takes: ``layers`` (oldest first; each
    ``(y0, x0, crop)`` of a bool mask, or None) weighted from ``alpha`` for the newest down
    to ``alpha / n`` for the oldest, the newer layer winning where two overlap. Returns
    ``(weight, top, left)`` — a float32 array over the layers' bounding box and where that
    box sits on the frame — or None when no layer is present.

    This is the one fade rule: the masks pane's onion skin and the montage's mask outline
    both take their weights here, so the two cannot disagree about how a trail fades.
    """
    present = [layer for layer in layers if layer is not None]
    if not present:
        return None
    top = min(y0 for y0, _x0, _crop in present)
    left = min(x0 for _y0, x0, _crop in present)
    bottom = max(y0 + crop.shape[0] for y0, _x0, crop in present)
    right = max(x0 + crop.shape[1] for _y0, x0, crop in present)
    weight = np.zeros((bottom - top, right - left), dtype=np.float32)
    n = len(layers)
    for age, layer in enumerate(reversed(layers)):  # newest first
        if layer is None:
            continue
        y0, x0, crop = layer
        share = np.float32(alpha * (1.0 - age / n))
        region = weight[
            y0 - top : y0 - top + crop.shape[0], x0 - left : x0 - left + crop.shape[1]
        ]
        np.maximum(region, np.where(crop, share, np.float32(0.0)), out=region)
    return weight, top, left


def tint_by_weight(
    canvas: np.ndarray, weight: np.ndarray, colour: RGB, *, top: int = 0, left: int = 0
) -> np.ndarray:
    """``colour`` blended into ``canvas`` by ``weight`` (float32 in ``[0, 1]``), whose top-left
    sits at ``(top, left)``; the rest of the canvas is untouched. Returns a new frame.

    **Blended over the weight's box, not the frame.** Outside it every weight is zero, so a
    blend there returns the canvas unchanged; doing it anyway was a float32 pass over the whole
    1080p frame per picture — about half of a montage tile, the largest item in drawing
    the pass. The output is byte-identical.
    """
    out = canvas.copy()
    bottom, right = top + weight.shape[0], left + weight.shape[1]
    tint = np.asarray(colour, dtype=np.float32)
    share = weight[..., None]
    window = out[top:bottom, left:right]
    blended = window.astype(np.float32) * (1.0 - share) + tint * share
    window[...] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    return out


def draw_onion_skin(
    canvas: np.ndarray,
    layers: Sequence[tuple[int, int, np.ndarray] | None],
    colour: RGB,
    *,
    alpha: float = ONION_ALPHA,
) -> np.ndarray:
    """``layers`` (oldest first) as a fading onion skin in ``colour`` on ``canvas``: one blend
    for all of them (:func:`fade_weights` then :func:`tint_by_weight`), not one per layer.
    Returns the canvas itself when there is nothing to draw, a new frame otherwise.
    """
    faded = fade_weights(layers, alpha=alpha)
    if faded is None:
        return canvas
    weight, top, left = faded
    if not weight.any():
        return canvas
    return tint_by_weight(canvas, weight, colour, top=top, left=left)


def mask_outline(mask: np.ndarray, thickness: int) -> np.ndarray:
    """The ring ``thickness`` pixels wide just **outside** a bool ``mask``: the mask dilated,
    minus the mask. A line drawn with it sits round the animal and never covers a paw — the
    montage's Raw-on look, where a tint over the animal would hide the fur it is there to show.
    Returns a bool array the mask's shape.
    """
    thickness = max(int(thickness), 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * thickness + 1, 2 * thickness + 1)
    )
    inside = np.asarray(mask, dtype=bool)
    grown = cv2.dilate(inside.astype(np.uint8), kernel) > 0
    return grown & ~inside
