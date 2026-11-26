"""The pure half of a recording visualization (DESIGN_PLAN §22 "Recording visualization").

Numbers in, numbers out — no stores, no Qt, no matplotlib: where each panel goes for a tile and a
video shape, how big its type is, which tree nodes a cut makes, which colour each frame takes,
and how a montage tiles the screen. ``api/recording_visualization.py`` reads the data and draws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from detail.core import chart_theme

#: The panels keep at least this share of the direction the video leaves free.
MIN_PANEL_SHARE = 0.30
#: The ethogram's height as a share of the tile's height (with a floor in px).
ETHOGRAM_SHARE = 0.010
ETHOGRAM_MIN_PX = 6
#: Type sizes in px at a 1080 px short side, and the floors no tile goes under.
FONT_PX = {"behavior": 34.0, "duration": 21.0, "label": 14.0, "time": 12.0}
FONT_FLOOR_PX = {"behavior": 16.0, "duration": 12.0, "label": 11.0, "time": 10.0}
#: Montage grids by tile count: (columns, rows).
MONTAGE_GRIDS = {1: (1, 1), 2: (2, 1), 3: (2, 2), 4: (2, 2), 5: (3, 2), 6: (3, 2)}
MAX_MONTAGE = 9

#: Frame label codes outside the coloured clusters.
GREY = -1  # in no coloured cluster: "other" for a cut, "unnamed" for names
NO_POSE = -2  # no embedding on this frame: drawn as a gap


@dataclass(frozen=True)
class Box:
    """A rectangle in tile pixels, top-left origin."""

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class Layout:
    """Where everything goes in one tile."""

    mode: str  # "stack" (video on top) or "side" (video on the left)
    video: Box
    ethogram: Box
    now: Box
    outline: Box
    pose: Box
    location: Box | None
    pad: float
    font_px: dict[str, float]


def font_sizes(tile_w: int, tile_h: int) -> dict[str, float]:
    """Each type size, scaled by the tile's short side and never under its floor."""
    unit = min(tile_w, tile_h) / 1080.0
    return {k: max(v * unit, FONT_FLOOR_PX[k]) for k, v in FONT_PX.items()}


def layout_for(tile_w: int, tile_h: int, aspect: float, location: bool) -> Layout:
    """Lay one visualization out in a ``tile_w`` x ``tile_h`` tile for a video ``aspect`` (w/h).

    Two arrangements are tried and the one giving the video more area is kept: *stack* (the video
    spans the width, panels below) and *side* (the video spans the height, panels to its right).
    Either way the panels keep ``MIN_PANEL_SHARE`` of the free direction, and they fill what is
    left as a row, a column or a 2x2 block by that region's shape.

    Raises:
        ValueError: A dimension or the aspect is not positive.
    """
    if tile_w <= 0 or tile_h <= 0 or aspect <= 0:
        raise ValueError(f"tile {tile_w}x{tile_h} and aspect {aspect} must be positive")
    pad = 0.03 * min(tile_w, tile_h)
    fonts = font_sizes(tile_w, tile_h)
    eth_h = max(ETHOGRAM_SHARE * tile_h, ETHOGRAM_MIN_PX)
    # The label row above the video (a small caption) costs one label line.
    cap = fonts["label"] * 1.6

    # stack
    s_w = tile_w - 2 * pad
    s_h = min(s_w / aspect, (1 - MIN_PANEL_SHARE) * tile_h - 4 * pad - eth_h - cap)
    s_w = s_h * aspect
    # side
    d_h = tile_h - 3 * pad - eth_h - cap
    d_w = min(d_h * aspect, (1 - MIN_PANEL_SHARE) * tile_w - 3 * pad)
    d_h = d_w / aspect
    if s_w * s_h >= d_w * d_h:
        mode, vw, vh = "stack", s_w, s_h
        vx, vy = pad + (tile_w - 2 * pad - vw) / 2, pad + cap
        eth = Box(vx, vy + vh + pad * 0.5, vw, eth_h)
        top = eth.y + eth_h + pad
        region = Box(pad, top, tile_w - 2 * pad, tile_h - top - pad)
    else:
        mode, vw, vh = "side", d_w, d_h
        vx = pad
        vy = pad + cap + (tile_h - 3 * pad - cap - eth_h - vh) / 2
        eth = Box(vx, vy + vh + pad * 0.5, vw, eth_h)
        left = vx + vw + pad
        region = Box(left, pad, tile_w - left - pad, tile_h - 2 * pad)
    now, outline, pose, loc = _panels(region, pad, fonts, location)
    return Layout(mode, Box(vx, vy, vw, vh), eth, now, outline, pose, loc, pad, fonts)


def _panels(r: Box, pad: float, fonts: dict, location: bool):
    """Split the free region into the now label, the outline trail, pose space and location."""
    now_h = fonts["behavior"] * 1.3 + fonts["duration"] * 1.6
    aspect = r.w / max(r.h, 1.0)
    if not location:
        if aspect >= 1.6:  # row: [now over outline] [pose]
            col = r.w * 0.30
            now = Box(r.x, r.y, col, now_h)
            outline = Box(r.x, r.y + now_h + pad, col, r.h - now_h - pad)
            pose = Box(r.x + col + pad, r.y, r.w - col - pad, r.h)
        else:  # column: now, then outline beside pose
            now = Box(r.x, r.y, r.w, now_h)
            rest = Box(r.x, r.y + now_h + pad, r.w, r.h - now_h - pad)
            ow = rest.w * 0.32
            outline = Box(rest.x, rest.y, ow, rest.h)
            pose = Box(rest.x + ow + pad, rest.y, rest.w - ow - pad, rest.h)
        return now, outline, pose, None
    if aspect > 2.2:  # row
        widths = (0.22, 0.43, 0.35)
        xs = [r.x]
        for w in widths[:-1]:
            xs.append(xs[-1] + (r.w - 2 * pad) * w + pad)
        cols = [Box(x, r.y, (r.w - 2 * pad) * w, r.h) for x, w in zip(xs, widths)]
        now = Box(cols[0].x, cols[0].y, cols[0].w, now_h)
        outline = Box(
            cols[0].x, cols[0].y + now_h + pad, cols[0].w, cols[0].h - now_h - pad
        )
        return now, outline, cols[1], cols[2]
    if aspect < 0.65:  # column
        now = Box(r.x, r.y, r.w, now_h)
        rest_h = r.h - now_h - 2 * pad
        top = Box(r.x, r.y + now_h + pad, r.w, rest_h * 0.55)
        ow = top.w * 0.32
        outline = Box(top.x, top.y, ow, top.h)
        pose = Box(top.x + ow + pad, top.y, top.w - ow - pad, top.h)
        loc = Box(r.x, top.y + top.h + pad, r.w, rest_h * 0.45)
        return now, outline, pose, loc
    # 2x2: [now / outline] [location] over [pose] spanning the left
    hw = (r.w - pad) / 2
    now = Box(r.x, r.y, hw, now_h)
    upper = (r.h - pad) * 0.42
    outline = Box(r.x, r.y + now_h + pad, hw, upper - now_h - pad)
    pose = Box(r.x, r.y + upper + pad, hw, r.h - upper - pad)
    loc = Box(r.x + hw + pad, r.y, hw, r.h)
    return now, outline, pose, loc


def montage_grid(n: int) -> tuple[int, int]:
    """``(columns, rows)`` for ``n`` tiles.

    Raises:
        ValueError: ``n`` is not 1..``MAX_MONTAGE``.
    """
    if not 1 <= n <= MAX_MONTAGE:
        raise ValueError(f"a montage holds 1 to {MAX_MONTAGE} visualizations, got {n}")
    return MONTAGE_GRIDS.get(n, (3, 3))


def montage_tiles(n: int, width: int, height: int) -> list[Box]:
    """The ``n`` tiles of a ``width`` x ``height`` montage, left to right, top to bottom, each a
    whole number of even pixels so every tile encodes and composes exactly."""
    cols, rows = montage_grid(n)
    tw, th = width // cols // 2 * 2, height // rows // 2 * 2
    out = []
    for i in range(n):
        c, r = i % cols, i // cols
        out.append(Box(c * tw, r * th, tw, th))
    return out


# --------------------------------------------------------------------------- the tree cut


def display_heights(
    children: np.ndarray, height: np.ndarray, n_leaves: int
) -> np.ndarray:
    """Each node's height for drawing and cutting: its merge height, raised to its children's.

    ``ClusterTree`` heights are monotone only within a transition, so a raw-height cut could split
    a child and keep its parent; this running maximum makes every parent at least as high as
    anything under it. Leaves are 0.
    """
    n_nodes = 2 * n_leaves - 1
    out = np.zeros(n_nodes, dtype=np.float64)
    for m in range(len(children)):
        a, b = (int(v) for v in children[m])
        out[n_leaves + m] = max(float(height[m]), out[a], out[b])
    return out


def cut_nodes(
    children: np.ndarray, height: np.ndarray, n_leaves: int, cut_height: float
) -> list[int]:
    """The nodes a cut at ``cut_height`` makes, largest first by leaf count.

    Descends from the root, splitting every node whose display height is above the cut, so the
    result is always a partition of the leaves.
    """
    shown = display_heights(children, height, n_leaves)
    root = 2 * n_leaves - 2
    stack, out = [root], []
    while stack:
        node = stack.pop()
        if node >= n_leaves and shown[node] > cut_height:
            a, b = (int(v) for v in children[node - n_leaves])
            stack.extend((a, b))
        else:
            out.append(node)
    size = _leaf_counts(children, n_leaves)
    return sorted(out, key=lambda nd: (-int(size[nd]), nd))


def _leaf_counts(children: np.ndarray, n_leaves: int) -> np.ndarray:
    size = np.ones(2 * n_leaves - 1, dtype=np.int64)
    for m in range(len(children)):
        a, b = (int(v) for v in children[m])
        size[n_leaves + m] = size[a] + size[b]
    return size


def label_raster(
    spans: list[dict], t_ns: np.ndarray, node_code: dict[int, int], embedded: np.ndarray
) -> np.ndarray:
    """One code per frame from labelled spans: the colour index of the span's node, ``GREY`` in
    no coloured node, ``NO_POSE`` where the frame has no embedding.

    ``spans`` are ``NodeView.labelled_runs`` rows of one recording and animal
    (``start_ns``/``end_ns``/``node``), ``t_ns`` the recording's per-frame times, ``node_code``
    each coloured node's index, and ``embedded`` the per-frame embedding mask.
    """
    out = np.full(len(t_ns), GREY, dtype=np.int64)
    for span in spans:
        code = node_code.get(int(span["node"]))
        if code is None:
            continue
        lo = int(np.searchsorted(t_ns, int(span["start_ns"]), "left"))
        hi = int(np.searchsorted(t_ns, int(span["end_ns"]), "right"))
        out[lo:hi] = code
    out[~np.asarray(embedded, dtype=bool)] = NO_POSE
    return out


# --------------------------------------------------------------------------- themes

#: The categorical palette, in fixed order: the chart theme's identity colours (Okabe-Ito, then
#: its frozen extension), validated per mode in `core/chart_theme.py` and the ones INTERPRET's tree
#: draws, so a cluster is one colour on screen and on a slide.
SERIES = {
    "light": tuple(chart_theme.LIGHT.identities(8)),
    "dark": tuple(chart_theme.DARK.identities(8)),
}

THEMES = {
    "dark": dict(
        bg="#000000",
        ink="#f0efec",
        muted="#8f8e88",
        cloud="#5a5954",
        cloud_alpha=0.55,
        ghost="#77766f",
        grey="#4a4945",
        gap="#262624",
        fade=0.7,
        fill="#2b2a28",
        old="#3d3c39",
        new="#f0efec",
        ring="#000000",
    ),
    "light": dict(
        bg="#ffffff",
        ink="#1f1f1e",
        muted="#8f8e88",
        cloud="#c4c3bc",
        cloud_alpha=0.5,
        ghost="#a3a29b",
        grey="#d3d2cc",
        gap="#efeee9",
        fade=0.7,
        fill="#ebeae5",
        old="#d3d2cc",
        new="#1f1f1e",
        ring="#ffffff",
    ),
}


def colour_of(theme: str, code: int) -> str:
    """A frame code's colour in ``theme``: a series slot, grey, or the gap colour."""
    t = THEMES[theme]
    if code == NO_POSE:
        return t["gap"]
    if code < 0 or code >= len(SERIES[theme]):
        return t["grey"]
    return SERIES[theme][code]


def nice_scale_bar(extent: float) -> float:
    """A round scale-bar length near a third of ``extent`` (1, 2 or 5 times a power of ten)."""
    if extent <= 0:
        return 1.0
    target = extent / 3
    p = 10 ** math.floor(math.log10(target))
    for m in (5, 2, 1):
        if m * p <= target:
            return float(m * p)
    return float(p)
