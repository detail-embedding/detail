"""A cluster's clips playing side by side: the clip montage (§22 *Clip montage*).

One cluster's members — :data:`DEFAULT_CLIPS` of them unless asked for more, **most typical
first** unless sorted otherwise (:data:`SORTS`) — each a tile cropped to its animal and labelled
``#rank · recording · start``, all playing **together**: every clip starts at once, a shorter one
holds its last frame until the longest has finished, and then all of them start again. One clock,
so a viewer's eye can run along a row and see whether the cluster is one behaviour. ``rank`` is
always the clip's typicality rank among the cluster's members (1 = most typical), whatever the
order the tiles are laid out in, so a tile names the same clip under every sort.

**What a tile shows** is chosen by :class:`MontageLayers`, the same five ticks the window offers.
The default (:data:`DEFAULT_LAYERS`, the user's choice of 2026-10-23) is **Mask + Mask trail**:

- **Raw** off: the masks alone and/or the keypoints alone, on white — the clip window's two side
  panes, drawn into one picture. The clip's animal is a **filled** tint.
- **Raw** on: the footage, with the clip's animal **outlined** (**Mask**) rather than tinted, so
  its fur and paws stay visible, and the keypoints drawn (**Keypoints**) on it. The outline is
  drawn at the tile's own scale, after the crop, :func:`outline_px` wide — so it reads the same in
  a 48-pixel tile and a 640-pixel GIF instead of vanishing when a 1080p frame is shrunk.
- **Mask trail** / **Keypoint trail**: the last half second of the animal's outline (filled on
  white, outlines on the footage), or of each chosen keypoint's path, fading with age
  (``ClipQueryPreview.clip_views``'s ``mask_trail`` and ``trail``). Each needs its parent tick.

Every picture comes from ``ClipQueryPreview.clip_views`` — the clip window's own renderer — so a
tile and the clip window never disagree about what a frame looks like.

**The crop** (:func:`crop_box`) is fixed for the whole clip, so a tile never jitters: the square
around the union of the animal's boxes over the clip, widened by :data:`CROP_MARGIN` either side and
never tighter than :data:`CROP_MIN_BODIES` of its typical box, so a still animal is not cropped to
its outline. An animal that crosses the arena gets the arena; one that sits still fills its tile.

**One clock, sampled at** :attr:`ClusterMontage.fps` — the footage's own rate up to
:data:`MONTAGE_MAX_FPS`. Fifty tiles at 30 fps would ask for 1,500 rendered pictures a second;
fifteen keeps real time and halves that. Time, not frame count, is what is synchronised: a clip's
picture at montage time ``t`` is its own frame at ``start + t`` (Tenet 8), so recordings filmed at
different rates still play at the same speed.

**Cost, and how many clips.** A tile is one ``clip_views`` picture at the footage's full
resolution, cropped and scaled; with **Raw** off it decodes no video (``footage=False``: the masks
and keypoints are stored data). Measured on a real 1080p temporal run (an_10, node 1394, in WSL,
six threads, the default look): 25 clips' first picture in 2.1 s and their 3.5 s pass in 6.0 s;
50 clips' in 3.9 s and 11.7 s. Raw on: 7.6 s and 16.9 s for the two passes. A held tile is never rendered twice: the cache is keyed by the clip and the frame it
shows, so after the short clips end only the long ones cost anything. Clips from one recording are
spread over separate decode contexts so each one's frames are read forward rather than seeking
between clips. The rendered-tile cache (:data:`CACHE_BYTES`) holds what a window has drawn (the
location montage's); a :meth:`ClusterMontage.subset` shares the root montage's decode contexts,
per-clip reads and cache, which is how **Export clips…** previews and plans its pieces without
reading anything twice. Preparing reads each recording's
``recording.yaml`` once (the clip list has just read it; ``prepare`` pins those reads, and threads
that miss one together wait for a single parse).

**Exports** draw at their own size, so they take the window's tiles only where the size and the
frame match, and put none of their own in the cache: what the window drew survives an export. A montage export draws its next picture while this one is encoded; a folder of per-clip
GIFs is written a clip per thread, each clip from its own decode context; a GIF chooses one palette
for all its frames (``api.animation``). Measured on the same run (WSL, median of 3, against the same code
before): 25 per-clip GIFs 19.0 s -> 9.6 s (Raw on 68.5 s -> 14.9 s); the 50-clip GIF 23.9 s -> 21.9 s
with Raw on (9.9 MB -> 1.4 MB), and with Raw off 17.4 s -> 16.5 s paired; the 50-clip mp4 17.1 s ->
16.3 s. What is left is drawing the tiles (about 0.1 s each on six threads that share the
interpreter), not writing them.

**Nothing here writes to the run.** A montage reads a clustering run and its recordings, and
writes only the files an export is pointed at.
"""

from __future__ import annotations

import csv
import functools
import math
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Sequence
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from detail.api.animation import (
    GIF,
    GIF_DEFAULT_FPS,
    GIF_DEFAULT_WIDTH,
    format_of,
    frame_at_offset,
    sample_offsets_ns,
    write_animation,
)
from detail.core.video_codecs import DEFAULT_VIDEO_CODEC
from detail.storage import read_cache
from detail.storage.roots import MissingRootError

#: How many clips a montage shows unless asked for more: the user's number (2026-10-23). The
#: window's clip-count box, ``--clips`` and every API default.
DEFAULT_CLIPS = 25
#: The orders a montage can be laid out in, and their names on screen. ``rank`` (typicality) is
#: kept on every clip whatever the order, and breaks every tie.
SORTS: dict[str, str] = {
    "typical": "Most typical first",
    "least-typical": "Least typical first",
    "recording": "By recording",
    "length": "Longest first",
    "random": "Random",
}
DEFAULT_SORT = "typical"
#: The fastest a montage clock runs (see the module docstring).
MONTAGE_MAX_FPS = 15.0
#: The crop's margin either side of the animal's boxes, as a share of the boxes' extent.
CROP_MARGIN = 0.15
#: The tightest a crop gets: this many of the animal's typical (median) box sides.
CROP_MIN_BODIES = 2.0
#: Tile side an export draws each clip at when no width is asked for.
EXPORT_TILE_PX = 200
#: The widest an mp4 export's grid gets when no width is asked for: a montage of hundreds of
#: clips shrinks its tiles rather than asking the encoder for a 5,000-pixel frame.
EXPORT_MAX_WIDTH = 3840
#: The bytes a montage's rendered-tile cache may hold before it drops the oldest.
CACHE_BYTES = 256 * 1024 * 1024
#: The grid's own colour behind the tiles and under the labels.
BACKGROUND = (38, 38, 38)
LABEL_INK = (235, 235, 235)
#: A tile with no picture: the recording cannot be read here, or, with Raw on, the frame did not
#: decode or the video is not on this machine (Raw off decodes no video, so it needs none).
_NO_PICTURE = (90, 90, 90)
#: Pixels between tiles.
GAP_PX = 2
#: Threads a montage draws tiles on (see ``ClusterMontage``).
WORKERS = 6
#: How many clips an overlay draws unless asked for more (§22 *Overlay*): the most whose
#: outlines can still be told apart by colour.
OVERLAY_CLIPS = 12
#: The overlay's square, in pixels, unless asked otherwise: the window draws at it, and an
#: export writes it.
OVERLAY_PX = 640
#: The overlay's canvas and its legend's ink.
OVERLAY_CANVAS = (255, 255, 255)
_LEGEND_INK = (40, 40, 40)
#: What an aligned tile, or an overlay's legend row, says on a frame the embedding would not
#: turn (and so embeds no row for).
NO_HEADING = "no heading"

Progress = Callable[[str, float, str], None] | None

#: Why a clip may have no picture on this machine: its video is missing, its video root is not
#: registered here (``MissingRootError``), or its run's files cannot be read. One clip's reason
#: is that clip's grey tile, never the whole montage's failure.
_UNREADABLE = (FileNotFoundError, KeyError, ValueError, OSError, MissingRootError)


@dataclass(frozen=True)
class MontageClip:
    """One tile: a member clip of the cluster, with its rank among the members (1 = most
    typical) and the recording's display name."""

    rank: int
    recording_id: str
    recording_name: str
    subject: object
    embed_run_id: str
    start_t_ns: int
    end_t_ns: int
    #: A ``run_preview.ClipSource`` when no embedding run stands behind the clip (a location
    #: export's tile, §16); ``embed_run_id`` is then empty.
    source: object = None
    #: Whether the method ranked this clip. A clip it could not (a bout assigned after a medoid
    #: fit, never compared) still has a rank, after every ranked one, and sorts last in either
    #: typicality order rather than leading the least typical (``ClipTable.order``).
    ranked: bool = True

    @property
    def key(self):
        """What the clip machinery reads the clip through: its source, or its embedding run."""
        return self.source if self.source is not None else self.embed_run_id


@dataclass(frozen=True)
class MontageOrigin:
    """Where a montage's clips came from: the run, the node, and what the node is called —
    written into a clip list and used to name a Query behaviour built from the clips."""

    run_id: str
    node: int
    #: The node's label, or None when it has none.
    name: str | None = None
    #: What the run calls a node: ``cluster`` (pose clustering) or ``behavior`` (temporal).
    unit: str = "cluster"

    @property
    def behavior_name(self) -> str:
        """``rearing``, or ``cluster 7`` for a node without a label."""
        return self.name or f"{self.unit} {self.node}"


@dataclass(frozen=True)
class MontageLayers:
    """What every tile draws — the montage window's ticks. The defaults are the window's
    opening ticks: the mask and its trail on white.

    ``mask_trail`` left unset follows ``mask``, so ``MontageLayers(mask=False, keypoints=True)``
    is a valid keypoints-only montage rather than an orphaned trail.

    Raises:
        ValueError: none of ``raw``, ``mask`` and ``keypoints`` is on, or a trail is asked for
            without what it trails.
    """

    raw: bool = False
    mask: bool = True
    keypoints: bool = False
    mask_trail: bool | None = None
    keypoint_trail: bool = False
    #: Which keypoints leave a trail, by name; ``None`` is every point.
    trail_points: tuple[str, ...] | None = None
    #: Turn every tile so its animal faces the same way, by the embedding's heading (§22
    #: *Align to heading*): up, or flipped as a ``mirror`` run flips.
    aligned: bool = False

    def __post_init__(self) -> None:
        if self.mask_trail is None:
            object.__setattr__(self, "mask_trail", bool(self.mask))
        if not (self.raw or self.mask or self.keypoints):
            raise ValueError(
                "a montage draws something: turn on at least one of raw, mask, keypoints"
            )
        if self.mask_trail and not self.mask:
            raise ValueError("a mask trail trails the mask: turn mask on with it")
        if self.keypoint_trail and not self.keypoints:
            raise ValueError(
                "a keypoint trail trails the keypoints: turn keypoints on with it"
            )


#: What a montage draws unless told otherwise — the window, the CLI and every API default.
DEFAULT_LAYERS = MontageLayers()


def blind_aliases(recording_ids: Sequence[str]) -> dict[str, str]:
    """``{recording_id: "A"}``, lettered in the order the recordings first appear (then
    ``AA``, ``AB`` … past 26) — the clip window's Blind mode names, shared by the montage.
    """
    out: dict[str, str] = {}
    for recording_id in recording_ids:
        if recording_id in out:
            continue
        n = len(out)
        letters = ""
        while True:
            letters = chr(ord("A") + n % 26) + letters
            n = n // 26 - 1
            if n < 0:
                break
        out[recording_id] = letters
    return out


def clock_tenths(t_ns: int) -> str:
    """``m:ss.t`` — a start time to a tenth of a second, which is what a label has room for."""
    tenths = int(round(max(int(t_ns), 0) / 1e8))
    minutes, rest = divmod(tenths, 600)
    return f"{minutes}:{rest // 10:02d}.{rest % 10}"


def montage_label(clip: MontageClip, alias: str | None = None) -> str:
    """``#3 · mouse_07 · 0:42.1``; with ``alias`` (Blind mode) ``#3 · recording C · 0:42.1``."""
    name = f"recording {alias}" if alias is not None else clip.recording_name
    return f"#{clip.rank} · {name} · {clock_tenths(clip.start_t_ns)}"


def montage_labels(
    clips: Sequence[MontageClip],
    *,
    blind: bool,
    aliases: dict[str, str] | None = None,
) -> list[str]:
    """Every tile's label, with Blind mode's aliases when ``blind``.

    ``aliases`` lets a caller letter the recordings over a larger set than ``clips`` — the
    window letters them over every member, most typical first, so recording A stays A on
    every page and under every sort.
    """
    if blind and aliases is None:
        aliases = blind_aliases([c.recording_id for c in clips])
    found = aliases if blind else {}
    return [montage_label(c, found.get(c.recording_id)) for c in clips]


def _file_safe(text: str) -> str:
    """``text`` with every character a file name might not take replaced by ``_``."""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in text)
    return "_".join(part for part in cleaned.split("_") if part) or "clip"


def clip_file_stem(clip: MontageClip, alias: str | None = None) -> str:
    """``3_mouse_07_0-42.1`` — a clip's file name without its suffix: rank, recording, start.

    With ``alias`` (Blind mode) the recording is ``recording-C``, so a folder of GIFs made
    blind stays blind. The clock's ``:`` becomes ``-``, which every file system takes.
    """
    name = f"recording-{alias}" if alias is not None else clip.recording_name
    when = clock_tenths(clip.start_t_ns).replace(":", "-")
    return f"{clip.rank}_{_file_safe(name)}_{when}"


def sort_clips(
    clips: Sequence[MontageClip], sort: str, seed: int = 0
) -> list[MontageClip]:
    """``clips`` in the order :data:`SORTS` names, ties broken by typicality rank.

    - ``typical``: rank 1 first — the cluster's centre outward.
    - ``least-typical``: the last ranked clip first — where impurities collect; clips the
      method could not rank stay last, as ``ClipTable.order("atypical")`` keeps them.
    - ``recording``: by recording name, then by start time within a recording.
    - ``length``: the longest clip first.
    - ``random``: a shuffle of the clips in rank order, seeded by ``seed`` (the node id, so
      a cluster's random order is the same every time it is opened).

    Raises:
        ValueError: an unknown sort.
    """
    if sort not in SORTS:
        raise ValueError(f"unknown sort {sort!r}: one of {', '.join(SORTS)}")
    if sort == "random":
        ranked = sorted(clips, key=lambda c: c.rank)
        shuffle = np.random.default_rng(int(seed)).permutation(len(ranked))
        return [ranked[int(i)] for i in shuffle]
    keys = {
        "typical": lambda c: c.rank,
        "least-typical": lambda c: (not c.ranked, -c.rank),
        "recording": lambda c: (c.recording_name, c.recording_id, c.start_t_ns, c.rank),
        "length": lambda c: (-(c.end_t_ns - c.start_t_ns), c.rank),
    }
    return sorted(clips, key=keys[sort])


def choose_clips(
    members: Sequence[MontageClip],
    *,
    count: int = DEFAULT_CLIPS,
    sort: str = DEFAULT_SORT,
    removed: Sequence[MontageClip] | set = (),
    seed: int = 0,
) -> list[MontageClip]:
    """The clips a montage of ``members`` shows: ``members`` without ``removed``, in ``sort``
    order, the first ``count`` of them — the window's clip-count box over the sorted order.
    ``seed`` is the ``random`` order's (:func:`sort_clips`).

    Raises:
        ValueError: ``count`` below 1, or an unknown sort.
    """
    if int(count) < 1:
        raise ValueError(f"a montage shows at least 1 clip, got {count}")
    gone = set(removed)
    return sort_clips([m for m in members if m not in gone], sort, seed)[: int(count)]


def with_context(clips: Sequence[MontageClip], seconds: float) -> list[MontageClip]:
    """``clips`` widened by ``seconds`` of footage either side — what a montage with context
    plays and exports (each tile then loops through the clip and its context). Label them
    from the clips as they were (:func:`montage_labels`), so a label still names the clip's own
    start. A clip near either end of its recording gets what footage there is when it is read.
    """
    pad = int(round(max(float(seconds), 0.0) * 1e9))
    if not pad:
        return list(clips)
    return [
        dataclasses.replace(
            clip,
            start_t_ns=max(int(clip.start_t_ns) - pad, 0),
            end_t_ns=int(clip.end_t_ns) + pad,
        )
        for clip in clips
    ]


def crop_box(
    boxes: np.ndarray,
    frame_wh: tuple[int, int],
    *,
    margin: float = CROP_MARGIN,
    min_bodies: float = CROP_MIN_BODIES,
) -> tuple[int, int, int, int]:
    """The pixel ``(x0, y0, x1, y1)`` a clip's tiles are cut from (see the module docstring).

    ``boxes`` are the animal's normalized per-frame boxes over the clip (NaN rows where it was
    not tracked). A square in pixels, centred on the union of the boxes, clamped inside the
    frame — narrower than square only where the frame itself is. No tracked frame at all: the
    whole frame.
    """
    width, height = int(frame_wh[0]), int(frame_wh[1])
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    boxes = boxes[np.isfinite(boxes).all(axis=1)]
    if not len(boxes) or width <= 0 or height <= 0:
        return (0, 0, max(width, 1), max(height, 1))
    px = boxes * np.array([width, height, width, height], dtype=np.float64)
    x0, y0 = px[:, 0].min(), px[:, 1].min()
    x1, y1 = px[:, 2].max(), px[:, 3].max()
    body = float(np.median(np.maximum(px[:, 2] - px[:, 0], px[:, 3] - px[:, 1])))
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * margin)
    side = max(side, min_bodies * body, 8.0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side_x, side_y = min(side, width), min(side, height)
    left = int(round(min(max(cx - side_x / 2.0, 0.0), width - side_x)))
    top = int(round(min(max(cy - side_y / 2.0, 0.0), height - side_y)))
    return (left, top, left + int(round(side_x)), top + int(round(side_y)))


def turn_degrees(heading_deg: float) -> float:
    """The angle a picture is turned by (``cv2.getRotationMatrix2D``'s, counter-clockwise on
    screen) so an animal heading ``heading_deg`` — image degrees, y down, 0 = facing +x, the
    embedding's convention (§14) — faces **up**."""
    return float(heading_deg) + 90.0


def turn_matrix(centre_xy, heading_deg: float, side: float) -> np.ndarray:
    """The ``2 × 3`` affine taking frame pixels into a ``side`` square: turned about
    ``centre_xy`` so the heading points up, that centre at the square's middle."""
    import cv2

    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    matrix = cv2.getRotationMatrix2D((cx, cy), turn_degrees(heading_deg), 1.0)
    matrix[0, 2] += side / 2.0 - cx
    matrix[1, 2] += side / 2.0 - cy
    return matrix


def turned_side(
    boxes: np.ndarray,
    deg: np.ndarray,
    frame_wh: tuple[int, int],
    *,
    margin: float = CROP_MARGIN,
) -> int:
    """An aligned tile's square for one clip, in pixels (§22 *Align to heading*): the largest
    of its boxes turned by **their own** frame's heading, widened by ``margin`` either side —
    fixed for the clip, so its animal is drawn at one scale throughout. A ``w × h`` box turned
    by ``a`` spans ``w|cos a| + h|sin a|`` across and ``w|sin a| + h|cos a|`` down (the
    embedding canvas pre-pass's bound). 8 when no frame has both a box and a heading."""
    width, height = float(frame_wh[0]), float(frame_wh[1])
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    deg = np.asarray(deg, dtype=np.float64).reshape(-1)
    n = min(len(boxes), len(deg))
    boxes, deg = boxes[:n], deg[:n]
    ok = np.isfinite(boxes).all(axis=1) & np.isfinite(deg)
    if not ok.any():
        return 8
    w = (boxes[ok, 2] - boxes[ok, 0]) * width
    h = (boxes[ok, 3] - boxes[ok, 1]) * height
    a = np.radians(deg[ok] + 90.0)
    c, s = np.abs(np.cos(a)), np.abs(np.sin(a))
    largest = float(np.max(np.maximum(w * c + h * s, w * s + h * c)))
    return max(int(math.ceil(largest * (1.0 + 2.0 * margin))), 8)


def overlay_colours(count: int) -> list[tuple[int, int, int]]:
    """``count`` colours for an overlay's clips, rank 1 first: hues evenly spaced round the
    wheel, so every clip is a different colour however many there are (§22 *Overlay*).
    """
    import colorsys

    out = []
    for i in range(max(int(count), 0)):
        r, g, b = colorsys.hsv_to_rgb((0.58 + i / max(int(count), 1)) % 1.0, 0.85, 0.82)
        out.append((int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
    return out


def outline_px(tile_px: int) -> int:
    """How wide the mask's outline is drawn in a tile of ``tile_px``: 2 pixels up to a
    160-pixel tile, then one more per 64 — so it stays a visible line in a 48-pixel tile and
    does not become a thread in a 640-pixel GIF."""
    return max(2, int(round(int(tile_px) / 64)))


def overlay_line_px(size: int) -> int:
    """How wide each clip's outline is in an overlay of ``size``: a line, not a tile's ring —
    a dozen animals' outlines lie on top of each other, and rings as wide as a tile's merge
    into one band. 2 pixels, one more per 200."""
    return max(2, int(round(int(size) / 200)))


def label_px(tile_px: int) -> int:
    """The height of the label strip under a tile of ``tile_px``."""
    return _font_px(tile_px) + 6


def _font_px(tile_px: int) -> int:
    return int(min(max(round(tile_px * 0.075), 9), 15))


def grid_extent(count: int, columns: int, tile_px: int) -> tuple[int, int]:
    """The ``(width, height)`` in pixels of :func:`compose`'s grid of ``count`` tiles."""
    columns = max(min(int(columns), max(count, 1)), 1)
    rows = math.ceil(max(count, 1) / columns)
    cell_h = tile_px + label_px(tile_px)
    return (
        columns * tile_px + GAP_PX * (columns + 1),
        rows * cell_h + GAP_PX * (rows + 1),
    )


def tile_origin(index: int, *, columns: int, tile_px: int) -> tuple[int, int]:
    """Where tile ``index``'s picture starts in a :func:`compose` grid, ``(x, y)`` pixels."""
    row, column = divmod(int(index), max(int(columns), 1))
    cell_h = tile_px + label_px(tile_px)
    return GAP_PX + column * (tile_px + GAP_PX), GAP_PX + row * (cell_h + GAP_PX)


def compose(
    tiles: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    columns: int,
    tile_px: int,
) -> np.ndarray:
    """Lay ``tiles`` (RGB, any size — scaled to ``tile_px``) out ``columns`` wide, each with
    its label under it, on the grid colour. What a montage window shows and what an export
    writes."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    count = len(tiles)
    columns = max(min(int(columns), max(count, 1)), 1)
    width, height = grid_extent(count, columns, tile_px)
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND
    for index, tile in enumerate(tiles):
        x, y = tile_origin(index, columns=columns, tile_px=tile_px)
        if tile.shape[0] != tile_px or tile.shape[1] != tile_px:
            tile = cv2.resize(
                tile,
                (tile_px, tile_px),
                interpolation=(
                    cv2.INTER_AREA if tile.shape[0] > tile_px else cv2.INTER_LINEAR
                ),
            )
        canvas[y : y + tile_px, x : x + tile_px] = tile
    picture = Image.fromarray(canvas, "RGB")
    draw = ImageDraw.Draw(picture)
    font = ImageFont.load_default(size=_font_px(tile_px))
    for index, label in enumerate(labels[:count]):
        x, y = tile_origin(index, columns=columns, tile_px=tile_px)
        text = _fit_label(label, lambda s: draw.textlength(s, font=font), tile_px - 2)
        draw.text((x + 1, y + tile_px + 2), text, fill=LABEL_INK, font=font)
    return np.asarray(picture)


def _fit_label(label: str, measure, width: float) -> str:
    """``label`` as it is when it fits ``width``, else cut short until it does.

    A cluster tile's ``#3 · mouse_07 · 0:42.1`` loses the recording's name first
    (``#3 · mouse_… · 0:42.1``), because the rank and the time are what tell two tiles apart.
    Any other label — a location montage's bare recording name — is cut in its middle
    (``2025_box3_…_arena_e``), because names in one experiment share a prefix and differ at
    the end. Whatever the shape, what comes back fits: an over-wide caption runs into the next
    tile's.
    """
    if measure(label) <= width:
        return label
    parts = label.split(" · ")
    if len(parts) >= 3:
        rank, name, when = parts[0], " · ".join(parts[1:-1]), parts[-1]
        for keep in range(len(name) - 1, 0, -1):
            text = f"{rank} · {name[:keep]}… · {when}"
            if measure(text) <= width:
                return text
        text = f"{rank} · {when}"
        if measure(text) <= width:
            return text
        label = rank
    for keep in range(len(label) - 1, 0, -1):
        tail = keep // 2
        text = f"{label[: keep - tail]}…{label[len(label) - tail :] if tail else ''}"
        if measure(text) <= width:
            return text
    return ""


@dataclass
class _Prepared:
    """One clip, resolved: its frames, their times, and its animal's boxes over them."""

    first: int
    t_ns: np.ndarray
    boxes: np.ndarray
    fps: float
    crop: tuple[int, int, int, int] | None = None
    #: The ``(width, height)`` of the picture ``crop`` was cut for, in pixels: the footage and
    #: the blank canvas a Raw-off tile draws on need not be the same size (a video that decodes
    #: larger than its record says), and a crop cut for one lands in a corner of the other.
    crop_size: tuple[int, int] | None = None
    #: The clip's heading on each of its frames (``run_preview.ClipHeadings``), read the first
    #: time an aligned tile or an overlay asks (:meth:`ClusterMontage.prepare_headings`).
    headings: object = None
    #: ``{(width, height): side}``: an aligned tile's square for the clip, in that picture's
    #: pixels (:func:`turned_side`).
    turn_sides: dict = field(default_factory=dict)


@dataclass
class _Shared:
    """What every page of one montage window shares: decode contexts, per-clip reads, the tile
    cache and the drawing threads (:meth:`ClusterMontage.subset`)."""

    dir: Path
    workers: int
    #: Show each clip's whole frame instead of the crop on its animal (a location montage's
    #: choice; the cluster window always crops).
    full_frame: bool = False
    previews: list = field(default_factory=list)
    previews_guard: threading.Lock = field(default_factory=threading.Lock)
    locks: dict = field(default_factory=dict)
    locks_guard: threading.Lock = field(default_factory=threading.Lock)
    prepared: dict = field(default_factory=dict)
    cache: OrderedDict = field(default_factory=OrderedDict)
    cache_bytes: int = 0
    cache_guard: threading.Lock = field(default_factory=threading.Lock)
    skeleton: object = None
    pool: object = None
    pool_guard: threading.Lock = field(default_factory=threading.Lock)

    def preview(self, slot: int):
        from detail.api.run_preview import ClipQueryPreview

        with self.previews_guard:
            while len(self.previews) <= slot:
                self.previews.append(ClipQueryPreview(self.dir))
            return self.previews[slot]


class ClusterMontage:
    """Render a cluster montage: tiles, whole frames, and exports.

    Built in two steps so a window can show itself first: construct (nothing is read), then
    :meth:`prepare` (reads every clip's timestamps and boxes, the one up-front cost, with
    progress). Every render method is safe to call from several threads at once: each decode
    context is locked while it is used, and the cache has its own lock.
    """

    def __init__(
        self,
        experiment_dir: Path | str,
        clips: Sequence[MontageClip],
        *,
        workers: int = WORKERS,
        full_frame: bool = False,
    ):
        """``workers`` threads draw tiles side by side: most of a cold tile is reading — a
        recording's metadata, its keypoints, a seek into its video — which a thread waits on
        without holding the interpreter. ``full_frame`` shows each clip's whole frame rather
        than :func:`crop_box`'s crop on its animal.

        Raises:
            ValueError: no clips.
        """
        shared = _Shared(Path(experiment_dir), max(int(workers), 1))
        shared.full_frame = bool(full_frame)
        self._setup(shared, clips)

    def _setup(self, shared: _Shared, clips: Sequence[MontageClip]) -> None:
        clips = list(clips)
        if not clips:
            raise ValueError("a montage needs at least one clip")
        self._shared = shared
        self.clips = clips
        # Clips from one recording go to different decode contexts (`slot`): a context holds one
        # reader per recording, and two clips sharing it would seek it back and forth every step.
        self._slot: list[int] = []
        seen: dict[str, int] = {}
        for clip in clips:
            self._slot.append(seen.get(clip.recording_id, 0))
            seen[clip.recording_id] = self._slot[-1] + 1

    def subset(self, clips: Sequence[MontageClip]) -> "ClusterMontage":
        """A montage of ``clips`` sharing this one's decode contexts, per-clip reads, tile cache
        and threads — a window's next page, or the clips it exports, without reading anything
        twice."""
        other = ClusterMontage.__new__(ClusterMontage)
        other._setup(self._shared, clips)
        other._owns_threads = False
        return other

    def _map(self, fn, items: list) -> list:
        """``[fn(item) for item in items]``, on the montage's threads."""
        pool = self._pool() if len(items) > 1 else None
        if pool is None:
            return [fn(item) for item in items]
        return list(pool.map(fn, items))

    def _pool(self):
        """The montage's drawing threads (made on first use), or None when it draws on one."""
        shared = self._shared
        with shared.pool_guard:
            if shared.workers == 1:
                return None
            if shared.pool is None:
                from concurrent.futures import ThreadPoolExecutor

                shared.pool = ThreadPoolExecutor(
                    max_workers=shared.workers, thread_name_prefix="montage"
                )
            return shared.pool

    def close(self) -> None:
        """Stop the montage's threads — every page's (it can still render afterwards, one
        tile at a time). Only the montage that made them closes them: a :meth:`subset` shares
        them with every other view of the same montage, so its ``close`` leaves them running
        (the recorded class "a shared resource closed from a view that does not own it").
        """
        if not getattr(self, "_owns_threads", True):
            return
        shared = self._shared
        with shared.pool_guard:
            pool, shared.pool, shared.workers = shared.pool, None, 1
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    # -- preparing ---------------------------------------------------------------- #
    def prepare(self, progress: Progress = None) -> None:
        """Read every clip's frame times and its animal's boxes; idempotent, and a clip another
        page already read is not read again."""
        prepared = self._shared.prepared
        missing = [i for i, c in enumerate(self.clips) if c not in prepared]
        if not missing:
            return
        done = [0]
        done_guard = threading.Lock()
        total = len(missing)

        def one(index: int) -> None:
            # Pinned: a clip's reads ask its recording's record five times over (the decode
            # context, the tracking run, the boxes), and the clip list read every record a
            # moment ago, so each ask is a stat that learns nothing — a round trip apiece on
            # a network share. A miss still reads (`read_cache.pinned`).
            with read_cache.pinned():
                found = self._prepare_one(index, self.clips[index])
            prepared[self.clips[index]] = found
            with done_guard:
                done[0] += 1
                count = done[0]
            if progress is not None:
                progress("preparing clips", count / total, f"clip {count} of {total}")

        if progress is not None:
            progress("preparing clips", 0.0, f"0 of {total}")
        self._map(one, missing)
        if progress is not None:
            progress("preparing clips", 1.0, f"{total} clips")

    def _prepare_one(self, index: int, clip: MontageClip) -> _Prepared:
        preview = self._preview(index)
        rec, emb = clip.recording_id, clip.key
        try:
            with self._lock(index):
                times = preview.timestamps(rec, emb)
                first = preview.t_ns_to_frame(rec, emb, clip.start_t_ns)
                last = max(preview.t_ns_to_frame(rec, emb, clip.end_t_ns), first)
                fps = preview.frame_rate(rec, emb)
                boxes = preview.subject_boxes(rec, emb, clip.subject, first, last)
        except _UNREADABLE:
            # A clip whose recording cannot be read still takes its place in the grid, as one
            # grey tile, rather than taking the montage down with it.
            return _Prepared(
                -1, np.zeros(1, np.int64), np.full((1, 4), np.nan), MONTAGE_MAX_FPS
            )
        t_ns = np.asarray(times[first : last + 1], dtype=np.int64)
        if not len(t_ns):
            t_ns = np.asarray([int(clip.start_t_ns)], dtype=np.int64)
        return _Prepared(int(first), t_ns, boxes, float(fps))

    def _ready(self) -> list[_Prepared]:
        prepared = self._shared.prepared
        if any(c not in prepared for c in self.clips):
            self.prepare()
        return [prepared[c] for c in self.clips]

    def _prep(self, index: int) -> _Prepared:
        found = self._shared.prepared.get(self.clips[index])
        if found is None:
            self.prepare()
            found = self._shared.prepared[self.clips[index]]
        return found

    def _preview(self, index: int):
        return self._shared.preview(self._slot[index])

    def _lock(self, index: int) -> threading.Lock:
        key = (self._slot[index], self.clips[index].recording_id)
        shared = self._shared
        with shared.locks_guard:
            return shared.locks.setdefault(key, threading.Lock())

    # -- headings (§22 *Align to heading*) ------------------------------------------ #
    def prepare_headings(self, progress: Progress = None) -> None:
        """Read every clip's heading on each of its frames (the embedding's rule), so an
        aligned tile or an overlay never waits on it; idempotent, and shared across pages.
        """
        self.prepare()
        missing = [
            i
            for i, p in enumerate(self._ready())
            if p.headings is None and p.first >= 0
        ]
        if not missing:
            return
        done = [0]
        done_guard = threading.Lock()
        total = len(missing)

        def one(index: int) -> None:
            self._headings(index)
            with done_guard:
                done[0] += 1
                count = done[0]
            if progress is not None:
                progress("reading headings", count / total, f"clip {count} of {total}")

        if progress is not None:
            progress("reading headings", 0.0, f"0 of {total}")
        self._map(one, missing)

    def _headings(self, index: int):
        """Clip ``index``'s ``ClipHeadings`` (read once)."""
        from detail.api.run_preview import ClipHeadings

        prepared = self._prep(index)
        if prepared.headings is None:
            clip = self.clips[index]
            n = len(prepared.t_ns)
            if prepared.first < 0:
                found = ClipHeadings(
                    np.full(n, np.nan, np.float32),
                    np.zeros(n, bool),
                    "rotate",
                    "This clip cannot be read here.",
                )
            else:
                try:
                    with self._lock(index):
                        found = self._preview(index).subject_headings(
                            clip.recording_id,
                            clip.key,
                            clip.subject,
                            prepared.first,
                            prepared.first + n - 1,
                        )
                except _UNREADABLE as exc:
                    found = ClipHeadings(
                        np.full(n, np.nan, np.float32),
                        np.zeros(n, bool),
                        "rotate",
                        f"Its heading cannot be read: {exc}",
                    )
            prepared.headings = found
        return prepared.headings

    def _require_headings(self, progress: Progress = None) -> None:
        """Read the headings an aligned export turns by; refuse when no clip has one, rather
        than write a file of grey tiles (the reason is the first clip's).

        Raises:
            ValueError: not one clip can be aligned.
        """
        self.prepare_headings(progress)
        if all(self._headings(i).note for i in range(len(self.clips))):
            raise ValueError(self.heading_note())

    def heading_note(self) -> str | None:
        """Why a clip has no heading to turn by — the first such clip's reason, with how many
        share one — or None when every clip has one (read on first ask)."""
        notes = [self._headings(i).note for i in range(len(self.clips))]
        missing = [n for n in notes if n]
        if not missing:
            return None
        if len(missing) == len(notes):
            return f"No clip can be aligned: {missing[0]}"
        return f"{len(missing)} of {len(notes)} clips cannot be aligned: {missing[0]}"

    def _turn_side(self, index: int, frame_wh: tuple[int, int]) -> int:
        prepared = self._prep(index)
        key = (int(frame_wh[0]), int(frame_wh[1]))
        side = prepared.turn_sides.get(key)
        if side is None:
            headings = self._headings(index)
            deg = headings.deg
            if headings.mode == "mirror":
                # No turn: the box as it stands (heading -90° turns by 0°), on each frame
                # that has a heading.
                deg = np.where(np.isfinite(deg), -90.0, np.nan)
            side = turned_side(prepared.boxes, deg, key)
            prepared.turn_sides[key] = side
        return side

    def _turn_at(self, index: int, frame: int, frame_wh: tuple[int, int]):
        """``(placement, None)`` turning clip ``index``'s ``frame`` as its embedding would, or
        ``(None, why not)``: no heading on the frame, or none for the clip."""
        prepared = self._prep(index)
        headings = self._headings(index)
        if headings.note:
            return None, NO_HEADING
        row = int(frame) - prepared.first
        if not 0 <= row < len(headings.deg) or not np.isfinite(headings.deg[row]):
            return None, NO_HEADING
        if headings.mode == "mirror":
            return _Mirror(bool(headings.mirrored[row])), None
        box = np.asarray(prepared.boxes[row], dtype=np.float64)
        if not np.isfinite(box).all():
            return None, NO_HEADING
        width, height = float(frame_wh[0]), float(frame_wh[1])
        centre = ((box[0] + box[2]) / 2.0 * width, (box[1] + box[3]) / 2.0 * height)
        side = self._turn_side(index, frame_wh)
        return (
            _Turn(turn_matrix(centre, float(headings.deg[row]), side), side),
            None,
        )

    # -- the clock ---------------------------------------------------------------- #
    @property
    def fps(self) -> float:
        """The montage clock's rate: the footage's (median over clips), at most 15."""
        rates = [p.fps for p in self._ready() if p.fps > 0]
        rate = float(np.median(rates)) if rates else MONTAGE_MAX_FPS
        return min(rate, MONTAGE_MAX_FPS)

    @property
    def duration_ns(self) -> int:
        """The longest clip's length: how long one pass of the montage lasts."""
        return max(int(p.t_ns[-1] - p.t_ns[0]) for p in self._ready())

    @property
    def n_steps(self) -> int:
        """How many pictures one pass is, at :attr:`fps`."""
        return len(sample_offsets_ns(self.duration_ns, self.fps))

    def step_offset_ns(self, step: int) -> int:
        """Montage time at ``step``, from the moment every clip starts."""
        return int(round(int(step) * 1e9 / self.fps))

    def frame_at(self, index: int, offset_ns: int) -> int:
        """Clip ``index``'s global frame at montage time ``offset_ns``: its own frame at
        ``start + offset``, or its **last** frame once it has ended (the hold)."""
        clip = self._prep(index)
        if clip.first < 0:
            return -1
        return clip.first + frame_at_offset(clip.t_ns, offset_ns)

    def frame_span(self, index: int) -> tuple[int, int]:
        """Clip ``index``'s first and last global frame, ``(-1, -1)`` when it cannot be read."""
        clip = self._prep(index)
        if clip.first < 0:
            return (-1, -1)
        return clip.first, clip.first + len(clip.t_ns) - 1

    def skeleton(self):
        """The keypoints' skeleton, once a tile has been drawn from a run that has one."""
        return self._shared.skeleton

    @property
    def experiment_dir(self) -> Path:
        """The experiment the clips are read from."""
        return self._shared.dir

    @property
    def footage_fps(self) -> float:
        """The footage's own frame rate (median over the clips), not capped as :attr:`fps`
        is — what an mp4 export plays at unless asked otherwise (``api.montage_export``).
        """
        rates = [p.fps for p in self._ready() if p.fps > 0]
        return float(np.median(rates)) if rates else MONTAGE_MAX_FPS

    def clip_boxes(self, index: int) -> np.ndarray:
        """Clip ``index``'s animal boxes over its frames, normalized; NaN rows where it was
        not tracked — what :func:`crop_box` cuts its tiles from."""
        return self._prep(index).boxes

    # -- tiles -------------------------------------------------------------------- #
    def _key(self, index: int, frame: int, layers: MontageLayers, size: int):
        return (self.clips[index], int(frame), layers, int(size))

    def tile(
        self, index: int, frame: int, layers: MontageLayers, size: int
    ) -> np.ndarray:
        """Clip ``index`` at global ``frame``, drawn per ``layers``, cropped and scaled to a
        ``size`` square. Cached by what it shows, so a holding clip is drawn once."""
        shared = self._shared
        key = self._key(index, frame, layers, size)
        with shared.cache_guard:
            found = shared.cache.get(key)
            if found is not None:
                shared.cache.move_to_end(key)
                return found
        picture = self._draw(index, int(frame), layers, int(size))
        with shared.cache_guard:
            if key not in shared.cache:
                shared.cache[key] = picture
                shared.cache_bytes += picture.nbytes
                while shared.cache_bytes > CACHE_BYTES and len(shared.cache) > 1:
                    _old, dropped = shared.cache.popitem(last=False)
                    shared.cache_bytes -= dropped.nbytes
        return picture

    def tiles_at(
        self, offset_ns: int, layers: MontageLayers, size: int
    ) -> list[np.ndarray]:
        """Every clip's tile at montage time ``offset_ns``."""
        return self._map(
            lambda i: self.tile(i, self.frame_at(i, offset_ns), layers, size),
            list(range(len(self.clips))),
        )

    def draw_first(
        self,
        layers: MontageLayers,
        size: int,
        *,
        on_tile: Callable[[int], None] | None = None,
    ) -> None:
        """Draw every clip's first picture (the one a window opens on), calling
        ``on_tile(index)`` as each lands — the slowest picture of a montage, since it is the
        one that opens every recording and seeks into every video."""

        def one(index: int) -> None:
            self.tile(index, self.frame_at(index, 0), layers, size)
            if on_tile is not None:
                on_tile(index)

        self._map(one, list(range(len(self.clips))))

    def _draw(
        self,
        index: int,
        frame: int,
        layers: MontageLayers,
        size: int,
        trail_s: float | None = None,
    ):
        import cv2

        clip = self.clips[index]
        prepared = self._prep(index)
        if frame < 0:
            return _blank(size, _NO_PICTURE, "video not found")
        outline = layers.raw and layers.mask
        preview = self._preview(index)
        try:
            with self._lock(index):
                views = preview.clip_views(
                    clip.recording_id,
                    clip.key,
                    clip.subject,
                    frame,
                    masks_on_frame=layers.raw and layers.mask,
                    points_on_frame=layers.raw and layers.keypoints,
                    trail=layers.keypoint_trail,
                    trail_points=(
                        None
                        if layers.trail_points is None
                        else list(layers.trail_points)
                    ),
                    mask_trail=layers.mask_trail and not layers.raw,
                    mask_trail_on_frame=layers.mask_trail and layers.raw,
                    mask_outline=outline,
                    footage=layers.raw,
                    trail_s=trail_s,
                )
        except _UNREADABLE + (IndexError,):
            views = None
        if views is None:
            return _blank(size, _NO_PICTURE, "no picture")
        if views.skeleton is not None and self._shared.skeleton is None:
            self._shared.skeleton = views.skeleton
        picture = _picture(views, layers)
        height, width = picture.shape[:2]
        flip = False
        place = None
        if layers.aligned and not self._shared.full_frame:
            place, reason = self._turn_at(index, frame, (width, height))
            if place is None:
                return _blank(size, _NO_PICTURE, reason)
            if isinstance(place, _Mirror):
                flip, place = place.flip, None
        if place is None:
            if prepared.crop is None or prepared.crop_size != (width, height):
                prepared.crop = (
                    (0, 0, width, height)
                    if self._shared.full_frame
                    else crop_box(prepared.boxes, (width, height))
                )
                prepared.crop_size = (width, height)
            place = _Cut(prepared.crop)
        fill = (0, 0, 0) if layers.raw else (255, 255, 255)
        square = place.picture(picture, fill)
        side = square.shape[0]
        tile = cv2.resize(
            square,
            (size, size),
            interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR,
        )
        if outline:
            # Each animal in its own colour when the clip names them (a location export's
            # tile); otherwise the clip's animal, or its pair, in the one mask tint.
            if views.subject_outlines is not None:
                trails = {
                    tuple(colour): layers
                    for layers, colour in views.history_outlines or []
                }
                outlines = [
                    (mask, trails.get(tuple(colour), []), colour)
                    for mask, colour in views.subject_outlines
                ]
            else:
                outlines = [(views.subject_mask, views.mask_history or [], None)]
            for mask, history, colour in outlines:
                tile = _outline_onto(
                    tile,
                    mask,
                    history,
                    lambda layer: place.mask(layer, size),
                    (height, width),
                    tint=colour,
                )
        return np.ascontiguousarray(tile[:, ::-1]) if flip else tile

    def _export_tile(
        self, index: int, frame: int, layers: MontageLayers, size: int, held: dict
    ) -> np.ndarray:
        """Clip ``index`` at ``frame`` for an export: the window's own tile when the cache has it
        at this size, else drawn here — and **not** put in the cache.

        An export draws at its own size (a 640-pixel GIF's grid of 77-pixel tiles, an mp4's
        200-pixel ones, a 636-pixel per-clip GIF), which the window never shows; caching those
        tiles would only push out the pass the window drew ahead (a folder of per-clip GIFs
        streams up to a gigabyte of them through a 256 MB cache). ``held`` is the export's own
        memory of each clip's last tile, which is all a holding clip needs: a clip's frame never
        goes backwards within a pass, so a repeat is always the previous one.
        """
        last = held.get(index)
        if last is not None and last[0] == frame:
            return last[1]
        shared = self._shared
        with shared.cache_guard:
            tile = shared.cache.get(self._key(index, frame, layers, size))
        if tile is None:
            tile = self._draw(index, int(frame), layers, int(size))
        held[index] = (frame, tile)
        return tile

    # -- whole pictures ------------------------------------------------------------- #
    def render(
        self,
        offset_ns: int,
        layers: MontageLayers,
        labels: Sequence[str],
        *,
        columns: int,
        tile_px: int,
        draw_px: int | None = None,
    ) -> np.ndarray:
        """The whole montage at montage time ``offset_ns``. Tiles are drawn at ``draw_px``
        (default ``tile_px``) and laid out at ``tile_px`` — a window draws at one size and
        shows at whatever its cells are, so a resize never re-decodes anything."""
        tiles = self.tiles_at(offset_ns, layers, draw_px or tile_px)
        return compose(tiles, labels, columns=columns, tile_px=tile_px)

    def export(
        self,
        path: Path | str,
        layers: MontageLayers,
        labels: Sequence[str],
        *,
        max_width: int | None = None,
        fps: float | None = None,
        speed: float = 1.0,
        progress: Progress = None,
        codec: str = DEFAULT_VIDEO_CODEC,
    ) -> Path:
        """Write one pass of the montage to ``path`` (``.mp4`` or ``.gif``).

        ``speed`` plays it faster (2.0) or slower (0.5) than real time: each written frame
        advances ``speed / fps`` seconds of footage, so the file's own rate stays ``fps``.
        ``codec`` is an mp4's encoding (``storage.video.VIDEO_CODECS``).

        ``fps`` defaults to the montage clock for an mp4 and to :data:`GIF_DEFAULT_FPS` for a
        GIF; ``max_width`` to a grid of :data:`EXPORT_TILE_PX` tiles for an mp4 (no wider than
        :data:`EXPORT_MAX_WIDTH`) and to :data:`GIF_DEFAULT_WIDTH` for a GIF — the GIF dialog's
        own defaults, so a script and the window write the same file. The grid is as near
        square as the clip count allows, and every frame samples real time at ``fps``.
        """
        kind = format_of(path)
        if max_width is None and kind == GIF:
            max_width = GIF_DEFAULT_WIDTH
        self.prepare(progress)
        if layers.aligned:
            self._require_headings(progress)
        rate = float(
            fps if fps is not None else (GIF_DEFAULT_FPS if kind == GIF else self.fps)
        )
        if speed <= 0:
            raise ValueError(f"speed must be positive, not {speed}")
        offsets = sample_offsets_ns(self.duration_ns, rate / float(speed))
        columns = math.ceil(math.sqrt(len(self.clips)))
        width = int(max_width) if max_width else None
        room = width if width is not None else EXPORT_MAX_WIDTH
        tile_px = max((room - GAP_PX * (columns + 1)) // columns, 16)
        if width is None:
            tile_px = min(tile_px, EXPORT_TILE_PX)

        held: dict = {}
        everyone = list(range(len(self.clips)))

        def picture(offset: int) -> np.ndarray:
            tiles = self._map(
                lambda i: self._export_tile(
                    i, self.frame_at(i, offset), layers, tile_px, held
                ),
                everyone,
            )
            return compose(tiles, labels, columns=columns, tile_px=tile_px)

        frames = _drawn_ahead(picture, offsets)
        try:
            return write_animation(
                path,
                frames,
                fps=rate,
                count=len(offsets),
                max_width=width,
                progress=progress,
                codec=codec,
            )
        finally:
            frames.close()

    def last_frame(
        self,
        layers: MontageLayers,
        labels: Sequence[str],
        *,
        max_width: int | None = None,
        progress: Progress = None,
    ) -> np.ndarray:
        """The montage's last picture with every trail covering the **whole** clip — one
        picture of where each animal went, rather than the last half second the player shows.
        A clip shorter than the montage shows its own last frame and its whole path."""
        self.prepare(progress)
        columns = math.ceil(math.sqrt(len(self.clips)))
        room = int(max_width) if max_width else EXPORT_MAX_WIDTH
        tile_px = max((room - GAP_PX * (columns + 1)) // columns, 16)
        if not max_width:
            tile_px = min(tile_px, EXPORT_TILE_PX)
        end = self.duration_ns
        tiles = self._map(
            lambda i: self._draw(
                i,
                self.frame_at(i, end),
                layers,
                tile_px,
                trail_s=self._clip_seconds(i),
            ),
            list(range(len(self.clips))),
        )
        return compose(tiles, labels, columns=columns, tile_px=tile_px)

    # -- the overlay (§22 *Overlay*) -------------------------------------------------- #
    @property
    def is_still(self) -> bool:
        """Every clip is one frame (a pose cluster's postures): an overlay of them is one
        picture, with no trail."""
        return self.duration_ns == 0

    def overlay(
        self,
        offset_ns: int,
        labels: Sequence[str],
        *,
        size: int = OVERLAY_PX,
    ) -> np.ndarray:
        """Every clip at montage time ``offset_ns`` drawn on top of the others in one
        ``size`` square, each aligned and in its own colour, with a legend beside it (see
        :func:`overlay_colours`). Cached like a tile, by the frames it shows, so a pass where
        every clip holds is drawn once."""
        frames = tuple(self.frames_at_step_offset(offset_ns))
        key = ("overlay", tuple(self.clips), frames, tuple(labels), int(size))
        shared = self._shared
        with shared.cache_guard:
            found = shared.cache.get(key)
            if found is not None:
                shared.cache.move_to_end(key)
                return found
        picture = self._overlay_picture(frames, labels, int(size))
        with shared.cache_guard:
            if key not in shared.cache:
                shared.cache[key] = picture
                shared.cache_bytes += picture.nbytes
                while shared.cache_bytes > CACHE_BYTES and len(shared.cache) > 1:
                    _old, dropped = shared.cache.popitem(last=False)
                    shared.cache_bytes -= dropped.nbytes
        return picture

    def frames_at_step_offset(self, offset_ns: int) -> list[int]:
        """Every clip's frame at montage time ``offset_ns``."""
        return [self.frame_at(i, offset_ns) for i in range(len(self.clips))]

    def _overlay_picture(
        self, frames: Sequence[int], labels: Sequence[str], size: int
    ) -> np.ndarray:
        trail = not self.is_still
        members = self._map(
            lambda i: self._overlay_member(i, int(frames[i]), trail),
            list(range(len(self.clips))),
        )
        return _compose_overlay(
            members, labels, overlay_colours(len(self.clips)), int(size), self._map
        )

    def _overlay_member(self, index: int, frame: int, trail: bool):
        """Clip ``index`` at ``frame`` as data for the overlay: its mask, onion skin and points
        and how to turn them (:class:`_OverlayMember`), or why it is left out."""
        clip = self.clips[index]
        if frame < 0:
            return "no picture"
        prepared = self._prep(index)
        headings = self._headings(index)
        row = int(frame) - prepared.first
        if (
            headings.note
            or not 0 <= row < len(headings.deg)
            or not np.isfinite(headings.deg[row])
            or not np.isfinite(prepared.boxes[row]).all()
        ):
            return NO_HEADING
        try:
            with self._lock(index):
                views = self._preview(index).clip_views(
                    clip.recording_id,
                    clip.key,
                    clip.subject,
                    int(frame),
                    masks_on_frame=False,
                    points_on_frame=False,
                    mask_trail=trail,
                    footage=False,
                    subject_data=True,
                )
        except _UNREADABLE + (IndexError,):
            views = None
        if views is None or views.subject_mask is None:
            return "no picture"
        if views.skeleton is not None and self._shared.skeleton is None:
            self._shared.skeleton = views.skeleton
        height, width = views.frame.shape[:2]
        box = np.asarray(prepared.boxes[row], dtype=np.float64)
        return _OverlayMember(
            mask=views.subject_mask,
            history=list(views.mask_history or []),
            points=views.subject_points,
            skeleton=views.skeleton,
            centre=(
                (box[0] + box[2]) / 2.0 * width,
                (box[1] + box[3]) / 2.0 * height,
            ),
            deg=float(headings.deg[row]),
            mirror=(
                bool(headings.mirrored[row]) if headings.mode == "mirror" else None
            ),
            side=self._turn_side(index, (width, height)),
            frame_hw=(height, width),
        )

    def export_overlay(
        self,
        path: Path | str,
        labels: Sequence[str],
        *,
        size: int = OVERLAY_PX,
        fps: float | None = None,
        progress: Progress = None,
        codec: str = DEFAULT_VIDEO_CODEC,
    ) -> Path:
        """Write the overlay to ``path``: one pass of the clips on the montage clock as an
        ``.mp4`` or ``.gif``, or the picture at the start as a ``.png`` — the one choice for
        single-frame members (:attr:`is_still`).

        Raises:
            ValueError: a video asked of single frames, a suffix that is none of the three,
                or no clip with a heading to turn by.
        """
        path = Path(path)
        self.prepare(progress)
        self._require_headings(progress)
        if path.suffix.lower() == ".png":
            import cv2

            from detail.core.visualize import rgb_to_bgr

            picture = self._overlay_picture(
                self.frames_at_step_offset(0), labels, int(size)
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), rgb_to_bgr(picture)):
                raise OSError(f"could not write {path}")
            if progress is not None:
                progress("writing", 1.0, path.name)
            return path
        kind = format_of(path)
        if self.is_still:
            raise ValueError(
                "these clips are single frames (a pose cluster's postures), so their overlay "
                "is one picture: write it as a .png"
            )
        rate = float(
            fps if fps is not None else (GIF_DEFAULT_FPS if kind == GIF else self.fps)
        )
        offsets = sample_offsets_ns(self.duration_ns, rate)
        frames = _drawn_ahead(
            lambda offset: self._overlay_picture(
                self.frames_at_step_offset(offset), labels, int(size)
            ),
            offsets,
        )
        try:
            return write_animation(
                path,
                frames,
                fps=rate,
                count=len(offsets),
                max_width=None,
                progress=progress,
                codec=codec,
            )
        finally:
            frames.close()

    def _clip_seconds(self, index: int) -> float:
        """Clip ``index``'s own length in seconds (its trail on the last tracing frame)."""
        t_ns = self._prep(index).t_ns
        return max(float(int(t_ns[-1]) - int(t_ns[0])) / 1e9, 0.0) if len(t_ns) else 0.0

    def export_last_frame(
        self,
        path: Path | str,
        layers: MontageLayers,
        labels: Sequence[str],
        *,
        max_width: int | None = None,
        progress: Progress = None,
    ) -> Path:
        """Write :meth:`last_frame` to ``path`` as a PNG."""
        import cv2

        picture = self.last_frame(
            layers, labels, max_width=max_width, progress=progress
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)):
            raise OSError(f"could not write {path}")
        return path

    def export_clip_gifs(
        self,
        out_dir: Path | str,
        layers: MontageLayers,
        labels: Sequence[str],
        stems: Sequence[str],
        *,
        max_width: int = GIF_DEFAULT_WIDTH,
        fps: float = GIF_DEFAULT_FPS,
        progress: Progress = None,
        on_written: Callable[[Path], None] | None = None,
    ) -> list[Path]:
        """Write each clip as a GIF of its own into ``out_dir``: ``<stem>.gif``, one pass of
        that clip at ``fps``, its tile ``max_width`` wide with its label under it.

        ``stems`` are the file names (see :func:`clip_file_stem`), one per clip. The montage's
        threads write a clip each, side by side; one thread draws all of a clip from its own
        decode context, so its frames are still read forward. The bar is weighed by the frames
        each GIF holds, not by the GIF count, so a long clip moves it as far as it costs. A
        stopped export (the progress callback raising) keeps the GIFs already finished —
        ``on_written`` has heard of each — and removes the ones it was writing
        (``api.animation``). Returned in the clips' order, whatever order they finished in.

        Raises:
            ValueError: ``stems`` and the clips differ in number, or two stems are the same.
        """
        if len(stems) != len(self.clips) or len(set(stems)) != len(stems):
            raise ValueError("one distinct file name per clip is needed")
        out = Path(out_dir)
        self.prepare(progress)
        if layers.aligned:
            self._require_headings(progress)
        out.mkdir(parents=True, exist_ok=True)
        width = int(max_width)
        tile_px = max(width - 2 * GAP_PX, 16)
        offsets = [
            sample_offsets_ns(int(p.t_ns[-1] - p.t_ns[0]), fps) for p in self._ready()
        ]
        counts = [len(o) for o in offsets]
        total = max(sum(counts), 1)
        shares = [0.0] * len(self.clips)  # how much of each GIF is written, 0..1
        finished: list[int] = []

        def report() -> None:
            if progress is not None:
                done = sum(share * n for share, n in zip(shares, counts))
                current = min(len(finished) + 1, len(self.clips))
                progress(
                    "writing GIFs",
                    min(done / total, 1.0),
                    f"GIF {current} of {len(self.clips)}",
                )

        def write_one(index: int, on_frame: Callable[[float], None]) -> Path:
            held: dict = {}

            def frames():
                for offset in offsets[index]:
                    frame = self.frame_at(index, offset)
                    tile = self._export_tile(index, frame, layers, tile_px, held)
                    yield compose([tile], [labels[index]], columns=1, tile_px=tile_px)

            return write_animation(
                out / f"{stems[index]}.gif",
                frames(),
                fps=float(fps),
                count=counts[index],
                max_width=width,
                progress=lambda _stage, fraction, _detail: on_frame(fraction),
            )

        def landed(index: int, path: Path) -> None:
            shares[index] = 1.0
            finished.append(index)
            if on_written is not None:
                on_written(path)

        pool = self._pool()
        if pool is None or len(self.clips) < 2:
            for index in range(len(self.clips)):

                def on_frame(fraction: float, index: int = index) -> None:
                    shares[index] = fraction
                    report()

                landed(index, write_one(index, on_frame))
        else:
            _write_side_by_side(pool, shares, write_one, landed, report)
        return [out / f"{stem}.gif" for stem in stems]

    def export_clip_list(
        self,
        path: Path | str,
        origin: MontageOrigin,
        *,
        progress: Progress = None,
    ) -> Path:
        """Write the clips as a CSV list, one row per clip in the montage's order
        (:data:`CLIP_LIST_COLUMNS`). Reading each clip's frame span is the one cost."""
        self.prepare(progress)
        rows = []
        for index, clip in enumerate(self.clips):
            first, last = self.frame_span(index)
            rows.append(_list_row(clip, origin, first, last))
        return write_clip_list(path, rows)


def _drawn_ahead(draw: Callable, items: Sequence):
    """``draw(item)`` for each of ``items`` in order, the next one drawn on a helper thread
    while the caller writes this one.

    An export is drawing (on the montage's threads, then laid out) and encoding (quantizing a
    GIF frame, compressing an H.264 one); one after the other, each waits for the other.
    Drawing one picture ahead lets them overlap, for one extra picture in memory. Pictures are
    drawn strictly in order, so every clip's frames are still read forward. Closing the
    generator (an export that failed or was cancelled) waits for the picture being drawn and
    draws nothing more.
    """
    from concurrent.futures import ThreadPoolExecutor

    items = list(items)
    if not items:
        return
    with ThreadPoolExecutor(1, thread_name_prefix="montage-export") as helper:
        pending = helper.submit(draw, items[0])
        try:
            for item in items[1:]:
                ready = pending.result()
                pending = helper.submit(draw, item)
                yield ready
            yield pending.result()
        finally:
            pending.cancel()


class _Stopped(Exception):
    """A per-clip GIF told to stop: the batch was cancelled, or another GIF failed."""


def _write_side_by_side(pool, shares: list, write_one, landed, report) -> None:
    """``write_one(index, on_frame)`` for every clip, on ``pool``; progress from this thread.

    Progress and cancellation stay on the calling thread: a progress callback raising is how a
    GUI bar cancels, and raised on a pool thread it would end only that one GIF. Each GIF's
    ``on_frame(fraction)`` records its share in ``shares``; this thread reports about ten times
    a second and as each GIF lands (``landed(index, path)``). On a stop or a failure the GIFs
    not yet started are cancelled, the ones being written stop at their next frame (and
    ``write_animation`` removes them), and any that finished meanwhile still land.
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    stop = threading.Event()

    def task(index: int) -> Path:
        def on_frame(fraction: float) -> None:
            if stop.is_set():
                raise _Stopped()
            shares[index] = fraction

        return write_one(index, on_frame)

    futures = {pool.submit(task, index): index for index in range(len(shares))}
    pending = set(futures)
    try:
        while pending:
            done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=futures.get):
                landed(futures[future], future.result())
            report()
    except BaseException:
        stop.set()
        for future in pending:
            future.cancel()
        done, _ = wait(pending)
        for future in sorted(done, key=futures.get):
            if not future.cancelled() and future.exception() is None:
                landed(futures[future], future.result())
        raise


class _Cut:
    """A tile cut from its frame unturned: ``crop_box``'s square (or the whole frame), padded
    square and centred — the montage's tile before §22 *Align to heading*."""

    def __init__(self, crop: tuple[int, int, int, int]) -> None:
        self.crop = crop

    def picture(self, picture: np.ndarray, fill) -> np.ndarray:
        x0, y0, x1, y1 = self.crop
        cut = picture[y0:y1, x0:x1]
        side = max(cut.shape[0], cut.shape[1], 1)
        square = np.empty((side, side, 3), dtype=np.uint8)
        square[:] = fill
        self.offset = ((side - cut.shape[0]) // 2, (side - cut.shape[1]) // 2)
        self.side = side
        oy, ox = self.offset
        square[oy : oy + cut.shape[0], ox : ox + cut.shape[1]] = cut
        return square

    def mask(self, layer, size: int) -> np.ndarray | None:
        return _into_tile(layer, self.crop, self.offset, self.side, size)


class _Turn:
    """A tile turned about its animal (§22 *Align to heading*): ``matrix`` takes frame pixels
    into a ``side`` square, the heading pointing up and the box's centre in the middle.
    """

    def __init__(self, matrix: np.ndarray, side: int) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.side = int(side)

    def picture(self, picture: np.ndarray, fill) -> np.ndarray:
        import cv2

        return cv2.warpAffine(
            picture,
            self.matrix,
            (self.side, self.side),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=tuple(int(v) for v in fill),
        )

    def mask(self, layer, size: int) -> np.ndarray | None:
        """A mask ``(y0, x0, bool crop)`` in frame pixels, turned into the square and scaled
        to ``size`` — the same steps the tile's picture took."""
        import cv2

        ly0, lx0, mask = layer
        matrix = self.matrix.copy()
        # The crop's pixel (0, 0) is the frame's (lx0, ly0).
        matrix[:, 2] += matrix[:, :2] @ np.array([float(lx0), float(ly0)])
        matrix *= float(size) / float(self.side)
        placed = (
            cv2.warpAffine(
                np.where(mask, 255, 0).astype(np.uint8),
                matrix,
                (int(size), int(size)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 127
        )
        return placed if placed.any() else None


@dataclass(frozen=True)
class _Mirror:
    """A ``mirror`` run's alignment of one frame: the unturned tile, flipped or not."""

    flip: bool


@dataclass(frozen=True)
class _OverlayMember:
    """One clip's frame in an overlay, as data (``clip_views(subject_data=True)``)."""

    mask: np.ndarray
    history: list
    points: tuple | None
    skeleton: object
    centre: tuple[float, float]
    deg: float
    #: A ``mirror`` run's flip on this frame; None for a turned (``rotate``) clip.
    mirror: bool | None
    side: int
    frame_hw: tuple[int, int]

    def placement(self, side: int) -> _Turn:
        """Into the overlay's shared ``side`` square: turned (or flipped) and centred."""
        if self.mirror is None:
            return _Turn(turn_matrix(self.centre, self.deg, side), side)
        cx, cy = self.centre
        sign = -1.0 if self.mirror else 1.0
        matrix = np.array(
            [[sign, 0.0, side / 2.0 - sign * cx], [0.0, 1.0, side / 2.0 - cy]]
        )
        return _Turn(matrix, side)

    def placed_points(self, place: _Turn, size: int) -> np.ndarray | None:
        """The keypoints ``(K, 2)`` normalized to the ``size`` canvas, or None without any."""
        if self.points is None:
            return None
        xy = np.asarray(self.points[0], dtype=np.float64)
        height, width = self.frame_hw
        px = xy * np.array([width, height], dtype=np.float64)
        moved = px @ place.matrix[:, :2].T + place.matrix[:, 2]
        return moved * (float(size) / float(place.side)) / float(size)


def _compose_overlay(
    members: Sequence,
    labels: Sequence[str],
    colours: Sequence,
    size: int,
    map_fn: Callable = lambda fn, items: [fn(item) for item in items],
) -> np.ndarray:
    """The overlay picture: every member's outline, trail and skeleton in its colour on one
    ``size`` square (rank 1 drawn last, on top), and the legend beside it. ``map_fn`` works
    out the members' outlines side by side (the montage's threads); they are blended in
    order, so the picture is the same however many threads drew it."""
    from detail.core.visualize import draw_keypoints

    drawn = [m for m in members if isinstance(m, _OverlayMember)]
    side = max((m.side for m in drawn), default=8)
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:] = OVERLAY_CANVAS
    places = {
        index: member.placement(side)
        for index, member in enumerate(members)
        if isinstance(member, _OverlayMember)
    }
    order = sorted(places, reverse=True)
    weights = map_fn(
        lambda index: _outline_weight(
            size,
            members[index].mask,
            members[index].history,
            lambda layer, p=places[index]: p.mask(layer, size),
            members[index].frame_hw,
            overlay_line_px(size),
        ),
        order,
    )
    for index, weight in zip(order, weights):
        canvas = _blend_outline(canvas, weight, colours[index])
    for index in reversed(range(len(members))):
        member = members[index]
        if index not in places or member.skeleton is None:
            continue
        xy = member.placed_points(places[index], size)
        if xy is not None:
            canvas = draw_keypoints(
                canvas,
                xy,
                member.skeleton,
                edge_color=colours[index],
                color=colours[index],
            )
    notes = tuple(m if isinstance(m, str) else None for m in members)
    legend = _overlay_legend(
        tuple(labels), tuple(tuple(int(v) for v in c) for c in colours), notes, size
    )
    return np.ascontiguousarray(np.hstack([canvas, legend]))


@functools.lru_cache(maxsize=32)
def _overlay_legend(
    labels: tuple[str, ...],
    colours: tuple[tuple[int, int, int], ...],
    notes: tuple[str | None, ...],
    size: int,
) -> np.ndarray:
    """The overlay's key, ``size`` high: each clip's colour and label, in columns, with why a
    clip is not in the picture at this moment. Kept: it is the same on almost every step of a
    pass, and drawing its text was a tenth of a step. Read-only, as every caller shares it.
    """
    from PIL import Image, ImageDraw, ImageFont

    font_px = _font_px(size)
    row = font_px + 8
    per_column = max((size - 8) // row, 1)
    count = len(colours)
    columns = max(math.ceil(count / per_column), 1)
    texts = [
        f"{labels[i] if i < len(labels) else ''}"
        + (f" · {notes[i]}" if i < len(notes) and notes[i] else "")
        for i in range(count)
    ]
    font = ImageFont.load_default(size=font_px)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    # Sized for the longest note every row could carry, so a video's frames keep one width.
    widest = [
        f"{labels[i] if i < len(labels) else ''} · {why}"
        for i in range(count)
        for why in (NO_HEADING, "no picture")
    ]
    text_w = max((probe.textlength(t, font=font) for t in widest), default=0.0)
    column_w = int(math.ceil(font_px + 6 + text_w + 12))
    legend = Image.new("RGB", (columns * column_w + 8, size), OVERLAY_CANVAS)
    draw = ImageDraw.Draw(legend)
    for i, text in enumerate(texts):
        column, place = divmod(i, per_column)
        x, y = 8 + column * column_w, 8 + place * row
        draw.rectangle((x, y, x + font_px, y + font_px), fill=tuple(colours[i]))
        draw.text((x + font_px + 6, y), text, fill=_LEGEND_INK, font=font)
    out = np.asarray(legend)
    out.setflags(write=False)
    return out


def _outline_onto(tile, subject_mask, history, place, frame_hw, tint=None):
    """``tile`` with the clip's animal outlined in ``tint`` (default: the mask tint), and its
    trail as fading outlines under it — drawn at the tile's own size (:func:`outline_px`),
    after the crop or the turn. ``place(layer)`` puts a ``(y0, x0, bool crop)`` mask where
    the tile's picture went, at the tile's size (None when it misses the tile).
    """
    weight = _outline_weight(tile.shape[0], subject_mask, history, place, frame_hw)
    return _blend_outline(tile, weight, tint)


def _outline_weight(
    size, subject_mask, history, place, frame_hw, thickness: int | None = None
) -> np.ndarray:
    """:func:`_outline_onto`'s ``(size, size)`` ink share: 1 on the outline, fading with age
    on the trail, 0 elsewhere — the ring ``thickness`` wide (:func:`outline_px` unless
    given). Independent of the picture under it, so an overlay works out its clips' on
    separate threads and blends them in order."""
    import cv2

    from detail.api.run_preview import _MASK_TRAIL_ALPHA

    thickness = outline_px(size) if thickness is None else int(thickness)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * thickness + 1, 2 * thickness + 1)
    )

    def ring(placed: np.ndarray):
        """``(rows, cols, edge)``: the ring round ``placed``, worked out only in its box
        widened by the ring's width — the one region a ring can reach, so the same pixels
        as over the whole tile at a fraction of the cost (an overlay is a dozen animals'
        trails on a 640-pixel square)."""
        x, y, w, h = cv2.boundingRect(placed.astype(np.uint8))
        rows = slice(max(y - thickness, 0), min(y + h + thickness, size))
        cols = slice(max(x - thickness, 0), min(x + w + thickness, size))
        inside = placed[rows, cols]
        grown = cv2.dilate(inside.astype(np.uint8), kernel) > 0
        # Outside the animal only: the line sits round it and never covers a paw.
        return rows, cols, grown & ~inside

    weight = np.zeros((size, size), dtype=np.float32)
    n = len(history)
    for age, layer in enumerate(reversed(history)):  # newest first
        if layer is None:
            continue
        placed = place(layer)
        if placed is not None:
            rows, cols, edge = ring(placed)
            alpha = np.float32(_MASK_TRAIL_ALPHA * (1.0 - age / n))
            region = weight[rows, cols]
            np.maximum(region, np.where(edge, alpha, np.float32(0.0)), out=region)
    if subject_mask is not None and tuple(subject_mask.shape[:2]) == tuple(frame_hw):
        now = place((0, 0, subject_mask))
        if now is not None:
            # The past outlines stop at the animal as it is now, so none crosses its fur.
            weight[now] = 0.0
            rows, cols, edge = ring(now)
            weight[rows, cols][edge] = 1.0
    return weight


def _blend_outline(tile, weight: np.ndarray, tint=None) -> np.ndarray:
    """``tile`` inked in ``tint`` (default: the mask tint) by ``weight``, only in the box
    where it is not 0; the same pixels as over the whole tile."""
    import cv2

    from detail.core.visualize import MASK_TINT

    if not weight.any():
        return tile
    x, y, w, h = cv2.boundingRect((weight > 0).astype(np.uint8))
    tint = np.asarray(MASK_TINT if tint is None else tint, dtype=np.float32)
    share = weight[y : y + h, x : x + w, None]
    out = tile.copy()
    blended = (
        tile[y : y + h, x : x + w].astype(np.float32) * (1.0 - share) + tint * share
    )
    out[y : y + h, x : x + w] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    return out


def _into_tile(layer, crop, offset, side, size) -> np.ndarray | None:
    """A mask ``(y0, x0, bool crop)`` in frame pixels, cut to the tile's crop, padded into its
    square and scaled to ``size`` — the same steps the tile's picture took. None when the mask
    lies outside the crop."""
    import cv2

    ly0, lx0, mask = layer
    x0, y0, x1, y1 = crop
    oy, ox = offset
    lh, lw = mask.shape[:2]
    ry0, ry1 = max(ly0, y0), min(ly0 + lh, y1)
    rx0, rx1 = max(lx0, x0), min(lx0 + lw, x1)
    if ry0 >= ry1 or rx0 >= rx1:
        return None
    square = np.zeros((side, side), dtype=np.uint8)
    square[oy + ry0 - y0 : oy + ry1 - y0, ox + rx0 - x0 : ox + rx1 - x0] = np.where(
        mask[ry0 - ly0 : ry1 - ly0, rx0 - lx0 : rx1 - lx0], 255, 0
    )
    scaled = cv2.resize(
        square,
        (size, size),
        interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR,
    )
    placed = scaled > 127
    return placed if placed.any() else None


def _blank(size: int, colour, text: str = "") -> np.ndarray:
    """A plain tile, saying why it is plain."""
    tile = np.empty((size, size, 3), dtype=np.uint8)
    tile[:] = colour
    if not text:
        return tile
    from PIL import Image, ImageDraw, ImageFont

    picture = Image.fromarray(tile, "RGB")
    draw = ImageDraw.Draw(picture)
    font = ImageFont.load_default(size=_font_px(size))
    text = _fit_label(text, lambda s: draw.textlength(s, font=font), size - 2)
    width = draw.textlength(text, font=font)
    draw.text(
        ((size - width) / 2, size / 2 - _font_px(size) / 2), text, fill=LABEL_INK,
        font=font,
    )  # fmt: skip
    return np.asarray(picture)


def _picture(views, layers: MontageLayers) -> np.ndarray:
    """The one picture a tile is cut from, per ``layers`` (see the module docstring)."""
    if layers.raw:
        return views.frame
    if layers.mask and layers.keypoints and views.pose is not None:
        # The keypoints over the masks: both panes are drawn on white, so whatever the pose pane
        # inked is laid over the masks pane.
        inked = (views.pose < 250).any(axis=2)
        out = views.masks.copy()
        out[inked] = views.pose[inked]
        return out
    if layers.keypoints and views.pose is not None:
        return views.pose
    if layers.mask:
        return views.masks
    return np.full_like(views.frame, 255)  # keypoints asked for, and the run has none


# -- the clip list ---------------------------------------------------------------------- #
#: A clip list's columns. The first ten are the list itself; the rest make it a clip-query CSV
#: as well (``storage/clip_query_csv.read_clip_csv``), so ``detail analysis query --from-csv``
#: takes a montage's list as its example clips, under ``query_name``.
CLIP_LIST_COLUMNS = [
    "rank",
    "recording_name",
    "recording_id",
    "subject",
    "start_t_ns",
    "end_t_ns",
    "start_s",
    "end_s",
    "node",
    "run_id",
    "embed_run_id",
    "start_frame",
    "end_frame",
    "query_name",
    "kind",
    "track_id",
    "score",
]


def _list_row(clip: MontageClip, origin: MontageOrigin, first: int, last: int) -> dict:
    return {
        "rank": clip.rank,
        "recording_name": clip.recording_name,
        "recording_id": clip.recording_id,
        "subject": clip.subject,
        "start_t_ns": int(clip.start_t_ns),
        "end_t_ns": int(clip.end_t_ns),
        "start_s": f"{clip.start_t_ns / 1e9:.3f}",
        "end_s": f"{clip.end_t_ns / 1e9:.3f}",
        "node": int(origin.node),
        "run_id": origin.run_id,
        "embed_run_id": clip.embed_run_id or "",
        "start_frame": int(first),
        "end_frame": int(last),
        "query_name": origin.behavior_name,
        # An example clip, in the clip-query schema's words; it has no match score.
        "kind": "query",
        "track_id": clip.subject,
        "score": "nan",
    }


def write_clip_list(path: Path | str, rows: Sequence[dict]) -> Path:
    """Write clip-list ``rows`` (:data:`CLIP_LIST_COLUMNS`) to ``path``, atomically."""
    from detail.core import atomic

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLIP_LIST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    atomic.replace(tmp, path)
    return path


# -- query examples --------------------------------------------------------------------- #
@dataclass(frozen=True)
class QueryExamples:
    """Montage clips as the example clips of one behaviour Query (§20) — what *Use selected
    as query examples* prefills, and what a script hands ``Experiment.run_analysis("query",
    …)`` after adding what a montage cannot know.

    The clips are the user's own choice, made by watching them: a method given them is given
    what a user could give it, which is what "Labels score a method; they never build one"
    allows.
    """

    behavior_name: str
    examples: tuple[dict, ...]
    #: The recordings the examples are on — a Query searches at least these.
    recording_ids: tuple[str, ...]
    #: The embedding the clustering read for these clips, or None when they span several.
    embed_run_id: str | None
    embed_run_ids: tuple[str, ...]
    #: The subject most of the examples follow (the run-level ``track_id``).
    track_id: object

    def config(self) -> dict:
        """The Query config these clips fill in: its recordings, embedding, subject and one
        behaviour. Everything else (method, its parameters, more recordings to search,
        non-examples) is left at Query's defaults for the caller to set.

        Raises:
            ValueError: the clips come from more than one embedding run — a Query searches one.
        """
        if self.embed_run_id is None:
            raise ValueError(
                "these clips come from several embedding runs "
                f"({', '.join(self.embed_run_ids)}); a Query searches one — choose clips "
                "from one of them"
            )
        return {
            "source_recording_ids": list(self.recording_ids),
            "embed_run_id": self.embed_run_id,
            "track_id": self.track_id,
            "behaviors": [
                {
                    "name": self.behavior_name,
                    "examples": [dict(c) for c in self.examples],
                    "negatives": [],
                }
            ],
        }


def query_examples(clips: Sequence[MontageClip], *, name: str) -> QueryExamples:
    """``clips`` as one Query behaviour named ``name`` (see :class:`QueryExamples`).

    Raises:
        ValueError: no clips.
    """
    clips = list(clips)
    if not clips:
        raise ValueError("choose at least one clip to use as an example")
    recordings = tuple(dict.fromkeys(c.recording_id for c in clips))
    embeds = tuple(dict.fromkeys(c.embed_run_id for c in clips))
    subjects = Counter(c.subject for c in clips)
    return QueryExamples(
        behavior_name=name,
        examples=tuple(
            {
                "recording_id": c.recording_id,
                "track_id": c.subject,
                "start_t_ns": int(c.start_t_ns),
                "end_t_ns": int(c.end_t_ns),
            }
            for c in clips
        ),
        recording_ids=recordings,
        embed_run_id=embeds[0] if len(embeds) == 1 else None,
        embed_run_ids=embeds,
        track_id=subjects.most_common(1)[0][0],
    )


# -- the cluster's clips --------------------------------------------------------------- #
def cluster_montage_members(
    experiment, run, node: int, *, progress: Progress = None
) -> list[MontageClip]:
    """Every one of ``node``'s members, most typical first, as montage tiles —
    ``run.typical_members``, so a node under the cache's floor gets its clips too. A window
    holds these and chooses from them (:func:`choose_clips`)."""
    members = run.typical_members(int(node), None, progress=progress)
    return _as_clips(experiment, run, members)


def cluster_montage_clips(
    experiment,
    run,
    node: int,
    *,
    limit: int = DEFAULT_CLIPS,
    sort: str = DEFAULT_SORT,
    progress: Progress = None,
) -> list[MontageClip]:
    """The ``limit`` clips a montage of ``node`` shows in ``sort`` order — the window's
    opening clips, for a script.

    Raises:
        ValueError: ``limit`` below 1, or an unknown sort.
    """
    if int(limit) < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if sort not in SORTS:
        raise ValueError(f"unknown sort {sort!r}: one of {', '.join(SORTS)}")
    if sort == "typical":
        # The head of the ranking is all that is needed, and the node cache answers it
        # without loading the fit.
        members = run.typical_members(int(node), int(limit), progress=progress)
        clips = _as_clips(experiment, run, members)
    else:
        clips = cluster_montage_members(experiment, run, node, progress=progress)
    return choose_clips(clips, count=limit, sort=sort, seed=int(node))


def _as_clips(experiment, run, members) -> list[MontageClip]:
    names = experiment.recording_names()
    state = run.state()
    return [
        MontageClip(
            rank=rank,
            recording_id=m["recording_id"],
            recording_name=names.get(m["recording_id"], m["recording_id"]),
            subject=m["subject"],
            embed_run_id=state.embed_run_for(m["recording_id"]),
            start_t_ns=int(m["start_t_ns"]),
            end_t_ns=int(m["end_t_ns"]),
            ranked=_is_ranked(m.get("typicality")),
        )
        for rank, m in enumerate(members, start=1)
    ]


def _is_ranked(typicality) -> bool:
    """A row without a typicality is ranked by its position; ``NaN`` means the method could
    not rank it."""
    return typicality is None or typicality == typicality


def montage_origin(run, node: int) -> MontageOrigin:
    """Where a montage of ``run``'s ``node`` comes from, with the node's label if it has one
    and the run's word for a node (the cluster tree's own: a temporal run's are behaviours).
    """
    from detail.api.temporal_clustering import TemporalClusteringRun

    names = run.node_names()
    return MontageOrigin(
        run_id=str(run.id),
        node=int(node),
        name=names.get(str(int(node))),
        unit="behavior" if isinstance(run, TemporalClusteringRun) else "cluster",
    )


def _cluster_clips_or_refuse(experiment, run, node, limit, sort, progress):
    clips = cluster_montage_clips(
        experiment, run, node, limit=limit, sort=sort, progress=progress
    )
    if not clips:
        raise ValueError(f"node {int(node)} has no member clips to put in a montage")
    return clips


def export_cluster_montage(
    experiment,
    run,
    node: int,
    path: Path | str,
    layers: MontageLayers = DEFAULT_LAYERS,
    *,
    blind: bool = False,
    limit: int = DEFAULT_CLIPS,
    sort: str = DEFAULT_SORT,
    max_width: int | None = None,
    fps: float | None = None,
    progress: Progress = None,
    codec: str = DEFAULT_VIDEO_CODEC,
    context_s: float = 0.0,
) -> Path:
    """Write ``node``'s montage of a clustering ``run`` to ``path`` (``.mp4`` or ``.gif``).
    ``context_s`` plays that much footage either side of each clip (:func:`with_context`);
    the labels still name each clip's own start.

    The scripted twin of the montage window's **Export video…** / **Export GIF…**: the same
    clips, order, crop, labels and clock. ``blind`` labels recordings A, B, … instead of by
    name.

    Raises:
        ValueError: the node has no clips, a bad suffix, or a bad ``layers`` / ``limit`` /
            ``sort``.
    """
    clips = _cluster_clips_or_refuse(experiment, run, node, limit, sort, progress)
    montage = ClusterMontage(experiment.dir, with_context(clips, context_s))
    try:
        return montage.export(
            path,
            layers,
            montage_labels(clips, blind=blind),
            max_width=max_width,
            fps=fps,
            progress=progress,
            codec=codec,
        )
    finally:
        montage.close()


def export_cluster_overlay(
    experiment,
    run,
    node: int,
    path: Path | str,
    *,
    blind: bool = False,
    limit: int = OVERLAY_CLIPS,
    sort: str = DEFAULT_SORT,
    size: int = OVERLAY_PX,
    fps: float | None = None,
    progress: Progress = None,
    codec: str = DEFAULT_VIDEO_CODEC,
) -> Path:
    """Write ``node``'s overlay (§22 *Overlay*) to ``path``: its ``limit`` clips in ``sort``
    order, each aligned to its heading and drawn over the others in its own colour, with its
    skeleton — an ``.mp4`` or ``.gif`` on the montage clock, or a ``.png`` (the one form for
    single-frame members, a pose cluster's). The scripted twin of the window's
    **Export overlay…**.

    Raises:
        ValueError: the node has no clips, a bad suffix, a video of single frames, or a bad
            ``limit`` / ``sort``.
    """
    clips = _cluster_clips_or_refuse(experiment, run, node, limit, sort, progress)
    montage = ClusterMontage(experiment.dir, clips)
    try:
        return montage.export_overlay(
            path,
            montage_labels(clips, blind=blind),
            size=size,
            fps=fps,
            progress=progress,
            codec=codec,
        )
    finally:
        montage.close()


def export_cluster_clip_gifs(
    experiment,
    run,
    node: int,
    out_dir: Path | str,
    layers: MontageLayers = DEFAULT_LAYERS,
    *,
    blind: bool = False,
    limit: int = DEFAULT_CLIPS,
    sort: str = DEFAULT_SORT,
    max_width: int = GIF_DEFAULT_WIDTH,
    fps: float = GIF_DEFAULT_FPS,
    progress: Progress = None,
    context_s: float = 0.0,
) -> list[Path]:
    """Write each of ``node``'s montage clips as its own GIF in ``out_dir``, named
    :func:`clip_file_stem` — the scripted twin of the window's *Export as a folder of GIFs*.
    ``context_s`` as :func:`export_cluster_montage` takes it.

    Raises:
        ValueError: the node has no clips, or a bad ``layers`` / ``limit`` / ``sort``.
    """
    clips = _cluster_clips_or_refuse(experiment, run, node, limit, sort, progress)
    aliases = blind_aliases([c.recording_id for c in clips]) if blind else {}
    montage = ClusterMontage(experiment.dir, with_context(clips, context_s))
    try:
        return montage.export_clip_gifs(
            out_dir,
            layers,
            montage_labels(clips, blind=blind, aliases=aliases or None),
            [clip_file_stem(c, aliases.get(c.recording_id)) for c in clips],
            max_width=max_width,
            fps=fps,
            progress=progress,
        )
    finally:
        montage.close()


def export_cluster_clip_list(
    experiment,
    run,
    node: int,
    path: Path | str,
    *,
    limit: int = DEFAULT_CLIPS,
    sort: str = DEFAULT_SORT,
    progress: Progress = None,
) -> Path:
    """Write ``node``'s montage clips as a CSV list (:data:`CLIP_LIST_COLUMNS`) — the scripted
    twin of the window's *Export as a CSV list*.

    Raises:
        ValueError: the node has no clips, or a bad ``limit`` / ``sort``.
    """
    clips = _cluster_clips_or_refuse(experiment, run, node, limit, sort, progress)
    montage = ClusterMontage(experiment.dir, clips)
    try:
        return montage.export_clip_list(
            path, montage_origin(run, node), progress=progress
        )
    finally:
        montage.close()
