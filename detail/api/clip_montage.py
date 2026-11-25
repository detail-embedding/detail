"""A cluster's clips written side by side: the clip montage's exports (§22 *Clip montage*).

What **Export ▾** on a clip grid writes (``gui/montage_export.py``) and what ``detail analysis
cluster-montage`` writes: one picture of many clips (an mp4 or a GIF), one GIF per clip, or a CSV
list of them. On screen the grids play through ``api.clip_tiles``; this module draws only what an
export writes, so nothing here is held between exports.

One cluster's members — :data:`DEFAULT_CLIPS` of them unless asked for more, **most typical
first** unless sorted otherwise (:data:`SORTS`) — each a tile cropped to its animal and labelled
``#rank · recording · start``, all playing **together**: every clip starts at once and each loops
from its own start the moment it ends, so no tile ever stands still; a pass is the longest clip's
length. ``rank`` is always the clip's typicality rank among the cluster's members (1 = most
typical), whatever the order the tiles are laid out in, so a tile names the same clip under every
sort.

**What a tile shows** is :class:`MontageLayers` — **Show ▾**'s layers (``show_menu.layers_of``).
The default (:data:`DEFAULT_LAYERS`) is **Mask + Mask trail**:

- **Raw** off: the masks alone and/or the keypoints alone, on white — the clip window's two side
  panes, drawn into one picture. The clip's animal is a **filled** tint.
- **Raw** on: the footage, with the clip's animal **outlined** (**Mask**) rather than tinted, so
  its fur and paws stay visible, and the keypoints drawn (**Keypoints**) on it. The outline is
  drawn at the tile's own scale, after the crop, :func:`outline_px` wide, so it reads the same in
  a 48-pixel tile and a 640-pixel GIF instead of vanishing when a 1080p frame is shrunk.
- **Mask trail** / **Keypoint trail**: the last half second of the animal's outline (filled on
  white, outlines on the footage), or of each chosen keypoint's path, fading with age. Each needs
  its parent tick.

Every picture is read through ``ClipQueryPreview.clip_frame`` — the clip window's own read — and
drawn by ``api.clip_views``' drawers, so a tile and the clip window never disagree about what a
frame holds; a tile's one addition is the outline, which the footage pane does not draw.
:attr:`MontageLayers.flavour` names the three looks in ``api.clip_render``'s words (``RAW``,
``OVERLAY``, ``MASK``), the vocabulary the rest of the tool uses for the same pictures.

**The crop** (:func:`crop_box`) is fixed for the whole clip, so a tile never jitters: the square
around the union of the animal's boxes over the clip (``ClipQueryPreview.clip_boxes``, read off
the detection rows), widened by :data:`CROP_MARGIN` either side and never tighter than
:data:`CROP_MIN_BODIES` of its typical box, so a still animal is not cropped to its outline.

**One clock, sampled at** :attr:`ClusterMontage.fps` — the footage's own rate up to
:data:`MONTAGE_MAX_FPS`. Time, not frame count, is what is synchronised: a clip's picture at
montage time ``t`` is its own frame at ``start + t`` (Tenet 8), so recordings filmed at different
rates still play at the same speed.

**Cost.** A tile is one ``clip_frame`` read at the footage's full resolution, cropped and scaled;
with **Raw** off it decodes no video (``footage=False``: the masks and keypoints are stored data)
— except for a recording whose record carries no frame size, where the montage, as the caller the
API leaves the decision to, reads the footage to size the tile. Within one loop a clip's frames
only go forward, so a frame an export samples twice in a row is drawn once. Clips from one
recording are spread over separate decode contexts so each one's frames are read forward.
:meth:`ClusterMontage.close` shuts the threads and drops the contexts and reads. Drawing before
:meth:`ClusterMontage.prepare` is refused rather than prepared behind the caller's back — the
port this replaced did that from inside a pool task and could wait on its own pool forever.

A GIF chooses one palette for all its frames (``api.animation``). **Nothing here writes to the
run**: a montage reads a clustering run and its recordings, and writes only the files an export
is pointed at.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from detail.api import clip_render
from detail.api.animation import (
    GIF,
    GIF_DEFAULT_FPS,
    GIF_DEFAULT_WIDTH,
    format_of,
    sample_offsets_ns,
    write_animation,
)
from detail.api.clip_views import (
    ClipViewSettings,
    blind_aliases,
    draw_footage,
    draw_masks_pane,
    draw_pose_pane,
    frame_turn,
    mark_no_axis,
    mirror_as_embedded,
    turn_mask,
    turn_matrix,
    turn_picture,
)
from detail.core import cpu
from detail.core import visualize
from detail.core.errors import UserInputError
from detail.core.visualize import MASK_TINT, ONION_ALPHA
from detail.storage import read_cache
from detail.storage.clip_query_csv import COLUMNS, ClipRow, write_clip_csv
from detail.storage.roots import MissingRootError

#: How many clips a montage holds unless asked for more: ``--clips`` and every API default.
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
#: The fastest a montage clock runs: fifty tiles at 30 fps would ask for 1,500 pictures a
#: second; fifteen keeps real time and halves that.
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
#: The grid's own colour behind the tiles and under the labels.
BACKGROUND = (38, 38, 38)
LABEL_INK = (235, 235, 235)
#: A tile with no picture: the recording cannot be read here, or, with Raw on, the frame did not
#: decode or the video is not on this machine (Raw off decodes no video, so it needs none).
NO_PICTURE = (90, 90, 90)
#: Pixels between tiles.
GAP_PX = 2

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
    #: Whether the method ranked this clip. A clip it could not (a bout assigned after a medoid
    #: fit, never compared) still has a rank, after every ranked one, and sorts last in either
    #: typicality order rather than leading the least typical (``ClipTable.order``).
    ranked: bool = True


@dataclass(frozen=True)
class MontageOrigin:
    """Where a montage's clips came from: the run, the node, and what the node is called —
    written into a clip list, which names its one query after the node."""

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
    """What every tile draws — **Show ▾**'s layers (``show_menu.layers_of``). The defaults are
    the mask and its trail on white.

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
    #: **Aligned**: each tile frame turned about the animal's middle so its body points up
    #: (``clip_views.turn_matrix``, the clip window's and the live montage's own turn), the crop
    #: kept at its as-recorded size; a side-view (``mirror``) run's tile is the as-recorded one,
    #: mirrored where its embedding mirrored the frame (``clip_views.mirror_as_embedded``), never
    #: turned. A frame with no body axis (on a side view, no embedded row) is drawn as recorded
    #: and marked; a run with no keypoints or heading definition draws every frame as recorded.
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

    @property
    def flavour(self) -> str:
        """The look in ``api.clip_render``'s words: ``RAW`` (the footage alone), ``OVERLAY``
        (the footage with the mask outline and/or the keypoints on it), ``MASK`` (stored data
        on white)."""
        if not self.raw:
            return clip_render.MASK
        return clip_render.OVERLAY if (self.mask or self.keypoints) else clip_render.RAW

    @property
    def settings(self) -> ClipViewSettings:
        """The read these layers need, as the clip window's own settings: the mask is never
        tinted onto the footage here (a tile outlines it at its own scale), the trails are read
        when a tick asks for them, and the pose is never body-aligned (an **Aligned** tile is
        turned after it is drawn, so the pose turns with everything else). ``aligned`` is
        carried so the read fetches a side-view run's flip (``ClipFrame.flip``); the drawers
        here never read it."""
        return ClipViewSettings(
            masks_on_frame=False,
            points_on_frame=self.raw and self.keypoints,
            trail=self.keypoint_trail,
            trail_points=self.trail_points,
            mask_trail=bool(self.mask_trail),
            pose_aligned=False,
            aligned=self.aligned,
        )


#: What a montage draws unless told otherwise — the CLI and every API default.
DEFAULT_LAYERS = MontageLayers()


# -- labels and names ------------------------------------------------------------------- #
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

    ``aliases`` lets a caller letter the recordings over a larger set than ``clips``, so
    recording A stays A under every sort.
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


# -- which clips ------------------------------------------------------------------------ #
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
    seed: int = 0,
) -> list[MontageClip]:
    """The clips a montage of ``members`` shows: ``members`` in ``sort`` order, the first
    ``count`` of them.
    ``seed`` is the ``random`` order's (:func:`sort_clips`).

    Raises:
        ValueError: ``count`` below 1, or an unknown sort.
    """
    if int(count) < 1:
        raise ValueError(f"a montage shows at least 1 clip, got {count}")
    return sort_clips(members, sort, seed)[: int(count)]


# -- the crop, the grid --------------------------------------------------------------- #
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


def outline_px(tile_px: int) -> int:
    """How wide the mask's outline is drawn in a tile of ``tile_px``: 2 pixels up to a
    160-pixel tile, then one more per 64 — so it stays a visible line in a 48-pixel tile and
    does not become a thread in a 640-pixel GIF."""
    return max(2, int(round(int(tile_px) / 64)))


def font_px(tile_px: int) -> int:
    """The label's font size for a tile of ``tile_px``."""
    return int(min(max(round(tile_px * 0.075), 9), 15))


def label_px(tile_px: int) -> int:
    """The height of the label strip under a tile of ``tile_px``."""
    return font_px(tile_px) + 6


def grid_size(columns: int, rows: int, tile_px: int) -> tuple[int, int]:
    """The ``(width, height)`` of a :func:`compose` grid: ``columns`` tiles of ``tile_px``
    with their label strips, :data:`GAP_PX` between and around them. **The one grid
    formula**: :func:`compose` and :func:`tile_origin` both measure with it, so an export's
    frame is exactly the size its tiles are placed in."""
    columns, rows = max(int(columns), 1), max(int(rows), 1)
    return (
        columns * tile_px + GAP_PX * (columns + 1),
        rows * (tile_px + label_px(tile_px)) + GAP_PX * (rows + 1),
    )


def tile_origin(index: int, *, columns: int, tile_px: int) -> tuple[int, int]:
    """Where tile ``index``'s picture starts in a :func:`compose` grid, ``(x, y)`` pixels."""
    row, column = divmod(int(index), max(int(columns), 1))
    cell_h = tile_px + label_px(tile_px)
    return GAP_PX + column * (tile_px + GAP_PX), GAP_PX + row * (cell_h + GAP_PX)


def export_grid(count: int, max_width: int | None) -> tuple[int, int]:
    """An export's ``(columns, tile_px)``: as near square as ``count`` allows, tiles sized to
    fill ``max_width`` — or :data:`EXPORT_TILE_PX` each, no wider than
    :data:`EXPORT_MAX_WIDTH` in all, when no width is asked for."""
    columns = max(math.ceil(math.sqrt(max(int(count), 1))), 1)
    room = int(max_width) if max_width else EXPORT_MAX_WIDTH
    tile_px = max((room - GAP_PX * (columns + 1)) // columns, 16)
    if not max_width:
        tile_px = min(tile_px, EXPORT_TILE_PX)
    return columns, int(tile_px)


def compose(
    tiles: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    columns: int,
    tile_px: int,
) -> np.ndarray:
    """Lay ``tiles`` (RGB, any size — scaled to ``tile_px``) out ``columns`` wide, each with
    its label under it, on the grid colour: one frame of an export."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    count = len(tiles)
    columns = max(min(int(columns), max(count, 1)), 1)
    rows = math.ceil(max(count, 1) / columns)
    width, height = grid_size(columns, rows, tile_px)
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
    font = ImageFont.load_default(size=font_px(tile_px))
    for index, label in enumerate(labels[:count]):
        x, y = tile_origin(index, columns=columns, tile_px=tile_px)
        text = fit_label(label, lambda s: draw.textlength(s, font=font), tile_px - 2)
        draw.text((x + 1, y + tile_px + 2), text, fill=LABEL_INK, font=font)
    return np.asarray(picture)


def fit_label(label: str, measure, width: float) -> str:
    """``label`` as it is when it fits ``width``, else with the recording's name cut short
    (``#3 · mouse_… · 0:42.1``): the rank and the time are what tell two tiles apart."""
    if measure(label) <= width:
        return label
    parts = label.split(" · ")
    if len(parts) == 3:
        rank, name, when = parts
        for keep in range(len(name) - 1, 0, -1):
            text = f"{rank} · {name[:keep]}… · {when}"
            if measure(text) <= width:
                return text
        text = f"{rank} · {when}"
        if measure(text) <= width:
            return text
        return rank
    return label


# -- the montage ---------------------------------------------------------------------- #
@dataclass
class _Prepared:
    """One clip, resolved: its frames, their times, its animal's boxes over them, how long one
    loop of it lasts (``cycle_ns``: its length plus one frame step, so its last frame shows for
    a step before it starts again), and whether its tiles need the footage even with Raw off
    (no recorded frame size)."""

    first: int
    t_ns: np.ndarray
    boxes: np.ndarray
    fps: float
    cycle_ns: int = 1
    needs_footage: bool = False
    crop: tuple[int, int, int, int] | None = None


@dataclass
class _Shared:
    """What a montage's threads share: decode contexts, per-clip reads, their locks and the
    drawing pool."""

    dir: Path
    workers: int
    previews: list = field(default_factory=list)
    previews_guard: threading.Lock = field(default_factory=threading.Lock)
    locks: dict = field(default_factory=dict)
    locks_guard: threading.Lock = field(default_factory=threading.Lock)
    prepared: dict = field(default_factory=dict)
    pool: object = None
    pool_guard: threading.Lock = field(default_factory=threading.Lock)

    def preview(self, slot: int):
        from detail.api.run_preview import ClipQueryPreview

        with self.previews_guard:
            while len(self.previews) <= slot:
                self.previews.append(ClipQueryPreview(self.dir))
            return self.previews[slot]


class ClusterMontage:
    """Write a cluster montage's exports: one picture of every clip, a GIF per clip, a CSV list.

    Built in two steps: construct (nothing is read), then :meth:`prepare` (reads every clip's
    timestamps and boxes, the one up-front cost, with progress); each export prepares first.
    :meth:`tile` is safe to call from several threads at once: each decode context is locked
    while it is used. Drawing **before**
    ``prepare`` raises rather than preparing on the caller's behalf: a prepare from inside a
    pool task waits on its own pool.
    """

    def __init__(
        self,
        experiment_dir: Path | str,
        clips: Sequence[MontageClip],
        *,
        workers: int | None = None,
    ):
        """``workers`` threads draw tiles side by side (``core.cpu.thread_workers()`` unless
        given): most of a cold tile is reading — a recording's metadata, its keypoints, a seek
        into its video — which a thread waits on without holding the interpreter.

        Raises:
            ValueError: no clips.
        """
        count = cpu.thread_workers() if workers is None else max(int(workers), 1)
        clips = list(clips)
        if not clips:
            raise ValueError("a montage needs at least one clip")
        self._shared = _Shared(Path(experiment_dir), count)
        self.clips = clips
        # Clips from one recording go to different decode contexts (`slot`): a context holds one
        # reader per recording, and two clips sharing it would seek it back and forth every step.
        self._slot: list[int] = []
        seen: dict[str, int] = {}
        for clip in clips:
            self._slot.append(seen.get(clip.recording_id, 0))
            seen[clip.recording_id] = self._slot[-1] + 1

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
        """Stop the threads and drop the decode contexts and the per-clip reads, so a finished
        export holds nothing."""
        shared = self._shared
        with shared.pool_guard:
            pool, shared.pool, shared.workers = shared.pool, None, 1
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        with shared.previews_guard:
            shared.previews.clear()
        shared.prepared.clear()
        with shared.locks_guard:
            shared.locks.clear()

    # -- preparing ---------------------------------------------------------------- #
    def prepare(self, progress: Progress = None) -> None:
        """Read every clip's frame times and its animal's boxes; idempotent."""
        prepared = self._shared.prepared
        missing = [i for i, c in enumerate(self.clips) if c not in prepared]
        if not missing:
            return
        done = [0]
        done_guard = threading.Lock()
        total = len(missing)

        def one(index: int) -> None:
            # Pinned: a clip's reads ask its recording's record several times over (the decode
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
        rec, emb = clip.recording_id, clip.embed_run_id
        try:
            with self._lock(index):
                times = preview.timestamps(rec, emb)
                first = preview.t_ns_to_frame(rec, emb, clip.start_t_ns)
                last = max(preview.t_ns_to_frame(rec, emb, clip.end_t_ns), first)
                fps = preview.frame_rate(rec, emb)
                boxes = preview.clip_boxes(rec, emb, clip.subject, first, last)
                needs_footage = preview.frame_size(rec) is None
        except _UNREADABLE:
            # A clip whose recording cannot be read still takes its place in the grid, as one
            # grey tile, rather than taking the montage down with it.
            return _Prepared(
                -1, np.zeros(1, np.int64), np.full((1, 4), np.nan), MONTAGE_MAX_FPS
            )
        t_ns = np.asarray(times[first : last + 1], dtype=np.int64)
        if not len(t_ns):
            t_ns = np.asarray([int(clip.start_t_ns)], dtype=np.int64)
        step = (
            int(np.median(np.diff(t_ns)))
            if len(t_ns) > 1
            else int(round(1e9 / fps)) if fps > 0 else 1
        )
        cycle = int(t_ns[-1] - t_ns[0]) + max(step, 1)
        return _Prepared(int(first), t_ns, boxes, float(fps), cycle, needs_footage)

    def _ready(self) -> list[_Prepared]:
        """Every clip's preparation.

        Raises:
            RuntimeError: a clip has not been prepared — call :meth:`prepare` first.
        """
        prepared = self._shared.prepared
        missing = [c for c in self.clips if c not in prepared]
        if missing:
            raise RuntimeError(
                f"call prepare() before drawing: {len(missing)} of {len(self.clips)} clips "
                "have not been read"
            )
        return [prepared[c] for c in self.clips]

    def _prep(self, index: int) -> _Prepared:
        found = self._shared.prepared.get(self.clips[index])
        if found is None:
            raise RuntimeError(
                f"call prepare() before drawing: clip {index} has not been read"
            )
        return found

    def _preview(self, index: int):
        return self._shared.preview(self._slot[index])

    def _lock(self, index: int) -> threading.Lock:
        key = (self._slot[index], self.clips[index].recording_id)
        shared = self._shared
        with shared.locks_guard:
            return shared.locks.setdefault(key, threading.Lock())

    # -- the clock ---------------------------------------------------------------- #
    @property
    def fps(self) -> float:
        """The montage clock's rate: the footage's (median over clips), at most 15."""
        rates = [p.fps for p in self._ready() if p.fps > 0]
        rate = float(np.median(rates)) if rates else MONTAGE_MAX_FPS
        return min(rate, MONTAGE_MAX_FPS)

    @property
    def duration_ns(self) -> int:
        """The longest clip's length: how long one pass of the montage lasts, within which a
        shorter clip loops."""
        return max(int(p.t_ns[-1] - p.t_ns[0]) for p in self._ready())

    def frame_at(self, index: int, offset_ns: int) -> int:
        """Clip ``index``'s global frame at montage time ``offset_ns``: its own frame at
        ``start + (offset mod its cycle)`` — the clip loops from its start the moment it ends,
        so no tile is ever still. A clip the montage clock samples more coarsely than its own
        rate shows the frames that fall under the samples, as any resampling does."""
        clip = self._prep(index)
        if clip.first < 0:
            return -1
        target = int(clip.t_ns[0]) + int(offset_ns) % max(int(clip.cycle_ns), 1)
        position = int(np.searchsorted(clip.t_ns, target, side="right")) - 1
        return clip.first + int(min(max(position, 0), len(clip.t_ns) - 1))

    def frame_span(self, index: int) -> tuple[int, int]:
        """Clip ``index``'s first and last global frame, ``(-1, -1)`` when it cannot be read."""
        clip = self._prep(index)
        if clip.first < 0:
            return (-1, -1)
        return clip.first, clip.first + len(clip.t_ns) - 1

    # -- tiles -------------------------------------------------------------------- #
    def tile(
        self, index: int, frame: int, layers: MontageLayers, size: int
    ) -> np.ndarray:
        """Clip ``index`` at global ``frame``, drawn per ``layers``, cropped and scaled to a
        ``size`` square; the grey no-picture tile when it cannot be read here."""
        clip = self.clips[index]
        prepared = self._prep(index)
        if frame < 0:
            return blank_tile(size, NO_PICTURE, "video not found")
        preview = self._preview(index)
        try:
            with self._lock(index):
                read = preview.clip_frame(
                    clip.recording_id,
                    clip.embed_run_id,
                    clip.subject,
                    frame,
                    layers.settings,
                    footage=layers.raw or prepared.needs_footage,
                )
        except _UNREADABLE + (IndexError,):
            read = None
        if read is None:
            return blank_tile(size, NO_PICTURE, "no picture")
        picture = _picture(read, layers, clip.subject)
        height, width = picture.shape[:2]
        if prepared.crop is None:
            prepared.crop = crop_box(prepared.boxes, (width, height))
        x0, y0, x1, y1 = prepared.crop
        fill = (0, 0, 0) if layers.raw else (255, 255, 255)
        if layers.aligned and read.axis is not None:
            if read.flip is not None:
                tile = self._as_recorded(
                    read, clip.subject, picture, layers, prepared.crop, size, fill
                )
                mirrored = mirror_as_embedded(tile, read.flip)
                return mark_no_axis(tile) if mirrored is None else mirrored
            ends = frame_turn(read, int(clip.subject))
            if ends is None:
                return mark_no_axis(
                    self._as_recorded(
                        read, clip.subject, picture, layers, prepared.crop, size, fill
                    )
                )
            matrix = turn_matrix(
                ends[0], ends[1], max(x1 - x0, y1 - y0, 1), (size, size)
            )
            tile = turn_picture(picture, matrix, (size, size), fill)
            if layers.raw and layers.mask:
                tile = _outline_turned(
                    tile,
                    read.subject_mask(clip.subject),
                    read.mask_history or [],
                    matrix,
                    (height, width),
                )
            return tile
        return self._as_recorded(
            read, clip.subject, picture, layers, prepared.crop, size, fill
        )

    @staticmethod
    def _as_recorded(
        read, subject, picture, layers, crop, size: int, fill
    ) -> np.ndarray:
        """The tile cut from ``picture`` by the clip's fixed crop, squared and scaled; the
        animal outlined at tile size on the footage."""
        import cv2

        height, width = picture.shape[:2]
        x0, y0, x1, y1 = crop
        cut = picture[y0:y1, x0:x1]
        side = max(cut.shape[0], cut.shape[1], 1)
        square = np.empty((side, side, 3), dtype=np.uint8)
        square[:] = fill
        oy, ox = (side - cut.shape[0]) // 2, (side - cut.shape[1]) // 2
        square[oy : oy + cut.shape[0], ox : ox + cut.shape[1]] = cut
        tile = cv2.resize(
            square,
            (size, size),
            interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR,
        )
        if layers.raw and layers.mask:
            tile = _outline_onto(
                tile,
                read.subject_mask(subject),
                read.mask_history or [],
                (x0, y0, x1, y1),
                (oy, ox),
                side,
                (height, width),
            )
        return tile

    def _export_tile(
        self, index: int, frame: int, layers: MontageLayers, size: int, held: dict
    ) -> np.ndarray:
        """Clip ``index`` at ``frame`` for an export, drawn at the export's own size.

        ``held`` is the export's memory of each clip's last tile: within one loop a clip's
        frames only go forward, so a repeated frame is the previous one; a clip that has looped
        round within the pass is drawn again. Nothing is kept past the export.
        """
        last = held.get(index)
        if last is not None and last[0] == frame:
            return last[1]
        tile = self.tile(index, int(frame), layers, int(size))
        held[index] = (frame, tile)
        return tile

    def export(
        self,
        path: Path | str,
        layers: MontageLayers,
        labels: Sequence[str],
        *,
        max_width: int | None = None,
        fps: float | None = None,
        progress: Progress = None,
    ) -> Path:
        """Write one pass of the montage to ``path`` (``.mp4`` or ``.gif``).

        ``fps`` defaults to the montage clock for an mp4 and to :data:`GIF_DEFAULT_FPS` for a
        GIF; ``max_width`` to a grid of :data:`EXPORT_TILE_PX` tiles for an mp4 (no wider than
        :data:`EXPORT_MAX_WIDTH`) and to :data:`GIF_DEFAULT_WIDTH` for a GIF — the GIF dialog's
        own defaults, so a script and **Export ▾** write the same file. The grid is
        :func:`export_grid`'s, and every frame samples real time at ``fps``.
        """
        kind = format_of(path)
        if max_width is None and kind == GIF:
            max_width = GIF_DEFAULT_WIDTH
        self.prepare(progress)
        rate = float(
            fps if fps is not None else (GIF_DEFAULT_FPS if kind == GIF else self.fps)
        )
        offsets = sample_offsets_ns(self.duration_ns, rate)
        columns, tile_px = export_grid(len(self.clips), max_width)
        width = int(max_width) if max_width else None

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
            )
        finally:
            frames.close()

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
        replace: bool = False,
    ) -> list[Path]:
        """Write each clip as a GIF of its own into ``out_dir``: ``<stem>.gif``, one pass of
        that clip at ``fps``, its tile ``max_width`` wide with its label under it.

        GIFs already in ``out_dir`` under these names are refused unless ``replace`` — a
        folder is chosen, the names in it are not, so writing over them is never assumed
        (#29). :func:`existing_gifs` is the check, for a surface that asks first.

        ``stems`` are the file names (see :func:`clip_file_stem`), one per clip. The montage's
        threads write a clip each, side by side; one thread draws all of a clip from its own
        decode context, so its frames are still read forward. The bar is weighed by the frames
        each GIF holds, not by the GIF count, so a long clip moves it as far as it costs. A
        stopped export (the progress callback raising) keeps the GIFs already finished —
        ``on_written`` has heard of each — and removes the ones it was writing
        (``api.animation``). Returned in the clips' order, whatever order they finished in.

        Raises:
            ValueError: ``stems`` and the clips differ in number, or two stems are the same.
            UserInputError: a GIF of one of these names is already there, and not ``replace``.
        """
        if len(stems) != len(self.clips) or len({s.casefold() for s in stems}) != len(
            stems
        ):
            raise ValueError("one distinct file name per clip is needed")
        there = existing_gifs(out_dir, stems)
        if there and not replace:
            raise UserInputError(
                f"{len(there):,} of these GIFs already exist in {out_dir} "
                f"({', '.join(there[:3])}{', …' if len(there) > 3 else ''}); choose another "
                "folder, or remove them first"
            )
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.prepare(progress)
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
        """Write the clips as a CSV list, one row per clip in the montage's order: a clip-query
        CSV (``storage/clip_query_csv``, every clip an example of the one query named after the
        node, so ``detail analysis query --from-csv`` takes it as example clips) with the
        montage's own columns after it (:data:`MONTAGE_LIST_COLUMNS`). Reading each clip's
        frame span is the one cost."""
        self.prepare(progress)
        rows, extra = [], {name: [] for name in MONTAGE_LIST_EXTRA}
        for index, clip in enumerate(self.clips):
            first, last = self.frame_span(index)
            rows.append(
                ClipRow(
                    kind="query",
                    rank=0,
                    recording_id=clip.recording_id,
                    track_id=int(clip.subject),
                    embed_run_id=clip.embed_run_id,
                    start_t_ns=int(clip.start_t_ns),
                    end_t_ns=int(clip.end_t_ns),
                    start_frame=int(first),
                    end_frame=int(last),
                    score=0.0,
                    query_id=f"node{int(origin.node)}",
                    query_name=origin.behavior_name,
                    example_idx=index,
                )
            )
            extra["rank_typical"].append(int(clip.rank))
            extra["recording_name"].append(clip.recording_name)
            extra["subject"].append(clip.subject)
            extra["node"].append(int(origin.node))
            extra["run_id"].append(origin.run_id)
            extra["start_s"].append(f"{clip.start_t_ns / 1e9:.3f}")
            extra["end_s"].append(f"{clip.end_t_ns / 1e9:.3f}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_clip_csv(path, rows, extra=extra)
        return path


def existing_gifs(out_dir: Path | str, stems: Sequence[str]) -> list[str]:
    """Which ``<stem>.gif`` of ``stems`` are already in ``out_dir``, by name."""
    out = Path(out_dir)
    return [f"{stem}.gif" for stem in stems if (out / f"{stem}.gif").exists()]


#: The clip list's own columns, after the clip-query ones: the clip's typicality rank (the
#: clip-query ``rank`` is a match rank, 0 for an example), its recording's name, the subject,
#: where the clips came from, and the times in seconds for a reader.
MONTAGE_LIST_EXTRA = (
    "rank_typical",
    "recording_name",
    "subject",
    "node",
    "run_id",
    "start_s",
    "end_s",
)
MONTAGE_LIST_COLUMNS = COLUMNS + list(MONTAGE_LIST_EXTRA)


# -- drawing helpers ------------------------------------------------------------------ #
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


def _picture(read, layers: MontageLayers, track_id) -> np.ndarray:
    """The one picture a tile is cut from, per ``layers``: the clip window's own drawers over
    the frame it read (see the module docstring)."""
    settings = layers.settings
    if layers.raw:
        return draw_footage(read, settings, int(track_id))
    masks = draw_masks_pane(read, settings, int(track_id)) if layers.mask else None
    pose = draw_pose_pane(read, settings, int(track_id)) if layers.keypoints else None
    if masks is not None and pose is not None:
        # The keypoints over the masks: both panes are drawn on white, so whatever the pose pane
        # inked is laid over the masks pane.
        inked = (pose < 250).any(axis=2)
        out = masks.copy()
        out[inked] = pose[inked]
        return out
    if pose is not None:
        return pose
    if masks is not None:
        return masks
    # Keypoints asked for, and the run has none: a white page the size of the frame.
    return np.full((*read.frame_hw, 3), 255, dtype=np.uint8)


def _outline_onto(tile, subject_mask, history, crop, offset, side, frame_hw):
    """``tile`` with the clip's animal outlined in the mask tint, and its trail as fading
    outlines under it — drawn at the tile's own size (:func:`outline_px`), after the crop.
    The fade is ``visualize.fade_weights``' (the one fade rule), the ring
    ``visualize.mask_outline``'s, the blend ``visualize.tint_by_weight``'s.
    """
    size = tile.shape[0]
    thickness = outline_px(size)
    rings = []
    for layer in history:  # oldest first, as fade_weights takes them
        placed = _into_tile(layer, crop, offset, side, size) if layer else None
        rings.append(
            None
            if placed is None
            else (0, 0, visualize.mask_outline(placed, thickness))
        )
    weight = np.zeros((size, size), dtype=np.float32)
    faded = visualize.fade_weights(rings, alpha=ONION_ALPHA)
    if faded is not None:
        part, top, left = faded
        weight[top : top + part.shape[0], left : left + part.shape[1]] = part
    if subject_mask is not None and tuple(subject_mask.shape[:2]) == tuple(frame_hw):
        now = _into_tile((0, 0, subject_mask), crop, offset, side, size)
        if now is not None:
            # The past outlines stop at the animal as it is now, so none crosses its fur.
            weight[now] = 0.0
            weight[visualize.mask_outline(now, thickness)] = 1.0
    if not weight.any():
        return tile
    return visualize.tint_by_weight(tile, weight, MASK_TINT)


def _outline_turned(tile, subject_mask, history, matrix, frame_hw) -> np.ndarray:
    """:func:`_outline_onto` for an **Aligned** tile: the outline and its trail placed by the
    tile's turn (``clip_views.turn_mask``) instead of its crop."""
    size = tile.shape[0]
    thickness = outline_px(size)
    rings = []
    for layer in history:
        placed = (
            turn_mask(layer[2], matrix, (size, size), at=(layer[0], layer[1]))
            if layer
            else None
        )
        rings.append(
            None
            if placed is None
            else (0, 0, visualize.mask_outline(placed, thickness))
        )
    weight = np.zeros((size, size), dtype=np.float32)
    faded = visualize.fade_weights(rings, alpha=ONION_ALPHA)
    if faded is not None:
        part, top, left = faded
        weight[top : top + part.shape[0], left : left + part.shape[1]] = part
    if subject_mask is not None and tuple(subject_mask.shape[:2]) == tuple(frame_hw):
        now = turn_mask(subject_mask, matrix, (size, size))
        if now is not None:
            weight[now] = 0.0
            weight[visualize.mask_outline(now, thickness)] = 1.0
    if not weight.any():
        return tile
    return visualize.tint_by_weight(tile, weight, MASK_TINT)


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


def blank_tile(size: int, colour, text: str = "") -> np.ndarray:
    """A plain tile, saying why it is plain."""
    tile = np.empty((size, size, 3), dtype=np.uint8)
    tile[:] = colour
    if not text:
        return tile
    from PIL import Image, ImageDraw, ImageFont

    picture = Image.fromarray(tile, "RGB")
    draw = ImageDraw.Draw(picture)
    font = ImageFont.load_default(size=font_px(size))
    width = draw.textlength(text, font=font)
    draw.text(
        ((size - width) / 2, size / 2 - font_px(size) / 2),
        text,
        fill=LABEL_INK,
        font=font,
    )
    return np.asarray(picture)


# -- the cluster's clips --------------------------------------------------------------- #
def montage_members(
    experiment, run, node: int, *, progress: Progress = None
) -> list[MontageClip]:
    """Every one of ``node``'s clips, most typical first, as montage tiles —
    ``run.typical_members``, so every node with members has its clips; :func:`montage_clips`
    chooses from them (:func:`choose_clips`)."""
    members = run.typical_members(int(node), None, progress=progress)
    return as_clips(experiment, run, members)


def as_clips(experiment, run, members) -> list[MontageClip]:
    """``NodeView.ranked`` rows as montage clips, ranked 1.. in the order given."""
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


def montage_clips(
    experiment,
    run,
    node: int,
    *,
    limit: int = DEFAULT_CLIPS,
    sort: str = DEFAULT_SORT,
    progress: Progress = None,
) -> list[MontageClip]:
    """The ``limit`` clips a montage of ``node`` shows in ``sort`` order, for a script or
    the CLI. Every order is drawn from the node's clips as the node
    cache lists them; ``random`` is seeded by the node, so it is the same on every call.

    Raises:
        ValueError: ``limit`` below 1, or an unknown sort.
    """
    if int(limit) < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if sort not in SORTS:
        raise ValueError(f"unknown sort {sort!r}: one of {', '.join(SORTS)}")
    if sort == "typical":
        members = run.typical_members(int(node), int(limit), progress=progress)
        clips = as_clips(experiment, run, members)
    else:
        clips = montage_members(experiment, run, node, progress=progress)
    return choose_clips(clips, count=limit, sort=sort, seed=int(node))


@contextlib.contextmanager
def cluster_montage(
    experiment,
    run,
    node: int,
    *,
    limit: int = DEFAULT_CLIPS,
    sort: str = DEFAULT_SORT,
    context_s: float = 0.0,
    progress: Progress = None,
) -> Iterator[tuple[ClusterMontage, list[MontageClip]]]:
    """A prepared :class:`ClusterMontage` of :func:`montage_clips`' clips, and the clips
    themselves — what a script or the CLI exports from — closed on the way out.

    ``context_s`` plays that much footage either side of each clip (:func:`with_context`);
    the clips handed back are the clips as they are, for labelling.

    Raises:
        ValueError: ``limit`` below 1, an unknown sort, or a node with no member clips.
    """
    clips = montage_clips(
        experiment, run, node, limit=limit, sort=sort, progress=progress
    )
    if not clips:
        raise ValueError(f"node {int(node)} has no member clips to put in a montage")
    montage = ClusterMontage(experiment.dir, with_context(clips, context_s))
    try:
        montage.prepare(progress)
        yield montage, clips
    finally:
        montage.close()
