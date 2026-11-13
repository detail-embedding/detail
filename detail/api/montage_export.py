"""Export a cluster montage's clips in several looks and formats at once (§22 *Clip montage*,
*Export clips*): the montage window's one **Export clips…** dialog and ``detail analysis
cluster-montage --export-dir``.

**What one export writes.** Each ticked *visual* (a look: :data:`VISUALS`, or the window's
current ticks) is its own set of files in its own subfolder of the folder chosen, named for the
visual (:func:`visual_folder`). In it go, per ticked format (MP4 and/or GIF, each with its own
width and frame rate), the **montage file** — every clip in one grid, ``montage.mp4`` — and/or
**one file per clip**, ``rank_recording_m-ss.t.mp4`` (:func:`~detail.api.clip_montage.clip_file_stem`;
Blind mode's letters when the caller labels blind). :class:`ExportPlan` is that list, built
before anything is drawn, so the count a window shows is the count written.

**Nothing new is drawn here.** Every file goes through the montage's own writers, on its decode
contexts and threads: a montage file or a per-clip mp4 is :meth:`ClusterMontage.export`, and a
look's per-clip GIFs are one :meth:`ClusterMontage.export_clip_gifs` batch, written side by side.
So whatever makes the window's exports faster makes this faster too, and this keeps no tiles of
its own. What this module adds is the
plan, the folder layout, one bar over the whole thing weighed by what each file costs, and an
estimate of that cost.

**Full size.** An mp4's width defaults to *full size*: each clip's tile at the size its crop has
in the recording's own pixels (:func:`native_tile_px`), so nothing is shrunk. A montage file's
tiles are one size, the largest crop's, as long as the grid stays within
:data:`~detail.api.clip_montage.EXPORT_MAX_WIDTH` — past that the tiles shrink to fit, since an
encoder asked for a 5,000-pixel frame refuses it. A GIF is always written at a width chosen,
640 px unless told otherwise.

**The estimate is approximate and says so.** :func:`estimate_export` multiplies the frames each
file holds by per-tile and per-pixel costs (:class:`ExportCosts`) measured on a real recording,
and checked against a whole real export (see the class). Treat it as "about": footage that
moves more compresses worse, and a busy machine draws slower.

**Cancelling** (the progress callback raising, as the GUI's Cancel does) keeps every file already
finished — ``on_written`` has heard of each — and removes the one being written
(``api.animation.write_animation``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from detail.api.animation import (
    GIF,
    GIF_DEFAULT_FPS,
    GIF_DEFAULT_WIDTH,
    GIF_FPS_RANGE,
    GIF_WIDTH_RANGE,
    MP4,
    sample_offsets_ns,
)
from detail.api.clip_montage import (
    DEFAULT_CLIPS,
    DEFAULT_LAYERS,
    DEFAULT_SORT,
    EXPORT_MAX_WIDTH,
    GAP_PX,
    ClusterMontage,
    MontageClip,
    MontageLayers,
    _file_safe,
    blind_aliases,
    clip_file_stem,
    crop_box,
    label_px,
    montage_labels,
)
from detail.core.video_codecs import DEFAULT_VIDEO_CODEC
from detail.core.video_codecs import VIDEO_CODECS

Progress = Callable[[str, float, str], None] | None

#: The visual named for the window's ticks as they are (the CLI's tick options).
CURRENT = "current"
#: What to write: the montage file, one file per clip, or both.
WRITE_MONTAGE = "montage"
WRITE_CLIPS = "clips"
WRITE_BOTH = "both"
WRITES: dict[str, str] = {
    WRITE_MONTAGE: "One montage file (all clips in a grid)",
    WRITE_CLIPS: "One file per clip",
    WRITE_BOTH: "Both",
}
DEFAULT_WRITE = WRITE_BOTH
#: The montage file's name in each visual's folder (a clip's always starts with its rank).
MONTAGE_STEM = "montage"
#: A clip whose recording's size cannot be read is drawn this wide at "full size".
FALLBACK_TILE_PX = 640


@dataclass(frozen=True)
class Visual:
    """One look an export can write: which of the montage's five ticks are on."""

    key: str
    label: str
    raw: bool = False
    mask: bool = False
    keypoints: bool = False
    mask_trail: bool = False
    keypoint_trail: bool = False

    @property
    def needs_keypoints(self) -> bool:
        return self.keypoints

    def layers(
        self, trail_points: Sequence[str] | None = None, *, align: bool = False
    ) -> MontageLayers:
        """The look as ``MontageLayers``; ``trail_points`` choose which keypoints trail, and
        ``align`` turns every tile to its heading (§22 *Align to heading*)."""
        return MontageLayers(
            raw=self.raw,
            mask=self.mask,
            keypoints=self.keypoints,
            mask_trail=self.mask_trail,
            keypoint_trail=self.keypoint_trail,
            trail_points=(
                tuple(trail_points)
                if self.keypoint_trail and trail_points is not None
                else None
            ),
            aligned=bool(align),
        )


#: What a look's name says when its tiles are turned to their heading.
ALIGNED = "aligned to heading"
#: The fixed looks the dialog lists under the current one, in its order (the user's list,
#: 2026-10-23). "Mask outline" is the mask on the footage, which the montage draws as a line.
VISUALS: tuple[Visual, ...] = (Visual("raw", "Raw footage", raw=True), Visual(
    "raw-outline", "Raw + mask outline", raw=True, mask=True
), Visual(
    "raw-outline-trail",
    "Raw + mask outline + mask trail",
    raw=True,
    mask=True,
    mask_trail=True,
), Visual(
    "raw-keypoints", "Raw + keypoints", raw=True, keypoints=True
), Visual(
    "raw-keypoints-trail",
    "Raw + keypoints + keypoint trail",
    raw=True,
    keypoints=True,
    keypoint_trail=True,
), Visual(
    "mask", "Mask", mask=True
), Visual(
    "mask-trail", "Mask + mask trail", mask=True, mask_trail=True
), Visual(
    "keypoints", "Keypoints", keypoints=True
), Visual(
    "keypoints-trail",
    "Keypoints + keypoint trail",
    keypoints=True,
    keypoint_trail=True,
    ),  # fmt: skip
    Visual(
        "mask-keypoints-trails",
        "Mask + keypoints + both trails",
        mask=True,
        keypoints=True,
        mask_trail=True,
        keypoint_trail=True,
    ),
)
VISUALS_BY_KEY: dict[str, Visual] = {v.key: v for v in VISUALS}
#: Every visual a caller can name: the current ticks, then the fixed looks.
VISUAL_CHOICES: tuple[str, ...] = (CURRENT, *VISUALS_BY_KEY)
#: How many trail-point names a label spells out before it counts them instead.
_NAMED_POINTS = 3


def describe_layers(layers: MontageLayers, point_count: int | None = None) -> str:
    """The montage's ticks in words — ``Mask + Mask trail``, or with chosen trail points
    ``Keypoints + Keypoint trail: nose, tail_base`` (``3 of 8 points`` past three names;
    ``point_count`` is the skeleton's size, when known)."""
    parts = [
        name
        for name, on in (
            ("Raw", layers.raw),
            ("Mask", layers.mask),
            ("Keypoints", layers.keypoints),
            ("Mask trail", layers.mask_trail),
            ("Keypoint trail", layers.keypoint_trail),
        )
        if on
    ]
    text = " + ".join(parts)
    points = layers.trail_points
    if layers.keypoint_trail and points is not None:
        if len(points) <= _NAMED_POINTS:
            text += ": " + (", ".join(points) if points else "no points")
        else:
            of = f" of {point_count}" if point_count else ""
            text += f": {len(points)}{of} points"
    if layers.aligned:
        text += f", {ALIGNED}"
    return text


def visual_folder(label: str, *, current: bool = False) -> str:
    """The subfolder a visual's files go in: its label made safe for any file system —
    ``Raw_mask_outline``; the current ticks' as ``current_Mask_Mask_trail``, so it never shares
    a folder with the fixed look it may equal."""
    name = _file_safe(label.replace(":", " ").replace(",", " "))
    return f"current_{name}" if current else name


@dataclass(frozen=True)
class ExportLook:
    """One ticked visual: its name in words, its folder and what it draws."""

    name: str
    folder: str
    layers: MontageLayers


def resolve_looks(
    keys: Sequence[str],
    *,
    current: MontageLayers | None = None,
    trail_points: Sequence[str] | None = None,
    point_count: int | None = None,
    align: bool = False,
) -> tuple[ExportLook, ...]:
    """The looks ``keys`` name (:data:`VISUAL_CHOICES`), in the dialog's order: the current
    ticks first, then the fixed looks. ``trail_points`` are every keypoint trail's points,
    the current one's too. ``align`` turns every fixed look to its heading (the current
    one is the ticks as they are, its own Align included), and says so in its name and
    folder.

    Raises:
        ValueError: an unknown or repeated key, none at all, or ``current`` named without the
            ticks it stands for.
    """
    keys = list(keys)
    if not keys:
        raise ValueError("choose at least one visual to export")
    unknown = [k for k in keys if k not in VISUAL_CHOICES]
    if unknown:
        raise ValueError(
            f"unknown visual(s) {', '.join(unknown)}: one of {', '.join(VISUAL_CHOICES)}"
        )
    if len(set(keys)) != len(keys):
        raise ValueError("each visual can be exported once")
    looks: list[ExportLook] = []
    if CURRENT in keys:
        if current is None:
            raise ValueError("the current visual needs the montage's ticks")
        layers = replace(
            current,
            trail_points=(
                tuple(trail_points)
                if current.keypoint_trail and trail_points is not None
                else None
            ),
        )
        name = describe_layers(layers, point_count)
        looks.append(ExportLook(name, visual_folder(name, current=True), layers))
    for visual in VISUALS:
        if visual.key in keys:
            label = f"{visual.label}, {ALIGNED}" if align else visual.label
            looks.append(
                ExportLook(
                    label,
                    visual_folder(label),
                    visual.layers(trail_points, align=align),
                )
            )
    return tuple(looks)


@dataclass(frozen=True)
class ExportFormat:
    """One file format and its size and rate. ``width`` ``None`` is full size (mp4 only);
    ``fps`` ``None`` is the footage's own rate for an mp4 and :data:`GIF_DEFAULT_FPS` for a GIF.
    ``codec`` is an mp4's encoding (``core.video_codecs.VIDEO_CODECS``); a GIF ignores it.

    Raises:
        ValueError: an unknown format or codec, a full-size GIF, or a width or rate out of
            range.
    """

    kind: str
    width: int | None = None
    fps: float | None = None
    codec: str = DEFAULT_VIDEO_CODEC

    def __post_init__(self) -> None:
        if self.kind not in (MP4, GIF):
            raise ValueError(f"a format is mp4 or gif, not {self.kind!r}")
        if self.codec not in VIDEO_CODECS:
            raise ValueError(
                f"unknown video codec {self.codec!r}; choose one of "
                f"{', '.join(VIDEO_CODECS)}"
            )
        if self.width is None and self.kind == GIF:
            raise ValueError("a GIF is written at a width: choose one")
        if self.width is not None and not (
            GIF_WIDTH_RANGE[0] <= int(self.width) <= EXPORT_MAX_WIDTH * 2
        ):
            raise ValueError(f"width {self.width} px is out of range")
        if self.fps is not None and not (GIF_FPS_RANGE[0] <= float(self.fps) <= 120.0):
            raise ValueError(f"{self.fps} fps is out of range")

    def resolved_fps(self, footage_fps: float) -> float:
        if self.fps is not None:
            return float(self.fps)
        return float(footage_fps) if self.kind == MP4 else GIF_DEFAULT_FPS


#: The dialog's and the CLI's formats unless told otherwise: an mp4 at full size and the
#: footage's own rate, and a GIF at 640 px and 12 fps.
DEFAULT_FORMATS: tuple[ExportFormat, ...] = (
    ExportFormat(MP4),
    ExportFormat(GIF, GIF_DEFAULT_WIDTH, GIF_DEFAULT_FPS),
)


@dataclass(frozen=True)
class PlannedFile:
    """One file an export will write: which look, which format, which clip (``None``: the
    montage file), and where, relative to the export folder."""

    look: ExportLook
    format: ExportFormat
    clip: int | None
    relpath: str


@dataclass(frozen=True)
class ExportPlan:
    """Everything one export writes (see the module docstring).

    ``clips`` are in the montage's order, with each one's tile ``labels`` and file ``stems``
    (:func:`~detail.api.clip_montage.clip_file_stem`, with Blind mode's letters when blind).

    Raises:
        ValueError: no looks, formats or clips; labels or stems that do not match the clips;
            two stems alike; two looks sharing a folder; a format twice; an unknown ``write``.
    """

    looks: tuple[ExportLook, ...]
    formats: tuple[ExportFormat, ...]
    write: str
    clips: tuple[MontageClip, ...]
    labels: tuple[str, ...]
    stems: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.looks:
            raise ValueError("choose at least one visual to export")
        if not self.formats:
            raise ValueError("choose at least one format: mp4 or gif")
        if not self.clips:
            raise ValueError("there are no clips to export")
        if self.write not in WRITES:
            raise ValueError(f"write is one of {', '.join(WRITES)}, not {self.write!r}")
        if len(self.labels) != len(self.clips) or len(self.stems) != len(self.clips):
            raise ValueError("one label and one file name per clip are needed")
        if len(set(self.stems)) != len(self.stems):
            raise ValueError("two clips would be written to the same file name")
        if len({look.folder for look in self.looks}) != len(self.looks):
            raise ValueError("two visuals would share a folder")
        if len({f.kind for f in self.formats}) != len(self.formats):
            raise ValueError("each format is written once")

    @property
    def writes_montage(self) -> bool:
        return self.write in (WRITE_MONTAGE, WRITE_BOTH)

    @property
    def writes_clips(self) -> bool:
        return self.write in (WRITE_CLIPS, WRITE_BOTH)

    def files_per_look_and_format(self) -> int:
        return int(self.writes_montage) + (len(self.clips) if self.writes_clips else 0)

    def files(self) -> list[PlannedFile]:
        """Every file, in the order written: per look, per format, the montage then each clip."""
        out: list[PlannedFile] = []
        for look in self.looks:
            for fmt in self.formats:
                if self.writes_montage:
                    out.append(
                        PlannedFile(
                            look, fmt, None, f"{look.folder}/{MONTAGE_STEM}.{fmt.kind}"
                        )
                    )
                if self.writes_clips:
                    out.extend(
                        PlannedFile(look, fmt, i, f"{look.folder}/{stem}.{fmt.kind}")
                        for i, stem in enumerate(self.stems)
                    )
        return out


def plan_for(
    clips: Sequence[MontageClip],
    looks: Sequence[ExportLook],
    formats: Sequence[ExportFormat],
    write: str = DEFAULT_WRITE,
    *,
    blind: bool,
    aliases: dict[str, str] | None = None,
) -> ExportPlan:
    """An :class:`ExportPlan` of ``clips`` with their labels and file names, blind or not.

    ``aliases`` letters the recordings over a larger set than ``clips`` (the window letters
    every member, so recording A is A in every export); by default they are lettered over
    ``clips``.
    """
    clips = list(clips)
    if blind and aliases is None:
        aliases = blind_aliases([c.recording_id for c in clips])
    found = aliases if blind else {}
    return ExportPlan(
        looks=tuple(looks),
        formats=tuple(formats),
        write=write,
        clips=tuple(clips),
        labels=tuple(montage_labels(clips, blind=blind, aliases=aliases)),
        stems=tuple(
            clip_file_stem(c, (found or {}).get(c.recording_id)) for c in clips
        ),
    )


# -- sizes ---------------------------------------------------------------------------- #
def recording_resolutions(
    experiment_dir: Path | str, recording_ids: Sequence[str]
) -> dict[str, tuple[int, int]]:
    """Each recording's video ``(width, height)``, read once per recording; one whose size
    cannot be read is left out."""
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    out: dict[str, tuple[int, int]] = {}
    for recording_id in dict.fromkeys(recording_ids):
        try:
            data = recording_store.read_recording(
                experiment_store.recording_path(experiment_dir, recording_id)
            )
        except (FileNotFoundError, KeyError, ValueError, OSError):
            continue
        res = (data.get("video") or {}).get("resolution")
        if res:
            out[recording_id] = (int(res[0]), int(res[1]))
    return out


def native_tile_px(
    montage: ClusterMontage, clips: Sequence[MontageClip] | None = None
) -> dict[MontageClip, int]:
    """Each clip's tile side at full size: its crop's side in the recording's own pixels.

    Reads each clip's boxes (``montage.prepare``) and each recording's size once.
    """
    subset = montage if clips is None else montage.subset(list(clips))
    subset.prepare()
    sizes = recording_resolutions(
        subset.experiment_dir, [c.recording_id for c in subset.clips]
    )
    out: dict[MontageClip, int] = {}
    for index, clip in enumerate(subset.clips):
        wh = sizes.get(clip.recording_id)
        if wh is None:
            out[clip] = FALLBACK_TILE_PX
            continue
        x0, y0, x1, y1 = crop_box(subset.clip_boxes(index), wh)
        out[clip] = max(int(max(x1 - x0, y1 - y0)), 16)
    return out


def montage_columns(count: int) -> int:
    """The montage file's columns: as near square as ``count`` allows (``ClusterMontage.export``)."""
    return math.ceil(math.sqrt(max(int(count), 1)))


def file_size(
    planned: PlannedFile, plan: ExportPlan, native: dict[MontageClip, int]
) -> tuple[int, int, int]:
    """``(width, height, tile_px)`` of ``planned``'s pictures, as ``ClusterMontage.export``
    lays them out for the width it is handed (:func:`export_width`)."""
    width = export_width(planned, plan, native)
    count = 1 if planned.clip is not None else len(plan.clips)
    columns = 1 if planned.clip is not None else montage_columns(count)
    tile = max((width - GAP_PX * (columns + 1)) // columns, 16)
    rows = math.ceil(count / columns)
    height = rows * (tile + label_px(tile)) + GAP_PX * (rows + 1)
    return width, height, tile


def export_width(
    planned: PlannedFile, plan: ExportPlan, native: dict[MontageClip, int]
) -> int:
    """The width ``planned`` is written at: the format's, or at full size the width that
    gives each tile its crop's own size (the montage's largest, within
    :data:`EXPORT_MAX_WIDTH`)."""
    if planned.format.width is not None:
        return int(planned.format.width)
    if planned.clip is not None:
        clip = plan.clips[planned.clip]
        return native.get(clip, FALLBACK_TILE_PX) + 2 * GAP_PX
    columns = montage_columns(len(plan.clips))
    largest = max((native.get(c, FALLBACK_TILE_PX) for c in plan.clips), default=16)
    room = (EXPORT_MAX_WIDTH - GAP_PX * (columns + 1)) // columns
    tile = max(min(largest, room), 16)
    return columns * tile + GAP_PX * (columns + 1)


# -- the estimate --------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExportCosts:
    """What a file costs to draw, encode and store, per tile and per pixel.

    Fitted 2026-10-23 to the dialog's default export on real data, after the export speed-ups
    (#807, #817): ``Toddes_SRNASEQ_C2_Hotplate`` (1080p, 21 fps), an_10 node 1385's 25 most
    typical clips, Current (Mask + mask trail) and Raw + mask outline, a full-size mp4 and a
    640-px 12-fps GIF, the montage and every clip — 104 files, timed step by step in WSL on a
    box running other work. It took 309 s and wrote 79 MB. Per step, a per-clip GIF batch
    (written side by side, one palette each) took 16 s on white and 25 s of footage; a per-clip
    mp4 of footage about 0.19 s a picture; the full-size montage mp4s 41 s and 55 s. Rough by
    nature: every number a window shows from these says "about". Timing the preview's own
    pictures to correct them was tried and dropped: drawn beside the montage's own drawing they
    ran about three times slower than an export does.
    """

    #: Seconds to draw one tile that decodes its footage (Raw on), before its size.
    tile_decoded_s: float = 0.12
    #: Seconds to draw one tile on white (Raw off): masks and keypoints, no footage.
    tile_white_s: float = 0.015
    #: Seconds each trail adds to a tile.
    trail_s: float = 0.012
    #: Seconds a tile's size adds, per megapixel of tile.
    tile_s_per_mpix: float = 0.040
    #: How much faster a montage file draws its tiles than one at a time (its threads).
    parallel: float = 2.4
    #: How much faster a batch of per-clip GIFs is written than one at a time (side by side).
    clip_gif_parallel: float = 2.5
    #: Seconds to lay out and encode a megapixel of output picture, by format; a GIF of
    #: footage costs far more than one of masks on white.
    mp4_s_per_mpix: float = 0.022
    gif_s_per_mpix_raw: float = 0.45
    gif_s_per_mpix_white: float = 0.05
    #: Bytes a megapixel of output picture takes in the file, by format; footage (Raw on)
    #: compresses worse than masks and keypoints on white.
    mp4_bytes_per_mpix_raw: float = 26_000.0
    mp4_bytes_per_mpix_white: float = 10_000.0
    gif_bytes_per_mpix_raw: float = 235_000.0
    gif_bytes_per_mpix_white: float = 13_000.0
    #: A montage mp4's grid of small tiles compresses about this much better per pixel.
    mp4_montage_bytes: float = 0.55
    #: The mp4 figures above were fitted on mp4v files. An H.264 file of the same pictures is
    #: about this share of the bytes and takes about this multiple of the time, measured in WSL
    #: on 20 s of real footage at 800x600 and 1280x720 (bytes 0.28-0.45 of mp4v's, preset
    #: veryfast 1.3x mp4v's encode time). Rough, as every number here is.
    h264_bytes: float = 0.35
    h264_encode: float = 1.3

    def tile_s(self, layers: MontageLayers, tile_px: int) -> float:
        """Seconds to draw one tile of ``layers`` at ``tile_px``."""
        base = self.tile_decoded_s if layers.raw else self.tile_white_s
        trails = int(bool(layers.mask_trail)) + int(bool(layers.keypoint_trail))
        area = (int(tile_px) ** 2) / 1e6
        return base + trails * self.trail_s + area * self.tile_s_per_mpix

    def bytes_per_mpix(self, kind: str, raw: bool, codec: str = "mp4v") -> float:
        """``codec`` is the one the file will be written in (``storage.video.written_codec``)."""
        if kind == MP4:
            each = self.mp4_bytes_per_mpix_raw if raw else self.mp4_bytes_per_mpix_white
            return each * (self.h264_bytes if codec == "h264" else 1.0)
        return self.gif_bytes_per_mpix_raw if raw else self.gif_bytes_per_mpix_white

    def encode_s_per_mpix(self, kind: str, raw: bool, codec: str = "mp4v") -> float:
        if kind == MP4:
            return self.mp4_s_per_mpix * (self.h264_encode if codec == "h264" else 1.0)
        return self.gif_s_per_mpix_raw if raw else self.gif_s_per_mpix_white


DEFAULT_COSTS = ExportCosts()


@dataclass(frozen=True)
class FileCost:
    """One planned file's frames, picture size, estimated seconds and bytes."""

    frames: int
    width: int
    height: int
    seconds: float
    bytes: float


@dataclass(frozen=True)
class Estimate:
    """What a whole export is expected to write: files, bytes and seconds (approximate)."""

    files: int
    bytes: float
    seconds: float


def file_costs(
    montage: ClusterMontage,
    plan: ExportPlan,
    native: dict[MontageClip, int],
    costs: ExportCosts = DEFAULT_COSTS,
) -> list[FileCost]:
    """The cost of each of ``plan.files()``, in order. Reads each clip's length
    (``montage.prepare``) the first time it is asked."""
    from detail.storage import video as video_store

    subset = montage.subset(list(plan.clips))
    subset.prepare()
    durations = [subset.subset([c]).duration_ns for c in plan.clips]
    footage_fps = subset.footage_fps
    whole = max(durations)
    out: list[FileCost] = []
    for planned in plan.files():
        fps = planned.format.resolved_fps(footage_fps)
        width, height, tile = file_size(planned, plan, native)
        if planned.clip is None:
            frames = len(sample_offsets_ns(whole, fps))
            # A clip that has ended holds its last tile, which the cache draws once.
            tiles = sum(min(len(sample_offsets_ns(d, fps)), frames) for d in durations)
            draw = tiles * costs.tile_s(planned.look.layers, tile) / costs.parallel
        else:
            frames = len(sample_offsets_ns(durations[planned.clip], fps))
            draw = frames * costs.tile_s(planned.look.layers, tile)
        kind, raw = planned.format.kind, planned.look.layers.raw
        mpix = frames * width * height / 1e6
        codec = video_store.written_codec(planned.format.codec)
        seconds = draw + mpix * costs.encode_s_per_mpix(kind, raw, codec)
        size = mpix * costs.bytes_per_mpix(kind, raw, codec)
        if planned.clip is not None and kind == GIF:
            seconds /= costs.clip_gif_parallel  # written side by side (export_clips)
        if planned.clip is None and kind == MP4:
            size *= costs.mp4_montage_bytes
        out.append(
            FileCost(
                frames=frames, width=width, height=height, seconds=seconds, bytes=size
            )
        )
    return out


def estimate_export(
    montage: ClusterMontage,
    plan: ExportPlan,
    native: dict[MontageClip, int],
    costs: ExportCosts = DEFAULT_COSTS,
) -> Estimate:
    """The whole export's files, bytes and seconds (see :func:`file_costs`)."""
    each = file_costs(montage, plan, native, costs)
    return Estimate(
        files=len(each),
        bytes=sum(c.bytes for c in each),
        seconds=sum(c.seconds for c in each),
    )


def about_size(n_bytes: float) -> str:
    """``about 180 MB`` / ``about 1.4 GB`` / ``under 1 MB``."""
    mb = n_bytes / 1e6
    if mb < 1:
        return "under 1 MB"
    if mb >= 1000:
        return f"about {mb / 1000:.1f} GB"
    return f"about {mb:.0f} MB" if mb >= 10 else f"about {mb:.1f} MB"


def about_time(seconds: float) -> str:
    """``under a minute`` / ``about 2 min`` / ``about 1 h 10 min``."""
    minutes = seconds / 60.0
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"about {max(int(round(minutes)), 1)} min"
    hours, rest = divmod(int(round(minutes)), 60)
    return f"about {hours} h {rest} min" if rest else f"about {hours} h"


def summary_text(plan: ExportPlan, estimate: Estimate | None = None) -> str:
    """``Will write 2 visuals × 2 formats × (1 montage + 25 clips) = 104 files into a new
    subfolder per visual · about 180 MB · about 2 min`` — the count from the plan, the rest
    from ``estimate`` when there is one."""
    looks, formats = len(plan.looks), len(plan.formats)
    parts = []
    if plan.writes_montage:
        parts.append("1 montage")
    if plan.writes_clips:
        n = len(plan.clips)
        parts.append(f"{n} clip" if n == 1 else f"{n} clips")
    each = " + ".join(parts)
    if len(parts) > 1:
        each = f"({each})"
    total = len(plan.looks) * len(plan.formats) * plan.files_per_look_and_format()
    text = (
        f"Will write {looks} visual{'s' if looks != 1 else ''} × {formats} "
        f"format{'s' if formats != 1 else ''} × {each} = {total} "
        f"file{'s' if total != 1 else ''} into a new subfolder per visual"
    )
    if estimate is not None:
        text += f"  ·  {about_size(estimate.bytes)}  ·  {about_time(estimate.seconds)}"
    return text


# -- writing -------------------------------------------------------------------------- #
def _what(planned: PlannedFile, plan: ExportPlan) -> str:
    if planned.clip is None:
        return "montage"
    return f"clip {planned.clip + 1} of {len(plan.clips)}"


def _steps(files: list[PlannedFile]) -> list[list[int]]:
    """``files``' indices in the order they are written: one file a step, except a look's
    per-clip GIFs, which are one step — ``ClusterMontage.export_clip_gifs`` writes them side
    by side on the montage's threads."""
    steps: list[list[int]] = []
    for index, planned in enumerate(files):
        batch = planned.clip is not None and planned.format.kind == GIF
        if batch and steps:
            before = files[steps[-1][0]]
            if (
                before.clip is not None
                and before.format == planned.format
                and before.look == planned.look
            ):
                steps[-1].append(index)
                continue
        steps.append([index])
    return steps


def export_clips(
    montage: ClusterMontage,
    plan: ExportPlan,
    out_dir: Path | str,
    *,
    native: dict[MontageClip, int] | None = None,
    costs: ExportCosts = DEFAULT_COSTS,
    progress: Progress = None,
    on_written: Callable[[Path], None] | None = None,
) -> list[Path]:
    """Write every file of ``plan`` under ``out_dir`` (see the module docstring), on
    ``montage``'s contexts, threads and cache; returns the paths written, in plan order.

    Every file goes through the montage's own writers: a montage file or a per-clip mp4 is
    ``ClusterMontage.export``, and a look's per-clip GIFs are one
    ``ClusterMontage.export_clip_gifs`` batch, written side by side. ``native`` is
    :func:`native_tile_px` when a caller already has it (read here when a format is full size
    and it is not given). The bar is weighed by each file's estimated cost
    (:func:`file_costs`), not counted by files, so a montage file moves it as far as it costs.
    Phases: ``preparing clips`` then ``exporting``.
    """
    out = Path(out_dir)
    subset = montage.subset(list(plan.clips))
    subset.prepare(progress)
    if native is None:
        native = native_tile_px(subset)
    footage_fps = subset.footage_fps
    files = plan.files()
    weights = [max(c.seconds, 1e-6) for c in file_costs(subset, plan, native, costs)]
    total = sum(weights)
    count = len(files)
    before = 0.0
    if progress is not None:
        progress("exporting", 0.0, f"file 1 of {count}")
    for step in _steps(files):
        first = files[step[0]]
        weight = sum(weights[i] for i in step)
        fps = first.format.resolved_fps(footage_fps)
        (out / first.relpath).parent.mkdir(parents=True, exist_ok=True)
        if len(step) == 1:
            what = f"{_what(first, plan)} (file {step[0] + 1} of {count})"
        else:
            what = f"clips (files {step[0] + 1} to {step[-1] + 1} of {count})"
        detail = f"{first.look.name} · {first.format.kind.upper()} · {what}"
        peak = [0.0]

        def report(_stage, fraction, inner, weight=weight, at=before, detail=detail):
            peak[0] = max(peak[0], float(fraction))
            if progress is not None:
                said = f"{detail}, {inner}" if inner.startswith("GIF ") else detail
                progress("exporting", min((at + peak[0] * weight) / total, 1.0), said)

        if len(step) > 1 or first.clip is not None and first.format.kind == GIF:
            indices = [files[i].clip for i in step]
            subset.subset([plan.clips[i] for i in indices]).export_clip_gifs(
                (out / first.relpath).parent,
                first.look.layers,
                [plan.labels[i] for i in indices],
                [plan.stems[i] for i in indices],
                max_width=export_width(first, plan, native),
                fps=fps,
                progress=report,
                on_written=on_written,
            )
        else:
            if first.clip is None:
                target, labels = subset, list(plan.labels)
            else:
                target = subset.subset([plan.clips[first.clip]])
                labels = [plan.labels[first.clip]]
            target.export(
                out / first.relpath,
                first.look.layers,
                labels,
                max_width=export_width(first, plan, native),
                fps=fps,
                progress=report,
                codec=first.format.codec,
            )
            if on_written is not None:
                on_written(out / first.relpath)
        before += weight
    if progress is not None:
        progress("exporting", 1.0, f"{count} files")
    return [out / f.relpath for f in files]


# -- for a script ------------------------------------------------------------------- #
def keypoints_available(
    experiment_dir: Path | str, clips: Sequence[MontageClip]
) -> bool:
    """Whether any of ``clips``' embedding runs read a keypoint run — what the looks that
    draw keypoints need. One ``embedding.yaml`` read per (recording, embedding run)."""
    from detail.api.run_preview import _embedding_meta
    from detail.storage import experiment as experiment_store

    for recording_id, embed_run_id in dict.fromkeys(
        (c.recording_id, c.embed_run_id) for c in clips
    ):
        try:
            meta = _embedding_meta(
                experiment_store.recording_path(experiment_dir, recording_id),
                embed_run_id,
            )
        except (FileNotFoundError, KeyError, ValueError, OSError):
            continue
        if (meta or {}).get("keypoint_run_id"):
            return True
    return False


def export_cluster_clips(
    experiment,
    run,
    node: int,
    out_dir: Path | str,
    visuals: Sequence[str] = (CURRENT,),
    *,
    current: MontageLayers = DEFAULT_LAYERS,
    trail_points: Sequence[str] | None = None,
    formats: Sequence[ExportFormat] = DEFAULT_FORMATS,
    write: str = DEFAULT_WRITE,
    blind: bool = False,
    limit: int = DEFAULT_CLIPS,
    sort: str = DEFAULT_SORT,
    align: bool = False,
    progress: Progress = None,
) -> list[Path]:
    """Write ``node``'s montage clips in each of ``visuals`` and ``formats`` under
    ``out_dir`` — the scripted twin of the montage window's **Export clips…**, with the
    window's defaults: the current ticks (``current``, the window's opening ones), an mp4 at
    full size and the footage's rate plus a 640-px 12-fps GIF, the montage and every clip.
    ``align`` turns the fixed looks to their heading, as the dialog's tick does.

    Raises:
        ValueError: the node has no clips; an unknown visual, format or ``write``; or a bad
            ``limit`` / ``sort``.
    """
    from detail.api.clip_montage import _cluster_clips_or_refuse

    looks = resolve_looks(
        visuals, current=current, trail_points=trail_points, align=align
    )
    clips = _cluster_clips_or_refuse(experiment, run, node, limit, sort, progress)
    plan = plan_for(clips, looks, formats, write, blind=blind)
    montage = ClusterMontage(experiment.dir, clips)
    try:
        return export_clips(montage, plan, out_dir, progress=progress)
    finally:
        montage.close()
