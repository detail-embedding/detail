"""A cluster's clips as montage tiles, drawn from this machine's video previews (§22).

The clustering page's montage plays every clip of the selected cluster at once. It reads the
preview cache only (§3 *Video preview cache*): a 480×270 proxy per part and the subject's masks at
proxy size, so a tile frame costs about a millisecond rather than a full-resolution decode. The
exports and the full-size clip window keep reading the source video.

- :func:`tracking_runs` — the tracking run whose masks a clip's subject is drawn from, per
  recording (the clustering run's embedding run says which).
- :func:`missing` — what a run's recordings still need built: a proxy (nothing to play yet) or
  only masks (the footage plays, outlined once they are built).
- :class:`ClipTiles` — the tile source: one clip per tile, opened on first ask, a frame at any
  offset from the clip's start, drawn as a :class:`TileLook` says. A clip outside the page's
  filters is drawn with its subject in grey (``matching=False``) rather than hidden. Without
  masks a tile is the plain footage, no mask or outline (the montage's **Masks** toggle), and
  ``matching`` changes nothing.
- :class:`TileLook` — what a tile draws: the footage, the masks and their trail, the keypoints,
  and the orientation. **Aligned** turns each tile frame so the body points up, from the
  keypoint run the embedding read and its heading definition (``api.clip_views`` — the one turn
  the clip window and the exports use too); the keypoints and headings are read once per clip,
  when it is opened, and kept with it. A frame with no axis is drawn as recorded and marked.

**One thread.** A :class:`ClipTiles` and its :class:`TileCatalog` are not thread-safe; the
montage calls them from its one picture worker and nowhere else.

**One budget for every montage open at once.** Each source kept :data:`CACHE_BYTES` of drawn
tiles and :data:`OPEN_CLIPS` open clips of its own, so INTERPRET with five cluster windows open
could hold six times both (review, 2026-11-09). Now every source that is drawing shares
:data:`CACHE_POOL_BYTES` and :data:`OPEN_POOL` evenly — one montage alone keeps the full
:data:`CACHE_BYTES`, two keep it each, more split the pool — down to a floor per source
(:data:`CACHE_FLOOR_BYTES`, :data:`OPEN_FLOOR`) under which a montage would re-decode the tiles
in its own view on every loop. A source counts from its first picture until it is closed, and
comes down to its share at the next picture it is asked for.

**A short clip is drawn whole, in one forward pass** (:data:`WHOLE_CLIP_FRAMES`). The montage's
one worker goes round the tiles while each tile's clock keeps running, so the frame a tile asks
for next is rarely the one after its last — and in a clip of half a second it is usually *behind*
the decoder, which costs a seek back to a keyframe (10 ms on a quiet laptop, 25–33 ms on a busy
one) where the next frame costs one. Asked frame by frame, 72 such tiles took a seek for nearly
every picture until each clip's frames had all happened to be drawn: 3 s on a quiet machine,
about 20 s beside 30 busy processes, a picture per tile every two or three seconds meanwhile.
So the first picture of a clip is still the one frame asked for (a montage fills before it
plays), and the next frame it lacks draws every frame it lacks, in order: one seek at most and a
decode per frame, none thrown away. A clip longer than the limit, or one whose frames would not
fit this source's share of the budget, is drawn frame by frame as before.
"""

from __future__ import annotations

from collections import OrderedDict
import threading
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detail.api import previews
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store

#: The subject's colour on a clip outside the page's filters: grey, still playing.
GREY = (150, 150, 150)
#: Open clips kept per montage. Each holds a decode cursor on the proxy and a few masks; a
#: cluster of two thousand clips scrolled end to end keeps the most recent of them.
OPEN_CLIPS = 160
#: Drawn tiles kept per montage, in bytes. A tile loops, and every pass after the first would
#: otherwise seek its proxy back to the clip's start and decode it again, and with short
#: clips that seek per loop held a page of playing tiles to a few frames a second. A
#: 120 px tile is 43 KB, so this holds about 4,000 of them — every frame of ~50 visible clips of
#: several seconds each.
CACHE_BYTES = 192 * 1024 * 1024
#: Drawn tiles kept by every montage open at once, together: two montages at full size.
CACHE_POOL_BYTES = 2 * CACHE_BYTES
#: The least one montage keeps however many are open: every frame of ~50 one-second clips.
CACHE_FLOOR_BYTES = 48 * 1024 * 1024
#: Open clips kept by every montage at once, together, and the least one keeps: about the
#: tiles a large window shows, so a clip in view is never closed to open another in view.
OPEN_POOL = 2 * OPEN_CLIPS
OPEN_FLOOR = 48

#: The longest clip, in tile frames, drawn whole on its second ask (the module docstring says
#: why). The pass holds the worker for about a decode per frame, so the limit keeps it near one
#: tick of the montage's clock: a longer clip's next frame is usually ahead of the decoder
#: anyway, which is the cheap direction.
WHOLE_CLIP_FRAMES = 48

_POOL_LOCK = threading.Lock()
#: The sources drawing now: a first picture, and neither closed nor collected.
_DRAWING: weakref.WeakSet = weakref.WeakSet()


def share() -> tuple[int, int]:
    """``(bytes, open clips)`` each drawing source may keep now."""
    with _POOL_LOCK:
        n = max(len(_DRAWING), 1)
    return (
        min(CACHE_BYTES, max(CACHE_FLOOR_BYTES, CACHE_POOL_BYTES // n)),
        min(OPEN_CLIPS, max(OPEN_FLOOR, OPEN_POOL // n)),
    )


@dataclass(frozen=True)
class TileLook:
    """What a montage tile draws — the pop-out window's **Show** ticks and its orientation.

    ``footage``: the proxy's picture (off: on white). ``masks``: the subject's mask, outlined on
    footage and filled on white; ``mask_trail``: its last half second, fading. ``keypoints``: the
    keypoint run the embedding read, drawn over the rest. ``aligned``: each frame turned so the
    body points up (``clip_views.turn_matrix``).

    Raises:
        ValueError: a tile that would draw nothing (none of footage, masks, keypoints).
    """

    footage: bool = True
    masks: bool = True
    mask_trail: bool = True
    keypoints: bool = False
    aligned: bool = False

    def __post_init__(self) -> None:
        if not (self.footage or self.masks or self.keypoints):
            raise ValueError(
                "a tile draws something: turn on at least one of footage, masks, keypoints"
            )

    @property
    def needs_pose(self) -> bool:
        """Whether a tile reads the clip's keypoints: to draw them, or to turn it."""
        return self.keypoints or self.aligned


@dataclass(frozen=True)
class TileClip:
    """What a tile plays: a subject of a recording from ``start_ns`` to ``end_ns`` (inclusive)."""

    recording_id: str
    subject: int
    start_ns: int
    end_ns: int


def tracking_runs(
    experiment_dir: Path | str, embed_runs: Mapping[str, str]
) -> dict[str, str]:
    """``{recording_id: tracking_run_id}`` from each recording's embedding run.

    A recording whose embedding run is gone (deleted since the fit) is left out; its clips draw
    the footage with no outline.
    """
    from detail.api.run_preview import _embedding_meta

    out: dict[str, str] = {}
    for recording_id, embed_run_id in embed_runs.items():
        recording_dir = experiment_store.recording_path(experiment_dir, recording_id)
        try:
            meta = _embedding_meta(recording_dir, embed_run_id)
        except (ValueError, FileNotFoundError):
            continue
        tracking = meta.get("tracking_run_id")
        if tracking:
            out[recording_id] = str(tracking)
    return out


def missing(experiment, tracking: Mapping[str, str], recording_ids) -> dict[str, list]:
    """What ``recording_ids`` still need built here: ``{"proxy": [...], "masks": [...]}``.

    A recording missing a part's proxy has no picture to play; one missing only its tracking
    run's masks plays its footage without the outline until they are built (the reader opens
    footage-only, ``PreviewReader.masks_missing``). One stat per part once the cache knows the
    file (``Recording.missing_previews``). A part whose video is not on this machine has
    nothing to build from and is not counted.
    """
    out: dict[str, list] = {"proxy": [], "masks": []}
    for recording_id in recording_ids:
        recording = experiment.get_recording(recording_id)
        tracking_run = tracking.get(recording_id)
        tasks = recording.missing_previews(
            [tracking_run] if tracking_run is not None else []
        )
        kinds = {task.kind for task in tasks}
        if previews.PROXY in kinds:
            out["proxy"].append(recording_id)
        elif previews.MASKS in kinds:
            out["masks"].append(recording_id)
    return out


class TileCatalog:
    """Each part's ``PreviewReader`` and where each recording's parts start, opened once.

    Shared by every :class:`ClipTiles` a page makes, so moving to another cluster reuses what
    the last one opened (a reader costs ~60 ms to open: its frame map and its masks).
    """

    def __init__(
        self,
        experiment,
        tracking: Mapping[str, str],
        embeds: Mapping[str, str] | None = None,
    ) -> None:
        """``embeds`` (``{recording_id: embed_run_id}``) is where a tile's keypoints and body
        axis come from (the keypoint run each embedding read); without it a tile draws neither.
        """
        self._experiment = experiment
        self._tracking = dict(tracking)
        self._embeds = dict(embeds or {})
        self._readers: dict[tuple[str, int], previews.PreviewReader | None] = {}
        self._part_starts: dict[str, np.ndarray] = {}
        self._points: dict[str, object] = {}
        self._stems: dict[str, list[str]] = {}

    def part_of(self, recording_id: str, t_ns: int) -> int:
        """The part of ``recording_id`` that holds ``t_ns``: the last one starting at or before."""
        starts = self._part_starts.get(recording_id)
        if starts is None:
            recording = self._experiment.get_recording(recording_id)
            data = recording_store.read_recording(recording.dir)
            parts = recording_store.video_files(data)
            firsts = []
            for index in range(len(parts)):
                values = recording_store.load_active_timestamps_cached(
                    recording.dir, data, index
                ).values
                firsts.append(int(values[0]) if len(values) else 0)
            starts = np.asarray(firsts or [0], dtype=np.int64)
            self._part_starts[recording_id] = starts
        index = int(np.searchsorted(starts, int(t_ns), side="right")) - 1
        return max(index, 0)

    def reader(self, recording_id: str, part: int) -> previews.PreviewReader:
        """The part's reader, with the subject masks of the recording's tracking run.

        Raises:
            PreviewMissingError: the proxy (or its masks) is not built on this machine.
        """
        key = (recording_id, int(part))
        if key not in self._readers:
            recording = self._experiment.get_recording(recording_id)
            self._readers[key] = previews.PreviewReader.open(
                recording, int(part), self._tracking.get(recording_id)
            )
        return self._readers[key]

    def points(self, recording_id: str):
        """The keypoints ``recording_id``'s embedding run read (``clip_views.ClipPoints``),
        opened once; ``None`` without an embedding run for it."""
        if recording_id not in self._points:
            embed_run_id = self._embeds.get(recording_id)
            found = None
            if embed_run_id is not None:
                from detail.api.clip_views import ClipPoints
                from detail.api.run_preview import _embedding_meta

                recording = self._experiment.get_recording(recording_id)
                try:
                    meta = _embedding_meta(recording.dir, embed_run_id)
                except (ValueError, FileNotFoundError):
                    meta = None
                if meta is not None:
                    found = ClipPoints.of(
                        Path(self._experiment.dir), recording.dir, meta, embed_run_id
                    )
            self._points[recording_id] = found
        return self._points[recording_id]

    def part_stem(self, recording_id: str, part: int) -> str:
        """The stem keypoint runs name ``recording_id``'s part ``part`` by."""
        stems = self._stems.get(recording_id)
        if stems is None:
            recording = self._experiment.get_recording(recording_id)
            data = recording_store.read_recording(recording.dir)
            stems = [
                recording_store.part_stem(f) for f in recording_store.video_files(data)
            ]
            self._stems[recording_id] = stems
        return stems[int(part)]

    def forget(self, recording_id: str) -> None:
        """Drop what was opened for ``recording_id`` — its previews were just (re)built."""
        for key in [k for k in self._readers if k[0] == recording_id]:
            del self._readers[key]


class ClipTiles:
    """The montage's tile source over ``clips``, in display order (``live_montage.TileSource``).

    A tile's clip is opened the first time it is asked for, at the size and colour asked, and
    reopened only when either changes. A clip that crosses into the next part plays to the end
    of the part it starts in.
    """

    def __init__(self, catalog: TileCatalog, clips: Sequence[TileClip]) -> None:
        self._catalog = catalog
        self._clips = list(clips)
        self._open: OrderedDict[int, tuple] = OrderedDict()
        #: ``{clip index: (tile_px, matching, masks, t_ns, {frame offset: tile})}``, oldest
        #: first. Keyed on everything that changes the picture, so a toggle never shows a stale
        #: tile.
        self._drawn: OrderedDict[int, tuple] = OrderedDict()
        self._bytes = 0
        #: Counted among the sources sharing the budget (:func:`share`).
        self._drawing = False

    def __len__(self) -> int:
        return len(self._clips)

    def frame(
        self,
        clip_index: int,
        offset_ns: int,
        tile_px: int,
        matching: bool,
        *,
        look: TileLook,
    ) -> np.ndarray:
        """Tile ``clip_index`` at ``offset_ns`` from its start: ``(tile_px, tile_px, 3)`` RGB,
        drawn as ``look`` says — the subject in grey when not ``matching`` and masks are drawn.

        Raises:
            PreviewMissingError: the clip's part has no preview on this machine.
        """
        index, tile_px = int(clip_index), int(tile_px)
        if not self._drawing:
            self._drawing = True
            with _POOL_LOCK:
                _DRAWING.add(self)
        self._trim()
        matching = bool(matching) or not look.masks  # no mask, nothing to colour
        key = (tile_px, matching, look)
        clip = self._clips[index]
        drawn = self._drawn.get(index)
        if drawn is not None and drawn[0] != key:
            self._forget_drawn(index)
            drawn = None
        if drawn is None:
            tile = self._clip(index, key)
            drawn = (key, tile.t_ns, {})
            self._drawn[index] = drawn
        self._drawn.move_to_end(index)
        t_ns, frames = drawn[1], drawn[2]
        wanted = clip.start_ns + int(offset_ns)
        offset = max(int(np.searchsorted(t_ns, wanted, side="right")) - 1, 0)
        picture = frames.get(offset)
        if picture is None:
            tile = self._clip(index, key)
            wanted_frames = (
                range(len(t_ns)) if self._draws_whole(len(t_ns), frames) else (offset,)
            )
            for missing in wanted_frames:
                if missing not in frames:
                    frames[missing] = tile.frame_at(missing)
                    self._bytes += frames[missing].nbytes
            picture = frames[offset]
            self._trim()
        return picture

    def _draws_whole(self, n_frames: int, frames: Mapping[int, np.ndarray]) -> bool:
        """Whether a clip's missing frame draws every frame it lacks: not its first picture, at
        most :data:`WHOLE_CLIP_FRAMES` long, and the whole clip fits this source's share.
        """
        if not frames or n_frames > WHOLE_CLIP_FRAMES:
            return False
        tile_bytes = next(iter(frames.values())).nbytes
        return n_frames * tile_bytes <= share()[0]

    def _trim(self) -> None:
        """Down to this source's share of the budget (:func:`share`), oldest first."""
        budget, most = share()
        while self._bytes > budget and len(self._drawn) > 1:
            self._forget_drawn(next(iter(self._drawn)))
        while len(self._open) > most:
            _index, (oldest, *_rest) = self._open.popitem(last=False)
            oldest.close()

    def _forget_drawn(self, index: int) -> None:
        drawn = self._drawn.pop(index, None)
        if drawn is not None:
            self._bytes -= sum(f.nbytes for f in drawn[2].values())

    def _clip(self, index: int, key: tuple):
        held = self._open.get(index)
        if held is not None and held[1] == key:
            self._open.move_to_end(index)
            return held[0]
        if held is not None:
            held[0].close()
        tile_px, matching, look = key
        clip = self._clips[index]
        part = self._catalog.part_of(clip.recording_id, clip.start_ns)
        reader = self._catalog.reader(clip.recording_id, part)
        tile = reader.clip(
            clip.subject,
            clip.start_ns,
            clip.end_ns,
            tile_px,
            footage=look.footage,
            mask=look.masks,
            trail=previews.TRAIL_FRAMES if look.mask_trail else 1,
            tint=None if matching else GREY,
        )
        if look.needs_pose:
            self._pose(tile, clip, part, look)
        self._open[index] = (tile, key)
        self._trim()
        return tile

    def _pose(self, tile, clip: TileClip, part: int, look: TileLook) -> None:
        """Read the clip's keypoints on its tile frames, once, and hand the tile its axis ends
        (to turn it) and its points (to draw them). A run with no keypoints or no heading
        definition leaves the tile as recorded: the switch was greyed with the reason.
        """
        points = self._catalog.points(clip.recording_id)
        if points is None or points.run is None:
            return
        source = tile.source_frames()
        stem = self._catalog.part_stem(clip.recording_id, part)
        xy, scores = points.rows_at(clip.subject, stem, source)
        axes = None
        if look.aligned:
            axis = points.axis(clip.subject)
            if axis is not None:
                axes = tile_axes(xy, scores, axis)
        tile.set_pose(
            axes=axes,
            points=(xy, scores, points.skeleton) if look.keypoints else None,
        )

    def close(self) -> None:
        """Let go of every open clip and every drawn tile."""
        if self._drawing:
            self._drawing = False
            with _POOL_LOCK:
                _DRAWING.discard(self)
        self._drawn.clear()
        self._bytes = 0
        while self._open:
            _index, (tile, *_rest) = self._open.popitem()
            tile.close()


def tile_axes(points: np.ndarray, scores: np.ndarray, axis) -> np.ndarray:
    """``(n, 2, 2)`` normalized body-axis ends (front, back) per row of ``points``
    ``(n, K, 2)``, NaN where a row has none — the per-frame turn of an aligned tile."""
    from detail.analysis import body_axis

    front, back = body_axis.row_axis_ends(
        points, scores, list(axis.front), list(axis.back), (1, 1), axis.cutoff
    )
    return np.stack([front, back], axis=1)
