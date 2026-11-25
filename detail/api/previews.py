"""Video previews: build them, ask what is missing, and draw tiles from them (§3 *Video
preview cache*).

A preview is a **display-only** copy of a video part, small enough that many tiles can play
at once: a proxy at 480x270 at most and 15 frames a second, the tracked animals' masks shrunk
to match, and a map from each proxy frame to the source frame it shows. The layout and keys
are ``storage/previews.py``'s; this module is what builds them and what reads them back.

**Nothing in the pipeline reads a preview** (``tests/storage/test_preview_isolation.py``). A
stage analyses the source video; a preview is for looking at, and every export reads the
source too.

Two builds per part:

- **proxy** — one sequential decode of the source (no seek, Tenet 11; no FFmpeg executable,
  Tenet 15), keeping a frame each time its ``t_ns`` reaches the next 1/15 s slot, shrunk in the
  decoder and written by ``storage.video.open_video_writer`` (H.264, or mp4v where cv2 has no
  H.264 encoder). The kept source frame indices are stored beside it (``src_frame``).
- **masks**, per complete tracking run — for every proxy frame, each tracked animal's mask from
  the detection rows (fragments of one animal OR-merged), shrunk to the proxy and re-encoded as
  COCO-RLE, with its box. No video is decoded.

:class:`PreviewReader` is the read side: given a clip (animal, start and end ``t_ns``) it gives
tile pictures at a requested size — a crop fixed for the clip from the stored boxes (the clip
montage's :func:`~detail.api.clip_montage.crop_box`), shrunk, with the mask outline and its
trail drawn **at tile size**, never on a full frame.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detail.core import visualize
from detail.core.progress import OperationCancelled, ThrottledProgress
from detail.storage import previews as store
from detail.storage import recording as recording_store
from detail.storage import video_roots

#: The proxy's frame rate: the montage plays at most 15 fps (``clip_montage.MONTAGE_MAX_FPS``).
PROXY_FPS = 15
#: The box a proxy is fitted inside, aspect kept and never enlarged.
PROXY_MAX_WH = (480, 270)
#: Proxy frames in a mask trail: half a second (``clip_views.CLIP_TRAIL_S``) at 15 fps.
TRAIL_FRAMES = 8
#: How much a part's masks cost against its proxy, per source frame — measured on one real
#: 1080p recording.
MASK_WORK_SHARE = 0.35
STAGE = "Building previews"
PROXY = "proxy"
MASKS = "masks"

_SLOT_SLACK = 100_000 * PROXY_FPS  # 0.1 ms, in slot units x 1e9

Progress = Callable[[str, float, str], None]


class PreviewMissingError(FileNotFoundError):
    """A preview that has not been built on this machine yet."""


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def proxy_size(width: int, height: int) -> tuple[int, int]:
    """The proxy's ``(width, height)`` for a source of ``width`` x ``height``: fitted inside
    :data:`PROXY_MAX_WH`, aspect kept, never enlarged, both sides even (H.264 needs it).
    """
    width, height = int(width), int(height)
    scale = min(PROXY_MAX_WH[0] / width, PROXY_MAX_WH[1] / height, 1.0)
    return (
        max(2, int(round(width * scale / 2.0)) * 2),
        max(2, int(round(height * scale / 2.0)) * 2),
    )


def proxy_frame_selection(t_ns: np.ndarray) -> np.ndarray:
    """Source frame indices the proxy keeps: each frame whose ``t_ns`` reaches the next
    1/15 s slot after the last one kept. A source slower than 15 fps keeps every frame; a
    variable-rate one keeps the frame nearest each slot from below, so no time is skipped.
    """
    t_ns = np.asarray(t_ns, dtype=np.int64)
    if not len(t_ns):
        return np.zeros(0, dtype=np.int64)
    # Integer slot numbers, with a 0.1 ms allowance: a 60 fps frame 8 sits at 133,333,333 ns,
    # a hair under the 2/15 s slot edge, and plain floor would push it to the next frame.
    slots = ((t_ns - t_ns[0]) * PROXY_FPS + _SLOT_SLACK) // 1_000_000_000
    first_in_slot = np.ones(len(slots), dtype=bool)
    first_in_slot[1:] = slots[1:] != slots[:-1]
    return np.flatnonzero(first_in_slot).astype(np.int64)


# --------------------------------------------------------------------------- #
# what needs building
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PreviewTask:
    """One build: a part's proxy, or one tracking run's masks for a part."""

    recording_id: str
    part_index: int
    kind: str  # PROXY | MASKS
    tracking_run_id: str | None = None
    #: Relative cost, in source frames (masks weighed by :data:`MASK_WORK_SHARE`).
    work: float = 1.0


def _part_video(recording_dir: Path, part: dict) -> Path | None:
    try:
        path = recording_store.resolve_part_video_path(recording_dir, part)
    except video_roots.MissingVideoRootError:
        return None
    return path if path.is_file() else None


def _complete_tracking(data: dict) -> list[tuple[dict, dict]]:
    """``(tracking entry, detection entry)`` for each complete tracking run on a complete
    detection run — what a mask store can be built from."""
    detections = {r.get("id"): r for r in data.get("detection_runs", [])}
    found = []
    for entry in data.get("tracking_runs", []):
        if entry.get("status") != "complete":
            continue
        detection = detections.get(entry.get("detection_run_id"))
        if detection is None or detection.get("status") != "complete":
            continue
        if detection.get("downsampling_run_id"):
            continue  # legacy frame space (§4); its rows do not index this video
        found.append((entry, detection))
    return found


def recording_tasks(
    recording_dir: Path | str,
    recording_id: str,
    *,
    tracking_run_ids: Sequence[str] | None = None,
) -> list[PreviewTask]:
    """The builds a recording is missing on this machine, proxies before masks.

    One ``stat`` per part once the cache index knows the file; a part never seen on this
    machine is fingerprinted (two 1 MiB reads). A part whose video is not here (a
    video-excluded export, an unregistered video root) is skipped — there is nothing to
    build from. ``tracking_run_ids`` limits the masks considered; ``None`` is every complete
    tracking run.
    """
    recording_dir = Path(recording_dir)
    data = recording_store.read_recording(recording_dir)
    files = data.get("video", {}).get("files", [])  # a metadata-only recording has none
    tracking = [
        pair
        for pair in _complete_tracking(data)
        if tracking_run_ids is None or pair[0]["id"] in tracking_run_ids
    ]
    proxies, masks = [], []
    for part_index, part in enumerate(files):
        path = _part_video(recording_dir, part)
        if path is None:
            continue
        frames = len(
            recording_store.load_active_timestamps_cached(
                recording_dir, data, part_index
            ).values
        )
        key = store.key_for_video(path)
        if not store.proxy_present(key):
            proxies.append(
                PreviewTask(recording_id, part_index, PROXY, None, float(frames))
            )
        for entry, detection in tracking:
            digest = store.masks_digest(entry, detection)
            if store.masks_complete(store.masks_path(key, entry["id"], digest)):
                continue
            masks.append(
                PreviewTask(
                    recording_id,
                    part_index,
                    MASKS,
                    entry["id"],
                    float(frames) * MASK_WORK_SHARE,
                )
            )
    return proxies + masks


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PreviewBuildReport:
    """What a batch of builds did: how many were built, and which failed and why.

    A failed build does not stop the batch — one unreadable part must not leave every other
    recording without previews — so failures are collected here for the caller to show.
    """

    built: int
    failed: tuple[tuple[PreviewTask, str], ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failed


def build_tasks(
    experiment_dir: Path | str,
    tasks: Sequence[PreviewTask],
    names: dict[str, str] | None = None,
    progress: Progress | None = None,
) -> PreviewBuildReport:
    """Build ``tasks`` in order under one bar.

    The bar is weighed by work (source frames), not counted by task, so a 30-minute part and
    a 30-second one are not each half of it. The phase line names the build in hand —
    ``proxy · <recording> · part 2 of 3`` — and advances, forced past the throttle, at every
    task boundary. A task that fails is recorded in the report and the batch moves on; a
    ``progress`` that raises ``OperationCancelled`` (the GUI's cancel) stops the whole batch.
    Either way the staged files of the task in hand are removed. Anything an interrupted
    build left behind earlier is swept first.
    """
    from detail.api.recording import Recording

    experiment_dir = Path(experiment_dir)
    names = names or {}
    total = sum(task.work for task in tasks) or 1.0
    done = 0.0
    throttled = ThrottledProgress(
        (lambda event: progress(*event)) if progress is not None else None
    )

    def waiting_at(fraction: float, phase: str):
        """What a build reports while another holds the cache's build lock — a GUI build
        queued behind a CLI one, or the reverse. Called every ``atomic.WAIT_POLL_S``, so a
        cancel (``progress`` raising) lands while it waits."""
        if progress is None:
            return None
        text = f"Waiting for another preview build to finish · {phase}"
        return lambda: progress(STAGE, min(1.0, fraction), text)

    if tasks:
        with store.build_lock(waiting=waiting_at(0.0, "starting")):
            store.sweep_stale_staging()
    parts_of: dict[str, int] = {}
    built = 0
    failed: list[tuple[PreviewTask, str]] = []
    proxy_failed: dict[tuple[str, int], str] = {}
    for number, task in enumerate(tasks, start=1):
        part = (task.recording_id, task.part_index)
        if task.kind == MASKS and part in proxy_failed:
            # Its masks are drawn onto the proxy and would decode the whole part again to
            # build it, failing the same way: one failure per part, said once.
            failed.append(
                (task, f"skipped: this part's proxy failed ({proxy_failed[part]})")
            )
            done += task.work
            continue
        try:
            recording = Recording(experiment_dir, task.recording_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            failed.append((task, f"recording unreadable: {exc}"))
            done += task.work
            continue
        if task.recording_id not in parts_of:
            parts_of[task.recording_id] = len(recording.video_parts)
        name = names.get(task.recording_id, recording.name)
        what = PROXY if task.kind == PROXY else f"{MASKS} {task.tracking_run_id}"
        phase = (
            f"{what} · {name} · part {task.part_index + 1} of "
            f"{parts_of[task.recording_id]}"
        )
        if len(tasks) > 1:
            phase = f"{phase} · {number} of {len(tasks)}"
        base = done

        def report(fraction: float, *, force: bool = False, base=base, task=task):
            throttled(
                (STAGE, min(1.0, (base + fraction * task.work) / total), phase),
                force=force,
            )

        report(0.0, force=True)
        error = _run_task(recording, task, report, waiting_at(base / total, phase))
        done += task.work
        if error is None:
            built += 1
        else:
            failed.append((task, error))
            if task.kind == PROXY:
                proxy_failed[part] = error
    if tasks:
        summary = f"{built} built" + (f", {len(failed)} failed" if failed else "")
        throttled((STAGE, 1.0, summary), force=True)
    return PreviewBuildReport(built, tuple(failed))


def _run_task(recording, task: PreviewTask, report, waiting=None) -> str | None:
    """One build; returns why it failed, or None. The isolation boundary of the batch: a
    cancel (``OperationCancelled``) is not a failure of this part and goes straight through.
    """
    try:
        if task.kind == PROXY:
            build_proxy(recording, task.part_index, report, waiting=waiting)
        else:
            build_masks(
                recording,
                task.part_index,
                task.tracking_run_id,
                report,
                waiting=waiting,
            )
    except OperationCancelled:
        raise
    except (
        Exception
    ) as exc:  # noqa: BLE001 — boundary: one part must not stop the batch
        return f"{type(exc).__name__}: {exc}"
    return None


def build_proxy(recording, part_index: int, report=None, *, waiting=None) -> str:
    """Build one part's proxy (skipped when already built); returns its key. Runs under the
    cache's build lock (``storage.previews.build_lock``); ``waiting`` is called while another
    build holds it, and may raise to stop waiting."""
    with store.build_lock(waiting=waiting):
        return _build_proxy(recording, part_index, report)


def build_masks(
    recording, part_index: int, tracking_run_id: str, report=None, *, waiting=None
) -> Path:
    """Build one tracking run's masks for one part at proxy size; returns the store path.
    Runs under the cache's build lock, and builds the part's proxy first if it is missing.

    For each proxy frame the rows read are those of the run's nearest *processed* frame
    (a run at ``every_n_frames`` > 1 has rows on every Nth source frame only), so a strided
    run still shows its animal on every proxy frame.
    """
    with store.build_lock(waiting=waiting):
        return _build_masks(recording, part_index, tracking_run_id, report)


def _build_proxy(recording, part_index: int, report=None) -> str:
    from detail.storage import video

    data = recording_store.read_recording(recording.dir)
    part = recording_store.video_files(data)[part_index]
    path = recording_store.resolve_part_video_path(recording.dir, part)
    key = store.key_for_video(path)
    if store.proxy_present(key):
        return key
    timestamps = recording_store.load_active_timestamps_cached(
        recording.dir, data, part_index
    ).values
    # The part's own size, not the recording's: parts of one recording can differ, and a
    # proxy fitted to another part's shape is stretched.
    source_wh = tuple(int(v) for v in video.probe_video(path)["resolution"])
    width, height = proxy_size(*source_wh)
    keep = proxy_frame_selection(timestamps)
    final = store.proxy_dir(key)
    tmp = store.staging_dir(final)
    try:
        writer, codec = video.open_video_writer(
            tmp / store.PROXY_FILE, PROXY_FPS, width, height
        )
        written: list[int] = []
        n_source = max(len(timestamps), 1)
        try:
            for idx, bgr in video.read_selected_frames(
                path, keep, timestamps=timestamps, size=(width, height)
            ):
                writer.write(bgr)
                written.append(idx)
                if report is not None:
                    report(idx / n_source)
        finally:
            writer.release()
        if not written:
            raise ValueError(f"{path.name}: no frame could be decoded for its preview")
        _check_proxy_complete(tmp / store.PROXY_FILE, len(written), path.name)
        store.write_frames(
            tmp,
            np.asarray(written, dtype=np.int64),
            {
                "width": width,
                "height": height,
                "source_width": int(source_wh[0]),
                "source_height": int(source_wh[1]),
                "fps": PROXY_FPS,
                "codec": codec,
                "n_source_frames": int(len(timestamps)),
            },
        )
    except BaseException:
        store.discard(tmp)
        raise
    store.swap_into_place(tmp, final, complete=store.proxy_complete)
    return key


def proxy_frames_in_file(proxy: Path) -> int:
    """How many frames the written proxy file holds, read from its packets (PyAV, no decode
    and no FFmpeg executable)."""
    from detail.storage import video

    return len(video.generate_timestamps(proxy, float(PROXY_FPS)))


def _check_proxy_complete(proxy: Path, n_written: int, source_name: str) -> None:
    """Refuse a proxy file holding fewer frames than were written to it.

    ``cv2.VideoWriter.write`` reports nothing — a full disk, or an encoder that dropped
    frames, leaves a short file and no error — and a short proxy marked complete is never
    rebuilt: every clip past its end shows the wrong frame or none. So the file is counted
    before it is swapped into place.

    Raises:
        OSError: the file is short or unreadable; the build fails and nothing is kept.
    """
    try:
        found = proxy_frames_in_file(proxy)
    except Exception as exc:  # noqa: BLE001 — any unreadable file is the same failure
        raise OSError(
            f"{source_name}: the preview written could not be read back ({exc}); the disk "
            "may be full. It was not kept and is rebuilt next time."
        ) from exc
    if found != n_written:
        raise OSError(
            f"{source_name}: the preview holds {found} of the {n_written} frames written "
            "(is the disk full?). It was not kept and is rebuilt next time."
        )


def _mask_resolution(recording, detection_run_id: str) -> tuple[int, int]:
    """``(height, width)`` the detection masks were encoded at (§9: the run's job.json)."""
    from detail.executor import rundir

    try:
        envelope = rundir.read_envelope(recording.dir / "detections" / detection_run_id)
        height, width = (int(v) for v in envelope["mask_resolution"])
        return height, width
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        resolution = recording.video_resolution
        if resolution is None:
            raise ValueError(
                f"{recording.name}: detection run {detection_run_id} records no mask "
                "resolution and the recording has no video resolution"
            ) from None
        return int(resolution[1]), int(resolution[0])


def _build_masks(recording, part_index: int, tracking_run_id: str, report=None) -> Path:
    import cv2
    from pycocotools import mask as mask_utils

    from detail.api.run_preview import _build_part, tracking_cadence, tracking_entries
    from detail.core.utils import decode_mask

    data = recording_store.read_recording(recording.dir)
    part = recording_store.video_files(data)[part_index]
    path = recording_store.resolve_part_video_path(recording.dir, part)
    key = store.key_for_video(path)
    entry, detection = tracking_entries(data, tracking_run_id)
    digest = store.masks_digest(entry, detection)
    final = store.masks_path(key, tracking_run_id, digest)
    if store.masks_complete(final):
        return final
    try:
        frames = store.read_frames(key)
    except FileNotFoundError:
        build_proxy(recording, part_index)
        frames = store.read_frames(key)
    cadence = tracking_cadence(data, tracking_run_id)
    source = _build_part(
        recording.dir,
        data,
        entry["detection_run_id"],
        part_index,
        cadence,
        recording.dir / "tracking" / tracking_run_id / "assignments",
        timestamps=recording_store.load_active_timestamps_cached(
            recording.dir, data, part_index
        ).values,
    )
    mask_hw = _mask_resolution(recording, entry["detection_run_id"])
    width, height = frames.width, frames.height
    n_proxy = len(frames.src_frame)
    tmp = store.staging_dir(final)
    try:
        writer = store.MaskWriter(
            tmp,
            {
                "built_from": digest,
                "tracking_run_id": tracking_run_id,
                "detection_run_id": entry["detection_run_id"],
                "proxy_key": key,
                "width": width,
                "height": height,
                "n_proxy_frames": n_proxy,
            },
        )
        reader, frame_idx = source["reader"], np.asarray(source["frame_idx"])
        assignment = source["assignment"]
        if reader is not None and assignment is not None and len(frame_idx):
            last = max(source["n_frames"] - 1, 0)
            target = np.clip(
                np.round(frames.src_frame / cadence).astype(np.int64) * cadence,
                0,
                last - (last % cadence),
            )
            lo = np.searchsorted(frame_idx, target, side="left")
            hi = np.searchsorted(frame_idx, target, side="right")
            block: dict = {"start": -1, "stop": -1}

            def rows(a: int, b: int) -> dict:
                if not (block["start"] <= a and b <= block["stop"]):
                    stop = min(len(frame_idx), max(b, a + store.FLUSH_ROWS))
                    block.update(
                        start=a,
                        stop=stop,
                        bbox=np.asarray(reader.column("bbox")[a:stop]),
                        rle=np.asarray(reader.column("mask_rle")[a:stop]),
                        oid=np.asarray(assignment.column("object_id")[a:stop]),
                    )
                s = a - block["start"]
                return {
                    name: block[name][s : s + (b - a)]
                    for name in ("bbox", "rle", "oid")
                }

            for proxy_frame in range(n_proxy):
                a, b = int(lo[proxy_frame]), int(hi[proxy_frame])
                if b > a:
                    got = rows(a, b)
                    merged: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                    for j in range(b - a):
                        oid = int(got["oid"][j])
                        counts = str(got["rle"][j])
                        if oid < 0 or not counts:
                            continue
                        mask = decode_mask(counts, mask_hw)
                        box = np.asarray(got["bbox"][j], dtype=np.float64)
                        if oid in merged:
                            old_mask, old_box = merged[oid]
                            mask = mask | old_mask
                            box = np.concatenate(
                                [
                                    np.minimum(box[:2], old_box[:2]),
                                    np.maximum(box[2:], old_box[2:]),
                                ]
                            )
                        merged[oid] = (mask, box)
                    for oid in sorted(merged):
                        mask, box = merged[oid]
                        small = (
                            cv2.resize(
                                mask.astype(np.uint8) * 255,
                                (width, height),
                                interpolation=cv2.INTER_AREA,
                            )
                            > 127
                        )
                        counts = mask_utils.encode(
                            np.asfortranarray(small.astype(np.uint8))
                        )["counts"].decode("ascii")
                        writer.add(proxy_frame, oid, box, counts)
                if report is not None and proxy_frame % 64 == 0:
                    report(proxy_frame / max(n_proxy, 1))
        writer.close()
    except BaseException:
        store.discard(tmp)
        raise
    store.swap_into_place(tmp, final, complete=store.masks_complete)
    return final


# --------------------------------------------------------------------------- #
# the machine-level cache
# --------------------------------------------------------------------------- #
def preview_cache_dir() -> Path:
    """Where this machine's previews live."""
    return store.cache_dir()


def preview_cache_size() -> int:
    """Bytes the preview cache holds on this machine."""
    return store.cache_size()


def clear_preview_cache() -> int:
    """Delete every preview on this machine (they rebuild when next needed); returns the
    bytes freed. Experiment data is never touched: the cache is outside every experiment.
    """
    return store.clear_cache()


# --------------------------------------------------------------------------- #
# reading: tiles
# --------------------------------------------------------------------------- #
_PROXY_PTS: dict[tuple[str, int], np.ndarray] = {}


def _proxy_pts(path: Path) -> np.ndarray:
    """The proxy file's own frame times, read from its packets (no decode; ~70 ms for a
    29-minute part) and kept per file. This is the *file's* seek index for ``FrameCursor``,
    not a time reference: a proxy frame's time is always its source frame's ``t_ns``."""
    from detail.storage import video

    stat = path.stat()
    key = (str(path), stat.st_mtime_ns)
    if key not in _PROXY_PTS:
        if len(_PROXY_PTS) > 64:
            _PROXY_PTS.clear()
        _PROXY_PTS[key] = video.generate_timestamps(path, float(PROXY_FPS))
    return _PROXY_PTS[key]


def _to_tile(image: np.ndarray, crop, tile_px: int, *, mask: bool = False):
    """``image`` cut to ``crop``, centred in a square, and scaled to ``tile_px``."""
    import cv2

    x0, y0, x1, y1 = crop
    cut = image[y0:y1, x0:x1]
    side = max(cut.shape[0], cut.shape[1], 1)
    if cut.shape[0] != cut.shape[1]:
        square = np.zeros((side, side, *cut.shape[2:]), dtype=cut.dtype)
        top, left = (side - cut.shape[0]) // 2, (side - cut.shape[1]) // 2
        square[top : top + cut.shape[0], left : left + cut.shape[1]] = cut
        cut = square
    # Area averaging only earns its cost on a large shrink; a clip crop is typically ~200 px
    # of proxy going to a 160 px tile, where it costs several times linear's and looks
    # the same.
    interpolation = cv2.INTER_AREA if side > 2 * tile_px else cv2.INTER_LINEAR
    if mask:
        scaled = cv2.resize(
            cut.astype(np.uint8) * 255, (tile_px, tile_px), interpolation=interpolation
        )
        return scaled > 127
    return cv2.resize(cut, (tile_px, tile_px), interpolation=interpolation)


#: Decode threads per open clip. A montage opens one clip per tile and a person can have several
#: open, so FFmpeg's machine-sized pools (a decode and a conversion thread per core) came
#: to thousands of threads with five cluster windows. Not 1: a tile's first picture seeks a
#: proxy whose keyframes are 250 frames apart, and one thread made that seek more than
#: twice as slow, which a fast scroll on a large cluster shows. Four were measurably slower
#: than FFmpeg's default pool; eight matched the default's speed at about half its threads
#: (no conversion pool).
TILE_DECODE_THREADS = 8


class PreviewClip:
    """One clip's tiles: a fixed crop, a frame at any offset, or every frame in order.

    ``proxy_frames[i]`` is the proxy frame of tile ``i`` and ``t_ns[i]`` the source time it
    shows (the part's own timestamp map — no second map). ``crop`` is ``(x0, y0, x1, y1)`` in
    proxy pixels, fixed for the clip so a tile never jitters. Not thread-safe: one clip, one
    caller; open one per tile.
    """

    def __init__(
        self,
        reader: "PreviewReader",
        object_id: int | None,
        proxy_frames: np.ndarray,
        crop: tuple[int, int, int, int],
        tile_px: int,
        *,
        footage: bool,
        mask: bool,
        trail: int,
        tint=None,
    ) -> None:
        self._reader = reader
        #: The animal's colour: the house mask tint unless the caller asks for another (the
        #: montage draws a clip outside the page's filters in grey).
        self.tint = visualize.MASK_TINT if tint is None else tuple(tint)
        self.object_id = object_id
        self.proxy_frames = proxy_frames
        self.t_ns = reader.proxy_t_ns[proxy_frames]
        self.crop = crop
        self.tile_px = int(tile_px)
        self._footage, self._mask, self._trail = bool(footage), bool(mask), int(trail)
        # A FrameCursor, not a raw container: it walks forward one decode per frame and lets
        # go of the file when idle, so a tile left open never pins the proxy (on Windows an
        # open handle would stop Clear from deleting it).
        self._decoder = None
        if footage:
            from detail.storage.video import FrameCursor

            self._decoder = FrameCursor(
                reader.frames.video,
                timestamps=_proxy_pts(reader.frames.video),
                threads=TILE_DECODE_THREADS,
            )
        self._rows: dict[int, int] = {}
        masks = reader.masks
        if masks is not None and object_id is not None and len(proxy_frames):
            lo = int(proxy_frames[0]) - max(self._trail - 1, 0)
            for row in masks.rows_for(object_id, lo, int(proxy_frames[-1])):
                self._rows[int(masks.proxy_frame[row])] = int(row)
        self._tile_masks: dict[int, np.ndarray | None] = {}
        self._rings: dict[int, np.ndarray | None] = {}
        #: The **Aligned** orientation (:meth:`set_pose`): each tile frame's body-axis ends,
        #: normalized ``(n, 2, 2)`` (front, back), NaN where the frame has none; ``None`` draws
        #: every frame as recorded.
        self._axes: np.ndarray | None = None
        #: A side-view run's **Aligned** orientation (:meth:`set_pose`): its embedding's flip per
        #: tile frame (``ClipPoints.flips``), the tile drawn as recorded and mirrored where
        #: flagged; ``None`` on any other run.
        self._flips: np.ndarray | None = None
        #: Keypoints drawn on the tile (:meth:`set_pose`): ``(points (n, K, 2), scores (n, K),
        #: skeleton)``, or ``None``.
        self._points: tuple | None = None
        self._full_masks: dict[int, np.ndarray | None] = {}

    def __len__(self) -> int:
        return len(self.proxy_frames)

    def __iter__(self) -> Iterator[np.ndarray]:
        for offset in range(len(self)):
            yield self.frame_at(offset)

    def __enter__(self) -> "PreviewClip":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._decoder is not None:
            self._decoder.close()

    def _tile_mask(self, proxy_frame: int) -> np.ndarray | None:
        if proxy_frame in self._tile_masks:
            return self._tile_masks[proxy_frame]
        row = self._rows.get(int(proxy_frame))
        tile = None
        if row is not None:
            from detail.core.utils import decode_mask

            masks = self._reader.masks
            full = decode_mask(str(masks.mask_rle[row]), (masks.height, masks.width))
            tile = _to_tile(full, self.crop, self.tile_px, mask=True)
            if not tile.any():
                tile = None
        if len(self._tile_masks) > 4 * max(self._trail, 1):
            self._tile_masks.pop(next(iter(self._tile_masks)))
        self._tile_masks[proxy_frame] = tile
        return tile

    def source_frames(self) -> np.ndarray:
        """The part-local source frame each tile frame shows — what a keypoint run is
        indexed by."""
        src = np.asarray(self._reader.frames.src_frame, dtype=np.int64)
        return src[np.clip(self.proxy_frames, 0, len(src) - 1)]

    def set_pose(
        self,
        *,
        axes: np.ndarray | None = None,
        points: tuple | None = None,
        flips: np.ndarray | None = None,
    ) -> None:
        """Turn the tiles so the body points up (``axes``: ``(n, 2, 2)`` normalized front and
        back per tile frame, NaN where there is none) and/or draw the keypoints on them
        (``points``: ``(points (n, K, 2), scores (n, K), skeleton)``), one row per tile frame.
        A side-view run is Aligned by ``flips`` instead (``(n,)``, ``ClipPoints.flips``): each
        tile drawn as recorded and mirrored where its embedding mirrored the frame, never
        turned, so it needs no axes (``clip_views.mirror_as_embedded``).

        Raises:
            ValueError: a row count that is not the clip's, or both ``axes`` and ``flips``.
        """
        if axes is not None and flips is not None:
            raise ValueError(
                "axes turn an overhead run's tiles and flips mirror a side view's: pass one"
            )
        for name, rows in (
            ("axes", axes),
            ("points", None if points is None else points[0]),
            ("flips", flips),
        ):
            if rows is not None and len(rows) != len(self):
                raise ValueError(
                    f"{name}: {len(rows)} rows for a clip of {len(self)} frames"
                )
        self._axes = None if axes is None else np.asarray(axes, dtype=np.float64)
        self._flips = None if flips is None else np.asarray(flips, dtype=np.float64)
        self._points = points

    def _matrix(self, offset: int) -> np.ndarray | None:
        """The turn for tile ``offset`` (``clip_views.turn_matrix``), or ``None``: as recorded,
        whether by choice, because the frame gives no axis, or because a side view is never
        turned."""
        if self._axes is None:
            return None
        ends = self._axes[offset]
        if not np.isfinite(ends).all():
            return None
        from detail.api.clip_views import turn_matrix

        wh = np.array(self._reader.size, dtype=np.float64)
        front, back = ends[0] * wh, ends[1] * wh
        if np.allclose(front, back):
            return None
        x0, y0, x1, y1 = self.crop
        return turn_matrix(
            front, back, max(x1 - x0, y1 - y0, 1), (self.tile_px, self.tile_px)
        )

    def frame_at(self, offset: int) -> np.ndarray:
        """Tile ``offset`` of the clip: ``(tile_px, tile_px, 3)`` uint8 RGB.

        With footage, the proxy frame with the animal outlined and its trail as fading
        outlines; without, its filled mask and trail on white. A frame past the proxy's
        end (a short file) is a grey tile. Turned so the body points up (a side view: mirrored
        where its embedding mirrored the frame), and the keypoints drawn, when :meth:`set_pose`
        asked; a frame with no axis (a side view: no embedded row) is drawn as recorded and
        marked "no axis"."""
        offset = min(max(int(offset), 0), len(self) - 1)
        proxy_frame = int(self.proxy_frames[offset])
        size = self.tile_px
        matrix = self._matrix(offset)
        if self._decoder is not None:
            found = self._decoder.frame(proxy_frame)
            rgb = None if found is None else visualize.bgr_to_rgb(found[1])
            if rgb is None:
                return np.full((size, size, 3), 90, dtype=np.uint8)
            if matrix is None:
                tile = _to_tile(rgb, self.crop, size)
            else:
                from detail.api.clip_views import turn_picture

                tile = turn_picture(rgb, matrix, (size, size), (0, 0, 0))
        else:
            tile = np.full((size, size, 3), 255, dtype=np.uint8)
        if self._mask and self._rows:
            outline = self._decoder is not None
            if matrix is None:
                history = [
                    self._trail_layer(frame, outline)
                    for frame in range(proxy_frame - self._trail + 1, proxy_frame)
                ]
                now = self._tile_mask(proxy_frame)
                ring = self._trail_layer(proxy_frame, outline) if outline else None
            else:
                history = [
                    self._turned_layer(frame, matrix, outline)
                    for frame in range(proxy_frame - self._trail + 1, proxy_frame)
                ]
                now = self._turned_layer(proxy_frame, matrix, False)
                ring = (
                    self._turned_layer(proxy_frame, matrix, True) if outline else None
                )
            tile = _draw_masks(tile, history, now, ring, self.tint)
        if self._points is not None:
            tile = self._draw_points(tile, offset, matrix)
        from detail.api.clip_views import mark_no_axis, mirror_as_embedded

        if self._flips is not None:
            mirrored = mirror_as_embedded(tile, float(self._flips[offset]))
            return mark_no_axis(tile) if mirrored is None else mirrored
        if self._axes is not None and matrix is None:
            tile = mark_no_axis(tile)
        return tile

    def _crop_matrix(self) -> np.ndarray:
        """``_to_tile``'s crop, square and scale as an affine: proxy pixels to tile pixels."""
        x0, y0, x1, y1 = self.crop
        side = max(x1 - x0, y1 - y0, 1)
        top, left = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
        k = self.tile_px / side
        return np.array([[k, 0.0, k * (left - x0)], [0.0, k, k * (top - y0)]])

    def _draw_points(self, tile: np.ndarray, offset: int, matrix) -> np.ndarray:
        """The subject's keypoints on ``tile``, placed the way its picture was."""
        from detail.api.clip_views import turn_points
        from detail.core.visualize import draw_keypoints

        points, scores, skeleton = self._points
        xy = points[offset]
        if not np.isfinite(xy).any():
            return tile
        size = (self.tile_px, self.tile_px)
        placed = turn_points(
            xy,
            self._crop_matrix() if matrix is None else matrix,
            self._reader.size,
            size,
        )
        edge = (235, 235, 235) if self._decoder is not None else (110, 110, 110)
        return draw_keypoints(
            tile, placed, skeleton, scores=scores[offset], edge_color=edge
        )

    def _full_mask(self, proxy_frame: int) -> np.ndarray | None:
        """The subject's mask on ``proxy_frame`` at proxy size, kept for the trail's frames."""
        if proxy_frame in self._full_masks:
            return self._full_masks[proxy_frame]
        row = self._rows.get(int(proxy_frame))
        full = None
        if row is not None:
            from detail.core.utils import decode_mask

            masks = self._reader.masks
            full = decode_mask(str(masks.mask_rle[row]), (masks.height, masks.width))
            if not full.any():
                full = None
        if len(self._full_masks) > 4 * max(self._trail, 1):
            self._full_masks.pop(next(iter(self._full_masks)))
        self._full_masks[proxy_frame] = full
        return full

    def _turned_layer(self, proxy_frame: int, matrix, outline: bool):
        """A frame's mask (or, ``outline``, its ring) turned by the **current** frame's matrix:
        an aligned trail is where the animal was relative to its body now."""
        from detail.api.clip_views import turn_mask

        full = self._full_mask(proxy_frame)
        if full is None:
            return None
        turned = turn_mask(full, matrix, (self.tile_px, self.tile_px))
        if turned is None or not outline:
            return turned
        from detail.api.clip_montage import outline_px

        return visualize.mask_outline(turned, outline_px(self.tile_px))

    def _trail_layer(self, proxy_frame: int, outline: bool) -> np.ndarray | None:
        """What a frame contributes to a trail: its outline ring on footage, its mask on
        white. Kept per frame, so walking a clip makes each ring once rather than once for
        every later frame whose trail it is in."""
        if not outline:
            return self._tile_mask(proxy_frame)
        if proxy_frame not in self._rings:
            from detail.api.clip_montage import outline_px

            mask = self._tile_mask(proxy_frame)
            if len(self._rings) > 4 * max(self._trail, 1):
                self._rings.pop(next(iter(self._rings)))
            self._rings[proxy_frame] = (
                None
                if mask is None
                else visualize.mask_outline(mask, outline_px(self.tile_px))
            )
        return self._rings[proxy_frame]


def _draw_masks(tile, history, now, ring, tint=visualize.MASK_TINT) -> np.ndarray:
    """The animal and its trail on a tile, at the tile's own size (``clip_montage``'s look:
    outlined on footage — ``ring`` is the current outline — filled on white without;
    ``visualize``'s fade and blend)."""
    size = tile.shape[0]
    layers = [None if m is None else (0, 0, m) for m in history]
    weight = np.zeros((size, size), dtype=np.float32)
    faded = visualize.fade_weights(layers, alpha=visualize.ONION_ALPHA)
    if faded is not None:
        part, top, left = faded
        weight[top : top + part.shape[0], left : left + part.shape[1]] = part
    if now is not None:
        if ring is not None:
            weight[now] = 0.0
            weight[ring] = 1.0
        else:
            weight[now] = 1.0
    if not weight.any():
        return tile
    return visualize.tint_by_weight(tile, weight, tint)


class PreviewReader:
    """One part's preview: its proxy frames, their source times, and (with a tracking run)
    its animals' masks — the fast path the clip montage draws from.

    Open with :meth:`open`; ask for a clip with :meth:`clip`. Reading needs nothing but the
    cache and the part's timestamp map: no source video is opened.
    """

    def __init__(
        self,
        frames: store.ProxyFrames,
        timestamps: np.ndarray,
        masks: store.MaskRows | None = None,
        *,
        masks_missing: bool = False,
    ) -> None:
        self.frames = frames
        self.masks = masks
        #: True when masks were asked for and are not built here yet: the reader still draws
        #: footage (whole-frame crop, no outline), so a montage can play while they build.
        self.masks_missing = bool(masks_missing)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        #: Source ``t_ns`` of each proxy frame, read off the part's own map (Tenet 8).
        self.proxy_t_ns = timestamps[np.clip(frames.src_frame, 0, len(timestamps) - 1)]

    @classmethod
    def open(
        cls, recording, part_index: int, tracking_run_id: str | None = None
    ) -> "PreviewReader":
        """The preview of ``recording``'s part ``part_index``, with ``tracking_run_id``'s masks.

        Masks asked for but not built yet do not refuse the open: the reader comes back
        footage-only with :attr:`masks_missing` set, and a caller reopens once they are built.

        Raises:
            PreviewMissingError: If the proxy is not built here (or the video is not here).
        """
        from detail.api.run_preview import tracking_entries

        data = recording_store.read_recording(recording.dir)
        part = recording_store.video_files(data)[part_index]
        path = _part_video(recording.dir, part)
        if path is None:
            raise PreviewMissingError(
                f"{recording.name} part {part_index + 1}: the video is not on this machine"
            )
        key = store.key_for_video(path)
        try:
            frames = store.read_frames(key)
        except FileNotFoundError as exc:
            raise PreviewMissingError(str(exc)) from None
        masks = None
        masks_missing = False
        if tracking_run_id is not None:
            entry, detection = tracking_entries(data, tracking_run_id)
            masks_at = store.masks_path(
                key, tracking_run_id, store.masks_digest(entry, detection)
            )
            try:
                masks = store.read_masks(masks_at)
            except FileNotFoundError:
                masks_missing = True  # footage now; the masks follow when built
        timestamps = recording_store.load_active_timestamps_cached(
            recording.dir, data, part_index
        ).values
        return cls(frames, timestamps, masks, masks_missing=masks_missing)

    @property
    def size(self) -> tuple[int, int]:
        """The proxy's ``(width, height)``."""
        return self.frames.width, self.frames.height

    def proxy_frame_at(self, t_ns: int) -> int:
        """The proxy frame showing source time ``t_ns``: the last one at or before it."""
        index = int(np.searchsorted(self.proxy_t_ns, int(t_ns), side="right")) - 1
        return min(max(index, 0), len(self.proxy_t_ns) - 1)

    def clip(
        self,
        object_id: int | None,
        start_t_ns: int,
        end_t_ns: int,
        tile_px: int = 160,
        *,
        footage: bool = True,
        mask: bool = True,
        trail: int = TRAIL_FRAMES,
        tint=None,
    ) -> PreviewClip:
        """A clip of animal ``object_id`` from ``start_t_ns`` to ``end_t_ns`` (inclusive), as
        ``tile_px`` tiles. The crop is the clip montage's (:func:`clip_montage.crop_box`) over
        the animal's stored boxes; the whole frame when it has none (or ``object_id`` is None).
        ``tint`` is the animal's RGB colour (default: the house mask tint).
        """
        from detail.api.clip_montage import crop_box

        lo = self.proxy_frame_at(start_t_ns)
        hi = max(lo, self.proxy_frame_at(end_t_ns))
        proxy_frames = np.arange(lo, hi + 1, dtype=np.int64)
        width, height = self.size
        boxes = np.zeros((0, 4))
        if self.masks is not None and object_id is not None:
            rows = self.masks.rows_for(object_id, lo, hi)
            boxes = self.masks.bbox[rows].astype(np.float64)
        crop = crop_box(boxes, (width, height))
        return PreviewClip(
            self,
            object_id,
            proxy_frames,
            crop,
            tile_px,
            footage=footage,
            mask=mask,
            trail=trail,
            tint=tint,
        )
