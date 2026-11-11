"""The frames a tracked subject is on, and how an example set samples them (§26 "Example sets").

``subject_frames`` is the one answer to "which frames does this track have a subject on, and
where is it": a track may hold several detections in one frame (an anchor tracker's secondaries),
and they are one subject, so the frame's box is their union — the same merge embedding does
(§13), so every stage sees the same animal. The keypoint stages (PR 4, 5) read it too.

Two sampling strategies, both copying tracked-subject frames into an example set unmarked:

- ``varied`` (the default) picks frames that look *different*, from the subject itself rather
  than the clock. A spread over time hands a resting animal five copies of one posture.

  0. **Recordings.** Above ``DEFAULT_FROM_RECORDINGS`` sources, only that many are read:
     ``choose_recordings`` cuts the full set into equal runs and picks one recording from each,
     so a large experiment is sampled from end to end rather than from its first recordings,
     and the wait stops growing with the recordings. ``from_recordings=None`` reads every one.
  1. **Shape pass, every tracked frame** (thinned to ``SHAPE_POOL`` per recording, no video
     decode): each frame's subject mask gives its posture and the direction of its long axis
     (``shape_row``). Frames whose box touches the frame edge, or whose area is far from the
     recording's median (a merged pair, a fragment), are never candidates: sampling the extremes
     of a feature space otherwise picks exactly those.
  2. **Groups.** k-means on the standardized shape rows. The member nearest each centre stands
     for its group, so a pick is a posture many frames share rather than one odd frame.
  3. **Embedding pass, representatives only.** Each is decoded, cropped to a padded square
     around its box, and embedded by the DINOv3 ViT the patch-matching method is built on
     (the largest installed of ``BACKBONES``), so "different" means different *to that
     backbone*, head and ears included, not only in outline. With no DINOv3 ViT installed the
     shape rows stand in, and ``sampled_by`` says so.
  4. **Choose.** An empty set opens with the ``n_typical`` representatives nearest the
     frame-weighted mean: the middle of the spectrum. Then ``n_frames`` farthest-first picks,
     each the representative farthest from every frame already in the set and picked so far.
     A second pass fills the gaps the set has rather than starting over.

- ``spread``: one random frame from each of ``n_frames`` equal time bins, spread across the
  recordings in proportion to their tracked frames. The removed keypoint audit drew its sets with
  this and only this (§26 "Auditing"): an audit needs an unbiased sample, not a varied one.

Each copied frame is decoded exactly (``read_frame_at`` with the part's timestamp map: Tenet
11's labeling-sampler exception, since an example frame's index is a label, not a join key, so it
may seek) and stored as a lossless PNG with its subject box. Whether ``varied`` beats ``spread``
on accuracy is **not measured** (§26 *Open questions* 11).
"""

from __future__ import annotations

import zlib
from pathlib import Path

import cv2
import numpy as np
import yaml

from detail.core import atomic
from detail.core.examples import ExampleFrame, ExampleSet
from detail.core.progress import ThrottledProgress
from detail.core.utils import decode_mask
from detail.core.visualize import bgr_to_rgb
from detail.executor import rundir
from detail.storage import example_sets
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage.video import read_frame_at
from detail.storage.zarr_io import DetectionPartReader, PartZarrReader

STRATEGIES = ("varied", "spread")
DEFAULT_STRATEGY = "varied"
#: A varied sample reads the shape of every tracked frame of every recording it is given, so the
#: wait grows with the recordings, and a few dozen already hold the postures a set needs. Above
#: this many, ``varied`` samples from this many, spread evenly through the full set.
DEFAULT_FROM_RECORDINGS = 20
#: Typical frames a varied sample opens an empty set with: the middle of the spectrum.
DEFAULT_N_TYPICAL = 5
# The shape pass reads at most this many tracked frames per recording, evenly spaced: a mask
# decode is about a millisecond at 1080p, and a whole long recording would add a minute without
# making the groups any better.
SHAPE_POOL = 3000
# Groups the shape pass makes. One representative of each is decoded and embedded, so this
# bounds the seeks; four per requested frame leaves farthest-first room to choose.
MIN_GROUPS, MAX_GROUPS, GROUPS_PER_FRAME = 40, 150, 4
# A box this close to the frame edge is a subject partly out of view: never a candidate.
EDGE_MARGIN = 0.005
# A mask area outside this range of the recording's median (two animals merged, a fragment) is
# never a candidate. The extremes of a feature space are where such frames sit.
AREA_RANGE = (0.4, 2.5)
#: DINOv3 ViT sizes the embedding pass uses, best first: the patch-matching method's own two.
BACKBONES = ("vit_large", "vit_base")
EMBED_BATCH = 16
# The padded square the embedding pass crops around a subject's box, as a fraction of the box.
CROP_PAD = 0.15


# --------------------------------------------------------------------------- #
# Which frames a track has a subject on
# --------------------------------------------------------------------------- #
def subject_frames(
    det: dict, object_id: np.ndarray, track_id: int, every_n_frames: int
) -> list[dict]:
    """The track's frames in this part at the run's cadence, in frame order:
    ``{frame_idx, t_ns, bbox, rows}``.

    A track may hold several detections in one frame (an anchor tracker's secondaries); they are
    one subject, so the frame's box is their union and ``rows`` are its detection rows (their
    masks OR together), the merge embedding does (§13).
    """
    keep = np.flatnonzero(np.asarray(object_id) == int(track_id))
    frame_col = np.asarray(det["frame_idx"])
    by_frame: dict[int, list[int]] = {}
    for row in keep.tolist():
        by_frame.setdefault(int(frame_col[row]), []).append(row)
    bbox = np.asarray(det["bbox"])
    t_ns = np.asarray(det["t_ns"])
    out = []
    for frame in sorted(by_frame):
        if frame % every_n_frames:
            continue
        rows = by_frame[frame]
        boxes = bbox[rows]
        out.append(
            {
                "frame_idx": frame,
                "t_ns": int(t_ns[rows[0]]),
                "bbox": (
                    float(boxes[:, 0].min()),
                    float(boxes[:, 1].min()),
                    float(boxes[:, 2].max()),
                    float(boxes[:, 3].max()),
                ),
                "rows": rows,
            }
        )
    return out


def tracked_frames(
    recording_dir: Path | str,
    tracking_run_id: str,
    track_ids: list[int] | None = None,
) -> list[dict]:
    """Every frame a track of the run has a subject on: ``{part_id, frame_idx, t_ns,
    track_id, bbox}``, the box the union of the track's detections in that frame.

    Raises:
        ValueError: No such tracking run in the recording.
    """
    rows, _size = _tracked_rows(recording_dir, tracking_run_id, track_ids)
    for row in rows:
        del row["det"], row["rows"]
    return rows


def tracked_frame(
    recording_dir: Path | str,
    tracking_run_id: str,
    part_id: str,
    frame_idx: int,
    track_id: int,
) -> dict | None:
    """One frame's row of :func:`tracked_frames`, or None when the track has no subject on it.

    Reads the one part's ``frame_idx`` column and then only that frame's rows: a hand-picked
    frame needs one box, and scanning the whole run for it (every column of every part, masks
    included) cost seconds on a long recording, paid on the labeling screen's first
    click on each frame.

    Raises:
        ValueError: No such tracking run in the recording.
    """
    recording_dir = Path(recording_dir)
    tracking_dir = recording_dir / "tracking" / tracking_run_id
    meta_path = tracking_dir / "track.yaml"
    if not meta_path.is_file():
        raise ValueError(
            f"no tracking run {tracking_run_id!r} in recording {recording_dir.name!r}"
        )
    track_meta = yaml.safe_load(atomic.read_text(meta_path))
    det_path = (
        recording_dir
        / "detections"
        / track_meta["detection_run_id"]
        / f"{part_id}.zarr"
    )
    assignment_path = tracking_dir / "assignments" / f"{part_id}.zarr"
    if not det_path.is_dir() or not assignment_path.is_dir():
        return None
    det = DetectionPartReader(det_path)
    rows = np.flatnonzero(det.column("frame_idx")[:] == int(frame_idx))
    if rows.size == 0:
        return None
    lo, hi = int(rows[0]), int(rows[-1]) + 1  # a frame's rows are contiguous
    object_id = np.asarray(PartZarrReader(assignment_path).column("object_id")[lo:hi])
    window = {
        "frame_idx": np.full(hi - lo, int(frame_idx)),
        "bbox": np.asarray(det.column("bbox")[lo:hi]),
        "t_ns": np.asarray(det.column("t_ns")[lo:hi]),
    }
    found = subject_frames(window, object_id, int(track_id), 1)
    if not found:
        return None
    item = found[0]
    return {
        "part_id": part_id,
        "frame_idx": item["frame_idx"],
        "t_ns": item["t_ns"],
        "track_id": int(track_id),
        "bbox": item["bbox"],
    }


def _tracked_rows(
    recording_dir: Path | str, tracking_run_id: str, track_ids: list[int] | None
) -> tuple[list[dict], tuple[int, int] | None]:
    """``tracked_frames``' rows plus each one's detection ``rows`` and part ``det`` (to decode
    its mask from), and the detection run's mask size ``(h, w)``, None when it has no masks.
    """
    recording_dir = Path(recording_dir)
    tracking_dir = recording_dir / "tracking" / tracking_run_id
    meta_path = tracking_dir / "track.yaml"
    if not meta_path.is_file():
        raise ValueError(
            f"no tracking run {tracking_run_id!r} in recording {recording_dir.name!r}"
        )
    track_meta = yaml.safe_load(atomic.read_text(meta_path))
    detection_dir = recording_dir / "detections" / track_meta["detection_run_id"]
    out = []
    for det_path in sorted(detection_dir.glob("part_*.zarr")):
        assignment_path = tracking_dir / "assignments" / det_path.name
        if not assignment_path.is_dir():
            continue
        det = _part_columns(DetectionPartReader(det_path))
        object_id = np.asarray(PartZarrReader(assignment_path).read()["object_id"])
        present = sorted({int(t) for t in object_id if int(t) >= 0})
        for track_id in present if track_ids is None else track_ids:
            for item in subject_frames(det, object_id, int(track_id), 1):
                out.append(
                    {
                        "part_id": det_path.stem,
                        "frame_idx": item["frame_idx"],
                        "t_ns": item["t_ns"],
                        "track_id": int(track_id),
                        "bbox": item["bbox"],
                        "rows": item["rows"],
                        "det": det,
                    }
                )
    return out, _mask_size(detection_dir)


#: What a tracked row needs of its detection part, read whole; the mask strings are not.
_ROW_COLUMNS = ("frame_idx", "bbox", "t_ns")


def _part_columns(reader: DetectionPartReader) -> dict:
    """A detection part as the sampler and the keypoint stages use it: the small columns read,
    ``mask_rle`` left on disk behind :class:`_LazyColumn`.

    On a long part most of a whole-part read is the mask strings, of which a sampler uses
    ~3,000 and Test cut-off ~12.
    """
    out = {name: reader.column(name)[:] for name in _ROW_COLUMNS}
    if "mask_rle" in reader.column_names:
        out["mask_rle"] = _LazyColumn(reader.column("mask_rle"))
    return out


class _LazyColumn:
    """A zarr column read by the chunk on first use, the last few chunks kept.

    Rows are asked for in time order, so a chunk is read once however many of its rows are
    used; a random order still works, it just reads a chunk again.
    """

    _KEEP = 8

    def __init__(self, array) -> None:
        self._array = array
        self._chunk = int(array.chunks[0])
        self._cache: dict[int, np.ndarray] = {}
        self._whole: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self._array.shape[0])

    def prefetch(self, rows) -> None:
        """Read the whole column now if ``rows`` touch more than half its chunks.

        A chunk read on its own costs about twice its share of one whole-column read:
        the shape pass's evenly spaced rows touch most chunks of a long part, and
        fetching them one at a time made the mask step several times slower.
        """
        chunks = {int(r) // self._chunk for r in rows}
        n_chunks = -(-len(self) // self._chunk)
        if len(chunks) * 2 > n_chunks:
            self._whole = self._array[:]

    def __getitem__(self, row) -> object:
        whole = self._whole
        if whole is not None:
            return whole[int(row)]
        row = int(row)
        index = row // self._chunk
        block = self._cache.get(index)
        if block is None:
            start = index * self._chunk
            block = self._array[start : start + self._chunk]
            if len(self._cache) >= self._KEEP:
                self._cache.pop(next(iter(self._cache)))
            self._cache[index] = block
        return block[row - index * self._chunk]


def _prefetch_masks(rows: list[dict]) -> None:
    """Tell each part's mask column which rows are coming (:meth:`_LazyColumn.prefetch`)."""
    by_part: dict[int, tuple] = {}
    for row in rows:
        column = row["det"].get("mask_rle")
        if isinstance(column, _LazyColumn):
            by_part.setdefault(id(column), (column, []))[1].extend(row["rows"])
    for column, wanted in by_part.values():
        column.prefetch(wanted)


def _mask_size(detection_dir: Path) -> tuple[int, int] | None:
    if not (detection_dir / rundir.JOB_FILENAME).is_file():
        return None
    size = rundir.read_envelope(detection_dir).get("mask_resolution")
    return tuple(int(v) for v in size) if size else None


# --------------------------------------------------------------------------- #
# Allocation and the spread draw
# --------------------------------------------------------------------------- #
def _allocate(counts: list[int], n: int) -> list[int]:
    """``n`` frames over sources with ``counts`` candidates: proportional, at least one each
    while ``n`` allows, never more than a source has."""
    total = sum(counts)
    if total == 0:
        return [0] * len(counts)
    n = min(n, total)
    want = [c * n / total for c in counts]
    take = [min(c, int(w)) for c, w in zip(counts, want)]
    if n >= sum(1 for c in counts if c):
        take = [max(t, 1) if c else 0 for t, c in zip(take, counts)]
    order = sorted(
        range(len(counts)), key=lambda i: want[i] - int(want[i]), reverse=True
    )
    while sum(take) < n:
        moved = False
        for i in order:
            if sum(take) >= n:
                break
            if take[i] < counts[i]:
                take[i] += 1
                moved = True
        if not moved:
            break
    while sum(take) > n:
        i = max(range(len(take)), key=lambda j: take[j])
        take[i] -= 1
    return take


def _spread(candidates: list[dict], k: int, rng: np.random.Generator) -> list[dict]:
    """``k`` candidates spread across time: one at random from each of ``k`` equal bins."""
    if k <= 0 or not candidates:
        return []
    ordered = sorted(
        candidates, key=lambda c: (c["part_id"], c["frame_idx"], c["track_id"])
    )
    edges = np.linspace(0, len(ordered), k + 1).astype(int)
    picks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            picks.append(ordered[int(rng.integers(lo, hi))])
    return picks


def choose_recordings(recording_ids: list[str], k: int, seed: int = 0) -> list[str]:
    """``k`` of ``recording_ids``, **stratified across the whole list**, in list order.

    The list is cut into ``k`` equal runs and one recording is picked from each at random, so the
    sample reaches every part of the set: a library imported in blocks (two strains, several
    days) is represented from each block, which taking the first ``k`` would not do. The same
    ``seed`` picks the same recordings; ``k`` of the list or more returns the list as it is.

    Raises:
        ValueError: ``k`` is below one.
    """
    if k < 1:
        raise ValueError(f"from_recordings must be 1 or more; got {k}")
    n = len(recording_ids)
    if k >= n:
        return list(recording_ids)
    edges = np.linspace(0, n, k + 1).astype(int)
    rng = np.random.default_rng(seed)
    return [
        recording_ids[int(rng.integers(low, high))]
        for low, high in zip(edges[:-1], edges[1:])
    ]


def _check_request(sources: dict[str, str], n_frames: int) -> None:
    if n_frames < 1:
        raise ValueError("sample at least one frame")
    if not sources:
        raise ValueError("choose at least one recording to sample frames from")


def _nothing_left(track_ids) -> ValueError:
    return ValueError(
        "the chosen tracking runs have no tracked frames left to sample"
        + (f" for tracks {track_ids}" if track_ids else "")
    )


def _part_index(recording: dict, part_id: str) -> int:
    for index, part in enumerate(recording_store.video_files(recording)):
        if recording_store.part_stem(part) == part_id:
            return index
    raise ValueError(f"the recording has no video part {part_id!r}")


def _require_videos(recording_dir: Path, part_ids: set[str]) -> None:
    """Refuse up front a recording whose video is missing: a varied pick may never land on it,
    and a user's missing video is something to fix, not a source to skip quietly."""
    data = recording_store.read_recording(recording_dir)
    files = recording_store.video_files(data)
    for part_id in sorted(part_ids):
        path = recording_store.resolve_part_video_path(
            recording_dir, files[_part_index(data, part_id)]
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"video not present: {path}; bring the source video to this machine to "
                "sample example frames from it"
            )


def _decode(experiment_dir: Path, recording_id: str, cand: dict) -> np.ndarray | None:
    """``cand``'s frame as RGB uint8, decoded exactly; None when the decoder lands anywhere
    but that frame (a picture is never paired with another frame's box)."""
    recording_dir = experiment_store.recording_path(experiment_dir, recording_id)
    data = recording_store.read_recording(recording_dir)
    files = recording_store.video_files(data)
    part_index = _part_index(data, cand["part_id"])
    tsmap = recording_store.load_active_timestamps_cached(
        recording_dir, data, part_index
    )
    decoded = read_frame_at(
        recording_store.resolve_part_video_path(recording_dir, files[part_index]),
        cand["frame_idx"],
        timestamps=tsmap.values,
    )
    if decoded is None or decoded[0] != cand["frame_idx"]:
        return None
    return bgr_to_rgb(decoded[1])


def _copy_frame(
    experiment_dir: Path,
    recording_id: str,
    tracking_run_id: str,
    cand: dict,
    add_image,
    *,
    sampled_by: str | None = None,
) -> tuple[ExampleFrame, tuple[int, int]] | None:
    """Decode ``cand``'s frame exactly, copy it in through ``add_image``, return its row and
    the frame's size, or None when it cannot be decoded exactly (``_decode``)."""
    rgb = _decode(experiment_dir, recording_id, cand)
    if rgb is None:
        return None
    frame = ExampleFrame(
        image=add_image(rgb),
        recording_id=recording_id,
        tracking_run_id=tracking_run_id,
        track_id=int(cand["track_id"]),
        part_id=str(cand["part_id"]),
        frame_idx=int(cand["frame_idx"]),
        t_ns=int(cand["t_ns"]),
        bbox=tuple(float(v) for v in cand["bbox"]),
        sampled_by=sampled_by,
    )
    return frame, (int(rgb.shape[1]), int(rgb.shape[0]))


def _key(row: dict) -> tuple[str, str, int, int]:
    """``ExampleFrame.key()`` for a candidate row."""
    return (row["recording_id"], row["part_id"], int(row["t_ns"]), int(row["track_id"]))


# --------------------------------------------------------------------------- #
# The two entry points: a sample into a set, a single picked frame
# --------------------------------------------------------------------------- #
def sample_frames(
    experiment_dir: Path | str,
    set_id: str,
    sources: dict[str, str],
    n_frames: int,
    *,
    strategy: str = DEFAULT_STRATEGY,
    n_typical: int = DEFAULT_N_TYPICAL,
    from_recordings: int | None = DEFAULT_FROM_RECORDINGS,
    track_ids: list[int] | None = None,
    seed: int = 0,
    device: str | None = None,
    progress=None,
) -> list[ExampleFrame]:
    """Decode frames from ``sources`` (recording id -> tracking run id) into the set's
    ``frames/`` and return their ``ExampleFrame`` rows, unmarked and **not yet in the set**:
    the caller appends them and writes the set (``api/keypoints.py``).

    ``varied``: ``n_frames`` farthest-first picks, opened by ``n_typical`` typical frames
    when the set is empty (module docstring), from ``from_recordings`` of the recordings when
    there are more than that (:func:`choose_recordings`; ``None`` uses every one). ``spread``:
    ``n_frames`` spread over time across every recording; ``n_typical`` and
    ``from_recordings`` do not apply. Frames the set already holds are not drawn again.
    ``device`` runs the DINOv3 pass (the fastest present when None).
    ``progress(stage, fraction, detail)`` names each phase as it starts.

    Raises:
        ValueError: An unknown strategy, a negative ``n_typical``, a ``from_recordings``
            below one, or ``draw_frames``' reasons.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown sampling strategy {strategy!r}; expected one of {STRATEGIES}"
        )
    if n_typical < 0:
        raise ValueError(f"n_typical must be 0 or more; got {n_typical}")
    if from_recordings is not None and from_recordings < 1:
        raise ValueError(f"from_recordings must be 1 or more; got {from_recordings}")
    experiment_dir = Path(experiment_dir)
    example_set = example_sets.read_set(experiment_dir, set_id)
    existing = {f.key() for f in example_set.frames}

    created: list[Path] = []

    def add_image(rgb):
        relative = example_sets.add_frame_image(experiment_dir, set_id, rgb)
        path = example_sets.set_dir(experiment_dir, set_id) / relative
        if path not in had:
            created.append(path)
        return relative

    frames_dir = example_sets.set_dir(experiment_dir, set_id) / example_sets.FRAMES_DIR
    had = set(frames_dir.glob("*.png")) if frames_dir.is_dir() else set()
    try:
        return _draw(
            experiment_dir,
            example_set,
            sources,
            n_frames,
            strategy=strategy,
            existing=existing,
            add_image=add_image,
            n_typical=n_typical,
            from_recordings=from_recordings,
            track_ids=track_ids,
            seed=seed,
            device=device,
            progress=progress,
        )
    except BaseException:
        # A cancelled or failed draw leaves no pictures behind: the set's YAML never names
        # them (the caller writes it after this returns), so they would only be litter. A
        # picture that was already there is the set's, and is kept.
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _draw(
    experiment_dir: Path,
    example_set: ExampleSet,
    sources: dict[str, str],
    n_frames: int,
    *,
    strategy: str,
    existing: set,
    add_image,
    n_typical: int,
    from_recordings: int | None,
    track_ids: list[int] | None,
    seed: int,
    device: str | None,
    progress,
) -> list[ExampleFrame]:
    if strategy == "spread":
        drawn = draw_frames(
            experiment_dir,
            sources,
            n_frames,
            exclude=existing,
            add_image=add_image,
            track_ids=track_ids,
            seed=seed,
            progress=progress,
            sampled_by="spread",
        )
        return [frame for frame, _wh in drawn]
    return varied_frames(
        experiment_dir,
        example_set,
        sources,
        n_frames,
        n_typical=n_typical if not example_set.frames else 0,
        from_recordings=from_recordings,
        add_image=add_image,
        track_ids=track_ids,
        seed=seed,
        device=device,
        progress=progress,
    )


def draw_frames(
    experiment_dir: Path | str,
    sources: dict[str, str],
    n_frames: int,
    *,
    exclude: set[tuple],
    add_image,
    track_ids: list[int] | None = None,
    seed: int = 0,
    progress=None,
    sampled_by: str | None = "spread",
) -> list[tuple[ExampleFrame, tuple[int, int]]]:
    """The ``spread`` sampler (example sets; the removed keypoint audit's sets too): ``n_frames``
    tracked frames from ``sources``, spread across recordings in proportion to their tracked
    frames and across time, none whose ``(recording_id, part_id, t_ns, track_id)`` is in
    ``exclude``. Each is decoded exactly and handed to ``add_image(rgb) -> relative path``;
    returns ``(row, (width, height))`` per frame copied.

    Raises:
        ValueError: No sources, ``n_frames`` below one, or no frames left to draw.
    """
    experiment_dir = Path(experiment_dir)
    _check_request(sources, n_frames)
    pools = {}
    for recording_id, tracking_run_id in sources.items():
        recording_dir = experiment_store.recording_path(experiment_dir, recording_id)
        frames = tracked_frames(recording_dir, tracking_run_id, track_ids)
        for f in frames:
            f["recording_id"] = recording_id
        pools[recording_id] = [f for f in frames if _key(f) not in exclude]
    recording_ids = list(pools)
    counts = [len(pools[r]) for r in recording_ids]
    if not sum(counts):
        raise _nothing_left(track_ids)
    take = _allocate(counts, n_frames)
    chosen = []
    for recording_id, k in zip(recording_ids, take):
        rng = np.random.default_rng(zlib.crc32(f"{recording_id}:{seed}".encode()))
        chosen.extend((recording_id, c) for c in _spread(pools[recording_id], k, rng))
    out: list[tuple[ExampleFrame, tuple[int, int]]] = []
    for i, (recording_id, cand) in enumerate(chosen):
        if progress is not None:
            progress("sampling frames", i / len(chosen), f"{i}/{len(chosen)} frames")
        copied = _copy_frame(
            experiment_dir,
            recording_id,
            sources[recording_id],
            cand,
            add_image,
            sampled_by=sampled_by,
        )
        if copied is not None:
            out.append(copied)
    if progress is not None:
        progress("sampling frames", 1.0, f"{len(out)}/{len(chosen)} frames")
    return out


def pick_frame(
    experiment_dir: Path | str,
    set_id: str,
    recording_id: str,
    tracking_run_id: str,
    part_id: str,
    frame_idx: int,
    track_id: int,
) -> ExampleFrame:
    """The one frame a user navigated to, copied into the set with the track's box on it, and
    returned **not yet in the set** (the caller appends and writes). ``sampled_by`` is None:
    hand-picked.

    Raises:
        ValueError: The track has no subject on that frame (there is no box to crop to), the
            set already holds the frame, or the frame cannot be decoded exactly.
    """
    experiment_dir = Path(experiment_dir)
    recording_dir = experiment_store.recording_path(experiment_dir, recording_id)
    cand = tracked_frame(
        recording_dir, tracking_run_id, part_id, int(frame_idx), int(track_id)
    )
    if cand is None:
        raise ValueError(
            f"track {track_id} of {tracking_run_id} has no subject on frame {frame_idx} "
            f"of {recording_id} {part_id}; an example frame needs the tracked box"
        )
    cand["recording_id"] = recording_id
    existing = {f.key() for f in example_sets.read_set(experiment_dir, set_id).frames}
    if _key(cand) in existing:
        raise ValueError(
            f"frame {frame_idx} of {recording_id} {part_id} (track {track_id}) is "
            f"already in example set {set_id!r}"
        )
    copied = _copy_frame(
        experiment_dir,
        recording_id,
        tracking_run_id,
        cand,
        lambda rgb: example_sets.add_frame_image(experiment_dir, set_id, rgb),
    )
    if copied is None:
        raise ValueError(
            f"frame {frame_idx} of {recording_id} {part_id} could not be decoded exactly"
        )
    return copied[0]


# --------------------------------------------------------------------------- #
# The varied strategy
# --------------------------------------------------------------------------- #
def shape_row(mask: np.ndarray | None, bbox) -> np.ndarray:
    """``[log area, elongation, e*cos 2a, e*sin 2a, solidity, extent]`` of one subject.

    Posture (how stretched, how compact, how much of its box it fills) and the direction of its
    long axis ``a``. The angle is doubled because a mask cannot tell head from tail, and weighted
    by the elongation ``e`` because a round blob has no direction worth comparing. Without a mask
    (a box-only detector), the box stands in.
    """
    x0, y0, x1, y1 = bbox
    if mask is None or not mask.any():
        bw, bh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        e = 1.0 - min(bw, bh) / max(bw, bh)
        return np.array(
            [np.log(bw * bh), e, e if bw >= bh else -e, 0.0, 1.0, 1.0],
            dtype=np.float64,
        )
    h, w = mask.shape
    c0, r0 = max(int(x0 * w) - 1, 0), max(int(y0 * h) - 1, 0)
    c1, r1 = min(int(np.ceil(x1 * w)) + 1, w), min(int(np.ceil(y1 * h)) + 1, h)
    crop = mask[r0:r1, c0:c1].astype(np.uint8)
    if not crop.any():
        crop = mask.astype(np.uint8)
    m = cv2.moments(crop, binaryImage=True)
    area = float(m["m00"])
    cov = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / area
    lo, hi = np.linalg.eigvalsh(cov)
    e = 1.0 - float(np.sqrt(max(lo, 0.0) / hi)) if hi > 0 else 0.0
    angle2 = float(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    # The hull of the outer outlines is the hull of every pixel (each pixel lies inside an
    # outline made of pixels). Built from every pixel it dominated this function; from
    # the outlines it is several times faster with identical rows.
    outlines, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull_area = float(cv2.contourArea(cv2.convexHull(np.vstack(outlines))))
    _x, _y, box_w, box_h = cv2.boundingRect(crop)
    box_area = float(box_w * box_h)
    return np.array(
        [
            np.log(area),
            e,
            e * np.cos(angle2),
            e * np.sin(angle2),
            min(area / hull_area, 1.0) if hull_area > 0 else 1.0,
            area / box_area,
        ],
        dtype=np.float64,
    )


def subject_mask(det: dict, rows: list[int], mask_size) -> np.ndarray | None:
    """The subject's mask on one frame: the OR of its detection ``rows``' masks (the rows
    ``subject_frames`` grouped), ``(H, W)`` bool at the detection run's ``mask_size``; None
    when the run kept no masks. The one merge both the sampler and the keypoint stages use,
    so every stage sees the same animal (§13)."""
    if mask_size is None:
        return None
    mask = decode_mask(str(det["mask_rle"][rows[0]]), mask_size)
    for r in rows[1:]:
        mask = mask | decode_mask(str(det["mask_rle"][r]), mask_size)
    return mask


def _subject_mask(row: dict, mask_size) -> np.ndarray | None:
    return subject_mask(row["det"], row["rows"], mask_size)


def _thin(rows: list[dict], cap: int, keep: set) -> list[dict]:
    """At most ``cap`` rows evenly spaced in time, plus every row whose key is in ``keep``."""
    if len(rows) <= cap:
        return rows
    picked = set(np.linspace(0, len(rows) - 1, cap).round().astype(int).tolist())
    return [r for i, r in enumerate(rows) if i in picked or _key(r) in keep]


def _fit_for_pick(row: dict, median_area: float) -> bool:
    """A frame a user can mark: the subject wholly in view, one animal's worth of area."""
    x0, y0, x1, y1 = row["bbox"]
    if min(x0, y0) < EDGE_MARGIN or max(x1, y1) > 1 - EDGE_MARGIN:
        return False
    ratio = float(np.exp(row["shape"][0])) / median_area
    return AREA_RANGE[0] <= ratio <= AREA_RANGE[1]


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 50):
    """``(labels, centres)``: k-means++ seeding, then Lloyd's iterations to a fixed point."""
    n = len(x)
    k = min(k, n)
    centres = [x[int(rng.integers(n))]]
    d2 = ((x - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        total = d2.sum()
        i = int(rng.choice(n, p=d2 / total)) if total > 0 else int(rng.integers(n))
        centres.append(x[i])
        d2 = np.minimum(d2, ((x - x[i]) ** 2).sum(1))
    c = np.array(centres)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        labels = ((x[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
        new = np.array(
            [x[labels == j].mean(0) if (labels == j).any() else c[j] for j in range(k)]
        )
        if np.allclose(new, c):
            break
        c = new
    return labels, c


def representatives(
    x: np.ndarray, k: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """``(index, group size)`` per non-empty group: the member nearest the group's centre."""
    labels, centres = kmeans(x, k, rng)
    out = []
    for j in range(len(centres)):
        members = np.flatnonzero(labels == j)
        if len(members):
            d = ((x[members] - centres[j]) ** 2).sum(1)
            out.append((int(members[int(d.argmin())]), int(len(members))))
    return out


def choose(
    feats: np.ndarray,
    weights: np.ndarray,
    anchors: np.ndarray,
    n_typical: int,
    n_varied: int,
) -> tuple[list[int], list[int]]:
    """``(typical, varied)`` row indices into ``feats``.

    Typical: the ``n_typical`` rows nearest the ``weights``-weighted mean. Each row stands for
    a group of frames, so the mean is over frames, not groups. Varied: ``n_varied`` picks, each
    the row farthest from every anchor and every row picked so far. With nothing yet to be far
    from, the first is the row farthest from the mean.
    """
    n = len(feats)
    mean = (feats * weights[:, None]).sum(0) / weights.sum()
    to_mean = ((feats - mean) ** 2).sum(1)
    typical = [int(i) for i in np.argsort(to_mean, kind="stable")[: min(n_typical, n)]]
    taken = set(typical)
    near = np.full(n, np.inf)
    for a in list(anchors) + [feats[i] for i in typical]:
        near = np.minimum(near, ((feats - a) ** 2).sum(1))
    varied = []
    for _ in range(min(n_varied, n - len(taken))):
        score = np.where(np.isfinite(near), near, to_mean)
        score[list(taken)] = -np.inf
        i = int(score.argmax())
        varied.append(i)
        taken.add(i)
        near = np.minimum(near, ((feats - feats[i]) ** 2).sum(1))
    return typical, varied


def installed_backbone() -> str | None:
    """The best DINOv3 ViT size installed, or None (the shape rows then stand in). A disk
    check only (``is_installed``), never the network."""
    from detail.core.installed import is_importable

    if not (is_importable("torch") and is_importable("transformers")):
        return None
    from detail.models.dinov3 import DINOv3ViT

    for variant in BACKBONES:
        if DINOv3ViT(variant=variant).is_installed():
            return variant
    return None


def subject_crop(rgb: np.ndarray, bbox, pad: float = CROP_PAD) -> np.ndarray:
    """A square crop around the subject's box, ``pad`` of the box's longer side on each side,
    filled with mid-grey where it runs off the frame."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h
    side = max((x1 - x0) * w, (y1 - y0) * h) * (1 + 2 * pad)
    side = int(round(max(side, 16)))
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    canvas = np.full((side, side, 3), 114, dtype=np.uint8)
    sx0, sy0 = max(left, 0), max(top, 0)
    sx1, sy1 = min(left + side, w), min(top + side, h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - top : sy1 - top, sx0 - left : sx1 - left] = rgb[sy0:sy1, sx0:sx1]
    return canvas


def embed_subjects(
    images: list[np.ndarray],
    bboxes: list,
    variant: str,
    device: str | None,
    progress=None,
) -> np.ndarray:
    """``(N, D)`` unit vectors: each subject's DINOv3 ViT embedding of its padded square
    crop (``subject_crop``), the backbone the patch-matching method is built on."""
    import torch

    from detail.core.utils import default_compute, resolve_device
    from detail.models.dinov3 import DINOv3ViT

    model = DINOv3ViT(variant=variant)
    if device is None:
        backends = model.supported_backends
        device = resolve_device(default_compute(backends), backends)
    model.load(device)
    out = []
    for start in range(0, len(images), EMBED_BATCH):
        stop = start + EMBED_BATCH
        crops = [
            subject_crop(rgb, bbox)
            for rgb, bbox in zip(images[start:stop], bboxes[start:stop])
        ]
        vectors = np.asarray(model.embed_batch(crops), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out.extend(vectors / norms)
        if progress is not None:
            done = min(stop, len(images))
            progress("embedding", done / len(images), f"{done}/{len(images)} frames")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.stack(out)


def varied_frames(
    experiment_dir: Path | str,
    example_set: ExampleSet,
    sources: dict[str, str],
    n_frames: int,
    *,
    n_typical: int,
    add_image,
    from_recordings: int | None = DEFAULT_FROM_RECORDINGS,
    track_ids: list[int] | None = None,
    seed: int = 0,
    device: str | None = None,
    progress=None,
) -> list[ExampleFrame]:
    """The ``varied`` strategy (module docstring): ``n_typical`` typical frames, then
    ``n_frames`` farthest-first picks, copied in. Each row's ``sampled_by`` says which, and the
    features that chose it (``dinov3_<size>`` or ``shape``).

    With more than ``from_recordings`` sources, only that many are read, stratified across the
    whole set (:func:`choose_recordings`) and named in the progress line, so a large experiment
    does not wait on every recording's masks. ``None`` reads every one.

    Raises:
        ValueError: No sources, ``n_frames`` below one, a ``from_recordings`` below one, or no
            frames left to draw.
        FileNotFoundError: A chosen recording's video is not on this machine.
    """
    experiment_dir = Path(experiment_dir)
    _check_request(sources, n_frames)

    def report(stage, fraction, detail):
        if progress is not None:
            progress(stage, fraction, detail)

    if from_recordings is not None:
        chosen = choose_recordings(sorted(sources), from_recordings, seed)
        if len(chosen) < len(sources):
            report(
                "choosing recordings",
                1.0,
                f"sampling from {len(chosen)} of {len(sources)} recordings",
            )
            sources = {recording_id: sources[recording_id] for recording_id in chosen}

    existing = {f.key() for f in example_set.frames}
    pool: list[dict] = []
    # The bar moves inside a recording, not only between them: a recording is seconds of
    # masks, and a bar that jumps once per recording reads as hung (DESIGN_PLAN §6).
    throttled = ThrottledProgress(lambda event: report(*event))
    for i, (recording_id, tracking_run_id) in enumerate(sources.items()):
        throttled(
            ("reading shapes", i / len(sources), f"{recording_id} · reading"),
            force=True,
        )
        recording_dir = experiment_store.recording_path(experiment_dir, recording_id)
        rows, mask_size = _tracked_rows(recording_dir, tracking_run_id, track_ids)
        _require_videos(recording_dir, {row["part_id"] for row in rows})
        for row in rows:
            row["recording_id"] = recording_id
        rows = _thin(rows, SHAPE_POOL, existing)
        _prefetch_masks(rows)
        for j, row in enumerate(rows):
            row["shape"] = shape_row(_subject_mask(row, mask_size), row["bbox"])
            del row["det"], row["rows"]
            throttled(
                (
                    "reading shapes",
                    (i + (j + 1) / len(rows)) / len(sources),
                    f"{recording_id} · {j + 1}/{len(rows)} frames",
                )
            )
        if rows:
            median = float(np.median([np.exp(r["shape"][0]) for r in rows]))
            for row in rows:
                row["fit"] = _fit_for_pick(row, median)
        pool.extend(rows)
    report("reading shapes", 1.0, f"{len(pool)} frames")

    fresh = [r for r in pool if _key(r) not in existing]
    if not fresh:
        raise _nothing_left(track_ids)
    candidates = [r for r in fresh if r["fit"]] or fresh
    shapes = np.array([r["shape"] for r in pool])
    centre, scale = shapes.mean(0), shapes.std(0)
    scale[scale == 0] = 1.0

    def standardized(rows: list[dict]) -> np.ndarray:
        return (np.array([r["shape"] for r in rows]) - centre) / scale

    k = int(np.clip(GROUPS_PER_FRAME * (n_frames + n_typical), MIN_GROUPS, MAX_GROUPS))
    report("grouping", 0.0, f"{len(candidates)} candidates")
    reps = representatives(standardized(candidates), k, np.random.default_rng(seed))
    report("grouping", 1.0, f"{len(reps)} groups")
    rep_rows = [candidates[i] for i, _ in reps]
    weights = np.array([size for _, size in reps], dtype=np.float64)

    variant = installed_backbone()
    if variant is None:
        feats = standardized(rep_rows)
        anchor_rows = [r for r in pool if _key(r) in existing]
        anchors = (
            standardized(anchor_rows) if anchor_rows else np.empty((0, feats.shape[1]))
        )
        features = "shape"
    else:
        images, kept = [], []
        for i, row in enumerate(rep_rows):
            report("decoding candidates", i / len(rep_rows), f"{i}/{len(rep_rows)}")
            rgb = _decode(experiment_dir, row["recording_id"], row)
            if rgb is not None:
                images.append(rgb)
                kept.append(i)
        report("decoding candidates", 1.0, f"{len(kept)}/{len(rep_rows)}")
        if not kept:
            raise _nothing_left(track_ids)
        rep_rows = [rep_rows[i] for i in kept]
        weights = weights[kept]
        anchor_images = example_sets.load_images(experiment_dir, example_set)
        embedded = embed_subjects(
            images + anchor_images,
            [r["bbox"] for r in rep_rows] + [f.bbox for f in example_set.frames],
            variant,
            device,
            progress=report,
        )
        feats, anchors = embedded[: len(images)], embedded[len(images) :]
        features = f"dinov3_{variant}"

    typical, varied = choose(feats, weights, anchors, n_typical, n_frames)
    picks = [(i, f"typical:{features}") for i in typical]
    picks += [(i, f"varied:{features}") for i in varied]
    out = []
    for j, (i, label) in enumerate(picks):
        report("copying frames", j / len(picks), f"{j}/{len(picks)} frames")
        row = rep_rows[i]
        copied = _copy_frame(
            experiment_dir,
            row["recording_id"],
            sources[row["recording_id"]],
            row,
            add_image,
            sampled_by=label,
        )
        if copied is not None:
            out.append(copied[0])
    report("copying frames", 1.0, f"{len(out)}/{len(picks)} frames")
    return out
