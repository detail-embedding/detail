"""Recording directory: recording.yaml read/write and timestamp maps (Section 3).

`recording.yaml` is append-only — each pipeline stage adds its section as it
completes. This module handles identity + the ``video`` section and the per-part
``frame_idx -> t_ns`` timestamp maps; later sections (streams, spatial,
detection_runs, ...) are added by their stages. All paths in recording.yaml are
relative to the recording directory.
"""

import contextlib
import copy
import csv
import os
import stat
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from detail.core import atomic
from detail.core.errors import UserInputError
from detail.storage import experiment
from detail.storage import read_cache
from detail.storage import video
from detail.storage.user_yaml import read_user_yaml

SCHEMA_VERSION = 1
RECORDING_FILENAME = "recording.yaml"

# recording.yaml is read a lot on the UI thread — every Recording property re-reads it,
# so a batch over N recordings parses it hundreds of times. Memoize the parsed dict
# keyed by (resolved path, mtime_ns): a stat is ~1000x cheaper than a read+parse, and
# the mtime key keeps it fresh across background-worker writes (atomic os.replace
# always bumps the mtime). read_recording returns a deep copy so callers can freely
# mutate their result without corrupting the cache.
_RECORDING_CACHE: dict[str, tuple[int, dict]] = {}
_RECORDING_CACHE_LOCK = threading.Lock()


def clear_recording_cache() -> None:
    """Drop the whole recording.yaml cache (test isolation / forced reload)."""
    with _RECORDING_CACHE_LOCK:
        _RECORDING_CACHE.clear()


def _cache_key(recording_dir: Path | str) -> str:
    """Normalized cache key for a recording dir's ``recording.yaml``. Read and write key
    through this same helper so a relative vs absolute spelling of the same directory
    can't split the entry (mtime keeps correctness regardless; this bounds duplicate
    growth). ``abspath`` normalizes without the symlink I/O of ``resolve``."""
    return os.path.abspath(Path(recording_dir) / RECORDING_FILENAME)


RECORDING_SUBDIRS = (
    "video/timestamps/generated",
    "detections",
    "tracking",
    "embeddings",
)


def create_recording(
    experiment_dir: Path | str,
    recording_id: str,
    name: str | None = None,
    created_at: str | None = None,
) -> Path:
    """Create a recording directory under the experiment and return its path.

    Writes a minimal recording.yaml (identity only); the ``video`` section is added
    when video is imported.

    Raises:
        ValueError: If ``recording_id`` is empty.
        FileExistsError: If a recording already exists at the target path.
    """
    if not recording_id:
        raise ValueError("recording_id must be non-empty")
    sdir = experiment.recording_path(experiment_dir, recording_id)
    if (sdir / RECORDING_FILENAME).exists():
        raise FileExistsError(f"a recording already exists at {sdir}")
    for sub in RECORDING_SUBDIRS:
        (sdir / sub).mkdir(parents=True, exist_ok=True)
    if name is None:
        name = recording_id
    if created_at is None:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_recording(sdir, {"id": recording_id, "name": name, "created_at": created_at})
    return sdir


def read_recording(recording_dir: Path | str) -> dict:
    """Read and validate a recording's ``recording.yaml``.

    **One round trip to the presence check.** The existence probe and the cache's mtime are
    the same ``stat``: asking ``is_file()`` first and then ``stat()`` pays twice to learn one
    fact, and this is read once per recording by every listing there is. Locally that is
    ~28 µs and invisible; on the network share these workspaces live on it was a third of the
    project rail's whole cost (#316).

    That leaves one round trip **per call**, and every ``Recording`` property is its own call —
    so a caller wanting several fields pays per field. Inside ``read_cache.pinned()`` a hit
    skips the stat entirely; see that function for the trade and its bounds.

    Raises:
        FileNotFoundError: If ``recording.yaml`` is absent.
        ValueError: If it is malformed, has the wrong ``schema_version``, or has
            no ``id``.
    """
    path = Path(recording_dir) / RECORDING_FILENAME
    if read_cache.is_pinned():
        with _RECORDING_CACHE_LOCK:
            cached = _RECORDING_CACHE.get(_cache_key(recording_dir))
        if cached is not None:
            return copy.deepcopy(cached[1])
    try:
        info = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        raise FileNotFoundError(f"no {RECORDING_FILENAME} in {recording_dir}") from None
    # `is_file()` refused a directory (or a socket) of that name; keep that, off the stat
    # already in hand, or the failure moves to `read_text` as an IsADirectoryError.
    if not stat.S_ISREG(info.st_mode):
        raise FileNotFoundError(f"no {RECORDING_FILENAME} in {recording_dir}")
    key = _cache_key(recording_dir)
    mtime = info.st_mtime_ns
    with _RECORDING_CACHE_LOCK:
        cached = _RECORDING_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return copy.deepcopy(cached[1])
    # Per key: threads that miss one record together wait for a single parse rather than
    # each parsing it (`read_cache.computing`); a thread wanting another record is not held.
    with read_cache.computing(("recording", key)):
        with _RECORDING_CACHE_LOCK:
            cached = _RECORDING_CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return copy.deepcopy(cached[1])
        recording = _parse_recording(path)
        with _RECORDING_CACHE_LOCK:
            _RECORDING_CACHE[key] = (mtime, recording)
    return copy.deepcopy(recording)


def _parse_recording(path: Path) -> dict:
    """``recording.yaml`` at ``path``, read and validated: the parse the cache guards."""
    recording = read_user_yaml(path)
    if not isinstance(recording, dict):
        raise UserInputError(f"{path} is not a mapping")
    if recording.get("schema_version") != SCHEMA_VERSION:
        raise UserInputError(
            f"{path} schema_version is {recording.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    if not recording.get("id"):
        raise UserInputError(f"{path} has no 'id'")
    return recording


def write_recording(recording_dir: Path | str, recording: dict) -> None:
    """Write ``recording.yaml`` with ``schema_version`` forced to the first key (Tenet 7).

    Atomic (tmp + ``os.replace``) because the import worker thread rewrites the
    recording while the GUI thread reads it (e.g. ``Experiment.recordings`` rebuilding
    the tree); a reader never sees a torn, half-truncated file. Routed through
    ``core.atomic`` so the replace rides out the transient Windows lock a
    concurrent reader provokes — ``os.replace`` there is not atomic against an
    open handle the way POSIX ``rename`` is."""
    if not recording.get("id"):
        raise ValueError("recording must have a non-empty 'id'")
    ordered = {"schema_version": SCHEMA_VERSION}
    for key, value in recording.items():
        if key != "schema_version":
            ordered[key] = value
    path = Path(recording_dir) / RECORDING_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(ordered, f, sort_keys=False)
    atomic.replace(tmp, path)
    # Drop the cache entry so a read right after this write (before the mtime clock
    # necessarily advances) re-reads the just-written file rather than a stale parse.
    with _RECORDING_CACHE_LOCK:
        _RECORDING_CACHE.pop(_cache_key(recording_dir), None)


@contextlib.contextmanager
def recording_lock(recording_dir: Path | str):
    """Serialize a recording.yaml read-modify-write across processes (an exclusive lock).

    Concurrent runs on one recording — several embedding runs pinned to different GPUs, say —
    each allocate a run id and append/update their record in recording.yaml. That read-modify-write
    has no natural atomicity, so without serialization two runs collide on the id and lose-update
    each other's records. This advisory lock (a sibling ``recording.yaml.lock``) makes the
    allocate+append and status-update critical sections mutually exclusive. Readers do not take it:
    ``write_recording`` is atomic, so a read always sees a whole file, and ``read_recording`` re-parses
    on any mtime change, so a caller inside the lock sees another process's committed write.
    """
    with atomic.file_lock(Path(recording_dir) / (RECORDING_FILENAME + ".lock")):
        yield


def set_name(recording_dir: Path | str, name: str) -> None:
    """Edit a recording's display name in recording.yaml (id/dir unchanged).

    Raises:
        ValueError: If ``name`` is empty.
    """
    if not name:
        raise ValueError("recording name must be non-empty")
    recording = read_recording(recording_dir)
    recording["name"] = name
    write_recording(recording_dir, recording)


def set_import_status(recording_dir: Path | str, status: str) -> None:
    """Set the recording's ``import_status`` (``ready`` | ``importing`` | ``failed``).

    Persisted so a crashed/interrupted import is visible + deletable, not stuck.
    Absent on older recordings → ``ready`` (read via ``.get``)."""
    recording = read_recording(recording_dir)
    recording["import_status"] = status
    write_recording(recording_dir, recording)


def remove_run_entry(recording_dir: Path | str, runs_key: str, run_id: str) -> bool:
    """Drop the ``run_id`` entry from ``recording.yaml[runs_key]``. Returns whether
    an entry was removed."""
    recording = read_recording(recording_dir)
    runs = recording.get(runs_key, [])
    kept = [r for r in runs if r.get("id") != run_id]
    if len(kept) == len(runs):
        return False
    recording[runs_key] = kept
    write_recording(recording_dir, recording)
    return True


def set_run_status(
    recording_dir: Path | str, runs_key: str, run_id: str, status: str
) -> bool:
    """Set the ``run_id`` entry's ``status`` in ``recording.yaml[runs_key]``. Returns
    whether an entry was updated."""
    recording = read_recording(recording_dir)
    updated = False
    for entry in recording.get(runs_key, []):
        if entry.get("id") == run_id:
            entry["status"] = status
            updated = True
    if updated:
        write_recording(recording_dir, recording)
    return updated


def reorder_video_parts(recording_dir: Path | str, new_order: list[int]) -> None:
    """Reorder a recording's video parts and re-offset global timestamps (Tenet 8).

    ``new_order`` is the current part indices in the desired sequence (a
    permutation of ``range(n)``). Each part keeps its file and timestamp CSV, but
    the list is reordered + re-indexed and every part's active timestamp map is
    rewritten so global ``t_ns`` reflects the new ordering (each part offset by
    the cumulative duration of the parts now before it). Offsets accumulate in
    integer ns, exactly as ``video_end_ns`` computes the next part's offset at import,
    so reordering into the order a recording already has rewrites nothing.

    Raises:
        ValueError: If ``new_order`` is not a permutation of the current parts.
    """
    recording_dir = Path(recording_dir)
    data = read_recording(recording_dir)
    files = (data.get("video") or {}).get("files", [])
    n = len(files)
    if sorted(new_order) != list(range(n)):
        raise ValueError(
            f"new_order must be a permutation of 0..{n - 1}, got {new_order}"
        )
    fps = float(data["video"]["fps_actual"])

    # Per (old) part: local timestamps (relative to its own start) + duration.
    locals_, durations_ns = [], []
    for entry in files:
        active = entry["timestamps"][entry["timestamps"]["active"]]
        values = TimestampMap.from_csv(recording_dir / active).values
        locals_.append(values - values[0])
        # Measured span, not len/fps_actual. This rewrites every part's timestamps
        # below, so using the advertised rate here would *degrade* a correct recording.
        durations_ns.append(video.part_duration_ns(values, fps))

    new_files = []
    offset_ns = 0
    for new_idx, old_idx in enumerate(new_order):
        entry = dict(files[old_idx])
        entry["index"] = new_idx
        active = entry["timestamps"][entry["timestamps"]["active"]]
        global_t_ns = locals_[old_idx] + offset_ns
        write_timestamps_csv(recording_dir / active, global_t_ns.astype(np.int64))
        offset_ns += durations_ns[old_idx]
        new_files.append(entry)

    data["video"]["files"] = new_files
    write_recording(recording_dir, data)


class TimestampMap:
    """Per-part ``frame_idx -> t_ns`` map (Section 3, Tenet 8).

    Backed by a dense int64 array indexed by frame_idx (the maps are contiguous
    from frame 0).
    """

    def __init__(self, t_ns: np.ndarray) -> None:
        self._t_ns = np.asarray(t_ns, dtype=np.int64)

    @classmethod
    def from_csv(cls, path: Path | str) -> "TimestampMap":
        """Load a ``frame_idx,t_ns`` CSV.

        Raises:
            ValueError: If the frame_idx column is not contiguous from 0.
        """
        frame_idx: list[int] = []
        t_ns: list[int] = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                frame_idx.append(int(row[0]))
                t_ns.append(int(row[1]))
        frames = np.asarray(frame_idx, dtype=np.int64)
        if not np.array_equal(frames, np.arange(len(frames), dtype=np.int64)):
            raise ValueError(f"timestamp map {path} frame_idx is not contiguous from 0")
        return cls(np.asarray(t_ns, dtype=np.int64))

    def t_ns(self, frame_idx: int) -> int:
        if not 0 <= frame_idx < len(self._t_ns):
            raise IndexError(
                f"frame_idx {frame_idx} out of range [0, {len(self._t_ns)})"
            )
        return int(self._t_ns[frame_idx])

    @property
    def values(self) -> np.ndarray:
        """The full int64 t_ns array, indexed by frame_idx."""
        return self._t_ns

    def __len__(self) -> int:
        return len(self._t_ns)


def write_timestamps_csv(path: Path | str, t_ns: np.ndarray) -> None:
    """Write a ``frame_idx,t_ns`` CSV; frame_idx is the row position (0-based).

    Atomic (tmp + ``core.atomic.replace``): ``reorder_video_parts`` rewrites a map the
    GUI may be reading through ``load_active_timestamps_cached`` at that moment.
    """
    path = Path(path)
    values = np.asarray(t_ns, dtype=np.int64)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "t_ns"])
        for i, t in enumerate(values):
            writer.writerow([i, int(t)])
    atomic.replace(tmp, path)


def video_end_ns(recording_dir: Path | str, recording: dict) -> int:
    """Where the next video part starts: the summed measured durations of every part.

    Integer ns from each part's active map (``video.part_duration_ns``), never the
    float ``total_duration_s``, so a part appended later lands on exactly the offset
    ``reorder_video_parts`` would give it.
    """
    video_meta = recording.get("video") or {}
    fps = float(video_meta.get("fps_actual") or 0.0)
    return sum(
        video.part_duration_ns(
            load_active_timestamps(recording_dir, recording, i).values, fps
        )
        for i in range(len(video_meta.get("files", [])))
    )


def active_timestamp_csv(
    recording_dir: Path | str, recording: dict, part_index: int
) -> Path:
    """Resolve the active (generated or user) timestamp CSV path for a video part."""
    part = recording["video"]["files"][part_index]
    timestamps = part["timestamps"]
    active = timestamps["active"]  # "generated" | "user"
    relative = timestamps[active]
    if relative is None:
        raise ValueError(f"part {part_index}: active timestamp map {active!r} is null")
    return Path(recording_dir) / relative


def load_active_timestamps(
    recording_dir: Path | str, recording: dict, part_index: int
) -> TimestampMap:
    """Load the active timestamp map for a video part."""
    return TimestampMap.from_csv(
        active_timestamp_csv(recording_dir, recording, part_index)
    )


def raise_on_legacy_downsampling(record: dict | None) -> None:
    """Refuse a run produced by the removed §4 downsampling stage.

    Such a run's ``frame_idx`` indexes the *downsampled* video, so reading its rows
    against the recording's video resolves the wrong frames — ds frame 100 is source
    frame 200 at ``sub_sample: 2``. That is wrong pictures with no error, so the run is
    refused rather than misread. Nothing can recover the mapping once the ds video is
    gone; the old runs are removed and the work is re-run with ``every_n_frames``.

    Raises:
        ValueError: If ``record`` carries a non-null ``downsampling_run_id``.
    """
    if record and record.get("downsampling_run_id"):
        raise ValueError(
            f"run was produced on downsampled video "
            f"({record['downsampling_run_id']!r}), a stage that no longer exists — its "
            "frame numbering cannot be read against this recording's video. Re-run "
            "detection with a every_n_frames, and remove the old runs."
        )


def video_files(recording: dict) -> list[dict]:
    """The recording's per-part video files — the one source every stage reads.

    There is exactly one frame space per recording: a stage that processes fewer frames
    does so with a ``every_n_frames`` over *these* files, so ``frame_idx`` always indexes
    the source video (§4).
    """
    return recording["video"]["files"]


def part_stem(part: dict) -> str:
    """A part's stable identifier — used for zarr/progress-file names and run records.

    Reference-in-place parts have no recording-relative ``file``, so every part written now
    carries an explicit ``id``; a legacy copied part falls back to its ``file`` stem.
    """
    stem = part.get("id")
    if stem:
        return stem
    return Path(part["file"]).stem


def resolve_part_video_path(recording_dir: Path | str, part: dict) -> Path:
    """The absolute path to a part's video file.

    A **copied** part stores a recording-relative ``file`` (joined onto the recording dir); a
    **reference-in-place** part stores a ``source`` ``{root, rel}`` that resolves through the
    machine-local video-root registry. Every video reader (the run workers, previews, the
    presence gate) goes through here so reference mode is transparent to them. Does not check the
    file exists — that is the caller's presence check.
    """
    source = part.get("source")
    if source is not None:
        from detail.storage import video_roots

        return video_roots.resolve_video_ref(source)
    return Path(recording_dir) / part["file"]


def load_active_timestamps_cached(
    recording_dir: Path | str, recording: dict, part_index: int
) -> TimestampMap:
    """:func:`load_active_timestamps`, memoized on the CSV's mtime.

    For a loop that resolves one frame at a time — the prompt test, the render
    preview, embedding's mean-colour pass — the map is both the seek index handed to
    ``read_frame_at`` and the ``t_ns`` source, so without this it re-parses a
    100k-row CSV per frame. A written map is immutable; the mtime key picks up a
    re-import. Shared **by reference** (``read_cache``) — treat the result as
    read-only.
    """
    path = active_timestamp_csv(recording_dir, recording, part_index)
    return read_cache.cached_read(
        ("active_timestamps", str(path)), path, lambda: TimestampMap.from_csv(path)
    )


def session_span_ns(recording_dir: Path | str) -> tuple[int, int] | None:
    """``(first_ns, last_ns)``: the recording's first and last frame on its active timestamp map.

    The whole recording, first part to last: ``t_ns`` runs from the start of the first part
    (Tenet 8), so the first part's first time and the last part's last time bound it. ``None``
    when the recording has no video manifest, no map, or an empty one. Read through
    :func:`load_active_timestamps_cached`.
    """
    try:
        data = read_recording(recording_dir)
    except FileNotFoundError:
        return None
    files = (data.get("video") or {}).get("files") or []
    if not files:
        return None
    try:
        first = load_active_timestamps_cached(recording_dir, data, 0).values
        last = load_active_timestamps_cached(recording_dir, data, len(files) - 1).values
    except (FileNotFoundError, ValueError):
        return None
    if not len(first) or not len(last):
        return None
    return int(first[0]), int(last[-1])


def part_start_frames(
    recording_dir: Path | str, recording: dict | None = None
) -> dict[str, int] | None:
    """``{part stem: the recording frame its local frame 0 is}`` — the one part-offset
    mechanism.

    Every per-part store (detections, assignments, embeddings, kinematics, keypoints) counts
    ``frame_idx`` from 0 inside its own part (§5 ``FrameResult.frame_idx``), while a frame a
    user scrubs to, an importer reads or a clip cutter seeks is counted across the recording.
    Add ``part_start_frames(...)[stem]`` to a part-local frame to get the recording one.

    **Keyed by part stem, never by position**: a reader that skips an empty part and counts
    the rest positionally places every later part one part too early. The counts come from
    each part's timestamp map, the authoritative frame count (Tenet 8); a stage's
    ``every_n_frames`` thins the rows it writes but never renumbers them, so there is one set
    of offsets for every stage. ``None`` if the recording has no video manifest (synthetic
    direct-call tests); a caller decides what that means for it.
    """
    if recording is None:
        try:
            recording = read_recording(recording_dir)
        except FileNotFoundError:
            return None
    files = (recording.get("video") or {}).get("files")
    if not files:
        return None
    offsets: dict[str, int] = {}
    total = 0
    for part_index, file in enumerate(files):
        offsets[part_stem(file)] = total
        if part_index < len(files) - 1:  # the last part's length offsets nothing
            total += len(
                load_active_timestamps_cached(
                    recording_dir, recording, part_index
                ).values
            )
    return offsets


def part_frame_offsets(recording_dir: Path | str) -> list[int] | None:
    """`part_start_frames` in part order — for a caller that walks every part positionally
    (tracking §13, the per-recording location export), so position and stem agree.
    ``None`` if the recording has no video manifest."""
    offsets = part_start_frames(recording_dir)
    return None if offsets is None else list(offsets.values())
