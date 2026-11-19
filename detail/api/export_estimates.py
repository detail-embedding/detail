"""How big each export target's output will be and how long it will take (issue #710).

One estimator per target, registered on the target (`ExportTarget.estimate`) and asked through
`exporters.estimate`. They exist because an export started from the top bar wrote far more
video than the drive had free, and the dialog had said nothing about it.

**Three kinds of figure, and each says which it is** (`Estimate.basis`):

- *counted*: read from what is on disk (a file's size, a recording's frame count, a zarr's row
  count). The ones that are exact are exact: a video copy is the sum of the files' sizes.
- *modelled*: a row count times a measured cost per row. The constants are below, each with the
  measurement it came from. They are right to within a small factor on the machine they were
  measured on and a different factor elsewhere, which is why the surfaces print "about".
- *timed*: a few dozen frames of the real video are decoded, scaled and encoded on *this* machine
  and the result is multiplied up. Video is where an estimate is worth having (it is where a
  mistake costs a drive) and where a constant would be least honest: encoder, resolution and
  content move the answer by 10x.

An estimator that cannot know (a video that is not on this machine, a comparison whose figure
count only exists once it is computed) returns ``None`` for what it cannot know and says why. A
guessed number would read as a measurement.

**Estimators never write an export.** The one thing they write is the timed estimators' scratch
clips in the system temp folder, and the disk-speed probe's 16 MiB file, deleted at once, beside
the destination (the speed that matters is the destination's, and on a network share it is not the
system drive's).
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from detail.api.exporters import Estimate, ExportScope, ExportSubject, format_size
from detail.storage import dirscan

# --------------------------------------------------------------------------- #
# Measured constants: timing the real `write_export` on one real recording, with
# `scope.recording_ids` narrowed to it.
# --------------------------------------------------------------------------- #
#: masks: the JSON around each mask (ids, box, area, score, and the
#: frame's `images` entry) is ~640 bytes on top of the run-length string, which is sampled.
MASKS_ROWS_PER_S = 21_000
MASKS_JSON_BYTES_PER_ROW = 640
MASKS_FALLBACK_RLE_BYTES = (
    320  # the mean string length measured; used when nothing is sampled
)
#: tracks: slow because each row is placed against the recording's ROIs and points; a
#: recording with none is faster.
TRACKS_ROWS_PER_S = 600
#: 352 measured on a calibrated 1080p recording once the body columns were added (it was 215
#: before them); an uncalibrated recording's rows are shorter, its mm columns being blank.
TRACKS_BYTES_PER_ROW = 350
#: vectors: rows of 1024 float32. The index CSV is ~51 bytes a row.
VECTORS_ROWS_PER_S = 100_000
VECTORS_INDEX_BYTES_PER_ROW = 51
#: timestamps.
TIMESTAMPS_BYTES_PER_ROW = 21
TIMESTAMPS_ROWS_PER_S = 500_000
#: instances / ethogram: the figures dominate, and a figure's size and drawing time depend on
#: what is in it more than on the row count, so these are the loosest constants here,
#: fitted over several analysis runs of one recording.
INSTANCES_BASE_BYTES = 45_000
INSTANCES_BYTES_PER_ROW = 1_050
INSTANCES_BASE_S = 0.4
INSTANCES_S_PER_ROW = 0.003
ETHOGRAM_BASE_BYTES = 20_000
ETHOGRAM_BYTES_PER_ROW = 190
ETHOGRAM_BASE_S = 0.12
ETHOGRAM_S_PER_ROW = 0.0004
#: frame_labels: a time per frame plus a time per behavior column, and a row is ~25 bytes
#: plus 2 per column.
FRAME_LABELS_BASE_S_PER_FRAME = 4.5e-6
FRAME_LABELS_S_PER_COLUMN = 1.45e-7
FRAME_LABELS_BASE_BYTES_PER_FRAME = 25
#: Drawing the points (and, opt-in, the tracking overlay) on a frame, per frame. Measured
#: on another keypoints video engine; this tree's `keypoints_video` is a different
#: implementation, so the figure is an
#: approximation of its cost and the estimate's basis says so.
DRAW_POINTS_S_PER_FRAME = 0.0003
DRAW_OVERLAY_S_PER_FRAME = 0.0009
DEFAULT_DISK_BYTES_PER_S = 100e6
_PROBE_BYTES = 16 * 1024 * 1024
_CACHE_TTL_S = 120.0
_RATE_CACHE: dict = {}


def _small(basis: str, bytes_: int = 4_000, seconds: float = 0.2) -> Estimate:
    return Estimate(bytes_, seconds, basis)


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def disk_rate(destination: Path | str | None) -> tuple[float, str]:
    """``(bytes per second, basis)`` writing to ``destination``'s drive, measured.

    Writes and flushes 16 MiB beside ``destination`` (its nearest existing folder) and deletes it.
    Remembered for two minutes, because the dialog asks on every change. An unwritable or
    unknown destination gets a stated default rather than an error: the question is cosmetic and
    must not be why an export cannot be offered.
    """
    default = (
        DEFAULT_DISK_BYTES_PER_S,
        f"assumes {DEFAULT_DISK_BYTES_PER_S / 1e6:.0f} MB/s (the destination's speed could not be "
        "measured)",
    )
    if destination is None:
        return default
    probe_dir = None
    for candidate in (
        Path(destination).expanduser(),
        *Path(destination).expanduser().parents,
    ):
        if candidate.is_dir():
            probe_dir = candidate
            break
    if probe_dir is None:
        return default
    key = str(probe_dir)
    cached = _RATE_CACHE.get(("disk", key))
    if cached and time.monotonic() - cached[1] < _CACHE_TTL_S:
        return cached[0]
    path = probe_dir / f".detail-speed-probe-{os.getpid()}.tmp"
    try:
        block = os.urandom(1024 * 1024)
        started = time.perf_counter()
        with open(path, "wb") as handle:
            for _ in range(_PROBE_BYTES // len(block)):
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        elapsed = max(time.perf_counter() - started, 1e-3)
    except OSError:
        return default
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    rate = min(max(_PROBE_BYTES / elapsed, 5e6), 3e9)
    answer = (
        rate,
        f"{rate / 1e6:.0f} MB/s, timed writing 16 MiB to the destination's drive",
    )
    _RATE_CACHE[("disk", key)] = (answer, time.monotonic())
    return answer


@dataclass(frozen=True)
class VideoRate:
    """What decoding, scaling and encoding a recording's video costs on this machine."""

    decode_s: float  # per frame, decode + scale to the output size
    encode_s: float  # per frame, at the output size
    bytes_per_frame: float
    setup_s: float  # once per output file: opening the encoder
    width: int
    height: int


def _scaled(frame, size: int | None):
    """``frame`` with its longest side at ``size`` (even sides, which H.264 needs)."""
    import cv2

    h, w = frame.shape[:2]
    if size:
        scale = float(size) / max(h, w)
        w, h = max(2, round(w * scale)), max(2, round(h * scale))
    w, h = w - w % 2, h - h % 2
    if (h, w) != frame.shape[:2]:
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return frame


def video_rate(
    video_path: Path | str,
    *,
    size: int | None = None,
    fps: float = 30.0,
    frames: int = 40,
) -> VideoRate:
    """Time ``frames`` frames from the middle of ``video_path`` through the real writer.

    The middle, not the start: a recording's first seconds are the animal being put in the box,
    and a size taken from a still scene is half the real one. Two encodes of ``frames/2`` and
    ``frames`` frames are timed and the *difference* is the cost per frame, so the encoder's start-up
    and the first (key)frame, which are paid once per file and not per frame, are not multiplied
    by 87,000.

    ``cap.set(CAP_PROP_POS_FRAMES)`` is used once, to reach the middle: the sequential-read rule
    permits a seek on a preview-class path where approximate position is acceptable, and this is
    one.
    """
    import cv2

    from detail.storage.video import open_video_writer

    path = Path(video_path)
    stat = path.stat()
    key = ("video", str(path), stat.st_mtime_ns, size)
    cached = _RATE_CACHE.get(key)
    if cached and time.monotonic() - cached[1] < 600.0:
        return cached[0]
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 4 * frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        decoded = []
        started = time.perf_counter()
        for _ in range(frames):
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(_scaled(frame, size))
        decode_elapsed = time.perf_counter() - started
    finally:
        cap.release()
    if len(decoded) < 4:
        raise ValueError(f"could not read frames from {path.name} to time them")
    height, width = decoded[0].shape[:2]

    def encode(count: int) -> tuple[float, float, int]:
        with tempfile.TemporaryDirectory(prefix="detail-estimate-") as scratch:
            out = Path(scratch) / "sample.mp4"
            opened = time.perf_counter()
            writer, _codec = open_video_writer(out, fps, width, height)
            setup = time.perf_counter() - opened
            started = time.perf_counter()
            for frame in decoded[:count]:
                writer.write(frame)
            writer.release()
            return time.perf_counter() - started, setup, out.stat().st_size

    half = max(len(decoded) // 2, 2)
    t_half, setup_a, b_half = encode(half)
    t_full, setup_b, b_full = encode(len(decoded))
    extra = len(decoded) - half
    marginal_s = (t_full - t_half) / extra
    marginal_bytes = (b_full - b_half) / extra
    # A noisy pair can difference to nothing; the whole-sample average is then the better guess.
    encode_s = marginal_s if marginal_s > 0 else t_full / len(decoded)
    bytes_per_frame = marginal_bytes if marginal_bytes > 0 else b_full / len(decoded)
    rate = VideoRate(
        decode_s=decode_elapsed / len(decoded),
        encode_s=encode_s,
        bytes_per_frame=bytes_per_frame,
        setup_s=(setup_a + setup_b) / 2,
        width=width,
        height=height,
    )
    _RATE_CACHE[key] = (rate, time.monotonic())
    return rate


# --------------------------------------------------------------------------- #
# What a recording holds
# --------------------------------------------------------------------------- #
def _frames_of(experiment_dir: Path, recording_id: str) -> int:
    """A recording's frame count: the timestamp map's, never cv2's estimate (Tenet 8)."""
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    recording_dir = experiment_store.recording_path(experiment_dir, recording_id)
    data = recording_store.read_recording(recording_dir)
    return sum(
        len(recording_store.load_active_timestamps_cached(recording_dir, data, index))
        for index in range(len(recording_store.video_files(data)))
    )


def _first_video(experiment_dir: Path, recording_id: str) -> Path | None:
    """The first video part of a recording that is on this machine, or ``None``."""
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    recording_dir = experiment_store.recording_path(experiment_dir, recording_id)
    data = recording_store.read_recording(recording_dir)
    for part in recording_store.video_files(data):
        try:
            path = recording_store.resolve_part_video_path(recording_dir, part)
        except FileNotFoundError:  # a different video at the reference: not this one
            continue
        if path.is_file():
            return path
    return None


def _timed_video(
    experiment_dir: Path,
    recording_ids: list[str],
    *,
    size: int | None,
    extra_s_per_frame: float,
    label: str,
    frames: dict[str, int] | None = None,
) -> Estimate:
    """Video written for ``recording_ids``: frames x a per-frame cost timed on the first one.

    The sample comes from one recording (the first that has its video here) and is applied to all
    of them: recordings of one experiment share a camera and a resolution, and timing each would
    cost a sample per recording on a large export. A recording whose video is not on this
    machine makes the whole answer unknowable, because the export would fail on it.
    """
    if not recording_ids:
        return Estimate(0, 0.0, f"no {label} to write for this scope")
    counts = frames or {rid: _frames_of(experiment_dir, rid) for rid in recording_ids}
    sample_video = None
    for rid in recording_ids:
        video = _first_video(experiment_dir, rid)
        if video is None:
            return Estimate(
                None,
                None,
                f"the video of {rid} is not on this machine, so its {label} cannot be sized",
            )
        sample_video = sample_video or video
    rate = video_rate(sample_video, size=size)
    total_frames = sum(counts.values())
    per_frame_s = rate.decode_s + rate.encode_s + extra_s_per_frame
    seconds = total_frames * per_frame_s + len(recording_ids) * rate.setup_s
    nbytes = int(total_frames * rate.bytes_per_frame)
    return Estimate(
        nbytes,
        seconds,
        f"{total_frames:,} frames of {len(recording_ids)} recording(s) at {rate.width}x{rate.height}: "
        f"{rate.bytes_per_frame / 1024:.1f} KB and {per_frame_s * 1000:.2f} ms a frame, timed on "
        "40 frames from the middle of the first video",
    )


# --------------------------------------------------------------------------- #
# experiment
# --------------------------------------------------------------------------- #
def bundle(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    from detail.storage import export as export_store

    files = export_store._included_files(
        subject.experiment_dir,
        include_video=bool(options.get("include_video", False)),
        include_datasets=bool(options.get("include_datasets", False)),
    )
    total = 0
    for rel in files:
        try:
            total += (subject.experiment_dir / rel).stat().st_size
        except OSError:
            continue  # gone since the walk; it would not be packed either
    # A tar header and padding per member, plus what is fixed: the manifest the bundle carries
    # and the end-of-archive blocks, rounded up to tar's 10 KiB record.
    total += 1024 * len(files) + 20_000
    rate, how = disk_rate(options.get("destination"))
    if options.get("gzip"):
        # Compression, not the disk, is the limit then — many times slower than a plain tar, for a
        # file barely smaller, since zarr chunks are compressed already (`storage/export.py`).
        gzip_rate = export_store._PACK_GZ.bytes_per_s
        if gzip_rate < rate:
            rate, how = gzip_rate, f"gzip's ~{gzip_rate / 1e6:.0f} MB/s"
    return Estimate(
        total,
        total / rate + 0.0005 * len(files) + 1.0,
        f"the {len(files):,} files it packs (counted), written at {how}",
    )


def metadata(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    recordings = subject.experiment_dir / "recordings"
    count = (
        sum(1 for p in recordings.iterdir() if p.is_dir()) if recordings.is_dir() else 0
    )
    return _small(f"one row for each of {count:,} recordings", 300 * count + 400, 0.2)


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #
def streams(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    from detail.api import export_targets as et
    from detail.storage import streams as streams_store

    total = 0
    manifests = et._canonical_streams(subject)
    for manifest in manifests:
        source = (
            streams_store.stream_dir(et._recording_dir(subject), str(manifest["name"]))
            / streams_store.CANONICAL_DIRNAME
            / streams_store.CANONICAL_FILENAME
        )
        total += source.stat().st_size
    total += 100 * len(manifests)
    rate, how = disk_rate(options.get("destination"))
    return Estimate(
        total, total / rate + 0.05, f"{len(manifests)} stream file(s) copied (counted)"
    )


def timestamps(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    frames = _frames_of(subject.experiment_dir, subject.id)
    return Estimate(
        frames * TIMESTAMPS_BYTES_PER_ROW + 40,
        0.05 + frames / TIMESTAMPS_ROWS_PER_S,
        f"{frames:,} frames, one short row each",
    )


def video(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    from detail.api import export_targets as et
    from detail.storage import recording as recording_store

    recording_dir = et._recording_dir(subject)
    total = 0
    for part in et._copied_parts(subject):
        path = recording_store.resolve_part_video_path(recording_dir, part)
        if path.is_file():
            total += path.stat().st_size
    rate, how = disk_rate(options.get("destination"))
    return Estimate(
        total,
        total / rate + 0.1,
        f"the video file(s) copied as they are (counted), at {how}",
    )


def clip(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    if scope.frames is None:
        return Estimate(0, 0.0, "no frame range chosen yet: nothing to write")
    lo, hi = scope.frames
    frames = hi - lo + 1
    renders = options.get("renders") or ("overlay",)
    estimate = _timed_video(
        subject.experiment_dir,
        [subject.id],
        size=int(options.get("size", 640)),
        extra_s_per_frame=0.0,
        label="clip",
        frames={subject.id: frames},
    )
    if estimate.bytes is None:
        return estimate
    # Several flavours cost one decode and one encode each.
    n = max(len(renders), 1)
    return Estimate(
        estimate.bytes * n,
        estimate.seconds * (1 + 0.8 * (n - 1)),
        f"{estimate.basis}; {n} flavour(s)",
    )


def geometry(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    return _small(
        "one short CSV of the field of view, calibration and regions", 3_000, 0.1
    )


# --------------------------------------------------------------------------- #
# processing runs
# --------------------------------------------------------------------------- #
def _rows(paths) -> int:
    from detail.storage.zarr_io import PartZarrReader

    return sum(PartZarrReader(p).n_rows for p in paths)


def masks(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    from detail.api import export_targets as et
    from detail.storage.zarr_io import PartZarrReader

    parts = et._detection_parts(subject, et._only(scope, options))
    rows, lengths = 0, []
    for _rid, run_dir in parts:
        for part in sorted(run_dir.glob("part_*.zarr")):
            reader = PartZarrReader(part)
            rows += reader.n_rows
            if reader.n_rows and len(lengths) < 1000:
                sample = reader.column("mask_rle")[: min(250, reader.n_rows)]
                lengths += [len(str(s)) for s in sample]
    mean_len = sum(lengths) / len(lengths) if lengths else MASKS_FALLBACK_RLE_BYTES
    per_row = mean_len + MASKS_JSON_BYTES_PER_ROW
    return Estimate(
        int(rows * per_row),
        rows / MASKS_ROWS_PER_S + 0.1 * len(parts),
        f"{rows:,} detections in {len(parts)} recording(s), {per_row:.0f} bytes each "
        f"(sampled mask length {mean_len:.0f})",
    )


def tracks(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    from detail.api import export_targets as et
    from detail.storage import experiment as experiment_store

    runs = et._tracking_runs(subject, et._only(scope, options))
    rows = 0
    for rid, run_id in runs:
        assignments = (
            experiment_store.recording_path(subject.experiment_dir, rid)
            / "tracking"
            / run_id
            / "assignments"
        )
        rows += _rows(sorted(assignments.glob("part_*.zarr")))
    return Estimate(
        rows * TRACKS_BYTES_PER_ROW,
        rows / TRACKS_ROWS_PER_S + 0.1 * len(runs),
        f"{rows:,} tracked detections in {len(runs)} recording(s); the time assumes "
        f"{TRACKS_ROWS_PER_S} a second (measured with 2 regions and 3 points, fewer are faster)",
    )


def vectors(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    from detail.api import export_targets as et
    from detail.storage.zarr_io import PartZarrReader

    parts = et._embedding_parts(subject, et._only(scope, options))
    rows, nbytes = 0, 0
    for _rid, _track, path in parts:
        reader = PartZarrReader(path)
        n = reader.n_rows
        if not n:
            continue
        column = reader.column("embedding")
        width = int(column.shape[1]) if len(column.shape) > 1 else 1
        rows += n
        nbytes += n * width * column.dtype.itemsize
    nbytes += rows * VECTORS_INDEX_BYTES_PER_ROW + 128
    rate, how = disk_rate(options.get("destination"))
    return Estimate(
        nbytes,
        0.2 + rows / VECTORS_ROWS_PER_S + nbytes / rate,
        f"{rows:,} vectors from {len(parts)} part(s) (counted), written at {how}",
    )


def _kp_sources(subject: ExportSubject, scope: ExportScope, options: dict):
    from detail.api import export_targets as et

    return et._keypoint_sources(subject, options, scope)


def _kp_shape(source) -> tuple[int, int, int, int]:
    """``(series rows, frames in the recording, series, keypoints)`` for one source — a series
    is a track, or a detect run's instance index on the frame (`PointSource`).

    A track's frames come from the coverage the run recorded (``total``); a run with none falls
    back to every frame of the part, which over-counts rather than under-counts. An instance
    index's rows are counted off the part's ``frame_idx`` column.
    """
    series_ids = source.series_ids()
    rows = frames = 0
    for part_id in source.part_ids():
        frames += source.frame_count(part_id)
        for series_id in series_ids:
            if part_id in source.series_parts(series_id):
                rows += source.row_count(series_id, part_id)
    return rows, frames, len(series_ids), len(source.skeleton.names)


@dataclass(frozen=True)
class _LongSample:
    """What writing the first rows of a source's long CSV cost, per row and per cell."""

    bytes_per_row: float
    cells_per_s: (
        float  # cells written per second, the unit a DLC table is also counted in
    )
    rows_per_s: float
    read_s_per_row: float
    chars_per_number: (
        float  # `repr` length of a coordinate or score, as DLC writes them
    )


def _kp_long_sample(source, n: int = 2000) -> _LongSample:
    """Write the first ``n`` rows of ``source``'s long table through the real writer and time it.

    The column set, the cell formats and the run's own numbers are the ones the export will
    have; nothing here is a guess about a format. The read of those rows is timed separately,
    since a run that is mostly reads and one that is mostly formatting scale differently.
    """
    import itertools

    from detail.api import keypoint_export

    key = ("kp-long", str(source.recording_dir), source.run.run_id, source.cutoff)
    cached = _RATE_CACHE.get(key)
    if cached and time.monotonic() - cached[1] < 600.0:
        return cached[0]
    started = time.perf_counter()
    rows = list(itertools.islice(keypoint_export.long_rows(source), n))
    read_s = time.perf_counter() - started
    columns = source.long_columns
    if not rows:
        sample = _LongSample(120.0, 500_000.0, 50_000.0, 0.0, 19.0)
    else:
        with tempfile.TemporaryDirectory(prefix="detail-estimate-") as scratch:
            out = Path(scratch) / "sample.csv"
            started = time.perf_counter()
            keypoint_export.write_csv(out, columns, rows)
            write_s = max(time.perf_counter() - started, 1e-6)
            size = out.stat().st_size
        at = [columns.index(c) for c in ("x_px", "y_px", "score")]
        numeric = [
            repr(v)
            for row in rows
            for v in (row[i] for i in at)
            if isinstance(v, float) and v == v
        ]
        sample = _LongSample(
            bytes_per_row=size / len(rows),
            cells_per_s=len(rows) * len(columns) / write_s,
            rows_per_s=len(rows) / write_s,
            read_s_per_row=read_s / len(rows),
            chars_per_number=(
                sum(len(x) for x in numeric) / len(numeric) if numeric else 19.0
            ),
        )
    _RATE_CACHE[key] = (sample, time.monotonic())
    return sample


def _kp_totals(sources) -> tuple[int, int, int, int]:
    """``(long-table rows, dense cells, parts, recordings)`` summed over ``sources``."""
    long_rows = cells = parts = 0
    for source in sources:
        rows, frames, tracks, k = _kp_shape(source)
        long_rows += rows * k
        cells += frames * tracks * k * 3  # x, y, likelihood per keypoint per track
        parts += len(source.part_ids())
    return long_rows, cells, parts, len(sources)


def keypoints(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    sources = _kp_sources(subject, scope, options)
    if not sources:
        return Estimate(0, 0.0, "no keypoint run to write for this scope")
    rows, _cells, _parts, recordings = _kp_totals(sources)
    sample = _kp_long_sample(sources[0])
    per_row_s = 1.0 / sample.rows_per_s + sample.read_s_per_row
    return Estimate(
        int(rows * sample.bytes_per_row) + 600 * recordings,  # coverage.csv beside it
        rows * per_row_s + 0.1 * recordings,
        f"{rows:,} long-table rows (frames x tracks x keypoints, counted) in {recordings} "
        f"recording(s); {sample.bytes_per_row:.0f} bytes and {per_row_s * 1e6:.0f} us a row, "
        "sampled from the first 2,000 rows through the real writer",
    )


def keypoints_dlc(
    subject: ExportSubject, scope: ExportScope, options: dict
) -> Estimate:
    sources = _kp_sources(subject, scope, options)
    if not sources:
        return Estimate(0, 0.0, "no keypoint run to write for this scope")
    _rows, cells, parts, recordings = _kp_totals(sources)
    sample = _kp_long_sample(sources[0])
    # A cell is `repr(float)` and a comma; the sample measured the real numbers' length.
    h5 = _importable("pandas") and _importable("tables")
    return Estimate(
        int(cells * (sample.chars_per_number + 1)) + 400 * parts,
        cells / sample.cells_per_s + 0.1 * recordings,
        f"{cells:,} cells (frames x tracks x keypoints x 3, counted) in {parts} part(s); "
        f"{sample.chars_per_number:.0f} characters a number and {sample.cells_per_s:,.0f} "
        "cells a second, from the long table's sample"
        + ("; the H5 written beside the CSV is not counted" if h5 else ""),
    )


def _importable(name: str) -> bool:
    from detail.core.installed import is_importable

    return is_importable(name)


def _kp_array_format(
    subject: ExportSubject, scope: ExportScope, options: dict, what: str
) -> Estimate:
    sources = _kp_sources(subject, scope, options)
    if not sources:
        return Estimate(0, 0.0, "no keypoint run to write for this scope")
    _rows, cells, _parts, _recordings = _kp_totals(sources)
    # Not a bound to refuse on: gzip makes the file smaller than this, by an amount that depends
    # on how much of the run is NaN, and a dialog that turned Export off on a figure that is
    # several times too big would be wrong in the direction that blocks someone.
    return Estimate(
        None,
        None,
        f"{what} compresses its arrays, so its size cannot be known before it is written "
        f"(uncompressed they would be about {format_size(cells * 8)}); not counted in the total",
    )


def keypoints_sleap(
    subject: ExportSubject, scope: ExportScope, options: dict
) -> Estimate:
    return _kp_array_format(subject, scope, options, "SLEAP's HDF5")


def keypoints_nwb(
    subject: ExportSubject, scope: ExportScope, options: dict
) -> Estimate:
    return _kp_array_format(subject, scope, options, "NWB")


def keypoints_video(
    subject: ExportSubject, scope: ExportScope, options: dict
) -> Estimate:
    sources = _kp_sources(subject, scope, options)
    if not sources:
        return Estimate(0, 0.0, "no keypoint run to write for this scope")
    extra = DRAW_POINTS_S_PER_FRAME + (
        DRAW_OVERLAY_S_PER_FRAME if options.get("overlay") else 0.0
    )
    frames = {
        s.recording_id: sum(s.frame_count(p) for p in s.part_ids()) for s in sources
    }
    video = _timed_video(
        subject.experiment_dir,
        [s.recording_id for s in sources],
        size=None,
        extra_s_per_frame=extra,
        label="video",
        frames=frames,
    )
    if video.bytes is None:
        return video
    data = keypoints(subject, scope, options)  # keypoints.csv beside the videos
    return Estimate(
        video.bytes + (data.bytes or 0),
        video.seconds + (data.seconds or 0.0),
        f"{video.basis}; drawing time ({extra * 1000:.1f} ms a frame) is measured on the "
        f"remote-detail engine, not this one; plus keypoints.csv ({data.basis})",
    )


def performance(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    return _small("one short report from the run's activity log", 3_000, 0.2)


# --------------------------------------------------------------------------- #
# analysis runs
# --------------------------------------------------------------------------- #
def _instances(subject: ExportSubject, scope: ExportScope):
    from detail.api import export_targets as et
    from detail.storage import behavior_output

    artifact = behavior_output.read_behavior_output(et._analysis_dir(subject))
    return artifact, et._narrow(list(artifact.instances), scope)


def instances(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    _artifact, rows = _instances(subject, scope)
    if not rows:
        return Estimate(0, 0.0, "no instances in this scope: nothing to write")
    return Estimate(
        INSTANCES_BASE_BYTES + INSTANCES_BYTES_PER_ROW * len(rows),
        INSTANCES_BASE_S + INSTANCES_S_PER_ROW * len(rows),
        f"{len(rows):,} instances, and the four figures drawn from them",
    )


def ethogram(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    _artifact, rows = _instances(subject, scope)
    return Estimate(
        ETHOGRAM_BASE_BYTES + ETHOGRAM_BYTES_PER_ROW * len(rows),
        ETHOGRAM_BASE_S + ETHOGRAM_S_PER_ROW * len(rows),
        f"{len(rows):,} instances drawn as one ethogram, plus a CSV per recording",
    )


def frame_labels(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    from detail.api import export_targets as et
    from detail.storage import frame_view

    run_dir = et._analysis_dir(subject)
    frames, columns, units, unreadable = 0, 0, 0, 0
    for recording_id, unit_subject in frame_view.discover_frame_views(run_dir):
        if not scope.includes(recording_id):
            continue
        try:
            view = frame_view.read_frame_view(run_dir, recording_id, unit_subject)
        except dirscan.UNREADABLE_RECORD_ERRORS:
            unreadable += 1  # the export will say why; the estimate leaves it out
            continue
        n = int(view.t_ns.shape[0])
        if scope.frames is not None:
            lo, hi = scope.frames
            n = max(0, min(n, hi + 1) - max(lo, 0))
        frames += n
        columns = max(columns, len(view.source_labels))
        units += 1
    per_frame_bytes = FRAME_LABELS_BASE_BYTES_PER_FRAME + 2 * columns
    per_frame_s = FRAME_LABELS_BASE_S_PER_FRAME + FRAME_LABELS_S_PER_COLUMN * columns
    return Estimate(
        frames * per_frame_bytes,
        0.05 * units + frames * per_frame_s,
        f"{frames:,} frames in {units} unit(s), {columns} behavior column(s) wide"
        + (f"; {unreadable} unreadable view(s) left out" if unreadable else ""),
    )


def clips(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    from detail.api import export_targets as et

    _artifact, rows = _instances(subject, scope)
    if not rows:
        return Estimate(0, 0.0, "no instances in this scope: nothing to write")
    cap = options.get("max_per_cluster")
    if cap:
        kept, seen = [], {}
        for row in rows:
            if seen.get(row.source_label, 0) < int(cap):
                kept.append(row)
                seen[row.source_label] = seen.get(row.source_label, 0) + 1
        rows = kept
    frames = sum(max(int(r.end_frame) - int(r.start_frame) + 1, 1) for r in rows)
    recordings = sorted({r.recording_id for r in rows})
    renders = options.get("renders") or ("overlay",)
    sample = next(
        (rid for rid in recordings if _first_video(subject.experiment_dir, rid)), None
    )
    if sample is None:
        return Estimate(None, None, "no video for these clips is on this machine")
    estimate = _timed_video(
        subject.experiment_dir,
        [sample],
        size=int(options.get("size", 640)),
        extra_s_per_frame=0.0,
        label="clip",
        frames={sample: frames},
    )
    if estimate.bytes is None:
        return estimate
    n = max(len(renders), 1)
    return Estimate(
        estimate.bytes * n,
        estimate.seconds * (1 + 0.8 * (n - 1)) + 0.2 * len(rows),
        f"{len(rows):,} bouts, {frames:,} frames; {estimate.basis}; {n} flavour(s)",
    )


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def report(subject: ExportSubject, scope: ExportScope, options: dict) -> Estimate:
    return Estimate(
        None,
        None,
        "a comparison writes one figure per measurement, and how many there are is only known "
        "once it is computed",
    )
