"""Measure rows over a location-export run — the locations source of a comparison (§24).

A behavior artifact (§19) says *what* an animal was doing; a location export (§16) says where
its body was and what shape it had, a row per frame. This turns one finished location-export
run into one value per recording, track and window for each measure below, in
``measures.COLUMNS``' own schema, so ``compare.observations``, the summaries, the tables and
the figures consume them unchanged. Nothing is re-measured from masks: the per-frame values
are the run's, and a comparison stays seconds.

**Locomotion** is measured on the path of the *body* centre (``body_cx_mm`` / ``body_cy_mm``,
which leaves the tail out when the run removed thin parts). Three rules, each from a failure on
real recordings (detail#70):

- **A raw centre path over-counts.** Frame-to-frame jitter adds distance for an animal standing
  still, so the path is a running mean over ``smooth_s`` before it is measured.
- **A gap or a jump is not movement.** The path breaks at a gap in tracking and at a raw step
  faster than ``jump_speed`` of the animal's own median body lengths per second. A speed and
  not a length per step, so the rule means the same thing at any sampling rate. Nothing is
  joined across a break, and a stretch too short to smooth is dropped, which is what removes
  a single mis-tracked sample.
- **A window takes the steps that fall in it.** The path is smoothed once per track and each
  step belongs to the window its time is in, so distance over adjacent windows sums to the
  distance over their union.

**Body shape** is the median over the window of the run's per-frame length, width and area.

Everything in millimetres depends on the recording's distance calibration, so a recording
exported without one is refused rather than measured in pixels beside recordings measured in
millimetres.

Pure over tables: numpy and csv only, no zarr, no Qt.
"""

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from detail.analysis import measures as measures_lib
from detail.analysis.locations import sampled_indices

NS_PER_S = 1_000_000_000
TRACKS_FILE = "tracks.csv"
MANIFEST_FILE = "locations.yaml"

#: The columns of ``tracks.csv`` a comparison reads. All of them arrived with the body
#: (detail#75); a run made before it has none and is refused, by name.
_BODY_COLUMNS = (
    "body_cx_mm",
    "body_cy_mm",
    "body_length_mm",
    "body_width_mm",
    "body_area_mm2",
)


class LocationSourceError(ValueError):
    """A location-export run cannot answer what a comparison asked of it."""


@dataclass(frozen=True)
class LocationMeasure:
    """One measure of the locations source: what a surface needs to print it.

    ``unit`` is how a value is rendered and is never ``""``: the report reads ``""`` as a plain
    count, whose pseudo-count is one bout. ``fraction`` is rendered as a percentage, like a
    behavior's share of time. ``fans_out`` is always ``False``; it is here because the report
    asks every measure, whatever its source.
    """

    measure_id: str
    label: str
    description: str
    unit: str
    fans_out: bool = False


def _measure(measure_id: str, label: str, unit: str, description: str):
    return measure_id, LocationMeasure(measure_id, label, description, unit)


#: The measures of the locations source, in the order every surface lists them.
MEASURES: dict[str, LocationMeasure] = dict(
    [
        _measure(
            "distance",
            "Distance travelled",
            "m",
            "Length of the smoothed path of the body's centre, summed within unbroken "
            "stretches of tracking.",
        ),
        _measure(
            "time_moving",
            "Time moving",
            "fraction",
            "Share of tracked time the body's centre moved faster than the moving threshold.",
        ),
        _measure(
            "speed_moving",
            "Speed when moving",
            "cm/s",
            "Mean speed over the time counted as moving. No value for an animal that never "
            "moved.",
        ),
        _measure(
            "body_length",
            "Body length",
            "mm",
            "Median length of the body along its long axis.",
        ),
        _measure(
            "body_width",
            "Body width",
            "mm",
            "Median width of the body across its long axis.",
        ),
        _measure(
            "body_elongation",
            "Body length : width",
            "ratio",
            "Median of the body's length divided by its width, frame by frame: how stretched "
            "out or balled up the animal is. It does not depend on the calibration.",
        ),
        _measure(
            "body_area",
            "Body area",
            "cm²",
            "Median area of the body.",
        ),
    ]
)


@dataclass(frozen=True)
class Settings:
    """How a location-export run is read into measures (``configs.comparison``).

    ``sample_fps`` thins the run's rows to at most that many a second per track before anything
    is measured, by the export's own rule (the row nearest each tick); ``None`` reads every row.
    A run exported at a lower rate is read as it is: no row is invented.
    """

    sample_fps: float | None
    smooth_s: float
    moving_cm_s: float
    jump_speed: float


@dataclass(frozen=True)
class Track:
    """One track of one recording, as the columns a measure reads, in time order."""

    t_ns: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    length_mm: np.ndarray
    width_mm: np.ndarray
    area_mm2: np.ndarray

    def thinned(self, sample_fps: float | None) -> "Track":
        if sample_fps is None or len(self.t_ns) == 0:
            return self
        keep = sampled_indices(self.t_ns, sample_fps)
        return Track(*(getattr(self, name)[keep] for name in self.__dataclass_fields__))


@dataclass(frozen=True)
class SmoothedPath:
    """A track's smoothed path as steps: when each was taken, how long, how fast."""

    t_ns: np.ndarray
    step_mm: np.ndarray
    speed_cm_s: np.ndarray
    dt_ns: float


def _running_mean(values: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(values, np.ones(n) / n, mode="valid")


def smoothed_path(track: Track, settings: Settings) -> SmoothedPath | None:
    """The track's path as steps between smoothed positions, broken at gaps and jumps.

    ``None`` for a track with too few samples to have a path at all.
    """
    t = track.t_ns.astype(np.float64) / NS_PER_S
    if len(t) < 3:
        return None
    dt = float(np.median(np.diff(t)))
    if not dt > 0:
        return None
    n = max(1, int(round(settings.smooth_s / dt)))
    x, y = track.x_mm, track.y_mm
    raw = np.hypot(np.diff(x), np.diff(y))
    elapsed = np.diff(t)
    broken = elapsed > 1.5 * dt
    body = float(np.nanmedian(track.length_mm)) if len(track.length_mm) else np.nan
    if np.isfinite(body) and body > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            broken |= raw / elapsed > settings.jump_speed * body
    edges = np.flatnonzero(broken) + 1
    times, steps, speeds = [], [], []
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(t)]):
        if hi - lo < n + 1:
            continue
        sx = _running_mean(x[lo:hi], n)
        sy = _running_mean(y[lo:hi], n)
        st = _running_mean(t[lo:hi], n)
        step = np.hypot(np.diff(sx), np.diff(sy))
        times.append((st[:-1] + st[1:]) / 2.0)
        steps.append(step)
        speeds.append(step / np.diff(st) / 10.0)
    if not steps:
        return None
    return SmoothedPath(
        t_ns=np.concatenate(times) * NS_PER_S,
        step_mm=np.concatenate(steps),
        speed_cm_s=np.concatenate(speeds),
        dt_ns=dt * NS_PER_S,
    )


def _median(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def window_values(
    track: Track,
    path: SmoothedPath | None,
    start_t_ns: int,
    end_t_ns: int,
    settings: Settings,
) -> dict[str, tuple[float | None, float | None, int]]:
    """``{measure_id: (value, denominator_ns, n)}`` over ``[start_t_ns, end_t_ns)``.

    ``value`` is ``None`` where the window has nothing to measure it from; ``n`` is the number
    of samples, or of steps, behind it. ``denominator_ns`` is the path time behind a share of
    time, which is what a pseudo-count in seconds is converted with.
    """
    out: dict[str, tuple[float | None, float | None, int]] = {}
    here = (track.t_ns >= start_t_ns) & (track.t_ns < end_t_ns)
    count = int(here.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        elongation = track.length_mm[here] / track.width_mm[here]
    out["body_length"] = (_median(track.length_mm[here]), None, count)
    out["body_width"] = (_median(track.width_mm[here]), None, count)
    out["body_elongation"] = (_median(elongation), None, count)
    area = _median(track.area_mm2[here])
    out["body_area"] = (None if area is None else area / 100.0, None, count)

    steps = np.zeros(0, dtype=bool)
    if path is not None:
        steps = (path.t_ns >= start_t_ns) & (path.t_ns < end_t_ns)
    n_steps = int(steps.sum())
    if n_steps == 0:
        for name in ("distance", "time_moving", "speed_moving"):
            out[name] = (None, None, 0)
        return out
    speed = path.speed_cm_s[steps]
    moving = speed > settings.moving_cm_s
    out["distance"] = (float(path.step_mm[steps].sum() / 1000.0), None, n_steps)
    out["time_moving"] = (float(moving.mean()), float(n_steps * path.dt_ns), n_steps)
    out["speed_moving"] = (
        float(speed[moving].mean()) if moving.any() else None,
        None,
        int(moving.sum()),
    )
    return out


def read_manifest(recording_dir: Path) -> dict:
    """One recording's ``locations.yaml``.

    Raises:
        LocationSourceError: The run wrote none for this recording.
    """
    path = Path(recording_dir) / MANIFEST_FILE
    if not path.is_file():
        raise LocationSourceError(
            f"{Path(recording_dir).name} has no {MANIFEST_FILE} in this location-export run"
        )
    with open(path, encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle) or {}


def require_body(manifest: Mapping, recording_id: str, run_id: str) -> None:
    """Refuse a recording the run measured no body for, or measured without a calibration.

    Raises:
        LocationSourceError: Naming the recording and what to do about it.
    """
    columns = (manifest.get("columns") or {}).get("tracks") or []
    missing = [name for name in _BODY_COLUMNS if name not in columns]
    if missing:
        raise LocationSourceError(
            f"location-export run {run_id} was made before the export measured the body "
            f"({recording_id} has no {missing[0]} column). Run the location export again "
            "and compare the new run."
        )
    if not manifest.get("pixels_per_mm"):
        raise LocationSourceError(
            f"{recording_id} had no distance calibration when run {run_id} exported it, so "
            "its distances and body sizes are not in millimetres. Calibrate the recording, "
            "run the location export again, and compare the new run."
        )


def read_tracks(recording_dir: Path) -> dict[str, Track]:
    """``{object_id: Track}`` from one recording's ``tracks.csv``, each in time order.

    A blank cell (a frame with an empty mask has no length) reads as NaN.
    """
    names = ("t_ns", *_BODY_COLUMNS)
    columns: dict[str, dict[str, list[float]]] = {}
    with open(
        Path(recording_dir) / TRACKS_FILE, newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            track = columns.setdefault(row["object_id"], {name: [] for name in names})
            for name in names:
                cell = row[name]
                track[name].append(float(cell) if cell != "" else np.nan)
    out = {}
    for object_id, track in columns.items():
        t_ns = np.asarray(track["t_ns"], dtype=np.int64)
        order = np.argsort(t_ns, kind="stable")
        out[object_id] = Track(
            t_ns[order],
            *(
                np.asarray(track[name], dtype=np.float64)[order]
                for name in _BODY_COLUMNS
            ),
        )
    return out


def location_rows(
    run_dir: Path,
    run_id: str,
    recordings: Sequence[str],
    *,
    settings: Settings,
    windows_for: Callable,
    measures: Sequence[str] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Measure rows for every recording, track and window of a location-export run.

    ``windows_for(start_t_ns, end_t_ns, frame_ns=...)`` is ``WindowSpec.windows_for``: each
    track's windows are cut from its own first to last sample. ``measures`` narrows what is
    emitted; ``None`` is every measure. A window with nothing to measure a value from emits a
    blank, which an observation drops rather than reading as zero.

    Raises:
        LocationSourceError: An unknown measure, a recording the run did not write, a run made
            before the body was measured, or a recording exported without a calibration.
    """
    chosen = list(MEASURES) if not measures else list(measures)
    unknown = [m for m in chosen if m not in MEASURES]
    if unknown:
        raise LocationSourceError(
            f"the locations source has no measure {', '.join(unknown)}; it has "
            f"{', '.join(MEASURES)}"
        )
    rows: list[dict] = []
    total = len(recordings)
    for done, recording_id in enumerate(recordings):
        if stop_check is not None and stop_check():
            break
        recording_dir = Path(run_dir) / recording_id
        require_body(read_manifest(recording_dir), recording_id, run_id)
        for object_id, whole in sorted(read_tracks(recording_dir).items()):
            track = whole.thinned(settings.sample_fps)
            if len(track.t_ns) == 0:
                continue
            path = smoothed_path(track, settings)
            frame_ns = (
                int(np.median(np.diff(track.t_ns))) if len(track.t_ns) > 1 else None
            )
            start = int(track.t_ns[0])
            end = int(track.t_ns[-1]) + (frame_ns or 1)
            for window in windows_for(start, end, frame_ns=frame_ns or None):
                values = window_values(
                    track, path, window.start_t_ns, window.end_t_ns, settings
                )
                for measure_id in chosen:
                    value, denominator, n = values[measure_id]
                    row = {
                        "recording_id": recording_id,
                        "subject": object_id,
                        "label": "",
                        "label_to": "",
                        "label_display": "",
                        "label_is_remainder": False,
                        "window_id": window.window_id,
                        "window_start_t_ns": window.start_t_ns,
                        "window_end_t_ns": window.end_t_ns,
                        "measure": measure_id,
                        "value": "" if value is None else value,
                        "denominator_ns": "" if denominator is None else denominator,
                        "n": n,
                    }
                    assert (
                        list(row) == measures_lib.COLUMNS
                    ), "the locations source writes the measures schema"
                    rows.append(row)
        if progress_cb is not None:
            progress_cb(done + 1, total)
    return rows


def run_fingerprint(run_dir: Path, recordings: Sequence[str]) -> dict:
    """What identifies the run's output for the recordings compared: each one's manifest and
    the size of its table. A finished run is never rewritten (§16), so this moves only when
    the run was replaced, which is what a comparison's drift check asks."""
    out = {}
    for recording_id in sorted(recordings):
        recording_dir = Path(run_dir) / recording_id
        manifest = recording_dir / MANIFEST_FILE
        tracks = recording_dir / TRACKS_FILE
        out[recording_id] = {
            "manifest": (
                manifest.read_text(encoding="utf-8-sig") if manifest.is_file() else None
            ),
            "tracks_bytes": tracks.stat().st_size if tracks.is_file() else None,
        }
    return out
