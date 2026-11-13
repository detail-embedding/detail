"""Location plots — a location export's paths, occupancy and speed, as a ``PlotSpec``.

Pure numpy over the run's own CSVs (DESIGN_PLAN §16 "Location plots"): ``read_tracks`` reads a
``tracks.csv`` once, ``read_frame`` its ``locations.yaml``; ``trace`` cuts one subject's path out
of it; ``build_plot`` turns a set of paths and a ``LocationPlotView`` into a library-free
``report/plot_spec.PlotSpec``. Nothing here draws.

Positions are read once, as the normalized centroid, and converted with the manifest's video
resolution and calibration — the conversion ``analysis/locations.py`` used to write the ``_px``
and ``_mm`` columns — so one read serves every unit.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detail.configs.location_plots import LocationPlotView
from detail.core import atomic
from detail.core import yaml_load
from detail.report.plot_spec import (
    Axis,
    Band,
    Camera,
    ColorScale,
    Heat,
    Line,
    Panel,
    PlotSpec,
)

#: The positions ``tracks.csv`` carries for every run; any other ``source`` names a keypoint.
CENTROID_SOURCES = ("mask", "bbox")

#: The group a recording with no value in the ``group_by`` column falls into — never dropped.
NO_VALUE = "(no value)"

#: Consecutive samples further apart than this many median sample intervals are a tracking gap:
#: the drawn line breaks there and no speed is computed across it.
GAP_INTERVALS = 5.0

_TITLES = {
    "trace": "Path",
    "heatmap": "Occupancy",
    "speed": "Speed over time",
    "trace3d": "Path over time",
}

_COLUMNS = ("frame_idx", "t_ns", "object_id")


@dataclass(frozen=True)
class FrameInfo:
    """A recording's frame, from its export manifest: pixel size and calibration."""

    width: float
    height: float
    pixels_per_mm: float | None

    def extent(self, units: str) -> tuple[float, float]:
        """The frame's ``(width, height)`` in ``units``."""
        if units == "norm":
            return 1.0, 1.0
        if units == "px":
            return self.width, self.height
        return self.width / self.pixels_per_mm, self.height / self.pixels_per_mm


@dataclass(frozen=True)
class TrackTable:
    """One ``tracks.csv``, both centroid sources, every subject: ``{object_id: arrays}``."""

    columns: dict[int, dict[str, np.ndarray]]

    def object_counts(self) -> list[tuple[int, int]]:
        """``(object_id, rows)`` for every subject, most-tracked first."""
        counts = [(oid, len(cols["t_ns"])) for oid, cols in self.columns.items()]
        return sorted(counts, key=lambda item: (-item[1], item[0]))

    def first_t_ns(self) -> int | None:
        """The recording's first tracked sample, over every subject — time zero of its plots,
        so two animals of one recording share one clock."""
        starts = [int(cols["t_ns"][0]) for cols in self.columns.values()]
        return min(starts) if starts else None


@dataclass(frozen=True)
class Subject:
    """One animal of one recording: its label (the user's name for it) and its track there."""

    label: str
    object_id: int
    rows: int
    #: True when the label came from the run's ``subject_labels``, not ``subject N``.
    labelled: bool


@dataclass(frozen=True)
class Trace:
    """One subject's path through one recording, at full resolution, in normalized units."""

    recording_id: str
    name: str
    object_id: int
    #: The animal's label — the identity a plot shows, groups and colours by.
    subject: str
    #: Whether the title should name the animal: the recording has several, or the run named it.
    show_subject: bool
    #: Whether ``subject`` came from the run's ``subject_labels`` rather than ``subject N``.
    subject_labelled: bool
    frame: FrameInfo
    t_s: np.ndarray
    x_norm: np.ndarray
    y_norm: np.ndarray
    #: The keypoint this path follows, or None for a centroid.
    point: str | None = None

    def xy(self, units: str) -> tuple[np.ndarray, np.ndarray]:
        """The path's coordinates in ``units``."""
        w, h = self.frame.extent(units)
        return self.x_norm * w, self.y_norm * h

    def breaks(self) -> np.ndarray:
        """``breaks[i]`` is True when samples ``i-1`` and ``i`` straddle a tracking gap."""
        out = np.zeros(len(self.t_s), dtype=bool)
        if len(self.t_s) < 2:
            return out
        dt = np.diff(self.t_s)
        positive = dt[dt > 0]
        if not len(positive):
            return out
        out[1:] = dt > GAP_INTERVALS * float(np.median(positive))
        return out

    def steps(self, units: str) -> tuple[np.ndarray, np.ndarray]:
        """``(distance, dt)`` between consecutive samples, NaN across a gap; length n-1."""
        x, y = self.xy(units)
        dist = np.hypot(np.diff(x), np.diff(y))
        dt = np.diff(self.t_s)
        broken = self.breaks()[1:] | (dt <= 0)
        return np.where(broken, np.nan, dist), np.where(broken, np.nan, dt)

    def speed(self, units: str) -> np.ndarray:
        """Per-sample speed in units/s (the step that arrives at the sample; NaN after a gap)."""
        if len(self.t_s) < 2:
            return np.zeros(len(self.t_s))
        dist, dt = self.steps(units)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = dist / dt
        return np.concatenate([step[:1], step])


def read_manifest(
    manifest_path: Path | str,
) -> tuple[FrameInfo, dict[int, str] | None, list[str]]:
    """The frame size and calibration a ``locations.yaml`` recorded, its tracking run's
    ``subject_labels`` (``{object_id: label}``; None for an export made before it recorded
    them, or a run that never named its animals), and its keypoints' names (empty when the
    export read no keypoint track run).

    Raises:
        ValueError: If the manifest has no video resolution (normalized coordinates cannot be
            turned into pixels without one).
    """
    data = yaml_load.safe_load(atomic.read_text(Path(manifest_path))) or {}
    resolution = data.get("video_resolution")
    if not resolution:
        raise ValueError(f"{manifest_path} records no video_resolution")
    ppm = data.get("pixels_per_mm")
    frame = FrameInfo(float(resolution[0]), float(resolution[1]), ppm or None)
    names = [str(n) for n in data.get("keypoint_names") or []]
    return frame, normalize_labels(data.get("subject_labels")), names


def normalize_labels(labels: dict | None) -> dict[int, str] | None:
    """``subject_labels`` with int keys (YAML may hand back ``"0"``) and str values."""
    if not labels:
        return None
    return {int(k): str(v) for k, v in labels.items()}


def subjects_of(
    table: TrackTable, labels: dict[int, str] | None
) -> tuple[list[Subject], list[str]]:
    """``(subjects with rows, labelled subjects with none)`` of one recording.

    A labelled track with no rows in ``tracks.csv`` is reported rather than dropped silently: the
    run named an animal that this export never saw. Without labels a subject is ``subject N``.
    """
    labels = labels or {}
    present = [
        Subject(labels.get(oid, f"subject {oid}"), oid, rows, oid in labels)
        for oid, rows in table.object_counts()
    ]
    seen = {s.object_id for s in present}
    missing = [label for oid, label in sorted(labels.items()) if oid not in seen]
    return present, missing


def read_tracks(tracks_csv: Path | str) -> TrackTable:
    """Read a ``tracks.csv``'s timing and both normalized centroids, split by ``object_id``.

    Only the seven columns plots use are parsed, read by name so column order does not matter.

    Raises:
        ValueError: If a column plots need is missing.
    """
    wanted = _COLUMNS + tuple(
        f"{src}_c{axis}_norm" for src in ("mask", "bbox") for axis in ("x", "y")
    )
    with open(tracks_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        missing = [c for c in wanted if c not in header]
        if missing:
            raise ValueError(f"{tracks_csv} has no column(s) {missing}")
        index = [header.index(c) for c in wanted]
        rows = [[row[i] for i in index] for row in reader if row]
    if not rows:
        return TrackTable({})
    raw = np.asarray(rows, dtype=object).T
    frame_idx = raw[0].astype(np.int64)
    t_ns = raw[1].astype(np.int64)
    object_id = raw[2].astype(np.int64)
    centroids = [np.asarray(col, dtype=np.float64) for col in raw[3:]]
    columns: dict[int, dict[str, np.ndarray]] = {}
    for oid in np.unique(object_id):
        rows_of = np.flatnonzero(object_id == oid)
        rows_of = rows_of[np.argsort(frame_idx[rows_of], kind="stable")]
        cols = {"frame_idx": frame_idx[rows_of], "t_ns": t_ns[rows_of]}
        for name, values in zip(wanted[3:], centroids):
            cols[name] = values[rows_of]
        columns[int(oid)] = cols
    return TrackTable(columns)


@dataclass(frozen=True)
class KeypointTable:
    """One ``keypoints.csv``: ``{(object_id, point): (t_ns, x_norm, y_norm)}``, a point that
    was not placed or is under the cut-off (``visible`` 0) as NaN."""

    series: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]]


def read_keypoints(keypoints_csv: Path | str) -> KeypointTable:
    """Read a ``keypoints.csv`` by column name, into per-animal, per-point series in frame order.

    Raises:
        ValueError: If a column plots need is missing.
    """
    wanted = (
        "frame_idx",
        "t_ns",
        "object_id",
        "keypoint",
        "x_norm",
        "y_norm",
        "visible",
    )
    with open(keypoints_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        missing = [c for c in wanted if c not in header]
        if missing:
            raise ValueError(f"{keypoints_csv} has no column(s) {missing}")
        index = [header.index(c) for c in wanted]
        rows = [[row[i] for i in index] for row in reader if row]
    grouped: dict[tuple[int, str], list] = {}
    for frame, t, oid, point, x, y, visible in rows:
        shown = visible == "1" and x != "" and y != ""
        grouped.setdefault((int(oid), point), []).append(
            (
                int(frame),
                int(t),
                float(x) if shown else np.nan,
                float(y) if shown else np.nan,
            )
        )
    series = {}
    for key, values in grouped.items():
        values.sort(key=lambda v: v[0])
        arr = np.asarray(values, dtype=np.float64)
        series[key] = (arr[:, 1].astype(np.int64), arr[:, 2], arr[:, 3])
    return KeypointTable(series)


def trace(
    table: TrackTable,
    frame: FrameInfo,
    subject: Subject,
    *,
    recording_id: str,
    name: str,
    source: str,
    multi_subject: bool,
    keypoints: KeypointTable | None = None,
    point: str | None = None,
) -> Trace:
    """One subject's path, timed from the recording's first tracked sample (any subject): its
    ``source`` centroid, or with ``point`` that keypoint's positions from ``keypoints``.
    """
    if point is not None:
        t_ns, x, y = keypoints.series[(subject.object_id, point)]
        return Trace(
            recording_id=recording_id,
            name=name,
            object_id=subject.object_id,
            subject=subject.label,
            show_subject=multi_subject or subject.labelled,
            subject_labelled=subject.labelled,
            frame=frame,
            t_s=(t_ns - table.first_t_ns()).astype(np.float64) / 1e9,
            x_norm=x,
            y_norm=y,
            point=point,
        )
    cols = table.columns[subject.object_id]
    return Trace(
        recording_id=recording_id,
        name=name,
        object_id=subject.object_id,
        subject=subject.label,
        show_subject=multi_subject or subject.labelled,
        subject_labelled=subject.labelled,
        frame=frame,
        t_s=(cols["t_ns"] - table.first_t_ns()).astype(np.float64) / 1e9,
        x_norm=cols[f"{source}_cx_norm"],
        y_norm=cols[f"{source}_cy_norm"],
    )


def recording_traces(
    table: TrackTable,
    frame: FrameInfo,
    labels: dict[int, str] | None,
    view: LocationPlotView,
    *,
    recording_id: str,
    name: str,
    keypoints: KeypointTable | None = None,
    keypoint_names: list[str] | None = None,
) -> tuple[list[Trace], list[str]]:
    """The chosen subjects of one recording as traces, and a note for each one left out.

    ``view.subjects`` names animals by label and is matched in *this* recording — track numbers
    differ between recordings and runs, a label does not. Unset is every subject with rows. A
    keypoint ``source``, or ``points`` on a path plot, reads ``keypoints`` (one trace per
    point per animal).

    Raises:
        ValueError: A keypoint is asked for that the run's skeleton does not have, or the run
            was exported without keypoints.
    """
    points = position_points(view)
    if points:
        known = list(keypoint_names or [])
        if keypoints is None or not known:
            raise ValueError(
                f"{', '.join(points)} is a keypoint, and this run was exported without a "
                "keypoint track run; use mask or bbox, or export again with keypoints"
            )
        unknown = [p for p in points if p not in known]
        if unknown:
            raise ValueError(
                f"no keypoint {', '.join(unknown)} in this run (it has {', '.join(known)})"
            )
    present, missing = subjects_of(table, labels)
    wanted = None if view.subjects is None else set(view.subjects)
    # By label, not track number, so every recording's panel lists its animals the same way.
    chosen = sorted(
        (s for s in present if wanted is None or s.label in wanted),
        key=lambda s: _order_key(s.label),
    )
    notes = [
        f"{name} · {label}: no tracked rows"
        for label in missing
        if wanted is None or label in wanted
    ]
    if not present:
        notes.append(f"{name}: no tracked rows")
    elif not chosen:
        notes.append(f"{name}: none of {', '.join(sorted(wanted))}")
    multi = len(present) > 1
    if not points:
        traces = [
            trace(
                table,
                frame,
                s,
                recording_id=recording_id,
                name=name,
                source=view.source,
                multi_subject=multi,
            )
            for s in chosen
        ]
        return traces, notes
    traces = []
    for s in chosen:
        for point in points:
            if (s.object_id, point) not in keypoints.series:
                notes.append(f"{name} · {s.label} · {point}: no keypoint rows")
                continue
            traces.append(
                trace(
                    table,
                    frame,
                    s,
                    recording_id=recording_id,
                    name=name,
                    source=view.source,
                    multi_subject=multi,
                    keypoints=keypoints,
                    point=point,
                )
            )
    return traces, notes


def position_points(view: LocationPlotView) -> list[str]:
    """The keypoints a view draws positions from: ``points`` on a path plot, else ``source``
    when it names a keypoint; empty for a centroid."""
    if view.points and view.kind in ("trace", "trace3d"):
        return list(view.points)
    if view.source not in CENTROID_SOURCES:
        return [view.source]
    return []


# --------------------------------------------------------------------------- #
# Aggregates (pure)
# --------------------------------------------------------------------------- #
def subsample(n: int, cap: int) -> np.ndarray:
    """At most ``cap`` evenly spaced indices into ``range(n)``, always keeping both ends."""
    if n <= cap:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, cap).round().astype(np.int64))


def drawn(tr: Trace, columns: list[np.ndarray], max_points: int) -> list[np.ndarray]:
    """``columns`` (per-sample arrays of ``tr``) thinned to ``max_points``, with a NaN row
    inserted wherever a tracking gap falls between two kept samples."""
    keep = subsample(len(tr.t_s), max_points)
    gaps = np.cumsum(tr.breaks())
    split = np.flatnonzero(np.diff(gaps[keep]) > 0) + 1
    out = []
    for col in columns:
        kept = np.asarray(col, dtype=np.float64)[keep]
        out.append(np.insert(kept, split, np.nan))
    return out


def occupancy(
    x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray
) -> np.ndarray:
    """The fraction of samples in each cell, ``grid[row, col]`` (rows are y); sums to 1.

    A sample on the far edge counts in the last cell; samples outside the frame are dropped
    before normalizing, so the grid describes where the animal was *in* the frame.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    grid, _, _ = np.histogram2d(y[finite], x[finite], bins=(y_edges, x_edges))
    total = grid.sum()
    return grid / total if total > 0 else grid


def cell_edges(extent: tuple[float, float], bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Square cells over ``extent`` (width, height): ``bins`` along the longer side."""
    w, h = extent
    cell = max(w, h) / bins
    nx, ny = max(1, round(w / cell)), max(1, round(h / cell))
    return np.linspace(0.0, w, nx + 1), np.linspace(0.0, h, ny + 1)


def binned_speed(
    tr: Trace, units: str, bin_s: float, t_end: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(bin centres, speed)``: distance travelled in each bin ÷ time tracked in it.

    A step is assigned to the bin its start falls in; a bin with no tracked time is NaN, so a
    dropout reads as missing rather than as standing still.
    """
    n_bins = max(1, int(np.ceil(t_end / bin_s))) if t_end > 0 else 1
    centres = (np.arange(n_bins) + 0.5) * bin_s
    if len(tr.t_s) < 2:
        return centres, np.full(n_bins, np.nan)
    dist, dt = tr.steps(units)
    ok = np.isfinite(dist) & np.isfinite(dt)
    which = np.minimum((tr.t_s[:-1][ok] / bin_s).astype(np.int64), n_bins - 1)
    dist_sum = np.bincount(which, weights=dist[ok], minlength=n_bins)
    time_sum = np.bincount(which, weights=dt[ok], minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(time_sum > 0, dist_sum / time_sum, np.nan)
    return centres, speed


def mean_sem(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column ``(mean, sem, n)`` over equal-length rows, ignoring NaN; SEM is NaN at n < 2."""
    stack = np.vstack(rows)
    finite = np.isfinite(stack)
    n = finite.sum(axis=0)
    total = np.where(finite, stack, 0.0).sum(axis=0)
    mean = np.where(n > 0, total / np.maximum(n, 1), np.nan)
    dev = np.where(finite, stack - mean, 0.0)
    var = (dev**2).sum(axis=0) / np.maximum(n - 1, 1)
    sem = np.where(n >= 2, np.sqrt(var / np.maximum(n, 1)), np.nan)
    return mean, sem, n


# --------------------------------------------------------------------------- #
# Units, groups and colours
# --------------------------------------------------------------------------- #
def is_grouped(view: LocationPlotView) -> bool:
    """Whether panels pool animals across recordings (a column, the animal label, or both)."""
    return view.group_by is not None or view.split_by_subject


def resolve_units(traces: list[Trace], view: LocationPlotView) -> str:
    """The units a plot is drawn in (DESIGN_PLAN §16 "Location plots").

    Positions pooled across recordings (a by-group path or heat map) are always ``norm``. Else
    the view's choice, or — unset — ``mm`` when every recording is calibrated, ``px`` otherwise.

    Raises:
        ValueError: ``mm`` was asked for and a plotted recording is uncalibrated.
    """
    if is_grouped(view) and view.kind != "speed":
        return "norm"
    uncalibrated = sorted({t.name for t in traces if not t.frame.pixels_per_mm})
    if view.units is None:
        return "px" if uncalibrated else "mm"
    if view.units == "mm" and uncalibrated:
        raise ValueError(
            f"mm needs a distance calibration; not calibrated: {', '.join(uncalibrated)}"
        )
    return view.units


def _order_key(text: str) -> tuple[bool, str]:
    return (NO_VALUE in text, text.lower())


def group_traces(
    traces: list[Trace], values: dict[str, str], view: LocationPlotView
) -> list[tuple[str, list[Trace]]]:
    """``[(group, animals)]``: each animal's Recording Information value, its label, or both
    (``wt · resident``), sorted with ``(no value)`` last."""
    groups: dict[str, list[Trace]] = {}
    for tr in traces:
        parts = []
        if view.group_by is not None:
            parts.append((values.get(tr.recording_id) or "").strip() or NO_VALUE)
        if view.split_by_subject:
            parts.append(tr.subject)
        groups.setdefault(" · ".join(parts), []).append(tr)
    return sorted(groups.items(), key=lambda kv: _order_key(kv[0]))


def subject_order(labels) -> dict[str, int]:
    """``{label: identity index}``, sorted by label. Built from **every** animal the run has, not
    the ones plotted, so leaving an animal out never repaints the others (colour follows the
    animal)."""
    return {label: i for i, label in enumerate(sorted(set(labels), key=_order_key))}


def _by_recording(
    traces: list[Trace], *, per_animal: bool = False
) -> list[list[Trace]]:
    """Traces gathered per recording (or per recording and animal), in the order they came."""
    out: dict[tuple, list[Trace]] = {}
    for tr in traces:
        key = (tr.recording_id, tr.object_id) if per_animal else (tr.recording_id,)
        out.setdefault(key, []).append(tr)
    return list(out.values())


def _shared_aspect(traces: list[Trace]) -> float | None:
    """The frames' common height/width, or None when the recordings disagree."""
    ratios = {round(t.frame.height / t.frame.width, 4) for t in traces}
    return ratios.pop() if len(ratios) == 1 else None


def _trace_title(tr: Trace) -> str:
    return f"{tr.name} · {tr.subject}" if tr.show_subject else tr.name


def _members_title(group: str, members: list) -> str:
    return f"{group} (n={len(members)})"


def _animals(members: list[Trace]) -> list[tuple[str, int]]:
    """The distinct animals behind a group's traces (several points of one animal are one)."""
    return sorted({(t.recording_id, t.object_id) for t in members})


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build_plot(
    traces: list[Trace],
    view: LocationPlotView,
    group_values: dict[str, str] | None = None,
    *,
    order: dict[str, int] | None = None,
    notes: list[str] | None = None,
) -> PlotSpec:
    """Describe ``view`` of ``traces`` (one per animal per recording) as a ``PlotSpec``.

    Without grouping, a path plot draws every chosen animal of a recording in one panel, and a
    heat map or speed plot gives each animal its own panel. Grouped (``view.group_by`` over
    ``group_values`` = ``{recording_id: value}``, and/or ``view.split_by_subject``), a panel is a
    group and every animal in it counts once. ``order`` is the run-wide ``subject_order`` (default:
    the plotted animals'); ``notes`` name what was left out, shown under the title.

    Raises:
        ValueError: No traces, ``group_by`` set without values, or ``mm`` with an uncalibrated
            recording.
    """
    if not traces:
        detail = f" ({'; '.join(notes)})" if notes else ""
        raise ValueError(f"nothing to plot: no animals selected{detail}")
    if view.group_by is not None and group_values is None:
        raise ValueError("group_by is set but no group values were given")
    units = resolve_units(traces, view)
    order = dict(
        order or subject_order(t.subject for t in traces if t.subject_labelled)
    )
    base = len(order)
    for (
        t
    ) in traces:  # an animal the run never named takes an identity by its track number
        order.setdefault(t.subject, base + t.object_id)
    grouped = (
        group_traces(traces, group_values or {}, view) if is_grouped(view) else None
    )
    keys = ([view.group_by] if view.group_by else []) + (
        ["animal"] if view.split_by_subject else []
    )
    title = _TITLES[view.kind] + (f" by {' × '.join(keys)}" if keys else "")
    note = f"Not plotted: {'; '.join(notes)}" if notes else ""
    if view.kind == "speed":
        return _speed_plot(traces, view, units, grouped, title, order, note)
    if view.kind == "heatmap":
        return _heat_plot(traces, view, units, grouped, title, note)
    return _path_plot(traces, view, units, grouped, title, order, note)


def _extent(traces: list[Trace], units: str) -> tuple[float, float]:
    sizes = [t.frame.extent(units) for t in traces]
    return max(s[0] for s in sizes), max(s[1] for s in sizes)


def _position_aspect(traces: list[Trace], units: str) -> float:
    if units != "norm":
        return 1.0
    return _shared_aspect(traces) or 1.0


def color_mode(view: LocationPlotView, panels: list[list[Trace]]) -> str:
    """What colours a path: ``view.color_by``, or for ``auto`` the animal when any panel holds
    more than one, otherwise time."""
    if view.color_by != "auto":
        return view.color_by
    several = any(len({t.subject for t in members}) > 1 for members in panels)
    return "subject" if several else "time"


def _color_scale(traces: list[Trace], mode: str, view, units: str) -> ColorScale | None:
    if mode in ("subject", "point"):  # identity colours, no scale
        return None
    if mode == "time":
        t_end = max((float(t.t_s[-1]) for t in traces if len(t.t_s)), default=0.0)
        return ColorScale("elapsed time (s)", 0.0, t_end or 1.0)
    speeds = np.concatenate([t.speed(units) for t in traces])
    speeds = speeds[np.isfinite(speeds)]
    top = float(np.percentile(speeds, view.speed_clip_pct)) if len(speeds) else 0.0
    return ColorScale(f"speed ({units}/s)", 0.0, top or 1.0)


def _path_line(tr: Trace, mode: str, view, units: str, three_d: bool, order) -> Line:
    x, y = tr.xy(units)
    if mode == "point":
        out = drawn(tr, [x, y] + ([tr.t_s] if three_d else []), view.max_points)
        return Line(
            x=out[0],
            y=out[1],
            z=out[2] if three_d else None,
            identity=view.points.index(tr.point),
            label=tr.point,
        )
    if mode == "subject":
        out = drawn(tr, [x, y] + ([tr.t_s] if three_d else []), view.max_points)
        return Line(
            x=out[0],
            y=out[1],
            z=out[2] if three_d else None,
            identity=order[tr.subject],
            label=tr.subject,
        )
    values = tr.t_s if mode == "time" else tr.speed(units)
    out = drawn(tr, [x, y, values] + ([tr.t_s] if three_d else []), view.max_points)
    return Line(
        x=out[0],
        y=out[1],
        z=out[3] if three_d else None,
        values=out[2],
        label=_trace_title(tr),
    )


def _path_plot(traces, view, units, grouped, title, order, note) -> PlotSpec:
    three_d = view.kind == "trace3d"
    several_points = len(position_points(view)) > 1
    if grouped is None:
        members = _by_recording(traces, per_animal=several_points)
        titles = [
            _trace_title(m[0]) if len(m) == 1 or several_points else m[0].name
            for m in members
        ]
    else:
        members = [m for _g, m in grouped]
        titles = [_members_title(g, _animals(m)) for g, m in grouped]
    # Several points are told apart by colour, so colour cannot also tell animals apart.
    mode = "point" if several_points else color_mode(view, members)
    panels = [
        Panel(
            title_, lines=[_path_line(t, mode, view, units, three_d, order) for t in m]
        )
        for title_, m in zip(titles, members)
    ]
    width, height = _extent(traces, units)
    t_end = max((float(t.t_s[-1]) for t in traces if len(t.t_s)), default=0.0)
    return PlotSpec(
        kind=view.kind,
        title=title,
        panels=panels,
        x=Axis(f"x ({units})", (0.0, width)),
        y=Axis(f"y ({units})", (0.0, height)),
        z=Axis("time (s)", (0.0, t_end or 1.0)) if three_d else None,
        color=_color_scale(traces, mode, view, units),
        camera=Camera(view.elev, view.azim) if three_d else None,
        aspect=_position_aspect(traces, units),
        invert_y=True,
        legend=mode in ("subject", "point"),
        note=note,
    )


def _heat_plot(traces, view, units, grouped, title, note) -> PlotSpec:
    width, height = _extent(traces, units)
    if units == "norm":
        aspect = _shared_aspect(traces)
        extent_for_cells = (1.0, aspect) if aspect else (1.0, 1.0)
        x_edges, y_edges = cell_edges(extent_for_cells, view.bins)
        y_edges = y_edges / (y_edges[-1] or 1.0)  # back onto 0..1 of the frame height
    else:
        x_edges, y_edges = cell_edges((width, height), view.bins)

    def grid_of(tr: Trace) -> np.ndarray:
        return occupancy(*tr.xy(units), x_edges, y_edges)

    if grouped is None:
        panels = [
            Panel(_trace_title(t), heat=Heat(grid_of(t), x_edges, y_edges))
            for t in traces
        ]
    else:
        # Each animal's grid sums to 1 before the mean, so each counts once (decision 5).
        panels = [
            Panel(
                _members_title(group, members),
                heat=Heat(
                    np.mean([grid_of(t) for t in members], axis=0), x_edges, y_edges
                ),
            )
            for group, members in grouped
        ]
    top = max((float(p.heat.grid.max()) for p in panels), default=0.0)
    return PlotSpec(
        kind="heatmap",
        title=title,
        panels=panels,
        x=Axis(f"x ({units})", (0.0, width)),
        y=Axis(f"y ({units})", (0.0, height)),
        color=ColorScale("share of time per cell", 0.0, top or 1.0),
        aspect=_position_aspect(traces, units),
        invert_y=True,
        note=note,
    )


def _speed_plot(traces, view, units, grouped, title, order, note) -> PlotSpec:
    t_end = max((float(t.t_s[-1]) for t in traces if len(t.t_s)), default=0.0)
    binned = {
        (t.recording_id, t.object_id): binned_speed(t, units, view.time_bin_s, t_end)
        for t in traces
    }
    y_label = f"speed ({units}/s)"
    if grouped is None:
        panels = [
            Panel(
                _trace_title(t),
                lines=[
                    Line(
                        *binned[(t.recording_id, t.object_id)],
                        identity=order[t.subject],
                        label=t.subject,
                    )
                ],
            )
            for t in traces
        ]
        legend = False
    else:
        by_animal_only = view.split_by_subject and view.group_by is None
        lines, bands = [], []
        for index, (group, members) in enumerate(grouped):
            identity = order[members[0].subject] if by_animal_only else index
            centres = binned[(members[0].recording_id, members[0].object_id)][0]
            mean, sem, _n = mean_sem(
                [binned[(t.recording_id, t.object_id)][1] for t in members]
            )
            lines.append(
                Line(
                    centres,
                    mean,
                    identity=identity,
                    label=_members_title(group, members),
                )
            )
            if len(members) > 1:
                bands.append(Band(centres, mean - sem, mean + sem, identity=identity))
        panels = [Panel("", lines=lines, bands=bands, note="mean ± SEM across animals")]
        legend = True
    tops = [
        float(np.nanmax(line.y))
        for p in panels
        for line in p.lines
        if np.isfinite(line.y).any()
    ] + [
        float(np.nanmax(band.hi))
        for p in panels
        for band in p.bands
        if np.isfinite(band.hi).any()
    ]
    top = max(tops, default=0.0)
    return PlotSpec(
        kind="speed",
        title=title,
        panels=panels,
        x=Axis("time (s)", (0.0, t_end or 1.0)),
        y=Axis(y_label, (0.0, top * 1.05 or 1.0)),
        legend=legend,
        note=note,
    )


# --------------------------------------------------------------------------- #
# Distance between tracks (§16 "Location plots" — Distance between animals)
# --------------------------------------------------------------------------- #
#: How two labels are joined into a pair's name.
PAIR_JOIN = " – "


@dataclass(frozen=True)
class PairSeries:
    """The distance between two tracks of one recording, on the frames both are tracked on.

    ``dist_norm`` / ``dist_px`` are NaN for a frame the measurement has no value on (an
    unmeasured edge). ``label`` is the pair's name, its animals' labels sorted.
    """

    recording_id: str
    name: str
    label: str
    object_ids: tuple[int, int]
    frame: FrameInfo
    t_s: np.ndarray
    dist_norm: np.ndarray | None
    dist_px: np.ndarray

    def dist(self, units: str) -> np.ndarray:
        """The distance in ``units``.

        Raises:
            ValueError: ``norm`` of a distance measured only in pixels (a mask edge).
        """
        if units == "px":
            return self.dist_px
        if units == "mm":
            return self.dist_px / self.frame.pixels_per_mm
        if self.dist_norm is None:
            raise ValueError(
                "a mask-edge distance is measured in pixels; plot it in px or mm"
            )
        return self.dist_norm


def pair_label(a: str, b: str) -> str:
    """``Intruder – Resident``: the two labels, sorted the way animals are ordered."""
    first, second = sorted((a, b), key=_order_key)
    return f"{first}{PAIR_JOIN}{second}"


def read_pair_edges(
    pairs_csv: Path | str,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None:
    """``{(object_id_a, object_id_b): (frame_idx, edge_px)}`` from a ``pair_relations.csv``,
    or None when the file is absent or the export did not measure edges (the column is blank).
    """
    path = Path(pairs_csv)
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("mask_edge_dist_px") not in (None, "")]
    if not rows:
        return None
    grouped: dict[tuple[int, int], list] = {}
    for r in rows:
        key = (int(r["object_id_a"]), int(r["object_id_b"]))
        grouped.setdefault(key, []).append(
            (int(r["frame_idx"]), float(r["mask_edge_dist_px"]))
        )
    out = {}
    for key, values in grouped.items():
        values.sort()
        arr = np.asarray(values, dtype=np.float64)
        out[key] = (arr[:, 0].astype(np.int64), arr[:, 1])
    return out


def recording_pairs(
    table: TrackTable,
    frame: FrameInfo,
    labels: dict[int, str] | None,
    view: LocationPlotView,
    *,
    recording_id: str,
    name: str,
    edges: dict | None = None,
) -> tuple[list[PairSeries], list[str]]:
    """Every pair of chosen animals in one recording as a distance series, and a note for
    what is left out.

    A centroid distance comes from ``tracks.csv`` itself (both animals' centroids on the frames
    both are on — the formula the export writes into ``pair_relations.csv``), so every run has
    it. A mask-edge distance only exists in ``pair_relations.csv`` (``edges``).

    Raises:
        ValueError: ``distance_by`` is ``mask_edge`` on an export that did not measure it.
    """
    if view.distance_by == "mask_edge" and edges is None:
        raise ValueError(
            "this run has no mask-edge distances; export it again with "
            "pair_edge_distance, or plot the distance between centroids"
        )
    present, _missing = subjects_of(table, labels)
    wanted = None if view.subjects is None else set(view.subjects)
    chosen = [s for s in present if wanted is None or s.label in wanted]
    if len(chosen) < 2:
        return [], [f"{name}: fewer than two animals, no pair"]
    t0 = table.first_t_ns()
    source = "bbox" if view.distance_by == "bbox_centroid" else "mask"
    pairs, notes = [], []
    for i, a in enumerate(chosen):
        for b in chosen[i + 1 :]:
            label = pair_label(a.label, b.label)
            if view.pairs is not None and label not in view.pairs:
                continue
            lo, hi = sorted((a.object_id, b.object_id))
            ca, cb = table.columns[lo], table.columns[hi]
            both, ia, ib = np.intersect1d(
                ca["frame_idx"], cb["frame_idx"], return_indices=True
            )
            t_s = (ca["t_ns"][ia] - t0).astype(np.float64) / 1e9
            if view.distance_by == "mask_edge":
                edge = edges.get((lo, hi))
                if edge is None:
                    notes.append(f"{name} · {label}: no mask-edge distance")
                    continue
                at = {int(f): d for f, d in zip(*edge)}
                dist_px = np.asarray([at.get(int(f), np.nan) for f in both])
                dist_norm = None
            else:
                dx = ca[f"{source}_cx_norm"][ia] - cb[f"{source}_cx_norm"][ib]
                dy = ca[f"{source}_cy_norm"][ia] - cb[f"{source}_cy_norm"][ib]
                dist_norm = np.hypot(dx, dy)
                dist_px = np.hypot(dx * frame.width, dy * frame.height)
            pairs.append(
                PairSeries(
                    recording_id=recording_id,
                    name=name,
                    label=label,
                    object_ids=(lo, hi),
                    frame=frame,
                    t_s=t_s,
                    dist_norm=dist_norm,
                    dist_px=dist_px,
                )
            )
    if not pairs and not notes and view.pairs is not None:
        notes.append(f"{name}: none of {', '.join(view.pairs)}")
    return pairs, notes


def binned_mean(
    t_s: np.ndarray, values: np.ndarray, bin_s: float, t_end: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(bin centres, mean value per bin)``; a bin with no finite value is NaN (a gap)."""
    n_bins = max(1, int(np.ceil(t_end / bin_s))) if t_end > 0 else 1
    centres = (np.arange(n_bins) + 0.5) * bin_s
    ok = np.isfinite(values)
    which = np.minimum((t_s[ok] / bin_s).astype(np.int64), n_bins - 1)
    total = np.bincount(which, weights=values[ok], minlength=n_bins)
    count = np.bincount(which, minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        return centres, np.where(count > 0, total / count, np.nan)


def share_histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The share of ``values`` in each bin (sums to 1 over the finite ones; zeros if none)."""
    finite = values[np.isfinite(values)]
    counts, _ = np.histogram(np.clip(finite, edges[0], edges[-1]), bins=edges)
    return counts / counts.sum() if counts.sum() else counts.astype(np.float64)


def distance_units(pairs: list[PairSeries], view: LocationPlotView) -> str:
    """Units of a distance plot: the auto rule (a distance is a scalar, like speed)."""
    uncalibrated = sorted({p.name for p in pairs if not p.frame.pixels_per_mm})
    if view.units is None:
        return "px" if uncalibrated else "mm"
    if view.units == "mm" and uncalibrated:
        raise ValueError(
            f"mm needs a distance calibration; not calibrated: {', '.join(uncalibrated)}"
        )
    return view.units


def build_distance_plot(
    pairs: list[PairSeries],
    view: LocationPlotView,
    group_values: dict[str, str] | None = None,
    *,
    order: dict[str, int] | None = None,
    notes: list[str] | None = None,
) -> PlotSpec:
    """Distance between tracks over time (``distance``) or as a histogram
    (``distance_hist``): a panel per recording with a line per pair, or one axes with a line
    per group — the mean across pairs, each pair counting once, ± SEM.

    Raises:
        ValueError: No pairs, ``group_by`` without values, or units the measure cannot give.
    """
    if not pairs:
        detail = f" ({'; '.join(notes)})" if notes else ""
        raise ValueError(f"nothing to plot: no pair of animals selected{detail}")
    if view.group_by is not None and group_values is None:
        raise ValueError("group_by is set but no group values were given")
    units = distance_units(pairs, view)
    order = dict(
        order
        or {
            label: i
            for i, label in enumerate(sorted({p.label for p in pairs}, key=_order_key))
        }
    )
    for p in pairs:
        order.setdefault(p.label, len(order))
    series = {id(p): p.dist(units) for p in pairs}
    hist = view.kind == "distance_hist"
    if hist:
        top = max(
            (float(np.nanmax(v)) for v in series.values() if np.isfinite(v).any()),
            default=0.0,
        )
        edges = np.linspace(0.0, top or 1.0, view.hist_bins + 1)
        x = (edges[:-1] + edges[1:]) / 2.0
        curves = {id(p): (x, share_histogram(series[id(p)], edges)) for p in pairs}
    else:
        t_end = max((float(p.t_s[-1]) for p in pairs if len(p.t_s)), default=0.0)
        curves = {
            id(p): binned_mean(p.t_s, series[id(p)], view.time_bin_s, t_end)
            for p in pairs
        }
    measure = {
        "mask_centroid": "mask centroids",
        "bbox_centroid": "box centroids",
        "mask_edge": "mask edges",
    }[view.distance_by]
    keys = ([view.group_by] if view.group_by else []) + (
        ["pair"] if view.split_by_subject else []
    )
    title = ("Distance between tracks" if not hist else "Distances between tracks") + (
        f" by {' × '.join(keys)}" if keys else ""
    )
    grouped = view.group_by is not None or view.split_by_subject
    if not grouped:
        panels = []
        for members in _pairs_by_recording(pairs):
            panels.append(
                Panel(
                    members[0].name,
                    lines=[
                        Line(
                            *curves[id(p)],
                            identity=order[p.label],
                            label=p.label,
                            step=hist,
                        )
                        for p in members
                    ],
                )
            )
    else:
        groups: dict[str, list[PairSeries]] = {}
        for p in pairs:
            parts = []
            if view.group_by is not None:
                parts.append(
                    ((group_values or {}).get(p.recording_id) or "").strip() or NO_VALUE
                )
            if view.split_by_subject:
                parts.append(p.label)
            groups.setdefault(" · ".join(parts), []).append(p)
        by_pair_only = view.split_by_subject and view.group_by is None
        lines, bands = [], []
        for index, (group, members) in enumerate(
            sorted(groups.items(), key=lambda kv: _order_key(kv[0]))
        ):
            identity = order[members[0].label] if by_pair_only else index
            x_values = curves[id(members[0])][0]
            mean, sem, _n = mean_sem([curves[id(p)][1] for p in members])
            lines.append(
                Line(
                    x_values,
                    mean,
                    identity=identity,
                    label=_members_title(group, members),
                    step=hist,
                )
            )
            if len(members) > 1:
                bands.append(
                    Band(x_values, mean - sem, mean + sem, identity=identity, step=hist)
                )
        panels = [Panel("", lines=lines, bands=bands, note="mean ± SEM across pairs")]
    tops = [
        float(np.nanmax(arr))
        for panel in panels
        for arr in [ln.y for ln in panel.lines] + [b.hi for b in panel.bands]
        if np.isfinite(arr).any()
    ]
    top = max(tops, default=0.0)
    dist_axis = f"distance between {measure} ({units})"
    if not hist:
        # A distance-over-time panel is short: its y label names the units, and the title
        # names what is measured.
        title += f": {measure}"
    x_axis = (
        Axis(dist_axis, (0.0, float(edges[-1])))
        if hist
        else Axis(
            "time (s)",
            (
                0.0,
                max((float(p.t_s[-1]) for p in pairs if len(p.t_s)), default=1.0)
                or 1.0,
            ),
        )
    )
    y_axis = Axis(
        "share of frames" if hist else f"distance ({units})", (0.0, top * 1.05 or 1.0)
    )
    return PlotSpec(
        kind=view.kind,
        title=title,
        panels=panels,
        x=x_axis,
        y=y_axis,
        legend=True,
        note=f"Not plotted: {'; '.join(notes)}" if notes else "",
    )


def _pairs_by_recording(pairs: list[PairSeries]) -> list[list[PairSeries]]:
    out: dict[str, list[PairSeries]] = {}
    for p in pairs:
        out.setdefault(p.recording_id, []).append(p)
    return list(out.values())
