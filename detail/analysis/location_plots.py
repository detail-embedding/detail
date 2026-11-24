"""Location plots — a location export's paths, occupancy, speed and body, as a ``PlotSpec``.

Pure numpy over the run's own CSVs (DESIGN_PLAN §16 "Location plots"): ``read_tracks`` reads a
``tracks.csv`` once, ``read_frame`` the frame its ``locations.yaml`` records; ``trace`` cuts the
animal's path out of it; ``build_plot`` turns a set of paths and a ``LocationPlotView`` into a
library-free ``report/plot_spec.PlotSpec``. Nothing here draws.

**One animal per recording.** Every tracker this tool has follows one animal, so a recording's
path is its one track. A ``tracks.csv`` with several (an imported multi-animal tracking) is not
guessed at: the recording is named under the title as not plotted.

**One path rule.** Where the path breaks, and how fast the animal went, are a comparison's own
(``analysis/location_measures``): the path is broken at a gap in tracking and at a step faster
than ``jump_speed`` body lengths a second, and speed is measured on its running mean over
``smooth_s``. So a speed plot and a comparison's speed measure one path, and a tracking jump is
neither drawn as a line across the arena nor counted as movement. The rule is applied in pixels,
where a body length means one thing in both directions, and the result converted to the plot's
units.

**The body** (§16 *The body*) is the run's per-frame centre, length, width, long axis and area,
all measures of the whole mask. A body plot reads those columns as they were measured: a run
exported with thin parts removed, an option the export no longer has, is drawn as it measured
them and says so under the title (``api/location_plots``), and a run made before them has none,
and is named as not plotted rather than measured some other way.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detail.analysis import location_measures
from detail.analysis.locations import sampled_indices
from detail.configs.location_plots import LocationPlotView
from detail.storage import location_export
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

#: The group a recording with no value in the ``group_by`` column falls into — never dropped.
NO_VALUE = "(no value)"

#: The plot kinds drawn from the body's measures rather than from its path.
BODY_KINDS = ("body", "body_hist")
#: The positions ``tracks.csv`` carries for every run; any other ``source`` names a keypoint.
CENTROID_SOURCES = ("body", "mask", "bbox")

#: Each position kind's title; ``{of}`` is the point it follows (:func:`followed`).
_TITLES = {
    "trace": "Path{of}",
    "heatmap": "Occupancy{of}",
    "speed": "Speed{of} over time",
    "trace3d": "Path{of} over time",
}
#: What a title says a centroid position is; the body, the default, goes unsaid.
_CENTROID_NAMES = {"body": "", "mask": "the mask's centre", "bbox": "the box's centre"}


def followed(view: LocationPlotView) -> str:
    """The point a position plot follows, as its title says it: ``" of the nose"``,
    ``" of the tail_base and nose"`` for several on a path, or ``""`` for the body.

    Every position plot was titled the same whatever it followed, so an exported path of the
    nose read as the body's.
    """
    if view.points and view.kind in ("trace", "trace3d"):
        names = [f"the {name}" for name in view.points]
        joined = (
            ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]
        )
        return f" of {joined}"
    if view.source in _CENTROID_NAMES:
        name = _CENTROID_NAMES[view.source]
        return f" of {name}" if name else ""
    return f" of the {view.source}"


#: ``body_measure`` → (what it is called, how its unit is written from the length unit).
_MEASURES = {
    "length": ("Body length", "{u}"),
    "width": ("Body width", "{u}"),
    "area": ("Body area", "{u}²"),
    "elongation": ("Body elongation", "length ÷ width"),
    "mask_area": ("Mask area", "{u}²"),
}

_KEY_COLUMNS = ("frame_idx", "t_ns", "object_id")
_CENTRE_COLUMNS = tuple(
    f"{src}_c{axis}_norm" for src in ("mask", "bbox") for axis in ("x", "y")
)
#: The body's columns plots read; all arrived together (§16 *The body*).
BODY_COLUMNS = (
    "body_cx_norm",
    "body_cy_norm",
    "body_length_px",
    "body_width_px",
    "body_angle_deg",
    "body_area_px",
)
#: Points on one drawn body outline.
_OUTLINE_POINTS = 37


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

    def length_scale(self, units: str) -> float:
        """What one pixel of length is in ``units`` (``px`` or ``mm``)."""
        return 1.0 if units == "px" else 1.0 / self.pixels_per_mm


@dataclass(frozen=True)
class TrackTable:
    """One ``tracks.csv``'s columns plots read, per ``object_id``, in frame order.

    ``has_body`` is False for an export made before the body was measured: its rows have no
    ``body_*`` columns.
    """

    columns: dict[int, dict[str, np.ndarray]]
    has_body: bool


def read_frame(manifest: dict, where: str) -> FrameInfo:
    """The frame size and calibration a ``locations.yaml`` recorded (its contents, as
    ``storage/location_export.read_location_manifest`` reads them; ``where`` names it).

    Raises:
        ValueError: If the manifest has no video resolution (normalized coordinates cannot be
            turned into pixels without one).
    """
    resolution = manifest.get("video_resolution")
    if not resolution:
        raise ValueError(f"{where} records no video_resolution")
    ppm = manifest.get("pixels_per_mm")
    return FrameInfo(float(resolution[0]), float(resolution[1]), ppm or None)


def _floats(values: list[str]) -> np.ndarray:
    """A CSV column as floats, a blank cell (a measure with no value on that row) as NaN."""
    return np.asarray([float(v) if v != "" else np.nan for v in values])


def read_tracks(tracks_csv: Path | str) -> TrackTable:
    """Read the columns plots use from a ``tracks.csv``, by name, split by ``object_id``.

    Read through the export's own reader (``storage/location_export.read_location_csv``), as a
    comparison reads it.

    Raises:
        ValueError: If a column every export has is missing.
    """
    header, table_rows = location_export.read_location_csv(tracks_csv)
    missing = [c for c in _KEY_COLUMNS + _CENTRE_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{tracks_csv} has no column(s) {missing}")
    has_body = all(c in header for c in BODY_COLUMNS)
    wanted = (
        _KEY_COLUMNS
        + _CENTRE_COLUMNS
        + (BODY_COLUMNS + ("mask_area_px",) if has_body else ())
    )
    index = [header.index(c) for c in wanted]
    rows = [[row[i] for i in index] for row in table_rows if row]
    if not rows:
        return TrackTable({}, has_body)
    raw = list(zip(*rows))
    frame_idx = np.asarray(raw[0], dtype=np.int64)
    t_ns = np.asarray(raw[1], dtype=np.int64)
    object_id = np.asarray(raw[2], dtype=np.int64)
    values = {name: _floats(list(col)) for name, col in zip(wanted[3:], raw[3:])}
    columns: dict[int, dict[str, np.ndarray]] = {}
    for oid in np.unique(object_id):
        rows_of = np.flatnonzero(object_id == oid)
        rows_of = rows_of[np.argsort(frame_idx[rows_of], kind="stable")]
        cols = {"frame_idx": frame_idx[rows_of], "t_ns": t_ns[rows_of]}
        cols.update({name: col[rows_of] for name, col in values.items()})
        columns[int(oid)] = cols
    return TrackTable(columns, has_body)


@dataclass(frozen=True)
class Body:
    """The body's per-sample measures (§16 *The body*), in pixels and degrees."""

    cx_norm: np.ndarray
    cy_norm: np.ndarray
    length_px: np.ndarray
    width_px: np.ndarray
    angle_deg: np.ndarray
    area_px: np.ndarray
    mask_area_px: np.ndarray


@dataclass(frozen=True)
class Trace:
    """One recording's animal: its path at full resolution in normalized units, and its body
    when the export measured one."""

    recording_id: str
    name: str
    frame: FrameInfo
    t_s: np.ndarray
    x_norm: np.ndarray
    y_norm: np.ndarray
    body: Body | None
    #: The keypoint this path follows, or ``None`` for a centre from ``tracks.csv``.
    point: str | None = None

    def xy(self, units: str) -> tuple[np.ndarray, np.ndarray]:
        """The path's coordinates in ``units``."""
        w, h = self.frame.extent(units)
        return self.x_norm * w, self.y_norm * h

    def _placed(self) -> np.ndarray:
        """Indices of the samples with a position: all of a centre's; a keypoint's, less
        the frames it was not placed or was under the cut-off (each a gap)."""
        return np.flatnonzero(np.isfinite(self.x_norm) & np.isfinite(self.y_norm))

    def _body_length_px(self, rows: np.ndarray) -> np.ndarray:
        return self.body.length_px[rows] if self.body is not None else np.zeros(0)

    def breaks(self, view: LocationPlotView) -> np.ndarray:
        """``breaks[i]``: sample ``i`` is the first after a gap or a jump. A sample with no
        position is a gap of its own; the drawing breaks there by itself."""
        out = np.zeros(len(self.t_s), dtype=bool)
        rows = self._placed()
        x, y = self.xy("px")
        out[rows] = location_measures.path_breaks(
            self.t_s[rows],
            x[rows],
            y[rows],
            self._body_length_px(rows),
            view.jump_speed,
        )
        return out

    def _stretches(
        self, view: LocationPlotView, units: str
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """``[(rows, step times, step lengths, step durations)]`` of each unbroken stretch of
        the smoothed path, lengths in ``units``: the comparison's own path
        (``location_measures.smoothed_segments``), so its speed is the speed a comparison
        measures. The rule runs in pixels, over the samples with a position, and the
        stretches are scaled per axis after. ``rows`` index this trace's samples.
        """
        rows = self._placed()
        x, y = self.xy("px")
        found = location_measures.smoothed_segments(
            self.t_s[rows],
            x[rows],
            y[rows],
            self._body_length_px(rows),
            view.smooth_s,
            view.jump_speed,
        )
        w, h = self.frame.extent(units)
        sx, sy = w / self.frame.width, h / self.frame.height
        return [
            (
                rows[s.rows],
                (s.t_s[:-1] + s.t_s[1:]) / 2.0,
                np.hypot(np.diff(s.x) * sx, np.diff(s.y) * sy),
                np.diff(s.t_s),
            )
            for s in found or []
        ]

    def steps(
        self, view: LocationPlotView, units: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(t_s, distance, dt)`` of every step of the smoothed path, in ``units``."""
        stretches = self._stretches(view, units)
        if not stretches:
            empty = np.zeros(0)
            return empty, empty, empty
        return tuple(
            np.concatenate([s[k] for s in stretches]) for k in (1, 2, 3)
        )  # type: ignore[return-value]

    def speed(self, view: LocationPlotView, units: str) -> np.ndarray:
        """Speed in units/s at each sample, read off its stretch of the smoothed path; NaN
        for a sample in no stretch (one too short to smooth, or with no position)."""
        out = np.full(len(self.t_s), np.nan)
        for rows, t, dist, dt in self._stretches(view, units):
            with np.errstate(divide="ignore", invalid="ignore"):
                speed = dist / dt
            out[rows] = np.interp(self.t_s[rows], t, speed)
        return out


def tracked_animal(table: TrackTable, name: str) -> tuple[int | None, str | None]:
    """``(object_id, None)`` for a recording's one animal, or ``(None, note)`` saying why it has
    none to plot: no tracked rows, or more than one animal."""
    if not table.columns:
        return None, f"{name}: no tracked rows"
    if len(table.columns) > 1:
        ids = ", ".join(str(oid) for oid in sorted(table.columns))
        return None, (
            f"{name}: {len(table.columns)} animals (tracks {ids}); location plots show one "
            "animal per recording"
        )
    return next(iter(table.columns)), None


@dataclass(frozen=True)
class KeypointTable:
    """One ``keypoints.csv``: ``{(object_id, point): (frame_idx, t_ns, x_norm, y_norm)}`` in
    frame order, a point that was not placed or is under the cut-off (``visible`` 0) as NaN —
    a gap, as §26 *Visibility is a view* draws it."""

    series: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def read_keypoints(keypoints_csv: Path | str) -> KeypointTable:
    """Read a ``keypoints.csv`` by column name, through the export's own reader, into
    per-animal, per-point series.

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
    header, table_rows = location_export.read_location_csv(keypoints_csv)
    missing = [c for c in wanted if c not in header]
    if missing:
        raise ValueError(f"{keypoints_csv} has no column(s) {missing}")
    index = [header.index(c) for c in wanted]
    grouped: dict[tuple[int, str], list] = {}
    for row in table_rows:
        if not row:
            continue
        frame, t, oid, point, x, y, visible = (row[i] for i in index)
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
        frames = np.asarray([v[0] for v in values], dtype=np.int64)
        t_ns = np.asarray([v[1] for v in values], dtype=np.int64)
        xy = np.asarray([v[2:] for v in values], dtype=np.float64)
        series[key] = (frames, t_ns, xy[:, 0], xy[:, 1])
    return KeypointTable(series)


def _body_of(cols: dict, rows: np.ndarray | None = None) -> Body:
    pick = (lambda a: a) if rows is None else (lambda a: a[rows])
    return Body(
        cx_norm=pick(cols["body_cx_norm"]),
        cy_norm=pick(cols["body_cy_norm"]),
        length_px=pick(cols["body_length_px"]),
        width_px=pick(cols["body_width_px"]),
        angle_deg=pick(cols["body_angle_deg"]),
        area_px=pick(cols["body_area_px"]),
        mask_area_px=pick(cols["mask_area_px"]),
    )


def trace(
    table: TrackTable,
    frame: FrameInfo,
    view: LocationPlotView,
    *,
    recording_id: str,
    name: str,
    keypoints: KeypointTable | None = None,
    point: str | None = None,
) -> tuple[Trace | None, str | None]:
    """The recording's animal as a ``Trace`` timed from its first tracked sample, positioned by
    ``view.source`` — or, with ``point``, by that keypoint's positions in ``keypoints``; or
    ``(None, note)`` when there is none to plot.

    ``body`` reads the body's centre, and on an export made before the body was measured the
    mask's centre, which is the same point: the body is the whole mask (§16), and such an
    export never removed thin parts. A keypoint path shares the track's clock and its body,
    read on the frames the keypoint has (``keypoints.csv`` is keyed as ``tracks.csv``).
    """
    oid, note = tracked_animal(table, name)
    if oid is None:
        return None, note
    cols = table.columns[oid]
    t0 = cols["t_ns"][0]
    if point is not None:
        found = keypoints.series.get((oid, point)) if keypoints is not None else None
        if found is None:
            return None, f"{name} · {point}: no keypoint rows"
        frames, t_ns, x, y = found
        at = np.searchsorted(cols["frame_idx"], frames)
        on_track = (at < len(cols["frame_idx"])) & (
            cols["frame_idx"][np.minimum(at, len(cols["frame_idx"]) - 1)] == frames
        )
        body = (
            _body_of(cols, at)
            if table.has_body and on_track.all() and len(frames)
            else None
        )
        return (
            Trace(
                recording_id=recording_id,
                name=name,
                frame=frame,
                t_s=(t_ns - t0).astype(np.float64) / 1e9,
                x_norm=x,
                y_norm=y,
                body=body,
                point=point,
            ),
            None,
        )
    source = view.source if view.source in CENTROID_SOURCES else "body"
    if source == "body" and not table.has_body:
        source = "mask"
    return (
        Trace(
            recording_id=recording_id,
            name=name,
            frame=frame,
            t_s=(cols["t_ns"] - t0).astype(np.float64) / 1e9,
            x_norm=cols[f"{source}_cx_norm"],
            y_norm=cols[f"{source}_cy_norm"],
            body=_body_of(cols) if table.has_body else None,
        ),
        None,
    )


def position_points(view: LocationPlotView) -> list[str]:
    """The keypoints a view draws positions from: ``points`` on a path plot, else ``source``
    when it names a keypoint; empty for a centre, and for a body plot, which reads the body.
    """
    if view.kind in BODY_KINDS:
        return []
    if view.points and view.kind in ("trace", "trace3d"):
        return list(view.points)
    if view.source not in CENTROID_SOURCES:
        return [view.source]
    return []


def recording_traces(
    table: TrackTable,
    frame: FrameInfo,
    view: LocationPlotView,
    *,
    recording_id: str,
    name: str,
    keypoints: KeypointTable | None = None,
    keypoint_names: list[str] | None = None,
) -> tuple[list[Trace], list[str]]:
    """One recording's paths — its animal's centre, or one per keypoint the view asks for —
    and a note for each that could not be drawn.

    Raises:
        ValueError: A keypoint is asked for that the run's skeleton does not have, or the run
            was exported without keypoints.
    """
    points = position_points(view)
    if not points:
        found, note = trace(table, frame, view, recording_id=recording_id, name=name)
        return ([found] if found is not None else []), ([note] if note else [])
    known = list(keypoint_names or [])
    if keypoints is None or not known:
        raise ValueError(
            f"{', '.join(points)}: this run was exported without a keypoint track run; "
            "use body, mask or bbox, or export again with keypoints"
        )
    unknown = [p for p in points if p not in known]
    if unknown:
        raise ValueError(
            f"no keypoint {', '.join(unknown)} in this run (it has {', '.join(known)})"
        )
    traces, notes = [], []
    for point in points:
        found, note = trace(
            table,
            frame,
            view,
            recording_id=recording_id,
            name=name,
            keypoints=keypoints,
            point=point,
        )
        if found is not None:
            traces.append(found)
        if note is not None and note not in notes:
            notes.append(note)
    return traces, notes


# --------------------------------------------------------------------------- #
# Aggregates (pure)
# --------------------------------------------------------------------------- #
def subsample(n: int, cap: int) -> np.ndarray:
    """At most ``cap`` evenly spaced indices into ``range(n)``, always keeping both ends."""
    if n <= cap:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, cap).round().astype(np.int64))


def drawn(
    tr: Trace, columns: list[np.ndarray], view: LocationPlotView
) -> list[np.ndarray]:
    """``columns`` (per-sample arrays of ``tr``) thinned to ``view.max_points``, with a NaN
    row inserted wherever a break falls between two kept samples."""
    keep = subsample(len(tr.t_s), view.max_points)
    breaks = np.cumsum(tr.breaks(view))
    split = np.flatnonzero(np.diff(breaks[keep]) > 0) + 1
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


def _time_bins(bin_s: float, t_end: float) -> tuple[int, np.ndarray]:
    n_bins = max(1, int(np.ceil(t_end / bin_s))) if t_end > 0 else 1
    return n_bins, (np.arange(n_bins) + 0.5) * bin_s


def binned_speed(
    tr: Trace, view: LocationPlotView, units: str, t_end: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(bin centres, speed)``: distance travelled on the smoothed path in each bin ÷ the time
    it took. A step belongs to the bin its time falls in; a bin with no step is NaN, so a
    dropout reads as missing rather than as standing still."""
    n_bins, centres = _time_bins(view.time_bin_s, t_end)
    t, dist, dt = tr.steps(view, units)
    if not len(t):
        return centres, np.full(n_bins, np.nan)
    which = np.minimum((t / view.time_bin_s).astype(np.int64), n_bins - 1)
    dist_sum = np.bincount(which, weights=dist, minlength=n_bins)
    time_sum = np.bincount(which, weights=dt, minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(time_sum > 0, dist_sum / time_sum, np.nan)
    return centres, speed


def binned_mean(
    t_s: np.ndarray, values: np.ndarray, bin_s: float, t_end: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(bin centres, mean value per bin)``; a bin with no finite value is NaN (a gap)."""
    n_bins, centres = _time_bins(bin_s, t_end)
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


def body_values(tr: Trace, measure: str, units: str) -> np.ndarray:
    """One body measure per sample, in ``units`` (``px`` or ``mm``; squared for an area)."""
    body = tr.body
    if measure == "elongation":
        with np.errstate(divide="ignore", invalid="ignore"):
            return body.length_px / body.width_px
    scale = tr.frame.length_scale(units)
    if measure in ("area", "mask_area"):
        area = body.area_px if measure == "area" else body.mask_area_px
        return area * scale * scale
    return (body.length_px if measure == "length" else body.width_px) * scale


def posture_outlines(
    tr: Trace, view: LocationPlotView, units: str, colour: np.ndarray | None
) -> Line | None:
    """The body's outline every ``view.posture_every_s`` seconds as one ``Line``: an ellipse
    as long and as wide as the body along its long axis, at the body's centre, each outline
    broken from the next by a NaN row and coloured as the path is at that moment.

    Drawn in pixels, where the long axis's angle was measured, and converted to ``units`` point
    by point, so an outline is the body's shape whatever the plot's units and axis direction.
    The moments are the samples a clock at ``1 / posture_every_s`` keeps by the export's own
    sampling rule (``analysis/locations.sampled_indices``).
    """
    body = tr.body
    if body is None or view.posture_every_s is None or not len(tr.t_s):
        return None
    at = sampled_indices(
        (tr.t_s * 1e9).astype(np.int64), 1.0 / float(view.posture_every_s)
    )
    phi = np.linspace(0.0, 2.0 * np.pi, _OUTLINE_POINTS)
    w, h = tr.frame.extent(units)
    sx, sy = w / tr.frame.width, h / tr.frame.height
    xs, ys, vs = [], [], []
    for i in at:
        half_l, half_w = body.length_px[i] / 2.0, body.width_px[i] / 2.0
        theta = np.radians(body.angle_deg[i])
        cx, cy = body.cx_norm[i] * tr.frame.width, body.cy_norm[i] * tr.frame.height
        if not np.isfinite([half_l, half_w, theta, cx, cy]).all():
            continue
        ex = half_l * np.cos(phi)
        ey = half_w * np.sin(phi)
        px = cx + ex * np.cos(theta) - ey * np.sin(theta)
        py = cy + ex * np.sin(theta) + ey * np.cos(theta)
        xs.append(np.r_[px * sx, np.nan])
        ys.append(np.r_[py * sy, np.nan])
        value = colour[i] if colour is not None else np.nan
        vs.append(np.r_[np.full(_OUTLINE_POINTS, value), np.nan])
    if not xs:
        return None
    return Line(
        x=np.concatenate(xs),
        y=np.concatenate(ys),
        values=np.concatenate(vs) if colour is not None else None,
        label="body outline",
    )


# --------------------------------------------------------------------------- #
# Units and groups
# --------------------------------------------------------------------------- #
def is_grouped(view: LocationPlotView) -> bool:
    """Whether panels pool animals across recordings."""
    return view.group_by is not None


def _length_units(traces: list[Trace], view: LocationPlotView, what: str) -> str:
    """Units of a speed, a length or an area: the view's, or auto (mm when every recording is
    calibrated, else px).

    Raises:
        ValueError: ``mm`` with an uncalibrated recording, or ``norm`` for a body measure.
    """
    uncalibrated = sorted({t.name for t in traces if not t.frame.pixels_per_mm})
    if view.units is None:
        return "px" if uncalibrated else "mm"
    if view.units == "mm" and uncalibrated:
        raise ValueError(
            f"mm needs a distance calibration; not calibrated: {', '.join(uncalibrated)}"
        )
    if view.units == "norm" and what == "body":
        raise ValueError(
            "a body is measured in px or mm, not in fractions of the frame"
        )
    return view.units


def resolve_units(traces: list[Trace], view: LocationPlotView) -> str:
    """The units a plot is drawn in (DESIGN_PLAN §16 "Location plots").

    Positions pooled across recordings (a by-group path or heat map) are always ``norm``. A
    speed or a body measure is a scalar and takes the length rule. Else the view's choice, or —
    unset — ``mm`` when every recording is calibrated, ``px`` otherwise.

    Raises:
        ValueError: ``mm`` was asked for and a plotted recording is uncalibrated, or ``norm``
            for a body measure.
    """
    if view.kind in BODY_KINDS:
        return _length_units(traces, view, "body")
    if view.kind == "speed":
        return _length_units(traces, view, "speed")
    if is_grouped(view):
        return "norm"
    return _length_units(traces, view, "position")


def _order_key(text: str) -> tuple[bool, str]:
    return (NO_VALUE in text, text.lower())


def group_traces(
    traces: list[Trace], values: dict[str, str]
) -> list[tuple[str, list[Trace]]]:
    """``[(group, animals)]`` by each recording's Recording Information value, sorted with
    ``(no value)`` last."""
    groups: dict[str, list[Trace]] = {}
    for tr in traces:
        value = (values.get(tr.recording_id) or "").strip() or NO_VALUE
        groups.setdefault(value, []).append(tr)
    return sorted(groups.items(), key=lambda kv: _order_key(kv[0]))


def _shared_aspect(traces: list[Trace]) -> float | None:
    """The frames' common height/width, or None when the recordings disagree."""
    ratios = {round(t.frame.height / t.frame.width, 4) for t in traces}
    return ratios.pop() if len(ratios) == 1 else None


def _members_title(group: str, members: list) -> str:
    return f"{group} (n={len(members)})"


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def build_plot(
    traces: list[Trace],
    view: LocationPlotView,
    group_values: dict[str, str] | None = None,
    *,
    notes: list[str] | None = None,
) -> PlotSpec:
    """Describe ``view`` of ``traces`` (one animal per recording) as a ``PlotSpec``.

    Ungrouped, each recording is a panel. Grouped (``view.group_by`` over ``group_values`` =
    ``{recording_id: value}``), a panel or a line is a group, and every animal in it counts
    once. ``notes`` name what was left out, shown under the title.

    Raises:
        ValueError: Nothing to plot, ``group_by`` set without values, units the measure cannot
            be given in, or ``mm`` with an uncalibrated recording.
    """
    notes = list(notes or [])
    if view.kind in BODY_KINDS:
        measured = [t for t in traces if t.body is not None]
        notes += [
            f"{t.name}: exported before the body was measured; export it again"
            for t in traces
            if t.body is None
        ]
        traces = measured
    if not traces:
        detail = f" ({'; '.join(notes)})" if notes else ""
        raise ValueError(f"nothing to plot{detail}")
    if view.group_by is not None and group_values is None:
        raise ValueError("group_by is set but no group values were given")
    units = resolve_units(traces, view)
    grouped = group_traces(traces, group_values or {}) if is_grouped(view) else None
    if view.kind in BODY_KINDS:
        return _body_plot(traces, view, units, grouped, notes)
    title = _TITLES[view.kind].format(of=followed(view))
    title += f" by {view.group_by}" if grouped else ""
    note = f"Not plotted: {'; '.join(notes)}" if notes else ""
    if view.kind == "speed":
        return _speed_plot(traces, view, units, grouped, title, note)
    if view.kind == "heatmap":
        return _heat_plot(traces, view, units, grouped, title, note)
    return _path_plot(traces, view, units, grouped, title, note)


def _extent(traces: list[Trace], units: str) -> tuple[float, float]:
    sizes = [t.frame.extent(units) for t in traces]
    return max(s[0] for s in sizes), max(s[1] for s in sizes)


def _position_aspect(traces: list[Trace], units: str) -> float:
    if units != "norm":
        return 1.0
    return _shared_aspect(traces) or 1.0


def _color_scale(
    traces: list[Trace], view: LocationPlotView, units: str, speeds
) -> ColorScale:
    if view.color_by == "time":
        t_end = max((float(t.t_s[-1]) for t in traces if len(t.t_s)), default=0.0)
        return ColorScale("elapsed time (s)", 0.0, t_end or 1.0)
    every = np.concatenate([speeds[id(t)] for t in traces])
    every = every[np.isfinite(every)]
    top = float(np.percentile(every, view.speed_clip_pct)) if len(every) else 0.0
    speed_units = units if units != "norm" else "norm"
    return ColorScale(f"speed ({speed_units}/s)", 0.0, top or 1.0)


def _path_plot(traces, view, units, grouped, title, note) -> PlotSpec:
    three_d = view.kind == "trace3d"
    # Several keypoints are told apart by colour, so colour cannot also say time or speed.
    by_point = len(position_points(view)) > 1
    if grouped is None:
        order = list(dict.fromkeys(t.recording_id for t in traces))
        members = [[t for t in traces if t.recording_id == sid] for sid in order]
        titles = [m[0].name for m in members]
    else:
        members = [m for _g, m in grouped]
        titles = [_members_title(g, {t.recording_id for t in m}) for g, m in grouped]
    speeds = (
        {id(t): t.speed(view, units) for t in traces}
        if view.color_by == "speed" and not by_point
        else {}
    )

    def colour_of(tr: Trace) -> np.ndarray | None:
        if by_point:
            return None
        return tr.t_s if view.color_by == "time" else speeds[id(tr)]

    def path(tr: Trace) -> Line:
        x, y = tr.xy(units)
        if by_point:
            out = drawn(tr, [x, y] + ([tr.t_s] if three_d else []), view)
            return Line(
                x=out[0],
                y=out[1],
                z=out[2] if three_d else None,
                identity=view.points.index(tr.point),
                label=tr.point,
            )
        cols = [x, y, colour_of(tr)] + ([tr.t_s] if three_d else [])
        out = drawn(tr, cols, view)
        return Line(
            x=out[0],
            y=out[1],
            z=out[3] if three_d else None,
            values=out[2],
            label=tr.name,
        )

    panels = []
    for title_, m in zip(titles, members):
        lines = [path(t) for t in m]
        if not three_d:
            # One outline per animal, whichever of its paths it is drawn beside.
            first = {}
            for t in m:
                first.setdefault(t.recording_id, t)
            lines += [
                outline
                for t in first.values()
                if (outline := posture_outlines(t, view, units, colour_of(t)))
                is not None
            ]
        panels.append(Panel(title_, lines=lines))
    missing_body = list(dict.fromkeys(t.name for t in traces if t.body is None))
    if view.posture_every_s is not None and not three_d and missing_body:
        extra = f"no body outline: {', '.join(missing_body)} (exported before the body)"
        note = f"{note}; {extra}" if note else f"Not plotted: {extra}"
    width, height = _extent(traces, units)
    t_end = max((float(t.t_s[-1]) for t in traces if len(t.t_s)), default=0.0)
    return PlotSpec(
        kind=view.kind,
        title=title,
        panels=panels,
        x=Axis(f"x ({units})", (0.0, width)),
        y=Axis(f"y ({units})", (0.0, height)),
        z=Axis("time (s)", (0.0, t_end or 1.0)) if three_d else None,
        color=None if by_point else _color_scale(traces, view, units, speeds),
        camera=Camera(view.elev, view.azim) if three_d else None,
        aspect=_position_aspect(traces, units),
        invert_y=True,
        legend=by_point,
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
            Panel(t.name, heat=Heat(grid_of(t), x_edges, y_edges)) for t in traces
        ]
    else:
        # Each animal's grid sums to 1 before the mean, so an animal tracked longer than its
        # group-mates does not outweigh them: the group's map is its animals', each once.
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


def _curves_plot(
    *,
    kind: str,
    title: str,
    curves: dict[int, tuple[np.ndarray, np.ndarray]],
    traces: list[Trace],
    grouped,
    x_axis: Axis,
    y_label: str,
    step: bool,
    note: str,
) -> PlotSpec:
    """A curve per recording (a panel each), or a mean ± SEM curve per group on one axes."""
    if grouped is None:
        panels = [
            Panel(t.name, lines=[Line(*curves[id(t)], identity=0, step=step)])
            for t in traces
        ]
        legend = False
    else:
        lines, bands = [], []
        for index, (group, members) in enumerate(grouped):
            x_values = curves[id(members[0])][0]
            mean, sem, _n = mean_sem([curves[id(t)][1] for t in members])
            lines.append(
                Line(
                    x_values,
                    mean,
                    identity=index,
                    label=_members_title(group, members),
                    step=step,
                )
            )
            if len(members) > 1:
                bands.append(
                    Band(x_values, mean - sem, mean + sem, identity=index, step=step)
                )
        panels = [Panel("", lines=lines, bands=bands, note="mean ± SEM across animals")]
        legend = True
    tops = [
        float(np.nanmax(arr))
        for panel in panels
        for arr in [ln.y for ln in panel.lines] + [b.hi for b in panel.bands]
        if np.isfinite(arr).any()
    ]
    top = max(tops, default=0.0)
    return PlotSpec(
        kind=kind,
        title=title,
        panels=panels,
        x=x_axis,
        y=Axis(y_label, (0.0, top * 1.05 or 1.0)),
        legend=legend,
        note=note,
    )


def _speed_plot(traces, view, units, grouped, title, note) -> PlotSpec:
    t_end = max((float(t.t_s[-1]) for t in traces if len(t.t_s)), default=0.0)
    curves = {id(t): binned_speed(t, view, units, t_end) for t in traces}
    return _curves_plot(
        kind="speed",
        title=title,
        curves=curves,
        traces=traces,
        grouped=grouped,
        x_axis=Axis("time (s)", (0.0, t_end or 1.0)),
        y_label=f"speed ({units}/s)",
        step=False,
        note=note,
    )


def _body_plot(traces, view, units, grouped, notes) -> PlotSpec:
    name, unit_pattern = _MEASURES[view.body_measure]
    unit = unit_pattern.format(u=units)
    values = {id(t): body_values(t, view.body_measure, units) for t in traces}
    hist = view.kind == "body_hist"
    by = f" by {view.group_by}" if grouped else ""
    note = f"Not plotted: {'; '.join(notes)}" if notes else ""
    if hist:
        finite = [v[np.isfinite(v)] for v in values.values()]
        every = np.concatenate(finite) if finite else np.zeros(0)
        lo = float(every.min()) if len(every) else 0.0
        hi = float(every.max()) if len(every) else 1.0
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, view.hist_bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2.0
        curves = {
            key: (centres, share_histogram(v, edges)) for key, v in values.items()
        }
        return _curves_plot(
            kind="body_hist",
            title=f"{name}{by}",
            curves=curves,
            traces=traces,
            grouped=grouped,
            x_axis=Axis(f"{name.lower()} ({unit})", (lo, hi)),
            y_label="share of samples",
            step=True,
            note=note,
        )
    t_end = max((float(t.t_s[-1]) for t in traces if len(t.t_s)), default=0.0)
    curves = {
        id(t): binned_mean(t.t_s, values[id(t)], view.time_bin_s, t_end) for t in traces
    }
    return _curves_plot(
        kind="body",
        title=f"{name} over time{by}",
        curves=curves,
        traces=traces,
        grouped=grouped,
        x_axis=Axis("time (s)", (0.0, t_end or 1.0)),
        y_label=f"{name.lower()} ({unit})",
        step=False,
        note=note,
    )
