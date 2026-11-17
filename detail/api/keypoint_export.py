"""Keypoint export writers (§26 "Analysis and export"): the DETAIL long table, DeepLabCut,
SLEAP and NWB, over one reader of a keypoint run.

Every target in `export_targets` that writes points goes through :class:`PointSource`, so
the cut-off, the ``source`` choice (the settled track run or the detect run's own placements)
and the "blank below the cut-off" option are decided once and every format agrees. The
formats are written from the same ``(frame_idx, t_ns, points, point_scores, changed)`` rows;
a missing point is NaN in each of them, never zero.

A track run's rows are per **subject** (``track_<id>``). A detect run's are **unassigned**
(§26, revised 2026-10-21): a pose instance belongs to no subject until a track run gives it
one, so the export never invents one. Its rows are grouped by **instance index per frame** —
the frame's first instance is ``instance_0``, its second ``instance_1`` — which is what
DeepLabCut and SLEAP call untracked individuals; the long table leaves ``track_id`` empty and
writes the ``instance`` column instead.

The optional writers import their backend inside the function (``pandas`` + ``tables`` for the
DeepLabCut H5, ``h5py`` for SLEAP, ``pynwb`` + ``ndx-pose`` for NWB), so the package imports
without the ``keypoints_export`` extra and a target without its backend says so rather than
failing mid-export.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from detail.core.skeleton import Skeleton
from detail.storage import recording as recording_store
from detail.storage.keypoint_runs import KeypointDetectRun, KeypointTrackRun

SOURCES = ("track", "detect")

#: The DETAIL long table (§26 *Analysis and export*): one row per frame × track × point.
LONG_COLUMNS = [
    "recording_id",
    "track_id",
    "part_id",
    "frame_idx",
    "t_ns",
    "keypoint",
    "x_px",
    "y_px",
    "x_norm",
    "y_norm",
    "score",
    "visible",
    "changed",
]
#: The long table of a detect run's unassigned instances: ``track_id`` stays (empty), and
#: ``instance`` — the instance's index on its frame — follows it.
INSTANCE_LONG_COLUMNS = LONG_COLUMNS[:2] + ["instance"] + LONG_COLUMNS[2:]

TIMESTAMPS_NAME = "timestamps.csv"
COVERAGE_NAME = "coverage.csv"

#: What a method's score is *not*: the sidecar and every format's own field say so.
LIKELIHOOD_NOTE = (
    "likelihood is the method's own point score, not a calibrated probability; it is "
    "comparable within one run and never across methods"
)


def default_cutoff(detect: KeypointDetectRun) -> float | None:
    """The detect method's display cut-off (``score_spec.default_cutoff``), or ``None``."""
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)
    spec = getattr(_method_class(detect), "score_spec", None)
    cutoff = getattr(spec, "default_cutoff", None)
    return None if cutoff is None else float(cutoff)


def _method_class(detect: KeypointDetectRun):
    """The detect run's method class, a trained student's included (``models.resolve``)."""
    from detail.models.resolve import keypoint_method_class

    return keypoint_method_class(
        detect.recording_dir.parent.parent, detect.meta.get("model_id")
    )


@dataclass
class PartRows:
    """One series of points in one part — a track's, or a detect run's ``instance_<i>`` —
    already filtered the way the export asked for."""

    frame_idx: np.ndarray  # (M,) int
    t_ns: np.ndarray  # (M,) int64
    points: (
        np.ndarray
    )  # (M, K, 2) float64, normalised [0, 1], NaN where absent or blanked
    scores: np.ndarray  # (M, K) float64, NaN where absent
    visible: np.ndarray  # (M, K) bool: placed and at or above the cut-off
    changed: np.ndarray  # (M, K) int8: 0 as detected · 1 another candidate · 2 moved


class PointSource:
    """One keypoint run read for export: a track run (the default), or a detect run — either
    the run itself, or the detect run behind a track run when ``source="detect"``.

    Rows come in **series**: a track run's are its subjects (``track_<id>``), a detect run's
    its unassigned instances by index on the frame (``instance_<i>``; see the module
    docstring). ``series_ids`` / ``rows`` / ``rows_in`` / ``at`` take a series id either way.

    ``cutoff`` ``None`` takes the detect method's display cut-off. ``blank_below_cutoff``
    writes a point under the cut-off as NaN in every format; off, its coordinates are kept and
    only ``visible`` says it is under (§26 *Visibility*: a view, never a filter).
    """

    def __init__(
        self,
        recording_dir: Path | str,
        run_id: str,
        *,
        source: str = "track",
        cutoff: float | None = None,
        blank_below_cutoff: bool = False,
    ) -> None:
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, not {source!r}")
        self.recording_dir = Path(recording_dir)
        self.recording = recording_store.read_recording(self.recording_dir)
        self.run_id = run_id
        #: The tracking run whose subjects the rows belong to; None for unassigned instances.
        self.tracking_run_id: str | None = None
        if run_id.startswith("kt_"):
            track = KeypointTrackRun(self.recording_dir, run_id)
            self.detect = KeypointDetectRun(
                self.recording_dir, track.meta["detect_run_id"]
            )
            self.kind = "detect" if source == "detect" else "track"
            self.run = self.detect if source == "detect" else track
            if self.kind == "track":
                self.tracking_run_id = str(track.meta["tracking_run_id"])
                self.method = str(track.meta["method"])
            else:
                self.method = str(self.detect.meta.get("model_id") or "")
        else:
            self.detect = KeypointDetectRun(self.recording_dir, run_id)
            self.kind = "detect"
            self.run = self.detect
            self.method = str(self.detect.meta.get("model_id") or "")
        self._instance_series: dict[str, dict[int, PartRows]] = {}
        self.skeleton: Skeleton = self.run.skeleton
        self.cutoff = default_cutoff(self.detect) if cutoff is None else float(cutoff)
        self.blank_below_cutoff = bool(blank_below_cutoff)
        video = self.recording.get("video") or {}
        resolution = video.get("resolution") or (0, 0)
        self.frame_wh = (int(resolution[0]), int(resolution[1]))
        self._timestamps: dict[str, np.ndarray] = {}

    # -- what was run ------------------------------------------------------------- #
    @property
    def recording_id(self) -> str:
        return str(self.recording.get("id") or self.recording_dir.name)

    @property
    def recording_name(self) -> str:
        return str(self.recording.get("name") or self.recording_id)

    @property
    def scorer(self) -> str:
        """DeepLabCut's ``scorer`` / NWB's: the tool, the method and the run."""
        return f"DETAIL_{self.method}_{self.run.run_id}"

    @property
    def score_meaning(self) -> str:
        from detail.api._runner_modules import PIPELINE_RUNNERS
        from detail.executor.runners import import_runner_modules

        import_runner_modules(PIPELINE_RUNNERS)
        spec = getattr(_method_class(self.detect), "score_spec", None)
        return str(getattr(spec, "meaning", None) or "the method's point score")

    @property
    def assigned(self) -> bool:
        """Whether the rows belong to subjects (a track run) or are unassigned instances."""
        return self.kind == "track"

    @property
    def long_columns(self) -> list[str]:
        return LONG_COLUMNS if self.assigned else INSTANCE_LONG_COLUMNS

    def series_name(self, series_id: int) -> str:
        """``track_<id>`` for a subject, ``instance_<i>`` for an unassigned instance index:
        DeepLabCut's individual, SLEAP's track name, NWB's ``PoseEstimation``."""
        return f"{'track' if self.assigned else 'instance'}_{int(series_id)}"

    def series_ids(self) -> list[int]:
        """The track ids, or the instance indices any frame reaches."""
        if self.assigned:
            return list(self.run.track_ids)
        found: set[int] = set()
        for part_id in self.part_ids():
            found.update(self._instances(part_id))
        return sorted(found)

    def series_parts(self, series_id: int) -> list[str]:
        """The parts one series has rows in."""
        if self.assigned:
            return self.run.part_ids(series_id)
        return [p for p in self.part_ids() if int(series_id) in self._instances(p)]

    def part_ids(self) -> list[str]:
        """Every part with rows, in play order."""
        if not self.assigned:
            return self.detect.part_ids()
        parts: set[str] = set()
        for track_id in self.run.track_ids:
            parts.update(self.run.part_ids(track_id))
        return sorted(parts)

    def part_stems(self) -> list[str]:
        """The recording's video parts, in play order (what frame counts come from)."""
        return [
            recording_store.part_stem(part)
            for part in recording_store.video_files(self.recording)
        ]

    def frame_count(self, part_id: str) -> int:
        """Frames in a part, from its timestamp map (CLAUDE.md: never from cv2)."""
        return len(self.timestamps(part_id))

    def timestamps(self, part_id: str) -> np.ndarray:
        """``t_ns`` per frame of a part."""
        if part_id not in self._timestamps:
            index = self.part_stems().index(part_id)
            tmap = recording_store.load_active_timestamps(
                self.recording_dir, self.recording, index
            )
            self._timestamps[part_id] = np.asarray(tmap.values, dtype=np.int64)
        return self._timestamps[part_id]

    def fps(self, part_id: str) -> float:
        """The recording's measured rate, else the part's timestamps, else 30."""
        fps = (self.recording.get("video") or {}).get("fps_actual")
        if fps:
            return float(fps)
        t_ns = self.timestamps(part_id)
        if len(t_ns) > 1 and t_ns[-1] > t_ns[0]:
            return float((len(t_ns) - 1) / ((int(t_ns[-1]) - int(t_ns[0])) / 1e9))
        return 30.0

    def coverage(self) -> dict:
        return self.run.coverage()

    def row_count(self, series_id: int, part_id: str) -> int:
        """Rows one series has in a part — a track's from the coverage the run recorded
        (``total``, else every frame of the part, which over-counts rather than under), an
        instance index's counted off the part's ``frame_idx`` column alone."""
        if self.assigned:
            recorded = self.coverage().get(f"track_{series_id}", {}).get(part_id, {})
            return int(recorded.get("total", self.frame_count(part_id)))
        frames = np.asarray(self.detect.reader(part_id).column("frame_idx")[:])
        return int((_instance_index(frames) == int(series_id)).sum())

    # -- the rows ----------------------------------------------------------------- #
    def rows(self, series_id: int, part_id: str) -> PartRows:
        if not self.assigned:
            return self._instances(part_id)[int(series_id)]
        return self._part_rows(self.run.rows(series_id, part_id))

    def _instances(self, part_id: str) -> dict[int, PartRows]:
        """A detect part's instances grouped by their index on the frame."""
        if part_id not in self._instance_series:
            raw = self.detect.rows(part_id)
            index = _instance_index(np.asarray(raw["frame_idx"]))
            self._instance_series[part_id] = {
                int(i): self._part_rows(
                    {
                        name: np.asarray(raw[name])[index == i]
                        for name in ("frame_idx", "t_ns", "points", "point_scores")
                    }
                )
                for i in np.unique(index).tolist()
            }
        return self._instance_series[part_id]

    def _part_rows(self, raw: dict) -> PartRows:
        points = np.asarray(raw["points"], dtype=np.float64)
        scores = np.asarray(raw["point_scores"], dtype=np.float64)
        placed = np.isfinite(points).all(axis=2)
        scores = np.where(placed, scores, np.nan)
        if self.cutoff is None:
            visible = placed
        else:
            visible = placed & (np.nan_to_num(scores, nan=-np.inf) >= self.cutoff)
        if self.blank_below_cutoff:
            points = np.where(visible[..., None], points, np.nan)
        changed = raw.get("changed")
        if changed is None:  # the detect run: everything is as detected
            changed = np.zeros(placed.shape, dtype=np.int8)
        return PartRows(
            frame_idx=np.asarray(raw["frame_idx"]).astype(np.int64),
            t_ns=np.asarray(raw["t_ns"]).astype(np.int64),
            points=points,
            scores=scores,
            visible=visible,
            changed=np.asarray(changed, dtype=np.int8),
        )

    def rows_in(self, part_id: str) -> dict[int, PartRows]:
        """Every series' rows in one part."""
        if not self.assigned:
            return dict(self._instances(part_id)) if part_id in self.part_ids() else {}
        return {
            track_id: self.rows(track_id, part_id)
            for track_id in self.run.track_ids
            if part_id in self.run.part_ids(track_id)
        }

    def at(self, series_id: int, part_id: str, frame_idx: int):
        """``(points (K, 2), scores (K,), visible (K,))`` at a frame, or ``None``."""
        if not self.assigned:
            rows = self._instances(part_id).get(int(series_id))
            if rows is None:
                return None
            i = int(np.searchsorted(rows.frame_idx, frame_idx))
            if i >= len(rows.frame_idx) or int(rows.frame_idx[i]) != int(frame_idx):
                return None
            return rows.points[i], rows.scores[i], rows.visible[i]
        found = self.run.at(series_id, part_id, frame_idx)
        if found is None:
            return None
        points, scores = (np.asarray(found[0], dtype=np.float64), np.asarray(found[1]))
        placed = np.isfinite(points).all(axis=1)
        if self.cutoff is None:
            visible = placed
        else:
            visible = placed & (np.nan_to_num(scores, nan=-np.inf) >= self.cutoff)
        if self.blank_below_cutoff:
            points = np.where(visible[:, None], points, np.nan)
        return points, scores, visible

    def to_px(self, points: np.ndarray) -> np.ndarray:
        return points * np.asarray(self.frame_wh, dtype=np.float64)


def _instance_index(frame_idx: np.ndarray) -> np.ndarray:
    """Each detect row's index among the rows of its frame (rows are written in frame
    order): ``[0, 0, 1, 0, 1, 2]`` for frames ``[0, 1, 1, 2, 2, 2]``."""
    frame_idx = np.asarray(frame_idx)
    if not len(frame_idx):
        return np.zeros(0, np.int64)
    starts = np.r_[True, frame_idx[1:] != frame_idx[:-1]]
    first = np.maximum.accumulate(np.where(starts, np.arange(len(frame_idx)), 0))
    return np.arange(len(frame_idx)) - first


# --------------------------------------------------------------------------- #
# The DETAIL long table and the sidecars every format ships
# --------------------------------------------------------------------------- #
def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not np.isfinite(value) else repr(float(value))
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    return str(value)


def write_csv(path: Path, columns: list[str], rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(v) for v in row])


def long_rows(source: PointSource):
    """The long table's rows for every (series, part) of a source, in order, in
    ``source.long_columns``: a track run's carry the ``track_id``; a detect run's leave it
    empty and carry the ``instance`` index on the frame."""
    names = source.skeleton.names
    w, h = source.frame_wh
    for part_id in source.part_ids():
        for series_id, rows in source.rows_in(part_id).items():
            who = [int(series_id)] if source.assigned else [None, int(series_id)]
            for i in range(len(rows.frame_idx)):
                for k, name in enumerate(names):
                    x, y = (float(rows.points[i, k, 0]), float(rows.points[i, k, 1]))
                    placed = bool(np.isfinite(x) and np.isfinite(y))
                    yield [
                        source.recording_id,
                        *who,
                        part_id,
                        int(rows.frame_idx[i]),
                        int(rows.t_ns[i]),
                        name,
                        x * w if placed else np.nan,
                        y * h if placed else np.nan,
                        x if placed else np.nan,
                        y if placed else np.nan,
                        float(rows.scores[i, k]),
                        bool(rows.visible[i, k]),
                        int(rows.changed[i, k]),
                    ]


def write_long_csv(path: Path, sources: list[PointSource]) -> Path:
    """One long CSV over every source given (one recording, or all of a run's); the sources
    of one export are all assigned or all unassigned (one ``source`` option)."""

    def _rows():
        for source in sources:
            yield from long_rows(source)

    columns = sources[0].long_columns if sources else LONG_COLUMNS
    write_csv(path, columns, _rows())
    return path


def write_timestamps(path: Path, sources: list[PointSource]) -> Path:
    """``recording_id, part_id, frame_idx, t_ns`` for every frame of every part the points
    were written for: the time DLC and SLEAP files cannot carry."""

    def _rows():
        for source in sources:
            for part_id in source.part_ids():
                t_ns = source.timestamps(part_id)
                for frame_idx, t in enumerate(t_ns.tolist()):
                    yield [source.recording_id, part_id, frame_idx, int(t)]

    write_csv(path, ["recording_id", "part_id", "frame_idx", "t_ns"], _rows())
    return path


def write_coverage(path: Path, sources: list[PointSource]) -> Path:
    """The run's own coverage per track and part, plus each point's count at the cut-off.

    A detect run's instances belong to no track: one row per part (``placed`` the frames with
    an instance, ``total`` the frames walked, as the run recorded them; ``track_id`` empty),
    then each point's count over every instance of the part."""
    columns = [
        "recording_id",
        "run_id",
        "track_id",
        "part_id",
        "placed",
        "total",
        "keypoint",
        "visible",
    ]

    def _rows():
        for source in sources:
            coverage = source.coverage()
            if not source.assigned:
                yield from _instance_coverage(source, coverage)
                continue
            for part_id in source.part_ids():
                for track_id, rows in source.rows_in(part_id).items():
                    recorded = coverage.get(f"track_{track_id}", {}).get(part_id, {})
                    placed = recorded.get("placed", int(rows.visible.any(axis=1).sum()))
                    total = recorded.get("total", len(rows.frame_idx))
                    yield [
                        source.recording_id,
                        source.run.run_id,
                        int(track_id),
                        part_id,
                        int(placed),
                        int(total),
                        "",
                        "",
                    ]
                    for k, name in enumerate(source.skeleton.names):
                        yield [
                            source.recording_id,
                            source.run.run_id,
                            int(track_id),
                            part_id,
                            "",
                            "",
                            name,
                            int(rows.visible[:, k].sum()),
                        ]

    write_csv(path, columns, _rows())
    return path


def _instance_coverage(source: PointSource, coverage: dict):
    for part_id in source.part_ids():
        series = source.rows_in(part_id)
        recorded = coverage.get(part_id, {})
        frames = (
            np.unique(np.concatenate([r.frame_idx for r in series.values()]))
            if series
            else np.zeros(0)
        )
        yield [
            source.recording_id,
            source.run.run_id,
            None,
            part_id,
            int(recorded.get("frames_with_instance", len(frames))),
            int(recorded.get("frames", source.frame_count(part_id))),
            "",
            "",
        ]
        for k, name in enumerate(source.skeleton.names):
            yield [
                source.recording_id,
                source.run.run_id,
                None,
                part_id,
                "",
                "",
                name,
                int(sum(int(r.visible[:, k].sum()) for r in series.values())),
            ]


# --------------------------------------------------------------------------- #
# DeepLabCut: multi-animal CSV (+ H5 with the extra), one per video part
# --------------------------------------------------------------------------- #
def _dense_part(source: PointSource, part_id: str):
    """``(series ids, xy_px (F, T, K, 2), likelihood (F, T, K))`` over every frame of the
    part, NaN where a series has no point — the per-frame layout DLC and SLEAP share. A
    detect run's series are its instance indices: DLC's and SLEAP's untracked individuals.
    """
    n_frames = source.frame_count(part_id)
    by_track = source.rows_in(part_id)
    track_ids = sorted(by_track)
    k = len(source.skeleton.names)
    xy = np.full((n_frames, len(track_ids), k, 2), np.nan)
    likelihood = np.full((n_frames, len(track_ids), k), np.nan)
    for t, track_id in enumerate(track_ids):
        rows = by_track[track_id]
        keep = (rows.frame_idx >= 0) & (rows.frame_idx < n_frames)
        frames = rows.frame_idx[keep]
        xy[frames, t] = source.to_px(rows.points[keep])
        likelihood[frames, t] = rows.scores[keep]
    return track_ids, xy, likelihood


def dlc_header(scorer: str, individuals: list[str], bodyparts: list[str]) -> list[list]:
    """DeepLabCut's four multi-animal header rows: scorer / individuals / bodyparts /
    coords, over ``x, y, likelihood`` per (individual, bodypart)."""
    coords = ["x", "y", "likelihood"]
    cells = [(ind, bp, c) for ind in individuals for bp in bodyparts for c in coords]
    return [
        ["scorer"] + [scorer] * len(cells),
        ["individuals"] + [ind for ind, _bp, _c in cells],
        ["bodyparts"] + [bp for _ind, bp, _c in cells],
        ["coords"] + [c for _ind, _bp, c in cells],
    ]


def write_dlc(source: PointSource, part_id: str, csv_path: Path, h5_path: Path) -> list:
    """The DeepLabCut analysis table for one part: the CSV always, the H5 (pandas table
    ``df_with_missing``, as DLC writes it) when ``pandas`` and ``tables`` are installed.
    """
    track_ids, xy, likelihood = _dense_part(source, part_id)
    individuals = [source.series_name(t) for t in track_ids]
    bodyparts = list(source.skeleton.names)
    n_frames = xy.shape[0]
    table = np.concatenate([xy, likelihood[..., None]], axis=3).reshape(n_frames, -1)
    rows = ([frame] + table[frame].tolist() for frame in range(n_frames))
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for header in dlc_header(source.scorer, individuals, bodyparts):
            writer.writerow(header)
        for row in rows:
            writer.writerow(
                [row[0]] + ["" if not np.isfinite(v) else repr(v) for v in row[1:]]
            )
    written = [csv_path]
    try:
        import pandas as pd
        import tables  # noqa: F401  (pandas' HDF writer needs it)
    except ImportError:
        return written
    columns = pd.MultiIndex.from_product(
        [[source.scorer], individuals, bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    frame = pd.DataFrame(table, columns=columns, index=pd.RangeIndex(n_frames))
    frame.to_hdf(h5_path, key="df_with_missing", mode="w", format="table")
    written.append(h5_path)
    return written


# --------------------------------------------------------------------------- #
# SLEAP analysis HDF5, one per video part
# --------------------------------------------------------------------------- #
def sleap_instance_scores(likelihood: np.ndarray) -> np.ndarray:
    """``(F, T)`` instance scores from ``(F, T, K)`` point scores: the mean over the points
    **placed**, NaN where none is.

    A missing point is NaN, and it was set to 0 before the mean, so an animal with two of
    four points placed at 0.9 scored 0.45 (#32). SLEAP's own mean instance score (its
    bottom-up multi-class models, ``nanmean`` of the peak values) skips a missing node too.
    """
    likelihood = np.asarray(likelihood, dtype=np.float64)
    placed = np.isfinite(likelihood)
    count = placed.sum(axis=2)
    total = np.where(placed, likelihood, 0.0).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def write_sleap(source: PointSource, part_id: str, path: Path, video_name: str) -> Path:
    """SLEAP's analysis HDF5 (``sleap-convert --format analysis``), as SLEAP writes it:
    every matrix stored transposed (``tracks`` as ``(n_tracks, 2, n_nodes, n_frames)``,
    ``track_occupancy`` as ``(n_frames, n_tracks)``), names as byte strings."""
    import h5py

    track_ids, xy, likelihood = _dense_part(source, part_id)
    skeleton = source.skeleton
    n_frames = xy.shape[0]
    occupancy = np.isfinite(xy).all(axis=3).any(axis=2)  # (F, T)
    instance_scores = sleap_instance_scores(likelihood)
    edges = [(skeleton.index(a), skeleton.index(b)) for a, b in skeleton.edges]
    provenance = {
        "source_software": "DETAIL",
        "run_id": source.run.run_id,
        "recording_id": source.recording_id,
        "part_id": part_id,
        "scorer": source.scorer,
        "point_scores": LIKELIHOOD_NOTE,
    }
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "track_names",
            data=np.array([source.series_name(t).encode() for t in track_ids]),
        )
        f.create_dataset(
            "node_names", data=np.array([n.encode() for n in skeleton.names])
        )
        f.create_dataset(
            "edge_names",
            data=np.array(
                [[a.encode(), b.encode()] for a, b in skeleton.edges]
            ).reshape(-1, 2),
        )
        f.create_dataset(
            "edge_inds", data=np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        )
        f.create_dataset(
            "tracks", data=np.transpose(xy, (1, 3, 2, 0)), compression="gzip"
        )
        f.create_dataset(
            "track_occupancy", data=occupancy.astype(np.uint8), compression="gzip"
        )
        f.create_dataset(
            "point_scores", data=np.transpose(likelihood, (1, 2, 0)), compression="gzip"
        )
        f.create_dataset("instance_scores", data=instance_scores.T, compression="gzip")
        f.create_dataset(
            "tracking_scores", data=np.full((len(track_ids), n_frames), np.nan)
        )
        f.create_dataset("labels_path", data=np.bytes_(""))
        f.create_dataset("video_path", data=np.bytes_(video_name))
        f.create_dataset("video_ind", data=np.int64(0))
        f.create_dataset("provenance", data=np.bytes_(json.dumps(provenance)))
    return path


# --------------------------------------------------------------------------- #
# NWB + ndx-pose, one per video part
# --------------------------------------------------------------------------- #
def write_nwb(
    source: PointSource,
    part_id: str,
    path: Path,
    *,
    video_name: str,
    experiment_name: str,
    session_start: datetime | None = None,
) -> Path:
    """A ``PoseEstimation`` per track (or per instance index, for a detect run's unassigned
    instances), one ``PoseEstimationSeries`` per point, the skeleton in the behavior module
    (ndx-pose ≥ 0.2's shape). Timestamps from the rows' ``t_ns``."""
    from ndx_pose import PoseEstimation, PoseEstimationSeries
    from ndx_pose import Skeleton as NwbSkeleton
    from ndx_pose import Skeletons
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.image import ImageSeries

    import detail

    skeleton = source.skeleton
    start = session_start or _session_start(source)
    nwbfile = NWBFile(
        session_description=(
            f"DETAIL keypoints: run {source.run.run_id} on recording "
            f"{source.recording_name} ({source.recording_id}), video part {part_id}"
        ),
        identifier=f"{experiment_name}/{source.recording_id}/{source.run.run_id}/{part_id}",
        session_start_time=start,
    )
    w, h = source.frame_wh
    # The source footage as an external-file ImageSeries (ndx-pose ≥ 0.4 links the pose to
    # it, and reads the frame size from it); the file itself stays beside the recording.
    video = ImageSeries(
        name=f"video_{part_id}",
        description=f"video part {part_id} of recording {source.recording_name}",
        external_file=[video_name],
        format="external",
        starting_frame=[0],
        dimension=[w, h],
        rate=source.fps(part_id),
        # The part's first frame on the recording's clock, which is the clock the poses'
        # timestamps use: a later part's map is recording-wide (`add_video_part`), so 0.0
        # put its video one part-length ahead of its poses (issue #23).
        starting_time=float(source.timestamps(part_id)[0]) / 1e9,
        num_samples=source.frame_count(part_id),
        unit="n.a.",
    )
    nwbfile.add_acquisition(video)
    edges = np.asarray(
        [[skeleton.index(a), skeleton.index(b)] for a, b in skeleton.edges],
        dtype=np.uint8,
    ).reshape(-1, 2)
    nwb_skeleton = NwbSkeleton(
        name=f"skeleton_{skeleton.digest()}", nodes=list(skeleton.names), edges=edges
    )
    behavior = nwbfile.create_processing_module(
        "behavior", "keypoints placed by DETAIL (§26)"
    )
    behavior.add(Skeletons(skeletons=[nwb_skeleton]))
    for track_id, rows in source.rows_in(part_id).items():
        px = source.to_px(rows.points)
        t_s = rows.t_ns.astype(np.float64) / 1e9
        who = source.series_name(track_id)
        series = [
            PoseEstimationSeries(
                name=name,
                description=f"{name} of {who}, in pixels",
                data=np.ascontiguousarray(px[:, k, :]),
                unit="pixels",
                reference_frame=(
                    "(0, 0) is the top-left corner of the video frame; x to the right, y "
                    "down, in pixels of the source video"
                ),
                timestamps=t_s,
                confidence=np.ascontiguousarray(rows.scores[:, k]),
                confidence_definition=f"{source.score_meaning}; {LIKELIHOOD_NOTE}",
            )
            for k, name in enumerate(skeleton.names)
        ]
        behavior.add(
            PoseEstimation(
                name=who,
                description=(
                    f"DETAIL {source.kind} run {source.run.run_id}, method "
                    f"{source.method}; a missing point is NaN"
                    + (
                        ""
                        if source.assigned
                        else "; instances are unassigned: instance_<i> is the i-th "
                        "pose on each frame, not one animal"
                    )
                ),
                pose_estimation_series=series,
                source_video=video,
                scorer=source.scorer,
                source_software="DETAIL",
                source_software_version=detail.__version__,
                skeleton=nwb_skeleton,
            )
        )
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    return path


def _session_start(source: PointSource) -> datetime:
    created = str(source.run.meta.get("created_at") or "")
    try:
        start = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError:
        start = datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return start
