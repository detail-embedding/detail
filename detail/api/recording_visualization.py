"""Recording visualization: one recording (or a range) beside its path through pose space and its
location, as a picture per frame, a video, or a montage (DESIGN_PLAN §22 "Recording
visualization").

Three layers, so the window, the CLI and a script share one path:

- :func:`load_data` reads everything a frame needs **except its pixels** through DETAIL's own
  readers — the clustering's trajectory (``pipeline.temporal_clustering.load_trajectory``) pooled
  as the method pooled it, the run's labelled spans (``NodeView.labelled_runs``), and Location
  Export's compute (``analysis.locations.compute_location_tables``) — into a small
  :class:`VizData` that pickles and saves, so worker processes start in a second.
- :class:`FrameReader` reads pixels: a frame for a scrub (``TrackingPreview.frame``, a seek, the
  preview path) or a span in order for an export (``frames_range``, one forward pass — Tenet 11).
- :class:`RecordingVisualization` draws (``api/recording_visualization_draw.Canvas``) and writes:
  ``frame``, ``estimate``, ``export``; :func:`export_montage` tiles several, from any experiment.
"""

from __future__ import annotations

import dataclasses
import functools
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from detail.analysis import recording_visualization as rv
from detail.core.visualize import rgb_to_bgr
from detail.configs.recording_visualization import (
    RecordingVisualizationConfig,
    RecordingVisualizationExport,
    speed_timing,
)

Progress = Callable[[str, float, str], None]

#: Frames sampled to measure an export's time and size before it runs.
ESTIMATE_FRAMES = 6
#: Output frames per worker chunk: small enough that progress moves, large enough that a
#: worker's start-up (imports, opening the video) is a few percent of its work.
CHUNK_FRAMES = 150


def _noop(_stage: str, _fraction: float, _detail: str) -> None:
    return None


# --------------------------------------------------------------------------- the data


@dataclasses.dataclass
class VizData:
    """Everything one visualization's frames need except their pixels. Pickles; saves to npz."""

    experiment_dir: str
    run_id: str
    recipe: dict
    recording_id: str
    recording_name: str
    recording_dir: str
    tracking_run_id: str
    embed_run_id: str
    track_id: int
    size: tuple[int, int]  # video (w, h)
    fps: float
    t_ns: np.ndarray  # (n,) every frame of the recording, recording-global
    first: int  # the visualization's first frame (recording-global)
    last: int  # one past its last frame
    xy: np.ndarray  # (n, 2) pose-space position, NaN where not embedded
    codes: np.ndarray  # (n,) colour code per frame (rv.GREY / rv.NO_POSE / 0..7)
    names: list[str]  # the coloured clusters, by code
    grey_name: str  # what a grey frame is: "other" or "unnamed"
    cloud: np.ndarray  # (k, 2) every pose, the grey cloud
    pca_var: np.ndarray  # (2,) explained variance
    pose_note: str  # how pose space was built, for the window
    positions: np.ndarray | None  # (n, 2) floor position, NaN where untracked
    pos_units: str  # "mm" or "px"
    floor: tuple[float, float]  # the floor's extent in pos_units
    align: str  # "rotate" (head up), "mirror" (nose right) or "none"
    body_half: float  # half the outline panel's span, px

    @property
    def aspect(self) -> float:
        return self.size[0] / max(self.size[1], 1)

    @property
    def n_frames(self) -> int:
        return self.last - self.first

    def save(self, path: Path | str) -> None:
        """Write to ``path`` (npz): the arrays as arrays, the rest as one JSON string."""
        import json

        arrays = {
            k: getattr(self, k) for k in ("t_ns", "xy", "codes", "cloud", "pca_var")
        }
        if self.positions is not None:
            arrays["positions"] = self.positions
        meta = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in arrays and f.name != "positions"
        }
        np.savez(path, meta=json.dumps(meta), **arrays)

    @classmethod
    def read(cls, path: Path | str) -> "VizData":
        import json

        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k] for k in z.files if k != "meta"}
        meta["size"] = tuple(meta["size"])
        meta["floor"] = tuple(meta["floor"])
        return cls(positions=arrays.pop("positions", None), **arrays, **meta)


def recipe_of(
    config: RecordingVisualizationConfig | dict,
) -> RecordingVisualizationConfig:
    """A recipe as the config model, from the model or a saved dict (bookkeeping keys dropped)."""
    if isinstance(config, RecordingVisualizationConfig):
        return config
    fields = RecordingVisualizationConfig.model_fields
    return RecordingVisualizationConfig.model_validate(
        {k: v for k, v in config.items() if k in fields}
    )


def load_data(experiment, run, config, progress: Progress | None = None) -> VizData:
    """Read one recipe's data. ``progress(stage, fraction, detail)`` follows each stage.

    Raises:
        ValueError: The recording is not one of the run's, the run follows two animals as one
            (joint), the range lies outside the recording, or a cut has no tree to cut.
    """
    from detail.storage import recording as recording_store

    progress = progress or _noop
    recipe = recipe_of(config)
    state = run.state()
    rid = recipe.recording_id
    if rid not in state.source_recording_ids:
        raise ValueError(
            f"{rid} is not one of {run.id}'s recordings ({', '.join(state.source_recording_ids)})"
        )
    if not isinstance(state.track_id, int) and not str(state.track_id).isdigit():
        raise ValueError(
            f"{run.id} follows a joint subject ({state.track_id}); a visualization follows one "
            "animal"
        )
    recording = experiment.get_recording(rid)
    rec_dir = Path(recording.dir)
    data = recording_store.read_recording(rec_dir)
    embed_run_id = state.embed_run_for(rid)
    embed_entry = {r["id"]: r for r in data.get("embedding_runs", [])}.get(
        embed_run_id, {}
    )
    tracking_run_id = embed_entry.get("tracking_run_id")
    if not tracking_run_id:
        raise ValueError(f"{embed_run_id} in {recording.name} names no tracking run")

    progress("timestamps", 0.0, "reading the recording's timestamps")
    t_ns = _recording_t_ns(rec_dir, data)
    first, last = _frame_range(recipe, t_ns, recording.name)
    w, h = recording.video_resolution or (1, 1)
    fps = float((data.get("video") or {}).get("fps_actual") or 30.0)

    progress("pose space", 0.05, "pooling the embeddings as the clustering did")
    reach = int((recipe.ghost_s + 2.0) * fps)
    xy, cloud, var, note = _pose_space(
        run, state, rec_dir, embed_run_id, recipe, t_ns, max(0, first - reach), last
    )

    progress("clusters", 0.45, "resolving the clusters")
    codes, names, grey = _codes(run, recipe, rid, t_ns, np.isfinite(xy[:, 0]))

    positions, units, floor = None, "px", (float(w), float(h))
    if recipe.location:
        back = int(recipe.location_window_s * fps) + 1
        positions, units, floor = _positions(
            recording,
            tracking_run_id,
            recipe.track_id,
            (max(0, first - back), last),
            len(t_ns),
            lambda f: progress("location", 0.55 + 0.4 * f, "Location Export's compute"),
        )
    align = str(embed_entry.get("align") or "none")
    progress("done", 1.0, "")
    return VizData(
        experiment_dir=str(experiment.dir),
        run_id=run.id,
        recipe=recipe.model_dump(),
        recording_id=rid,
        recording_name=recording.name,
        recording_dir=str(rec_dir),
        tracking_run_id=tracking_run_id,
        embed_run_id=embed_run_id,
        track_id=recipe.track_id,
        size=(int(w), int(h)),
        fps=fps,
        t_ns=t_ns,
        first=first,
        last=last,
        xy=xy.astype(np.float32),
        codes=codes,
        names=names,
        grey_name=grey,
        cloud=cloud.astype(np.float32),
        pca_var=var,
        pose_note=note,
        positions=None if positions is None else positions.astype(np.float32),
        pos_units=units,
        floor=floor,
        align=align,
        body_half=_body_half(
            rec_dir, embed_run_id, experiment.dir, first, last, (w, h)
        ),
    )


def recode(data: VizData, run, config) -> VizData:
    """``data`` with its colours re-resolved for ``config``'s clusters — the window's cluster
    choice, a second, not a reload. The recording and range must be ``data``'s own.

    Raises:
        ValueError: ``config`` is for another recording or range.
    """
    recipe = recipe_of(config)
    old = recipe_of(data.recipe)
    same = ("recording_id", "track_id", "start_s", "end_s")
    if any(getattr(recipe, k) != getattr(old, k) for k in same):
        raise ValueError(
            "recode keeps the recording and range; load the new ones instead"
        )
    t_ns = np.asarray(data.t_ns)
    codes, names, grey = _codes(
        run, recipe, data.recording_id, t_ns, np.isfinite(data.xy[:, 0])
    )
    return dataclasses.replace(
        data, recipe=recipe.model_dump(), codes=codes, names=names, grey_name=grey
    )


def _recording_t_ns(rec_dir: Path, data: dict) -> np.ndarray:
    """Every frame's ``t_ns``, recording-global, from the active timestamp maps (the frame count
    is the maps' length, never cv2's estimate)."""
    from detail.storage import recording as recording_store

    parts = (data.get("video") or {}).get("files") or []
    out = [
        recording_store.load_active_timestamps_cached(rec_dir, data, i).values
        for i in range(len(parts))
    ]
    return np.concatenate(out).astype(np.int64) if out else np.zeros(0, np.int64)


def _frame_range(recipe, t_ns: np.ndarray, name: str) -> tuple[int, int]:
    if recipe.start_s is None:
        return 0, len(t_ns)
    lo = int(np.searchsorted(t_ns, int(recipe.start_s * 1e9), "left"))
    hi = int(np.searchsorted(t_ns, int(recipe.end_s * 1e9), "left"))
    if hi - lo < 2:
        end = t_ns[-1] / 1e9 if len(t_ns) else 0.0
        raise ValueError(
            f"{recipe.start_s}–{recipe.end_s} s holds no frames of {name} (0–{end:.1f} s)"
        )
    return lo, hi


def _trajectory_span(rec_dir, embed_run_id, track_id, t_lo: int, t_hi: int):
    """A track's embeddings with ``t_lo <= t_ns <= t_hi``: ``(features, t_ns)``, reading only those
    rows of each part (a minute of a half-hour session, not the whole trajectory)."""
    from detail.storage.zarr_io import PartZarrReader

    feats, times = [], []
    run_dir = Path(rec_dir) / "embeddings" / embed_run_id / f"track_{track_id}"
    for part in sorted(run_dir.glob("part_*.zarr")):
        reader = PartZarrReader(part)
        n = reader.n_rows
        if n == 0:
            continue
        t = np.asarray(reader.column("t_ns")[:n], dtype=np.int64)
        lo, hi = int(np.searchsorted(t, t_lo, "left")), int(
            np.searchsorted(t, t_hi, "right")
        )
        if hi <= lo:
            continue
        feats.append(np.asarray(reader.column("embedding")[lo:hi], dtype=np.float32))
        times.append(t[lo:hi])
    if not feats:
        raise ValueError(
            f"track {track_id} has no embedded frames in this stretch ({embed_run_id})"
        )
    return np.concatenate(feats), np.concatenate(times)


def _pose_space(run, state, rec_dir, embed_run_id, recipe, t_ns, lo, hi):
    """Project frames ``lo..hi`` into the plane of the clustering's own features."""
    from sklearn.decomposition import PCA

    from detail.analysis import clip_query, segmentation
    from detail.cluster_methods.codebook_viterbi import _blocks_of, _unit

    window_s = state.method_params.get("window_s")
    pad = int(float(window_s or 0) * 1e9) + 1
    feats, f_t = _trajectory_span(
        rec_dir,
        embed_run_id,
        recipe.track_id,
        int(t_ns[lo]) - pad,
        int(t_ns[hi - 1]) + pad,
    )
    fit = run.fitted()
    window = 1
    if window_s:
        window = max(1, clip_query.band_rows(f_t, int(float(window_s) * 1e9)))
    pooled, rows, _ = segmentation.build_features(
        feats, _blocks_of(f_t, None), window=window, pool="mean"
    )
    rows = np.asarray(rows)
    pooled = _unit(np.asarray(pooled, dtype=np.float64))
    sample = getattr(fit, "sample", None)
    if sample is not None and len(sample) >= 3:
        basis = np.asarray(sample, dtype=np.float64)
        note = (
            f"The first two principal components of the clustering's features: embeddings pooled "
            f"over {window_s} s, fitted on its {len(basis):,}-frame codebook sample."
        )
    else:
        take = np.linspace(0, len(pooled) - 1, min(len(pooled), 6000)).astype(int)
        basis = pooled[take]
        note = (
            "The first two principal components of the per-frame embeddings, fitted on this "
            "stretch of the recording: the method keeps no frame sample of its own."
        )
    pca = PCA(2).fit(basis)
    xy = np.full((len(t_ns), 2), np.nan)
    at = np.searchsorted(t_ns, f_t[rows])
    ok = (at < len(t_ns)) & (t_ns[np.minimum(at, len(t_ns) - 1)] == f_t[rows])
    xy[at[ok]] = pca.transform(pooled[ok])
    return xy, pca.transform(basis), pca.explained_variance_ratio_, note


def _codes(run, recipe, rid, t_ns, embedded):
    """Per-frame colour codes and the coloured clusters' names."""
    view = run.node_view()
    if recipe.clusters == "named":
        names_of = run.node_names()
        spans = [s for s in view.labelled_runs(names_of) if _ours(s, rid, recipe)]
        weight: dict[str, int] = {}
        for s in view.labelled_runs(names_of):
            name = names_of[str(s["node"])]
            weight[name] = weight.get(name, 0) + int(s["end_ns"]) - int(s["start_ns"])
        order = sorted(weight, key=lambda n: -weight[n])[: len(rv.SERIES["dark"])]
        node_code = {int(k): order.index(v) for k, v in names_of.items() if v in order}
        return rv.label_raster(spans, t_ns, node_code, embedded), order, "unnamed"
    tree = run.hierarchy()
    nodes = rv.cut_nodes(
        tree.children, tree.height, tree.n_leaves, float(recipe.cut_height)
    )
    numbered = {str(node): str(i + 1) for i, node in enumerate(nodes)}
    # The cut is the tree as fitted: clip edits belong to named labels, not to a cut.
    spans = [
        s
        for s in view.labelled_runs(numbered, apply_edits=False)
        if _ours(s, rid, recipe)
    ]
    included = [n for n in recipe.cut_included if 1 <= n <= len(nodes)]
    node_code = {nodes[n - 1]: i for i, n in enumerate(included)}
    return (
        rv.label_raster(spans, t_ns, node_code, embedded),
        [f"cluster {n}" for n in included],
        "other",
    )


def _ours(span: dict, rid: str, recipe) -> bool:
    return span["recording_id"] == rid and str(span["subject"]) == str(recipe.track_id)


def _positions(recording, tracking_run_id, track_id, frames, n, on_fraction):
    """The track's mask centroid per frame, by Location Export's own compute (mm when
    calibrated, px otherwise), and the floor's extent in the same units."""
    from detail.analysis.locations import compute_location_tables

    tables = compute_location_tables(
        recording,
        tracking_run_id,
        lambda done, total: on_fraction(done / max(total, 1)),
        frames=frames,
    )
    columns, rows = tables.tracks
    col = {c: i for i, c in enumerate(columns)}
    mm = bool(recording.pixels_per_mm)
    cx, cy = ("mask_cx_mm", "mask_cy_mm") if mm else ("mask_cx_px", "mask_cy_px")
    out = np.full((n, 2), np.nan)
    for row in rows:
        if int(row[col["object_id"]]) != int(track_id):
            continue
        f = int(row[col["frame_idx"]])
        if 0 <= f < n and row[col[cx]] != "":
            out[f] = (float(row[col[cx]]), float(row[col[cy]]))
    w, h = recording.video_resolution or (1, 1)
    scale = (1.0 / recording.pixels_per_mm) if mm else 1.0
    return out, ("mm" if mm else "px"), (float(w) * scale, float(h) * scale)


def _body_half(rec_dir, embed_run_id, experiment_dir, first, last, size) -> float:
    """Half the outline panel's span: most of the animal's front-to-back length, px."""
    points = _keypoints(experiment_dir, rec_dir, embed_run_id)
    cols = (
        points.axis_columns if points is not None and points.run is not None else None
    )
    if cols is None:
        return 0.18 * max(size)
    lengths = []
    for fno in np.linspace(first, last - 1, 200).astype(int):
        kp = _points_at(points, rec_dir, int(fno), size, 0)
        if kp is not None:
            front, back = kp[cols[0]].mean(0), kp[cols[1]].mean(0)
            lengths.append(float(np.hypot(*(front - back))))
    return 0.95 * float(np.percentile(lengths, 95)) if lengths else 0.18 * max(size)


# --------------------------------------------------------------------------- pixels


def _keypoints(experiment_dir, rec_dir, embed_run_id):
    """The keypoints the embedding read (``api.clip_views.ClipPoints``), or ``None``."""
    from detail.api.clip_views import ClipPoints
    from detail.api.run_preview import _embedding_meta

    try:
        meta = _embedding_meta(Path(rec_dir), embed_run_id)
        return ClipPoints.of(Path(experiment_dir), Path(rec_dir), meta, embed_run_id)
    except (FileNotFoundError, ValueError, KeyError):
        return None


@functools.lru_cache(maxsize=32)
def _parts(rec_dir: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """A recording's part offsets and stems, read once per recording."""
    from detail.storage import recording as recording_store

    offsets = tuple(recording_store.part_frame_offsets(rec_dir) or [0])
    data = recording_store.read_recording(rec_dir)
    stems = tuple(
        recording_store.part_stem(f) for f in recording_store.video_files(data)
    )
    return offsets, stems


def _part_of(rec_dir, fno: int) -> tuple[int, int, str]:
    offsets, stems = _parts(str(rec_dir))
    part = int(np.searchsorted(offsets, fno, "right")) - 1
    return part, fno - offsets[part], stems[part]


def _points_at(points, rec_dir, fno: int, size, track_id: int):
    if points is None or points.run is None:
        return None
    _part, local, stem = _part_of(rec_dir, fno)
    got = points.at(stem, local).get(int(track_id))
    if got is None:
        return None
    return np.asarray(got[0], dtype=float) * np.asarray(size, dtype=float)


class FrameReader:
    """A visualization's pixels: the video frame, the animal's mask, its keypoints (px)."""

    def __init__(self, data: VizData) -> None:
        from detail.api.run_preview import TrackingPreview

        self.data = data
        self._tracking = TrackingPreview(data.recording_dir, data.tracking_run_id)
        self._points = _keypoints(
            data.experiment_dir, data.recording_dir, data.embed_run_id
        )

    @property
    def axis_columns(self):
        p = self._points
        return p.axis_columns if p is not None and p.run is not None else None

    def keypoints(self, fno: int):
        return _points_at(
            self._points,
            self.data.recording_dir,
            fno,
            self.data.size,
            self.data.track_id,
        )

    def _mask(self, detections) -> np.ndarray:
        w, h = self.data.size
        out = np.zeros((h, w), bool)
        for det in detections:
            if det.object_id == self.data.track_id and det.mask is not None:
                out |= det.mask.astype(bool)
        return out

    def frame(self, fno: int):
        """``(rgb, mask, keypoints)`` at one frame, by a seek — for a scrub, never an export."""
        part, local, _stem = _part_of(self.data.recording_dir, fno)
        got = self._tracking.frame(part, local, with_masks=True)
        if got is None:
            return None
        _actual, rgb, dets = got
        return rgb, self._mask(dets), self.keypoints(fno)

    def mask_only(self, fno: int) -> np.ndarray:
        """The mask at ``fno`` without decoding the picture (the outline trail's past frames)."""
        part, local, _stem = _part_of(self.data.recording_dir, fno)
        w, h = self.data.size
        _actual, _canvas, dets = self._tracking.blank_frame(part, local, (h, w))
        return self._mask(dets)

    def span(self, first: int, last: int, step: int = 1):
        """``(fno, rgb, mask, keypoints)`` for ``first..last-1`` every ``step``, one forward pass
        per video part (Tenet 11)."""
        offsets, _stems = _parts(str(self.data.recording_dir))
        bounds = list(offsets) + [len(self.data.t_ns)]
        for part in range(len(offsets)):
            lo, hi = max(first, bounds[part]), min(last, bounds[part + 1])
            if lo >= hi:
                continue
            for actual, rgb, dets in self._tracking.frames_range(
                part,
                lo - bounds[part],
                hi - 1 - bounds[part],
                with_masks=True,
                step=step,
            ):
                fno = bounds[part] + int(actual)
                yield fno, rgb, self._mask(dets), self.keypoints(fno)


# --------------------------------------------------------------------------- drawing + writing


class RecordingVisualization:
    """One visualization, ready to draw and export."""

    def __init__(self, data: VizData) -> None:
        self.data = data
        self._reader: FrameReader | None = None
        self._canvases: dict = {}

    @property
    def reader(self) -> FrameReader:
        if self._reader is None:
            self._reader = FrameReader(self.data)
        return self._reader

    def canvas(self, theme: str, width: int, height: int):
        from detail.api.recording_visualization_draw import Canvas

        key = (theme, width, height)
        if key not in self._canvases:
            self._canvases[key] = Canvas(self.data, self.reader, theme, width, height)
        return self._canvases[key]

    def frame(
        self, fno: int, theme: str = "dark", size: tuple[int, int] = (1920, 1080)
    ):
        """The picture at recording frame ``fno`` (RGB uint8), or None when it cannot decode."""
        got = self.reader.frame(int(fno))
        if got is None:
            return None
        return self.canvas(theme, *size).draw(int(fno), *got)

    def estimate(self, export: RecordingVisualizationExport) -> dict:
        """Measure, then scale: ``{"seconds", "bytes_per_file", "files", "frames", "workers"}``."""
        return estimate([self], export)

    def export(
        self,
        path,
        export: RecordingVisualizationExport,
        progress: Progress | None = None,
    ):
        """Write one file per theme (``<stem>_<theme>.mp4``); returns the paths written."""
        return export_montage([self], path, export, progress)


def themed_path(path: Path | str, theme: str, themes: Sequence[str]) -> Path:
    p = Path(path)
    if p.suffix.lower() != ".mp4":
        p = p.with_suffix(".mp4")
    return p if len(themes) == 1 else p.with_name(f"{p.stem}_{theme}{p.suffix}")


def _timing(
    vizs: Sequence[RecordingVisualization], export: RecordingVisualizationExport
):
    """``(fps_out, source frames per output frame, output frame count)`` for the montage clock."""
    src_fps = max(v.data.fps for v in vizs)
    fps_out, speed = speed_timing(export.speed, src_fps)
    per_out = src_fps * speed / fps_out
    longest = max(v.data.n_frames / v.data.fps for v in vizs)
    return fps_out, per_out, max(1, int(longest * fps_out / speed))


def _workers(export: RecordingVisualizationExport, n_chunks: int) -> int:
    """Renderers to start: the export's choice, else the usable cores (`core/cpu`, the one place
    cores are counted) less two for the window and the joiner, never more than the chunks.
    """
    from detail.core.cpu import usable_cores

    default = max(1, (usable_cores() or 4) - 2)
    return max(1, min(export.workers or default, n_chunks))


def estimate(
    vizs: Sequence[RecordingVisualization], export: RecordingVisualizationExport
) -> dict:
    """Render and encode ``ESTIMATE_FRAMES`` frames of each visualization, then scale the time per
    frame and bytes per frame to the export's length, workers and themes."""
    from detail.storage.video import open_video_writer

    fps_out, _per_out, n_out = _timing(vizs, export)
    w, h = export.width, export.height
    tiles = rv.montage_tiles(len(vizs), w, h)
    theme = export.themes[0]
    pictures, t0 = [], time.perf_counter()
    for k in range(ESTIMATE_FRAMES):
        board = np.zeros((h, w, 3), np.uint8)
        for v, tile in zip(vizs, tiles):
            d = v.data
            fno = d.first + int((k + 0.5) * d.n_frames / ESTIMATE_FRAMES)
            pic = v.frame(fno, theme, (int(tile.w), int(tile.h)))
            if pic is not None:
                board[
                    int(tile.y) : int(tile.y + tile.h),
                    int(tile.x) : int(tile.x + tile.w),
                ] = pic
        pictures.append(board)
    per_frame = (time.perf_counter() - t0) / ESTIMATE_FRAMES
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.mp4"
        writer, _codec = open_video_writer(probe, fps_out, w, h, exact_colour=True)
        for pic in pictures:
            for _ in range(
                4
            ):  # held a few frames, so the probe has inter frames, as a video does
                writer.write(rgb_to_bgr(pic))
        writer.release()
        bytes_per_frame = probe.stat().st_size / (4 * len(pictures))
    chunks = max(1, -(-n_out // CHUNK_FRAMES))
    workers = _workers(export, chunks)
    # Workers share the machine and each pays its start-up once.
    seconds = (n_out * per_frame / workers) * 1.25 + 8.0
    return {
        "seconds": seconds * len(export.themes),
        "bytes_per_file": int(bytes_per_frame * n_out),
        "files": len(export.themes),
        "frames": n_out,
        "workers": workers,
    }


def export_montage(
    vizs: Sequence[RecordingVisualization],
    path,
    export: RecordingVisualizationExport,
    progress: Progress | None = None,
) -> list[Path]:
    """Write ``vizs`` side by side on one clock (one visualization fills the screen), one file
    per theme, rendered in parallel chunks and joined without re-encoding.

    Raises:
        ValueError: More visualizations than a montage holds.
    """
    import concurrent.futures as cf
    import multiprocessing as mp

    from detail.storage.video import concat_mp4

    progress = progress or _noop
    rv.montage_grid(len(vizs))
    fps_out, per_out, n_out = _timing(vizs, export)
    written = []
    with tempfile.TemporaryDirectory(prefix="detail-viz-") as tmp:
        caches = []
        for i, v in enumerate(vizs):
            cache = Path(tmp) / f"viz_{i}.npz"
            v.data.save(cache)
            caches.append(str(cache))
        bounds = list(range(0, n_out, CHUNK_FRAMES)) + [n_out]
        chunks = list(zip(bounds[:-1], bounds[1:]))
        for t, theme in enumerate(export.themes):
            out = themed_path(path, theme, export.themes)
            tasks = [
                (
                    caches,
                    theme,
                    export.width,
                    export.height,
                    a,
                    b,
                    per_out,
                    fps_out,
                    str(Path(tmp) / f"{theme}_{i:04d}.mp4"),
                )
                for i, (a, b) in enumerate(chunks)
            ]
            done = 0
            ctx = mp.get_context("spawn")
            with cf.ProcessPoolExecutor(
                _workers(export, len(tasks)), mp_context=ctx
            ) as pool:
                futures = [pool.submit(_render_chunk, task) for task in tasks]
                try:
                    for fut in cf.as_completed(futures):
                        fut.result()
                        done += 1
                        progress(
                            f"rendering {theme}",
                            (t + done / len(tasks)) / len(export.themes),
                            f"{done}/{len(tasks)} chunks",
                        )
                except BaseException:
                    for f in futures:
                        f.cancel()
                    raise
            concat_mp4([task[-1] for task in tasks], out)
            written.append(out)
    progress("done", 1.0, ", ".join(str(p) for p in written))
    return written


def _render_chunk(task) -> int:
    """One worker: output frames ``[a, b)`` of a montage into ``out`` (spawn-safe: everything it
    needs comes from the cached data and the stores)."""
    from detail.storage.video import open_video_writer

    caches, theme, width, height, a, b, per_out, fps_out, out = task
    vizs = [RecordingVisualization(VizData.read(c)) for c in caches]
    tiles = rv.montage_tiles(len(vizs), width, height)
    streams = []
    for v, tile in zip(vizs, tiles):
        d = v.data
        firsts = [d.first + int(round(k * per_out)) for k in (a, b)]
        lo = min(firsts[0], d.last - 1)
        hi = max(lo + 1, min(firsts[1], d.last))
        streams.append(_held(v, tile, theme, lo, hi, per_out, d.first))
    writer, _codec = open_video_writer(out, fps_out, width, height, exact_colour=True)
    try:
        for k in range(a, b):
            board = np.zeros((height, width, 3), np.uint8)
            for s, tile in zip(streams, tiles):
                pic = s(k)
                board[
                    int(tile.y) : int(tile.y + tile.h),
                    int(tile.x) : int(tile.x + tile.w),
                ] = pic
            writer.write(rgb_to_bgr(board))
    finally:
        writer.release()
    return b - a


def _held(v: RecordingVisualization, tile, theme, lo, hi, per_out, start):
    """A tile's picture for montage step ``k``: drawn from one forward pass over ``lo..hi``, and
    held, dimmed, on its last frame once its visualization has ended."""
    canvas = v.canvas(theme, int(tile.w), int(tile.h))
    frames = v.reader.span(lo, hi)
    cur = {"fno": -1, "pixels": None, "drawn": -1, "pic": None, "ended": None}

    def ended_picture():
        # Always the visualization's own last frame, read one way: every chunk computes its
        # hold separately, and one that held "whatever it drew last" made the held tile jump
        # at the chunk boundary where the next chunk started after the end.
        if cur["ended"] is None:
            got = v.reader.frame(v.data.last - 1)
            last = (
                canvas.draw(v.data.last - 1, *got)
                if got is not None
                else canvas.blank()
            )
            cur["ended"] = canvas.dim(last)
        return cur["ended"]

    def at(k: int):
        fno = start + int(round(k * per_out))
        if fno >= v.data.last:
            return ended_picture()
        while cur["fno"] < fno:
            try:
                f, rgb, mask, kp = next(frames)
            except StopIteration:
                break
            cur["fno"], cur["pixels"] = f, (rgb, mask, kp)
        if cur["pixels"] is not None and cur["drawn"] != cur["fno"]:
            cur["pic"] = canvas.draw(cur["fno"], *cur["pixels"])
            cur["drawn"] = cur["fno"]
        return cur["pic"] if cur["pic"] is not None else canvas.blank()

    return at
