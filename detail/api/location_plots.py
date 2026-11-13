"""Plots of a completed location-export run (DESIGN_PLAN §16 "Location plots").

``Experiment.location_plots(run_id)`` returns a ``LocationPlots``: the read boundary the CLI and
the GUI's result page draw through. It reads the run's per-recording ``tracks.csv`` /
``locations.yaml`` on demand (nothing is written by the run) and keeps what it read, so a second
plot of the same recordings — another kind, another colour, a turned camera — re-reads nothing.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from detail.analysis import location_plots as plots
from detail.configs.location_plots import LocationMontageView, LocationPlotView
from detail.core.video_codecs import DEFAULT_VIDEO_CODEC
from detail.report.plot_spec import PlotSpec
from detail.storage import location_export

#: ``(stage, fraction, detail)`` — the synchronous-API progress shape (CLAUDE.md).
Progress = Callable[[str, float, str], None] | None

MODULE_ID = "location_export"


@dataclass
class LocationMontage:
    """A location export's recordings ready to play side by side: the montage engine, one
    label per tile, and what was left out (§16 "Preview and montage")."""

    montage: object
    labels: list[str]
    notes: list[str] = field(default_factory=list)


#: The plot kinds drawn from pairs of tracks rather than from one animal's path.
DISTANCE_KINDS = ("distance", "distance_hist")


class LocationPlots:
    """Draw a location-export run's paths, occupancy and speed.

    Raises:
        ValueError: (at construction) the run is not a location export, or has no completed
            recording to plot.
    """

    def __init__(self, experiment, run_id: str) -> None:
        record = experiment.analysis_record(run_id)
        if record.get("module_id") != MODULE_ID:
            raise ValueError(
                f"{run_id} is a {record.get('module_id')!r} run, not a location export"
            )
        self._experiment = experiment
        self.run_id = run_id
        self._run_dir = experiment.analysis_run_dir(run_id)
        per_recording = record.get("per_recording") or {}
        selected = list(record.get("recording_ids") or per_recording)
        self._recordings = [
            sid
            for sid in selected
            if (per_recording.get(sid) or {}).get("status") == "complete"
            and (self._run_dir / sid / location_export.TRACKS_FILE).exists()
        ]
        if not self._recordings:
            raise ValueError(f"{run_id} has no completed recording to plot")
        self._tracking_runs = dict(record.get("tracking_run_ids") or {})
        if not self._tracking_runs and record.get("tracking_run_id"):
            self._tracking_runs = {
                sid: record["tracking_run_id"] for sid in self._recordings
            }
        params = record.get("params") or {}
        self._keypoint_runs = dict(
            params.get("keypoint_run_ids") or record.get("keypoint_run_ids") or {}
        )
        self._tables: dict[str, plots.TrackTable] = {}
        self._frames: dict[str, plots.FrameInfo] = {}
        self._labels: dict[str, dict[int, str] | None] = {}
        self._keypoint_names: dict[str, list[str]] = {}
        self._keypoint_tables: dict[str, plots.KeypointTable | None] = {}
        self._edges: dict[str, dict | None] = {}
        self._lock = threading.Lock()

    @property
    def recording_ids(self) -> list[str]:
        """The run's completed recordings, in run order — what a plot can draw."""
        return list(self._recordings)

    @property
    def group_columns(self) -> list[str]:
        """The Recording Information columns a plot can group by."""
        return list(self._experiment.metadata_columns)

    def names(self) -> dict[str, str]:
        """``{recording_id: name}`` for the run's recordings (a deleted one shows its id)."""
        known = self._experiment.recording_names()
        return {sid: known.get(sid, sid) for sid in self._recordings}

    def subjects(self, recording_id: str) -> list[plots.Subject]:
        """One recording's animals with tracked rows, labelled, most-tracked first."""
        self._table(recording_id)
        present, _missing = plots.subjects_of(
            self._tables[recording_id], self._labels[recording_id]
        )
        return present

    def subject_choices(self, progress: Progress = None) -> list[str]:
        """Every animal label across the run's recordings, sorted — what a picker offers.

        Reads every recording's ``tracks.csv`` (a run without labels names its animals by
        track, and only the rows say which tracks there are), so it reports progress; what it
        reads is kept for the plots that follow.
        """
        labels = set()
        for index, sid in enumerate(self._recordings):
            if progress is not None:
                progress("reading", index / len(self._recordings), self.names()[sid])
            labels.update(s.label for s in self.subjects(sid))
        if progress is not None:
            progress("reading", 1.0, f"{len(self._recordings)} recordings")
        order = plots.subject_order(labels)
        return sorted(order, key=order.get)

    def _table(self, recording_id: str) -> plots.TrackTable:
        with self._lock:
            cached = self._tables.get(recording_id)
        if cached is not None:
            return cached
        table = plots.read_tracks(
            self._run_dir / recording_id / location_export.TRACKS_FILE
        )
        self._manifest(recording_id)
        with self._lock:
            self._tables[recording_id] = table
        return table

    def _manifest(self, recording_id: str) -> None:
        """Read one recording's frame and animal labels (a small YAML, no rows), once."""
        with self._lock:
            if recording_id in self._frames:
                return
        frame, labels, keypoint_names = plots.read_manifest(
            self._run_dir / recording_id / location_export.MANIFEST_FILE
        )
        if labels is None:
            labels = self._tracking_run_labels(recording_id)
        with self._lock:
            self._frames[recording_id] = frame
            self._labels[recording_id] = labels
            self._keypoint_names[recording_id] = keypoint_names

    def _keypoints(self, recording_id: str) -> plots.KeypointTable | None:
        """One recording's ``keypoints.csv``, read once; None when the export had none."""
        with self._lock:
            if recording_id in self._keypoint_tables:
                return self._keypoint_tables[recording_id]
        path = self._run_dir / recording_id / location_export.KEYPOINTS_FILE
        table = plots.read_keypoints(path) if path.is_file() else None
        with self._lock:
            self._keypoint_tables[recording_id] = table
        return table

    def _pair_edges(self, recording_id: str) -> dict | None:
        """One recording's measured mask-edge distances, read once; None when not measured."""
        with self._lock:
            if recording_id in self._edges:
                return self._edges[recording_id]
        edges = plots.read_pair_edges(
            self._run_dir / recording_id / location_export.PAIR_FILE
        )
        with self._lock:
            self._edges[recording_id] = edges
        return edges

    @property
    def keypoint_names(self) -> list[str]:
        """The keypoints the run exported (its skeleton's points, in order); empty when the
        export read no keypoint track run. Reads the manifests only."""
        names: list[str] = []
        for sid in self._recordings:
            self._manifest(sid)
            names.extend(n for n in self._keypoint_names[sid] if n not in names)
        return names

    def pair_choices(self, progress: Progress = None) -> list[str]:
        """Every pair name (``Intruder – Resident``) across the run's recordings, sorted."""
        found = set()
        for index, sid in enumerate(self._recordings):
            if progress is not None:
                progress("reading", index / len(self._recordings), self.names()[sid])
            labels = [s.label for s in self.subjects(sid)]
            for i, a in enumerate(labels):
                found.update(plots.pair_label(a, b) for b in labels[i + 1 :])
        if progress is not None:
            progress("reading", 1.0, f"{len(self._recordings)} recordings")
        return sorted(found, key=plots.subject_order(found).get)

    def _run_wide_pair_order(self) -> dict[str, int]:
        """Colour order over every pair of *labelled* animals in the run (the manifests)."""
        found = set()
        for sid in self._recordings:
            self._manifest(sid)
            labels = sorted(set((self._labels[sid] or {}).values()))
            for i, a in enumerate(labels):
                found.update(plots.pair_label(a, b) for b in labels[i + 1 :])
        return plots.subject_order(found)

    def _tracking_run_labels(self, recording_id: str) -> dict[int, str] | None:
        """``subject_labels`` from the tracking run, for an export made before its manifest
        recorded them. None when the run or the recording is gone, or never named its animals.
        """
        run_id = self._tracking_runs.get(recording_id)
        if not run_id:
            return None
        try:
            recording = self._experiment.resolve_recording(recording_id)
        except KeyError:
            return None
        return plots.normalize_labels(recording.subject_labels(run_id))

    def _run_wide_order(self) -> dict[str, int]:
        """Colour order over every *labelled* animal of the run (cheap: the manifests), so
        leaving a recording or an animal out never repaints the rest."""
        labels = []
        for sid in self._recordings:
            self._manifest(sid)
            labels.extend((self._labels[sid] or {}).values())
        return plots.subject_order(labels)

    def traces(
        self,
        view: LocationPlotView,
        recording_ids: list[str] | None = None,
        progress: Progress = None,
    ) -> tuple[list[plots.Trace], list[str]]:
        """The chosen animals of each selected recording, and a note per one left out.

        Raises:
            ValueError: A recording is not one of the run's completed ones.
        """
        chosen = self._chosen(recording_ids)
        names = self.names()
        out, notes = [], []
        for index, sid in enumerate(chosen):
            if progress is not None:
                progress("reading", index / len(chosen), names[sid])
            table = self._table(sid)
            uses_points = bool(plots.position_points(view))
            traces, left_out = plots.recording_traces(
                table,
                self._frames[sid],
                self._labels[sid],
                view,
                recording_id=sid,
                name=names[sid],
                keypoints=self._keypoints(sid) if uses_points else None,
                keypoint_names=self._keypoint_names[sid],
            )
            out.extend(traces)
            notes.extend(left_out)
        if progress is not None:
            progress("reading", 1.0, f"{len(chosen)} recordings")
        return out, notes

    def pairs(
        self,
        view: LocationPlotView,
        recording_ids: list[str] | None = None,
        progress: Progress = None,
    ) -> tuple[list[plots.PairSeries], list[str]]:
        """Every chosen pair of animals in each selected recording, and what is left out.

        Raises:
            ValueError: A recording is not one of the run's, or ``mask_edge`` on a recording
                whose export did not measure it.
        """
        chosen = self._chosen(recording_ids)
        names = self.names()
        out, notes = [], []
        for index, sid in enumerate(chosen):
            if progress is not None:
                progress("reading", index / len(chosen), names[sid])
            table = self._table(sid)
            edges = self._pair_edges(sid) if view.distance_by == "mask_edge" else None
            try:
                found, left_out = plots.recording_pairs(
                    table,
                    self._frames[sid],
                    self._labels[sid],
                    view,
                    recording_id=sid,
                    name=names[sid],
                    edges=edges,
                )
            except ValueError as exc:
                raise ValueError(f"{names[sid]}: {exc}") from exc
            out.extend(found)
            notes.extend(left_out)
        if progress is not None:
            progress("reading", 1.0, f"{len(chosen)} recordings")
        return out, notes

    def _chosen(self, recording_ids: list[str] | None) -> list[str]:
        if recording_ids is None:
            return self.recording_ids
        unknown = [r for r in recording_ids if r not in self._recordings]
        if unknown:
            raise ValueError(
                f"not a completed recording of {self.run_id}: {', '.join(unknown)}"
            )
        if not recording_ids:
            raise ValueError("nothing to plot: no recordings selected")
        return list(recording_ids)

    def build(
        self,
        view: LocationPlotView,
        recording_ids: list[str] | None = None,
        progress: Progress = None,
    ) -> PlotSpec:
        """``view`` of the selected recordings (default: all) as a ``PlotSpec``.

        Every chosen animal is a unit (``view.subjects``, by label, matched per recording);
        an animal or recording left out is named in the spec's note.

        Raises:
            ValueError: An unknown recording or ``group_by`` column, no animal to plot, or
                ``mm`` units with an uncalibrated recording.
        """
        values = None
        if view.group_by is not None:
            columns = self.group_columns
            if view.group_by not in columns:
                have = ", ".join(columns) or "none"
                raise ValueError(
                    f"no Recording Information column {view.group_by!r} (columns: {have})"
                )
            metadata = self._experiment.recording_metadata
            values = {
                sid: (metadata.get(sid) or {}).get(view.group_by, "")
                for sid in self._recordings
            }
        if view.kind in DISTANCE_KINDS:
            pairs, notes = self.pairs(view, recording_ids, progress)
            if progress is not None:
                progress("drawing", 0.0, view.kind)
            spec = plots.build_distance_plot(
                pairs, view, values, order=self._run_wide_pair_order(), notes=notes
            )
            if progress is not None:
                progress("drawing", 1.0, view.kind)
            return spec
        traces, notes = self.traces(view, recording_ids, progress)
        if progress is not None:
            progress("drawing", 0.0, view.kind)
        spec = plots.build_plot(
            traces, view, values, order=self._run_wide_order(), notes=notes
        )
        if progress is not None:
            progress("drawing", 1.0, view.kind)
        return spec

    def export(
        self,
        view: LocationPlotView,
        out_stem: Path | str,
        recording_ids: list[str] | None = None,
        *,
        formats: tuple[str, ...] = ("png",),
        theme: str = "light",
        progress: Progress = None,
    ) -> list[Path]:
        """Write ``view`` as ``<out_stem>.<fmt>`` for each of ``formats``: png, svg and pdf
        pictures (matplotlib), and ``html``, one self-contained interactive page (plotly).
        """
        from detail.report import plot_html, plot_render

        spec = self.build(view, recording_ids, progress)
        if progress is not None:
            progress("writing", 0.0, ", ".join(formats))
        pictures = tuple(f for f in formats if f != "html")
        paths = (
            plot_render.render(spec, out_stem, formats=pictures, theme=theme)
            if pictures
            else []
        )
        if "html" in formats:
            paths.append(
                plot_html.render_html(
                    spec, Path(out_stem).with_suffix(".html"), theme=theme
                )
            )
        if progress is not None:
            progress("writing", 1.0, ", ".join(formats))
        return paths

    # -- preview and montage (§16 "Preview and montage") -------------------------------- #
    def montage(
        self,
        view: LocationMontageView,
        recording_ids: list[str] | None = None,
        subjects: list[str] | None = None,
        progress: Progress = None,
    ) -> LocationMontage:
        """The selected recordings (default: all) as one tile each, every chosen animal
        (``subjects``, by label; default: all) in its colour, over ``view``'s window.

        Raises:
            ValueError: A recording is not one of the run's, or nothing is left to play.
        """
        from detail.api.clip_montage import ClusterMontage, MontageClip
        from detail.api.run_preview import ClipSource
        from detail.core import chart_theme

        chosen = self._chosen(recording_ids)
        names = self.names()
        known = set(self._experiment.recording_names())
        order = self._run_wide_order()
        wanted = None if subjects is None else set(subjects)
        clips, labels, notes = [], [], []
        for index, sid in enumerate(chosen):
            if progress is not None:
                progress("reading", index / len(chosen), names[sid])
            if sid not in known:
                notes.append(
                    f"{names[sid]}: the recording is no longer in the experiment"
                )
                continue
            table = self._table(sid)
            present, _missing = plots.subjects_of(table, self._labels[sid])
            animals = [a for a in present if wanted is None or a.label in wanted]
            if not animals:
                notes.append(f"{names[sid]}: none of the chosen animals")
                continue
            colors = []
            base = len(order)
            for animal in animals:
                identity = order.get(animal.label, base + animal.object_id)
                colors.append(
                    (
                        animal.object_id,
                        chart_theme.rgb(chart_theme.LIGHT.identity(identity)),
                    )
                )
            starts = [int(table.columns[a.object_id]["t_ns"][0]) for a in animals]
            ends = [int(table.columns[a.object_id]["t_ns"][-1]) for a in animals]
            if view.full_length:
                start, end = min(starts), max(ends)
            else:
                start = int(round(view.start_s * 1e9))
                end = min(int(round((view.start_s + view.length_s) * 1e9)), max(ends))
                if start >= end or start > max(ends):
                    notes.append(f"{names[sid]}: nothing tracked in the window")
                    continue
            source = ClipSource(
                tracking_run_id=str(self._tracking_runs.get(sid)),
                keypoint_track_run_id=self._keypoint_runs.get(sid),
                subject_colors=tuple(colors),
            )
            clips.append(
                MontageClip(
                    rank=len(clips) + 1,
                    recording_id=sid,
                    recording_name=names[sid],
                    # The tile draws every animal its ClipSource names; the clip's own
                    # subject is the first of them.
                    subject=animals[0].object_id,
                    embed_run_id="",
                    start_t_ns=start,
                    end_t_ns=end,
                    source=source,
                )
            )
            labels.append(names[sid])
        if progress is not None:
            progress("reading", 1.0, f"{len(chosen)} recordings")
        if not clips:
            detail = f" ({'; '.join(notes)})" if notes else ""
            raise ValueError(f"nothing to play{detail}")
        montage = ClusterMontage(
            self._experiment.dir, clips, full_frame=view.full_frame
        )
        return LocationMontage(montage, labels, notes)

    def export_montage(
        self,
        view: LocationMontageView,
        path: Path | str,
        recording_ids: list[str] | None = None,
        subjects: list[str] | None = None,
        *,
        last_frame: bool = False,
        max_width: int | None = None,
        progress: Progress = None,
        codec: str = DEFAULT_VIDEO_CODEC,
    ) -> Path:
        """Write the montage to ``path``: a ``.gif`` or ``.mp4`` at ``view.speed``, or with
        ``last_frame`` a ``.png`` of its last picture with every trail covering the whole
        clip. ``codec`` is an mp4's encoding (``core.video_codecs.VIDEO_CODECS``).

        Raises:
            ValueError: Nothing to play, or no layer ticked.
        """
        built = self.montage(view, recording_ids, subjects, progress)
        layers = view.layers()
        try:
            if last_frame:
                return built.montage.export_last_frame(
                    path, layers, built.labels, max_width=max_width, progress=progress
                )
            return built.montage.export(
                path,
                layers,
                built.labels,
                max_width=max_width,
                speed=view.speed,
                progress=progress,
                codec=codec,
            )
        finally:
            built.montage.close()
