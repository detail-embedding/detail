"""Plots of a completed location-export run (DESIGN_PLAN §16 "Location plots").

``Experiment.location_plots(run_id)`` returns a ``LocationPlots``: the read boundary the CLI and
the GUI's result page draw through. It reads the run's per-recording ``tracks.csv`` /
``locations.yaml`` on demand (nothing is written by the run) and keeps what it read, so a second
plot of the same recordings — another kind, another colour, a turned camera — re-reads nothing.
"""

import dataclasses
import threading
from collections.abc import Callable
from pathlib import Path

from detail.analysis import location_plots as plots
from detail.configs.location_plots import LocationPlotView
from detail.report.plot_spec import PlotSpec
from detail.storage import location_export

#: ``(stage, fraction, detail)`` — the synchronous-API progress shape (CLAUDE.md).
Progress = Callable[[str, float, str], None] | None

MODULE_ID = "location_export"

#: The formats an export writes: pictures (matplotlib) and one interactive page (plotly).
FORMATS = ("png", "svg", "pdf", "html")

#: Under the title of a plot of a run exported with thin parts removed: its body columns, and
#: the path rule's body length, are not the whole mask's (§16 *The body*).
THIN_PARTS_NOTE = (
    "Body measured with thin parts removed, which the export no longer does: export again "
    "to measure the whole mask"
)


class LocationPlots:
    """Draw a location-export run's paths, occupancy, speed and body.

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
        params = record.get("params") or {}
        self.thin_parts_removed = bool(params.get("remove_thin_parts"))
        selected = list(params.get("source_recording_ids") or per_recording)
        self._recordings = [
            sid
            for sid in selected
            if (per_recording.get(sid) or {}).get("status") == "complete"
            and (self._run_dir / sid / location_export.TRACKS_FILE).exists()
        ]
        if not self._recordings:
            raise ValueError(f"{run_id} has no completed recording to plot")
        self._tables: dict[str, plots.TrackTable] = {}
        self._manifests: dict[str, dict] = {}
        self._keypoint_tables: dict[str, plots.KeypointTable | None] = {}
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

    def _manifest(self, recording_id: str) -> dict:
        """One recording's ``locations.yaml``, read once (a small YAML, no rows)."""
        with self._lock:
            cached = self._manifests.get(recording_id)
        if cached is not None:
            return cached
        manifest = location_export.read_location_manifest(self._run_dir / recording_id)
        if manifest is None:
            raise ValueError(
                f"{self.names()[recording_id]} has no {location_export.MANIFEST_FILE} in "
                f"{self.run_id}"
            )
        with self._lock:
            self._manifests[recording_id] = manifest
        return manifest

    def body_measured(self) -> bool:
        """Whether the export measured the body of any of its recordings (``body`` plots need
        it). Reads the manifests only: each lists the columns its ``tracks.csv`` has."""
        return any(
            all(
                name in ((self._manifest(sid).get("columns") or {}).get("tracks") or [])
                for name in plots.BODY_COLUMNS
            )
            for sid in self._recordings
        )

    @property
    def keypoint_names(self) -> list[str]:
        """The keypoints the run exported (its skeleton's points, in order); empty when the
        export read no keypoint track run. Reads the manifests only."""
        names: list[str] = []
        for sid in self._recordings:
            found = self._manifest(sid).get("keypoint_names") or []
            names.extend(str(n) for n in found if str(n) not in names)
        return names

    def _keypoints(self, recording_id: str) -> plots.KeypointTable | None:
        """One recording's ``keypoints.csv``, read once; ``None`` when the export had none."""
        with self._lock:
            if recording_id in self._keypoint_tables:
                return self._keypoint_tables[recording_id]
        path = self._run_dir / recording_id / location_export.KEYPOINTS_FILE
        table = plots.read_keypoints(path) if path.is_file() else None
        with self._lock:
            self._keypoint_tables[recording_id] = table
        return table

    def _table(self, recording_id: str) -> plots.TrackTable:
        with self._lock:
            cached = self._tables.get(recording_id)
        if cached is not None:
            return cached
        table = plots.read_tracks(
            self._run_dir / recording_id / location_export.TRACKS_FILE
        )
        with self._lock:
            self._tables[recording_id] = table
        return table

    def traces(
        self,
        view: LocationPlotView,
        recording_ids: list[str] | None = None,
        progress: Progress = None,
    ) -> tuple[list[plots.Trace], list[str]]:
        """The animal of each selected recording — its centre, or a path per keypoint the
        view asks for — and a note per path left out.

        Raises:
            ValueError: A recording is not one of the run's completed ones, or a keypoint the
                run did not export.
        """
        chosen = self._chosen(recording_ids)
        names = self.names()
        uses_points = bool(plots.position_points(view))
        out, notes = [], []
        for index, sid in enumerate(chosen):
            if progress is not None:
                progress("reading", index / len(chosen), names[sid])
            manifest = self._manifest(sid)
            frame = plots.read_frame(
                manifest, f"{names[sid]}'s {location_export.MANIFEST_FILE}"
            )
            found, left_out = plots.recording_traces(
                self._table(sid),
                frame,
                view,
                recording_id=sid,
                name=names[sid],
                keypoints=self._keypoints(sid) if uses_points else None,
                keypoint_names=[str(n) for n in manifest.get("keypoint_names") or []],
            )
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
        """``view`` of the selected recordings (default: all) as a ``PlotSpec``; a recording
        left out is named in the spec's note, and so is a body measured with thin parts
        removed.

        Raises:
            ValueError: An unknown recording or ``group_by`` column, nothing to plot, or units
                the plot cannot be given in.
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
        traces, notes = self.traces(view, recording_ids, progress)
        if progress is not None:
            progress("measuring", 0.0, view.kind)
        spec = plots.build_plot(traces, view, values, notes=notes)
        if self.thin_parts_removed:
            note = f"{THIN_PARTS_NOTE}. {spec.note}" if spec.note else THIN_PARTS_NOTE
            spec = dataclasses.replace(spec, note=note)
        if progress is not None:
            progress("measuring", 1.0, view.kind)
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
        pictures (matplotlib), and ``html``, one self-contained interactive page (plotly) that
        opens in any browser with no network: hover to read a point, zoom, and turn a 3-D plot,
        every panel with it.

        Raises:
            ValueError: A format that is not one of :data:`FORMATS`, or what :meth:`build`
                raises.
        """
        from detail.report import plot_html, plot_render

        unknown = [f for f in formats if f not in FORMATS]
        if unknown or not formats:
            raise ValueError(
                f"export formats are {', '.join(FORMATS)}; not {', '.join(unknown) or 'none'}"
            )
        spec = self.build(view, recording_ids, progress)
        pictures = tuple(f for f in formats if f != "html")
        paths = (
            plot_render.render(
                spec, out_stem, formats=pictures, theme=theme, progress=progress
            )
            if pictures
            else []
        )
        if "html" in formats:
            if progress is not None:
                progress("writing", 0.0, "html")
            paths.append(
                plot_html.render_html(
                    spec, Path(out_stem).with_suffix(".html"), theme=theme
                )
            )
            if progress is not None:
                progress("writing", 1.0, "html")
        return paths
