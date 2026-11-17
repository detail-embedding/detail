"""Read one analysis run's behavior artifact as numbers (§23).

Measures are pure windowed queries over the §19 artifact -- how many bouts, how much total time, a
rate against observed time, a mean bout duration -- and they have been computable since #207 with
no way to ask for them. The only caller was §24's Comparison, so getting the numbers meant running
a group contrast: naming grouping columns and an aggregation path you might not want, to get back
group statistics rather than the per-unit table underneath them.

This is that table, and the export a person can hand to R or a spreadsheet. It computes nothing
Comparison does not; it stops one layer earlier.

**Read-only, so not an `AuthoredRun`.** That base exists for runs carrying mutable authored state
-- a cutoff, a granularity, a set of names -- with a lifecycle to protect. Measuring authors
nothing and can be asked of the same run any number of times.
"""

from collections.abc import Callable
from pathlib import Path

import yaml

from detail.analysis import measures as measures_lib
from detail.configs.windows import WindowSpec
from detail.storage import analysis_runs as analysis_store
from detail.storage import behavior_output
from detail.storage import frame_view

MEASURES_FILENAME = "measures.csv"
FRAMES_FILENAME = "frames.csv"
MANIFEST_FILENAME = "measures.yaml"
SCHEMA_VERSION = 1

#: The per-frame CSV's columns. `source_label` is the producer's own id and `label` the name a
#: person gave it -- both, because a cluster has an id before it has a name and may never get one.
FRAME_COLUMNS = ["recording_id", "subject", "t_ns", "observed", "source_label", "label"]


class MeasuresError(ValueError):
    """A run could not be measured."""


class Measures:
    """The measures of one analysis run's behavior artifact."""

    def __init__(self, experiment, run_id: str) -> None:
        self._experiment = experiment
        self.id = run_id

    def _run_dir(self) -> Path:
        run_dir = analysis_store.analysis_dir(Path(self._experiment.dir), self.id)
        if not behavior_output.has_behavior_output(run_dir):
            raise MeasuresError(
                f"run {self.id!r} has no behavior output to measure. Measures read the §19 "
                f"artifact (instances + coverage + manifest), which only a producing module "
                f"writes."
            )
        return run_dir

    def _current_run_dir(self, progress=None) -> Path:
        """The run directory, with any frame views an authored edit deferred rebuilt first.

        A windowed measure reads the frame view, and an edit leaves it to be brought current by
        whoever reads next (§21) — which is here, before a single view is opened. Long only then,
        and it reports through ``progress``.
        """
        run_dir = self._run_dir()
        run = self._experiment.authored_run(self.id)
        if run is not None:
            run.bring_frame_views_current(progress=progress)
        return run_dir

    def table(
        self,
        *,
        windows: WindowSpec | None = None,
        measures: list[str] | None = None,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> list[dict]:
        """The tidy table: one row per ``(recording, subject, label, window, measure)``.

        ``windows`` defaults to the whole coverage span. A sub-window needs the producer's frame
        view -- ``coverage.csv`` says how much time was observed and not *where* -- and
        ``measure_run`` refuses rather than dividing by the recording's total.

        ``progress(stage, fraction, detail)`` reports the one long step: rebuilding frame views
        that edits since the last read deferred.
        """
        spec = windows or WindowSpec()
        return measures_lib.measure_run(
            self._current_run_dir(progress),
            windows_for=lambda cov: spec.windows_for(cov.start_t_ns, cov.end_t_ns),
            measures=measures,
        )

    def export(
        self,
        out_dir: Path | str,
        *,
        windows: WindowSpec | None = None,
        measures: list[str] | None = None,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> Path:
        """Write ``measures.csv``, ``frames.csv`` and ``measures.yaml`` into a new folder under
        ``out_dir``; return that folder. Each export gets its own, so a second one — another
        window rule, another run — never replaces the first (#29).

        The frame table is the reason this is an export rather than a single file: the frame view
        is zarr, so a per-frame classification is not otherwise readable outside this tool. The
        instance table is deliberately **not** copied -- the artifact already writes one beside
        the run, and a second copy is two files that can disagree.
        """
        from detail.api import exporters

        run_dir = self._current_run_dir(progress)
        out = exporters.new_export_folder(
            exporters.ExportSubject(
                "analysis_run", Path(self._experiment.dir), self.id
            ),
            out_dir,
            what="measures",
        )
        spec = windows or WindowSpec()

        def report(stage: str, fraction: float, detail: str = "") -> None:
            if progress is not None:
                progress(stage, fraction, detail)

        report("measuring", 0.0, self.id)
        rows = measures_lib.measure_run(
            run_dir,
            windows_for=lambda cov: spec.windows_for(cov.start_t_ns, cov.end_t_ns),
            measures=measures,
        )
        measures_lib.write_measures(out / MEASURES_FILENAME, rows)
        report("measuring", 1.0, f"{len(rows)} row(s)")

        report("expanding frames", 0.0, "")
        self._write_frames(run_dir, out / FRAMES_FILENAME, progress=progress)
        report("expanding frames", 1.0, "")

        (out / MANIFEST_FILENAME).write_text(
            yaml.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.id,
                    "windows": spec.model_dump(),
                    "measures": measures,
                    "files": [MEASURES_FILENAME, FRAMES_FILENAME],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return out

    def _write_frames(self, run_dir: Path, path: Path, *, progress=None) -> None:
        """The frame view as CSV: one row per observed-or-not frame, with what was active on it."""
        import csv

        artifact = behavior_output.read_behavior_output(run_dir)
        names = {
            source: labels
            for source, labels in (artifact.manifest.get("names") or {}).items()
        }
        units = frame_view.discover_frame_views(run_dir)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FRAME_COLUMNS)
            writer.writeheader()
            for index, (recording_id, subject) in enumerate(units):
                if progress is not None:
                    progress(
                        "expanding frames",
                        index / max(len(units), 1),
                        recording_id,
                    )
                view = frame_view.read_frame_view(run_dir, recording_id, subject)
                t_ns = view.t_ns[:]
                observed = view.observed[:]
                columns = {
                    source: view.root[source][:] for source in view.source_labels
                }
                for row in range(len(t_ns)):
                    active = [s for s, col in columns.items() if col[row]]
                    writer.writerow(
                        {
                            "recording_id": recording_id,
                            "subject": "" if subject is None else subject,
                            "t_ns": int(t_ns[row]),
                            "observed": int(bool(observed[row])),
                            "source_label": "|".join(active),
                            "label": "|".join(
                                name
                                for source in active
                                for name in names.get(source, [])
                            ),
                        }
                    )
