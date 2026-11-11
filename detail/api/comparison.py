"""Run and hold a comparison (§24): group statistics across recording metadata.

The question both research tasks exist to serve — *does this differ between these groups?* — turned
into a saved object. §23 measures the §19 artifact, `analysis/compare.py` collapses and groups it,
and this writes the record plus its tidy `stats.csv`. The publication tree (tables, figures,
summary, manifest) is an on-demand `export`, re-runnable from stored data with no recompute.

**Not a benchmark** (§18, since removed), which compared *method variants* against ground truth and
needed labels a user of the querying and clustering modules does not have. Same export primitives,
nothing else in common.

**Scope is descriptive.** No tests, no corrections, no models: behavioral counts are overdispersed
and sessions within an animal are not independent, so getting inference right needs choices this
tool has no business making for a user. `stats.csv` — long form, one row per observation with its
factors attached — is the handoff to a mixed model elsewhere.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from detail.analysis import compare, measures, provenance
from detail.configs.comparison import ComparisonConfig
from detail.core.utils import utc_timestamp
from detail.report.tables import write_tidy_csv
from detail.storage import analysis_runs as analysis_store
from detail.storage import behavior_output
from detail.storage import comparisons as comparisons_store
from detail.storage import experiment as experiment_store
from detail.storage import metadata as metadata_store
from detail.storage import snapshots as snapshots_store

if (
    TYPE_CHECKING
):  # annotation only -- never pull matplotlib into an import of this module
    from detail.report.style import FigureStyle

STATS_FILENAME = "stats.csv"

#: The stages :func:`run_comparison` reports, in the order it emits them. Declared here, beside
#: the calls, because a surface folding a per-stage fraction into a whole-run one needs the order
#: — and the GUI kept its own copy of this tuple, which a renamed or reordered stage would have
#: left silently mis-folding the bar. `tests/api/test_comparison.py` pins it against the literals
#: actually passed below.
STAGES = ("loading", "measuring", "grouping", "writing")


class ComparisonStopped(RuntimeError):
    """Raised when ``stop_check`` ended a comparison before it had measured every unit."""


class ComparisonRunError(ValueError):
    """A comparison could not be run against the artifact or metadata it names."""


def _artifact_dir(experiment_dir: Path, source_run_id: str) -> Path:
    run_dir = analysis_store.analysis_dir(experiment_dir, source_run_id)
    if not behavior_output.has_behavior_output(run_dir):
        raise ComparisonRunError(
            f"run {source_run_id!r} has no behavior output to compare. A comparison reads the "
            f"§19 artifact (instances + coverage + manifest), which only a producing module writes."
        )
    return run_dir


def _resolve_artifact(
    experiment, source_run_id: str, source_snapshot: str | None, *, progress=None
) -> tuple[Path, str | None]:
    """The directory to measure, and the snapshot id that names it (§21).

    A named snapshot resolves to that frozen copy. Otherwise the run's current artifact is
    **pinned first** and the pin is what gets measured, so what the record cites and where the
    numbers came from cannot differ — pinning afterwards would leave a window between the two.

    A snapshot is self-contained (the three artifact files plus the frame view §23 needs), so the
    caller measures it exactly as it would measure a run directory.

    Returns ``(dir, snapshot_id)``. ``snapshot_id`` is ``None`` only when there was nothing to
    pin, which ``_artifact_dir`` has already refused by the time this returns. ``progress`` reports
    the pin's rebuild of any frame views an edit deferred.
    """
    experiment_dir = Path(experiment.dir)
    run_dir = _artifact_dir(experiment_dir, source_run_id)
    if source_snapshot:
        sdir = snapshots_store.snapshot_dir(run_dir, source_snapshot)
        if not behavior_output.has_behavior_output(sdir):
            raise ComparisonRunError(
                f"run {source_run_id!r} has no snapshot {source_snapshot!r}. Its snapshots are: "
                f"{snapshots_store.discover_snapshots(run_dir) or 'none'}"
            )
        return sdir, source_snapshot
    # Pin *through the run handle* where there is one, so the snapshot records the interpretation
    # behind it. Pinning by directory alone leaves `state: {}` — which is empty on exactly the
    # snapshots most worth reading later, the ones a comparison used.
    run = experiment.authored_run(source_run_id)
    if run is not None:
        pinned = run.pin(reason="comparison", progress=progress)
    else:
        pinned = snapshots_store.pin(
            run_dir, reason="comparison", code_sha=provenance.code_version()
        )
    if pinned is None:
        return run_dir, None
    return snapshots_store.snapshot_dir(run_dir, pinned), pinned


def _validate_label_space(manifest: dict, recordings: list[str]) -> None:
    """Refuse to pool a per-recording label space across recordings.

    Cluster 3 in one recording is unrelated to cluster 3 in another when a method labels each
    recording independently (§19, §22), so grouping them would average unrelated things into one
    number that looks exactly like a real one.
    """
    if manifest.get("label_space") == "per_recording" and len(set(recordings)) > 1:
        raise ComparisonRunError(
            "this artifact has a per-recording label space, so its labels do not mean the same "
            "thing in two recordings and cannot be pooled. Compare within one recording, or use "
            "an artifact whose labels are shared (label_space: pooled)."
        )


def run_comparison(
    experiment,
    config,
    *,
    progress: Callable[[str, float, str], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> "Comparison":
    """Measure, aggregate, group, and write the record. Returns a :class:`Comparison` handle.

    ``progress(stage, fraction, detail)`` reports through the measuring pass, which reads a frame
    view per unit and is minutes at cohort scale — a silent long op is indistinguishable from a
    hang (CLAUDE.md). ``stop_check()`` is polled per unit; a stopped run measures what it reached
    and writes nothing, so there is no half-record to find later.
    """
    cfg = (
        config
        if isinstance(config, ComparisonConfig)
        else ComparisonConfig.model_validate(config)
    )
    experiment_dir = Path(experiment.dir)
    run_dir, snapshot_id = _resolve_artifact(
        experiment, cfg.source_run_id, cfg.source_snapshot, progress=progress
    )

    def report(stage: str, fraction: float, detail: str = "") -> None:
        if progress is not None:
            progress(stage, fraction, detail)

    report("loading", 0.0, cfg.source_run_id)
    artifact = behavior_output.read_behavior_output(run_dir)
    keep = set(cfg.recordings) if cfg.recordings else None
    covered = [
        row.recording_id
        for row in artifact.coverage
        if keep is None or row.recording_id in keep
    ]
    available = sorted({row.recording_id for row in artifact.coverage})
    if not covered:
        raise ComparisonRunError(
            f"no recording in {cfg.source_run_id!r} matches the filter {cfg.recordings!r}. "
            f"The artifact covers: {available}"
        )
    if keep is not None:
        # A *partly* matching filter used to be silently narrowed to the intersection -- so a
        # typo, or a cohort where one recording was never analyzed, compared a smaller set than
        # the one asked for. And `cfg.as_record()` stores the request, so the record went on
        # naming a recording that never entered a single number (Tenet 9: persist what was
        # read, not what was asked, which the `source_snapshot` line below already does).
        missing = sorted(keep - set(available))
        if missing:
            raise ComparisonRunError(
                f"{missing} is not covered by {cfg.source_run_id!r}, so a comparison filtered "
                f"to {sorted(keep)} would silently compare {sorted(set(covered))} instead. "
                f"The artifact covers: {available}"
            )
    _validate_label_space(artifact.manifest, covered)

    report("measuring", 0.0, f"{len(covered)} recording(s)")
    rows = measures.measure_run(
        run_dir,
        windows_for=lambda cov: cfg.windows.windows_for(cov.start_t_ns, cov.end_t_ns),
        measures=cfg.measures,
        progress_cb=lambda done, total: report(
            "measuring", done / max(total, 1), f"{done}/{total} recording(s)"
        ),
        # Per unit, because this is the pass the caller waits on: it reads a frame view each
        # time and is minutes at cohort scale, and a fraction that only moved at its ends left
        # every surface showing 0% for the whole of it.
        stop_check=stop_check,
    )
    report("measuring", 1.0, f"{len(rows)} row(s)")
    if stop_check is not None and stop_check():
        # Nothing is written. A stopped pass has measured some prefix of the units, and a record
        # built from it is a comparison over a fraction of the cohort that says `complete` --
        # indistinguishable, later, from one that covered everything. Losing a cancelled run is
        # the cheaper of the two: it is the one the user just asked to stop.
        raise ComparisonStopped(
            f"cancelled after measuring {len(rows)} row(s); nothing was written"
        )

    report("grouping", 0.0, "")
    metadata = metadata_store.read_metadata(experiment_dir)
    obs = compare.observations(
        rows,
        metadata,
        group_by=cfg.group_by,
        aggregate_to=cfg.aggregate_to,
        recordings=cfg.recordings,
    )
    summaries = compare.summarize(obs, coverage_by_group=compare.coverage_by_group(obs))
    report("grouping", 1.0, f"{len(summaries)} group summaries")

    report("writing", 0.0, "")
    comparison_id = comparisons_store.next_comparison_id(experiment_dir)
    cdir = comparisons_store.comparison_dir(experiment_dir, comparison_id)
    cdir.mkdir(parents=True, exist_ok=True)
    write_tidy_csv(
        cdir / STATS_FILENAME,
        compare.stats_rows(obs, aggregate_to=cfg.aggregate_to, group_by=cfg.group_by),
        compare.STATS_COLUMNS,
    )
    record = {
        "id": comparison_id,
        **cfg.as_record(),
        # The *resolved* snapshot, not the user's `None` — a record that stored the request
        # rather than what was read could not say which numbers it holds (Tenet 9).
        "source_snapshot": snapshot_id,
        # Ordered as tuples and joined afterwards, never joined and then sorted: `|` sorts
        # after every letter, so the two disagree whenever one group's first component is a
        # prefix of another's, and every reader of this list would be off by a permutation
        # from the observations it renders beside (tests/analysis/test_group_ordering.py).
        "groups": ["|".join(g) for g in sorted({s.group for s in summaries})],
        "n_observations": len(obs),
        # A comparison has no members; the recording lineage, label hashes and video fingerprints
        # capture all the same (§18.5), which is what `drift` re-checks.
        "provenance": provenance.capture(
            experiment_dir,
            recording_ids=sorted(set(covered)),
            members=[],
            # The hash of **what was measured** — the snapshot, not the run. `drift` compares it
            # against where the run is *now*, and that asymmetry is the whole mechanism.
            source={
                "run_id": cfg.source_run_id,
                "snapshot_id": snapshot_id,
                "content_hash": snapshots_store.artifact_hash(run_dir),
            },
        ),
        "status": "complete",
        "created_at": utc_timestamp(),
    }
    comparisons_store.write_comparison_yaml(cdir, record)
    report("writing", 1.0, comparison_id)
    return Comparison(experiment, comparison_id)


class Comparison:
    """Handle for a saved comparison: read its record, check drift, export tables and figures."""

    def __init__(self, experiment, comparison_id: str) -> None:
        self._experiment = experiment
        self._id = comparison_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def dir(self) -> Path:
        return comparisons_store.comparison_dir(self._experiment.dir, self._id)

    def record(self) -> dict:
        return comparisons_store.read_comparison_yaml(self.dir)

    def drift(self) -> list[str]:
        """What changed since this was run — edited labels, a swapped video, moved code, or the
        source analysis being re-interpreted.

        A stale comparison is indistinguishable from a current one by looking at it, which is why
        every surface that renders these numbers checks (§18.5). None of it means the stored
        numbers are wrong: they were measured from a pinned snapshot and stay exactly reproducible
        (§21). What drift says is that a re-run would not match them.
        """
        record = self.record()
        experiment_dir = Path(self._experiment.dir)
        source_run_id = record.get("source_run_id")
        current = provenance.capture(
            experiment_dir,
            recording_ids=sorted(record["provenance"].get("recordings", {})),
            members=[],
            # Deliberately the **run's current** artifact, not the snapshot this comparison read:
            # the question drift answers is whether the run has moved on since.
            source=(
                {
                    "run_id": source_run_id,
                    "content_hash": snapshots_store.artifact_hash(
                        analysis_store.analysis_dir(experiment_dir, source_run_id)
                    ),
                }
                if source_run_id
                else None
            ),
        )
        return provenance.drift(record.get("provenance", {}), current)

    def export(
        self,
        out_dir: Path | str | None = None,
        *,
        formats: tuple[str, ...] = ("pdf", "svg", "png"),
        style: "FigureStyle | None" = None,
        theme: str | None = None,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> dict:
        """Write the publication tree (tables + figures + summary + manifest); return the manifest.

        Defaults into the comparison's own directory, because §24 makes a comparison a saved object
        *holding its outputs* rather than a record pointing at a separate export.

        Args:
            out_dir: where to write; the comparison's own folder when omitted.
            formats: the figure formats to write.
            style: the figure style (default :data:`detail.report.style.DEFAULT_STYLE`).
            theme: ``"light"`` (white, for print — the default) or ``"dark"`` (for slides).
                Overrides ``style``'s theme when given; recorded in the manifest.
            progress: ``(stage, fraction, detail)``, reported through the render — one
                matplotlib figure per measurement, minutes on a large comparison.

        Raises:
            ValueError: If ``theme`` is not ``"light"`` or ``"dark"``.
        """
        from detail.analysis.comparison_report import export_comparison
        from detail.report.style import DEFAULT_STYLE

        out = Path(out_dir) if out_dir is not None else self.dir
        style = style or DEFAULT_STYLE
        if theme is not None:
            style = style.with_theme(theme)
        return export_comparison(
            Path(self._experiment.dir),
            self.record(),
            out,
            formats=formats,
            style=style,
            progress=progress,
        )
