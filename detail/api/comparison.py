"""Run and hold a comparison (§24): group statistics across recording metadata.

The question both research tasks exist to serve — *does this differ between these groups?* — turned
into a saved object. §23 measures the §19 artifact (or `analysis/node_measures.py` measures a
clustering run's node cache), `analysis/compare.py` collapses and groups it and resolves the
design, and this writes the record plus its tidy `stats.csv`. The publication tree (tables,
figures, summary, manifest) is an on-demand `export`, re-runnable from stored data with no
recompute.

**Not a benchmark** (§18, since removed), which compared *method variants* against ground truth and
needed labels a user of the querying and clustering modules does not have. Same export primitives,
nothing else in common.

**Scope is descriptive.** No tests, no corrections, no models: behavioral counts are overdispersed
and sessions within an animal are not independent, so getting inference right needs choices this
tool has no business making for a user. `stats.csv` — long form, one row per observation with its
factors attached — is the handoff to a mixed model elsewhere.

**One computation, two uses.** :func:`prepare_comparison` measures, groups and resolves the
design without writing anything; :func:`run_comparison` writes what it prepared, and
:func:`screen_comparison` ranks it live — the difference screen (§22) a tab reads. Both rank through
`comparison_report.rank_record`, the one record → ranking mapping the export and the detail page
also use, so a live screen and the saved comparison cannot disagree.
"""

import hashlib
import json
import math
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from detail.analysis import cohort, compare, measures, node_measures, provenance
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
NODES_FILENAME = "nodes.csv"

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


def _node_view(experiment, source_run_id: str):
    """The clustering run's node view, or why its nodes cannot be screened (§24 *The node
    source*). A stale cache is refused, naming what changed, never silently read."""
    run = experiment.authored_run(source_run_id)
    if run is None or not hasattr(run, "node_view"):
        raise ComparisonRunError(
            f"run {source_run_id!r} is not a clustering run, so it has no nodes to screen. The "
            f"node source needs a temporal clustering run; source 'labels' measures any "
            f"run's behavior output."
        )
    if not run.has_node_cache():
        raise ComparisonRunError(
            f"run {source_run_id!r} has no node cache to measure. Rebuild it (the workspace's "
            f"floor control, or `rebuild_node_cache`) and run the comparison again."
        )
    view = run.node_view()
    if view.stale:
        raise ComparisonRunError(
            f"run {source_run_id!r}'s node cache is stale: {view.stale}. Rebuild the node cache "
            f"before screening its nodes, or the ranking describes a fit that is gone."
        )
    return view


def node_source_hash(built_from: dict) -> str:
    """What ``drift`` compares for the node source: a digest of what the cache was built from."""
    return hashlib.sha256(
        json.dumps(dict(built_from), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _screened_labels(
    experiment, source_run_id: str, snapshot_id: str | None, *, columns
) -> dict[str, list[dict]]:
    """``{label: [origin, ...]}`` for the labels of the measured artifact whose clusters a
    difference screen over one of ``columns`` chose (``run_labels.screened``), read from the
    state the snapshot froze -- the labels *as measured*, not as they are now."""
    from detail.storage import run_labels

    if snapshot_id is None:
        return {}
    run_dir = analysis_store.analysis_dir(experiment.dir, source_run_id)
    try:
        record = snapshots_store.read_snapshot(run_dir, snapshot_id)
    except (OSError, ValueError, KeyError):
        return {}
    return run_labels.screened(record.get("state") or {}, columns)


def kept_nodes_of(ranking) -> list[int]:
    """The node ids a node screen's ranking kept, in rank order — what the filter dock turns
    into leaves. Empty for ``None`` (nothing to rank) or a ranking of labels.

    Takes the ranking rather than the comparison: re-ranking re-reads ``stats.csv`` and redraws
    the null, seconds over a large family, so a caller holding one ranks once."""
    if ranking is None:
        return []
    out = []
    for item in ranking.kept_items():
        node = node_measures.node_of(item.measure_key[1])
        if node is not None:
            out.append(node)
    return out


def label_origin(experiment, comparison_id: str, node: int, *, ancestors_of):
    """The origin a label on ``node`` carries from the saved screen ``comparison_id``
    (``run_labels.LabelOrigin``), or ``None`` when the screen did not point at the node -- it
    is not a cluster the screen kept, nor under or above one. ``ancestors_of(node)`` is the
    run's root-first path above a node (``AuthoredRun.ancestors_of``).

    Raises:
        ValueError: ``comparison_id`` is not a difference screen (``source: nodes``).
    """
    return label_origins(experiment, comparison_id, [node], ancestors_of=ancestors_of)[
        node
    ]


def label_origins(
    experiment, comparison_id: str, nodes: Sequence[int], *, ancestors_of
) -> dict:
    """:func:`label_origin` for each of ``nodes``, ``{node: LabelOrigin | None}``, ranking the
    screen once however many nodes are named.

    Raises:
        ValueError: ``comparison_id`` is not a difference screen (``source: nodes``).
    """
    from detail.analysis import comparison_report
    from detail.storage import run_labels

    comparison = experiment.get_comparison(comparison_id)
    record = comparison.record()
    if record.get("source") != "nodes":
        raise ValueError(
            f"{comparison_id} compares labels, not clusters; a label's origin is a screen "
            f"over every cluster (measure: every cluster)"
        )
    ranking = comparison.ranking()
    kept_texts: dict[int, str] = {}
    if ranking is not None:
        for item in ranking.kept_items():
            kept = node_measures.node_of(item.measure_key[1])
            if kept is not None:
                kept_texts[kept] = comparison_report.suggestion(item, ranking, record)
    return {
        node: run_labels.label_origin(
            node,
            comparison=comparison_id,
            kept_texts=kept_texts,
            column=str(record.get("design_column") or ""),
            screen=screen_label(record),
            ancestors_of=ancestors_of,
        )
        for node in nodes
    }


def labels_from_screen(experiment, comparison_id: str) -> list[dict]:
    """The labels written from the saved screen ``comparison_id`` as its source run holds them
    now -- ``[{node, name, kept, relation}]``, by node -- what the screen's page lists (§22).
    Empty for a labels comparison, a run that cannot be opened, or a screen nothing was
    labelled from."""
    comparison = experiment.get_comparison(comparison_id)
    try:
        record = comparison.record()
    except (OSError, ValueError, KeyError):
        return []
    if record.get("source") != "nodes":
        return []
    run = experiment.authored_run(record.get("source_run_id") or "")
    if run is None:
        return []
    try:
        state = run.state()
    except (OSError, ValueError, KeyError, FileNotFoundError):
        return []
    names = run.node_names()
    out = []
    for node, origin in sorted(state.label_origins.items(), key=lambda kv: int(kv[0])):
        if origin.get("comparison") != comparison_id or not names.get(node):
            continue
        out.append(
            {
                "node": int(node),
                "name": names[node],
                "kept": str(origin.get("kept", "")),
                "relation": str(origin.get("relation", "")),
            }
        )
    return out


def _check_filter(cfg: ComparisonConfig, available: list[str]) -> list[str]:
    """The recordings a comparison covers, refusing a filter the source cannot honour.

    A *partly* matching filter used to be silently narrowed to the intersection -- so a typo, or
    a cohort where one recording was never analyzed, compared a smaller set than the one asked
    for. And `cfg.as_record()` stores the request, so the record went on naming a recording that
    never entered a single number (Tenet 9: persist what was read, not what was asked).
    """
    keep = set(cfg.recordings) if cfg.recordings else None
    covered = [r for r in available if keep is None or r in keep]
    if not covered:
        raise ComparisonRunError(
            f"no recording in {cfg.source_run_id!r} matches the filter {cfg.recordings!r}. "
            f"The source covers: {sorted(set(available))}"
        )
    if keep is not None:
        missing = sorted(keep - set(available))
        if missing:
            raise ComparisonRunError(
                f"{missing} is not covered by {cfg.source_run_id!r}, so a comparison filtered "
                f"to {sorted(keep)} would silently compare {sorted(set(covered))} instead. "
                f"The source covers: {sorted(set(available))}"
            )
    return covered


@dataclass
class Prepared:
    """A comparison computed and not yet written — what the record and a live screen share.

    ``record`` holds every resolved field the saved record carries apart from its id, provenance
    and timestamp; ``hierarchy`` is the node source's ``nodes.csv`` rows, ``None`` for a flat
    source; ``obs`` is what ``stats.csv`` stores.
    """

    rows: list[dict]
    obs: list
    design: compare.Design
    record: dict
    hierarchy: list[dict] | None
    run_dir: Path
    snapshot_id: str | None
    covered: list[str]
    source_hash: str


def prepare_comparison(
    experiment,
    config,
    *,
    progress: Callable[[str, float, str], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> Prepared:
    """Measure, aggregate, group and resolve the design, writing nothing.

    ``progress(stage, fraction, detail)`` reports through the measuring pass, which reads a frame
    view per unit and is minutes at cohort scale — a silent long op is indistinguishable from a
    hang (CLAUDE.md). ``stop_check()`` is polled per unit; a stopped run measures what it reached
    and raises, so there is no half-record to find later.
    """
    cfg = (
        config
        if isinstance(config, ComparisonConfig)
        else ComparisonConfig.model_validate(config)
    )
    experiment_dir = Path(experiment.dir)

    def report(stage: str, fraction: float, detail: str = "") -> None:
        if progress is not None:
            progress(stage, fraction, detail)

    hierarchy = None
    screened: dict[str, list[dict]] = {}
    if cfg.source == "nodes":
        if cfg.windows.kind != "whole":
            raise ComparisonRunError(
                f"the node source measures the whole recording; windows '{cfg.windows.kind}' "
                f"are not available over nodes (§24)."
            )
        # Refused rather than ignored: a snapshot pins the run's labelled artifact, which the
        # node source does not read, and a screen without its ranking family (share of time)
        # or its presence measure (total time) ranks nothing.
        if cfg.source_snapshot:
            raise ComparisonRunError(
                f"the node source measures run {cfg.source_run_id}'s current node cache; "
                f"snapshot {cfg.source_snapshot!r} pins its labelled artifact, which a screen "
                f"over every cluster does not read. Leave the snapshot unset (§24)."
            )
        if cfg.measures and set(cfg.measures) != set(node_measures.MEASURES):
            raise ComparisonRunError(
                f"the node source always measures {', '.join(node_measures.MEASURES)}; "
                f"got {', '.join(cfg.measures)}. Leave measures unset (§24)."
            )
        report("loading", 0.0, cfg.source_run_id)
        view = _node_view(experiment, cfg.source_run_id)
        run_dir = analysis_store.analysis_dir(experiment_dir, cfg.source_run_id)
        snapshot_id = None
        covered = _check_filter(cfg, list(view.recording_ids))
        chosen = list(node_measures.MEASURES)
        # The node source's own floor (§24): the cache has none since every cluster has clips.
        candidate_floor = float(cfg.candidate_floor_s)
        report("measuring", 0.0, f"{len(covered)} recording(s)")
        rows = node_measures.node_rows(
            view,
            floor_s=candidate_floor,
            measures=chosen,
            recordings=covered,
            progress_cb=lambda done, total: report(
                "measuring", done / max(total, 1), f"{done}/{total} node(s)"
            ),
            stop_check=stop_check,
        )
        hierarchy = node_measures.hierarchy_rows(view, candidate_floor)
        source_hash = node_source_hash(view.built_from)
        ranking_measure = node_measures.RANKING_MEASURE
    else:
        run_dir, snapshot_id = _resolve_artifact(
            experiment, cfg.source_run_id, cfg.source_snapshot, progress=progress
        )
        report("loading", 0.0, cfg.source_run_id)
        artifact = behavior_output.read_behavior_output(run_dir)
        covered = _check_filter(cfg, [row.recording_id for row in artifact.coverage])
        _validate_label_space(artifact.manifest, covered)
        report("measuring", 0.0, f"{len(covered)} recording(s)")
        rows = measures.measure_run(
            run_dir,
            windows_for=lambda cov: cfg.windows.windows_for(
                cov.start_t_ns, cov.end_t_ns
            ),
            measures=cfg.measures,
            progress_cb=lambda done, total: report(
                "measuring", done / max(total, 1), f"{done}/{total} recording(s)"
            ),
            # Per unit, because this is the pass the caller waits on: it reads a frame view each
            # time and is minutes at cohort scale, and a fraction that only moved at its ends left
            # every surface showing 0% for the whole of it.
            stop_check=stop_check,
        )
        source_hash = snapshots_store.artifact_hash(run_dir)
        ranking_measure = None
        screened = _screened_labels(
            experiment, cfg.source_run_id, snapshot_id, columns=cfg.group_by
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
    kept_rows = [r for r in rows if r["recording_id"] in set(covered)]
    design = compare.resolve_design_from_rows(
        kept_rows,
        metadata,
        group_by=cfg.group_by,
        aggregate_to=cfg.aggregate_to,
        levels=cfg.levels,
    )
    # A recording with no value for a grouping column sits outside the comparison, not only
    # the design (§24): its rows reached the observations, the groups, the figures and
    # `stats.csv` as a level with no name. Out before anything is grouped, and named.
    left_out = cohort.left_out(metadata, covered, cfg.group_by)
    outside = set(left_out)
    wanted = (
        set(design.levels)
        if design.levels is not None and design.column is not None
        else None
    )
    column = design.column

    def predicate(factors: dict) -> bool:
        if factors["recording_id"] in outside:
            return False
        return wanted is None or str(factors.get(column, "")).strip() in wanted

    obs = compare.observations(
        rows,
        metadata,
        group_by=cfg.group_by,
        aggregate_to=cfg.aggregate_to,
        recordings=cfg.recordings,
        predicate=predicate,
    )
    summaries = compare.summarize(obs, coverage_by_group=compare.coverage_by_group(obs))
    groups = sorted({s.group for s in summaries})
    # The contrast as run, resolved (Tenet 9): the paired pair, or a two-group between
    # design's two groups, read as the second against the first. A trend has none.
    levels = design.levels
    if (
        levels is None
        and design.kind == compare.BETWEEN
        and design.column
        and len(groups) == 2
    ):
        levels = ("|".join(groups[0]), "|".join(groups[1]))
    n_units = len({o.key for o in obs})
    min_units = max(1, math.ceil(cfg.min_units_fraction * n_units)) if n_units else 1
    report("grouping", 1.0, f"{len(summaries)} group summaries")

    record = {
        **cfg.as_record(),
        # The *resolved* snapshot, not the user's `None` — a record that stored the request
        # rather than what was read could not say which numbers it holds (Tenet 9).
        "source_snapshot": snapshot_id,
        # The resolved design (§24 *Designs*): what ran, in the sentence every surface prints.
        "design": design.kind,
        "design_column": design.column,
        "levels": list(levels) if levels else None,
        "design_reason": design.reason,
        "n_units": n_units,
        "min_units": min_units,
        "ranking_measure": ranking_measure,
        # Ordered as tuples and joined afterwards, never joined and then sorted: `|` sorts
        # after every letter, so the two disagree whenever one group's first component is a
        # prefix of another's, and every reader of this list would be off by a permutation
        # from the observations it renders beside (tests/analysis/test_group_ordering.py).
        "groups": ["|".join(g) for g in groups],
        "n_observations": len(obs),
        # The recordings with no value for a grouping column, left out above (§24).
        "left_out": list(left_out),
    }
    if cfg.source != "nodes":
        # The labels a difference screen over one of these columns chose (§24): a comparison
        # that scores them is scoring what its own contrast selected, and the record is the
        # one place that can say so -- the number itself looks like any other.
        record["screened_labels"] = screened
    return Prepared(
        rows=rows,
        obs=obs,
        design=design,
        record=record,
        hierarchy=hierarchy,
        run_dir=run_dir,
        snapshot_id=snapshot_id,
        covered=covered,
        source_hash=source_hash,
    )


def run_comparison(
    experiment,
    config,
    *,
    progress: Callable[[str, float, str], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> "Comparison":
    """Measure, aggregate, group, and write the record. Returns a :class:`Comparison` handle."""
    prepared = prepare_comparison(
        experiment, config, progress=progress, stop_check=stop_check
    )
    cfg = (
        config
        if isinstance(config, ComparisonConfig)
        else ComparisonConfig.model_validate(config)
    )
    experiment_dir = Path(experiment.dir)

    def report(stage: str, fraction: float, detail: str = "") -> None:
        if progress is not None:
            progress(stage, fraction, detail)

    report("writing", 0.0, "")
    comparison_id = comparisons_store.next_comparison_id(experiment_dir)
    cdir = comparisons_store.comparison_dir(experiment_dir, comparison_id)
    cdir.mkdir(parents=True, exist_ok=True)
    write_tidy_csv(
        cdir / STATS_FILENAME,
        compare.stats_rows(
            prepared.obs, aggregate_to=cfg.aggregate_to, group_by=cfg.group_by
        ),
        compare.STATS_COLUMNS,
    )
    if prepared.hierarchy is not None:
        write_tidy_csv(
            cdir / NODES_FILENAME, prepared.hierarchy, node_measures.HIERARCHY_COLUMNS
        )
    record = {
        "id": comparison_id,
        **prepared.record,
        # A comparison has no members; the recording lineage, label hashes and video fingerprints
        # capture all the same (§18.5), which is what `drift` re-checks.
        "provenance": provenance.capture(
            experiment_dir,
            recording_ids=sorted(set(prepared.covered)),
            members=[],
            # The hash of **what was measured** — the snapshot, not the run (or the node cache's
            # digest, for the node source). `drift` compares it against where the run is *now*,
            # and that asymmetry is the whole mechanism.
            source={
                "run_id": cfg.source_run_id,
                "snapshot_id": prepared.snapshot_id,
                "content_hash": prepared.source_hash,
            },
        ),
        "status": "complete",
        "created_at": utc_timestamp(),
    }
    comparisons_store.write_comparison_yaml(cdir, record)
    report("writing", 1.0, comparison_id)
    return Comparison(experiment, comparison_id)


@dataclass
class Screen:
    """A live difference screen (§22): the ranking, with the observations and resolved record it
    was computed from. Nothing on disk; ``save`` is :func:`run_comparison` with the same config.
    """

    ranking: object | None
    obs: list
    record: dict
    hierarchy: list[dict] | None


def screen_comparison(
    experiment,
    config,
    *,
    progress: Callable[[str, float, str], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> Screen:
    """The same computation as :func:`run_comparison`, ranked and not written.

    Through `comparison_report.rank_record`, the one record → ranking mapping the saved
    comparison's export and detail page use, so a screen and the comparison saved from it cannot
    disagree (§22 *The difference screen*).
    """
    from detail.analysis import comparison_report

    prepared = prepare_comparison(
        experiment, config, progress=progress, stop_check=stop_check
    )
    ancestors = (
        node_measures.ancestors_from_rows(prepared.hierarchy)
        if prepared.hierarchy is not None
        else None
    )
    ranking = comparison_report.rank_record(prepared.obs, prepared.record, ancestors)
    return Screen(ranking, prepared.obs, prepared.record, prepared.hierarchy)


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

    def ranking(self):
        """This comparison's ranking with its floor, re-ranked from its stored observations
        through `comparison_report.rank_record` — the one record → ranking mapping (§22) — or
        ``None`` where there is nothing to rank. Seconds over a large family: call it off the
        UI thread."""
        from detail.analysis import comparison_report

        obs = comparison_report._observations(comparison_report._read_stats(self.dir))
        return comparison_report.rank_record(
            obs, self.record(), comparison_report.read_hierarchy(self.dir)
        )

    def kept_nodes(self) -> list[int]:
        """The node ids a node screen kept, in rank order — what the filter dock turns into
        leaves. Empty for a labels comparison, or a screen that kept nothing. Ranks; a caller
        already holding :meth:`ranking` passes it to :func:`kept_nodes_of` instead."""
        return kept_nodes_of(self.ranking())

    def _current_source(self, record: dict) -> dict | None:
        """Where the source is *now* — the run's current artifact, or its current node cache."""
        source_run_id = record.get("source_run_id")
        if not source_run_id:
            return None
        experiment_dir = Path(self._experiment.dir)
        if record.get("source") == "nodes":
            run = self._experiment.authored_run(source_run_id)
            if run is None or not getattr(run, "has_node_cache", lambda: False)():
                return {"run_id": source_run_id}
            return {
                "run_id": source_run_id,
                "content_hash": node_source_hash(run.node_view().built_from),
            }
        return {
            "run_id": source_run_id,
            "content_hash": snapshots_store.artifact_hash(
                analysis_store.analysis_dir(experiment_dir, source_run_id)
            ),
        }

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
        current = provenance.capture(
            experiment_dir,
            recording_ids=sorted(record["provenance"].get("recordings", {})),
            members=[],
            # Deliberately the **run's current** artifact, not the snapshot this comparison read:
            # the question drift answers is whether the run has moved on since.
            source=self._current_source(record),
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


def node_screens(experiment, source_run_id: str) -> list:
    """The saved node screens (`source: nodes`) over one clustering run, newest first — what
    INTERPRET's Differences line lists (§22)."""
    out = []
    for comparison in experiment.comparisons:
        try:
            record = comparison.record()
        except (OSError, ValueError, KeyError):
            continue
        if (
            record.get("source") == "nodes"
            and record.get("source_run_id") == source_run_id
        ):
            out.append(comparison)
    return list(reversed(out))


def screen_label(record: dict) -> str:
    """How a saved screen reads in a list: ``drug vs vehicle · paired``, ``dose · trend`` or
    ``session · profile``."""
    design = record.get("design") or "between"
    levels = record.get("levels") or []
    if design in ("trend", "profile"):
        what = str(record.get("design_column") or "")
    elif len(levels) == 2:
        what = f"{levels[1]} vs {levels[0]}"
    else:
        what = ", ".join(record.get("group_by") or [])
    return f"{what} · {design}"
