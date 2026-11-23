"""Query (§20) — the cohort search half.

A user points at a few example clips of a behavior they can already name; this searches every
selected `(recording, subject)` for stretches that resemble them, under a method from the §20
registry. It is §19's first producer: calibration (PR 2) turns the stored search results into
the behavior artifact.

**The search stores no cutoff.** A shard holds whatever the method's `SearchResult` is, and the
module never looks inside it. Turning a cutoff into instances is `instances_at`, applied later
and as often as the user likes — which is what stops a change of mind costing another search.

**A unit is `(behavior x recording x subject)`**, not a recording. That is the fan-out axis, the
resume key, and what the progress bar counts: a two-behavior run over six recordings is twelve
searches, and calling them recordings would claim twelve of those.

**A failing behavior does not take the others down.** Shards are per unit, so a behavior whose
search raises leaves the rest complete and reviewable, and the record names which finished. One
failure voiding a cohort-scale search is the outcome resume exists to prevent.

**The module draws a method's background, never the method** (§20). A method that needs one
declares a `SamplingRequest`; `draw_background` draws it **once, before any unit is searched**,
and records strategy, scope, size, seed *and the rows drawn* in `background.npz`. One sample is
shared by every behavior, and it excludes the exemplar spans of every behavior in the run and
**nothing else** — with no labels in the tool the remainder is the unlabeled trajectory, which
is by construction the pool a user could reproduce. Narrowing it by what *looks* like the
behavior would rebuild #191's leak: the pool would be chosen using the answer.

Long trajectories are searched in **chunks**, when the method declares an overlap. The curve
is split; the extraction is not — chunks are merged by the method and the instances come out of
one pass over the whole thing (§20, Parallelism). See `_search_unit` for where the split is
planned and why that is safe to recompute.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detail.analysis import clip_query as clip_query_analysis
from detail.configs.query import DEFAULT_METHOD, QueryConfig
from detail.configs.method_params import resolve_method_params
from detail.core.progress import (
    ThrottledProgress,
    best_effort,
    remaining_seconds,
    split_units,
)
from detail.core.registries import QUERY_METHODS
from detail.core.utils import utc_timestamp
from detail.executor.runners import AnalysisModule, job_params, register_analysis_module
from detail.pipeline.behavior_artifact import (
    bring_frame_views_current,
    coverage_for,
    write_artifact,
)
from detail.query_methods import import_query_methods
from detail.storage import analysis_runs as analysis_store
from detail.storage import experiment as experiment_store
from detail.storage import query_runs

_log = logging.getLogger(__name__)

# Phase labels, in order. §20 requires the search to name its phases and advance them: the
# numeric fraction is pinned through `preparing` (the background draw happens once, before
# fan-out), so a static label there reads as a hang however healthy the run is.
PHASE_LOADING = "loading"
PHASE_PREPARING = "preparing"
PHASE_SEARCHING = "searching"
PHASE_WRITING = "writing"

# There is deliberately no `LOAD_SHARE` here. The trajectory load used to claim a flat 20% of
# a unit — the milestone the *method* could not supply, back when one exemplar over one
# continuous stretch reported only its terminal 1.0. `retrieval.subsequence_dtw` reports from
# inside its own DP now, so that milestone is measured rather than invented, and keeping the
# 20% would be the same defect the other way up: the load is a sliver of the unit, so
# the bar jumped to 20% having done almost nothing and the ETA read a second left with
# most of the search to go. The load reports its progress without claiming a share.


@dataclass
class QueryProgressEvent:
    """Progress for one query run.

    Field names are a contract with `gui/workers.py::fit_progress_fraction`, which folds them
    as `(recordings_done + frames_done / frames_total) / recordings_total`. The outer pair
    counts **search units**, not recordings — the label the GUI renders says so.
    """

    recordings_done: int = 0
    recordings_total: int = 0
    frames_done: int = 0
    frames_total: int = 0
    units_done: int = 0
    units_total: int = 0
    elapsed_s: float = 0.0
    eta_s: float | None = None
    message: str = ""


@dataclass
class SearchUnit:
    """One `(behavior, recording, subject)` search — the fan-out and resume key."""

    behavior_id: str
    recording_id: str
    subject: int

    def to_dict(self) -> dict:
        return {
            "behavior_id": self.behavior_id,
            "recording_id": self.recording_id,
            "subject": self.subject,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SearchUnit":
        return cls(
            behavior_id=str(data["behavior_id"]),
            recording_id=str(data["recording_id"]),
            subject=int(data["subject"]),
        )


@dataclass
class QueryJob:
    """One Query run: every behavior, every searched subject, one method.

    `units` is the fan-out: one entry per `(behavior, recording, subject)`. How a unit is split
    into chunks is decided when it is searched, from the trajectory and the method's declared
    overlap — see `QueryModule._search_unit`.
    """

    job_id: str
    search_recording_ids: list[str]
    embed_run_id: str
    track_id: int
    method: str
    method_params: dict
    behaviors: list[dict]
    units: list[SearchUnit]
    trajectory_split_gap_ns: int | None
    created_at: str
    background_seed: int = 0
    min_match_ns: int = 0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "search_recording_ids": list(self.search_recording_ids),
            "embed_run_id": self.embed_run_id,
            "track_id": self.track_id,
            "method": self.method,
            "method_params": dict(self.method_params),
            "behaviors": [dict(b) for b in self.behaviors],
            "units": [u.to_dict() for u in self.units],
            "trajectory_split_gap_ns": self.trajectory_split_gap_ns,
            "created_at": self.created_at,
            "background_seed": self.background_seed,
            "min_match_ns": self.min_match_ns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QueryJob":
        return cls(
            job_id=str(data["job_id"]),
            search_recording_ids=list(data.get("search_recording_ids") or []),
            embed_run_id=str(data["embed_run_id"]),
            track_id=int(data.get("track_id", 0)),
            method=str(data["method"]),
            method_params=dict(data.get("method_params") or {}),
            behaviors=[dict(b) for b in (data.get("behaviors") or [])],
            units=[SearchUnit.from_dict(u) for u in (data.get("units") or [])],
            trajectory_split_gap_ns=data.get("trajectory_split_gap_ns"),
            created_at=str(data.get("created_at") or ""),
            background_seed=int(data.get("background_seed") or 0),
            min_match_ns=int(data.get("min_match_ns") or 0),
        )


def _min_match_ns(config: dict) -> int:
    """The run's minimum match length in ns, from ``min_match_s`` or an already-resolved
    ``min_match_ns`` (a re-run hands back the validated config)."""
    if config.get("min_match_ns") is not None:
        return int(config["min_match_ns"])
    return round(float(config.get("min_match_s") or 0.0) * 1e9)


def min_match_rows(min_match_ns: int, t_ns: np.ndarray) -> int:
    """The fewest rows a match on this timeline may span to last ``min_match_ns``.

    A match of ``n`` rows lasts ``n`` frame periods — so one row at 30 fps is 33 ms, not the
    zero its inclusive ``t_ns == end_t_ns`` would suggest. Seconds in the config, rows only
    here, where the cadence is known: the same setting is a different row count at every
    cadence (``tests/configs/test_unit_conventions.py``).
    """
    if min_match_ns <= 0:
        return 0
    period = clip_query_analysis.nominal_period_ns(t_ns)
    if period <= 0:
        return 0  # under two rows: no cadence, and no match longer than one row exists
    # The tolerance absorbs integer-ns periods: 1 s at 30 fps is 1e9 / 33_333_333 =
    # 30.0000003 rows, which a bare ceil would make 31.
    return int(np.ceil(min_match_ns / period - 1e-3))


def resolve_method(method_id: str):
    """The registered `QueryMethod` instance for `method_id`.

    Raises:
        ValueError: naming what is registered, since a method id is a user-facing choice.
    """
    import_query_methods()
    cls = QUERY_METHODS.get(method_id)
    if cls is None:
        raise ValueError(
            f"unknown query method {method_id!r}; registered: {sorted(QUERY_METHODS)}"
        )
    return cls()


def append_history(run_dir: Path | str, entry: dict) -> None:
    """Append one operation to `history.jsonl` — provenance, never authoritative (§21).

    §20's storage tree omits this file; §21 requires it of every authored run, and the two
    reconcile toward §21. Nothing reads it to decide anything: it exists so a result can be
    traced back to the state of the run when it was produced.
    """
    from detail.storage.append_log import append_line

    # Through `append_line`, so a line a crash cut short is not glued onto the next one.
    append_line(
        Path(run_dir) / query_runs.HISTORY_FILENAME,
        json.dumps({"at": utc_timestamp(), **entry}),
    )


def _plan_chunks(
    n_rows: int, overlap: int | None, target: int
) -> list[tuple[int, int]]:
    """Half-open row ranges covering `n_rows`, overlapping by `overlap`.

    `overlap is None` means the method declares itself unsplittable, so the unit is searched
    whole however long it is — the case §20 says the module should know about up front rather
    than discover at hour six.

    Chunks **overlap** rather than abut: a match ending near a seam needs its full window of
    context, and the method's declared overlap is the longest match it can produce. Consecutive
    starts advance by `target - overlap`, so the seam between two chunks is covered whole by
    both and no match can be cut by the split.
    """
    if overlap is None or n_rows <= target:
        return [(0, n_rows)]
    stride = max(target - overlap, 1)
    starts = list(range(0, n_rows, stride))
    chunks = [(s, min(s + target, n_rows)) for s in starts]
    # A final stride can land a chunk entirely inside its predecessor's reach; drop it rather
    # than emit a shard whose rows another shard already owns.
    return [c for i, c in enumerate(chunks) if i == 0 or c[1] > chunks[i - 1][1]]


def _capture_provenance(experiment_root, job: QueryJob) -> dict:
    """The §18.5 provenance block for a Query run: code, environment, per-recording lineage.

    Cheap by design — ``file_fingerprint`` is size plus a head/tail hash, never a full read of
    a multi-gigabyte video.

    A recording whose video has moved cannot be fingerprinted, and that must not discard a
    cohort-scale search that has already finished: the missing lineage is recorded as an error
    inside the block rather than raised, because every other claim the artifact makes is still
    true. The catch is narrow on purpose — a moved file (``OSError``) or a drifted record
    (``KeyError`` / ``ValueError``). Anything else is a bug in the capture and should surface.
    """
    from detail.analysis import provenance

    try:
        return provenance.capture(
            experiment_root,
            recording_ids=list(job.search_recording_ids),
            members=[(job.method, job.embed_run_id)],
        )
    except (OSError, KeyError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def unit_timeline(experiment_root, embed_run_id: str, recording_id: str, subject: int):
    """One searched unit's ``(frame_idx, t_ns)``.

    The **timeline**, never the trajectory: reprojection needs to know *when* each searched row
    happened, and the embedding column that comes with the trajectory is 589 MB and 0.3 s
    against under a megabyte and 7 ms for these two. That substitution is only sound because
    the two index rows identically — both walk ``sorted(glob("part_*.zarr"))`` and skip empty
    parts the same way — so row ``k`` is the same row in both. Every projected timestamp rests
    on it, which is why it is a guard and not a comment.
    """
    from detail.api.run_preview import load_track_timeline

    return load_track_timeline(
        experiment_store.recording_path(experiment_root, recording_id),
        embed_run_id,
        subject,
    )


def project_artifact(
    experiment_root, run_dir, *, progress=None, results=None, reuse: bool = False
) -> Path:
    """Rebuild the §19 behavior artifact from this run's stored search results.

    The whole of calibration, and the reason the search stores a cutoff-free result: turning a
    cutoff into instances is a read of what is already on disk, so a change of mind costs a
    reprojection rather than another search.

    **A behavior with no cutoff is absent, not zero.** §19 reads a label that is in the
    manifest's vocabulary and has no rows inside coverage as a *real zero* — the behavior was
    looked for and never happened. An uncalibrated behavior has not been looked for yet, so
    listing it would state something false in a table that stays perfectly well-formed. It
    therefore contributes to no vocabulary, no name, and no row.

    Coverage is written for every searched unit regardless, including units that returned
    nothing at the chosen cutoff. That is what makes a genuine zero readable as one.

    **A unit is searched only where its shard is on disk.** The run's plan lists every unit it
    was asked to search, and a cancelled or partly failed search did not reach all of them. A
    unit missing a shard for any behavior the artifact names is left out of coverage (and so
    states no instances): ``coverage.csv`` is per unit, not per behavior (§19), so a unit one
    named behavior never looked at cannot be a real zero for any of them. With nothing
    calibrated the vocabulary is empty, and every behavior must have searched the unit.

    **A clip a reviewer removed is left out at every cutoff** (§20 *The audit*): a candidate whose
    span is in its behavior's ``removed`` is not an instance, so ``instances.csv``, the instance
    features and the frame view all leave it out. It is an edit to the output only; the search
    results, and every reader of the method's own candidates (the frequency curve, the audit
    pool), still see it.

    With no behavior calibrated the artifact is still written, with real coverage and an empty
    vocabulary — the honest statement that these units were analyzed and nothing is named yet.
    §21 requires it: an ``open`` run *has* an artifact, because the statistics are what the
    user is deciding against.

    ``progress(stage, fraction, detail)`` is driven throughout: this reads a shard and writes a
    frame-view zarr per unit, which on a cohort is long enough that a silent run is
    indistinguishable from a hang. ``results`` lets a caller that already holds the loaded
    search results (a calibration surface, which loaded them to draw its curve) hand them in,
    so committing a cutoff does not re-read and re-extract what is already in memory. It is
    keyed ``{behavior_id: {(recording_id, subject): SearchResult}}`` — exactly what
    ``QueryRun.load_results`` returns per behavior.
    """
    run_dir = Path(run_dir)
    state = query_runs.read_query_state(run_dir)
    record = analysis_store.read_analysis_yaml(run_dir)
    method = resolve_method(state.method)
    gap_ns = (record.get("params") or {}).get("trajectory_split_gap_ns")
    calibrated = [b for b in state.behaviors if b.cutoff is not None]
    removed = {b.id: b.removed_keys() for b in calibrated}
    cached = results or {}

    def report(stage: str, fraction: float, detail: str = "") -> None:
        if progress is not None:
            progress(stage, fraction, detail)

    instances: list = []
    coverage: list = []
    analyzed: dict[tuple, object] = {}
    recordings = list(state.search_recording_ids)
    for index, recording_id in enumerate(recordings):
        report("projecting", index / max(len(recordings), 1), recording_id)
        frame_idx, t_ns = unit_timeline(
            experiment_root, state.embed_run_id, recording_id, state.track_id
        )
        if len(t_ns) == 0:
            continue
        unit = (recording_id, state.track_id)
        if not all(
            _searched(run_dir, method, behavior.id, unit, cached)
            for behavior in (calibrated or state.behaviors)
        ):
            # A behavior the artifact names never searched this unit (its search was cancelled
            # before it, or raised on it): it is missing, not a zero, so it is not covered.
            continue
        analyzed[unit] = t_ns
        coverage.append(coverage_for(recording_id, state.track_id, t_ns, gap_ns))
        for behavior in calibrated:
            result = cached.get(behavior.id, {}).get(unit)
            if result is None:
                result = method.load(
                    query_runs.shard_path(
                        run_dir,
                        behavior.id,
                        recording_id,
                        state.track_id,
                        method.shard_suffix,
                    )
                )
            instances.extend(
                instance
                for instance in instances_from(
                    method.instances_at(result, behavior.cutoff, state.method_params),
                    recording_id=recording_id,
                    subject=state.track_id,
                    behavior_id=behavior.id,
                    t_ns=t_ns,
                    frame_idx=frame_idx,
                )
                if query_runs.clip_key(instance) not in removed[behavior.id]
            )

    report("writing", 0.0, "behavior output")
    # Every unit of this run reads the same embedding run, which is what the search was given.
    write_artifact(
        experiment_root,
        run_dir,
        instances=instances,
        coverage=coverage,
        manifest=_manifest(
            record.get("id") or run_dir.name, record, state, method, calibrated
        ),
        analyzed_t_ns=analyzed,
        embed_runs={unit: state.embed_run_id for unit in analyzed},
        gap_ns=gap_ns,
        progress=progress,
        reuse=reuse,
        example_ids=query_runs.example_instance_ids(state.behaviors, instances),
    )
    report("writing", 1.0, "done")
    return run_dir


def _searched(run_dir, method, behavior_id: str, unit: tuple, cached: dict) -> bool:
    """Whether ``behavior_id``'s search reached ``unit``: its result is in hand or on disk."""
    if unit in cached.get(behavior_id, {}):
        return True
    recording_id, subject = unit
    return query_runs.shard_path(
        run_dir, behavior_id, recording_id, subject, method.shard_suffix
    ).is_file()


def missing_units(run_dir) -> dict[str, list[str]]:
    """``{behavior_id: [recording_id, ...]}`` for every planned unit the search never reached.

    Read from the shards, the evidence, rather than from the record; empty for a search that
    reached every unit. Only the units are read from here: whether a run fell short at all is
    its record's statement (``status``, ``behaviors_failed``), which is what
    ``QueryRun._refuse_if_incomplete`` asks first.
    """
    run_dir = Path(run_dir)
    state = query_runs.read_query_state(run_dir)
    method = resolve_method(state.method)
    missing: dict[str, list[str]] = {}
    for behavior in state.behaviors:
        for recording_id in state.search_recording_ids:
            if not _searched(
                run_dir, method, behavior.id, (recording_id, state.track_id), {}
            ):
                missing.setdefault(behavior.id, []).append(recording_id)
    return missing


def project_frames(experiment_root, run_dir, *, progress=None) -> int:
    """Bring the frame views an authored edit deferred up to date; return how many were rebuilt.

    The edit's other half (``write_artifact``'s ``reuse``): it wrote the intervals and left the
    views. Each is rebuilt from the unit's searched timeline, the same rows its coverage was built from.
    """
    run_dir = Path(run_dir)
    state = query_runs.read_query_state(run_dir)
    record = analysis_store.read_analysis_yaml(run_dir)
    return bring_frame_views_current(
        experiment_root,
        run_dir,
        rows_for=lambda unit: unit_timeline(
            experiment_root, state.embed_run_id, unit[0], unit[1]
        )[1],
        gap_ns=(record.get("params") or {}).get("trajectory_split_gap_ns"),
        progress=progress,
    )


def _manifest(run_id: str, record: dict, state, method, calibrated: list) -> dict:
    """The §19 manifest for a Query run.

    ``score.direction`` is **higher_is_better**. A distance-scored producer would say the
    opposite (§19's example once did, for a removed clip-query module); §20 makes ``confidence`` canonical —
    ``[0, 1]``, higher better, uniform across methods — and it is confidence that lands in the
    score column. Copying the example would invert every ranking built on this artifact, with
    nothing failing anywhere.

    ``overlap`` is ``allowed`` because N behaviors overlap each other the moment there are two,
    and the writer checks ``exclusive`` against the rows. It costs §23's ``transitions``, which
    refuses without ``exclusive`` — worth knowing, and better than a declaration that happens
    to be true only for the runs that searched a single behavior.
    """
    return {
        "producer": {"module_id": QueryModule.module_id, "run_id": run_id},
        "origin": "predicted",
        "label_space": "pooled",
        "overlap": "allowed",
        "vocabulary": sorted({b.name for b in calibrated}),
        "names": {b.id: [b.name] for b in calibrated},
        "behavior_types": {b.name: "state" for b in calibrated},
        "score": {"direction": "higher_is_better", "meaning": method.score_meaning},
        "method": {
            "method_id": state.method,
            "params": dict(state.method_params),
            "cutoffs": {b.id: b.cutoff for b in calibrated},
            # A search setting, so it shaped every extent in the artifact (§20).
            "min_match_s": int((record.get("params") or {}).get("min_match_ns") or 0)
            / 1e9,
        },
        # Captured once, at the end of the search, and copied through on every reprojection:
        # provenance describes the run's *inputs*, which a cutoff does not change, so
        # re-capturing per cutoff would be work that cannot alter its own answer.
        "provenance": record.get("provenance") or {},
    }


@register_analysis_module
class QueryModule(AnalysisModule):
    """§20's cohort search: N behaviors over a recording selection, one method."""

    module_id = "query"
    id_prefix = "an"
    requires = ["embedding"]
    selects = ("embedding", "embed_run_id")
    config_model = QueryConfig
    job_class = QueryJob
    phases = (PHASE_LOADING, PHASE_PREPARING, PHASE_SEARCHING, PHASE_WRITING)
    progress_event_class = QueryProgressEvent
    # frames_total is a per-mille fraction (core.progress.split_units)
    progress_counts_frames = False
    # An authored run (§21): calibration happens in place, so there is no "refine into a new
    # run" — the affordance would imply an immutability this lifecycle does not have.
    mutable = True
    # Rows per chunk before a splittable method's unit is divided. Sized so the DTW cost
    # surface stays well under a gigabyte at typical exemplar lengths, not tuned.
    chunk_target_rows = 200_000

    def validate(self, experiment_dir, config: dict) -> dict:
        from detail.api.run_preview import (
            embedding_track_ids,
            load_track_timeline,
            validate_comparability,
        )

        experiment_dir = Path(experiment_dir)
        # Bounds and enums only. The nested halves — the searched recordings and the
        # behaviors — stay outside the model, so the enriched dict is returned unchanged
        # rather than merging a partial model_dump() over it.
        QueryConfig.model_validate(config)
        search = list(config.get("source_recording_ids") or [])
        embed_run_id = config.get("embed_run_id")
        track_id = int(config.get("track_id", 0))
        if not search:
            raise ValueError("query needs at least one recording to search")
        if not embed_run_id:
            raise ValueError("query needs an embed_run_id")

        # The model carries the default, so a config that names no method gets one rather
        # than a KeyError. `out["method"]` below records what was resolved.
        method = resolve_method(config.get("method") or DEFAULT_METHOD)
        raw_behaviors = list(config.get("behaviors") or [])
        if not raw_behaviors:
            raise ValueError("query needs at least one behavior to search for")

        validate_comparability(
            [
                (experiment_store.recording_path(experiment_dir, sid), embed_run_id)
                for sid in search
            ]
        )
        for sid in search:
            available = embedding_track_ids(
                experiment_store.recording_path(experiment_dir, sid), embed_run_id
            )
            if track_id not in available:
                raise ValueError(
                    f"track {track_id} is not in embedding run {embed_run_id!r} of "
                    f"recording {sid!r} (has tracks {available})"
                )

        gap_ns = config.get("trajectory_split_gap_ns")
        if gap_ns is None and config.get("trajectory_split_gap_s") is not None:
            gap_ns = round(float(config["trajectory_split_gap_s"]) * 1e9)

        # Timelines, not trajectories: a span check needs *when* the subject was embedded, and
        # the embedding column dwarfs it (589 MB and 0.3 s against under a megabyte and 7 ms).
        timelines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for sid in search:
            timelines[sid] = load_track_timeline(
                experiment_store.recording_path(experiment_dir, sid),
                embed_run_id,
                track_id,
            )

        problems: list[str] = []
        behaviors: list[dict] = []
        names_seen: set[str] = set()
        for i, raw in enumerate(raw_behaviors):
            bid = query_runs.behavior_id(i)
            name = (str(raw.get("name")).strip() if raw.get("name") else "") or bid
            if name in names_seen:
                problems.append(f"{name!r}: duplicate behavior name")
            names_seen.add(name)
            examples, issues = self._resolve_clips(
                raw.get("examples") or [],
                name,
                "example",
                search,
                timelines,
                gap_ns,
                method,
            )
            problems.extend(issues)
            negatives, neg_issues = self._resolve_clips(
                raw.get("negatives") or [],
                name,
                "non-example",
                search,
                timelines,
                gap_ns,
                method,
            )
            problems.extend(neg_issues)
            problems.extend(self._check_bounds(method, name, examples, negatives))
            behaviors.append(
                {
                    "id": bid,
                    "name": name,
                    "examples": examples,
                    "negatives": negatives,
                }
            )

        if problems:
            # Every violation together (Tenet 10): a user who marked five behaviors and got
            # two of them wrong should see both, not discover the second after fixing the
            # first and relaunching.
            raise ValueError(
                "query cannot run as configured:\n  " + "\n  ".join(problems)
            )

        out = dict(config)
        out["source_recording_ids"] = search
        out["embed_run_id"] = embed_run_id
        out["track_id"] = track_id
        out["behaviors"] = behaviors
        out["method"] = method.method_id
        out["method_params"] = resolve_method_params(
            method, config.get("method_params")
        )
        out["trajectory_split_gap_ns"] = gap_ns
        out.pop("trajectory_split_gap_s", None)
        out["min_match_ns"] = _min_match_ns(config)
        out.pop("min_match_s", None)
        return out

    def _resolve_clips(
        self, raw, behavior_name, kind, search, timelines, gap_ns, method
    ) -> tuple[list[dict], list[str]]:
        """Normalize clip specs to `t_ns` spans, collecting problems rather than raising.

        Exemplars come from the **search set**: a behavior demonstrated on a recording that is
        not being quantified cannot be checked against what the run returns.

        Whether a clip must sit inside one continuous stretch is **the method's** answer
        (``ExemplarBounds.requires_contiguous``), asked here rather than assumed. Hardcoding
        it refused clips for ``soft_knn``, which pools exemplar frames into an unordered
        index and cannot tell the difference.
        """
        require_contiguous = method.exemplars.requires_contiguous
        out: list[dict] = []
        problems: list[str] = []
        for spec in raw:
            sid = spec.get("recording_id")
            if sid not in search:
                problems.append(
                    f"{behavior_name!r}: {kind} clip is on {sid!r}, which is not being "
                    "searched — mark clips on the recordings this run quantifies"
                )
                continue
            frame_idx, t_ns = timelines[sid]
            if "start_t_ns" in spec and "end_t_ns" in spec:
                start_t_ns, end_t_ns = int(spec["start_t_ns"]), int(spec["end_t_ns"])
            elif "start_frame" in spec and "end_frame" in spec:
                start_t_ns, end_t_ns = clip_query_analysis.frame_span_to_t_ns(
                    frame_idx, t_ns, int(spec["start_frame"]), int(spec["end_frame"])
                )
            else:
                problems.append(
                    f"{behavior_name!r}: {kind} clip needs a (start_t_ns, end_t_ns) or "
                    f"(start_frame, end_frame) span (got {sorted(spec)})"
                )
                continue
            check = clip_query_analysis.check_span(
                frame_idx,
                t_ns,
                start_t_ns,
                end_t_ns,
                gap_ns,
                require_contiguous=require_contiguous,
            )
            if not check.ok:
                problems.append(
                    f"{behavior_name!r}: {kind} clip {sid} "
                    f"{start_t_ns / 1e9:.2f}-{end_t_ns / 1e9:.2f} s {check.reason}"
                )
                continue
            out.append(
                {
                    "recording_id": sid,
                    "track_id": int(spec.get("track_id", 0)),
                    "start_t_ns": int(check.start_t_ns),
                    "end_t_ns": int(check.end_t_ns),
                    "n_rows": int(check.end_row - check.start_row + 1),
                }
            )
        return out, problems

    def _check_bounds(self, method, behavior_name, examples, negatives) -> list[str]:
        """The method's declared `ExemplarBounds` / `NegativeSupport` — a refusal, not a warning.

        Warnings (below ~2 s, fewer than three clips) belong to the marking surface, which can
        say so while the user is still choosing. What a method *requires* is checked here,
        before any compute, naming the bound and what was given.
        """
        problems: list[str] = []
        bounds = method.exemplars
        n_clips, n_rows = len(examples), sum(e["n_rows"] for e in examples)
        if n_clips < bounds.min_clips:
            problems.append(
                f"{behavior_name!r}: {method.method_id} needs at least "
                f"{bounds.min_clips} example clip(s), got {n_clips}"
            )
        if bounds.max_clips is not None and n_clips > bounds.max_clips:
            problems.append(
                f"{behavior_name!r}: {method.method_id} takes at most "
                f"{bounds.max_clips} example clip(s), got {n_clips}"
            )
        if n_clips and n_rows < bounds.min_total_rows:
            problems.append(
                f"{behavior_name!r}: {method.method_id} needs at least "
                f"{bounds.min_total_rows} embedded rows across its examples, got {n_rows}"
            )
        if bounds.max_total_rows is not None and n_rows > bounds.max_total_rows:
            problems.append(
                f"{behavior_name!r}: {method.method_id} takes at most "
                f"{bounds.max_total_rows} embedded rows across its examples, got {n_rows}"
            )

        support = method.negatives
        if negatives and support.support == "unusable":
            problems.append(
                f"{behavior_name!r}: {method.method_id} cannot use non-example clips — it "
                "scores distance to the examples and has nothing to score them against. "
                "Remove them or pick a method whose negatives are 'optional' or 'required'."
            )
        if support.support == "required" and len(negatives) < max(support.min_clips, 1):
            problems.append(
                f"{behavior_name!r}: {method.method_id} requires at least "
                f"{max(support.min_clips, 1)} non-example clip(s), got {len(negatives)}"
            )
        return problems

    def build_job(self, run_id: str, config: dict) -> QueryJob:
        units = [
            SearchUnit(
                behavior_id=behavior["id"],
                recording_id=sid,
                subject=int(config["track_id"]),
            )
            for sid in config["source_recording_ids"]
            for behavior in config["behaviors"]
        ]
        return QueryJob(
            job_id=run_id,
            search_recording_ids=list(config["source_recording_ids"]),
            embed_run_id=config["embed_run_id"],
            track_id=int(config["track_id"]),
            method=config["method"],
            method_params=dict(config.get("method_params") or {}),
            behaviors=[dict(b) for b in config["behaviors"]],
            units=units,
            trajectory_split_gap_ns=config.get("trajectory_split_gap_ns"),
            created_at=utc_timestamp(),
            background_seed=int(config.get("background_seed") or 0),
            min_match_ns=_min_match_ns(config),
        )

    def run_dir_for(self, experiment_root, job):
        return analysis_store.analysis_dir(experiment_root, job.job_id)

    def output_manifest(self, run_id: str, config: dict) -> list[dict]:
        from detail.storage.behavior_output import behavior_output_artifacts

        # Derived, never hand-listed: the artifact grew a file once already (the frame view),
        # and a hand-written list is complete until the day it stops being.
        return behavior_output_artifacts()

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        run_dir = Path(run_dir)
        experiment_root = run_dir.parent.parent
        method = resolve_method(job.method)
        emit = ThrottledProgress(progress_cb)
        started = time.monotonic()
        units_total = max(len(job.units), 1)
        last_message: str | None = None

        # Handed to `_search_unit`, so it is called from *inside* the per-behavior isolation
        # boundary below — the shape that made a bookkeeping error read as a failed behavior in
        # #289. The write it bottoms out in is already best-effort; saying so here keeps the
        # boundary honest whatever `report` grows later.
        @best_effort("query progress")
        def report(done: float, phase: str, note: str = "") -> None:
            nonlocal last_message
            elapsed = time.monotonic() - started
            outer, inner, inner_total = split_units(done, units_total)
            fraction = done / units_total if units_total else 0.0
            message = f"{phase} · {note}" if note else phase
            force = message != last_message
            last_message = message
            emit(
                QueryProgressEvent(
                    recordings_done=outer,
                    recordings_total=units_total,
                    frames_done=inner,
                    frames_total=inner_total,
                    units_done=outer,
                    units_total=units_total,
                    elapsed_s=elapsed,
                    eta_s=remaining_seconds(fraction, elapsed) if units_total else None,
                    message=message,
                ),
                force=force,
            )

        report(0, PHASE_LOADING)
        # The generic GUI launch is module-agnostic and never pre-writes the authored state,
        # so the runner is self-sufficient from the job — the same reason the removed
        # clip_annotation initialized its own.
        if not query_runs.has_query_state(run_dir):
            query_runs.write_query_state(run_dir, _state_from_job(job))
        append_history(run_dir, {"op": "search", "units": len(job.units)})

        report(0, PHASE_PREPARING)
        # The background is a barrier: drawn once, before any unit is searched, so every unit
        # is scored against the same one (§20). It reports its own phase because on a cohort
        # it streams every searched subject, and the numeric fraction is pinned at 0 through
        # all of it — a static label here reads as a hang however healthy the run is.
        if method.sampling is not None:
            report(0, PHASE_PREPARING, f"background · {method.sampling.size} rows")
        background = draw_background(experiment_root, job, method, run_dir)
        states = _prepare_behaviors(experiment_root, job, method, background)

        completed: set[str] = set()
        failed: dict[str, str] = {}
        for index, unit in enumerate(job.units):
            if stop_event is not None and stop_event.is_set():
                break
            note = f"{unit.behavior_id} · {unit.recording_id}"
            report(index, PHASE_SEARCHING, note)
            try:
                self._search_unit(
                    experiment_root,
                    run_dir,
                    job,
                    method,
                    states[unit.behavior_id],
                    unit,
                    lambda inner, _i=index, _n=note: report(
                        _i + inner, PHASE_SEARCHING, _n
                    ),
                )
                completed.add(unit.behavior_id)
            except Exception as exc:  # one behavior's failure is not the run's
                failed.setdefault(unit.behavior_id, f"{type(exc).__name__}: {exc}")
            report(index + 1, PHASE_SEARCHING, note)

        report(units_total, PHASE_WRITING)
        interrupted = stop_event is not None and stop_event.is_set()
        record = analysis_store.read_analysis_yaml(run_dir)
        record["module_id"] = self.module_id
        record["embed_run_id"] = job.embed_run_id
        record["track_id"] = job.track_id
        record["params"] = job_params(job)
        record["behaviors_searched"] = sorted(completed - set(failed))
        record["behaviors_failed"] = dict(sorted(failed.items()))
        record["artifacts"] = self.output_manifest(job.job_id, {})
        # Captured once, here, rather than on every reprojection: it fingerprints the run's
        # *inputs* — code, environment, videos, embedding lineage — and a cutoff changes none
        # of them, so re-capturing per calibration would be work that cannot alter its answer.
        record["provenance"] = _capture_provenance(experiment_root, job)
        # An authored run's compute never writes a terminal status: the runner puts it back to
        # `open` and only the user closes it (§21). `partial` still means interrupted compute,
        # which §19 reads as "no artifact yet".
        record["status"] = "partial" if interrupted else "open"
        analysis_store.write_analysis_yaml(run_dir, record)

        if not interrupted:
            # §21: an `open` run *has* an artifact. Withholding it until the user finalizes
            # would invert the workflow — the statistics are what they are supposed to be
            # looking at while deciding. Nothing is calibrated yet, so this writes real
            # coverage and an empty vocabulary: these units were analyzed, nothing is named.
            report(units_total, PHASE_WRITING, "behavior output")
            project_artifact(experiment_root, run_dir)
        report(units_total, PHASE_WRITING, "done")

    def _search_unit(
        self, experiment_root, run_dir, job, method, state, unit, progress
    ) -> None:
        """Search one `(behavior, recording, subject)`, writing one shard per chunk.

        Resume keys off the shard: a chunk already on disk is never recomputed.

        **The split is planned here, not at `build_job`.** §20 puts it at `build_job` so a
        resume cannot recompute a different plan after the worker count changed — but this
        rule takes no worker count. It is a pure function of the trajectory's row count and
        the method's declared overlap, both fixed for a given job and dataset, so recomputing
        it yields the identical plan and the hazard does not arise. `build_job` also has no
        route to the experiment directory, and giving the job one would put an absolute path
        in `job.json` (a hard rule). If a future split rule *does* depend on the environment,
        it has to move and be recorded.
        """
        from detail.api.run_preview import load_track_trajectory

        emb, _frame_idx, t_ns = load_track_trajectory(
            experiment_store.recording_path(experiment_root, unit.recording_id),
            job.embed_run_id,
            unit.subject,
        )
        # The load claims **no** share of the unit — see the phase constants above for why —
        # but it still reports, so a slow one (a 589 MB subject, a network mount) refreshes
        # `elapsed_s` rather than leaving the last event to go stale for its whole duration.
        # A fraction that has not moved while time passed correctly lengthens the ETA.
        progress(0.0)
        overlap = method.chunk_overlap_rows(state, job.method_params)
        chunks = _plan_chunks(len(t_ns), overlap, self.chunk_target_rows)
        gap = clip_query_analysis._gap_for(t_ns, job.trajectory_split_gap_ns)
        # From the whole unit's timeline, not per chunk, so every chunk of a unit agrees.
        min_rows = min_match_rows(job.min_match_ns, t_ns)
        n_chunks = max(len(chunks), 1)
        merged_path = query_runs.shard_path(
            run_dir,
            unit.behavior_id,
            unit.recording_id,
            unit.subject,
            method.shard_suffix,
        )
        if merged_path.is_file():
            return
        chunk_paths: list[Path] = []
        for k, (lo, hi) in enumerate(chunks):
            path = query_runs.shard_path(
                run_dir,
                unit.behavior_id,
                unit.recording_id,
                unit.subject,
                method.shard_suffix,
                chunk=None if n_chunks == 1 else k,
            )
            chunk_paths.append(path)
            if path.is_file():
                continue
            runs = clip_query_analysis.segment_runs(t_ns[lo:hi], gap)
            result = method.search(
                state,
                emb[lo:hi],
                runs,
                job.method_params,
                row_offset=lo,
                progress_cb=lambda f, _k=k: progress((_k + f) / n_chunks),
                min_rows=min_rows,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            method.serialize(result, path)
        if n_chunks > 1:
            # Merge **here**, once, so a unit is one shard by the time calibration reads it.
            # `merge` takes the prepared state, which the search already holds and reprojection
            # would have to rebuild — loading every exemplar recording's embeddings on every
            # cutoff change, for a step whose answer never varies with the cutoff. Chunks are
            # dropped only after the merged shard lands, so an interrupted merge resumes from
            # them rather than re-searching.
            method.serialize(
                method.merge(
                    [method.load(path) for path in chunk_paths],
                    state,
                    job.method_params,
                ),
                merged_path,
            )
            for path in chunk_paths:
                path.unlink()


def _state_from_job(job: QueryJob) -> query_runs.QueryState:
    return query_runs.QueryState(
        embed_run_id=job.embed_run_id,
        track_id=job.track_id,
        method=job.method,
        method_params=dict(job.method_params),
        behaviors=[
            query_runs.Behavior(
                id=b["id"],
                name=b["name"],
                examples=list(b.get("examples") or []),
                negatives=list(b.get("negatives") or []),
            )
            for b in job.behaviors
        ],
        search_recording_ids=list(job.search_recording_ids),
    )


def _exemplar_rows(t_ns: np.ndarray, behaviors, recording_id: str, subject: int) -> set:
    """Every row of this subject inside **any** behavior's example or non-example clip.

    Any behavior's, not the one being prepared: §20 draws one background for the whole run,
    "excluding the exemplar spans of *every* behavior in the run, not just the one being
    scored. Per-behavior exclusion would make the draw behavior-dependent for no gain."
    """
    taken: set = set()
    for behavior in behaviors:
        for spec in [*behavior.get("examples", []), *(behavior.get("negatives") or [])]:
            if spec["recording_id"] != recording_id or int(
                spec.get("track_id", 0)
            ) != int(subject):
                continue
            lo, hi = clip_query_analysis.span_to_rows(
                t_ns, int(spec["start_t_ns"]), int(spec["end_t_ns"])
            )
            taken.update(range(lo, hi + 1))
    return taken


def _draw_uniform(load, units, behaviors, size: int, seed: int):
    """``size`` rows drawn uniformly from the unlabeled remainder, streaming.

    A seeded reservoir — give every eligible row a random key and keep the ``size`` largest —
    which is a uniform sample without replacement computed in one pass, with **one trajectory
    resident at a time**. The same construction `query_eval` uses; materializing every
    searched subject at once was a memory regression there and would be a worse one here,
    where a run can span a whole cohort.

    **The exemplar spans are excluded and nothing else is.** That is not a simplification: it
    is what makes the pool one a user could reproduce. With no labels in the tool the
    remainder *is* the unlabeled trajectory, positives and all — so the background contains
    the target behavior in proportion to how common it is, which depresses confidence exactly
    where the behavior is frequent. §20 calls that a property of background-using methods
    rather than a defect, and the calibration page surfaces it. Excluding rows because they
    *look* like the behavior would rebuild #191's leak in a new place: the negative pool
    would be chosen using the answer, and the score would stop being reachable by a user.
    """
    rng = np.random.default_rng(seed)
    keys = np.empty(0)
    vectors = None
    addresses: list[tuple[str, int, int]] = []
    for recording_id, subject in units:
        loaded = load(recording_id, subject)
        if loaded is None:
            continue
        emb, t_ns = loaded
        excluded = _exemplar_rows(t_ns, behaviors, recording_id, subject)
        eligible = np.array(
            [r for r in range(len(emb)) if r not in excluded], dtype=np.int64
        )
        if not len(eligible):
            continue
        draw = rng.random(len(eligible))
        vectors = (
            emb[eligible] if vectors is None else np.vstack([vectors, emb[eligible]])
        )
        keys = np.concatenate([keys, draw])
        addresses.extend((recording_id, subject, int(r)) for r in eligible)
        if len(keys) > size:
            keep = np.argpartition(-keys, size - 1)[:size]
            keep.sort()  # keep the record in stream order, so a diff of two draws reads
            vectors, keys = vectors[keep], keys[keep]
            addresses = [addresses[i] for i in keep]
    if vectors is None:
        return None
    return vectors.astype(np.float32), addresses


#: The draw strategies the module implements (§20: "a named strategy from a small registry the
#: module implements"). A method names one through its ``SamplingRequest``; an unknown name is
#: refused at validate rather than silently defaulting to uniform.
SAMPLE_STRATEGIES = {"uniform": _draw_uniform}


def draw_background(experiment_root, job, method, run_dir):
    """The run's background sample, drawn once before fan-out — or ``None``.

    ``None`` when the method asks for none (``sampling is None``), which is every method but
    soft-kNN today, so that path is unchanged.

    **Reloaded rather than redrawn on resume.** A seed reproduces a draw only against an
    unchanged trajectory; reading back what was written makes a resumed run use the same
    background even if a recording has since been re-embedded, and it is what the recorded
    rows describe.
    """
    from detail.api.run_preview import load_track_trajectory

    request = method.sampling
    if request is None:
        return None
    stored = query_runs.read_background(run_dir)
    if stored is not None:
        return stored
    strategy = SAMPLE_STRATEGIES.get(request.strategy)
    if strategy is None:
        raise ValueError(
            f"{method.method_id} asks for the {request.strategy!r} sampling strategy; "
            f"this module implements {sorted(SAMPLE_STRATEGIES)}"
        )

    def load(recording_id: str, subject: int):
        try:
            emb, _frame_idx, t_ns = load_track_trajectory(
                experiment_store.recording_path(experiment_root, recording_id),
                job.embed_run_id,
                subject,
            )
        except (KeyError, ValueError, FileNotFoundError):
            _log.warning(
                "query background: %s track %s unreadable, skipped",
                recording_id,
                subject,
            )
            return None
        return np.asarray(emb, dtype=np.float32), np.asarray(t_ns)

    # `cohort` scope is every searched subject. A background drawn per recording would
    # calibrate confidence to that recording, so one cutoff would mean different things in
    # different animals (§20); the scope is the method's declaration, not the module's choice.
    units = sorted({(u.recording_id, u.subject) for u in job.units})
    drawn = strategy(load, units, job.behaviors, int(request.size), job.background_seed)
    if drawn is None:
        raise ValueError(
            f"{method.method_id} needs a background of {request.size} rows and no searched "
            "subject had any row outside an example clip to draw from"
        )
    vectors, addresses = drawn
    sample = query_runs.Background(
        vectors=vectors,
        recording_ids=[a[0] for a in addresses],
        subjects=[a[1] for a in addresses],
        rows=[a[2] for a in addresses],
        strategy=request.strategy,
        scope=request.scope,
        size=int(request.size),
        seed=int(job.background_seed),
    )
    query_runs.write_background(run_dir, sample)
    return sample


def instances_from(
    candidates, *, recording_id: str, subject: int, behavior_id: str, t_ns, frame_idx
) -> list:
    """A method's candidates as §19 instances, for one unit.

    Factored out because two callers need the *same* conversion at different cutoffs:
    ``project_artifact`` converts at the behavior's chosen cutoff, and the reliability audit
    converts at a permissive one, because its sample must be independent of a cutoff the user
    is still deciding. A second copy of this would drift on the next field §19 adds, and the
    drift would be silent — the fields below are exactly the ones a producer is not allowed to
    set by hand.

    No ``label``, ``instance_id`` or ``audited``: all three are derived, and a passed value is
    **silently discarded** rather than refused. ``_expand`` replaces ``label`` from the
    manifest's names map, so a producer that sets one and expects it to survive is wrong with
    nothing failing. The manifest owns the name; this owns the span.
    """
    from detail.storage import behavior_output

    return [
        behavior_output.BehaviorInstance(
            recording_id=recording_id,
            subject=subject,
            t_ns=int(t_ns[c.start_row]),
            end_t_ns=int(t_ns[c.end_row]),
            start_frame=int(frame_idx[c.start_row]),
            end_frame=int(frame_idx[c.end_row]),
            source_label=behavior_id,
            score=float(c.confidence),
        )
        for c in candidates
    ]


def _prepare_behaviors(experiment_root, job, method, background=None) -> dict:
    """One prepared method state per behavior, from its example clips' embedding rows."""
    from detail.api.run_preview import load_track_trajectory

    cache: dict[tuple[str, int], tuple] = {}

    def rows(spec: dict) -> np.ndarray:
        key = (spec["recording_id"], int(spec.get("track_id", 0)))
        if key not in cache:
            cache[key] = load_track_trajectory(
                experiment_store.recording_path(experiment_root, key[0]),
                job.embed_run_id,
                key[1],
            )
        emb, _frame_idx, t_ns = cache[key]
        lo, hi = clip_query_analysis.span_to_rows(
            t_ns, int(spec["start_t_ns"]), int(spec["end_t_ns"])
        )
        return emb[lo : hi + 1]

    # One sample, shared by every behavior — §20's "one background sample, shared across
    # behaviors". `None` for a method that asked for none, which is the pre-existing path.
    sample = None if background is None else background.vectors
    return {
        b["id"]: method.prepare(
            [rows(e) for e in b["examples"]],
            job.method_params,
            negatives=[rows(n) for n in (b.get("negatives") or [])] or None,
            sample=sample,
        )
        for b in job.behaviors
    }
