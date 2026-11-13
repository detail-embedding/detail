"""Temporal clustering (§22) — discover a vocabulary of behaviors, and label bouts with it.

§22's second discovery module; the first, `pipeline/pose_clustering.py`, was removed. Where that
one found postures — a cluster was a shape, an instance a run of frames holding it — this one finds
behaviors: a cluster is a kind of movement and an instance is a **bout**, a stretch of trajectory
between two transitions.

An **authored run** (§21), the same two halves. **The fit** is computed, expensive and immutable —
segmentation plus the O(bouts²) DTW matrix, stored as an opaque shard the module never looks
inside. **Interpretation** is authored, cheap and instant — the granularity and the names — and
every authored change reprojects the §19 artifact.

Three things differ from the removed pose module, and none of them is cosmetic:

- **An instance arrives with its extent.** A bout is already a span, so there is no run-length
  encoding and no minimum-run floor here; the minimum length was applied during segmentation,
  before the comparison existed.
- **Coverage is genuinely partial.** Every frame gets a posture, so a pose run covers everything it
  analyzed. Bouts do not tile a timeline: segments below the minimum length are dropped and
  tracking gaps are not bouts. A temporal run reporting 60% of observed time is correct, not
  broken.
- **The vocabulary can be learned from a sample; the labeling never is.** The comparison is
  quadratic and ``max_segments`` is the only bound on it, so a capped fit forms its clusters from
  part of the data — and then places every remaining bout against them, so the artifact still
  carries an instance per bout. Bounding what a method *learns from* is a quality trade; bounding
  what it *labels* would be missing data, which for a module whose job is finding every occurrence
  is not a trade at all. The method carries a ``subsample_note`` saying which happened, and the
  workspace shows it.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from detail.core import run_selection
from detail.core.installed import is_importable
from detail.core.progress import PhaseClock, ThrottledProgress
from detail.core.utils import (
    coerce_series_key,
    subject_key,
    utc_timestamp,
)
from detail.executor.runners import (
    AnalysisModule,
    job_params,
    register_analysis_module,
)
from detail.pipeline import clustering_remote
from detail.pipeline.behavior_artifact import (
    UnitFrames,
    bring_frame_views_current,
    coverage_for,
    require_embedded_track,
    write_artifact,
)
from detail.storage import analysis_runs as analysis_store
from detail.storage import experiment as experiment_store
from detail.storage import temporal_runs


@dataclass
class TemporalClusteringProgressEvent:
    """Progress for one temporal-clustering fit.

    Field names are a contract with ``gui/workers.py::fit_progress_fraction``. The fit is one
    global operation rather than a per-recording fan-out, so the outer pair counts **phases** and
    the inner pair carries the fraction within one — §22: the fit "reports a *global* phase
    fraction rather than per-recording progress".
    """

    recordings_done: int = 0
    recordings_total: int = 0
    frames_done: int = 0
    frames_total: int = 0
    elapsed_s: float = 0.0
    eta_s: float | None = None
    message: str = ""


@dataclass
class TemporalClusteringJob:
    """One temporal-clustering run: the subjects to fit on, and the method to fit with."""

    job_id: str
    source_recording_ids: list[str]
    # The single id when every recording's run shares it, else None; the per-recording
    # map is what the runner reads (§6.5: runs are matched on settings, not on id).
    embed_run_id: str | None
    track_id: int
    method: str
    method_params: dict = field(default_factory=dict)
    trajectory_split_gap_ns: int | None = None
    created_at: str = ""
    embed_run_ids: dict[str, str] = field(default_factory=dict)
    # Where it runs (§6 "Remote analysis"); None is this machine, and a job written
    # before the field existed reads as that.
    compute: dict | None = None

    def embed_run_for(self, recording_id: str) -> str:
        """``recording_id``'s own embedding run — ``embed_run_ids`` when recorded, else the
        single ``embed_run_id`` a run from before the map meant in every recording."""
        return run_selection.require_run_for_recording(self, recording_id)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "source_recording_ids": list(self.source_recording_ids),
            "embed_run_id": self.embed_run_id,
            "embed_run_ids": dict(self.embed_run_ids),
            "track_id": self.track_id,
            "method": self.method,
            "method_params": dict(self.method_params),
            "trajectory_split_gap_ns": self.trajectory_split_gap_ns,
            "created_at": self.created_at,
            "compute": clustering_remote.compute_dict(self.compute),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemporalClusteringJob":
        return cls(
            job_id=str(data["job_id"]),
            source_recording_ids=list(data.get("source_recording_ids") or []),
            embed_run_id=data.get("embed_run_id") or None,
            embed_run_ids=dict(data.get("embed_run_ids") or {}),
            track_id=coerce_series_key(data.get("track_id", 0)),
            method=str(data["method"]),
            method_params=dict(data.get("method_params") or {}),
            trajectory_split_gap_ns=data.get("trajectory_split_gap_ns"),
            created_at=str(data.get("created_at") or ""),
            compute=data.get("compute"),
        )


def resolve_method(method_id: str):
    """The registered temporal-cluster method, or an error naming the ones there are."""
    from detail.cluster_methods import import_temporal_cluster_methods
    from detail.core.registries import TEMPORAL_CLUSTER_METHODS

    import_temporal_cluster_methods()
    if method_id not in TEMPORAL_CLUSTER_METHODS:
        raise ValueError(
            f"unknown temporal-clustering method {method_id!r}; available: "
            f"{sorted(TEMPORAL_CLUSTER_METHODS)}"
        )
    return TEMPORAL_CLUSTER_METHODS[method_id]()


def append_history(run_dir: Path | str, entry: dict) -> None:
    """Append one operation to ``history.jsonl`` — provenance, never authoritative (§21)."""
    path = Path(run_dir) / temporal_runs.HISTORY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": utc_timestamp(), **entry}) + "\n")
        f.flush()


def load_trajectory(
    experiment_root, embed_run_id: str, recording_id: str, subject: int
):
    """One subject's raw ``(features, frame_idx, t_ns)``.

    Unwindowed on purpose: the window, the pooling and the transition detector are the *method's*
    (§22), so handing it pre-pooled features would apply one method's segmentation choice to every
    method that follows.
    """
    from detail.api.run_preview import load_track_trajectory

    recording_dir = experiment_store.recording_path(experiment_root, recording_id)
    return load_track_trajectory(recording_dir, embed_run_id, subject)


def load_timeline(experiment_root, embed_run_id: str, recording_id: str, subject: int):
    """One subject's ``(frame_idx, t_ns)`` **without** its embeddings.

    Coverage asks *when* a subject was embedded, not what it looked like, and the embedding column
    dwarfs the timeline — one 2.5-hour track is 589 MB and ~0.3 s through the full trajectory
    against 7 ms and under a megabyte for these two columns. Reprojection runs on every authored
    edit, so this is the difference between a name committing instantly and visibly stalling.
    """
    from detail.api.run_preview import load_track_timeline

    recording_dir = experiment_store.recording_path(experiment_root, recording_id)
    return load_track_timeline(recording_dir, embed_run_id, subject)


def project_artifact(
    experiment_root, run_dir, *, progress=None, fit=None, reuse: bool = False
) -> Path:
    """Rebuild the §19 behavior artifact from this run's stored fit.

    Turning a granularity into instances is a re-cluster of a matrix that is already on disk, so a
    change of mind costs a reprojection rather than another fit.

    **With no granularity chosen the artifact is still written**, with real coverage and an empty
    vocabulary — §21 requires it: an ``open`` run *has* an artifact, because the statistics are
    what the user is deciding against.
    """
    run_dir = Path(run_dir)
    state = temporal_runs.read_temporal_state(run_dir)
    record = analysis_store.read_analysis_yaml(run_dir)
    method = resolve_method(state.method)
    gap_ns = (record.get("params") or {}).get("trajectory_split_gap_ns")

    def report(stage: str, fraction: float, detail: str = "") -> None:
        if progress is not None:
            progress(stage, fraction, detail)

    loaded = fit if fit is not None else _load_fit(run_dir, method)

    instances: list = []
    coverage: list = []
    analyzed: dict[tuple, object] = {}
    frames_of: dict[str, UnitFrames] = {}
    recordings = list(state.source_recording_ids)
    for index, recording_id in enumerate(recordings):
        report("projecting", index / max(len(recordings), 1), recording_id)
        frame_idx, t_ns = load_timeline(
            experiment_root,
            state.embed_run_for(recording_id),
            recording_id,
            state.track_id,
        )
        # Coverage is built from these rows, so the frame view must be too, or the windowed
        # denominator disagrees with the observed time it is supposed to refine.
        analyzed[(recording_id, state.track_id)] = t_ns
        frames_of[recording_id] = UnitFrames(t_ns, frame_idx)
        coverage.append(coverage_for(recording_id, state.track_id, t_ns, gap_ns))

    # Labels drive the output (§22): every member reports its deepest labelled node, and the
    # rest is one unlabelled bucket. There is no granularity to wait for, so a fitted run always
    # has instances, labelled or not.
    from detail.api import authored_run

    for instance in authored_run.labelled_instances(loaded, method, state):
        row = _row_for(method, loaded, instance, state, frames_of)
        if row is not None:
            instances.append(row)

    report("writing", 0.0, "behavior output")
    # Every unit of this run reads the same embedding run, which is what the fit was given.
    write_artifact(
        experiment_root,
        run_dir,
        instances=instances,
        coverage=coverage,
        manifest=_manifest(
            record.get("id") or run_dir.name, record, state, method, loaded
        ),
        analyzed_t_ns=analyzed,
        embed_runs={unit: state.embed_run_for(unit[0]) for unit in analyzed},
        gap_ns=gap_ns,
        progress=progress,
        reuse=reuse,
    )
    report("writing", 1.0, "done")
    return run_dir


def project_frames(experiment_root, run_dir, *, progress=None) -> int:
    """Bring the frame views an authored edit deferred up to date; return how many were rebuilt.

    The edit's other half (``write_artifact``'s ``reuse``): it wrote the intervals and left the
    views. Each is rebuilt from the unit's embedding timeline, the same rows its coverage was built from.
    """
    run_dir = Path(run_dir)
    state = temporal_runs.read_temporal_state(run_dir)
    record = analysis_store.read_analysis_yaml(run_dir)
    return bring_frame_views_current(
        experiment_root,
        run_dir,
        rows_for=lambda unit: load_timeline(
            experiment_root, state.embed_run_for(unit[0]), unit[0], unit[1]
        )[1],
        gap_ns=(record.get("params") or {}).get("trajectory_split_gap_ns"),
        progress=progress,
    )


def member_resolver(method, fit, subject):
    """``instance -> (recording_id, subject, start_ns, end_ns)``, asked of the method (§22).

    ``subject`` is the run's own (``state.track_id``), returned for every member: a run has one,
    and the method's ``subject`` is its int bookkeeping.
    """
    subject = subject_key(subject)

    def resolve(instance):
        where = method.resolve(fit, instance)
        if not where:
            return None
        return (
            where["recording_id"],
            subject,
            int(where["t_ns"]),
            int(where["end_t_ns"]),
        )

    return resolve


def _row_for(method, fit, instance, state, frames_of):
    """One ``BehaviorInstance`` from a method instance, resolved back to a recording and time.

    **Asked of the method, not read off the fit.** §22: "the module never looks inside a fit".
    Indexing the fit's own arrays here worked for exactly as long as there was one method — the
    second one stores rows rather than bouts, and the module raised deep inside a reprojection.
    An instance the method cannot place is dropped rather than written, because a row claiming a
    span across two recordings is worse than a missing one.

    ``subject`` comes from ``state.track_id``, not ``where["subject"]`` — a temporal-clustering
    run has exactly one ``track_id`` for its whole lifetime (never per-recording), so the two
    always agree, and ``state.track_id`` is the one that is the real subject: a method's own
    ``subjects`` array is a plain ``int64`` internal grouping key, not a value meant to be read
    back as the subject.
    """
    from detail.api import authored_run
    from detail.storage import behavior_output

    where = method.resolve(fit, instance)
    if not where:
        return None
    # The method's own frame fields are trajectory positions; the video frame comes from time.
    frames = frames_of[where["recording_id"]]
    return behavior_output.BehaviorInstance(
        recording_id=where["recording_id"],
        subject=state.track_id,
        t_ns=int(where["t_ns"]),
        end_t_ns=int(where["end_t_ns"]),
        start_frame=frames.frame_at(where["t_ns"]),
        end_frame=frames.frame_at(where["end_t_ns"]),
        source_label=authored_run.source_label_of(instance.cluster_id, state),
    )


def _manifest(run_id: str, record: dict, state, method, fit=None) -> dict:
    """The §19 manifest for a temporal-clustering run.

    ``overlap`` is ``exclusive``: bouts are disjoint by construction, so no two instances cover the
    same moment. That is a statement about *overlap*, not about completeness — unlike the pose
    module's, whose exclusivity also happens to mean full coverage, these instances leave the gaps
    between bouts unaccounted for and the coverage rows are where that shows.

    No ``score``: a cluster assignment has no confidence, and a number invented to fill the column
    would be read as one. (``medoid_dist`` exists, but "far from the medoid" is not a probability
    and presenting it as one is exactly the mistake §19 warns about.)
    """
    names = {
        str(cluster_id): ([name] if name else [])
        for cluster_id, name in sorted(state.names.items())
    }
    # Whatever the method thinks is worth recording about its own fit. Asked rather than read:
    # the counts that matter differ per method, and reaching into the fit for them is the same
    # layering break `resolve` exists to close.
    counts = method.fit_summary(fit) if fit is not None else {}
    return {
        "producer": {"module_id": TemporalClusteringModule.module_id, "run_id": run_id},
        "origin": "predicted",
        "label_space": "pooled",
        "overlap": "exclusive",
        "vocabulary": sorted({n for n in state.names.values() if n}),
        "names": names,
        # Parallel to `names`: same keys, same order, one path per label. The writer pairs them
        # positionally, so they are built from one sorted iteration rather than two.
        "label_paths": {
            str(cluster_id): ([path] if path else [])
            for cluster_id, path in sorted(state.resolved_paths.items())
        },
        "behavior_types": {n: "state" for n in state.names.values() if n},
        "method": {
            "method_id": state.method,
            "params": dict(state.method_params),
            "granularity": state.granularity,
            **counts,
        },
        "provenance": record.get("provenance") or {},
    }


def _load_fit(run_dir: Path, method):
    """The stored fit, or an error saying the run never finished one."""
    path = temporal_runs.fit_path(run_dir, method.shard_suffix)
    if not path.exists():
        raise FileNotFoundError(
            f"run {Path(run_dir).name} has no stored fit at {path.name} — its fit never "
            "completed, so there is nothing to interpret"
        )
    return method.load(path)


@register_analysis_module
class TemporalClusteringModule(clustering_remote.RemoteClusteringFit, AnalysisModule):
    """§22's temporal clustering: one behavior vocabulary pooled across a recording selection."""

    module_id = "temporal_clustering"
    id_prefix = "an"
    requires = ["embedding"]
    selects = ("embedding", "embed_run_id")
    config_model = None  # set below, after the import that would otherwise cycle
    job_class = TemporalClusteringJob
    # The phases a user waits through, in order — the runner names them via ``self.phases``
    # rather than a local tuple so the declaration and the messages cannot drift apart. The
    # method's own sub-phases (segmenting, comparing, clustering) ride in as the note on
    # "clustering", because the comparison dwarfs everything else and a label frozen through it
    # reads as hung.
    phases = ("loading", "clustering", "persisting", "projecting", "caching")
    progress_event_class = TemporalClusteringProgressEvent
    # frames_total is a per-mille fraction
    progress_counts_frames = False
    # An authored run (§21): interpretation happens in place, so there is no "refine into a new
    # run" — the affordance would imply an immutability this lifecycle does not have.
    mutable = True

    def validate(self, experiment_dir, config: dict) -> dict:
        from detail.api import remote as api_remote
        from detail.configs.temporal_clustering import TemporalClusteringConfig

        from detail.configs.clustering_legacy import drop_legacy_floor

        # A run started from one saved with a size floor carries its key; it means nothing now.
        config = drop_legacy_floor(config)
        experiment_dir = Path(experiment_dir)
        model = TemporalClusteringConfig.model_validate(config)
        # First, so a target not registered here is refused before anything is read.
        compute = api_remote.analysis_compute(
            config.get("compute"), "temporal clustering"
        )
        recordings = list(model.source_recording_ids)
        if not recordings:
            raise ValueError(
                "temporal clustering needs at least one recording selected"
            )

        method = resolve_method(model.method)
        missing = [m for m in method.requires if not is_importable(m)]
        if missing:
            raise ValueError(
                f"temporal-clustering method {model.method!r} needs {', '.join(missing)}, from "
                'the `analysis` extra: install it with `uv pip install -e ".[analysis]"`, or '
                "choose another method"
            )
        # Refused, not dropped: pydantic ignores an unknown field, so a knob the method does not
        # have — or one it no longer has, like codebook_viterbi's `fit_rows` — would do nothing
        # and the run would look as if it had honoured it. Launch only: a stored run's params are
        # read back without this check, so a run recorded before a rename still opens.
        given = set(config.get("method_params") or {})
        unknown = sorted(given - set(method.config_model.model_fields))
        if unknown:
            raise ValueError(
                f"{', '.join(unknown)} is not a parameter of temporal-clustering method "
                f"{model.method!r}; it accepts "
                f"{', '.join(sorted(method.config_model.model_fields))}"
                + (
                    ". `fit_rows` was replaced by `fit_share` (percent of each recording) and "
                    "`codewords`"
                    if "fit_rows" in unknown
                    else ""
                )
            )
        # Each recording's own run with the selected settings (§6.5), refusing rather than
        # guessing when the default is ambiguous (§22, Tenet 9).
        selection = self.resolve_selection(experiment_dir, recordings, config)
        resolved_track_id = require_embedded_track(
            experiment_dir, selection["embed_run_ids"], model.track_id
        )

        # The resolved value is written back (Tenet 9): a default resolved from the experiment's
        # current contents makes the same command mean something different once a run is added.
        return {
            **config,
            **model.model_dump(),
            **selection,
            "track_id": resolved_track_id,
            "compute": compute,
            "method_params": method.config_model()
            .model_validate(config.get("method_params") or {})
            .model_dump(),
        }

    def build_job(self, run_id: str, config: dict) -> TemporalClusteringJob:
        """Carried by name from the validated config, never a hand-listed dict.

        A hand-written mapping is complete until the day a field is added, and nothing fails when
        it stops being — the defect `tests/configs/test_record_completeness.py` exists for.
        """
        return TemporalClusteringJob(
            job_id=run_id,
            source_recording_ids=list(config["source_recording_ids"]),
            embed_run_id=config.get("embed_run_id"),
            embed_run_ids=dict(config.get("embed_run_ids") or {}),
            track_id=coerce_series_key(config.get("track_id", 0)),
            method=str(
                config.get("method") or self.config_model.model_fields["method"].default
            ),
            method_params=dict(config.get("method_params") or {}),
            trajectory_split_gap_ns=config.get("trajectory_split_gap_ns"),
            created_at=utc_timestamp(),
            compute=config.get("compute"),
        )

    def run_dir_for(self, experiment_root, job):
        return analysis_store.analysis_dir(experiment_root, job.job_id)

    def output_manifest(self, run_id: str, config: dict) -> list[dict]:
        from detail.storage.behavior_output import behavior_output_artifacts

        return behavior_output_artifacts()

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        run_dir = Path(run_dir)
        experiment_root = run_dir.parent.parent
        method = resolve_method(job.method)
        emit = ThrottledProgress(progress_cb)
        started = time.monotonic()
        # The phase's own progress over the phase's own time (`PhaseClock`): the phases are
        # weighted equally on the bar but are nothing alike in length, so a whole-run estimate
        # would read minutes on a long caching phase. It also replaces a call that passed
        # `(elapsed, fraction)` to `remaining_seconds(fraction, elapsed)`, which returned None,
        # so a clustering fit never showed an ETA at all.
        clock = PhaseClock(started)
        # ``self.phases`` (declared on the module, §16) rather than a local tuple, so the
        # declaration and the messages driving it cannot drift apart.
        phases = self.phases
        last_message: str | None = None

        def report(phase: str, within: float, note: str = "") -> None:
            nonlocal last_message
            done = phases.index(phase) + max(0.0, min(within, 1.0))
            elapsed = time.monotonic() - started
            message = f"{phase} · {note}" if note else phase
            force = message != last_message
            last_message = message
            emit(
                TemporalClusteringProgressEvent(
                    recordings_done=phases.index(phase),
                    recordings_total=len(phases),
                    frames_done=int(within * 1000),
                    frames_total=1000,
                    elapsed_s=elapsed,
                    eta_s=clock.eta(phase, within, time.monotonic()),
                    message=message,
                ),
                force=force,
            )

        # The method's own `subjects` array is int64 bookkeeping (dtw_finch.py). Every read-out
        # takes the run's own state.track_id instead (`_row_for`, `member_resolver`).
        subject_placeholder = subject_key(job.track_id)
        trajectories = []
        for index, recording_id in enumerate(job.source_recording_ids):
            report(
                "loading", index / max(len(job.source_recording_ids), 1), recording_id
            )
            features, _frame_idx, t_ns = load_trajectory(
                experiment_root,
                job.embed_run_for(recording_id),
                recording_id,
                job.track_id,
            )
            trajectories.append((recording_id, subject_placeholder, features, t_ns))
        report("loading", 1.0)

        fit = method.fit(
            trajectories,
            dict(job.method_params),
            lambda fraction, phase: report("clustering", fraction, phase),
        )

        report("persisting", 0.0)
        temporal_runs.fit_dir(run_dir).mkdir(parents=True, exist_ok=True)
        method.serialize(fit, temporal_runs.fit_path(run_dir, method.shard_suffix))

        state = temporal_runs.TemporalClusteringState(
            embed_run_id=job.embed_run_id,
            embed_run_ids=dict(job.embed_run_ids),
            track_id=job.track_id,
            method=job.method,
            method_params=dict(job.method_params),
            source_recording_ids=list(job.source_recording_ids),
            granularity=None,
        )
        temporal_runs.write_temporal_state(run_dir, state)
        append_history(run_dir, {"op": "fit", "method": job.method})
        report("persisting", 1.0)

        # An authored run's compute never writes a terminal status: the runner puts it back to
        # `open` at the very end and only the user closes it (§21). A run whose fit finished
        # perfectly is still open, because a vocabulary is not a result until somebody has
        # interpreted it. The params go first: `project_artifact` reads them.
        record = analysis_store.read_analysis_yaml(run_dir)
        record["artifacts"] = self.output_manifest(job.job_id, {})
        # Every number this run was launched with, derived from the job rather than
        # hand-listed (Tenet 9: a run is reproducible from its own metadata alone). Without it
        # the record names its recordings and its artifacts and nothing else — which is what
        # `an_17` on disk looks like, and why Refine re-opened such a run at defaults.
        record["params"] = job_params(job)
        analysis_store.write_analysis_yaml(run_dir, record)

        report("projecting", 0.0)
        project_artifact(
            experiment_root,
            run_dir,
            progress=lambda stage, fraction, detail="": report(
                "projecting", fraction, detail
            ),
            fit=fit,
        )
        report("projecting", 1.0)

        report("caching", 0.0)
        # §22: the evidence every node card reads, computed once while the fit is in memory so
        # that selecting a node never waits. Its own phase, naming each step as it goes.
        from detail.api import node_cache

        node_cache.write_for_run(
            experiment_root,
            run_dir,
            method=method,
            fit=fit,
            fit_path=temporal_runs.fit_path(run_dir, method.shard_suffix),
            embed_run_for=job.embed_run_for,
            params=dict(job.method_params),
            subject=job.track_id,
            progress=lambda stage, fraction, detail="": report(
                "caching", fraction, detail
            ),
        )
        report("caching", 1.0)

        # Open only now, with the artifact **and** the node cache on disk. Marked open before
        # them, the run page built the workspace mid-fit: its tabs read an artifact that did not
        # exist yet ("no behavior_output.yaml"), and INTERPRET offered to build a cache the fit
        # was still writing. Until here the record says `running`, which the page shows as a
        # run in progress.
        record = analysis_store.read_analysis_yaml(run_dir)
        record["status"] = "open"
        analysis_store.write_analysis_yaml(run_dir, record)


def _set_config_model() -> None:
    """Attach the config model after import, so the module and its config cannot cycle."""
    from detail.configs.temporal_clustering import TemporalClusteringConfig

    TemporalClusteringModule.config_model = TemporalClusteringConfig


_set_config_model()
