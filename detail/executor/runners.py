"""Registry mapping job types to their pipeline runners (Section 6).

A runner does the actual work for a job type; a resolver tells the executor where
the job's run directory is (co-located with its outputs, e.g. detections/det_01/).
Pipeline stages self-register on import — excluding a stage means not importing
its module. Both the executor (parent) and the worker (subprocess) populate this
registry by importing the modules named in ``DETAIL_RUNNER_MODULES``.
"""

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from detail.core.dataclasses import (
    register_job_type,
    register_progress_event_type,
)
from detail.core.utils import utc_timestamp

RUNNER_MODULES_ENV = "DETAIL_RUNNER_MODULES"


@dataclass(frozen=True)
class RunnerSpec:
    run: Callable  # (job, run_dir, progress_cb, stop_event) -> None
    run_dir_for: Callable  # (experiment_root, job) -> Path


JOB_RUNNERS: dict[str, RunnerSpec] = {}


def register_runner(job_type: str, run_dir_for: Callable) -> Callable:
    """Register the runner callable for ``job_type`` along with its run-dir resolver."""

    def decorator(fn: Callable) -> Callable:
        JOB_RUNNERS[job_type] = RunnerSpec(run=fn, run_dir_for=run_dir_for)
        return fn

    return decorator


def import_runner_modules(modules: Iterable[str]) -> None:
    """Import each module so it self-registers its runner(s)."""
    for name in modules:
        importlib.import_module(name)


# --------------------------------------------------------------------------- #
# Processing modules (§6.5)
# --------------------------------------------------------------------------- #
PROCESSING_MODULES: dict[str, "ProcessingModule"] = {}


class ProcessingModule(ABC):
    """One executor-backed pipeline stage, referenced by string id (§6.5).

    A module consumes a single ``upstream`` input and writes its own derived
    output (Tenet 5). Subclasses declare identity + dependency metadata and
    implement the per-stage hooks; the base supplies the standard run-record.
    Registration wires the module registry, ``JOB_RUNNERS``, and the dynamic
    job/progress-event maps in one step — there is no way to half-register.

    Scope: executor-backed, output-producing stages only. Synchronous
    inspection tools (prompt-test §8, render-preview §14) are not modules.
    """

    module_id: str  # registry key, job tag, recording.yaml root, GUI nav id
    id_prefix: str  # "det" -> det_01, det_02, ...
    runs_key: str  # the recording.yaml list this module appends to
    upstream: str | None  # "video" | another module_id | None
    job_class: type  # the typed, serializable Job dataclass
    # Whether this module's progress ``frames_total`` is a count of the frames it processed —
    # the one fact the performance report needs before it may divide by seconds and call the
    # result fps. Many runners use the bar's pair for something else: a fraction out of 1000
    # (``core.progress.split_units``), rows of a table, or a count weighted so a multi-pass
    # stage moves evenly. Reading those as frames invents a rate. Required (no default), for
    # ``phases``' reason: a default is indistinguishable from a module that forgot to say.
    # ``tests/executor/test_progress_counts_frames.py`` is the guard.
    progress_counts_frames: bool
    progress_event_class: type | None = None
    # The pydantic config model (§6.5) — feeds the docs parameter reference; None
    # until the module is migrated off a dict config.
    config_model: type | None = None
    # Run identity (§6.5 "Upstream run specs"): the record keys whose values define
    # this run's spec, and the (upstream_module_id, record_ref_field) its lineage
    # recurses through. A module that roots at the recording's video (detection) has
    # no lineage parent and leaves ``lineage_ref`` None.
    identity_fields: tuple[str, ...] = ()
    # What an identity field *means* when a record predates it. Only for a field whose
    # absence has one honest reading — ``every_n_frames`` missing means the run processed
    # every frame, because that was the only behaviour then. Without this, adding a field
    # to ``identity_fields`` turns every run already on disk into an unresolvable
    # per-recording sentinel (a safe but total false-split: no batch would ever again
    # match two recordings' existing runs). A field with no such reading must be left out
    # and recovered from job.json instead.
    identity_defaults: dict[str, object] = {}
    lineage_ref: tuple[str, str] | None = None
    # Auxiliary inputs (§6.5, §26): ``(module_id, record_ref_field)`` pairs a run *may* read
    # beside its lineage parent — embedding's alignment reads a ``keypoint_track`` run. Each
    # must descend from the same lineage parent (the chain stays linear). A run that used one
    # carries that input's identity under ``"aux"`` in its own spec, resolves a batch
    # ``<module>_spec`` per recording, and is a dependent of it (``_downstream_edges``).
    aux_refs: tuple[tuple[str, str], ...] = ()
    # The stages whose finished run offers this module as its next step in the GUI (*Run X →*).
    # ``None`` = the upstream plus every ``aux_refs`` module, which is right when the order a user
    # meets is the order the data flows. Set it where the two differ: keypoint detect reads only
    # a detection run but is met after tracking, because keypoint track needs both (§26).
    offered_after: tuple[str, ...] | None = None
    # A **batch** module runs once over the whole selected subset instead of one
    # isolated job per recording (§6.5) — for a stage that needs cross-recording state, e.g.
    # a transform fit across all selected recordings. The per-recording hooks below are then
    # unused; ``BatchProcessingModule`` supplies ``validate_batch`` / ``run_batch``.
    batch: bool = False

    @abstractmethod
    def validate(self, recording_dir, recording_data: dict, config: dict) -> dict:
        """Run all pre-run checks (Tenet 10) and return the normalized config.

        Raises specific, actionable errors on bad input. May enrich ``config``
        with derived values that ``build_job`` / ``run_record`` consume.
        """

    @abstractmethod
    def build_job(self, recording_id: str, run_id: str, config: dict):
        """Construct the typed, serializable job from validated config."""

    @abstractmethod
    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        """The stage's compute — invoked by the executor via ``JOB_RUNNERS``."""

    @abstractmethod
    def run_dir_for(self, experiment_root, job):
        """Absolute run directory, co-located with the stage's outputs."""

    @abstractmethod
    def output_path(self, run_id: str) -> str:
        """Recording-relative output directory recorded in recording.yaml."""

    def resolve_shared_config(
        self,
        experiment_dir,
        recording_ids: list[str],
        config: dict,
        on_recording: Callable[[str], None] | None = None,
    ) -> dict:
        """Resolve config values that must be **identical across the selected
        recordings**, once, before the per-recording ``validate`` loop.

        A per-recording ``validate`` cannot see the other recordings, so any
        data-dependent value it derives locally (embedding's shared canvas, sized to
        *this* recording's largest mask) comes out different per recording — and the
        resulting runs are then not comparable, which blocks every cross-recording
        consumer (query, query-eval, clustering). Resolving here and
        enriching ``config`` makes the per-recording resolution deterministic.

        An override that reads each recording calls ``on_recording(recording_id)`` before
        it does, so the launch can say which recording it is on (§6 *Progress reporting*).

        Runs only on the non-batch path (a batch module already sees the whole
        subset in ``validate_batch``). Default: no shared values, config unchanged.
        """
        return config

    def recording_parallelism(self) -> int:
        """Max recordings of this module to run concurrently in one processing run.

        Default 1 (serial). CPU transcode stages override to fan out across cores;
        GPU stages stay 1 because they already parallelize across devices within a
        single job, so concurrent recordings would only oversubscribe the GPU."""
        return 1

    def extra_record_fields(self, config: dict) -> dict:
        """Module-specific keys merged into the recording.yaml run record."""
        return {}

    def run_record(self, run_id: str, config: dict, status: str) -> dict:
        """The recording.yaml run record — a lightweight index. Full inputs and
        params live in the run's job.json (Tenet 9).

        ``processing_run_id`` (the experiment run that launched this per-recording run,
        when batch-launched) lets the GUI show which ``run_NNN`` produced a stage
        run; ``created_at`` dates it. Both feed the run labels.
        """
        record = {
            "id": run_id,
            "status": status,
            "output": self.output_path(run_id),
            "created_at": utc_timestamp(),
            **self.extra_record_fields(config),
        }
        if config.get("processing_run_id") is not None:
            record["processing_run_id"] = config["processing_run_id"]
        return record

    def recompute_identity_from_job(self, job) -> dict:
        """Identity fields that ``extra_record_fields`` derives (not a straight copy
        of config) — recovered from a typed job for *legacy* run records written
        before those fields were recorded (§6.5 "Upstream run specs"). Default none;
        detection overrides for ``prompt_fingerprint``, tracking for ``hyperparams``.
        """
        return {}

    def resolve_upstream_spec(
        self, recording_dir, recording_data: dict, config: dict
    ) -> dict:
        """Resolve a batch ``<upstream>_spec`` in ``config`` to *this* recording's own
        upstream run id, in place (§6.5). A no-op when the module has no lineage
        parent or the config carries an explicit id instead of a spec. Raises
        ``ValueError`` if the recording has no run matching the spec.
        """
        if self.lineage_ref is None and not self.aux_refs:
            return config
        from detail.api import run_spec

        return run_spec.resolve_into_config(self, recording_dir, recording_data, config)


@dataclass
class _BatchNoJob:
    """Placeholder job type for a batch module — registered so the id reserves its slot
    in the shared ``JOB_RUNNERS`` tag space, but never serialized (a batch run has no
    per-recording job; the coordinator calls ``run_batch`` directly)."""


class BatchProcessingModule(ProcessingModule):
    """A processing module that runs **once over the whole selected subset** rather than
    one isolated job per recording (§6.5). Its cross-recording step (e.g. a transform fit
    pooled across the selection) can't ride the per-recording coordinator, so
    ``RunOrchestrator`` calls ``run_batch`` a single time and ``Experiment.run_processing``
    validates the subset once via ``validate_batch``.

    Subclasses implement identity + the two batch hooks; the per-recording
    ``ProcessingModule`` hooks are never invoked and raise if called (a wiring bug).
    Registration is unchanged — the id still reserves its ``JOB_RUNNERS`` slot — so the
    module appears in the launch dropdown like any other.
    """

    batch = True
    job_class = _BatchNoJob

    @abstractmethod
    def validate_batch(
        self, experiment_root, recording_ids: list[str], config: dict
    ) -> dict:
        """Pre-run checks over the whole subset (Tenet 10); returns the normalized
        config stored in the run record (there is no per-recording re-validation)."""

    @abstractmethod
    def run_batch(
        self, experiment, recording_ids: list[str], config: dict, emit_progress, cancel
    ) -> str:
        """Run the stage once over ``recording_ids`` — the shared compute + every recording's
        output. ``emit_progress(stage, fraction, detail)`` drives the aggregate bar;
        ``cancel`` is a pollable stop flag. Returns the produced run id (e.g. ``emb_NN``).
        """

    # The per-recording *job* seams are dead for a batch module — calling one is a wiring
    # bug. `output_path` is deliberately NOT among them: it says where a recording's output
    # lives, and a batch module still writes one output per recording. Sweeping it in here made Experiment.delete_run raise
    # NotImplementedError out of a Qt slot, which PyQt turns into a process abort — so a
    # batch module declares its own, exactly like a per-recording one.
    def _no_per_recording(self, *_a, **_k):
        raise NotImplementedError(
            f"{type(self).__name__} is a batch module — it has no per-recording "
            "validate/build_job/runner; use validate_batch / run_batch"
        )

    validate = _no_per_recording
    build_job = _no_per_recording
    runner = _no_per_recording
    run_dir_for = _no_per_recording


def register_processing_module(cls: type) -> type:
    """Register a ProcessingModule and wire all three seams (§6.5).

    Populates ``PROCESSING_MODULES``, registers the runner in ``JOB_RUNNERS``,
    and registers the module's job (and progress-event) types so they serialize
    without editing core. Decorates the class; instantiates it once. Raises if
    ``module_id`` collides with any already-registered family (the ``JOB_RUNNERS``
    tag space is shared across processing, analysis, and raw runners).
    """
    instance = cls()
    _check_id_free(instance.module_id)
    PROCESSING_MODULES[instance.module_id] = instance
    JOB_RUNNERS[instance.module_id] = RunnerSpec(
        run=instance.runner, run_dir_for=instance.run_dir_for
    )
    register_job_type(instance.module_id, instance.job_class)
    if instance.progress_event_class is not None:
        register_progress_event_type(instance.module_id, instance.progress_event_class)
    return cls


# --------------------------------------------------------------------------- #
# Analysis modules (§16)
# --------------------------------------------------------------------------- #
ANALYSIS_MODULES: dict[str, "AnalysisModule"] = {}


def _check_id_free(module_id: str) -> None:
    """Module ids are experiment-globally unique across all families: the ``JOB_RUNNERS``
    tag space is shared, so a duplicate would silently clobber a runner (last writer
    wins). Turn that into a startup error."""
    if module_id in JOB_RUNNERS:
        raise ValueError(
            f"module id {module_id!r} is already registered; module ids are "
            "experiment-globally unique across processing, analysis, and raw runners"
        )


# Identity, not configuration: these say *which* run this is, not what it did, so they are the
# only job fields a record's ``params`` leaves out.
JOB_ONLY_KEYS = ("job_id", "created_at")


def job_params(job) -> dict:
    """A run record's ``params``, derived from the job rather than hand-listed.

    Every module that hand-wrote this dict eventually drifted from its own config. ``query_eval``
    lost its matcher that way once and was rewritten to carry fields by name; ``cluster_eval``
    lost ``stream_name`` the same way and nobody noticed for 230 runs, because the record stays
    well-formed — it is simply short a key no reader counts.

    A record that cannot name every input that shaped its numbers is not reproducible from its own
    metadata, which Tenet 9 requires. Deriving the dict makes that hold by construction instead of
    by everyone remembering.
    """
    return {k: v for k, v in job.to_dict().items() if k not in JOB_ONLY_KEYS}


class AnalysisModule(ABC):
    """One executor-backed, experiment-scoped analysis (§16).

    Unlike a ``ProcessingModule`` (§6.5), an analysis run is a single
    **experiment-level** job — one ``analysis/<run_id>/``, one executor job (the
    dataset-export/training shape of §12), not a per-recording fan-out. It consumes
    derived data across a *selection* of recordings (referenced by id in ``config``)
    and writes free-form artifacts declared via ``output_manifest`` rather than a
    per-recording zarr. Registration is a strict subset of
    ``register_processing_module`` — none of the recording.yaml record fields
    (``runs_key`` / ``output_path`` / ``run_record``), and ``build_job`` takes no
    ``recording_id``.
    """

    module_id: str  # registry key, job tag, GUI nav id — experiment-globally unique
    id_prefix: str  # "an" -> an_01, an_02, ...
    requires: list[str]  # artifact kinds consumed, e.g. ["tracking"]
    job_class: type  # the typed, serializable Job dataclass
    # The run's named steps, in the order it moves through them (§16, issue #393) — a GUI run
    # page reads this to show a phase in flight *and* what is still ahead, the way
    # ``recording_ids`` lets a processing page show queued recordings. The strings are not new:
    # they are the ``message`` a runner already emits at each phase boundary, moved somewhere
    # readable before the run reaches them rather than invented for this.
    #
    # Required (no default) rather than defaulted to ``()``, for the reason ``metric_history`` is
    # abstract on ``Trainer`` rather than defaulted to "none": a module with genuinely one phase
    # still declares ``("running",)`` or similar, because a default meaning "I have none" is
    # indistinguishable from a module that forgot to say — and the surface then silently renders
    # no phase rows, with nothing failing. ``tests/configs/test_record_completeness.py`` guards
    # every registered module has one.
    phases: tuple[str, ...]
    # Whether this module's progress ``frames_total`` is a count of the frames it processed —
    # the one fact the performance report needs before it may divide by seconds and call the
    # result fps. Many runners use the bar's pair for something else: a fraction out of 1000
    # (``core.progress.split_units``), rows of a table, or a count weighted so a multi-pass
    # stage moves evenly. Reading those as frames invents a rate. Required (no default), for
    # ``phases``' reason: a default is indistinguishable from a module that forgot to say.
    # ``tests/executor/test_progress_counts_frames.py`` is the guard.
    progress_counts_frames: bool
    progress_event_class: type | None = None
    # The pydantic config model (§6.5) — feeds the docs parameter reference; None
    # until the module is migrated off a dict config.
    config_model: type | None = None
    # Run identity (§6.5 "Upstream run specs"): the (upstream_module_id,
    # config_ref_field) this analysis selects per recording — explicit, since the ref
    # field is not always ``{module}_run_id`` (the clip modules use ``embed_run_id``).
    selects: tuple[str, str] | None = None
    # An **authored** run (§21): its compute finishes but the run stays open while a user
    # works on it, and it is edited in place rather than seeded into a new run. Declared here
    # rather than only on the GUI module UI so the API and CLI know it too — a surface that
    # decides "does this run have something to show" by listing status strings goes stale the
    # day a module adds a lifecycle, and nothing fails when it does.
    mutable: bool = False

    @abstractmethod
    def validate(self, experiment_dir, config: dict) -> dict:
        """Confirm the selected recordings + their referenced runs exist and are
        complete (Tenet 10); return the normalized config. Raises specific,
        actionable errors on bad input."""

    def resolve_selection(
        self, experiment_dir, recording_ids: list[str], config: dict
    ) -> dict:
        """The upstream run this analysis reads in each recording, as the config fields a
        ``validate()`` writes back (Tenet 9): ``<ref>s``, the per-recording map, and ``<ref>``,
        the single id when every recording's agrees (else ``None``).

        Resolved on settings, never on the shared per-recording id (§6.5) — see
        ``run_spec.resolve_analysis_selection``, which this drives from ``selects``.
        """
        from detail.api import run_spec
        from detail.core import run_selection

        mapping = run_spec.resolve_analysis_selection(
            self, experiment_dir, recording_ids, config
        )
        _upstream, ref_field = self.selects
        return {
            ref_field: run_selection.uniform_run_id(mapping),
            f"{ref_field}s": mapping,
        }

    @abstractmethod
    def build_job(self, run_id: str, config: dict):
        """Construct the typed, serializable job from validated config."""

    @abstractmethod
    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        """The analysis compute — invoked by the executor via ``JOB_RUNNERS``."""

    @abstractmethod
    def run_dir_for(self, experiment_root, job):
        """Absolute run directory — ``analysis/<run_id>/`` under the experiment root."""

    @abstractmethod
    def output_manifest(self, run_id: str, config: dict) -> list[dict]:
        """Artifacts produced — ``[{path, kind, label}, ...]`` for the result view.
        ``path`` is relative to the run dir (no absolute paths in experiment files)."""


def register_analysis_module(cls: type) -> type:
    """Register an AnalysisModule and wire all three seams (§16).

    A strict subset of ``register_processing_module``: populates
    ``ANALYSIS_MODULES``, registers the runner in ``JOB_RUNNERS``, and registers
    the module's job (and progress-event) types so they serialize without editing
    core. Raises if ``module_id`` collides with any already-registered family.
    """
    instance = cls()
    _check_id_free(instance.module_id)
    ANALYSIS_MODULES[instance.module_id] = instance
    JOB_RUNNERS[instance.module_id] = RunnerSpec(
        run=instance.runner, run_dir_for=instance.run_dir_for
    )
    register_job_type(instance.module_id, instance.job_class)
    if instance.progress_event_class is not None:
        register_progress_event_type(instance.module_id, instance.progress_event_class)
    return cls
