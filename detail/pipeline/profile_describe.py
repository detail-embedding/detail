"""Profile clusters — an analysis module (§16) over a clustering run's tree (§22).

One executor job profiles many nodes of one temporal-clustering run. Its module id is
``profile_describe``, the id it had while it also ran a model's description of each cluster, so
the runs made then still open and their records still read (a record may list a ``describe``
part per cluster, which nothing here reads any more).

Each cluster is the single-cluster path — ``Experiment.profile_cluster`` — with its result
written into **this job's own run directory** (``analysis/<an_NN>/profiles/node_<id>/``, the
same layout) and then **placed** beside the clustering run (``api/profile_describe.place``),
where its Interpret tab reads it. Placement is its own idempotent step, run here after every
cluster; a remote run, whose copy-back returns only its own run directory, runs it after the
copy-back. ``analysis.yaml`` is the batch's record: which run, the settings as resolved, and
each cluster's status and times, rewritten after every cluster so a stopped or killed job still
says which finished.

**Resumable by construction.** A cluster whose profile already answers these settings is skipped
(``api/profile_describe.cluster_states``), so running the job again after a cancel, a crash or a
failed cluster does only what is left.

**A failed cluster is a row, not the end.** A profile with nothing to measure is recorded with
its reason and the job goes on; a Cancel keeps every finished cluster.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

from detail.api import profile_describe as batch
from detail.configs.profile_describe import RETIRED_FIELDS, ProfileDescribeConfig
from detail.core.progress import (
    DONE_PHASE,
    MESSAGE_SEPARATOR,
    WITHIN_RESOLUTION,
    OperationCancelled,
    ThrottledProgress,
    remaining_seconds,
)
from detail.core.utils import utc_timestamp
from detail.executor.runners import (
    AnalysisModule,
    job_params,
    register_analysis_module,
)
from detail.storage import analysis_runs as analysis_store

#: The job's own fields beside the config's: the resolved node list, the source run's
#: recordings (recorded as the run's ``recording_ids``), where it runs, and identity.
_JOB_EXTRA = (
    ("job_id", str),
    ("nodes", list),
    ("source_recording_ids", list),
    ("compute", object),
    ("created_at", str),
)
#: Extras a job written before they existed reads as ``None``.
_OPTIONAL_EXTRA = ("compute",)


def _job_class():
    """The job dataclass: every ``ProfileDescribeConfig`` field by name, so a field the
    profile config gains is carried to the worker and the record without an edit here
    (``tests/configs/test_record_completeness.py``). A ``job.json`` written before the
    description step was cut carries its settings too; ``from_dict`` reads only known fields.
    """
    names = [name for name, _type in _JOB_EXTRA]
    specs = list(_JOB_EXTRA) + [
        (name, object)
        for name in ProfileDescribeConfig.model_fields
        if name not in names
    ]

    def to_dict(self) -> dict:
        """Every field as written to ``job.json``. ``compute`` is held as a dict, and the
        remote shepherd sets it to the cluster's ``ComputeTarget`` before writing the cluster's
        envelope, as it does for every job family: written as its dict, or the envelope could
        not be written at all."""
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        if hasattr(out.get("compute"), "to_dict"):
            out["compute"] = out["compute"].to_dict()
        return out

    @classmethod
    def from_dict(cls, d: dict):
        known = {f.name for f in dataclasses.fields(cls)}
        defaults = ProfileDescribeConfig.model_fields
        values = {}
        for name in known:
            if name in d:
                values[name] = d[name]
            elif name in defaults:  # a job written before the field existed
                values[name] = defaults[name].get_default(call_default_factory=True)
            elif name in _OPTIONAL_EXTRA:
                values[name] = None
            else:
                values[name] = d[name]  # identity: a KeyError is the honest answer
        return cls(**values)

    return dataclasses.make_dataclass(
        "ProfileDescribeJob",
        specs,
        namespace={"to_dict": to_dict, "from_dict": from_dict},
        module=__name__,
    )


ProfileDescribeJob = _job_class()


@dataclass
class ProfileDescribeProgressEvent:
    """Progress in the §16 analysis event's field names (``gui/workers.py::
    fit_progress_fraction`` folds ``(recordings_done + frames_done/frames_total) /
    recordings_total``): the outer pair counts **clusters** finished (skipped ones count from
    the start), the inner pair the share of the running cluster's profile. ``eta_s`` is over the
    work this job does, not the clusters it skipped. ``unit_label`` names the units on every
    surface; ``fold_units`` asks the CLI for one bar over the whole job rather than one per
    cluster."""

    recordings_done: int
    recordings_total: int
    frames_done: int
    frames_total: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None
    unit_label: str = "cluster"
    fold_units: bool = True


#: A profile's own stages, as this job's phases (``batch.PHASES``) name them, and where each
#: sits inside its cluster's share of the bar: ``measuring`` has a fraction (most of the time is
#: reading tracks); the fit and the context have none and hold the bar where it got to (§6
#: rule 1: a phase with no sub-count holds).
_STAGES = {
    "sampling": ("measuring", 0.0),
    "measuring": ("measuring", 0.0),
    "fitting": ("fitting", 0.8),
    "context": ("fitting", 0.95),
}
_MEASURE_SHARE = 0.8


@register_analysis_module
class ProfileDescribeModule(AnalysisModule):
    module_id = batch.MODULE_ID
    config_model = ProfileDescribeConfig
    id_prefix = "an"
    # Reads another analysis run, not a recording's run: nothing a recording must carry.
    requires: list[str] = []
    job_class = ProfileDescribeJob
    phases = batch.PHASES
    progress_event_class = ProfileDescribeProgressEvent
    # frames_total is a WITHIN_RESOLUTION fraction
    progress_counts_frames = False
    # §6 "Remote analysis": the job re-sends the clustering run it reads, and is placed beside
    # that run once it is back.
    remote_capable = True

    def refreshed_inputs(self, experiment_root, job) -> list[Path]:
        """The clustering run: its node cache, and the profiles already placed there, which the
        job's skip check reads."""
        return [analysis_store.analysis_dir(experiment_root, job.source_run_id)]

    def after_copy_back(self, experiment_root, job, run_dir) -> None:
        """Place what the job made on the cluster beside the clustering run here."""
        from detail.api.experiment import Experiment

        batch.place(Experiment(experiment_root), job.job_id)

    def remote_resources(self, job) -> dict:
        """One GPU, the fewest a job may ask for: the work is CPU, and Tillicum refuses a job
        that asks for none. The node's CPUs and memory scale down with it."""
        return {"gpus": 1}

    def validate(self, experiment_dir, config: dict) -> dict:
        """The settings as they will run (Tenet 9): the clusters resolved to a node list, the
        source run's recordings. The job then depends on nothing it was not handed.

        Raises:
            ValueError: Not a clustering run, no node cache, an id outside the tree, no
                cluster to run, or another of these jobs already running on the run.
            FileNotFoundError: The run does not exist.
        """
        from detail.api import remote as api_remote
        from detail.api.experiment import Experiment

        given = dict(config)
        given.pop("source_recording_ids", None)  # read from the run, never chosen
        for retired in RETIRED_FIELDS:  # an older run's settings, from "start from"
            given.pop(retired, None)
        given.pop("served_on", None)
        compute = api_remote.analysis_compute(
            given.pop("compute", None), "profile clusters"
        )
        model = ProfileDescribeConfig.model_validate(given)
        experiment = Experiment(experiment_dir)
        run_id = model.source_run_id
        record = experiment.analysis_record(run_id)
        run = experiment._clustering_run(run_id)
        try:
            view = run.node_view()
        except FileNotFoundError:
            raise ValueError(
                f"{run_id} has no node cache yet; open it once (or rebuild its clips) "
                "before profiling its clusters"
            ) from None
        nodes = batch.select_nodes(view, model.clusters, model.node_ids)
        if not nodes:
            raise ValueError(
                f"{run_id} has no cluster to profile under clusters={model.clusters!r}"
            )
        _refuse_a_second_job(experiment, run_id)
        return {
            **model.model_dump(),
            "nodes": nodes,
            "source_recording_ids": list(record.get("recording_ids") or []),
            "compute": compute,
        }

    def build_job(self, run_id: str, config: dict):
        fields = {f.name for f in dataclasses.fields(ProfileDescribeJob)}
        values = {k: v for k, v in config.items() if k in fields}
        values.update(job_id=run_id, created_at=utc_timestamp())
        return ProfileDescribeJob.from_dict(values)

    def run_dir_for(self, experiment_root, job):
        return analysis_store.analysis_dir(experiment_root, job.job_id)

    def output_manifest(self, run_id: str, config: dict) -> list[dict]:
        """The profiles as the job wrote them, before placement; the placed copies are the
        clustering run's, listed per cluster in ``clusters``."""
        return [
            {
                "path": "profiles",
                "kind": "profiles",
                "label": "Cluster profiles (placed beside the run)",
            }
        ]

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        from detail.api.experiment import Experiment

        run_dir = Path(run_dir)
        experiment = Experiment(
            run_dir.parent.parent
        )  # <experiment>/analysis/<run_id>/
        entries, status = run_clusters(
            experiment, job, run_dir, progress_cb, stop_event
        )
        record = analysis_store.read_analysis_yaml(run_dir)
        record.update(_summary(job, entries))
        record["status"] = status
        record["artifacts"] = self.output_manifest(job.job_id, {})
        # Every setting this run was launched with, derived from the job (Tenet 9).
        record["params"] = job_params(job)
        analysis_store.write_analysis_yaml(run_dir, record)


def _refuse_a_second_job(experiment, run_id: str) -> None:
    """Two of these jobs on one run would profile the same clusters twice and place their
    profiles over each other; the second is refused while the first runs."""
    from detail.core.dataclasses import ACTIVE_STATUSES
    from detail.executor import rundir

    for ref in experiment.analysis_runs:
        if ref.module_id != batch.MODULE_ID or ref.status not in ACTIVE_STATUSES:
            continue
        try:  # the job envelope names it from submit on; the record only once it runs
            source = rundir.read_envelope(experiment.analysis_run_dir(ref.id))[
                "job"
            ].get("source_run_id")
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if source == run_id:
            raise ValueError(
                f"{ref.id} is already profiling {run_id}'s clusters; wait for it or "
                "cancel it"
            )


def _config_of(job) -> ProfileDescribeConfig:
    return ProfileDescribeConfig.model_validate(
        {f: getattr(job, f) for f in ProfileDescribeConfig.model_fields}
    )


def _summary(job, entries: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return {"source_run_id": job.source_run_id, "counts": counts, "clusters": entries}


def _write_progress_record(run_dir: Path, job, entries: list[dict]) -> None:
    """The record as it stands, rewritten after each cluster (atomically), so a run stopped
    or killed part-way still lists which clusters finished."""
    record = analysis_store.read_analysis_yaml(run_dir)
    record.update(_summary(job, entries))
    record["params"] = job_params(job)
    analysis_store.write_analysis_yaml(run_dir, record)


def run_clusters(experiment, job, run_dir, progress_cb, stop_event):
    """Profile each of ``job.nodes``; ``(entries, status)``.

    ``entries`` is one dict per node in order — ``{node, name, status, profile, error}`` — and
    ``status`` the run's: ``complete`` when every cluster finished or was already done,
    ``partial`` when some did and others failed or were not reached, ``failed`` when none did.
    """
    from detail.api import cluster_profile

    config = _config_of(job)
    run_id = job.source_run_id
    run = experiment._clustering_run(run_id)
    names = run.node_names()
    states = {
        s.node: s
        for s in batch.cluster_states(experiment, run_id, list(job.nodes), config)
    }
    entries = [_entry(node, names, states[int(node)]) for node in job.nodes]
    todo = [e for e in entries if e["status"] == "pending"]

    total = len(entries)
    emit = ThrottledProgress(progress_cb)
    started = time.monotonic()
    clock = {"done": total - len(todo), "work": 0}
    current = {"phase": None}

    def report(entry, phase: str, within: float, detail: str = "") -> None:
        if stop_event.is_set():
            raise OperationCancelled("stop requested")
        elapsed = time.monotonic() - started
        name = f" ({entry['name']})" if entry["name"] else ""
        message = MESSAGE_SEPARATOR.join(
            part
            for part in (
                phase,
                f"cluster {clock['done'] + 1} of {total}",
                f"node {entry['node']}{name}",
                detail,
            )
            if part
        )
        emit(
            ProfileDescribeProgressEvent(
                recordings_done=clock["done"],
                recordings_total=total,
                frames_done=round(min(max(within, 0.0), 1.0) * WITHIN_RESOLUTION),
                frames_total=WITHIN_RESOLUTION,
                elapsed_s=elapsed,
                eta_s=remaining_seconds(
                    (clock["work"] + within) / (len(todo) or 1), elapsed
                ),
                message=message,
            ),
            force=phase != current["phase"],
        )
        current["phase"] = phase

    _write_progress_record(run_dir, job, entries)
    stopped = False
    for entry in todo:
        if stop_event.is_set():
            stopped = True
            break
        node = entry["node"]
        began = time.monotonic()

        def on_profile(stage, fraction, detail, _entry=entry):
            phase, share = _STAGES.get(stage, ("fitting", 0.95))
            if stage == "measuring":
                share = _MEASURE_SHARE * min(max(float(fraction), 0.0), 1.0)
            report(_entry, phase, share, detail)

        report(entry, "measuring", 0.0, "drawing members")
        try:
            profile = experiment.profile_cluster(
                run_id, node, batch.profile_config(config).model_dump(),
                save=False, progress=on_profile,
            )  # fmt: skip
        except OperationCancelled:
            entry.update(status="cancelled")
            entry["profile"]["status"] = "cancelled"
            stopped = True
            break
        except Exception as exc:  # noqa: BLE001 — one cluster's isolation boundary
            _failed(entry, exc, time.monotonic() - began)
            clock["done"] += 1
            clock["work"] += 1
            _write_progress_record(run_dir, job, entries)
            continue
        entry["profile"].update(_profile_fields(profile))
        # Into this job's own run directory, then placed beside the run, where the Interpret
        # tab reads it.
        cluster_profile.save_profile(run_dir, profile)
        batch.place(experiment, job.job_id, [node])
        entry["status"] = "complete"
        entry["seconds"] = round(time.monotonic() - began, 2)
        clock["done"] += 1
        clock["work"] += 1
        _write_progress_record(run_dir, job, entries)

    for entry in entries:
        if entry["status"] == "pending":
            entry["status"] = "not run"
    emit(
        ProfileDescribeProgressEvent(
            recordings_done=clock["done"],
            recordings_total=total,
            frames_done=0,
            frames_total=0,
            elapsed_s=time.monotonic() - started,
            eta_s=0.0,
            message=DONE_PHASE if not stopped else f"{DONE_PHASE} · stopped",
        ),
        force=True,
    )
    finished = sum(1 for e in entries if e["status"] in ("complete", "skipped"))
    if finished == total:
        status = "complete"
    elif finished:
        status = "partial"
    else:
        status = "failed" if not stopped else "partial"
    return entries, status


def _entry(node, names: dict, state) -> dict:
    """One cluster's row before the job reaches it: a done profile marked ``skipped``."""
    return {
        "node": int(node),
        "name": str(names.get(str(int(node))) or ""),
        "status": "skipped" if state.done else "pending",
        "profile": {"status": "skipped" if state.profile_done else "pending"},
    }


def _profile_fields(profile) -> dict:
    e = profile.explanation
    return {
        "status": "done",
        "seconds": float(profile.elapsed_s),
        "accuracy": float(e.accuracy),
        "floor": float(e.floor),
        "separated": bool(e.separated),
    }


def _failed(entry: dict, exc: BaseException, seconds: float) -> None:
    entry["status"] = "failed"
    entry["profile"].update(status="failed", seconds=round(seconds, 2))
    entry["error"] = f"profile: {type(exc).__name__}: {exc}"
